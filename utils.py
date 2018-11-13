import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import unicodedata

from nltk import sent_tokenize, word_tokenize


def clones(module, n):
    """
    Produces N identical layers
    """
    return torch.nn.ModuleList([
        copy.deepcopy(module) for _ in range(n)
    ])


class LayerNorm(nn.Module):
    """
    - Construct a layer normalization module
    - See https://arxiv.org/abs/1607.06450 for details
    - Generally applied before the non-linearity?
    """
    def __init__(self, features: int, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # gain?
        self.b_2 = nn.Parameter(torch.zeros(features))  # bias?
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def scaled_dot_prod_attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention' used in transformer.
    see: http://nlp.seas.harvard.edu/2018/04/03/attention.html#attention
    -----------------------------------------------------------
    - The output is computed as a weighted sum of the values (V),
    where the weight assigned to each value is computed by a
    compatibility function of the query (Q) with the corresponding key (K).
    - We call our particular attention "Scaled Dot-Product Attention".
    - The input consists of queries (Q) and keys (K) of dimension d_k
    and values (V) of dimension d_v.
    - We compute the dot products of the query (Q) with all keys (K),
    divide each by sqrt(d_k) and apply a softmax function
    to obtain the weights on the values (V).
    - In practice, we compute the attention function on a set of
    queries simultaneously, packed together into a matrix Q.
    The keys and values are also packed together into matrices K and V.
    We compute the matrix of outputs as:
    Attention(Q, K, V) = softmax( Q.K / sqrt(d_k) ) * V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # replace 0's by -inf in mask?
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def unicode_to_ascii(s):
    """
    Turn a Unicode string to plain ASCII
    thanks to http://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    """
    Lowercase, trim, and remove non-letter characters
    """
    return unicode_to_ascii(s.lower().strip())


def undo_word_tokenizer(tokens, symbol):
    symbols = word_tokenize(symbol)
    j = 0  # index going through symbols

    new_words = []  # list of new tokens to be returned
    tmp_w = ''  # tmp string to store tokens we try to match

    for t in tokens:
        # match, save token & move pointer to the right
        if t == symbols[j]:
            tmp_w += t + ' '  # save this token in the tmp_word
            j += 1            # move pointer of symbols to the right

            # find a complete match! add token & reset
            if j >= len(symbols):
                new_words.append(tmp_w.replace(' ', ''))
                tmp_w = ''  # reset tmp_word
                j = 0       # reset pointer of symbols

        # no match, reset
        else:
            if len(tmp_w) > 0:
                # save the previous tokens we tried to match
                new_words.extend(word_tokenize(tmp_w))
            new_words.append(t)  # save this token
            tmp_w = ''           # reset tmp_word
            j = 0                # reset pointer of symbols

    return new_words


class Dictionary(object):
    """
    Custom dictionary for word-to-idx and idx-to-word.
    Used in Language modeling.
    see: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/data_utils.py
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.word2cnt = {}
        self.trimmed = False  # flag for trimmed vocab

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.word2cnt[word] = 1
            self.idx2word[self.idx] = word
            self.idx += 1
        else:
            self.word2cnt[word] += 1

    def trim(self, min_count):
        """
        Remove words below a certain count threshold
        :param min_count: threshold
        """
        if self.trimmed:  # if already trimmed, return
            return
        self.trimmed = True  # flag to true so that we don't do it again

        keep_words = []

        for w, c in self.word2cnt.items():
            if c >= min_count:
                keep_words.append({'w': w, 'c': c, 'idx': self.word2idx[w]})

        print("keeping %d words from %d = %.4f" % (
            len(keep_words), len(self.word2idx), len(keep_words) / len(self.word2idx)
        ))

        # Reinitialize dictionaries
        self.word2idx = {}
        self.idx2word = {}
        self.word2cnt = {}
        self.idx = 0
        for e in keep_words:
            self.word2idx[e['w']] = e['idx']
            self.word2cnt[e['w']] = e['c']
            self.idx2word[e['idx']] = e['w']
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    """
    General object to hold a corpus of text from an input file
    Used in Language modeling.
    """
    def __init__(self, pad_tag='<pad>', unk_tag='<unk>', sos_tag='<sos>', eos_tag='<eos>'):
        self.dictionary = Dictionary()
        self.pad_tag = pad_tag  # pad symbol
        self.unk_tag = unk_tag  # unknown word
        self.sos_tag = sos_tag  # start-of-sentence tag
        self.eos_tag = eos_tag  # end-of-sentence tag
        self.dictionary.add_word(self.pad_tag)
        self.dictionary.add_word(self.unk_tag)
        self.dictionary.add_word(self.eos_tag)

    def get_data(self, path):
        """
        :param path: path to a readable file
        :return: list of (src, tgt) pairs where src is all possible contexts and tgt is the next sentence
        """
        src = []  # list of contexts
        tgt = []  # list of next sentences

        with open(path, 'r') as f:
            for line in f:

                # skip empty lines
                if len(line.strip().split()) == 0:
                    continue

                sentences = sent_tokenize(line)  # list of sentences in this line

                # for all sentences except but the last one,
                # add words to dictionary & make a (src - tgt) pair
                for s_id in range(len(sentences) - 1):
                    # lowercase, strip, to ascii
                    src_sents = normalize_string(  # take all sentences before i+1 as src
                        (' ' + self.eos_tag + ' ' + self.sos_tag + ' ').join(sentences[:s_id+1])
                    )
                    tgt_sent = normalize_string(sentences[s_id+1])

                    # list of words in the sentences
                    src_words = [self.sos_tag] + word_tokenize(src_sents) + [self.eos_tag]
                    src_words = undo_word_tokenizer(src_words, self.sos_tag)
                    src_words = undo_word_tokenizer(src_words, self.eos_tag)

                    tgt_words = [self.sos_tag] + word_tokenize(tgt_sent) + [self.eos_tag]
                    tgt_words = undo_word_tokenizer(tgt_words, self.sos_tag)
                    tgt_words = undo_word_tokenizer(tgt_words, self.eos_tag)

                    # add words to dictionary
                    # always add words of the tgt sentences
                    for word in tgt_words:
                        self.dictionary.add_word(word)
                    # only add words of the first sentence
                    if s_id == 0:
                        for word in src_words:
                            self.dictionary.add_word(word)

                    src.append(' '.join(src_words))
                    tgt.append(' '.join(tgt_words))

        return src, tgt

    def to_str(self, idx_sents):
        """
        Convert a batch of idx to strings
        :param idx_sents: list of idx sentence [ [id1 id2 id3], ..., [id1 id2 id3] ]
        :return strings: list of sentences [ "w1 w2 w3", ..., "w1 w2 w3" ]
        """
        if isinstance(idx_sents, torch.Tensor):
            idx_sents = idx_sents.numpy()

        strings = []
        for idx_sent in idx_sents:
            str_sent = ""
            for idx_word in idx_sent:
                str_sent += self.dictionary.idx2word[idx_word] + " "
            strings.append(str_sent.strip())
        return strings

    def to_idx(self, str_sents):
        """
        Convert a batch of strings to idx
        :param str_sents: list of sentences [ "w1 w2 w3", ..., "w1 w2 w3" ]
        :return idx: list of idx sentences [ [id1 id2 id3], ..., [id1 id2 id3] ]
        """
        idx = []
        for str_sent in str_sents:
            idx_sent = []
            for word in str_sent.split():
                idx_word = self.dictionary.word2idx.get(
                    word,
                    self.dictionary.word2idx[self.unk_tag]
                )
                idx_sent.append(idx_word)
            idx.append(idx_sent)
        return idx

    def fill_seq(self, seq, max_len, fill_token=None):
        """
        Pad a sequence with a fill_token until the seq reaches max_len
        """
        if fill_token is None:
            fill_token = self.dictionary.word2idx[self.pad_tag]
        padded_seq = copy.copy(seq)
        padded_seq += [fill_token] * (max_len - len(seq))
        return padded_seq


class AttentionModule(nn.Module):
    """
    Compute attention weights for a given hidden state (h_t) and a list of encoder outputs (outs)
    Different modes available: dot, general, concat

    Tutorial: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
              |_> "Implementing an attention module"
    """
    def __init__(self, decoder_size, encoder_size, method='general'):
        """
        :param decoder_size: size of the decoder hidden state
        :param encoder_size: size of the encoder hidden state
        :param method: 'dot' or 'general' or 'concat'
        """
        super(AttentionModule, self).__init__()

        assert method.lower().strip() in ['concat', 'general', 'dot']
        self.method = method.lower().strip()

        # wn = lambda x: nn.utils.weight_norm(x)

        if self.method == 'general':
            self.attn = nn.Linear(encoder_size, decoder_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(encoder_size + decoder_size, decoder_size)
            self.v = nn.Linear(decoder_size, 1)

    def forward(self, h_t, outs):
        """
        :param h_t: hidden state of the decoder ~(bs, dec_seq=1, dec_size)
        :param outs: output vectors of the encoder ~(bs, enc_seq, enc_size)
        :return attention weights: ~(bs, dec_seq=1, enc_seq)
        """

        # make sure inputs have the same batch size
        assert h_t.size(0) == outs.size(0)
        # make sure inputs have the same number of dimensions
        assert len(h_t.size()) == len(outs.size()) == 3

        if self.method == 'dot':
            # make sure inputs have the same hidden size
            assert h_t.size(2) == outs.size(2)
            # swap (transpose) dimension 1 & 2 of outs
            tmp_outs = outs.permute(0, 2, 1)  # ~(bs, hid_size, enc_seq)

            # bmm performs a batch dot product between 2 tensors of 3D
            # h_t~(bs, 1, hid_size) DOT tmp_outs~(bs, hid_size, enc_seq)
            grid = torch.bmm(h_t, tmp_outs)   # ~(bs, dec_seq=1, enc_seq)

            del tmp_outs

        elif self.method == 'general':
            tmp_outs = self.attn(outs)            # ~(bs, enc_seq, dec_size)
            # swap (transpose) dimension 1 & 2
            tmp_outs = tmp_outs.permute(0, 2, 1)  # ~(bs, dec_size, enc_seq)

            # bmm performs a batch dot product between 2 tensors of 3D
            # h_t~(bs, 1, dec_size) DOT tmp_outs~(bs, dec_size, enc_seq)
            grid = torch.bmm(h_t, tmp_outs)       # (bs, dec_seq=1, enc_seq)

            del tmp_outs

        elif self.method == 'concat':
            # expand h_t to be of same sequence_length as outs
            # (bs, 1, dec_size) --> ~(bs, enc_seq, dec_size)
            tmp_h_t = h_t.expand((h_t.size(0), outs.size(1), h_t.size(2)))

            # concatenate tmp_h_t and outs along the hidden_size dim
            concat = torch.cat((tmp_h_t, encoder_outputs), 2)  # ~(bs, enc_seq, dec_size + enc_size)

            concat = self.attn(concat)    # ~(bs, enc_seq, dec_size)
            grid = self.v(concat)         # ~(bs, enc_seq, 1)
            grid = grid.permute(0, 2, 1)  # ~(bs, dec_seq=1, enc_seq)

            del tmp_h_t, concat

        # grid is now of shape ~(bs, dec_seq=1, enc_seq)

        attn_weights = F.softmax(grid, dim=2)  # ~(bs, dec_seq=1, enc_seq)

        # make sure to compute softmax over valid tokens only
        mask = (grid != 0).float()                          # ~(bs, dec_seq=1, enc_seq)
        attn_weights *= mask                                # ~(bs, dec_seq=1, enc_seq)
        normalizer = attn_weights.sum(dim=2, keepdim=True)  # ~(bs, dec_seq=1, 1)
        attn_weights /= normalizer                          # ~(bs, dec_seq=1, enc_seq)

        return attn_weights  # ~(bs, dec_seq=1, enc_seq)


def set_gradient(model, value):
    """
    Make all parameters of the model trainable or not
    """
    for p in model.parameters():
        p.requires_grad = value


def separate_list(array, separator):
    """
    Split an array of idx around a separator of any length
    :param array: list of idx [0,a,b,c,1, ..., 0,a,b,c,1]
    :param separator: list of separator tokens [1]
    :return: a list of list [ [0,a,b,c], [...], [0,a,b,c], []]

    See: https://stackoverflow.com/a/34732203
    """
    results = []
    a = array[:]
    i = 0
    while i <= len(a) - len(separator):
        if a[i: i+len(separator)] == separator:
            results.append(a[:i])
            a = a[i+len(separator):]
            i = 0
        else:
            i += 1
    results.append(a)
    return results
