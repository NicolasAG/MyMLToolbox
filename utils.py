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
        :return: one batch of examples where each example is a line of idx ~(n_lines, max_len)
        """
        # Add words to the dictionary
        with open(path, 'r') as f:
            lines = 0
            max_len = 0
            for line in f:
                # skip empty lines
                if len(line.strip().split()) == 0:
                    continue

                tokens = 0  # number of tokens in each line
                sents = sent_tokenize(line)  # list of sentences in this line
                for sent in sents:
                    sent = normalize_string(sent)  # lowercase, strip, to ascii
                    words = [self.sos_tag] + word_tokenize(sent) + [self.eos_tag]  # list of words in this sentence
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)

                # keep track of longest line
                if tokens > max_len:
                    max_len = tokens
                lines += 1

        # Tokenize the file content
        ids = torch.LongTensor(lines, max_len).fill_(self.dictionary.word2idx[self.pad_tag])
        with open(path, 'r') as f:
            i = 0  # line number
            for line in f:
                # skip empty lines
                if len(line.strip().split()) == 0:
                    continue

                j = 0  # token number
                sents = sent_tokenize(line)
                for sent in sents:
                    sent = normalize_string(sent)  # lowercase, strip, to ascii
                    words = [self.sos_tag] + word_tokenize(sent) + [self.eos_tag]
                    for word in words:
                        ids[i, j] = self.dictionary.word2idx[word]
                        j += 1
                i += 1

        return ids

    def to_str(self, idx_sents):
        """
        Convert a batch of idx to strings
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
