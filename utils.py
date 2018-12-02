import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import copy
import math
import unicodedata

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from nltk import sent_tokenize, word_tokenize

from gensim.models import KeyedVectors


def str2bool(v):
    """
    Used in argument parser for custom argument type
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected. Got %s' % v)


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


def load_google_word2vec(fname):
    """
    Loads the 3Mx300 matrix
    """
    return KeyedVectors.load_word2vec_format(fname, binary=True)


def load_glove_word2vec(fname):
    """
    Loads word vecs from gloVe
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        for i, line in enumerate(f):
            ll = line.split()
            word = ll[0].lower().strip()
            word_vecs[word] = np.array(ll[1:], dtype='float32')
            if '__length__' not in word_vecs:
                word_vecs['__length__'] = len(word_vecs[word])
    return word_vecs


def set_word_embeddings(embedding_layer, w2v, dictionary, requires_grad=True):
    """
    set embedding layer of an rnn to a specific word2vec dictionary
    :param embedding_layer: torch.nn.Embedding layer
    :param w2v: mapping from word to vectors
    :param corpus: Corpus object with Dictionary
    :param requires_grad: fine-tune the embeddings
    """

    if embedding_layer.weight.requires_grad:
        params = embedding_layer.weight.detach().cpu().numpy()
    else:
        params = embedding_layer.weight.cpu().numpy()

    # embed = np.random.uniform(-0.1, 0.1, size=(len(dictionary), 300))
    count = 0
    for tok_id in range(len(dictionary)):
        try:
            # embed[tok_id] = w2v[dictionary.idx2word[tok_id]]
            params[tok_id] = w2v[dictionary.idx2word[tok_id]]
            count += 1
        except KeyError:
            pass

    print("Got %d/%d = %.6f pretrained embeddings" % (
        count, len(dictionary), float(count) / len(dictionary)
    ))

    params = torch.from_numpy(params).float()  # convert numpy array to torch float tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params = params.to(device)
    embedding_layer.weight = torch.nn.Parameter(params, requires_grad=requires_grad)
    # embedding_layer.weight.requires_grad = requires_grad


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
    """
    Merge back tokens matching a given symbol
    :param tokens: list of tokens
    :param symbol: symbol that should not be tokenized
    :return: new list of tokens
    """
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
    def __init__(self, required=None):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.word2cnt = {}
        self.trimmed = False  # flag for trimmed vocab

        if required is None:
            self.required = []
        else:
            self.required = required

        for token in self.required:
            self.add_word(token)

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

        # add required tokens
        for token in self.required:
            keep_words.append({'w': token,
                               'c': self.word2cnt[token]})

        for w, c in self.word2cnt.items():
            if c >= min_count and w not in self.required:
                keep_words.append({'w': w, 'c': c})

        print("keeping %d words from %d = %.4f" % (
            len(keep_words), len(self.word2idx), len(keep_words) / len(self.word2idx)
        ))

        # Reinitialize dictionaries
        self.word2idx = {}
        self.idx2word = {}
        self.word2cnt = {}
        self.idx = 0
        for e in keep_words:
            self.word2idx[e['w']] = self.idx
            self.word2cnt[e['w']] = e['c']
            self.idx2word[self.idx] = e['w']
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    """
    General object to hold a corpus of text from an input file
    Used in Language modeling.
    """
    def __init__(self, pad_tag='<pad>', unk_tag='<unk>', sos_tag='<sos>', eos_tag='<eos>'):
        self.dictionary = Dictionary(required=[pad_tag, unk_tag, sos_tag, eos_tag])
        self.pad_tag = pad_tag  # pad symbol
        self.unk_tag = unk_tag  # unknown word
        self.sos_tag = sos_tag  # start-of-sentence tag
        self.eos_tag = eos_tag  # end-of-sentence tag

    def get_data_from_lines(self, path, max_n_lines=-1, max_context_size=-1, max_seq_length=-1,
                            reverse_tgt=False, debug=False, add_to_dict=True):
        """
        Reads an input file where each line is considered as one conversation with more than one sentence.
        :param path: path to a readable file
        :param max_n_lines: consider top lines
        :param max_context_size: number of sentences to keep in the context
        :param max_seq_length: max number of tokens in one sequence
        :param reverse_tgt: reverse tokens of the tgt sequence
        :param debug: print a few item examples
        :param add_to_dict: add words to dictionary
        :return: list of (src, tgt) pairs where src is all possible contexts and tgt is the next sentence
        """
        src = []  # list of contexts
        tgt = []  # list of next sentences

        truncated_src = 0  # number of truncated source sequences
        truncated_tgt = 0  # number of truncated target sequences
        src_tokens_lost = 0  # number of tokens removed after truncation
        tgt_tokens_lost = 0  # number of tokens removed after truncation

        f = open(path, 'r')
        num_lines = sum(1 for _ in f)
        print("%d lines" % num_lines)
        f.close()

        with open(path, 'r') as f:

            # bar = pyprind.ProgBar(num_lines, stream=sys.stdout)
            line_done = 0

            for line in f:

                # skip empty lines
                if len(line.strip().split()) == 0:
                    continue

                sentences = sent_tokenize(line)  # list of sentences in this line
                start = 0  # index of the first sentences to keep in the `src`

                # for all sentences except the last one,
                # add words to dictionary & make a (src - tgt) pair
                for s_id in range(len(sentences) - 1):
                    if max_context_size > 0:
                        sentences_to_consider = sentences[start: s_id + 1]
                        # move pointer to the right when `src` reached its max capacity
                        if len(sentences_to_consider) == max_context_size:
                            start += 1  # sliding window ->->
                    else:
                        # take all sentences before i+1 as src
                        sentences_to_consider = sentences[:s_id+1]

                    # lowercase, strip, to ascii
                    src_sents = normalize_string(
                        (' ' + self.eos_tag + ' ' + self.sos_tag + ' ').join(sentences_to_consider)
                    )
                    tgt_sent = normalize_string(sentences[s_id+1])

                    # list of words in the sentences
                    src_words = [self.sos_tag] + word_tokenize(src_sents) + [self.eos_tag]
                    src_words = undo_word_tokenizer(src_words, self.sos_tag)
                    src_words = undo_word_tokenizer(src_words, self.unk_tag)
                    src_words = undo_word_tokenizer(src_words, self.eos_tag)
                    if 0 < max_seq_length < len(src_words):
                        src_tokens_lost += len(src_words) - max_seq_length
                        # truncate source sentence at the beginning
                        src_words = [self.sos_tag] + src_words[-(max_seq_length-1):]
                        truncated_src += 1

                    tgt_words = [self.sos_tag] + word_tokenize(tgt_sent) + [self.eos_tag]
                    tgt_words = undo_word_tokenizer(tgt_words, self.sos_tag)
                    tgt_words = undo_word_tokenizer(tgt_words, self.unk_tag)
                    tgt_words = undo_word_tokenizer(tgt_words, self.eos_tag)
                    if 0 < max_seq_length < len(tgt_words):
                        tgt_tokens_lost += len(tgt_words) - max_seq_length
                        # truncate target sentence at the tail
                        tgt_words = tgt_words[:(max_seq_length-1)] + [self.eos_tag]
                        truncated_tgt += 1

                    # add words to dictionary
                    if add_to_dict:
                        # always add words of the tgt sentences
                        for word in tgt_words:
                            self.dictionary.add_word(word)
                        # only add words of the first src sentence
                        if s_id == 0:
                            for word in src_words:
                                self.dictionary.add_word(word)

                    if reverse_tgt:
                        tgt_words = tgt_words[::-1]

                    src.append(' '.join(src_words))
                    tgt.append(' '.join(tgt_words))

                # bar.update()
                line_done += 1
                if line_done % 1000 == 0:
                    print("#", end='')

                if 0 < max_n_lines <= line_done:
                    break

            print("")

        if debug:
            # print a few random examples
            start1 = 0
            start2 = 50
            for src_ex, tgt_ex in zip(src[start1: start1+3], tgt[start1: start1+3]):
                print('src:', src_ex)
                print('tgt:', tgt_ex)
                print('')
            if len(src) > start2+3:
                for src_ex, tgt_ex in zip(src[start2: start2+3], tgt[start2: start2+3]):
                    print('src:', src_ex)
                    print('tgt:', tgt_ex)
                    print('')

        if truncated_src > 0:
            print("Truncated %d (%d) / %d = %4f source sentences" % (
                truncated_src, src_tokens_lost, len(src), truncated_src / len(src)
            ))
        if truncated_tgt > 0:
            print("Truncated %d (%d) / %d = %4f target sentences" % (
                truncated_tgt, tgt_tokens_lost, len(tgt), truncated_tgt / len(tgt)
            ))

        return src, tgt

    def to_str(self, idx_sents, filter_pad=False):
        """
        Convert a batch of idx to strings
        :param idx_sents: list of idx sentence [ [id1 id2 id3], ..., [id1 id2 id3] ]
        :param filter_pad: remove <pad> tokens in the string format
        :return strings: list of sentences [ "w1 w2 w3", ..., "w1 w2 w3" ]
        """
        if isinstance(idx_sents, torch.Tensor):
            if idx_sents.is_cuda:
                idx_sents = idx_sents.cpu()
            idx_sents = idx_sents.numpy()

        strings = []
        for idx_sent in idx_sents:
            str_sent = [self.dictionary.idx2word.get(x, self.unk_tag) for x in idx_sent]

            if filter_pad:
                # filter out '<pad>'
                str_sent = filter(lambda x: x != self.pad_tag, str_sent)

            str_sent = ' '.join(str_sent)
            strings.append(str_sent)
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
            assert h_t.size(2) == outs.size(2), "encoder " \
                "hidden size: %d != decoder hidden size: %d" % (outs.size(2), h_t.size(2))
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
            concat = torch.cat((tmp_h_t, outs), 2)  # ~(bs, enc_seq, dec_size + enc_size)

            concat = self.attn(concat)    # ~(bs, enc_seq, dec_size)
            grid = self.v(concat)         # ~(bs, enc_seq, 1)
            grid = grid.permute(0, 2, 1)  # ~(bs, dec_seq=1, enc_seq)

            del tmp_h_t, concat

        # grid is now of shape ~(bs, dec_seq=1, enc_seq)

        attn_weights = F.softmax(grid, dim=2)  # ~(bs, dec_seq=1, enc_seq)

        # make sure to compute softmax over valid tokens only
        mask = (grid != 0).float()                          # ~(bs, dec_seq=1, enc_seq)
        attn_weights = attn_weights * mask                  # ~(bs, dec_seq=1, enc_seq)
        # ^ WARNING: the above line cannot use *= because Pytorch will give this error:
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
        # Thanks to: https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/5
        normalizer = attn_weights.sum(dim=2, keepdim=True)  # ~(bs, dec_seq=1, 1)
        attn_weights /= normalizer                          # ~(bs, dec_seq=1, enc_seq)

        return attn_weights  # ~(bs, dec_seq=1, enc_seq)


def set_gradient(model, value: bool):
    """
    Make all parameters of the model trainable or not
    """
    for p in model.parameters():
        p.requires_grad = value


def split_list(array, separator):
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


def _sequence_mask(seq_lengths, max_len):
    """
    Build a mask for a batch of sequences of various lengths
    --> used in masked_cross_entropy() below
    :param seq_lengths: length of each sequence ~(bs)
    :param max_len: maximum length of a sequence in the batch
    :return: mask ~(bs, max_len)
    """
    assert seq_lengths.max().item() == max_len
    bs = seq_lengths.size(0)

    seq_range = torch.arange(0, max_len).long()  # [0,1,2,3,...,max_len-1]
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    '''
    seq_range_expand = [
        [0, 1, 2, 3, ..., max_len-1],
        [0, 1, 2, 3, ..., max_len-1],
        [0, 1, 2, 3, ..., max_len-1],
        [0, 1, 2, 3, ..., max_len-1],
        ...
    ]
    ~(bs, max_len)
    '''

    seq_lengths_expand = (
        seq_lengths.unsqueeze(1).expand_as(seq_range_expand)
    )
    '''
    seq_lengths_expand = [
        [length_1, length_1, ..., length_1],
        [length_2, length_2, ..., length_2],
        [length_3, length_3, ..., length_3],
        [length_4, length_4, ..., length_4],
        ...
    ]
    # ~(bs, max_len)
    '''

    # move to GPU if available
    if seq_lengths_expand.is_cuda:
        seq_range_expand = seq_range_expand.cuda()

    return seq_range_expand < seq_lengths_expand


def masked_cross_entropy(logits, target, lengths):
    """
    Compute the negative log softmax of a batch of sequences.
    The loss is defined as the negative log-likelihood
    --> used in hred language modelling
    :param logits: FloatTensor containing unormalized probabilities for each class
                   ~(bs, max_len, num_classes)
    :param target: LongTensor containing the index of the true class for each step
                   ~(bs, max_len)
    :param lengths: LongTensor containing the length of each data in a batch
                    ~(bs)

    :return loss: an average loss value masked by the length.
    """
    bs, max_len, num_classes = logits.size()

    # grab the log softmax proba of each class
    logits_flat = logits.view(-1, num_classes)          # ~(bs*max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=1)  # ~(bs*max_len, num_classes)

    # grab the negative log likelihood (-log proba) of the true tokens
    target_flat = target.view(-1, 1)         # ~(bs*max_len, 1)
    losses_flat = - torch.gather(
        log_probs_flat, dim=1, index=target_flat
    )                                        # ~(bs*max_len, 1)

    # mask out loss of the padded parts of the sequence
    losses = losses_flat.view(bs, max_len)   # ~(bs, max_len)
    mask = _sequence_mask(lengths, max_len)  # ~(bs, max_len)
    losses *= mask.float()

    # compute the average loss
    loss = losses.sum() / lengths.float().sum()
    return loss


def show_attention(input_sequence, output_words, attentions, name=""):
    """
    :param input_sequence: list of input strings
    :param output_words: list of output words
    :param attentions: ~(max_tgt_len, max_src_len)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')

    if isinstance(input_sequence, str):
        input_sequence = input_sequence.split()

    # set up axes
    ax.set_xticklabels([''] + input_sequence + ['<EOC>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    if len(name) > 0:
        if not os.path.exists('images'):
            os.makedirs('images')
        plt.savefig('images/' + str(name) + '.png')
    else:
        plt.show()
    plt.close()
