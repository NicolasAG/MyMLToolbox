import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

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

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    """
    General object to hold a corpus of text from an input file
    Used in Language modeling.
    """
    def __init__(self, pad_tag='<pad>', unk_tag='<unk>', eos_tag='<eos>'):
        self.dictionary = Dictionary()
        self.pad_tag = pad_tag  # pad symbol
        self.unk_tag = unk_tag  # unknown word
        self.eos_tag = eos_tag  # end-of-sentence tag
        self.dictionary.add_word(self.pad_tag)
        self.dictionary.add_word(self.unk_tag)
        self.dictionary.add_word(self.eos_tag)

    def get_data(self, path):
        """
        :param path: path to a readable file
        :return: one batch of examples where each example is a line of idx
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
                    words = word_tokenize(sent) + [self.eos_tag]  # list of words in this sentence
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
                    words = word_tokenize(sent) + [self.eos_tag]
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


