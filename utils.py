import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import copy
import math
import json
import codecs
import logging
import unicodedata
import pickle as pkl

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import nltk
from nltk import sent_tokenize, word_tokenize

from gensim.models import KeyedVectors

from MyMLToolbox.external.subword_nmt.learn_bpe import learn_bpe
from MyMLToolbox.external.subword_nmt.apply_bpe import BPE


# Check nltk dependencies
try:
    nltk.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


logger = logging.getLogger(__name__)


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
    :param dictionary: Dictionary object with str2idx & idx2str mapping
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

    logger.info("Got %d/%d = %.6f pretrained embeddings" % (
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
    Merge back tokens matching a given symbol after calling nltk.word_tokenize
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


def put_back_bpe_separator(tokens, seprator):
    """
    Merge back tokens matching the bpe separator after calling nltk.word_tokenize
    :param tokens: list of tokens
    :param seprator: bpe separator like '@@'
    :return: new list of tokens
    """
    symbols = word_tokenize(seprator)

    start = 0
    j = 0  # index going through symbols

    new_words = []  # list of new tokens to be returned
    tmp_w = ''  # tmp string to store tokens we try to match

    for i, tok in enumerate(tokens):
        # match, save token & move pointer to the right
        if tok == symbols[j]:
            if len(tmp_w) == 0:
                # this is the first time we match, save the position of the previous token
                start = i-1

            tmp_w += tok + ' '  # save this token in the tmp_word
            j += 1  # move pointer of symbols to the right

            # find a complete match! add token & reset
            if j >= len(symbols):
                tmp_w = tokens[start] + ' ' + tmp_w  # put the first token in front
                new_words = new_words[:-1]  # ignore the previous token as we include it in tmp_w
                new_words.append(tmp_w.replace(' ', ''))
                tmp_w = ''  # reset tmp_word
                j = 0  # reset pointer of symbols

        # no match, reset
        else:
            if len(tmp_w) > 0:
                # save the previous tokens we tried to match
                new_words.extend(word_tokenize(tmp_w))
            new_words.append(tok)  # save this token
            tmp_w = ''  # reset tmp_word
            j = 0  # reset pointer of symbols

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

        logger.info("keeping %d words from %d = %.4f" % (
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
        self.bpe = None  # path to the BPE object -- see MyMLToolbox/external/subword-nmt/apply_bpe.BPE()

    def learn_bpe(self, input_file, output_prefix, target_size):
        """
        Just a util function that calls MyMLToolbox/external/subword-nmt/learn_bpe.learn_bpe()
        :param input_file: txt file to read and learn bpe
        :param output_prefix: path to the code file
        :param target_size: target vocabulary size
        """
        # only learn BPE codes if they do no exist yet.
        if not os.path.isfile(output_prefix + '.codes'):
            logger.info("Learning BPE codes...")
            #fin = codecs.open(input_file, mode='r', encoding='utf-8')
            fin = open(input_file, mode='r', encoding='utf-8')
            #fout = codecs.open(output_prefix + '.codes', mode='w', encoding='utf-8')
            fout = open(output_prefix + '.codes', mode='w', encoding='utf-8')
            learn_bpe(fin, fout, target_size)
            fin.close()
            fout.close()

        logger.info("Building BPE object...")
        codes = codecs.open(output_prefix + '.codes', mode='r', encoding='utf-8')
        self.bpe = BPE(codes)
        codes.close()

    def already_preprocessed(self, path, max_context_size=-1, max_seq_length=-1, reverse_tgt=False):
        """
        Check if data was already preprocessed with these specific arguments.
        If so, return it, else,  return empty list
        :param path: prefix to a preprocessed file
        :param max_n_examples: consider top k examples - default -1
        :param max_context_size: number of sentences to keep in the context
        :param max_seq_length: max number of tokens in one sequence
        :param reverse_tgt: reverse tokens of the tgt sequence
        :param add_to_dict: add words to dictionary - default True
        """
        # check if data has been preprocessed before
        if os.path.isfile(path + '.preprocessed-mcs%d-msl%d-rt%d-bpe%d.pkl' % (
                max_context_size, max_seq_length, reverse_tgt, self.bpe is not None
        )):
            with open(path + '.preprocessed-mcs%d-msl%d-rt%d-bpe%d.pkl' % (
                    max_context_size, max_seq_length, reverse_tgt, self.bpe is not None
            ), 'rb') as f:
                # src, tgt = pkl.load(f)
                data = pkl.load(f)
            # assert len(src) == len(tgt)
            return data

        # return [], []
        return None

    def save_preprocessed_data(self, path, data, max_context_size=-1, max_seq_length=-1, reverse_tgt=False):
        """
        save preprocessed data
        """
        with open(path + '.preprocessed-mcs%d-msl%d-rt%d-bpe%d.pkl' % (
                max_context_size, max_seq_length, reverse_tgt, self.bpe is not None
        ), 'wb') as f:
            pkl.dump(data, f, pkl.HIGHEST_PROTOCOL)

    @staticmethod
    def print_few_examples(data, prefix='sent'):
        start1 = 0
        start2 = 50
        for sent in data[start1: start1+3]:
            logger.info('%s: %s' % (prefix, sent))
            logger.info('')
        if len(data) > start2 + 3:
            for sent in data[start2: start2 + 3]:
                logger.info('%s: %s' % (prefix, sent))
                logger.info('')

    @staticmethod
    def print_few_hred_examples(src, tgt):
        # print a few random examples
        start1 = 0
        start2 = 50
        for src_ex, tgt_ex in zip(src[start1: start1 + 3], tgt[start1: start1 + 3]):
            logger.info('src: %s' % src_ex)
            logger.info('tgt: %s' % tgt_ex)
            logger.info('')
        if len(src) > start2 + 3:
            for src_ex, tgt_ex in zip(src[start2: start2 + 3], tgt[start2: start2 + 3]):
                logger.info('src: %s' % src_ex)
                logger.info('tgt: %s' % tgt_ex)
                logger.info('')

    def preprocess_sentence(self, sentence, add_to_dict, max_seq_length, truncate_at_tail=True,
                            reverse_tokens=False, bpe=True):
        """
        Pre-process a string sentence (BPE, word tokenize, normalize, truncate, add to dict...)
        :param sentence: the string to process
        :param add_to_dict: add tokens to vocabulary
        :param max_seq_length: maximum number of tokens
        :param truncate_at_tail: keep the beginning and ignore the end of the sentence
        :param reverse_tokens: reverse the order of the tokens
        :param bpe: perform BPE if possible
        :return: processed sentence, number of ignored tokens
        """
        if bpe and self.bpe is not None:
            sentence = self.bpe.process_line(sentence)

        # lowercase, strip, to ascii
        sentence = normalize_string(sentence)

        # list of words in the sentence
        tokens = [self.sos_tag] + word_tokenize(sentence) + [self.eos_tag]
        tokens = undo_word_tokenizer(tokens, self.sos_tag)
        tokens = undo_word_tokenizer(tokens, self.unk_tag)
        tokens = undo_word_tokenizer(tokens, self.eos_tag)
        if self.bpe is not None:
            tokens = put_back_bpe_separator(tokens, self.bpe.separator)

        # truncate if too long
        if 0 < max_seq_length < len(tokens):
            tokens_lost = len(tokens) - max_seq_length
            if truncate_at_tail:
                # truncate sentence at the tail
                tokens = tokens[:(max_seq_length - 1)] + [self.eos_tag]
            else:
                # truncate sentence at the beginning
                tokens = [self.sos_tag] + tokens[-(max_seq_length - 1):]
        else:
            tokens_lost = 0

        # add words to dictionary
        if add_to_dict:
            # always add words of the tgt sentences
            for word in tokens:
                self.dictionary.add_word(word)

        if reverse_tokens:
            tokens = tokens[::-1]

        return ' '.join(tokens), tokens_lost

    def get_data_from_sentences(self, data_list, max_n_examples=-1, max_seq_length=-1, truncate_at_tail=True,
                                reverse_tokens=False, debug=False, add_to_dict=True):
        """
        Reads a list of sentences
        :param data_list: list of sentences
        :param max_n_examples: max number of sentences to return
        :param max_seq_length: max number of tokens per sentence
        :param truncate_at_tail: keep the beginning and ignore the end of the sentence
        :param reverse_tokens: reverse sentence tokens
        :param debug: print a few item examples
        :param add_to_dict: add words to dictionary
        :return: list of processed sentences
        """
        data = []
        truncated_sent = 0  # number of truncated sentences
        total_tokens_lost = 0     # number of tokens removed after truncation

        for sent in data_list:
            # process sentence
            processed_sent, tokens_lost = self.preprocess_sentence(
                sent,
                add_to_dict=add_to_dict,
                max_seq_length=max_seq_length,
                truncate_at_tail=truncate_at_tail,
                reverse_tokens=reverse_tokens
            )
            if tokens_lost > 0:
                total_tokens_lost += tokens_lost
                truncated_sent += 1

            data.append(processed_sent)

            if len(data) % 10000 == 0:
                print("#", end='')

            if 0 < max_n_examples <= len(data):
                break

        logger.info("")
        if debug:
            self.print_few_examples(data)

        if truncated_sent > 0:
            logger.info("Truncated %d (%d) / %d = %4f sentences" % (
                truncated_sent, total_tokens_lost, len(data), truncated_sent / len(data)
            ))

        return data

    def get_hreddata_from_array(self, data_list, max_n_examples=-1, max_context_size=-1, max_seq_length=-1,
                                reverse_tgt=False, debug=False, add_to_dict=True):
        """
        Reads a list of elements where each element is a list of sentences
        :param data_list: list of list of strings
        :param max_n_examples: consider top examples
        :param max_context_size: number of sentences to keep in the context
        :param max_seq_length: max number of tokens in one sequence
        :param reverse_tgt: reverse tokens of the tgt sequence
        :param debug: print a few item examples
        :param add_to_dict: add words to dictionary
        :return: (src, tgt) pair where src is all possible contexts and tgt is all the next sentences
        """
        src, tgt = [], []  # list of contexts & next sentences

        truncated_src = 0  # number of truncated source sequences
        truncated_tgt = 0  # number of truncated target sequences
        src_tokens_lost = 0  # number of tokens removed after truncation
        tgt_tokens_lost = 0  # number of tokens removed after truncation

        for sentences in data_list:

            start = 0  # index of the first sentences to keep in the `src`
            # for all sentences except the last one, make a (src - tgt) pair
            for s_id in range(len(sentences) - 1):
                if max_context_size > 0:
                    sentences_to_consider = sentences[start: s_id + 1]
                    # move pointer to the right when `src` reached its max capacity
                    if len(sentences_to_consider) == max_context_size:
                        start += 1  # sliding window ->->
                else:
                    # take all sentences before i+1 as src
                    sentences_to_consider = sentences[:s_id + 1]

                # perform BPE in advance for source sentences due to ' <eos> <sos> '.join()
                if self.bpe is not None:
                    for i, sent in enumerate(sentences_to_consider):
                        sentences_to_consider[i] = self.bpe.process_line(sent)
                # process source sentences
                processed_src, tokens_lost = self.preprocess_sentence(
                    (' ' + self.eos_tag + ' ' + self.sos_tag + ' ').join(sentences_to_consider),
                    add_to_dict=add_to_dict,
                    max_seq_length=max_seq_length,
                    truncate_at_tail=False,  # ignore the beginning of the sentence & keep the end
                    reverse_tokens=False,
                    bpe=False  # do not perform BPE again
                )
                if tokens_lost > 0:
                    src_tokens_lost += tokens_lost
                    truncated_src += 1

                # process target sentence
                processed_tgt, tokens_lost = self.preprocess_sentence(
                    sentences[s_id + 1],
                    add_to_dict=add_to_dict,
                    max_seq_length=max_seq_length,
                    truncate_at_tail=True,  # ignore the end of the sentence & keep the beginning
                    reverse_tokens=reverse_tgt
                )
                if tokens_lost > 0:
                    tgt_tokens_lost += tokens_lost
                    truncated_tgt += 1

                src.append(processed_src)
                tgt.append(processed_tgt)

                if len(src) % 10000 == 0:
                    print("#", end='')

                if 0 < max_n_examples <= len(src):
                    break

            if 0 < max_n_examples <= len(src):
                break

        logger.info("")

        if debug:
            self.print_few_hred_examples(src, tgt)

        if truncated_src > 0:
            logger.info("Truncated %d (%d) / %d = %4f source sentences" % (
                truncated_src, src_tokens_lost, len(src), truncated_src / len(src)
            ))
        if truncated_tgt > 0:
            logger.info("Truncated %d (%d) / %d = %4f target sentences" % (
                truncated_tgt, tgt_tokens_lost, len(tgt), truncated_tgt / len(tgt)
            ))

        return src, tgt

    def get_copydata_from_array(self, json_path, max_n_examples=-1, max_seq_length=-1,
                                debug=False, add_to_dict=True):
        """
        Reads an array where each item is considered as one story with sentences splitted by \n.
        :param json_path: path to a json file
        :param max_n_examples: consider top examples
        :param max_seq_length: max number of tokens in one sequence
        :param debug: print a few item examples
        :param add_to_dict: add words to dictionary
        :return: list of (src, tgt) pairs where src and tgt are the same sentence
        """
        # check if data has been preprocessed before
        data = self.already_preprocessed(
            json_path, max_context_size=-1, max_seq_length=max_seq_length, reverse_tgt=False
        )
        if data is not None:
            src, tgt = data

            # add words to dictionary
            if add_to_dict:
                # add all words in the source sentence
                for src_words in src:
                    for word in src_words.split():
                        self.dictionary.add_word(word)
                # add all words in the target sentence
                for tgt_words in tgt:
                    for word in tgt_words.split():
                        self.dictionary.add_word(word)

            # remove extra examples if needed
            if 0 < max_n_examples <= len(src):
                src = src[:max_n_examples]
                tgt = tgt[:max_n_examples]
                return src, tgt
            else:
                logger.info("previously processed data has only %d examples" % len(src))
                logger.info("this experiment asked for %d examples" % max_n_examples)
                reprocess_data = input("reprocess data (yes): ")
                if reprocess_data.lower() in ['n', 'no', '0']:
                    logger.info("ok, working with %d examples then..." % len(src))
                    return src, tgt

        # else:
        src = []  # list of contexts
        tgt = []  # list of next sentences

        truncated_src = 0  # number of truncated source sequences
        truncated_tgt = 0  # number of truncated target sequences
        src_tokens_lost = 0  # number of tokens removed after truncation
        tgt_tokens_lost = 0  # number of tokens removed after truncation

        # f = open(json_path, 'r')
        # array = json.load(f)
        # f.close()
        array = json_path

        logger.info("%d items" % len(array))

        for item in array:

            # skip empty items
            if len(item.strip().split()) == 0:
                continue

            sentences = item.split('\n')  # list of sentences in this item

            # for all sentences add words to dictionary & make a (src - tgt) pair
            for s in sentences:

                # Process BPE this sentence
                if self.bpe is not None:
                    s = self.bpe.process_line(s)

                # lowercase, strip, to ascii
                src_sent = normalize_string(
                    self.sos_tag + ' ' + s + ' ' + self.eos_tag
                )
                tgt_sent = normalize_string(
                    self.sos_tag + ' ' + s + ' ' + self.eos_tag
                )

                # list of words in the source sentences
                src_words = word_tokenize(src_sent)
                src_words = undo_word_tokenizer(src_words, self.sos_tag)
                src_words = undo_word_tokenizer(src_words, self.unk_tag)
                src_words = undo_word_tokenizer(src_words, self.eos_tag)
                if self.bpe is not None:
                    src_words = put_back_bpe_separator(src_words, self.bpe.separator)

                # truncate if too long
                if 0 < max_seq_length < len(src_words):
                    src_tokens_lost += len(src_words) - max_seq_length
                    # truncate source sentence at the beginning
                    src_words = [self.sos_tag] + src_words[-(max_seq_length - 1):]
                    truncated_src += 1

                # list of words in the target sentence
                tgt_words = word_tokenize(tgt_sent)
                tgt_words = undo_word_tokenizer(tgt_words, self.sos_tag)
                tgt_words = undo_word_tokenizer(tgt_words, self.unk_tag)
                tgt_words = undo_word_tokenizer(tgt_words, self.eos_tag)
                if self.bpe is not None:
                    tgt_words = put_back_bpe_separator(tgt_words, self.bpe.separator)

                # truncate if too long
                if 0 < max_seq_length < len(tgt_words):
                    tgt_tokens_lost += len(tgt_words) - max_seq_length
                    # truncate target sentence at the tail
                    tgt_words = tgt_words[:(max_seq_length - 1)] + [self.eos_tag]
                    truncated_tgt += 1

                # add words to dictionary
                if add_to_dict:
                    for word in src_words:
                        self.dictionary.add_word(word)

                src.append(' '.join(src_words))
                tgt.append(' '.join(tgt_words))

                if len(src) % 10000 == 0:
                    print("#", end='')

                if 0 < max_n_examples <= len(src):
                    break

            if 0 < max_n_examples <= len(src):
                break

        logger.info("")

        if debug:
            self.print_few_hred_examples(src, tgt)

        if truncated_src > 0:
            logger.info("Truncated %d (%d) / %d = %4f source sentences" % (
                truncated_src, src_tokens_lost, len(src), truncated_src / len(src)
            ))
        if truncated_tgt > 0:
            logger.info("Truncated %d (%d) / %d = %4f target sentences" % (
                truncated_tgt, tgt_tokens_lost, len(tgt), truncated_tgt / len(tgt)
            ))

        self.save_preprocessed_data(json_path, (src, tgt), max_seq_length=max_seq_length)

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
        :param h_t: hidden state of the decoder ~(bs, dec_len=1, dec_size)
        :param outs: output vectors of the encoder ~(bs, max_src_len, n_dir*enc_size)
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
        else:
            raise NotImplementedError("Unknown attention method:", self.method)

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


def count_parameters(model):
    """
    Return the number of trainable parameters in a Pytorch model
    :param model: torch.nn Model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GaussianNoise(nn.Module):
    """
    Gaussian noise regularizer.
    Thanks to https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/4
    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x
