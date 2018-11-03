"""
Tutorial followed from Pytorch Github Tuto
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/main.py

also look at: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

import sys
sys.path.append('..')

from utils import Dictionary, Corpus


if __name__ == '__main__':

    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    embed_size = 128
    hidden_size = 1024
    num_layers = 2
    num_epochs = 5
    num_samples = 1000  # number of words to be sampled
    batch_size = 8
    seq_length = 30
    learning_rate = 0.002

    # Load dataset
    corpus = Corpus()
    idx_data = corpus.get_data('train.txt')  # ~(line, max_len)
    vocab_size = len(corpus.dictionary)
    num_batches = idx_data.size(0) // batch_size

    print("ids:", idx_data, idx_data.size())
    print("str:")
    for sent in corpus.to_str(idx_data):
        print(sent)
    print("vocab:", vocab_size)
    print("num_batches:", num_batches)
