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

from utils import Dictionary, Corpus


class EncoderLayer(nn.Module):
    """
    Encoder is made up of embedding layer + some lstms
    """
    def __init__(self, in_size: int, out_size: int, dropout=0.1, bidir=True):
        super(EncoderLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.lstm = nn.LSTM(input_size=self.in_size,
                            hidden_size=self.out_size,
                            batch_first=True,  # input and output tensors are provided as (bs, seq, feature)
                            dropout=dropout,
                            bidirectional=bidir)

    def forward(self, x, mask):
        # apply self attention
        x = self.sublayer[0](x, lambda a: self.self_attn.forward(a, a, a, mask))
        # apply feed-forward
        return self.sublayer[1](x, self.feed_forward)


def make_model(src_vocab: int, tgt_vocab: int, n: int,
               d_model=512, d_ff=2048,
               h=8, dropout=0.1):
    """
    Helper: construct a model from hyper parameters
    :param src_vocab: vocab size at the input level
    :param tgt_vocab: vocab size at the output level
    :param n: number of Encoder and Decoder layers
    :param d_model: dim of model (embeddings)
    :param d_ff: dim of feed-forward net
    :param h: number of attention heads
    :param dropout: dropout rate
    :return: an EncoderDecoder model
    """
    from models.encdec import \
        EncoderDecoder, \
        Encoder, \
        Decoder, \
        Generator

    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(
            d_model, c(attn), c(ff), dropout
        ), n=n),
        Decoder(DecoderLayer(
            d_model, c(attn), c(attn), c(ff), dropout
        ), n=n),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    return model


