"""
# Implementation of HRED model in PyTorch
# Paper : https://arxiv.org/abs/1507.04808
# Tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
#
# python hred.py <training_data> <dictionary>
"""

import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    """
    Encoder working on word vectors, producing a sentence encoding

    Tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-encoder
    """
    def __init__(self, gate, vocab_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.gate = gate
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.n_dir = 2
        else:
            self.n_dir = 1

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        if gate == 'gru':
            self.rnn = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
                dropout=dropout,
                bidirectional=self.bidirectional
            )

        elif gate == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
                dropout=dropout,
                bidirectional=self.bidirectional
            )

        else:
            raise NotImplementedError("unknown encoder gate %s" % gate)

    def forward(self, x, h):
        """
        GRU doc:
        -in1- input ~(seq_len, batch, input_size): tensor containing the features of the input sequence.
                    The input can also be a packed variable length sequence.
                    See torch.nn.utils.rnn.pack_padded_sequence() for details.
        -in2- h_0 ~(num_layers*num_directions, batch, hidden_size): tensor containing the initial hidden
                  state for each element in the batch. Defaults to zero if not provided.
        -out1- output ~(seq_len, batch, num_directions*hidden_size): tensor containing the output
                      features h_t from the last layer of the GRU, for each t.
                      If a torch.nn.utils.rnn.PackedSequence has been given as the input,
                      the output will also be a packed sequence. For the unpacked case, the directions
                      can be separated using output.view(seq_len, batch, num_directions, hidden_size),
                      with forward and backward being direction 0 and 1 respectively.
                      Similarly, the directions can be separated in the packed case.
        -out2- h_n ~(num_layers*num_directions, batch, hidden_size): tensor containing the hidden
                   state for t = seq_len. Like output, the layers can be separated using
                   h_n.view(num_layers, num_directions, batch, hidden_size).

        LSTM doc:
        -in1- input ~(seq_len, batch, input_size): tensor containing the features of the input sequence.
                    The input can also be a packed variable length sequence.
                    See torch.nn.utils.rnn.pack_padded_sequence() or torch.nn.utils.rnn.pack_sequence() for details.
        -in2- h_0 ~(num_layers*num_directions, batch, hidden_size): tensor containing the initial hidden
                  state for each element in the batch. Defaults to zero if not provided.
        -in3- c_0 ~(num_layers*num_directions, batch, hidden_size): tensor containing the initial cell
                  state for each element in the batch. Defaults to zero if not provided.
        -out1- output ~(seq_len, batch, num_directions*hidden_size): tensor containing the output
                      features (h_t) from the last layer of the LSTM, for each t.
                      If a torch.nn.utils.rnn.PackedSequence has been given as the input,
                      the output will also be a packed sequence. For the unpacked case, the directions
                      can be separated using output.view(seq_len, batch, num_directions, hidden_size),
                      with forward and backward being direction 0 and 1 respectively.
                      Similarly, the directions can be separated in the packed case.
        -out2- h_n ~(num_layers*num_directions, batch, hidden_size): tensor containing the hidden
                   state for t = seq_len. Like output, the layers can be separated using
                   h_n.view(num_layers, num_directions, batch, hidden_size).
        -out3- c_n ~(num_layers*num_directions, batch, hidden_size): tensor containing the cell
                   state for t = seq_len. Like output, the layers can be separated using
                   c_n.view(num_layers, num_directions, batch, hidden_size).
        """
        x = self.embedding(x)  # ~(bs, seq, size)

        if self.gate == 'lstm':
            h, c = h  # decompose lstm unit into hidden state & cell state
            out, (h, c) = self.rnn(x, h, c)
            # out ~(bs, seq, n_dir*size)
            # h   ~(bs, n_dir*n_layers, size)
            # c   ~(bs, n_dir*n_layers, size)

            # separate the directions with forward and backward being direction 0 and 1 respectively.
            out = out.view(out.size(0), out.size(1), self.n_dir, self.hidden_size)
            h = h.view(h.size(0), self.n_layers, self.n_dir, self.hidden_size)
            c = h.view(c.size(0), self.n_layers, self.n_dir, self.hidden_size)
            # out ~(bs, seq, n_dir, size)
            # h   ~(bs, n_layers, n_dir, size)
            # c   ~(bs, n_layers, n_dir, size)

            h = (h, c)  # merge back the lstm unit

        else:
            out, h = self.rnn(x, h)
            # out ~(bs, seq, n_dir*size)
            # h   ~(bs, n_dir*n_layers, size)

            # separate the directions with forward and backward being direction 0 and 1 respectively.
            out = out.view(out.size(0), out.size(1), self.n_dir, self.hidden_size)
            h = h.view(h.size(0), self.n_layers, self.n_dir, self.hidden_size)
            # out ~(bs, seq, n_dir, size)
            # h   ~(bs, n_layers, n_dir, size)

        return out, h

    def init_hidden(self, bs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.gate == 'lstm':
            return (
                torch.zeros(bs, self.n_layers*self.n_dir, self.hidden_size).to(device),
                torch.zeros(bs, self.n_layers*self._n_dir, self.hidden_size).to(device)
            )
        else:
            return torch.zeros(bs, self.n_layers*self._n_dir, self.hidden_size).to(device)


class ContextRNN(nn.Module):
    """
    Encoder working on sentence vectors, producing a context encoding

    Tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-encoder
    """
    def __init__(self, gate, hidden_size, n_layers=1, dropout=0.1, bidirectional=True):
        super(ContextRNN, self).__init__()
        self.gate = gate
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.n_dir = 2
        else:
            self.n_dir = 1

        if gate == 'gru':
            self.rnn = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
                dropout=dropout,
                bidirectional=self.bidirectional
            )

        elif gate == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
                dropout=dropout,
                bidirectional=self.bidirectional
            )

        else:
            raise NotImplementedError("unknown encoder gate %s" % gate)

    def forward(self, x, h):
        """
        :param x: input sequence of vectors ~(bs, seq, size)
        :param h: initial hidden state ~(bs, n_dir*n_layers, size)
        :return: out ~(bs, seq, n_dir, size): output features h_t from the last layer, for each t
                 h ~(bs, n_layers, n_dir, size): hidden state for t = seq_len
        """
        if self.gate == 'lstm':
            h, c = h  # decompose lstm unit into hidden state & cell state
            out, (h, c) = self.rnn(x, h, c)
            # out ~(bs, seq, n_dir*size)
            # h   ~(bs, n_dir*n_layers, size)
            # c   ~(bs, n_dir*n_layers, size)

            # separate the directions with forward and backward being direction 0 and 1 respectively.
            out = out.view(out.size(0), out.size(1), self.n_dir, self.hidden_size)
            h = h.view(h.size(0), self.n_layers, self.n_dir, self.hidden_size)
            c = h.view(c.size(0), self.n_layers, self.n_dir, self.hidden_size)
            # out ~(bs, seq, n_dir, size)
            # h   ~(bs, n_layers, n_dir, size)
            # c   ~(bs, n_layers, n_dir, size)

            h = (h, c)  # merge back the lstm unit

        else:
            out, h = self.rnn(x, h)
            # out ~(bs, seq, n_dir*size)
            # h   ~(bs, n_dir*n_layers, size)

            # separate the directions with forward and backward being direction 0 and 1 respectively.
            out = out.view(out.size(0), out.size(1), self.n_dir, self.hidden_size)
            h = h.view(h.size(0), self.n_layers, self.n_dir, self.hidden_size)
            # out ~(bs, seq, n_dir, size)
            # h   ~(bs, n_layers, n_dir, size)

        return out, h

    def init_hidden(self, bs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.gate == 'lstm':
            return (
                torch.zeros(bs, self.n_layers*self.n_dir, self.hidden_size).to(device),
                torch.zeros(bs, self.n_layers*self._n_dir, self.hidden_size).to(device)
            )
        else:
            return torch.zeros(bs, self.n_layers*self._n_dir, self.hidden_size).to(device)


class AttnDecoderRNN(nn.Module):
    """
    Decoder network taking as input the context vector, the previous hidden state,
    and the previously predicted token (or the ground truth previous token ie: teacher forcing)

    The context vector is concatenated with the previous hidden state to form the new hidden state.

    Tutorial: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#attention-decoder
    - - -
    Attention allows the decoder network to "focus" on a different part of the encoder's outputs
    for every step of the decoder's own outputs.
    First we calculate a set of attention weights.
    These will be multiplied by the encoder output vectors to create a weighted combination.
    The result (called attn_applied in the code) should contain information about that specific
    part of the input sequence, and thus help the decoder choose the right output words.
    - -
    Calculating the attention weights is done with another feed-forward layer (attn),
    using the decoder's input and hidden state as inputs.
    Because there are sentences of all sizes in the training data, to actually create
    and train this layer we have to choose a maximum sentence length (input length,
    for encoder outputs) that it can apply to.
    Sentences of the maximum length will use all the attention weights, while shorter
    sentences will only use the first few.
    """
    def __init__(self, gate, hidden_size, vocab_size, n_layers=1, dropout=0.1, max_len=64):
        super(AttnDecoderRNN, self).__init__()

        self.gate = gate
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        if gate == 'gru':
            self.rnn = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
                dropout=dropout,
                bidirectional=False
            )

        elif gate == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
                dropout=dropout,
                bidirectional=False
            )

        else:
            raise NotImplementedError("unknown encoder gate %s" % gate)

        # Attention layers
        self.attn = nn.Linear(self.hidden_size * 2, max_length)
        self.att_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Output layer mapping back to large vocabulary
        self.out = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, *input):
        pass


# TODO: continue with online tuto: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#attention-decoder
# also look at Koustuv code here: https://github.com/koustuvsinha/hred-py/blob/master/hred_pytorch.py
