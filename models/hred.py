"""
# Implementation of HRED model in PyTorch
# Paper : https://arxiv.org/abs/1507.04808
# python hred.py <training_data> <dictionary>
"""

import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    """
    Encoder working on word vectors, producing a sentence encoding
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
            h = (h, c)  # merge back the lstm unit

        else:
            out, h = self.rnn(x, h)
            # out ~(bs, seq, n_dir*size)
            # h   ~(bs, n_dir*n_layers, size)

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
        if self.gate == 'lstm':
            h, c = h  # decompose lstm unit into hidden state & cell state
            out, (h, c) = self.rnn(x, h, c)
            # out ~(bs, seq, n_dir*size)
            # h   ~(bs, n_dir*n_layers, size)
            # c   ~(bs, n_dir*n_layers, size)
            h = (h, c)  # merge back the lstm unit

        else:
            out, h = self.rnn(x, h)
            # out ~(bs, seq, n_dir*size)
            # h   ~(bs, n_dir*n_layers, size)

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

# TODO: continue ...
# check Koustuv implementation: https://github.com/koustuvsinha/hred-py/blob/master/hred_pytorch.py
# check pytorch tuto: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/main.py#L30-L50
