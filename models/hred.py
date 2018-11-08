"""
# Implementation of HRED model in PyTorch
# Paper : https://arxiv.org/abs/1507.04808
# Tutorial: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
#
# python hred.py <training_data> <dictionary>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import AttentionModule


class EncoderRNN(nn.Module):
    """
    Encoder working on word vectors, producing a sentence encoding
    The encoder will take a batch of word sequences, a LongTensor of size (batch_size x max_len),
    and output an encoding for each word, a FloatTensor of size (batch_size x max_len x hidden_size)

    Tutorial: https://render.githubusercontent.com/view/ipynb?commit=c520c52e68e945d88fff563dba1c028b6ec0197b&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f7370726f2f70726163746963616c2d7079746f7263682f633532306335326536386539343564383866666635363364626131633032386236656330313937622f736571327365712d7472616e736c6174696f6e2f736571327365712d7472616e736c6174696f6e2d626174636865642e6970796e62&nwo=spro%2Fpractical-pytorch&path=seq2seq-translation%2Fseq2seq-translation-batched.ipynb&repository_id=79684696&repository_type=Repository#The-Encoder
    """
    def __init__(self, rnn_type, vocab_size, embedding_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.n_dir = 2
        else:
            self.n_dir = 1

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=embedding_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
                dropout=dropout,
                bidirectional=self.bidirectional
            )

        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=embedding_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
                dropout=dropout,
                bidirectional=self.bidirectional
            )

        else:
            raise NotImplementedError("unknown encoder type %s" % self.rnn_type)

    def forward(self, x, lengths, h_0=None):
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
        x = self.embedding(x)  # ~(bs, seq, embedding_size)
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)

        if self.rnn_type == 'lstm':
            # decompose lstm unit into hidden state & cell state
            if h_0 is not None:
                h_0, c_0 = h_0
            else:
                c_0 = None

            out, (h_t, c_t) = self.rnn(packed, h_0, c_0)
            # unpack (back to padded)
            out, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(out)
            # out ~(bs, seq, n_dir*size)
            # h_t ~(bs, n_dir*n_layers, size)
            # c_t ~(bs, n_dir*n_layers, size)

            '''
            # separate the directions with forward and backward being direction 0 and 1 respectively.
            out = out.view(out.size(0), out.size(1), self.n_dir, self.hidden_size)
            h_t = h_t.view(h_t.size(0), self.n_layers, self.n_dir, self.hidden_size)
            c_t = c_t.view(c_t.size(0), self.n_layers, self.n_dir, self.hidden_size)
            # out ~(bs, seq, n_dir, size)
            # h_t ~(bs, n_layers, n_dir, size)
            # c_t ~(bs, n_layers, n_dir, size)
            '''

            h_t = (h_t, c_t)  # merge back the lstm unit

        else:
            out, h_t = self.rnn(packed, h_0)
            # unpack (back to padded)
            out, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(out)
            # out ~(bs, seq, n_dir*size)
            # h_t ~(bs, n_dir*n_layers, size)

            '''
            # separate the directions with forward and backward being direction 0 and 1 respectively.
            out = out.view(out.size(0), out.size(1), self.n_dir, self.hidden_size)
            h_t = h_t.view(h_t.size(0), self.n_layers, self.n_dir, self.hidden_size)
            # out ~(bs, seq, n_dir, size)
            # h_t ~(bs, n_layers, n_dir, size)
            '''

        return out, h_t

    def init_hidden(self, bs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.rnn_type == 'lstm':
            return (
                torch.zeros(bs, self.n_layers*self.n_dir, self.hidden_size).to(device),
                torch.zeros(bs, self.n_layers*self.n_dir, self.hidden_size).to(device)
            )
        else:
            return torch.zeros(bs, self.n_layers*self.n_dir, self.hidden_size).to(device)


class ContextRNN(nn.Module):
    """
    Encoder working on sentence vectors, producing a context encoding
    The encoder will take a batch of sentence encodings, a LongTensor of size (bs, seq, n_dir*size),
    and output an encoding for each sentence, a FloatTensor of size (batch_size x max_len x hidden_size)

    TODO: add attention between this encoder and the previous one
    """
    def __init__(self, rnn_type, input_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=True):
        super(ContextRNN, self).__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.n_dir = 2
        else:
            self.n_dir = 1

        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
                dropout=dropout,  # add Dropout on the outputs of each layer except the last one
                bidirectional=self.bidirectional
            )

        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
                dropout=dropout,  # add Dropout on the outputs of each layer except the last one
                bidirectional=self.bidirectional
            )

        else:
            raise NotImplementedError("unknown encoder type %s" % self.rnn_type)

    def forward(self, x, lengths, h_0=None):
        """
        :param x: input sequence of vectors ~(bs, seq, size)
        :param h_0: initial hidden state ~(bs, n_dir*n_layers, size)
        :return: out ~(bs, seq, n_dir*size): output features h_t from the last layer, for each t
                 h ~(bs, n_layers*n_dir, size): hidden state for t = seq_len
        """
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)

        if self.rnn_type == 'lstm':
            # decompose lstm unit into hidden state & cell state
            if h_0 is not None:
                h_0, c_0 = h_0
            else:
                c_0 = None

            out, (h_t, c_t) = self.rnn(packed, h_0, c_0)
            # unpack (back to padded)
            out, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(out)
            # out ~(bs, seq, n_dir*size)
            # h_t ~(bs, n_dir*n_layers, size)
            # c_t ~(bs, n_dir*n_layers, size)

            '''
            # separate the directions with forward and backward being direction 0 and 1 respectively.
            out = out.view(out.size(0), out.size(1), self.n_dir, self.hidden_size)
            h_t = h_t.view(h_t.size(0), self.n_layers, self.n_dir, self.hidden_size)
            c_t = c_t.view(c_t.size(0), self.n_layers, self.n_dir, self.hidden_size)
            # out ~(bs, seq, n_dir, size)
            # h_t ~(bs, n_layers, n_dir, size)
            # c_t ~(bs, n_layers, n_dir, size)
            '''

            h_t = (h_t, c_t)  # merge back the lstm unit

        else:
            out, h_t = self.rnn(packed, h_0)
            # unpack (back to padded)
            out, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(out)
            # out ~(bs, seq, n_dir*size)
            # h_t ~(bs, n_dir*n_layers, size)

            '''
            # separate the directions with forward and backward being direction 0 and 1 respectively.
            out = out.view(out.size(0), out.size(1), self.n_dir, self.hidden_size)
            h_t = h_t.view(h_t.size(0), self.n_layers, self.n_dir, self.hidden_size)
            # out ~(bs, seq, n_dir, size)
            # h_t ~(bs, n_layers, n_dir, size)
            '''

        return out, h_t

    def init_hidden(self, bs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.rnn_type == 'lstm':
            return (
                torch.zeros(bs, self.n_layers*self.n_dir, self.hidden_size).to(device),
                torch.zeros(bs, self.n_layers*self.n_dir, self.hidden_size).to(device)
            )
        else:
            return torch.zeros(bs, self.n_layers*self.n_dir, self.hidden_size).to(device)


class HREDDecoder(nn.Module):
    """
    Decoder network taking as input the context vector, the previous hidden state,
    and the previously predicted token (or the ground truth previous token ie: teacher forcing)

    The context vector is concatenated with the previous hidden state to form an intermediate (big) hidden state.
    The intermediate (big) hidden state is passed to a linear layer and reduced to the original hidden state before
    being fed into the RNN.

    Tutorial: https://github.com/placaille/nmt-comp550/blob/master/src/model.py#L101-L130

    Extra: Boltzmann Temperature term (alpha) to improve sample quality
    ``
    if o_t is the generator's pre-logit activation and W is the word embedding matrix then the conditional
    distribution of the generator is given by G(x_t | x_1:t−1) = softmax(o_t . W / alpha).
    Decreasing alpha below 1.0 will increase o_t and thus decrease the entropy of G's conditional probability.
    This is a useful tool to reduce the probability of mistakes in NLG and thus improve sample quality.
    Concretely, temperature tuning naturally moves the model in quality/diversity space.
    ``
    from: https://arxiv.org/pdf/1811.02549.pdf
    """
    def __init__(self, rnn_type, vocab_size, embedding_size, hidden_size, context_size,
                 n_layers=1, dropout=0.1, alpha=1.0):
        """
        :param rnn_type: 'lstm' or 'gru'
        :param vocab_size: number of tokens in vocabulary
        :param embedding_size: embedding size of all tokens
        :param hidden_size: size of RNN hidden state
        :param context_size: size of the previously encoded context
        :param alpha: Boltzmann Temperature term
        """
        super(HREDDecoder, self).__init__()

        self.rnn_type     = rnn_type
        self.hidden_size  = hidden_size
        self.n_layers     = n_layers
        self.alpha        = alpha

        self.embedding           = nn.Embedding(vocab_size, embedding_size)
        self.dropout             = nn.Dropout(dropout)
        self.intermediate_hidden = nn.Linear(self.hidden_size + context_size, self.hidden_size)
        self.output              = nn.Linear(self.hidden_size, vocab_size)

        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=embedding_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
                dropout=dropout,  # add Dropout on the outputs of each layer except the last one
                bidirectional=False
            )

        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=embedding_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
                dropout=dropout,  # add Dropout on the outputs of each layer except the last one
                bidirectional=False
            )

        else:
            raise NotImplementedError("unknown encoder type %s" % self.rnn_type)

    def forward(self, x, h_tm1, cntx):
        """
        :param x: batch of input tokens - LongTensor ~(bs)
        :param h_tm1: previous hidden state ~(bs, n_layers, hidden_size)
        :param cntx: context vector ~(bs, n_layers, context_size)
        """
        x = self.embedding(x).view(x.size(0), 1, -1)  # ~(bs, seq=1, embedding_size)
        x = self.dropout(x)

        if self.rnn_type == 'lstm':
            # decompose lstm unit into hidden state & cell state
            if h_tm1 is not None:
                h_tm1, c_tm1 = h_tm1
            else:
                c_tm1 = None

            # concatenate the context and project it to hidden_size
            decoder_hidden = torch.cat((h_tm1, cntx), 2)  # ~(bs, n_layers, hidden_size + context_size)
            decoder_hidden = self.intermediate_hidden(decoder_hidden)  # ~(bs, n_layers, hidden_size)

            out, (h_t, c_t) = self.rnn(x, decoder_hidden, c_tm1)
            # out ~(bs, seq=1, hidden_size)
            # h_t ~(bs, n_layers, hidden_size)
            # c_t ~(bs, n_layers, hidden_size)

            h_t = (h_t, c_t)  # merge back the lstm unit

        else:
            # concatenate the context and project it to hidden_size
            decoder_hidden = torch.cat((h_tm1, cntx), 2)  # ~(bs, n_layers, hidden_size + context_size)
            decoder_hidden = self.intermediate_hidden(decoder_hidden)  # ~(bs, n_layers, hidden_size)

            out, h_t = self.rnn(x, decoder_hidden)
            # out ~(bs, seq=1, hidden_size)
            # h_t ~(bs, n_layers, hidden_size)

        out = out.view(out.size(0), -1)  # ~(bs, hidden_size)
        out = self.output(out)  # ~(bs, vocab_size)
        out /= self.alpha  # divide by Boltzmann Temperature term before applying softmax

        return out, h_t

    def init_hidden(self, bs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.rnn_type == 'lstm':
            return (
                torch.zeros(bs, self.n_layers, self.hidden_size).to(device),
                torch.zeros(bs, self.n_layers, self.hidden_size).to(device)
            )
        else:
            return torch.zeros(bs, self.n_layers, self.hidden_size).to(device)


# TODO: after implementing attention module in utils, continue tuto:
# https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
# https://github.com/placaille/nmt-comp550/blob/master/src/model.py#L133


class AttnDecoderRNN(nn.Module):
    """
    Decoder network taking as input the context vector, the previous hidden state,
    and the previously predicted token (or the ground truth previous token ie: teacher forcing)

    The context vector is concatenated with the previous hidden state to form the new hidden state.

    Tutorial: https://render.githubusercontent.com/view/ipynb?commit=c520c52e68e945d88fff563dba1c028b6ec0197b&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f7370726f2f70726163746963616c2d7079746f7263682f633532306335326536386539343564383866666635363364626131633032386236656330313937622f736571327365712d7472616e736c6174696f6e2f736571327365712d7472616e736c6174696f6e2d626174636865642e6970796e62&nwo=spro%2Fpractical-pytorch&path=seq2seq-translation%2Fseq2seq-translation-batched.ipynb&repository_id=79684696&repository_type=Repository#Attention-Decoder
    - - -
    Attention allows the decoder network to "focus" on a different part of the encoder's outputs
    for every step of the decoder's own outputs.
    First we calculate a set of attention weights.
    These will be multiplied by the encoder output vectors to create a weighted combination.
    The result (called attn_applied in the code) should contain information about that specific
    part of the input sequence, and thus help the decoder choose the right output words.
    - - -
    Calculating the attention weights is done with another feed-forward layer (attn),
    using the decoder's input and hidden state as inputs.
    Because there are sentences of all sizes in the training data, to actually create
    and train this layer we have to choose a maximum sentence length (input length,
    for encoder outputs) that it can apply to.
    Sentences of the maximum length will use all the attention weights, while shorter
    sentences will only use the first few.
    """
    def __init__(self, rrn_type, hidden_size, vocab_size, n_layers=1, dropout=0.1, max_length=64):
        super(AttnDecoderRNN, self).__init__()

        self.rrn_type = rrn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        if self.rrn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
                dropout=dropout,  # add Dropout on the outputs of each layer except the last one
                bidirectional=False
            )

        elif self.rrn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
                dropout=dropout,  # add Dropout on the outputs of each layer except the last one
                bidirectional=False
            )

        else:
            raise NotImplementedError("unknown encoder type %s" % self.rrn_type)

        # attn takes in previous hidden state (h_t-1) and current input (x_t)
        # attn is used to computes attention weights
        self.attn = nn.Linear(self.hidden_size * 2, max_length)
        # attn_combine takes in the weighted context (c_attn) and current input (x_t)
        # attn_combine is used to re-evaluate c_attn in terms of x_t
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # Dropout layer used after the input embeddings
        self.dropout = nn.Dropout(dropout)
        # Output layer mapping back to large vocabulary
        self.out = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, x, h_0, c):
        """
        :param x: input sequence (teacher forced or not) ~(bs, seq)
        :param h_0: initial hidden state ~(bs, n_dir*n_layers, size)
        :param c: encoder output

        :return: out ~(bs, seq, n_dir, size): output features h_t from the last layer, for each t
                 h ~(bs, n_layers, n_dir, size): hidden state for t = seq_len
                 attn_weights
        """
        x = self.embedding(x)  # ~(bs, seq, size)
        x = self.dropout(x)

        att_weights = F.softmax(
            self.attn(torch.cat((x, h_0), 1)), dim=1
        )


# TUTO:
# pytorch tuto notebooks with batch https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation
# Lucas & Philippe code: https://github.com/placaille/nmt-comp550/tree/master/src

