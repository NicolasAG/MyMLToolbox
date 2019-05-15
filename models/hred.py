"""
# Implementation of HRED model in PyTorch
# Paper : https://arxiv.org/abs/1507.04808
"""
import random
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from MyMLToolbox.utils import AttentionModule, GaussianNoise, split_list


logger = logging.getLogger(__name__)


class HREDEncoder(nn.Module):
    """
    Parent class of both sentence encoder and context encoder.

    Tutorial: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
              |_> "The Encoder"
    """
    def __init__(self, rnn_type, input_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=True):
        super(HREDEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        if bidirectional:
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
                dropout=dropout,
                bidirectional=bidirectional
            )

        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
                dropout=dropout,
                bidirectional=bidirectional
            )

        else:
            raise NotImplementedError("unknown rnn type %s" % self.rnn_type)

        self._init_weights()

    def forward(self, x, lengths, h_0=None):
        """
        :param x: input sequence of vectors ~(bs, max_src_len, size)
        :param lengths: length of each sequence in x ~(bs)
        :param h_0: initial hidden state ~(n_dir*n_layers, bs, size)
        :return: out ~(bs, max_src_len, n_dir*size): output features h_t from the last layer, for each t
                 h   ~(n_layers*n_dir, bs, size): hidden state for t = seq_len
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # sort input by its length for pack_padded_sequence
        x, sorted_idx, sorted_lengths = self._sort_by_length(
            x, torch.LongTensor(lengths).to(device), dim=0
        )

        packed = torch.nn.utils.rnn.pack_padded_sequence(x, sorted_lengths, batch_first=True)

        if self.rnn_type == 'lstm':
            out, (h_t, c_t) = self.rnn(packed, h_0)
            # unpack (back to padded)
            out, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # out ~(bs, max_src_len, n_dir*size)
            # h_t ~(n_dir*n_layers, bs, size)
            # c_t ~(n_dir*n_layers, bs, size)

            # unsort tensor batches to match original input order
            out = self._unsort_tensor(out, sorted_idx, dim=0)
            h_t = self._unsort_tensor(h_t, sorted_idx, dim=1)  # batch index is on dim 1
            c_t = self._unsort_tensor(c_t, sorted_idx, dim=1)  # batch index is on dim 1

            '''
            # separate the directions with forward and backward being direction 0 and 1 respectively.
            out = out.view(out.size(0), out.size(1), self.n_dir, self.hidden_size)
            h_t = h_t.view(self.n_layers, self.n_dir, h_t.size(1),  self.hidden_size)
            c_t = c_t.view(self.n_layers, self.n_dir, c_t.size(1), self.hidden_size)
            # out ~(bs, max_src_len, n_dir, size)
            # h_t ~(n_layers, n_dir, bs, size)
            # c_t ~(n_layers, n_dir, bs, size)
            '''

            h_t = (h_t, c_t)  # merge back the lstm unit

        else:
            out, h_t = self.rnn(packed, h_0)
            # unpack (back to padded)
            out, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # out ~(bs, max_src_len, n_dir*size)
            # h_t ~(n_dir*n_layers, bs, size)

            # unsort tensor batches to match original input order
            out = self._unsort_tensor(out, sorted_idx, dim=0)
            h_t = self._unsort_tensor(h_t, sorted_idx, dim=1)  # batch index is on dim 1

            '''
            # separate the directions with forward and backward being direction 0 and 1 respectively.
            out = out.view(out.size(0), out.size(1), self.n_dir, self.hidden_size)
            h_t = h_t.view(self.n_layers, self.n_dir, h_t.size(1), self.hidden_size)
            # out ~(bs, max_src_len, n_dir, size)
            # h_t ~(n_layers, n_dir, bs, size)
            '''

        return out, h_t  # ~(bs, seq, n_dir*size) & ~(n_layers*n_dir, bs, size)

    def _sort_by_length(self, sequence, lengths, dim=0):
        """
        Sort a Tensor content by the length of each sequence
        :param sequence: LongTensor ~ (bs, ...)
        :param lengths: LongTensor ~ (bs)
        :param dim: dimension along which the index tensor is applicable (dim of size bs)
        :return: LongTensor with sorted content & list of sorted indices & sorted lengths
        """
        # Sort lengths
        sorted_lengths, sorted_idx = lengths.sort(descending=True)
        # Sort variable tensor by indexing at the sorted_idx
        sequence = sequence.index_select(dim, sorted_idx)
        return sequence, sorted_idx, sorted_lengths

    def _unsort_tensor(self, sorted_tensor, sorted_idx, dim=0):
        """
        Revert a Tensor to its original order. Undo the `_sort_by_length` function
        :param sorted_tensor: Tensor with content sorted by length ~ (bs, ...)
        :param sorted_idx: list of ordered indices ~ (bs)
        :param dim: dimension along which the index tensor is applicable (dim of size bs)
        :return: Unsorted Tensor
        """
        # Sort the sorted idx to get the original positional idx
        _, pos_idx = sorted_idx.sort()
        # Unsort the tensor
        original = sorted_tensor.index_select(dim, pos_idx)
        return original

    def _init_weights(self):
        """
        Initialise hidden-to-hidden weights to be orthogonal
        """
        for name, param in self.rnn.named_parameters():
            if name.startswith('weight_hh'):
                torch.nn.init.orthogonal_(param.data)

    def init_hidden(self, bs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.rnn_type == 'lstm':
            return (
                torch.zeros(self.n_layers*self.n_dir, bs, self.hidden_size).to(device),
                torch.zeros(self.n_layers*self.n_dir, bs, self.hidden_size).to(device)
            )
        else:
            return torch.zeros(self.n_layers*self.n_dir, bs, self.hidden_size).to(device)


class SentenceEncoder(HREDEncoder):
    """
    Encoder working on word vectors, producing a sentence encoding
    The encoder will take a batch of word sequences, a LongTensor of size (batch_size x max_len),
    and output an encoding for each word, a FloatTensor of size (batch_size x max_len x hidden_size)
    """
    def __init__(self, rnn_type, vocab_size, embedding_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=True):
        """
        :param rnn_type: 'lstm' or 'gru'
        :param vocab_size: number of tokens in vocabulary
        :param embedding_size: embedding size of all tokens
        :param hidden_size: size of RNN hidden state
        """
        super(SentenceEncoder, self).__init__(
            rnn_type, embedding_size, hidden_size, n_layers, dropout, bidirectional
        )
        # extra for sentence encoder: define embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)

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
        # extra for sentence encoder: feed sequence to the embedding layer
        x = self.embedding(x)  # ~(bs, max_src_len, embedding_size)
        return super(SentenceEncoder, self).forward(x, lengths, h_0)


class ContextEncoder(HREDEncoder):
    """
    Encoder working on sentence vectors, producing a context encoding
    The encoder will take a batch of sentence encodings, a LongTensor of size (bs, seq, n_dir*size),
    and output an encoding for each sentence, a FloatTensor of size (batch_size x max_len x hidden_size)

    TODO: add attention between this encoder and the previous one
    """
    def __init__(self, rnn_type, input_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=True):
        """
        :param rnn_type: 'lstm' or 'gru'
        :param input_size: embedding size of the input vectors
        :param hidden_size: size of RNN hidden state
        """
        super(ContextEncoder, self).__init__(
            rnn_type, input_size, hidden_size, n_layers, dropout, bidirectional
        )

    def forward(self, x, lengths, h_0=None):
        """
        :param x: input sequence of vectors ~(bs, max_src_len, size)
        :param lengths: length of each sequence in x ~(bs)
        :param h_0: initial hidden state ~(n_dir*n_layers, bs, size)
        :return: out ~(bs, max_src_len, n_dir*size): output features h_t from the last layer, for each t
                h    ~(n_layers*n_dir, bs, size): hidden state for t = seq_len
        """
        return super(ContextEncoder, self).forward(x, lengths, h_0)


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
    def __init__(self, rnn_type, vocab_size, embedding_size, hidden_size, extra_inp_dim=0, extra_hid_dim=0,
                 n_layers=1, h_dropout=0.0, i_dropout=0.0, i_noise=0.0, alpha=1.0):
        """
        :param rnn_type: 'lstm' or 'gru'
        :param vocab_size: number of tokens in vocabulary
        :param embedding_size: embedding size of all tokens
        :param hidden_size: size of RNN hidden state
        :param extra_inp_dim: size of the previously encoded extra information to concat to decoder input vector
        :param extra_hid_dim: size of the previously encoded extra information to concat to decoder hidden vector
        :param n_layers: number of stacked rnn layers
        :param h_dropout: dropout between each rnn layers
        :param i_dropout: input dropout (on the word embeddings)
        :param i_noise: gaussian variance of input noise (on the word embedding)
        :param alpha: Boltzmann Temperature term (set to 1 during training, change during inference)
        """
        super(HREDDecoder, self).__init__()

        self.rnn_type     = rnn_type
        self.hidden_size  = hidden_size
        self.extra_h      = extra_hid_dim
        self.extra_i      = extra_inp_dim
        self.n_layers     = n_layers
        self.alpha        = alpha
        self.vocab_size   = vocab_size  # attribute used in test_hred.step()

        self.embedding    = nn.Embedding(vocab_size, embedding_size)
        self.dropout      = nn.Dropout(i_dropout) if i_dropout > 0.0 else None
        self.noise        = GaussianNoise(i_noise) if i_noise > 0.0 else None

        self.extra_to_inp = nn.Linear(self.extra_i + embedding_size, embedding_size) if extra_inp_dim > 0 else None
        self.extra_to_hid = nn.Linear(self.extra_h + self.hidden_size, self.hidden_size) if extra_hid_dim > 0 else None
        self.output       = nn.Linear(self.hidden_size, vocab_size)

        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=embedding_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,   # input and output tensors are provided as (batch, seq, feature)
                dropout=h_dropout,  # add Dropout on the outputs of each layer except the last one
                bidirectional=False
            )

        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=embedding_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                bias=True,
                batch_first=True,   # input and output tensors are provided as (batch, seq, feature)
                dropout=h_dropout,  # add Dropout on the outputs of each layer except the last one
                bidirectional=False
            )

        else:
            raise NotImplementedError("unknown rnn type %s" % self.rnn_type)

        self._init_weights()

    def forward(self, x, h_tm1, extra_i=None, extra_h=None, enc_outs=None):
        """
        :param x: batch of input tokens - LongTensor ~(bs)
        :param h_tm1: previous hidden state ~(n_layers, bs, hidden_size)
        :param extra_i: any other information
                        ~(n_layers, bs, size)
        :param extra_h: context vector of all encoder layers at the last time step or any other information
                        ~(n_layers, bs, size)
        :param enc_outs: only used for the attention decoder

        :return: out          ~(bs, vocab_size)
                 h_t          ~(n_layers, bs, hidden_size)
                 None
        """
        x = self.embedding(x).view(x.size(0), 1, -1)  # ~(bs, seq=1, embedding_size)
        if self.noise is not None:
            x = self.noise(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if extra_i is not None:
            assert self.extra_to_inp is not None
            # reshape extra_i to fit input space: from (n_layers, bs, size) to (bs, seq=1, size)
            extra_i = extra_i[-1]           # ~(bs, size)
            extra_i = extra_i.unsqueeze(1)  # ~(bs, 1, size)
            x = torch.tanh(self.extra_to_inp(torch.cat((x, extra_i), dim=2)))

        # decompose lstm unit into hidden state & cell state
        if self.rnn_type == 'lstm':
            h_tm1, c_tm1 = h_tm1

        if extra_h is not None:
            # concatenate the context and project it to hidden_size
            assert self.extra_to_hid is not None
            decoder_hidden = torch.cat((h_tm1, extra_h), dim=2)             # ~(n_layers, bs, hidden_size + context_size)
            decoder_hidden = torch.tanh(self.extra_to_hid(decoder_hidden))  # ~(n_layers, bs, hidden_size)
        else:
            decoder_hidden = h_tm1
        del h_tm1

        # feed in input & new context to RNN
        if self.rnn_type == 'lstm':
            out, h_t = self.rnn(x, (decoder_hidden, c_tm1))
            # out ~(bs, seq=1, hidden_size)
            # h_t ~(n_layers, bs, hidden_size), ~(n_layers, bs, hidden_size)
        else:
            out, h_t = self.rnn(x, decoder_hidden)
            # out ~(bs, seq=1, hidden_size)
            # h_t ~(n_layers, bs, hidden_size)

        out = out.squeeze(1)    # ~(bs, hidden_size)
        out = self.output(out)  # ~(bs, vocab_size)
        out /= self.alpha  # divide by Boltzmann Temperature term before applying softmax

        return out, h_t, None

    def _init_weights(self):
        """
        Initialise hidden-to-hidden weights to be orthogonal
        """
        for name, param in self.rnn.named_parameters():
            if name.startswith('weight_hh'):
                torch.nn.init.orthogonal_(param.data)

    def init_hidden(self, bs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.rnn_type == 'lstm':
            return (
                torch.zeros(self.n_layers, bs, self.hidden_size).to(device),
                torch.zeros(self.n_layers, bs, self.hidden_size).to(device)
            )
        else:
            return torch.zeros(self.n_layers, bs, self.hidden_size).to(device)


class AttentionDecoder(HREDDecoder):
    """
    Decoder network taking as input the context vector, the previous hidden state,
    and the previously predicted token (or the ground truth previous token ie: teacher forcing).

    The context vector is concatenated with the previous hidden state to form an intermediate (big) hidden state.
    The intermediate (big) hidden state is passed to a linear layer and reduced to the original hidden state before
    being fed into the RNN.

    In addition the AttentionDecoder takes as input the encoder outputs (enc_outs) to build
    the attention weights and condition the output of the RNN on this new weighted context.

    Tutorial: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
              |_> "Implementing the Bahdanau et al. model"

    Paper: Effective Approaches to Attention-based Neural Machine Translation
    Minh-Thang Luong, Hieu Pham, Christopher D. Manning
    https://arxiv.org/abs/1508.04025

    Attention allows the decoder network to "focus" on a different part of the encoder's outputs
    for every step of the decoder's own outputs.
    First, we calculate the attention weights with another feed-forward layer (attn),
    using the decoder's input and hidden state as inputs (see utils.AttentionModule).
    These are then multiplied by the encoder output vectors to create a weighted combination.
    """
    def __init__(self, rnn_type, vocab_size, embedding_size, hidden_size, encoder_size,
                 extra_inp_dim=0, extra_hid_dim=0,
                 n_layers=1, h_dropout=0.0, i_dropout=0.0, i_noise=0.0,
                 attn_mode='general', alpha=1.0):
        """
        :param rnn_type: 'lstm' or 'gru'
        :param vocab_size: number of tokens in vocabulary
        :param embedding_size: embedding size of all tokens
        :param hidden_size: size of RNN hidden state
        :param encoder_size: size of the encoder outputs (used for computing attention weights)
        :param extra_inp_dim: size of the previously encoded extra information to concat to decoder input vector
        :param extra_hid_dim: size of the previously encoded extra information to concat to decoder hidden vector
        :param n_layers: number of stacked rnn layers
        :param h_dropout: dropout between each rnn layers
        :param i_dropout: input dropout (on the word embeddings)
        :param i_noise: gaussian variance of input noise (on the word embedding)
        :param attn_mode: 'general', 'dot', or 'concat'
        :param alpha: Boltzmann Temperature term (set to 1 during training, change during inference)
        """
        super(AttentionDecoder, self).__init__(
            rnn_type, vocab_size, embedding_size, hidden_size, extra_inp_dim, extra_hid_dim,
            n_layers, h_dropout, i_dropout, i_noise, alpha
        )
        # Attention decoder extra: attention layer & attention concatenation
        self.attn = AttentionModule(hidden_size, encoder_size, attn_mode)
        self.attn_enc_to_hid = nn.Linear(self.hidden_size + encoder_size, self.hidden_size)

    def forward(self, x, h_tm1, extra_i=None, extra_h=None, enc_outs=None):
        """
        :param x: batch of input tokens - LongTensor ~(bs)
        :param h_tm1: previous hidden state ~(n_layers, bs, hidden_size)
        :param extra_i: any extra information
                        ~(n_layers, bs, size)
        :param extra_h: context vector of all layers at the last time step or any other information
                        ~(n_layers, bs, size)
        :param enc_outs: encoder output vectors of the last layer at each time step
                         ~(bs, max_src_len, n_dir*size)

        :return: dec_out      ~(bs, vocab_size)
                 h_t          ~(n_layers, bs, hidden_size)
                 attn_weights ~(bs, seq=1, max_src_len)
        """
        # Note: this is called at each decoding step, one token at a time...
        x = self.embedding(x).view(x.size(0), 1, -1)  # ~(bs, seq=1, embedding_size)
        if self.noise is not None:
            x = self.noise(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if extra_i is not None:
            assert self.extra_to_inp is not None
            # reshape extra_i to fit input space: from (n_layers, bs, size) to (bs, seq=1, size)
            extra_i = extra_i[-1]           # ~(bs, size)
            extra_i = extra_i.unsqueeze(1)  # ~(bs, 1, size)
            x = torch.tanh(self.extra_to_inp(torch.cat((x, extra_i), dim=2)))

        ####
        # Pass input x into the RNN
        ####
        # decompose lstm unit into hidden state & cell state
        if self.rnn_type == 'lstm':
            h_tm1, c_tm1 = h_tm1

        if extra_h is not None:
            # concatenate the context and project it to hidden_size
            assert self.extra_to_hid is not None
            decoder_hidden = torch.cat((h_tm1, extra_h), dim=2)             # ~(n_layers, bs, hidden_size + context_size)
            decoder_hidden = torch.tanh(self.extra_to_hid(decoder_hidden))  # ~(n_layers, bs, hidden_size)
        else:
            decoder_hidden = h_tm1
        del h_tm1

        # feed in input & new context to RNN
        if self.rnn_type == 'lstm':
            dec_out, h_t = self.rnn(x, (decoder_hidden, c_tm1))
            # out ~(bs, seq=1, hidden_size)
            # h_t ~(n_layers, bs, hidden_size),  ~(n_layers, bs, hidden_size)
            del c_tm1
        else:
            dec_out, h_t = self.rnn(x, decoder_hidden)
            # out ~(bs, seq=1, hidden_size)
            # h_t ~(n_layers, bs, hidden_size)
        del decoder_hidden

        ####
        # Compute attention weights
        ####
        # enc_outs ~(bs, enc_seq, enc_size)
        attn_weights = self.attn(dec_out, enc_outs)  # ~(bs, dec_seq=1, max_src_len)

        # build context from encoder outputs & attention weights
        enc_outs = enc_outs * attn_weights.permute(0, 2, 1)  # ~(bs, max_src_len, enc_size)
        enc_outs = enc_outs.sum(dim=1)                       # ~(bs, enc_size)

        # get new outputs after concatenating weighted context
        dec_out = dec_out.squeeze(1)                           # ~(bs, hidden_size)
        dec_out = torch.cat((enc_outs, dec_out), 1)            # ~(bs, enc_size + hidden_size)
        dec_out = torch.tanh((self.attn_enc_to_hid(dec_out)))  # ~(bs, hidden_size)

        # get probability distribution over vocab size
        dec_out = self.output(dec_out)  # ~(bs, vocab_size)
        dec_out /= self.alpha  # divide by Boltzmann Temperature term before applying softmax

        return dec_out, h_t, attn_weights


default_params = {
    'embedding_size': 300,
    'gensen': False,
    # sentence encoder
    'sent_enc_rnn_type': 'lstm',
    'sent_enc_hidden_size': 300,
    'sent_enc_n_layers': 1,
    'sent_enc_bidirectional': True,
    'sent_enc_dropout': 0.1,
    # context encoder
    'cont_enc_rnn_type': 'lstm',
    'cont_enc_hidden_size': 300,
    'cont_enc_n_layers': 1,
    'cont_enc_bidirectional': True,
    'cont_enc_dropout': 0.1,
    # decoder
    'dec_extra_h': True,
    'dec_extra_i': False,
    'dec_rnn_type': 'lstm',
    'dec_hidden_size': 300,
    'dec_n_layers': 1,
    'dec_dropout': 0.1,
    'dec_attn_mode': 'general',  # None, 'general', 'dot', or 'concat'
    'dec_alpha': 1.0
}


##############################
# --------- HRED ----------- #
##############################


def build_hred(vocab_size, args=None):
    # Get context RNN input size
    if args:
        if args.sent_enc_bidirectional:
            cont_enc_input_size = args.sent_enc_hidden_size * 2
        else:
            cont_enc_input_size = args.sent_enc_hidden_size
        if args.gensen:
            cont_enc_input_size += 2048
    else:
        if default_params['sent_enc_bidirectional']:
            cont_enc_input_size = default_params['sent_enc_hidden_size'] * 2
        else:
            cont_enc_input_size = default_params['sent_enc_hidden_size']
        if default_params['gensen']:
            cont_enc_input_size += 2048

    sent_encoder = SentenceEncoder(
        rnn_type=args.sent_enc_rnn_type if args else default_params['sent_enc_rnn_type'],
        vocab_size=vocab_size,
        embedding_size=args.embedding_size if args else default_params['embedding_size'],
        hidden_size=args.sent_enc_hidden_size if args else default_params['sent_enc_hidden_size'],
        n_layers=args.sent_enc_n_layers if args else default_params['sent_enc_n_layers'],
        dropout=args.sent_enc_dropout if args else default_params['sent_enc_dropout'],
        bidirectional=args.sent_enc_bidirectional if args else default_params['sent_enc_bidirectional']
    )

    context_encoder = ContextEncoder(
        rnn_type=args.cont_enc_rnn_type if args else default_params['cont_enc_rnn_type'],
        input_size=cont_enc_input_size,
        hidden_size=args.cont_enc_hidden_size if args else default_params['cont_enc_hidden_size'],
        n_layers=args.cont_enc_n_layers if args else default_params['cont_enc_n_layers'],
        dropout=args.cont_enc_dropout if args else default_params['cont_enc_dropout'],
        bidirectional=args.cont_enc_bidirectional if args else default_params['cont_enc_bidirectional']
    )
    ###
    # Check if using attention
    ###
    if args:
        attention = args.dec_attn_mode
    else:
        attention = default_params['dec_attn_mode']
    ###
    # Get context encoder size
    # - used to compute ATTN weights
    # - used to define the size of the extra information to pass to the decoder
    ###
    if args:
        if args.cont_enc_bidirectional:
            cont_enc_hidden_size = args.cont_enc_hidden_size * 2
        else:
            cont_enc_hidden_size = args.cont_enc_hidden_size
    else:
        if default_params['cont_enc_bidirectional']:
            cont_enc_hidden_size = default_params['cont_enc_hidden_size'] * 2
        else:
            cont_enc_hidden_size = default_params['cont_enc_hidden_size']
    ###
    # define extra info sizes
    ###
    if args:
        # size of extra info to give to decoder hidden state
        if args.dec_extra_h:
            extra_h_size = cont_enc_hidden_size
        else:
            extra_h_size = 0
        # size of extra info to give to decoder inputs
        if args.dec_extra_i:
            extra_i_size = cont_enc_hidden_size
        else:
            extra_i_size = 0
    else:
        # size of extra info to give to decoder hidden state
        if default_params['dec_extra_h']:
            extra_h_size = cont_enc_hidden_size
        else:
            extra_h_size = 0
        # size of extra info to give to decoder inputs
        if default_params['dec_extra_i']:
            extra_i_size = cont_enc_hidden_size
        else:
            extra_i_size = 0
    ###
    # check hidden sizes for dot product attention
    ###
    if attention is not None and attention.strip().lower() == 'dot':
        # hidden size of context encoder & decoder must be the same
        if args:
            if cont_enc_hidden_size != args.dec_hidden_size:
                # hidden size of context encoder & decoder must be the same
                logger.warning(
                    "when using dot product attention, "
                    "the hidden size of the decoder (%d) must be the same as "
                    "the size of the context encoder vector (%d). Setting it to %d" % (
                        args.dec_hidden_size, cont_enc_hidden_size, cont_enc_hidden_size
                ))
                dec_hidden_size = cont_enc_hidden_size
            else:
                dec_hidden_size = args.dec_hidden_size
        else:
            if cont_enc_hidden_size != default_params['dec_hidden_size']:
                # hidden size of context encoder & decoder must be the same
                logger.warning(
                    "when using dot product attention, "
                    "the hidden size of the decoder (%d) must be the same as "
                    "the size of the context encoder vector (%d). Setting it to %d" % (
                        default_params['dec_hidden_size'], cont_enc_hidden_size, cont_enc_hidden_size
                ))
                dec_hidden_size = cont_enc_hidden_size
            else:
                dec_hidden_size = default_params['dec_hidden_size']
    else:
        dec_hidden_size = args.dec_hidden_size if args else default_params['dec_hidden_size']
    ###
    # Check that context n_layers and decoder n_layers are the same
    ###
    if args:
        if args.cont_enc_n_layers != args.dec_n_layers:
            logger.warning(
                "the number of layers in the decoder (%d) must be the same as "
                "the number of layers in the context encoder (%d). Setting it to %d" % (
                    args.dec_n_layers, args.cont_enc_n_layers, args.cont_enc_n_layers)
            )
            dec_n_layers = args.cont_enc_n_layers
        else:
            dec_n_layers = args.dec_n_layers
    else:
        if default_params['cont_enc_n_layers'] != default_params['dec_n_layers']:
            logger.warning(
                "the number of layers in the decoder (%d) must be the same as "
                "the number of layers in the context encoder (%d). Setting it to %d" % (
                    default_params['dec_n_layers'], default_params['cont_enc_n_layers'],
                    default_params['cont_enc_n_layers'])
            )
            dec_n_layers = default_params['cont_enc_n_layers']
        else:
            dec_n_layers = default_params['dec_n_layers']
    ###
    # Check that context rnn_type and decoder rnn_type are the same
    ###
    if args:
        if args.cont_enc_rnn_type != args.dec_rnn_type:
            logger.warning(
                "the rnn_type of the decoder (%s) must be the same as "
                "the rnn_type of the context encoder (%s). Setting it to %s" % (
                    args.dec_rnn_type, args.cont_enc_rnn_type, args.cont_enc_rnn_type)
            )
            dec_rnn_type = args.cont_enc_rnn_type
        else:
            dec_rnn_type = args.dec_rnn_type
    else:
        if default_params['cont_enc_rnn_type'] != default_params['dec_rnn_type']:
            logger.warning(
                "the rnn_type of the decoder (%s) must be the same as "
                "the rnn_type of the context encoder (%s). Setting it to %s" % (
                    default_params['dec_rnn_type'], default_params['cont_enc_rnn_type'],
                    default_params['cont_enc_rnn_type'])
            )
            dec_rnn_type = default_params['cont_enc_rnn_type']
        else:
            dec_rnn_type = default_params['dec_rnn_type']

    if attention is not None:
        decoder = AttentionDecoder(
            rnn_type=dec_rnn_type,
            vocab_size=vocab_size,
            embedding_size=args.embedding_size if args else default_params['embedding_size'],
            hidden_size=dec_hidden_size,
            encoder_size=cont_enc_hidden_size,  # used to compute attn weights
            extra_inp_dim=extra_i_size,  # size of extra info to pass to decoder inputs
            extra_hid_dim=extra_h_size,  # size of extra info to pass to decoder hidden states
            n_layers=dec_n_layers,
            h_dropout=args.dec_h_dropout if args else default_params['dec_dropout'],
            i_dropout=args.dec_i_dropout if args else default_params['dec_dropout'],
            i_noise=args.dec_i_noise if args else 0.0,
            attn_mode=attention,
            alpha=args.dec_alpha if args else default_params['dec_alpha']
        )
    else:
        decoder = HREDDecoder(
            rnn_type=dec_rnn_type,
            vocab_size=vocab_size,
            embedding_size=args.embedding_size if args else default_params['embedding_size'],
            hidden_size=dec_hidden_size,
            extra_inp_dim=extra_i_size,  # size of extra info to pass to decoder inputs
            extra_hid_dim=extra_h_size,  # size of extra info to pass to decoder hidden states
            n_layers=dec_n_layers,
            h_dropout=args.dec_h_dropout if args else default_params['dec_dropout'],
            i_dropout=args.dec_i_dropout if args else default_params['dec_dropout'],
            i_noise=args.dec_i_noise if args else 0.0,
            alpha=args.dec_alpha if args else default_params['dec_alpha']
        )

    return sent_encoder, context_encoder, decoder


class HREDData(data.Dataset):

    def __init__(self, src, tgt, corpus):
        self.src_str = src
        self.tgt_str = tgt
        self.corpus = corpus

        # transform string sentences into idx sentences
        self.src_idx = self.corpus.to_idx(self.src_str)
        self.tgt_idx = self.corpus.to_idx(self.tgt_str)

        self.ids = list(range(len(self.src_str)))

    def __getitem__(self, index):
        source = self.src_idx[index]
        target = self.tgt_idx[index]

        # split sentences around each " <eos>"
        sentences = split_list(source, [self.corpus.dictionary.word2idx[self.corpus.eos_tag]])
        # add <eos> back to all sentences except empty ones
        sentences = [s + [self.corpus.dictionary.word2idx[self.corpus.eos_tag]] for s in sentences if len(s) > 0]

        sentence_lengths = [len(s) for s in sentences]  # number of tokens per source sentence
        nof_sentences = len(sentences)                  # number of source sentences
        target_length = len(target)                     # number of tokens in target sentence

        return sentences, sentence_lengths, nof_sentences, target, target_length

    def __len__(self):
        return len(self.ids)


def hred_collate(dataset, corpus):
    b_sentences, b_sentence_lengths, b_nof_sentences, b_target, b_target_length = zip(*dataset)

    # flatten the batch of list of sentences into one batch of sentences
    b_sentences = [sentence   for sublist in b_sentences   for sentence in sublist]
    # same thing with the lengths: flatten the batch of list of lengths into one batch of lengths
    b_sentence_lengths = [length   for sublist in b_sentence_lengths   for length in sublist]

    max_sentence_len = max(b_sentence_lengths)  # max length of source sentences
    max_target_len = max(b_target_length)       # max length of target sentences
    # Fill in shorter sentences to make a tensor
    b_src_pp = [corpus.fill_seq(seq, max_sentence_len) for seq in b_sentences]
    b_tgt = [corpus.fill_seq(seq, max_target_len) for seq in b_target]

    b_src_pp = torch.LongTensor(b_src_pp)  # ~(bs++, seq_len)
    b_tgt = torch.LongTensor(b_tgt)        # ~(bs, seq_len)

    return b_src_pp, b_tgt, b_sentence_lengths, b_nof_sentences, b_target_length


def hred_minibatch_generator(dataset, corpus, batch_size, shuffle=False, num_workers=0):
    """
    Return a Pytorch DataLoader and its corresponding Dataset for an HRED model.
    :param dataset: tuple of (source, target) sequences
    :param corpus: MyMLToolbox.utils.Corpus object
    :param batch_size: number of examples per batch
    :param shuffle: True or False
    :param num_workers: default to 0
    """
    src, tgt = dataset
    dataset = HREDData(src, tgt, corpus)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  collate_fn=lambda batch: hred_collate(batch, corpus))
    return data_loader, dataset


##############################
# ------- SEQ-TO-SEQ ------- #
##############################


def build_seq2seq(vocab_size, args=None):
    """
    No context encoder
    """
    encoder = SentenceEncoder(
        rnn_type=args.sent_enc_rnn_type if args else default_params['sent_enc_rnn_type'],
        vocab_size=vocab_size,
        embedding_size=args.embedding_size if args else default_params['embedding_size'],
        hidden_size=args.sent_enc_hidden_size if args else default_params['sent_enc_hidden_size'],
        n_layers=args.sent_enc_n_layers if args else default_params['sent_enc_n_layers'],
        dropout=args.sent_enc_dropout if args else default_params['sent_enc_dropout'],
        bidirectional=args.sent_enc_bidirectional if args else default_params['sent_enc_bidirectional']
    )

    ###
    # Check if using attention
    ###
    if args:
        attention = args.dec_attn_mode
    else:
        attention = default_params['dec_attn_mode']
    ###
    # Get context encoder size
    # - used to compute ATTN weights
    # - used to define the size of the extra information to pass to the decoder
    ###
    if args:
        if args.sent_enc_bidirectional:
            enc_hidden_size = args.sent_enc_hidden_size * 2  # encoder is used to compute ATTN weights
        else:
            enc_hidden_size = args.sent_enc_hidden_size
        if args.gensen:
            enc_hidden_size += 2048
    else:
        if default_params['sent_enc_bidirectional']:
            enc_hidden_size = default_params['sent_enc_hidden_size'] * 2  # encoder is used to compute ATTN weights
        else:
            enc_hidden_size = default_params['sent_enc_hidden_size']
        if default_params['gensen']:
            enc_hidden_size += 2048
    ###
    # define extra info sizes
    ###
    if args:
        # size of extra info to give to decoder hidden state
        if args.dec_extra_h:
            extra_h_size = enc_hidden_size
        else:
            extra_h_size = 0
        # size of extra info to give to decoder inputs
        if args.dec_extra_i:
            extra_i_size = enc_hidden_size
        else:
            extra_i_size = 0
    else:
        # size of extra info to give to decoder hidden state
        if default_params['dec_extra_h']:
            extra_h_size = enc_hidden_size
        else:
            extra_h_size = 0
        # size of extra info to give to decoder inputs
        if default_params['dec_extra_i']:
            extra_i_size = enc_hidden_size
        else:
            extra_i_size = 0
    ###
    # check hidden sizes for dot product attention
    ###
    if attention is not None and attention.strip().lower() == 'dot':
        # hidden size of encoder & decoder must be the same
        if args:
            if enc_hidden_size != args.dec_hidden_size:
                # NOTE: enc_hidden_size is the same as sent_enc_hidden_size but adapted for bidirectional possibility
                logger.warning(
                    "when using dot product attention, "
                    "the hidden size of the decoder (%d) must be the same as "
                    "the size of the encoder (%d). Setting it to %d" % (
                        args.dec_hidden_size, enc_hidden_size, enc_hidden_size
                ))
                dec_hidden_size = enc_hidden_size
            else:
                dec_hidden_size = args.dec_hidden_size
        else:
            if enc_hidden_size != default_params['dec_hidden_size']:
                # NOTE: enc_hidden_size is the same as sent_enc_hidden_size but adapted for bidirectional possibility
                logger.warning(
                    "when using dot product attention, "
                    "the hidden size of the decoder (%d) must be the same as "
                    "the size of the encoder (%d). Setting it to %d" % (
                        default_params['dec_hidden_size'], enc_hidden_size, enc_hidden_size
                ))
                dec_hidden_size = enc_hidden_size
            else:
                dec_hidden_size = default_params['dec_hidden_size']
    else:
        dec_hidden_size = args.dec_hidden_size if args else default_params['dec_hidden_size']
    ###
    # Check that sentence n_layers and decoder n_layers are the same
    ###
    if args:
        if args.sent_enc_n_layers != args.dec_n_layers:
            logger.warning(
                "the number of layers in the decoder (%d) must be the same as "
                "the number of layers in the encoder (%d). Setting it to %d" % (
                args.dec_n_layers, args.sent_enc_n_layers, args.sent_enc_n_layers)
            )
            dec_n_layers = args.sent_enc_n_layers
        else:
            dec_n_layers = args.dec_n_layers
    else:
        if default_params['sent_enc_n_layers'] != default_params['dec_n_layers']:
            logger.warning(
                "the number of layers in the decoder (%d) must be the same as "
                "the number of layers in the encoder (%d). Setting it to %d" % (
                    default_params['dec_n_layers'], default_params['sent_enc_n_layers'],
                    default_params['sent_enc_n_layers'])
            )
            dec_n_layers = default_params['sent_enc_n_layers']
        else:
            dec_n_layers = default_params['dec_n_layers']
    ###
    # Check that encoder rnn_type and decoder rnn_type are the same
    ###
    if args:
        if args.sent_enc_rnn_type != args.dec_rnn_type:
            logger.warning(
                "the rnn_type of the decoder (%s) must be the same as "
                "the rnn_type of the encoder (%s). Setting it to %s" % (
                    args.dec_rnn_type, args.sent_enc_rnn_type, args.sent_enc_rnn_type)
            )
            dec_rnn_type = args.sent_enc_rnn_type
        else:
            dec_rnn_type = args.dec_rnn_type
    else:
        if default_params['sent_enc_rnn_type'] != default_params['dec_rnn_type']:
            logger.warning(
                "the rnn_type of the decoder (%s) must be the same as "
                "the rnn_type of the encoder (%s). Setting it to %s" % (
                    default_params['dec_rnn_type'], default_params['sent_enc_rnn_type'],
                    default_params['sent_enc_rnn_type'])
            )
            dec_rnn_type = default_params['sent_enc_rnn_type']
        else:
            dec_rnn_type = default_params['dec_rnn_type']

    if attention is not None:
        decoder = AttentionDecoder(
            rnn_type=dec_rnn_type,
            vocab_size=vocab_size,
            embedding_size=args.embedding_size if args else default_params['embedding_size'],
            hidden_size=dec_hidden_size,
            encoder_size=enc_hidden_size,  # used to compute attn weights
            extra_inp_dim=extra_i_size,  # size of extra info to pass to decoder inputs
            extra_hid_dim=extra_h_size,  # size of extra info to pass to decoder hidden states
            n_layers=dec_n_layers,
            h_dropout=args.dec_h_dropout if args else default_params['dec_dropout'],
            i_dropout=args.dec_i_dropout if args else default_params['dec_dropout'],
            i_noise=args.dec_i_noise if args else 0.0,
            attn_mode=attention,
            alpha=args.dec_alpha if args else default_params['dec_alpha']
        )
    else:
        decoder = HREDDecoder(
            rnn_type=dec_rnn_type,
            vocab_size=vocab_size,
            embedding_size=args.embedding_size if args else default_params['embedding_size'],
            hidden_size=dec_hidden_size,
            extra_inp_dim=extra_i_size,  # size of extra info to pass to decoder inputs
            extra_hid_dim=extra_h_size,  # size of extra info to pass to decoder hidden states
            n_layers=dec_n_layers,
            h_dropout=args.dec_h_dropout if args else default_params['dec_dropout'],
            i_dropout=args.dec_i_dropout if args else default_params['dec_dropout'],
            i_noise=args.dec_i_noise if args else 0.0,
            alpha=args.dec_alpha if args else default_params['dec_alpha']
        )

    return encoder, decoder


class Seq2SeqData(data.Dataset):

    def __init__(self, src, tgt, corpus):
        self.src_str = src
        self.tgt_str = tgt
        self.corpus = corpus

        # transform string sentences into idx sentences
        self.src_idx = self.corpus.to_idx(self.src_str)
        self.tgt_idx = self.corpus.to_idx(self.tgt_str)

        self.ids = list(range(len(self.src_str)))

    def __getitem__(self, index):
        source = self.src_idx[index]
        target = self.tgt_idx[index]
        return source, target, len(source), len(target)

    def __len__(self):
        return len(self.ids)


def seq2seq_collate(dataset, corpus):
    inputs, outputs, n_tok_srcs, n_tok_tgts = zip(*dataset)

    max_src = max(n_tok_srcs)  # max length of source sentences
    max_tgt = max(n_tok_tgts)  # max length of target sentences
    # Fill in shorter sentences to make a tensor
    b_src = [corpus.fill_seq(seq, max_src) for seq in inputs]
    b_tgt = [corpus.fill_seq(seq, max_tgt) for seq in outputs]

    b_src = torch.LongTensor(b_src)  # ~(bs, max_src_len)
    b_tgt = torch.LongTensor(b_tgt)  # ~(bs, max_tgt_len)

    return b_src, b_tgt, n_tok_srcs, n_tok_tgts


def seq2seq_minibatch_generator(dataset, corpus, batch_size, shuffle=False, num_workers=0):
    """
    Return a Pytorch DataLoader and its corresponding Dataset for a standard seq2seq model.
    :param dataset: tuple of (source, target) sequences
    :param corpus: MyMLToolbox.utils.Corpus object
    :param batch_size: number of examples per batch
    :param shuffle: True or False
    :param num_workers: default to 0
    """
    src, tgt = dataset
    dataset = Seq2SeqData(src, tgt, corpus)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  collate_fn=lambda batch: seq2seq_collate(batch, corpus))
    return data_loader, dataset
