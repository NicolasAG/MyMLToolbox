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


class SentenceEncoder(nn.Module):
    """
    Encoder working on word vectors, producing a sentence encoding
    The encoder will take a batch of word sequences, a LongTensor of size (batch_size x max_len),
    and output an encoding for each word, a FloatTensor of size (batch_size x max_len x hidden_size)

    Tutorial: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
              |_> "The Encoder"
    """
    def __init__(self, rnn_type, vocab_size, embedding_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=True):
        """
        :param rnn_type: 'lstm' or 'gru'
        :param vocab_size: number of tokens in vocabulary
        :param embedding_size: embedding size of all tokens
        :param hidden_size: size of RNN hidden state
        """
        super(SentenceEncoder, self).__init__()
        self.rnn_type      = rnn_type
        self.hidden_size   = hidden_size
        self.n_layers      = n_layers
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
            raise NotImplementedError("unknown rnn type %s" % self.rnn_type)

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


class ContextEncoder(nn.Module):
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
        super(ContextEncoder, self).__init__()
        self.rnn_type      = rnn_type
        self.hidden_size   = hidden_size
        self.n_layers      = n_layers
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
            raise NotImplementedError("unknown rnn type %s" % self.rnn_type)

    def forward(self, x, lengths, h_0=None):
        """
        :param x: input sequence of vectors ~(bs, seq, size)
        :param lengths: length of each sequence in x ~(bs)
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

        return out, h_t  # ~(bs, seq, n_dir*size) & ~(bs, n_layers*n_dir, size)

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
        :param alpha: Boltzmann Temperature term (set to 1 during training, change during inference)
        """
        super(HREDDecoder, self).__init__()

        self.rnn_type     = rnn_type
        self.hidden_size  = hidden_size
        self.n_layers     = n_layers
        self.alpha        = alpha

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout   = nn.Dropout(dropout)
        self.concat    = nn.Linear(self.hidden_size + context_size, self.hidden_size)
        self.output    = nn.Linear(self.hidden_size, vocab_size)

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
            raise NotImplementedError("unknown rnn type %s" % self.rnn_type)

    def forward(self, x, h_tm1, context):
        """
        :param x: batch of input tokens - LongTensor ~(bs)
        :param h_tm1: previous hidden state ~(bs, n_layers, hidden_size)
        :param context: context vector ~(bs, 1, n_dir*size)
        """
        x = self.embedding(x).view(x.size(0), 1, -1)  # ~(bs, seq=1, embedding_size)
        x = self.dropout(x)

        # decompose lstm unit into hidden state & cell state
        if self.rnn_type == 'lstm':
            if h_tm1 is not None:
                h_tm1, c_tm1 = h_tm1
            else:
                c_tm1 = None

        # expand context to be of same n_layers as h_tm1
        # (bs, 1, context_size) --> ~(bs, n_layers, context_size)
        context = context.expand(context.size(0),
                                 self.n_layers,
                                 context.size(2))

        # concatenate the context and project it to hidden_size
        decoder_hidden = torch.cat((h_tm1, context), 2)       # ~(bs, n_layers, hidden_size + context_size)
        decoder_hidden = F.tanh(self.concat(decoder_hidden))  # ~(bs, n_layers, hidden_size)

        # feed in input & new context to RNN
        if self.rnn_type == 'lstm':
            out, (h_t, c_t) = self.rnn(x, decoder_hidden, c_tm1)
            # out ~(bs, seq=1, hidden_size)
            # h_t ~(bs, n_layers, hidden_size)
            # c_t ~(bs, n_layers, hidden_size)
            h_t = (h_t, c_t)  # merge back the lstm unit
        else:
            out, h_t = self.rnn(x, decoder_hidden)
            # out ~(bs, seq=1, hidden_size)
            # h_t ~(bs, n_layers, hidden_size)

        out = out.squeeze(1)    # ~(bs, hidden_size)
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


class AttentionDecoder(nn.Module):
    """
    Decoder network taking as input the context vector, the previous hidden state,
    and the previously predicted token (or the ground truth previous token ie: teacher forcing)

    The context vector is concatenated with the previous hidden state to form the new hidden state.

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
    def __init__(self, rnn_type, vocab_size, embedding_size, hidden_size, context_size,
                 n_layers=1, dropout=0.1, attn_mode='general', alpha=1.0):
        """
        :param rnn_type: 'lstm' or 'gru'
        :param vocab_size: number of tokens in vocabulary
        :param embedding_size: embedding size of all tokens
        :param hidden_size: size of RNN hidden state
        :param context_size: size of the encoder outputs
        :param attn_mode: 'general', 'dot', or 'concat'
        :param alpha: Boltzmann Temperature term (set to 1 during training, change during inference)
        """
        super(AttnDecoder, self).__init__()

        self.rnn_type    = rnn_type
        self.hidden_size = hidden_size
        self.n_layers    = n_layers
        self.alpha       = alpha

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout   = nn.Dropout(dropout)
        self.attn      = AttentionModule(hidden_size, context_size, attn_mode)
        self.concat    = nn.Linear(self.hidden_size + context_size, self.hidden_size)
        self.output    = nn.Linear(self.hidden_size, vocab_size)

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
            raise NotImplementedError("unknown rnn type %s" % self.rnn_type)

    def forward(self, x, h_tm1, enc_outs):
        """
        :param x: batch of input tokens - LongTensor ~(bs)
        :param h_tm1: previous hidden state ~(bs, n_layers, hidden_size)
        :param enc_outs: encoder output vectors ~(bs, enc_seq, n_dir*size)

        :return: out          ~(bs, vocab_size)
                 h_t          ~(bs, n_layers, hidden_size)
                 attn_weights ~(bs, seq=1, enc_seq)
        """
        x = self.embedding(x).view(x.size(0), 1, -1)  # ~(bs, seq=1, embedding_size)
        x = self.dropout(x)

        ####
        # Pass input x into the RNN
        ####
        if self.rnn_type == 'lstm':
            # decompose lstm unit into hidden state & cell state
            if h_tm1 is not None:
                h_tm1, c_tm1 = h_tm1
            else:
                c_tm1 = None
            tmp_out, (h_t, c_t) = self.rnn(x, h_tm1, c_tm1)
            # tmp_out ~(bs, seq=1, hidden_size)
            # h_t     ~(bs, n_layers, hidden_size)
            # c_t     ~(bs, n_layers, hidden_size)
        else:
            tmp_out, h_t = self.rnn(x, h_tm1)
            # tmp_out ~(bs, seq=1, hidden_size)
            # h_t     ~(bs, n_layers, hidden_size)

        # Compute attention weights
        # enc_outs ~(bs, enc_seq, enc_size)
        attn_weights = self.attn(tmp_out, enc_outs)  # ~(bs, dec_seq=1, enc_seq)

        # build context from encoder outputs & attention weights
        context = enc_outs * attn_weights.permute(0, 2, 1)  # ~(bs, enc_seq, enc_size)
        context = context.sum(dim=1)                        # ~(bs, enc_size)

        # get new outputs after concatenating weighted context
        tmp_out = tmp_out.squeeze(1)                # ~(bs, hidden_size)
        tmp_out = torch.cat((context, tmp_out), 1)  # ~(bs, enc_size + hidden_size)
        out = F.tanh((self.concat(tmp_out)))        # ~(bs, hidden_size)

        # get probability distribution over vocab size
        out = self.output(out)  # ~(bs, vocab_size)
        out /= self.alpha  # divide by Boltzmann Temperature term before applying softmax

        # merge back the lstm unit
        if self.rnn_type == 'lstm':
            h_t = (h_t, c_t)

        return out, h_t, att_weights

    def init_hidden(self, bs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.rnn_type == 'lstm':
            return (
                torch.zeros(bs, self.n_layers, self.hidden_size).to(device),
                torch.zeros(bs, self.n_layers, self.hidden_size).to(device)
            )
        else:
            return torch.zeros(bs, self.n_layers, self.hidden_size).to(device)


default_params = {
    'embedding_size': 300,
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
    'dec_rnn_type': 'lstm',
    'dec_hidden_size': 300,
    'dec_n_layers': 1,
    'dec_dropout': 0.1,
    'dec_attn_mode': None,
    'dec_alpha': 1.0
}


def build_model(vocab_size, args=None):

    # Get context RNN input size
    if args:
        if args.sent_enc_bidirectional:
            cont_enc_input_size = args.sent_enc_hidden_size * 2
        else:
            cont_enc_input_size = args.sent_enc_hidden_size
    else:
        if default_params['sent_enc_bidirectional']:
            cont_enc_input_size = default_params['sent_enc_hidden_size'] * 2
        else:
            cont_enc_input_size = default_params['sent_enc_hidden_size']

    # Get decoder context size
    if args:
        if args.cont_enc_bidirectional:
            dec_context_size = args.cont_enc_hidden_size * 2
        else:
            dec_context_size = args.cont_enc_hidden_size
    else:
        if default_params['cont_enc_bidirectional']:
            dec_context_size = default_params['cont_enc_hidden_size'] * 2
        else:
            dec_context_size = default_params['cont_enc_hidden_size']

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

    # Check if using attention
    if args:
        if args.dec_attn_mode is not None:
            use_attention = True
        else:
            use_attention = False
    else:
        if default_params['dec_attn_mode'] is not None:
            use_attention = True
        else:
            use_attention = False

    if use_attention:
        decoder = AttentionDecoder(
            rnn_type=args.dec_rnn_type if args else default_params['dec_rnn_type'],
            vocab_size=vocab_size,
            embedding_size=args.embedding_size if args else default_params['embedding_size'],
            hidden_size=args.dec_hidden_size if args else default_params['dec_hidden_size'],
            context_size=dec_context_size,
            n_layers=args.dec_n_layers if args else default_params['dec_n_layers'],
            dropout=args.dec_dropout if args else default_params['dec_dropout'],
            attn_mode=args.dec_attn_mode if args else default_params['dec_attn_mode'],
            alpha=args.dec_alpha if args else default_params['dec_alpha']
        )
    else:
        decoder = HREDDecoder(
            rnn_type=args.dec_rnn_type if args else default_params['dec_rnn_type'],
            vocab_size=vocab_size,
            embedding_size=args.embedding_size if args else default_params['embedding_size'],
            hidden_size=args.dec_hidden_size if args else default_params['dec_hidden_size'],
            context_size=dec_context_size,
            n_layers=args.dec_n_layers if args else default_params['dec_n_layers'],
            dropout=args.dec_dropout if args else default_params['dec_dropout'],
            alpha=args.dec_alpha if args else default_params['dec_alpha']
        )

    return sent_encoder, context_encoder, decoder
