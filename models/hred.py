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

    def forward(self, x, lengths, h_0=None):
        """
        :param x: input sequence of vectors ~(bs, seq, size)
        :param lengths: length of each sequence in x ~(bs)
        :param h_0: initial hidden state ~(n_dir*n_layers, bs, size)
        :return: out ~(bs, seq, n_dir*size): output features h_t from the last layer, for each t
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
            # out ~(bs, seq, n_dir*size)
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
            # out ~(bs, seq, n_dir, size)
            # h_t ~(n_layers, n_dir, bs, size)
            # c_t ~(n_layers, n_dir, bs, size)
            '''

            h_t = (h_t, c_t)  # merge back the lstm unit

        else:
            out, h_t = self.rnn(packed, h_0)
            # unpack (back to padded)
            out, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # out ~(bs, seq, n_dir*size)
            # h_t ~(n_dir*n_layers, bs, size)

            # unsort tensor batches to match original input order
            out = self._unsort_tensor(out, sorted_idx, dim=0)
            h_t = self._unsort_tensor(h_t, sorted_idx, dim=1)  # batch index is on dim 1

            '''
            # separate the directions with forward and backward being direction 0 and 1 respectively.
            out = out.view(out.size(0), out.size(1), self.n_dir, self.hidden_size)
            h_t = h_t.view(self.n_layers, self.n_dir, h_t.size(1), self.hidden_size)
            # out ~(bs, seq, n_dir, size)
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
        x = self.embedding(x)  # ~(bs, seq, embedding_size)
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
        :param x: input sequence of vectors ~(bs, seq, size)
        :param lengths: length of each sequence in x ~(bs)
        :param h_0: initial hidden state ~(n_dir*n_layers, bs, size)
        :return: out ~(bs, seq, n_dir*size): output features h_t from the last layer, for each t
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
        self.vocab_size   = vocab_size  # attribute used in test_hred.step()

        self.embedding  = nn.Embedding(vocab_size, embedding_size)
        self.dropout    = nn.Dropout(dropout)
        self.pre_concat = nn.Linear(self.hidden_size + context_size, self.hidden_size)
        self.output     = nn.Linear(self.hidden_size, vocab_size)

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

    def forward(self, x, h_tm1, context, enc_outs):
        """
        :param x: batch of input tokens - LongTensor ~(bs)
        :param h_tm1: previous hidden state ~(n_layers, bs, hidden_size)
        :param context: context vector of all layers at the last time step
                        ~(n_layers, bs, n_dir*size)
        :param enc_outs: not used in the classic HRED decoder

        :return: out          ~(bs, vocab_size)
                 h_t          ~(n_layers, bs, hidden_size)
                 None
        """
        x = self.embedding(x).view(x.size(0), 1, -1)  # ~(bs, seq=1, embedding_size)
        x = self.dropout(x)

        # decompose lstm unit into hidden state & cell state
        if self.rnn_type == 'lstm':
            if h_tm1 is not None:
                h_tm1, c_tm1 = h_tm1
            else:
                c_tm1 = None

        # concatenate the context and project it to hidden_size
        decoder_hidden = torch.cat((h_tm1, context), 2)           # ~(n_layers, bs, hidden_size + context_size)
        decoder_hidden = torch.tanh(self.pre_concat(decoder_hidden))  # ~(n_layers, bs, hidden_size)

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
        super(AttentionDecoder, self).__init__(
            rnn_type, vocab_size, embedding_size, hidden_size, context_size, n_layers, dropout, alpha
        )
        # Attention decoder extra: attention layer & attention concatenation
        self.attn = AttentionModule(hidden_size, context_size, attn_mode)
        self.post_concat = nn.Linear(self.hidden_size + context_size, self.hidden_size)

    def forward(self, x, h_tm1, context, enc_outs):
        """
        :param x: batch of input tokens - LongTensor ~(bs)
        :param h_tm1: previous hidden state ~(n_layers, bs, hidden_size)
        :param context: context vector of all layers at the last time step
                        ~(n_layers, bs, n_dir*size)
        :param enc_outs: encoder output vectors of the last layer at each time step
                         ~(bs, enc_seq, n_dir*size)

        :return: out          ~(bs, vocab_size)
                 h_t          ~(n_layers, bs, hidden_size)
                 attn_weights ~(bs, seq=1, enc_seq)
        """
        x = self.embedding(x).view(x.size(0), 1, -1)  # ~(bs, seq=1, embedding_size)
        x = self.dropout(x)

        ####
        # Pass input x into the RNN
        ####
        # decompose lstm unit into hidden state & cell state
        if self.rnn_type == 'lstm':
            if h_tm1 is not None:
                h_tm1, c_tm1 = h_tm1
            else:
                c_tm1 = None

        # concatenate the context and project it to hidden_size
        decoder_hidden = torch.cat((h_tm1, context), 2)           # ~(n_layers, bs, hidden_size + context_size)
        decoder_hidden = torch.tanh(self.pre_concat(decoder_hidden))  # ~(n_layers, bs, hidden_size)

        # feed in input & new context to RNN
        if self.rnn_type == 'lstm':
            tmp_out, h_t = self.rnn(x, (decoder_hidden, c_tm1))
            # out ~(bs, seq=1, hidden_size)
            # h_t ~(n_layers, bs, hidden_size),  ~(n_layers, bs, hidden_size)
        else:
            tmp_out, h_t = self.rnn(x, decoder_hidden)
            # out ~(bs, seq=1, hidden_size)
            # h_t ~(n_layers, bs, hidden_size)

        ####
        # Compute attention weights
        ####
        # enc_outs ~(bs, enc_seq, enc_size)
        attn_weights = self.attn(tmp_out, enc_outs)  # ~(bs, dec_seq=1, enc_seq)

        # build context from encoder outputs & attention weights
        w_context = enc_outs * attn_weights.permute(0, 2, 1)  # ~(bs, enc_seq, enc_size)
        w_context = w_context.sum(dim=1)                      # ~(bs, enc_size)

        # get new outputs after concatenating weighted context
        tmp_out = tmp_out.squeeze(1)                   # ~(bs, hidden_size)
        tmp_out = torch.cat((w_context, tmp_out), 1)   # ~(bs, enc_size + hidden_size)
        out = torch.tanh((self.post_concat(tmp_out)))  # ~(bs, hidden_size)

        # get probability distribution over vocab size
        out = self.output(out)  # ~(bs, vocab_size)
        out /= self.alpha  # divide by Boltzmann Temperature term before applying softmax

        return out, h_t, attn_weights


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
    'dec_attn_mode': 'general',  # None, 'general', 'dot', or 'concat'
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
    # Get decoder context size
    ###
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
    ###
    # check hidden sizes for dot product attention
    ###
    if attention.strip().lower() == 'dot':
        # hidden size of context encoder & decoder must be the same
        if args:
            if dec_context_size != args.dec_hidden_size:
            # NOTE: dec_context_size is the same as cont_enc_hidden_size but adapted for bidirectional possibility
                print(
                    "WARNING: when using dot product attention, "
                    "the hidden size of the decoder (%d) must be the same as "
                    "the hidden size of the context encoder (%d). Setting it to %d" % (
                    args.dec_hidden_size, dec_context_size, dec_context_size
                ))
                dec_hidden_size = dec_context_size
            else:
                dec_hidden_size = args.dec_hidden_size
        else:
            if dec_context_size != default_params['dec_hidden_size']:
            # NOTE: dec_context_size is the same as cont_enc_hidden_size but adapted for bidirectional possibility
                print(
                    "WARNING: when using dot product attention, "
                    "the hidden size of the decoder (%d) must be the same as "
                    "the hidden size of the context encoder (%d). Setting it to %d" % (
                    default_params['dec_hidden_size'], dec_context_size, dec_context_size
                ))
                dec_hidden_size = dec_context_size
            else:
                dec_hidden_size = default_params['dec_hidden_size']
    else:
        dec_hidden_size = args.dec_hidden_size if args else default_params['dec_hidden_size']
    ###
    # Check that context n_layers and decoder n_layers are the same
    ###
    if args:
        if args.cont_enc_n_layers != args.dec_n_layers:
            print(
                "WARNING: the number of layers in the decoder (%d) must be the same as "
                "the number of layers in the context encoder (%d). Setting it to %d" % (
                args.dec_n_layers, args.cont_enc_n_layers, args.cont_enc_n_layers)
            )
            dec_n_layers = args.cont_enc_n_layers
        else:
            dec_n_layers = args.dec_n_layers
    else:
        if default_params['cont_enc_n_layers'] != default_params['dec_n_layers']:
            print(
                "WARNING: the number of layers in the decoder (%d) must be the same as "
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
            print(
                "WARNING: the rnn_type of the decoder (%s) must be the same as "
                "the rnn_type of the context encoder (%s). Setting it to %s" % (
                    args.dec_rnn_type, args.cont_enc_rnn_type, args.cont_enc_rnn_type)
            )
            dec_rnn_type = args.cont_enc_rnn_type
        else:
            dec_rnn_type = args.dec_rnn_type
    else:
        if default_params['cont_enc_rnn_type'] != default_params['dec_rnn_type']:
            print(
                "WARNING: the rnn_type of the decoder (%s) must be the same as "
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
            context_size=dec_context_size,
            n_layers=dec_n_layers,
            dropout=args.dec_dropout if args else default_params['dec_dropout'],
            attn_mode=attention,
            alpha=args.dec_alpha if args else default_params['dec_alpha']
        )
    else:
        decoder = HREDDecoder(
            rnn_type=dec_rnn_type,
            vocab_size=vocab_size,
            embedding_size=args.embedding_size if args else default_params['embedding_size'],
            hidden_size=dec_hidden_size,
            context_size=dec_context_size,
            n_layers=dec_n_layers,
            dropout=args.dec_dropout if args else default_params['dec_dropout'],
            alpha=args.dec_alpha if args else default_params['dec_alpha']
        )

    return sent_encoder, context_encoder, decoder
