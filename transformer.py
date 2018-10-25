"""
Tutorial followed from Harvard NLP "The Annotated Transformer"
http://nlp.seas.harvard.edu/2018/04/03/attention.html

Also must read: http://jalammar.github.io/illustrated-transformer
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from utils import \
    LayerNorm,\
    clones,\
    scaled_dot_prod_attention,\
    to_gpu

from encdec import subsequent_mask


"""
EMBEDDINGS
"""


class Embeddings(nn.Module):
    """
    - Similarly to other sequence transduction models, we use learned
    embeddings to convert the input tokens and output tokens to
    vectors of dimension d_model.
    - We also use the usual learned linear transformation and
    softmax function to convert the decoder output to predicted
    next-token probabilities.
    - In our model, we share the same weight matrix between the
    two embedding layers and the pre-softmax linear transformation.
    - In the embedding layers, we multiply those weights by sqrt(d_model).
    """
    def __init__(self, d_model: int, vocab: int):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    - Since our model contains no recurrence and no convolution,
    in order for the model to make use of the order of the sequence,
    we must inject some information about the relative or absolute
    position of the tokens in the sequence.
    - To this end, we add "positional encodings" to the input
    embeddings at the bottoms of the encoder and decoder stacks.
    - The positional encodings have the same dimension d_model as
    the embeddings, so that the two can be summed.

    - In this work, we use sine and cosine functions of different
    frequencies:
    -- PE_{pos, 2i} = sin( pos / 10000^{2i/d_model} )
    -- PE_{pos, 2i+1} = cos( pos / 10000^{2i/d_model} )
    where pos is the position and i is the dimension.
    That is, each dimension of the positional encoding corresponds
    to a sinusoid.
    - The wavelengths form a geometric progression from 2PI to 10,000 2PI.
    - We chose this function because we hypothesized it would
    allow the model to easily learn to attend by relative positions,
    since for any fixed offset k, PE_{pos+k} can be represented as
    a linear function of PE_{pos}.

    - In addition, we apply dropout to the sums of the embeddings
    and the positional encodings in both the encoder and decoder
    stacks.
    - For the base model, we use a rate of P_drop = 0.1
    """
    def __init__(self, d_model: int, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in a log space.
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # ~(max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000) / d_model)
        )
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0)  # ~(1, max_len, d_model)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x: Embeddings):
        x = x + self.pos_enc[:, :x.size(1)]
        return self.dropout(x)


"""
SUB LAYERS
"""

"""
Attention sub-layer:
--------------------
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

- Multi-head attention allows the model to jointly attend to
information from different representation subspaces at
different positions.
- MultiHead(Q,K,V) = Concat(head_1, ..., head_h).W^Out
with head_i = Attention(Q . W_i^Q , K . W_i^K , V . W_i^V)
with parameters:
-- W_i^Q ~(d_model, d_k),
-- W_i^K ~(d_model, d_k),
-- W_i^V ~(d_model, d_v),
-- W^Out ~(h*d_v, d_model)

- In this work we employ h=8 parallel attention layers, or heads.
For each of these we use d_k = d_v = d_model / h = 64.
Due to the reduced dimension of each head, the total
computational cost is similar to that of single-head
attention with full dimensionality.

==> Read "Self-Attention in Detail" from http://jalammar.github.io/illustrated-transformer/
for better explanation and visualisation.
"""


class MultiHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout=0.1):
        """Take in model size and number of heads"""
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # ex: h=8, d_model=512, d_k=64
        self.h = h
        # 4 linear modules from d_model to d_model
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # attention weights computed
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure http://nlp.seas.harvard.edu/images/the-annotated-transformer_38_0.png
        """
        if mask is not None:
            mask = mask.unsqueeze(1)  # add dimension of size one at position 1

        n_batches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.linears[0](query)
        query = query.view(n_batches, -1, self.h, self.d_k).transpose(1, 2)

        key = self.linears[1](key)
        key = key.view(n_batches, -1, self.h, self.d_k).transpose(1, 2)

        value = self.linears[2](value)
        value = value.view(n_batches, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = scaled_dot_prod_attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) Concat using a view and apply a final linear layer
        x = x.transpose(1, 2).contiguous().view(
            n_batches, -1, self.h*self.d_k
        )
        return self.linears[3](x)


"""
Applications of Attention:
--------------------------
1) In "encoder-decoder attention" layers:
-- the queries (Q) come from the previous decoder layer,
-- the memory keys (K) and values (V) from the output of the encoder.
This allows every position in the decoder to attend over all
positions in the input sequence.

2) The encoder contains self-attention layers.
In a self-attention layer all of the keys (K), values (V) and
queries (Q) come from the same place, in this case, the output
of the previous layer in the encoder.
Each position in the encoder can attend to all positions in the
previous layer of the encoder.

3) Similarly, self-attention layers in the decoder allow each
position in the decoder to attend to all positions in the decoder
up to and including that position.
We need to prevent leftward information flow in the decoder to
preserve the auto-regressive property.
We implement this inside of scaled dot product attention by
masking out (setting to -1e9) all values in the input of the
softmax which correspond to illegal connections.
"""


class PositionwiseFeedForward(nn.Module):
    """
    Implements FF sub-layer.
    - In addition to attention sub-layers, each of the layers in our
    encoder and decoder contains a fully connected feed-forward network,
    which is applied to each position separately and identically.
    - This consists of two linear transformations with a ReLU
    activation in between.
    - While the linear transformations are the same across different
    positions, they use different parameters from layer to layer.
    Another way of describing this is as two convolutions with
    kernel size 1.
    - The dimensionality of input and output is d_model = 512,
    and the inner-layer has dimensionality d_ff = 2048
    """
    def __init__(self, d_model: int, d_ff: int, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(
            self.dropout(
                F.relu(
                    self.w_1(x)
        )))


class SublayerConnection(nn.Module):
    """
    - A residual connection followed by a layer norm.
    - The output of each sub-layer is LayerNorm(x + Sublayer(x))
    where Sublayer(x) is the function implemented by
    the sub-layer itself
    - We apply dropout to the output of each sub-layer, before
    it is added to the sub-layer input and normalized
    """
    def __init__(self, size: int, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size
        `sublayer` can be any function: self-attn, feed-forward, ...
        """
        # return x + self.dropout(sublayer(self.norm(x)))  # Note for simplicity the norm is first as opposed to last.
        return self.norm(x + self.dropout(sublayer(x)))


"""
ENCODER LAYERS
"""


class EncoderLayer(nn.Module):
    """
    - Encoder is made up of self-attention and feed forward
    - Each layer has 2 sub-layers: the first is a multi-head
    attention mechanism, and the second is a simple position-wise
    fully connected feed forward network.
    """
    def __init__(self, size: int,
                 self_attn: MultiHeadAttention,
                 feed_forward: PositionwiseFeedForward,
                 dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 2 sub-layers: 1 self-attention + 1 feed-forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        """
        Follow Figure (left) for connections
        http://nlp.seas.harvard.edu/images/the-annotated-transformer_14_0.png
        """
        # apply self attention
        x = self.sublayer[0](x, lambda a: self.self_attn.forward(a, a, a, mask))
        # apply feed-forward
        return self.sublayer[1](x, self.feed_forward)


"""
DECODER LAYERS
"""


class DecoderLayer(nn.Module):
    """
    - In addition to the two sub-layers in each encoder layer,
    the decoder inserts a third sub-layer, which performs a
    multi-head attention over the output of the encoder stack.
    - Similar to the encoder, we employ residual connections
    around each of the sub-layers, followed by layer normalization.
    """
    def __init__(self, size: int,
                 self_attn: MultiHeadAttention,
                 src_attn: MultiHeadAttention,
                 feed_forward: PositionwiseFeedForward,
                 dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 3 sub-layers: 1 self-attention + 1 source-attention + 1 feed-forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, context, src_mask, tgt_mask):
        """
        Follow Figure (right) for connections
        http://nlp.seas.harvard.edu/images/the-annotated-transformer_14_0.png
        """
        c = context
        # apply self attention on the target sentence
        x = self.sublayer[0](x, lambda a: self.self_attn.forward(a, a, a, tgt_mask))
        # apply source attention on the encoded context
        x = self.sublayer[1](x, lambda a: self.src_attn.forward(a, c, c, src_mask))
        # apply feed-forward
        return self.sublayer[2](x, self.feed_forward)


"""
OPTIMIZER:
"""

"""
- We used Adam with beta_1 = 0.9, beta_2=0.98 and eps=1e-9
- We varied the learning rate over the course of training:
--> lr = d_model^(-0.5) * min{step_num^(-0.5), step_num*warmup_steps^(-1.5)}
- This corresponds to increasing the learning rate linearly
for the first warmup_steps training steps, and decreasing it
thereafter proportionally to the inverse square root of the step number.
- We used warmup_steps = 4000.

==> NOTE: This part is very important. Need to train with this
setup of the model
"""


class NoamOpt:
    """
    Optimizer wrapper that implements rate.
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """
        Update parameters and rate.
        """
        self._step += 1  # increment step
        rate = self.rate()  # update rate accordingly
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()  # do a normal optimizer step

    def rate(self, step=None):
        """
        Implement learning rate strategy described above.
        ie: lr = d_model^(-0.5) * min{step_num^(-0.5), step_num*warmup_steps^(-1.5)}
        """
        if step is None:
            step = self._step
        return self.factor * (
                self.model_size**(-0.5) *
                min(step**(-0.5), step * self.warmup**(-1.5))
        )


"""
REGULARIZATION
"""

"""
Label smoothing
---------------
- During training, we employed label smoothing of value e_ls = 0.1.
- This hurts perplexity, as the model learns to be more unsure,
but improves accuracy and BLEU score.

- We implement label smoothing using the KL div loss.
- Instead of using a one-hot target distribution, we create a
distribution that has confidence of the correct word and the
rest of the smoothing mass distributed throughout the vocabulary.
"""


class LabelSmoothing(nn.Module):
    """
    Implement label smoothing.
    """
    def __init__(self, size: int, padding_idx: int, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


"""
BUILD MODEL WITH SPECIFIC HYPER PARAMETERS
"""


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
    from encdec import \
        EncoderDecoder, \
        Encoder, \
        Decoder, \
        Generator

    c = copy.deepcopy
    attn = to_gpu(MultiHeadAttention(h, d_model, dropout))
    ff = to_gpu(PositionwiseFeedForward(d_model, d_ff, dropout))
    position = to_gpu(PositionalEncoding(d_model, dropout))
    model = to_gpu(EncoderDecoder(
        Encoder(EncoderLayer(
            d_model, c(attn), c(ff), dropout
        ), n=n),
        Decoder(DecoderLayer(
            d_model, c(attn), c(attn), c(ff), dropout
        ), n=n),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    ))

    # This was important from their code.
    # initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


if __name__ == '__main__':
    print("it compiles!")

    # Test sub-sequence masking
    plt.figure(figsize=(5, 5))
    plt.imsave('mask_test.png', subsequent_mask(20)[0])
    print("sub-sequence masking figure saved!")
    plt.close()

    # Test positional encoding
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))
    plt.plot(np.arange(100), y[0, :, 4:8].numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.savefig('pos_emb_test.png')
    print("positional encoding figure saved!")
    plt.close()

    # Test custom optimizer learning rates
    # Example of the curves for different model sizes and for
    #  optimization hyper-parameters.
    opts = [NoamOpt(512, 1, 4000, None),
            NoamOpt(512, 1, 8000, None),
            NoamOpt(256, 1, 4000, None),
            NoamOpt(256, 1, 8000, None)]
    plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000", "256:8000"])
    plt.savefig('custom_lr_test.png')
    print("custom learning rates figure saved!")
    plt.close()

    # Test regularization with label smoothing
    # Here we can see how the mass is distributed to the words
    #  based on confidence.
    criterion = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0],
                                 [0, 0.2, 0.7, 0.1, 0]])
    v = criterion.forward(predict.log(), torch.LongTensor([4, 3, 2, 1, 0]))
    # show the target distribution expected by the system.
    plt.imsave('label_smoothing_test.png', criterion.true_dist)
    print("label smoothing test image saved!")
    plt.close()

    # Label smoothing actually starts to penalize the model
    #  if it gets very confident about a given choice.
    criterion = LabelSmoothing(5, 0, 0.1)
    def loss(x):
        d = x + 3
        pred = torch.FloatTensor([[0, x/d, 1/d, 1/d, 1/d]])
        return criterion.forward(pred.log(), torch.LongTensor([1]))[0]
    plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
    plt.savefig('label_smoothing_penalize_test.png')
    print("label smoothing penalize test image saved!")
    plt.close()

    # Test model production
    tmp_model = make_model(10, 10, 2)
    print(tmp_model)
    print("it builds!")
