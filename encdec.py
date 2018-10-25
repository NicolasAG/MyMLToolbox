import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import clones, LayerNorm


class Encoder(nn.Module):
    """
    - Core encoder is a stack of n layers
    with a layer normalization at the end
    """
    def __init__(self, layer, n=1):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Pass the input (and mask) through each layer in turn
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    - Generic n layer decoder with masking and layer
    normalization at the end
    """
    def __init__(self, layer, n=1):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, context, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, context, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    """
    - Define standard linear + softmax projection step
    - Usually used to map from model dimension to vocabulary size
    """
    def __init__(self, d_model: int, vocab: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class EncoderDecoder(nn.Module):
    """
    - A standard Encoder-Decoder architecture
    - Base for the transformer and many other models
    """
    def __init__(self, encoder: Encoder,
                 decoder: Decoder,
                 src_embed, tgt_embed,
                 generator: Generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        """
        Encodes source tokens into a context
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, context, src_mask, tgt, tgt_mask):
        """
        Decodes context
        """
        return self.decoder(self.tgt_embed(tgt), context,
                            src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences
        """
        return self.decode(self.encode(src, src_mask),
                           src_mask, tgt, tgt_mask)


def subsequent_mask(size: int):
    """
    - We also modify the decoder stack to prevent positions
    from attending to subsequent positions.
    - This masking, combined with fact that the output embeddings are
    offset by one position, ensures that the predictions for
    position i can depend only on the known outputs at positions
    less than i.
    """
    att_shape = (1, size, size)
    # zero-out elements lower than the 1st diagonal
    mask = np.tril(np.ones(att_shape).astype('uint8'))
    '''
    ex with size = 5:
    array([[[1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1]]], dtype=uint8)

    the attention mask shows the position each tgt word (row)
    is allowed to look at (column)
    '''
    return torch.from_numpy(mask)


class Batch:
    """
    Object for holding a batch of data with mask during training.
    - We define a batch object that holds the src and target
    sentences for training, as well as constructing the masks.
    """
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.n_tokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words.
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt_mask.size(1)).type_as(tgt_mask)
        return tgt_mask
