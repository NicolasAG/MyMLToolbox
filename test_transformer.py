"""
Script followed from `The Annotated Transformer`
from Harvard NLP group
http://nlp.seas.harvard.edu/2018/04/03/attention.html#training
"""
import os

import numpy as np
import torch
import time
import seaborn
seaborn.set_context(context='talk')

from encdec import Batch, subsequent_mask
from transformer import make_model, NoamOpt, LabelSmoothing


"""
- We create a generic training and scoring function to keep
track of loss.
- We pass in a generic loss compute function that also handles
parameter updates.
"""


def run_epoch(data_iter, p_model, loss_compute):
    """
    Standard training and logging function
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = p_model(batch.src, batch.tgt,
                      batch.src_mask, batch.tgt_mask)
        loss = loss_compute(out, batch.tgt_y, batch.n_tokens)
        total_loss += loss
        total_tokens += batch.n_tokens
        tokens += batch.n_tokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch: step #%.4d, loss: %.6f, tokens p.sec: %.6f" % (
                i, loss / batch.n_tokens, tokens / elapsed
            ))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


"""
- Sentence pairs were batched together by approximate sequence length.
- Each training batch contained a set of sentence pairs containing
approximately 25000 source tokens and 3,125 target tokens.
- We will use torch text for batching.
- Here we create batches in a torchtext function that ensures our
batch size padded to the maximum batchsize does not surpass a
threshold (3,125 if we have 1 gpu).
"""
global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count):
    """
    Keep augmenting batch and calculate total number of tokens + padding.
    """
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


"""
Synthetic data
"""
# We can begin by trying out a simple copy-task.
# Given a random set of input symbols from a small vocab,
#  the goal is to generate back those same symbols.


def data_gen(v, batch, n_batches):
    """
    Generate random data for a src-tgt copy task
    :param v: size of vocabulary
    :param batch: batch size
    :param n_batches: number of batches
    """
    for i in range(n_batches):
        data = torch.from_numpy(np.random.randint(1, v, size=(batch, 10)))
        data[:, 0] = 1  # all examples starts with a '1'
        yield Batch(data, data, 0)  # source & target is the same data, pad = 0


"""
Loss Computation
"""


class SimpleLossCompute:
    """
    A simple loss compute and train function
    """
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)  # compute log softmax on vocab size
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss[0] * norm


"""
Greedy Decoding
"""


def greedy_decode(p_model, src, src_mask, max_len, start_symbol):
    # encode source text
    context = p_model.encode(src, src_mask)
    # create target sequence one token at a time..
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src)

    for i in range(max_len-1):
        out = model.decode(context=context, src_mask=src_mask,
                           tgt=ys, tgt_mask=subsequent_mask(ys.size(1)).type_as(src))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word[0]
        # append predicted word to target sequence
        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(src).fill_(next_word)],
            dim=1
        )

    return ys


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use gpu 1

    # Train the simple copy task
    vocab_size = 11
    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.0)
    model = make_model(vocab_size, vocab_size, 2)  # model with same src_vocab and tgt_vocab

    # if torch.cuda.is_available:
    #     print("cuda available!! put model on GPU")
    #     model.cuda()
    # else:
    #     print("cuda not available :(")

    # print(model)
    print("model built!")

    optimizer = NoamOpt(
        model_size=model.src_embed[0].d_model,  # model.src_embed = [Embeddings, PositionalEncoding]
        factor=1,
        warmup=400,
        optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    )

    # Train for 10 epochs
    print("##############")
    print("## TRAINING ##")
    print("##############")
    for epoch in range(10):
        model.train()  # put model in training mode
        run_epoch(data_gen(v=vocab_size, batch=30, n_batches=20),
                  model,
                  SimpleLossCompute(model.generator, criterion, optimizer))

        model.eval()  # put model in evaluation mode
        print(run_epoch(
            data_gen(v=vocab_size, batch=30, n_batches=5),
            model,
            SimpleLossCompute(model.generator, criterion, opt=None)
        ))

    # Evaluate
    print("##############")
    print("## TESTING ##")
    print("##############")
    model.eval()
    src = torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    src_mask = torch.ones(1, 1, 10)
    print(greedy_decode(
        p_model=model, src=src, src_mask=src_mask, max_len=10, start_symbol=1
    ))


# continue "A REAL WORLD EXAMPLE" from http://nlp.seas.harvard.edu/2018/04/03/attention.html



