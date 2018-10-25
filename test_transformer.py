"""
Script followed from `The Annotated Transformer`
from Harvard NLP group
http://nlp.seas.harvard.edu/2018/04/03/attention.html#training
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import time
import seaborn
seaborn.set_context(context='talk')

from transformer import make_model
from encdec import Batch

model = make_model(10, 10, 2)
# print(model)
print("model built!")

"""
- We create a generic training and scoring function to keep
track of loss.
- We pass in a generic loss compute function that also handles
parameter updates.
"""


def run_epoch(data_iter, model, loss_compute):
    """
    Standard training and logging function
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model(batch.src, batch.tgt,
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
Optimizer:
----------
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


def get_std_opt(p_model):
    return NoamOpt(
        p_model.src_embed[0].d_model, 2, 4000, torch.optim.Adam(
            p_model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )
    )


# Example of the curves for different model sizes and for
#  optimization hyper-parameters.
'''opts = [NoamOpt(512, 1, 4000, None),
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None),
        NoamOpt(256, 1, 8000, None)]
plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
plt.legend(["512:4000", "512:8000", "256:4000", "256:8000"])
plt.savefig('custom_lr_test.png')
print("custom learning rates figure saved!")'''


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


# Here we can see how the mass is distributed to the words
#  based on confidence.
'''criterion = LabelSmoothing(5, 0, 0.4)
predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0]])
v = criterion(predict.log(), torch.LongTensor([4, 3, 2, 1, 0]))
# show the target distribution expected by the system.
plt.imsave('label_smoothing_test.png', criterion.true_dist)
print("label smoothing test image saved!")'''

# Label smoothing actually starts to penalize the model
#  if it gets very confident about a given choice.
'''criterion = LabelSmoothing(5, 0, 0.1)
def loss(x):
    d = x + 3
    predict = torch.FloatTensor([[0, x/d, 1/d, 1/d, 1/d]])
    return criterion(predict.log(), torch.LongTensor([1]))[0]
plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
plt.savefig('label_smoothing_penalize_test.png')
print("label smoothing penalize test image saved!")'''


# continue "A FIRST EXAMPLE" from http://nlp.seas.harvard.edu/2018/04/03/attention.html



