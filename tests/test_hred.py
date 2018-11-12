"""
Tutorial followed from:

Pytorch Tuto
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/main.py

Pytorch Tuto:
https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

also look at
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Lucas code:
https://github.com/placaille/nmt-comp550/tree/master/src


"""
import numpy as np
import torch
import random
import pickle as pkl
import os
import time

import sys
sys.path.append('..')

from utils import Dictionary, Corpus, set_gradient
from hred import build_model


def minibatch_generator(bs, src, tgt, corpus, shuffle=True):
    """
    Generator used to feed mini-batches
    :param bs: batch size
    :param src: list of source sentences
    :param tgt: list of tgt sentences
    :param corpus: utils.Corpus object
    """
    # transform string sentences into idx sentences
    src = corpus.to_idx(src)
    tgt = corpus.to_idx(tgt)

    nb_elem = len(src)  # number of examples in total
    indices = list(range(nb_elem))

    if shuffle:
        random.shuffle(indices)

    while nb_elem > 0:  # while number there are still some items left
        b_src = []  # batch sources
        b_tgt = []  # batch targets
        len_src = []  # lengths of sources
        len_tgt = []  # lengths of targets

        count = 0  # number of items in a batch
        while count < bs and nb_elem > 0:
            ind = indices.pop()  # remove and return last item
            count += 1           # will add 1 item to a batch
            nb_elem -= 1         # one item was removed from all

            b_src.append(src[ind])
            b_tgt.append(tgt[ind])
            len_src.append(len(src[ind]))
            len_tgt.append(len(tgt[ind]))

        # Fill in shorter sentences to make a tensor
        max_src = max(len_src)  # max length of source sentences
        max_tgt = max(len_tgt)  # max length of target sentences

        b_src = [corpus.fill_seq(seq, max_src) for seq in b_src]
        b_tgt = [corpus.fill_seq(seq, max_tgt) for seq in b_tgt]

        # Sort the lists by len_src for pack_padded_sentence later
        b_sorted = [
            (bs, bt, ls, lt) for (bs, bt, ls, lt) in sorted(
                zip(b_src, b_tgt, len_src, len_tgt),
                key=lambda v: v[2],  # using len_src
                reverse=True         # descending order
            )
        ]
        # unzip to individual listss
        b_src, b_tgt, len_src, len_tgt = zip(*b_sorted)

        b_src = torch.LongTensor(b_src)  # ~(bs, seq_len)
        b_tgt = torch.LongTensor(b_tgt)  # ~(bs, seq_len)
        yield b_src, b_tgt, len_src, len_tgt


if __name__ == '__main__':

    batch_size = 8
    max_epoch = 10
    patience = 10

    ##########################################################################
    # Device configuration
    ##########################################################################
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_idx = torch.cuda.current_device()
        print("\nUsing GPU", torch.cuda.get_device_name(device_idx))
    else:
        device = torch.device('cpu')
        print("\nNo GPU available :(")

    ##########################################################################
    # Load dataset
    ##########################################################################
    print("\nLoading data...")
    corpus = Corpus()
    train_src, train_tgt = corpus.get_data('test_hred_train_data.txt')
    test_src, test_tgt = corpus.get_data('test_hred_test_data.txt')
    # make sure source and target have the same number of examples
    assert len(train_src) == len(train_tgt)
    assert len(test_src) == len(test_tgt)

    vocab_size = len(corpus.dictionary)
    print("vocab:", vocab_size)

    num_train_batches = len(train_src) // batch_size
    num_test_batches = len(test_src) // batch_size

    '''
    print("train sentences:")
    for src, tgt in zip(train_src, train_tgt):
        print('src:', src)
        print('tgt:', tgt)
    print("train idx:")
    for src, tgt in zip(corpus.to_idx(train_src), corpus.to_idx(train_tgt)):
        print('src:', src)
        print('tgt:', tgt)
    '''
    print("number of training pairs:", len(train_src))
    print("train batches:", num_train_batches)
    '''
    print("test sentences:")
    for src, tgt in zip(test_src, test_tgt):
        print('src:', src)
        print('tgt:', tgt)
    print("test idx:")
    for src, tgt in zip(corpus.to_idx(test_src), corpus.to_idx(test_tgt)):
        print('src:', src)
        print('tgt:', tgt)
    '''
    print("number of testing pairs:", len(test_src))
    print("test batches:", num_test_batches)

    # initialize batches
    train_batches = minibatch_generator(
        bs=batch_size, src=train_src, tgt=train_tgt, corpus=corpus, shuffle=True
    )
    test_batches = minibatch_generator(
        bs=batch_size, src=test_src, tgt=test_tgt, corpus=corpus, shuffle=False
    )

    # Save dictionary for generation
    print("saving dictionary...")
    with open('hred_vocab.pt', 'wb') as f:
        pkl.dump(corpus.dictionary, f)
    print("done.")

    ##########################################################################
    # Build the model
    ##########################################################################
    print("\nBuilding model...")

    sentence_encoder, context_encoder, decoder = build_model(len(corpus.dictionary))

    sentence_encoder.to(device)
    context_encoder.to(device)
    decoder.to(device)

    print("sentence encoder:", sentence_encoder)
    print("context encoder:", context_encoder)
    print("decoder:", decoder)

    optimizer = torch.optim.Adam(
        list(sentence_encoder.parameters()) +
        list(context_encoder.parameters()) +
        list(decoder.parameters())
    )
    # scheduler to reduce learning rate by 4 (*0.25) if valid loss doesn't decrease for 2 epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=0.25,
        verbose=True,
        patience=2
    )

    ##########################################################################
    # Training code
    ##########################################################################
    print("\nStarting training...")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, max_epoch+1):
        epoch_start_time = time.time()

        # Turn on training mode which enables dropout
        sentence_encoder.train()
        context_encoder.train()
        decoder.train()

        set_gradient(sentence_encoder, True)
        set_gradient(context_encoder, True)
        set_gradient(decoder, True)

        total_loss = 0.0
        start_time = time.time()

        for n_batch, batch in enumerate(train_batches):
            ###
            # STEP
            # https://github.com/placaille/nmt-comp550/blob/master/src/main.py#L231
            ###
            loss, _, _ = step()  # TODO: implement that! https://github.com/placaille/nmt-comp550/blob/master/src/utils.py#L111

            total_loss += loss
            if n_batch % 10 == 0:
                current_loss = total_loss / 10
                elapsed = time.time() - start_time
                print("| epoch %3d | %3d/%3d batches | ms/batch %6f | loss %6f | ppl %6f" % (
                    epoch, n_batch, num_train_batches, elapsed*1000 / 10, current_loss, np.exp(current_loss)
                ))
                total_loss = 0
                start_time = time.time()

        # TODO: continue from here: https://github.com/placaille/nmt-comp550/blob/master/src/main.py#L257

