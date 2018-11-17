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
sys.path.append('../..')

from utils import Dictionary, Corpus, set_gradient, split_list, masked_cross_entropy, show_attention
from beam_wrapper import BSWrapper
from models.hred import build_hred, AttentionDecoder


def minibatch_generator(bs, src, tgt, corpus, shuffle=True):
    """
    Generator used to feed mini-batches
    :param bs: batch size
    :param src: list of source sentences
    :param tgt: list of tgt sentences
    :param corpus: utils.Corpus object
    """
    # Note: in HRED, we first encode each sentences *independently*,
    # then we encode the list of sentences as different contexts.
    # We make the decision that 'bs' represents the number of contexts in a batch,
    # hence, the number of sentences might be much greater (bs_sent >> bs).

    # transform string sentences into idx sentences
    src = corpus.to_idx(src)
    tgt = corpus.to_idx(tgt)

    nb_elem = len(src)  # number of examples in total
    indices = list(range(nb_elem))

    if shuffle:
        random.shuffle(indices)

    while nb_elem > 0:  # while there are still some items left
        b_src_pp = []    # batch of individual sentences
        len_src_pp = []  # number of tokens in each sentence
        len_src = []     # number of sentences for each context

        b_tgt = []       # batch of target sentences
        len_tgt = []     # number of tokens in target sentences

        count = 0  # number of items in a batch
        while count < bs and nb_elem > 0:
            ind = indices.pop()  # remove and return last item
            count += 1           # will add 1 item to a batch
            nb_elem -= 1         # one item was removed from all

            context = src[ind]
            target  = tgt[ind]

            # split sentences around each " <eos>"
            sentences = split_list(context, [corpus.dictionary.word2idx[corpus.eos_tag]])
            # add <eos> back to all sentences except empty ones
            sentences = [s + [corpus.dictionary.word2idx[corpus.eos_tag]] for s in sentences if len(s) > 0]

            b_src_pp.extend(sentences)      # add a bunch of individual sentences
            len_src_pp.extend([len(s) for s in sentences])  # add a bunch of sentence lengths
            len_src.append(len(sentences))  # number of sentences in this context
            b_tgt.append(target)            # append target sentence
            len_tgt.append(len(target))     # number of tokens in target sentence

        # Fill in shorter sentences to make a tensor
        max_src_pp = max(len_src_pp)  # max length of source sentences
        max_tgt    = max(len_tgt)     # max length of target sentences

        b_src_pp = [corpus.fill_seq(seq, max_src_pp) for seq in b_src_pp]
        b_tgt = [corpus.fill_seq(seq, max_tgt) for seq in b_tgt]

        b_src_pp = torch.LongTensor(b_src_pp)  # ~(bs++, seq_len)
        b_tgt = torch.LongTensor(b_tgt)        # ~(bs, seq_len)
        yield b_src_pp, b_tgt, len_src_pp, len_src, len_tgt


def process_one_batch(sent_enc, cont_enc, decoder, batch, corpus, optimizer=None, beam_size=0):
    """
    Perform a training or validation step over 1 batch
    :param sent_enc: sentence encoder (see hred.SentenceEncoder)
    :param cont_enc: context encoder (see hred.ContextEncoder)
    :param decoder: decoder (see hred.HREDDecoder or hred.AttentionDecoder)
    :param batch: batch of (src, tgt) pairs
    :param corpus: utils.Corpus object that hold vocab
    :param optimizer: if specified, do a training step otherwise evaluate
    :param beam_size: if specified, decode using beam search. no training.
    """
    b_src_pp, b_tgt, len_src_pp, len_src, len_tgt = batch
    # move to GPU
    b_src_pp = b_src_pp.to(device)
    b_tgt = b_tgt.to(device)

    max_src_pp = b_src_pp.size(1)  # max number of tokens in source sentences
    bs_pp      = b_src_pp.size(0)  # extended batch size (bcs of each individual sentences)
    max_tgt    = b_tgt.size(1)     # max number of tokens in target sentences
    bs         = b_tgt.size(0)     # actual batch size of contexts
    assert bs == len(len_src)
    max_src = max(len_src)         # max number of sentences in one context

    # Reset gradients
    if optimizer:
        optimizer.zero_grad()

    # Initialize hidden state of encoders and decoder
    sent_enc_h0 = sent_enc.init_hidden(bs_pp)
    cont_enc_h0 = cont_enc.init_hidden(bs)
    dec_h0 = decoder.init_hidden(bs)  # can be initialized to 0 since we pass the context
                                      # at each time step no matter what

    ##########################
    # SENTENCE ENCODER
    ##########################
    # sentence encoder takes in: x       ~(bs, seq)
    #                            lengths ~(bs)
    #                            h_0     ~(n_dir*n_layers, bs, size)
    sent_enc_out, _ = sent_enc(b_src_pp, len_src_pp, sent_enc_h0)
    # and returns: sent_enc_out ~(bs_pp, seq, n_dir*size)
    #              sent_enc_ht  ~(n_dir*n_layers, bs_pp, size)

    # grab the encoding of the sentence at its last time step
    sent_enc_out = sent_enc_out[
        range(sent_enc_out.size(0)),           # take each sentence
        list(map(lambda l: l-1, len_src_pp)),  # take t = len(s)-1
        :                                      # take full encoding (ie: n_dir * size)
    ]  # ~(bs_pp, n_dir * size)

    # Put back together the sentences belonging to the same context
    b_src = torch.zeros(bs, max_src, sent_enc_out.size(1)).to(device)
    # ^ this will be the input to the context encoder ~(bs, seq, size)
    batch_pp_index = 0  # keep track of where we are in the batch++ of individual sentences
    for batch_index, length in enumerate(len_src):
        b_src[batch_index, 0:length, :] = sent_enc_out[batch_pp_index: batch_pp_index+length]
        batch_pp_index += length

    ##########################
    # CONTEXT ENCODER
    ##########################
    # context encoder takes in: x       ~(bs, seq, size)
    #                           lengths ~(bs)
    #                           h_0     ~(n_dir*n_layers, bs, size)
    cont_enc_out, cont_enc_ht = cont_enc(b_src, len_src, cont_enc_h0)
    # and returns: cont_enc_out ~(bs, seq, n_dir*size)
    #              cont_enc_ht  ~(n_dir*n_layers, bs, size)

    # split lstm state into hidden state and cell state
    if cont_enc.rnn_type == 'lstm':
        cont_enc_ht, cont_enc_ct = cont_enc_ht

    # grab the last hidden state of each layer to feed in each time step of the decoder as the 'context'
    cont_enc_ht = cont_enc_ht.view(
        cont_enc.n_layers,                    # n_layers
        cont_enc_ht.size(1),                  # bs
        cont_enc.n_dir * cont_enc_ht.size(2)  # n_dir*size
    )

    ##########################
    # PREPARE DECODER
    ##########################
    # initial hidden state of decoder
    dec_hid = dec_h0
    # Create SOS tokens for decoder input
    dec_input = torch.LongTensor([corpus.dictionary.word2idx[corpus.sos_tag]] * bs)  # ~(bs)
    # Create tensor that will hold all the outputs of the decoder
    dec_outputs = torch.zeros(bs, max_tgt, decoder.vocab_size)  # ~(bs, seq, vocab)
    predictions = torch.zeros(bs, max_tgt).long()               # ~(bs, seq)
    # Create tensor that will hold all the attention weights
    decoder_attentions = torch.zeros(bs, max_src, max_tgt)

    # move tensors to GPU
    dec_input = dec_input.to(device)
    dec_outputs = dec_outputs.to(device)
    predictions = predictions.to(device)

    ###############
    # BEAM SEARCH #
    ###############
    if beam_size > 0:
        try:
            maxl = args.max_length
        except NameError:
            maxl = max_length
        beam_searcher = BSWrapper(
            decoder, dec_hid, cont_enc_ht, bs, maxl, beam_size, corpus,
            cont_enc_out if isinstance(decoder, AttentionDecoder) else None
        )
        predictions = torch.LongTensor(beam_searcher.decode())  # ~(bs, max_len)

        return None, predictions, None

    ##########
    # DECODE #
    ##########
    else:
        # define teacher forcing probability
        use_teacher_forcing = False  # do not feed in ground truth token to decoder
        try:
            tf_prob = args.teacher_forcing_prob
        except NameError:
            tf_prob = teacher_forcing_prob

        if random.random() < tf_prob:
            use_teacher_forcing = True

        # decode the target sentence
        for step in range(max_tgt):

            # decoder takes in: x        ~(bs)
            #                   h_tm1    ~(n_layers, bs, hidden_size)
            #                   context  ~(n_layers, bs, n_dir*size)
            #                   enc_outs ~(bs, enc_seq, n_dir*size)
            dec_out, dec_hid, attn_weights = decoder(dec_input, dec_hid, cont_enc_ht, cont_enc_out)
            # and returns: dec_out      ~(bs, vocab_size)
            #              dec_hid      ~(n_layers, bs, hidden_size)
            #              attn_weights ~(bs, seq=1, enc_seq)

            if attn_weights is not None:
                decoder_attentions[:, :attn_weights.size(2), step] += attn_weights.squeeze(1).cpu()

            # get highest scoring token and value
            top_val, top_idx = dec_out.topk(1, dim=1)  # ~(bs, 1)
            top_token = top_idx.squeeze()   # ~(bs)
            if use_teacher_forcing:
                dec_input = b_tgt[:, step]  # ~(bs)
            else:
                dec_input = top_token       # ~(bs)

            # store outputs and tokens for later loss computing
            dec_outputs[:, step, :] = dec_out  # dec_outputs ~(bs, seq, vocab)
            predictions[:, step] = top_token   # predictions ~(bs, seq)

        # compute loss
        loss = masked_cross_entropy(
            dec_outputs,                          # ~(bs, seq, vocab)
            b_tgt,                                # ~(bs, seq)
            torch.LongTensor(len_tgt).to(device)  # ~(bs)
        )

        # update parameters
        if optimizer:
            loss.backward()
            try:
                _clip = args.clip
            except NameError:
                _clip = clip
            if _clip > 0.0:
                torch.nn.utils.clip_grad_norm_(sent_enc.parameters(), _clip)
                torch.nn.utils.clip_grad_norm_(cont_enc.parameters(), _clip)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), _clip)
            optimizer.step()

    return loss.item(), predictions, decoder_attentions


def main():
    # Hyper-parameters...
    batch_size = 8
    max_epoch = 50
    log_interval = 1  # lo stats every k batch
    show_attention_interval = -1

    ##########################################################################
    # Load dataset
    ##########################################################################
    print("\nLoading data...")
    corpus = Corpus()
    train_src, train_tgt = corpus.get_data_from_lines('../train_data.txt')
    test_src, test_tgt = corpus.get_data_from_lines('../test_data.txt')
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

    # Save dictionary for generation
    print("saving dictionary...")
    with open('hred_vocab.pt', 'wb') as f:
        pkl.dump(corpus.dictionary, f)
    print("done.")

    ##########################################################################
    # Build the model
    ##########################################################################
    print("\nBuilding model...")

    sentence_encoder, context_encoder, decoder = build_hred(len(corpus.dictionary))

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
    print("\nStart training...")
    print("-" * 100)

    best_valid_loss = float('inf')
    best_epoch = 0
    patience = 5

    for epoch in range(1, max_epoch+1):
        epoch_start_time = time.time()

        # initialize batches
        train_batches = minibatch_generator(
            bs=batch_size, src=train_src, tgt=train_tgt, corpus=corpus, shuffle=True
        )

        # Turn on training mode which enables dropout
        sentence_encoder.train()
        context_encoder.train()
        decoder.train()

        set_gradient(sentence_encoder, True)
        set_gradient(context_encoder, True)
        set_gradient(decoder, True)

        ####
        # Training pass
        ####
        train_loss = 0.0
        iters = 0.0
        start_time = time.time()

        for n_batch, batch in enumerate(train_batches):
            loss, predictions, attentions = process_one_batch(
                sentence_encoder, context_encoder, decoder, batch, corpus, optimizer
            )
            train_loss += loss
            iters += 1

            if n_batch % log_interval == 0:
                elapsed = time.time() - start_time
                print("| epoch %3d | %3d/%3d batches | ms/batch %4f | train loss %6f | train ppl %6f" % (
                    epoch, n_batch+1, num_train_batches, elapsed * 1000 / log_interval,
                    loss, np.exp(loss)
                ))
                start_time = time.time()

        train_loss = train_loss / iters

        # initialize batches
        test_batches = minibatch_generator(
            bs=batch_size, src=test_src, tgt=test_tgt, corpus=corpus, shuffle=False
        )

        # Turn on evaluation mode which disables dropout
        sentence_encoder.eval()
        context_encoder.eval()
        decoder.eval()

        set_gradient(sentence_encoder, False)
        set_gradient(context_encoder, False)
        set_gradient(decoder, False)

        ####
        # Validation pass
        ####
        valid_loss = 0.0
        iters = 0.0

        for n_batch, batch in enumerate(test_batches):
            loss, predictions, attentions = process_one_batch(
                sentence_encoder, context_encoder, decoder, batch, corpus
            )
            valid_loss += loss
            iters += 1

            if show_attention_interval > 0 and n_batch % show_attention_interval == 0:
                b_src_pp, b_tgt, len_src_pp, len_src, len_tgt = batch

                bs = b_tgt.size(0)  # actual batch size of contexts
                assert bs == len(len_src)

                # convert tensors to numpy array
                b_src_pp = b_src_pp.numpy()
                b_tgt = b_tgt.numpy()

                # Put back together the sentences belonging to the same context
                b_src = []  # ~(bs, max_len)
                b_pp_index = 0  # keep track of where we are in the batch++ of individual sentences
                for batch_index, length in enumerate(len_src):
                    sentences = []  # list of strings
                    for s in b_src_pp[b_pp_index: b_pp_index + length]:
                        sentences.append(' '.join([corpus.dictionary.idx2word[x] for x in s]))
                    b_src.append(sentences)
                    b_pp_index += length

                for i in range(bs):
                    src_sequence = b_src[i]
                    # b_tgt ~(bs, max_tgt_len)
                    tgt_sequence = [corpus.dictionary.idx2word[x] for x in b_tgt[i, :]]
                    # attentions ~(bs, max_src, max_tgt)
                    att_sequence = attentions[i].transpose(1, 0)  # ~(max_tgt_len, max_src_len)
                    show_attention(src_sequence, tgt_sequence, att_sequence, name=str(n_batch)+':'+str(i))

        valid_loss /= iters
        scheduler.step(valid_loss)

        print("|-epoch %3d-| took %4f s | train loss %6f | train ppl %6f | valid loss %6f | valid ppl %6f" % (
            epoch, time.time() - epoch_start_time, train_loss, np.exp(train_loss), valid_loss, np.exp(valid_loss)
        ))

        # Save the model if the validation loss improved
        if valid_loss < best_valid_loss:
            best_epoch = epoch
            best_valid_loss = valid_loss
            patience = 5  # reset patience
            print("| Improved! | patience is %3d | Saving model parameters..." % patience)
            torch.save(sentence_encoder.state_dict(), "hred_sent_enc.pt")
            torch.save(context_encoder.state_dict(), "hred_cont_enc.pt")
            torch.save(decoder.state_dict(), "hred_decoder.pt")
        else:
            patience -= 1
            print("| Worsened! | patience is %3d" % patience)

        print("-" * 100)
        if patience <= 0:
            break

    ##########################################################################
    # Testing code
    ##########################################################################
    print("Testing begins...")

    with open("hred_sent_enc.pt", "rb") as f:
        sentence_encoder.load_state_dict(torch.load(f))
    with open("hred_cont_enc.pt", "rb") as f:
        context_encoder.load_state_dict(torch.load(f))
    with open("hred_decoder.pt", "rb") as f:
        decoder.load_state_dict(torch.load(f))

    # initialize batches
    test_batches = minibatch_generator(
        bs=batch_size, src=test_src, tgt=test_tgt, corpus=corpus, shuffle=False
    )

    # Turn on evaluation mode which disables dropout
    sentence_encoder.eval()
    context_encoder.eval()
    decoder.eval()

    set_gradient(sentence_encoder, False)
    set_gradient(context_encoder, False)
    set_gradient(decoder, False)

    src_sentences  = []  # store context sentence
    pred_sentences = []  # store predicted sentences
    gold_sentences = []  # store ground truth sentences

    for n_batch, batch in enumerate(test_batches):
        _, predictions, _ = process_one_batch(
            sentence_encoder, context_encoder, decoder, batch,
            corpus, optimizer=None, beam_size=3
        )
        # predictions ~(bs, max_len)

        src_pp, gold, len_src_pp, len_src, len_tgt = batch

        bs = gold.size(0)
        max_n_sent = max(len_src)     # max number of sentences in one context
        max_n_toks = max(len_src_pp)  # max number of tokens per sentence

        # Put back together the sentences belonging to the same context
        src = torch.zeros(bs, max_n_sent, max_n_toks)
        batch_pp_index = 0  # keep track of where we are in the batch++ of individual sentences
        for batch_index, length in enumerate(len_src):
            src[batch_index, 0:length, :] = src_pp[batch_pp_index: batch_pp_index + length]
            batch_pp_index += length

        # convert back to numpy
        predictions = predictions.numpy()  # ~(bs, max_len)
        gold = gold.numpy()                # ~(bs, max_tgt_len)
        src = src.numpy()                  # ~(bs, max_n_sent, max_n_toks)

        for i in range(predictions.shape[0]):
            # get tokens from the predicted indices
            src_tokens = corpus.to_str(src[i], filter_pad=True)  # list of sentences
            src_tokens = ' '.join(src_tokens)  # full context
            pred_tokens = [corpus.dictionary.idx2word[x] for x in predictions[i]]
            gold_tokens = [corpus.dictionary.idx2word[x] for x in gold[i]]

            # filter out '<pad>'
            pred_tokens = filter(lambda x: x != corpus.pad_tag, pred_tokens)
            gold_tokens = filter(lambda x: x != corpus.pad_tag, gold_tokens)

            src_sentences.append(src_tokens)
            pred_sentences.append(pred_tokens)
            gold_sentences.append(gold_tokens)

    with open("hred_test_predictions.txt", "w") as f:
        for s_sent, p_sent, g_sent in zip(src_sentences, pred_sentences, gold_sentences):
            f.write('src: ' + s_sent + '\n')
            f.write('gold: ' + ' '.join(g_sent) + '\n')
            f.write('pred: ' + ' '.join(p_sent) + '\n\n')

    print("hred_test_predictions.txt is saved.")


if __name__ == '__main__':
    # Hyper-parameters...
    teacher_forcing_prob = 0.99  # used in step()
    max_length = 100             # used in step() for beam search
    clip = 0.25                  # used in step() for clipping gradient norm
    seed = 14                    # random seed for reproducibility

    # Set the random seed manually for reproducibility.
    random.seed(seed)
    torch.manual_seed(seed)

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_idx = torch.cuda.current_device()
        print("\nUsing GPU", torch.cuda.get_device_name(device_idx))
        # Set the random seed manually for reproducibility.
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device('cpu')
        print("\nNo GPU available :(")

    main()
