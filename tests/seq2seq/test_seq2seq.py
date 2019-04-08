"""
Tutorial followed from:

Pytorch seq-2-seq Tuto
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/main.py

Pytorch translation Tuto with attention decoder:
https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

also look at
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Lucas code:
https://github.com/placaille/nmt-comp550/tree/master/src
"""
import numpy as np
import torch
import random
import logging
import pickle as pkl
import os
import time
from nltk import sent_tokenize

import sys
sys.path.append('../../..')

from MyMLToolbox.utils import Dictionary, Corpus, set_gradient, masked_cross_entropy, show_attention
from MyMLToolbox.beam_wrapper import BSWrapper
from MyMLToolbox.models.hred import build_seq2seq, AttentionDecoder, seq2seq_minibatch_generator


# SETUP LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s.%(funcName)s l%(lineno)s [%(levelname)s]: %(message)s",
    # filename='out.txt'
)
logger = logging.getLogger(__name__)


def process_one_batch(encoder, decoder, batch, corpus, optimizer=None, beam_size=0):
    """
    Perform a training or validation step over 1 batch
    :param encoder: sentence encoder (see hred.SentenceEncoder)
    :param decoder: decoder (see hred.HREDDecoder or hred.AttentionDecoder)
    :param batch: batch of (src, tgt) pairs
    :param corpus: utils.Corpus object that hold vocab
    :param optimizer: if specified, do a training step otherwise evaluate
    :param beam_size: if specified, decode using beam search. no training.
    """
    b_src, b_tgt, len_src, len_tgt = batch

    # move to GPU
    b_src = b_src.to(device)  # ~(bs, max_src_len)
    b_tgt = b_tgt.to(device)  # ~(bs, max_tgt_len)

    bs, max_src = b_src.size()
    _, max_tgt = b_tgt.size()

    # Reset gradients
    if optimizer:
        optimizer.zero_grad()

    # Initialize hidden state of encoders and decoder
    encoder_h0 = encoder.init_hidden(bs)
    decoder_h0 = decoder.init_hidden(bs)  # can be initialized to 0 since we pass the context
                                          # at each time step no matter what

    ##########################
    # ENCODER
    ##########################
    # encoder takes in: x       ~(bs, max_src_len)
    #                   lengths ~(bs)
    #                   h_0     ~(n_dir*n_layers, bs, size)
    encoder_out, encoder_ht = encoder(b_src, len_src, encoder_h0)
    # and returns: encoder_out ~(bs, max_src_len, n_dir*size)
    #              encoder_ht  ~(n_dir*n_layers, bs, size)

    # split lstm state into hidden state and cell state
    if encoder.rnn_type == 'lstm':
        encoder_ht, encoder_ct = encoder_ht

    # grab the last hidden state of each layer to feed in each time step of the decoder as the 'context'
    encoder_ht = encoder_ht.view(
        encoder.n_layers, bs, encoder.n_dir * encoder_ht.size(2)
    )  # ~(n_layers, bs, n_dir*size)

    ##########################
    # PREPARE DECODER
    ##########################
    # initial hidden state of decoder
    decoder_ht = decoder_h0
    # Create SOS tokens for decoder input
    decoder_in = torch.LongTensor([corpus.dictionary.word2idx[corpus.sos_tag]] * bs)  # ~(bs)
    # Create tensor that will hold all the outputs of the decoder
    decoder_outputs = torch.zeros(bs, max_tgt, decoder.vocab_size)  # ~(bs, max_tgt_len, vocab)
    predictions = torch.zeros(bs, max_tgt).long()                   # ~(bs, max_tgt_len)
    # Create tensor that will hold all the attention weights
    decoder_attentions = torch.zeros(bs, max_src, max_tgt)          # ~(bs, max_src_len, max_tgt_len)

    # move tensors to GPU
    decoder_in = decoder_in.to(device)
    decoder_outputs = decoder_outputs.to(device)
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
            decoder, decoder_ht, encoder_ht, bs, maxl, beam_size, corpus,
            encoder_out if isinstance(decoder, AttentionDecoder) else None
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

        # decode one token at a time
        for step in range(max_tgt):

            # decoder takes in: x        ~(bs)
            #                   h_tm1    ~(n_layers, bs, hidden_size)
            #                   context  ~(n_layers, bs, n_dir*size)
            #                   enc_outs ~(bs, max_src_len, n_dir*size)
            decoder_out, decoder_ht, attn_weights = decoder(decoder_in, decoder_ht, encoder_ht, encoder_out)
            # and returns: decoder_out  ~(bs, vocab_size)
            #              decoder_ht   ~(n_layers, bs, hidden_size)
            #              attn_weights ~(bs, seq=1, max_src_len)

            if attn_weights is not None:
                decoder_attentions[:, :attn_weights.size(2), step] += attn_weights.squeeze(1).cpu()

            # get highest scoring token and value
            top_val, top_idx = decoder_out.topk(1, dim=1)  # ~(bs, 1)
            top_token = top_idx.squeeze()    # ~(bs)
            if use_teacher_forcing:
                decoder_in = b_tgt[:, step]  # ~(bs)
            else:
                decoder_in = top_token       # ~(bs)

            # store outputs and tokens for later loss computing
            decoder_outputs[:, step, :] = decoder_out  # ~(bs, max_tgt_len, vocab)
            predictions[:, step] = top_token           # ~(bs, max_tgt_len)

        # compute loss
        loss = masked_cross_entropy(
            decoder_outputs,                      # ~(bs, max_tgt_len, vocab)
            b_tgt,                                # ~(bs, max_tgt_len)
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
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), _clip)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), _clip)
            optimizer.step()

    return loss.item(), predictions, decoder_attentions


def load_data(path, corpus):
    # check if data has been preprocessed before
    data = corpus.already_preprocessed(path, max_context_size=1)

    if data is not None:
        src, tgt = data

        # add words to dictionary
        # add all words in the source sentence
        for src_words in src:
            for word in src_words.split():
                corpus.dictionary.add_word(word)
        # add all words in the target sentence
        for tgt_words in tgt:
            for word in tgt_words.split():
                corpus.dictionary.add_word(word)

        logger.info("previously processed data has only %d examples" % len(src))
        logger.info("this experiment asked for -1 examples")
        reprocess_data = input("reprocess data (yes): ")
        if reprocess_data.lower() in ['n', 'no', '0']:
            logger.info("ok, working with %d examples then..." % len(src))
            return src, tgt

    # count number of lines
    f = open(path, 'r')
    num_lines = sum(1 for _ in f)
    logger.info("%d lines" % num_lines)
    f.close()

    # get a list of list of strings
    data_list = []
    with open(path, 'r') as f:
        for line in f:
            # skip empty lines
            if len(line.strip().split()) == 0:
                continue

            sentences = sent_tokenize(line)  # list of sentences in this line
            data_list.append(sentences)

    src, tgt = corpus.get_hreddata_from_array(data_list, max_context_size=1, debug=True)

    # save it for later
    corpus.save_preprocessed_data(path, (src, tgt), max_context_size=1)

    return src, tgt


def main():
    # Hyper-parameters...
    batch_size = 8
    max_epoch = 20
    log_interval = 1  # lo stats every k batch
    show_attention_interval = -1

    ##########################################################################
    # Load dataset
    ##########################################################################
    logger.info("")
    logger.info("Loading data...")
    corpus = Corpus()
    corpus.learn_bpe('../train_data.txt', '../bpe10', 10)
    train_src, train_tgt = load_data('../train_data.txt', corpus)
    test_src, test_tgt = load_data('../test_data.txt', corpus)
    # make sure source and target have the same number of examples
    assert len(train_src) == len(train_tgt)
    assert len(test_src) == len(test_tgt)

    # initialize batches
    train_batches, _ = seq2seq_minibatch_generator(
        (train_src, train_tgt), corpus, batch_size, shuffle=True
    )
    test_batches, _ = seq2seq_minibatch_generator(
        (test_src, test_tgt), corpus, batch_size, shuffle=False
    )

    vocab_size = len(corpus.dictionary)
    logger.info("vocab: %d" % vocab_size)

    num_train_batches = len(train_src) // batch_size
    num_test_batches = len(test_src) // batch_size

    '''
    logger.info("train sentences:")
    for src, tgt in zip(train_src, train_tgt):
        logger.info('src: %s' % src)
        logger.info('tgt: %s' % tgt)
    logger.info("train idx:")
    for src, tgt in zip(corpus.to_idx(train_src), corpus.to_idx(train_tgt)):
        logger.info('src: %s' % src)
        logger.info('tgt: %s' % tgt)
    '''
    logger.info("number of training pairs: %d" % len(train_src))
    logger.info("train batches: %d" % num_train_batches)
    '''
    logger.info("test sentences:")
    for src, tgt in zip(test_src, test_tgt):
        logger.info('src: %s' % src)
        logger.info('tgt: %s' % tgt)
    logger.info("test idx:")
    for src, tgt in zip(corpus.to_idx(test_src), corpus.to_idx(test_tgt)):
        logger.info('src: %s' % src)
        logger.info('tgt: %s' % tgt)
    '''
    logger.info("number of testing pairs: %d" % len(test_src))
    logger.info("test batches: %d" % num_test_batches)

    # Save dictionary for generation
    logger.info("saving dictionary...")
    with open('seq2seq_vocab.pt', 'wb') as f:
        pkl.dump(corpus.dictionary, f)
    logger.info("done.")

    ##########################################################################
    # Build the model
    ##########################################################################
    logger.info("")
    logger.info("Building model...")

    encoder, decoder = build_seq2seq(len(corpus.dictionary))

    encoder.to(device)
    decoder.to(device)

    logger.info("encoder: %s" % encoder)
    logger.info("decoder: %s" % decoder)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
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
    logger.info("")
    logger.info("Start training...")
    logger.info("-" * 100)

    best_valid_loss = float('inf')
    best_epoch = 0
    patience = 5

    for epoch in range(1, max_epoch+1):
        epoch_start_time = time.time()

        # Turn on training mode which enables dropout
        encoder.train()
        decoder.train()

        set_gradient(encoder, True)
        set_gradient(decoder, True)

        train_loss = 0.0
        iters = 0.0
        start_time = time.time()

        for n_batch, batch in enumerate(train_batches):
            loss, predictions, attentions = process_one_batch(
                encoder, decoder, batch, corpus, optimizer
            )
            train_loss += loss
            iters += 1

            if n_batch % log_interval == 0:
                elapsed = time.time() - start_time
                logger.info("| epoch %3d | %3d/%3d batches | ms/batch %4f | train loss %6f | train ppl %6f" % (
                    epoch, n_batch+1, num_train_batches, elapsed*1000 / log_interval,
                    loss, np.exp(loss)
                ))
                start_time = time.time()

        train_loss = train_loss / iters

        # Turn on evaluation mode which disables dropout
        encoder.eval()
        decoder.eval()

        set_gradient(encoder, False)
        set_gradient(decoder, False)

        valid_loss = 0.0
        iters = 0.0

        for n_batch, batch in enumerate(test_batches):
            loss, predictions, attentions = process_one_batch(
                encoder, decoder, batch, corpus
            )
            valid_loss += loss
            iters += 1

            if show_attention_interval > 0 and n_batch % show_attention_interval == 0:
                b_src, b_tgt, len_src, len_tgt = batch

                bs, max_src = b_src.size()
                _, max_tgt = b_tgt.size()

                # convert tensors to numpy array
                b_src = b_src.numpy()  # ~(bs, max_src_len)
                b_tgt = b_tgt.numpy()  # ~(bs, max_tgt_len)

                src_sequences = corpus.to_str(b_src)

                tgt_sequences = corpus.to_str(b_tgt)
                for i in range(bs):
                    src_sequence = src_sequences[i]
                    tgt_sequence = tgt_sequences[i]
                    # attentions ~(bs, max_src, max_tgt)
                    att_sequence = attentions[i].transpose(1, 0)  # ~(max_tgt_len, max_src_len)
                    show_attention(src_sequence, tgt_sequence, att_sequence, name=str(n_batch) + ':' + str(i))

        valid_loss /= iters
        scheduler.step(valid_loss)

        logger.info("|-epoch %3d-| took %4f s | train loss %6f | train ppl %6f | valid loss %6f | valid ppl %6f" % (
            epoch, time.time() - epoch_start_time, train_loss, np.exp(train_loss), valid_loss, np.exp(valid_loss)
        ))

        # Save the model if the validation loss improved
        if valid_loss < best_valid_loss:
            best_epoch = epoch
            best_valid_loss = valid_loss
            patience = 5  # reset patience
            logger.info("| Improved! | patience is %3d | Saving model parameters..." % patience)
            torch.save(encoder.state_dict(), "seq2seq_encoder.pt")
            torch.save(decoder.state_dict(), "seq2seq_decoder.pt")
        else:
            patience -= 1
            logger.info("| Worsened! | patience is %3d" % patience)

        logger.info("-" * 100)
        if patience <= 0:
            break

    ##########################################################################
    # Testing code
    ##########################################################################
    logger.info("Testing begins...")

    with open("seq2seq_encoder.pt", "rb") as f:
        encoder.load_state_dict(torch.load(f))
    with open("seq2seq_decoder.pt", "rb") as f:
        decoder.load_state_dict(torch.load(f))

    # initialize batches
    test_batches, _ = seq2seq_minibatch_generator(
        (test_src, test_tgt), corpus, batch_size, shuffle=False
    )

    # Turn on evaluation mode which disables dropout
    encoder.eval()
    decoder.eval()

    set_gradient(encoder, False)
    set_gradient(decoder, False)

    src_sentences = []   # store context sentence
    tgt_sentences = []   # store ground truth sentences
    pred_sentences = []  # store predicted sentences

    for n_batch, batch in enumerate(test_batches):
        _, predictions, _ = process_one_batch(
            encoder, decoder, batch, corpus, optimizer=None, beam_size=3
        )

        b_src, b_tgt, len_src, len_tgt = batch

        bs, max_src = b_src.size()
        _, max_tgt = b_tgt.size()

        # convert back to numpy
        b_src = b_src.numpy()              # ~(bs, max_src_len)
        b_tgt = b_tgt.numpy()              # ~(bs, max_tgt_len)
        predictions = predictions.numpy()  # ~(bs, max_len)

        # get tokens from the predicted indices
        src_tokens = corpus.to_str(b_src, filter_pad=True)  # ~(bs, length)
        tgt_tokens = corpus.to_str(b_tgt, filter_pad=True)  # ~(bs, length)
        pred_tokens = corpus.to_str(predictions, filter_pad=True)  # ~(bs, length)

        src_sentences.extend(src_tokens)
        tgt_sentences.extend(tgt_tokens)
        pred_sentences.extend(pred_tokens)

    with open("seq2seq_test_predictions.txt", "w") as f:
        for s_sent, g_sent, p_sent in zip(src_sentences, tgt_sentences, pred_sentences):
            f.write('src: ' + s_sent + '\n')
            f.write('gold: ' + g_sent + '\n')
            f.write('pred: ' + p_sent + '\n\n')

    logger.info("seq2seq_test_predictions.txt is saved.")


if __name__ == '__main__':
    # Hyper-parameters...
    teacher_forcing_prob = 0.99  # used in step()
    max_length = 100             # used in step() for beam search
    clip = 0.25                  # used in step() for clipping gradient norm
    seed = 14

    # Set the random seed manually for reproducibility.
    random.seed(seed)
    torch.manual_seed(seed)

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_idx = torch.cuda.current_device()
        logger.info("")
        logger.info("Using GPU: %s" % torch.cuda.get_device_name(device_idx))
        # Set the random seed manually for reproducibility.
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device('cpu')
        logger.info("")
        logger.info("No GPU available :(")

    main()
