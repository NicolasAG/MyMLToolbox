import numpy as np
import collections

import torch
from torch.nn import functional

# great thanks from the following repo for the implementation https://github.com/GuessWhatGame/guesswhat/blob/master/src/guesswhat/models/qgen/qgen_beamsearch_wrapper.py
# and to this one: https://github.com/placaille/nmt-comp550/blob/master/src/beam_wrapper.py

BeamToken = collections.namedtuple(
    'BeamToken',
    [
        'path',           # outputed words
        'word_id',        # next word inputs
        'decoder_state',  # state of the decoder after outputing path[-1]
        'score',          # \sum log(prob(w))
        'prev_beam',      # chain the previous beam - to store the hidden state for each outputed words
        'encoder_ht',     # last hidden states of the encoder - used in all hred decoders
        'encoder_out'     # outputs of the encoder - used for attentive decoder
    ]
)


def unloop_beam_series(current_beam):
    """
    Build the full beam sequence by using the chain-list structure
    """
    sequence = [current_beam]
    while current_beam.prev_beam is not None:
        current_beam = current_beam.prev_beam
        sequence.append(current_beam)
    return sequence[::-1]  # reverse sequence


def create_initial_beam(decoder_state, i, enc_ht, enc_out):
    """
    Create the initial BeamToken object tuple
    :param decoder_state: hidden state from decoder rnn ~(n_layers, bs, hidden_size)
    :param i: batch index
    :param enc_ht: last hidden states of the encoder - used in all hred decoders ~(n_layers, bs, hidden_size)
    :param enc_out: outputs of the encoder - used for attentive decoder ~(bs, max_enc_len, hidden_size)
    :return:
    """
    # define initial hidden state for this batch
    if len(decoder_state) == 2:
        # means we are using lstm
        h_t, c_t = decoder_state
        dec_hid = (
            h_t[:, i].unsqueeze(1),  # ~(n_layers, 1, hidden_size)
            c_t[:, i].unsqueeze(1)   # ~(n_layers, 1, hidden_size)
        )
    else:
        dec_hid = decoder_state[:, i].unsqueeze(1)  # ~(n_layers, 1, hidden_size)

    # define last encoder hidden states for this batch
    enc_ht = enc_ht[:, i].unsqueeze(1)  # ~(n_layers, 1, hidden_size)

    # define encoder outputs for this batch
    if enc_out is not None:
        # means we are using attention
        enc_out = enc_out[i].unsqueeze(0)  # ~(1, max_src_len, n_dir*size)

    return BeamToken(
        path=[[]],              # no outputed words yet
        word_id=[[]],           # no next words yet
        decoder_state=dec_hid,  # initial state of the decoder
        score=0,                # initial probability is 1 and log(1) = 0
        prev_beam=None,         # no previous beam
        encoder_ht=enc_ht,      # last hidden states of the encoder
        encoder_out=enc_out     # outputs of the encoder for attention
    )


class BSWrapper(object):
    def __init__(self, decoder, decoder_state, enc_ht, batch_size, max_len, beam_size,
                 corpus, enc_out=None, reverse=False):
        self.decoder = decoder
        self.corpus = corpus
        self.max_len = max_len
        self.beam_size = beam_size
        self.batch_size = batch_size

        if reverse:
            self.start_token = self.corpus.eos_tag
            self.final_token = self.corpus.sos_tag
        else:
            self.start_token = self.corpus.sos_tag
            self.final_token = self.corpus.eos_tag

        self.beam = [
            create_initial_beam(decoder_state, i, enc_ht, enc_out)
            for i in range(self.batch_size)
        ]  # ~(bs)

    def decode(self):
        init_input = [
            np.array([self.corpus.dictionary.word2idx[self.start_token]])
            for _ in range(self.batch_size)
        ]  # ~(bs, len=1 for now)

        for i, one_beam in enumerate(self.beam):
            # Prepare beam by appending answer and removing previous path
            one_beam.word_id[0].append(init_input[i][0])
            one_beam.path[0] = list()

            # execute beam search
            new_beam = self.eval_one_beam(one_beam)

            # Store current beam (with rnn state)
            self.beam[i] = new_beam

        # Compute output
        tokens = [b.path[0] for b in self.beam]  # ~(bs, len)
        seq_lengths = [len(q) for q in tokens]   # ~(bs)

        tokens_pad = np.full(
            shape=(self.batch_size, max(seq_lengths)),
            fill_value=self.corpus.dictionary.word2idx[self.corpus.pad_tag]
        )
        for i, (q, l) in enumerate(zip(tokens, seq_lengths)):
            tokens_pad[i, :l] = q

        return tokens_pad  # ~(bs, max_len)

    def eval_one_beam(self, initial_beam, keep_trajectories=False):
        """
        Perform beam search
        :param keep_trajectories: Keep trace of the previous beam if we want to keep the trajectory
        """
        to_evaluate = [initial_beam]
        memory = []

        for depth in range(self.max_len):

            # evaluate all the current tokens
            for beam_token in to_evaluate:

                # if token is final, directly put it into memory
                if beam_token.word_id[0][-1] == self.corpus.dictionary.word2idx[self.final_token]:
                    memory.append(beam_token)
                    continue

                dec_input = torch.LongTensor(beam_token.word_id).to(
                    torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                )
                dec_hid = beam_token.decoder_state
                enc_ht = beam_token.encoder_ht
                enc_out = beam_token.encoder_out

                # evaluate next step

                # decoder takes in: x        ~(bs=1)
                #                   h_tm1    ~(n_layers, bs=1, hidden_size)
                #                   context  ~(n_layers, bs=1, n_dir*size)
                #                   enc_outs ~(bs=1, enc_seq, n_dir*size)
                dec_out, dec_hid, attn_weights = self.decoder(dec_input, dec_hid, enc_ht, enc_out)
                # and returns: dec_out      ~(bs=1, vocab_size)
                #              dec_hid      ~(n_layers, bs=1, hidden_size)
                #              attn_weights ~(bs=1, seq=1, enc_seq)

                # reshape tensor (remove batch_size=1)
                log_p = functional.log_softmax(dec_out, dim=1)
                log_p = log_p.cpu().numpy()[0]  # ~(vocab)

                # put into memory the k-best tokens of this sample (k=beam_size)
                k_best_word_indices = np.argpartition(log_p, -self.beam_size)[-self.beam_size:]
                for w_idx in k_best_word_indices:
                    memory.append(
                        BeamToken(
                            path=[beam_token.path[0] + [w_idx]],
                            word_id=[[w_idx]],
                            decoder_state=dec_hid,
                            score=beam_token.score + log_p[w_idx],  # log(a*b) = log(a) + log(b)
                            prev_beam=beam_token if keep_trajectories else None,  # Keep trace of the previous beam if we want to keep the trajectory
                            encoder_ht=enc_ht,
                            encoder_out=enc_out
                        )
                    )

            # retrieve best beams in memory
            scores = [beam.score / len(beam.path[0]) for beam in memory]
            k_best_word_indices = np.argpartition(scores, -self.beam_size)[-self.beam_size:]

            to_evaluate = [memory[i] for i in k_best_word_indices]
            memory = []  # reset memory

        # eventually, pick the best one
        final_scores = [beam.score / len(beam.path[0]) for beam in to_evaluate]
        best_beam_idx = np.argmax(final_scores)
        best_beam = to_evaluate[best_beam_idx]
        return best_beam
