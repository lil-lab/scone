"""
Implementation of the constrained action sequence predictor. This model
predicts three things at each step: an action, a location, and an argument.

Attributes:
    RNNBuilder (dy._RNNBuilder): the type of RNN used in this network.
    BEG, EOS, NO_ARG, SEP, CURRENT_SEP, ACTION_SEP (string): all represent
        special tokens in the input and output sequences.
    LEN_LIMIT (int): the length limit of sequence generation.
    self._dropout: the dropout probability during training.

Todo:
    * Refactor so that all of the model files use the same base class with
      constants included in only one file.
"""

import random

import numpy
import time

import dynet as dy

from model_util import flatten_triple
from basic_model import SconeModel
from util import attend
from vocabulary import Vocabulary, BEG, EOS, NO_ARG, ACTION_SEP

LEN_LIMIT = 10
RNNBuilder = dy.LSTMBuilder

def sample_any_tok(prob_dist, vocabulary):
    """Samples a token from a probability distribution.

    Args:
        prob_dist (dy.Expression): Probability distribution over
            actions.
        vocab_dict (list of any): List that uniquely identifies a set
            of choices to make.

    Raises:
        ValueError, if the sampled token does not exist in the vocabulary
            dictionary.

    Returns:
        (str, dy.Expression) where the string represents the token
            sampled and the expression represents the probability of sampling it.
    """
    rnd = random.random()
    i = -1
    for i, prob in enumerate(prob_dist.value()):
        rnd -= prob
        if rnd <= 0:
            break
    if i > len(vocabulary):
        raise ValueError("Sampled token with index " +
                         str(i) +
                         " but vocab dict has length " +
                         str(len(vocabulary)))
    sampled_token = vocabulary.lookup_token(i)
    return (sampled_token, prob_dist[i])

class ConstrainedContextSeq2SeqEmbeddings(SconeModel):
    """Model that predicts a sequence of actions (action and arguments).

    Attributes:

    Todo:
        * Consider refactoring. E.g., have a class for an encoder and a
            decoder.
        * Fewer parameters in the constructor.
    """

    def __init__(self,
                 in_vocab,
                 output_vocabularies,
                 state_encoder_builder,
                 valid_action_fn,
                 args):
        SconeModel.__init__(self,
                            state_encoder_builder,
                            in_vocab,
                            args.embeddings_size,
                            args.num_enc_layers,
                            args.encoder_size,
                            args.decoder_size,
                            RNNBuilder)

        self.args = args
        self._dropout = 0.

        # Output vocabs and embeddings.
        self.output_action_vocabulary = Vocabulary(output_vocabularies[0], [EOS, BEG])
        self.output_location_vocabulary = Vocabulary(output_vocabularies[1], [NO_ARG, BEG])
        self.output_argument_vocabulary = Vocabulary(output_vocabularies[2], [NO_ARG, BEG])

        # All outputs vocabulary.
        all_vocabulary_list = []
        self._valid_action_indices = []
        index = 0
        for action in self.output_action_vocabulary:
            for location in self.output_location_vocabulary:
                for argument in self.output_argument_vocabulary:
                    if action != BEG and location != BEG and argument != BEG:
                        if valid_action_fn(action, location, argument):
                            self._valid_action_indices.append(index)
                        all_vocabulary_list.append((action, location, argument))
                        index += 1
        self._all_output_vocabulary = Vocabulary(all_vocabulary_list, [])

        self._output_action_embeddings = self._pc.add_lookup_parameters(
            (len(self.output_action_vocabulary),
             args.embeddings_size),
            name="output-action-embeddings")
        self._output_location_embeddings = self._pc.add_lookup_parameters(
            (len(self.output_location_vocabulary),
             args.embeddings_size),
            name="output-location-embeddings")
        self._output_argument_embeddings = self._pc.add_lookup_parameters(
            (len(self.output_argument_vocabulary),
             args.embeddings_size),
            name="output-argument-embeddings")

        # Action decoder RNN.
        self._dec_input_size = args.encoder_size * 2 \
            + args.encoder_size * 2 \
            + self._state_encoder.item_size() * 2 \
            + args.embeddings_size * 3
        self._decoder = RNNBuilder(args.num_dec_layers,
                                   self._dec_input_size,
                                   args.decoder_size,
                                   self._pc)

        situated_in_size = self._dec_input_size
        if self.args.always_initial_state:
            self._state_attention_winitial = self._pc.add_parameters(
                (self.args.encoder_size * 2 + self.args.decoder_size,
                 self._state_encoder.item_size()),
                name="state-attention-winitial")
            self._state_attention_winitial2 = self._pc.add_parameters(
                (self.args.encoder_size * 2 + self.args.decoder_size,
                 self._state_encoder.item_size()),
                name="state-attention-winitial2")
            situated_in_size += 2 * self._state_encoder.item_size()

        # MLP parameters to mix the situated embedding.
        self._situated_w = self._pc.add_parameters(
            (self._dec_input_size, situated_in_size),
            name="situated-w")
        self._situated_b = self._pc.add_parameters((self._dec_input_size),
                                                   name="situated-b")

        # Project the RNN output to a vector that is the length of the output
        # vocabulary.
        self._final_w = self._pc.add_parameters(
            (args.decoder_size, args.decoder_size), name="final-w")

        self._output_w_action = self._pc.add_parameters(
            (len(self.output_action_vocabulary) - 1, args.decoder_size),
            name="output-w-action")
        self._output_w_location = self._pc.add_parameters(
            (len(self.output_location_vocabulary) - 1, args.decoder_size),
            name="output-w-location")
        self._output_w_argument = self._pc.add_parameters(
            (len(self.output_argument_vocabulary) - 1, args.decoder_size),
            name="output-w-argument")

    def probability_of_token(self, token, probability_dist):
        return probability_dist[self._all_output_vocabulary.lookup_index(tuple(token))]

    def set_dropout(self, amount):
        """ Sets the dropout amount for the model, changes during various learning stages.

        Inputs:
            amount (float): Amount of dropout to apply.
        """
        self._dropout = amount

    def compute_entropy(self, distribution):
        """ Gets the entropy of a probability distribution that may contain zeroes.

        Inputs:
            probability_distribution (dy.Expression): The probability distribution.

        Returns:
            dy.Expression representing the entropy.
        """
        num_actions = len(self.output_action_vocabulary) - 1
        num_locations = len(self.output_location_vocabulary) - 1
        num_arguments = len(self.output_argument_vocabulary) - 1
        valid_mask = numpy.zeros(num_actions * num_locations * num_arguments)
        for index in self._valid_action_indices:
            valid_mask[index] = 1.
        # This mask is one for all valid indices, and zero for all others.
        valid_mask = dy.inputTensor(valid_mask)

        # This basically replaces everything in the probability distribution
        # with the original value (if valid), or zero (if not valid).
        valid_probs = dy.cmult(valid_mask, distribution)

        # The inverse of valid mask, this gives a value of 1. if something is invalid.
        invalid_probs = 1.-valid_mask

        # The result of this operation is that everything that's valid gets its
        # original probability, and everything that's not gets a probability of 1.
        probs = valid_probs + invalid_probs

        # dy.log(probs) will give log(p(action)) if action is valid, and
        # log(1)=0 for invalid actions.
        # then entropies will be zero for everything that isn't valid, and the
        # actual p log(p) otherwise.
        entropies = dy.cmult(probs, dy.log(probs + 0.00000000001))
        return -dy.sum_elems(entropies)


    def action_probabilities(self, distribution):
        num_actions = len(self.output_action_vocabulary) - 1
        num_locations = len(self.output_location_vocabulary) - 1
        num_arguments = len(self.output_argument_vocabulary) - 1
        zeroes = numpy.zeros(num_locations * num_arguments)
        ones = numpy.ones(num_locations * num_arguments)
        
        actions_masks = []
        probs = { }
        action_idx = 0
        for action in self.output_action_vocabulary:
            if action != BEG:
                masks = numpy.concatenate(
                            (numpy.repeat(zeroes, action_idx),
                             ones,
                             numpy.repeat(zeroes, num_actions - action_idx - 1)))
                actions_masks = dy.reshape(dy.inputTensor(masks),
                                           (num_actions * num_locations * num_arguments, 1))
                action_prob = dy.sum_elems(dy.cmult(actions_masks, distribution))
                probs[action] = action_prob
                action_idx += 1
        return probs

    def group_tokens(self, string):
        """ Groups tokens from a flat list of strings into action sequence.

        Inputs:
            string (list of str): Flat action sequence.

        Returns:
            list of tuple, representing parameterized actions.
        """
        seq = []
        current_triple = []
        for token in string:
            if token in self.output_action_vocabulary:
                if len(current_triple) == 3:
                    # Push the current triple and add this one
                    seq.append(current_triple)
                elif len(current_triple) < 3 and current_triple:
                    # Means that there were no arguments
                    current_triple.extend(
                        [NO_ARG for _ in range(3 - len(current_triple))])
                    assert len(current_triple) == 3
                    seq.append(current_triple)
                current_triple = [token]
            elif token in self.output_location_vocabulary:
                assert len(current_triple) == 1, \
                    "Location " + str(token) + " must follow an action," \
                    + " but current triple was " + str(current_triple)
                current_triple.append(token)
            elif token in self.output_argument_vocabulary:
                assert len(current_triple) == 2, \
                    "Argument " + str(token) + " must follow an action and location," \
                    + " but current triple was " + str(current_triple)
                current_triple.append(token)
        if len(current_triple) < 3 and current_triple:
            current_triple.extend(
                [NO_ARG for _ in range(3 - len(current_triple))])
        assert len(current_triple) == 3 or not current_triple
        if len(current_triple) == 3:
            seq.append(current_triple)
        return seq

    def _out_to_int(self, string, add_eos=False):
        if add_eos:
            string = list(string) + [EOS]
        else:
            string = list(string)

        return [(self.output_action_vocabulary.lookup_index(tok[0]),
                 self.output_location_vocabulary.lookup_index(tok[1]),
                 self.output_argument_vocabulary.lookup_index(tok[2])) \
                    for tok in self.group_tokens(string)]

    def _get_probs(self, rnn_output, restrict=None):
        final_w = dy.parameter(self._final_w)
        output_w_action = dy.parameter(self._output_w_action)
        output_w_location = dy.parameter(self._output_w_location)
        output_w_argument = dy.parameter(self._output_w_argument)

        intermediate_state = final_w * rnn_output
        if self.args.final_nonlinearity:
            intermediate_state = dy.tanh(intermediate_state)
        action_scores = output_w_action * intermediate_state
        location_scores = output_w_location * intermediate_state
        argument_scores = output_w_argument * intermediate_state

        flattened_scores = flatten_triple(action_scores, location_scores, argument_scores)
        if restrict or self.args.syntax_restricted:
            restrict_tokens = self._valid_action_indices
            if restrict:
                restrict_tokens = restrict
            return dy.exp(dy.log_softmax(flattened_scores,
                                         restrict=restrict_tokens))
        else:
            probs = dy.softmax(flattened_scores)
        return probs

    def _predict(self, rnn_output, fsa_restricted=False, fsa=None):
        # Forces a forward pass to get value.
        probs = self._get_probs(
            rnn_output,
            restrict=fsa.valid_actions(self._all_output_vocabulary) if fsa_restricted else None).value()
        max_tuple = numpy.argmax(probs)
        predicted_token = self._all_output_vocabulary.lookup_token(max_tuple)

        return (predicted_token, probs[max_tuple])

    def _init_decoder(self):
        return self._decoder.initial_state().add_input(dy.vecInput(self._dec_input_size))

    def _embed_predicted_triple(self, triple):
        return dy.concatenate([self._output_action_embeddings[triple[0]],
                               self._output_location_embeddings[triple[1]],
                               self._output_argument_embeddings[triple[2]]])

    def _decoder_input_embedding(self,
                                 rnn_state,
                                 previous_triple,
                                 encoded_string,
                                 enc_state,
                                 encoded_history,
                                 training=False,
                                 initial_state=None):
        attention_vecs = {}

        # Compute attention over encodded string.
        utterance_attn, utterance_dist = attend(encoded_string,
                                                rnn_state.h()[-1],
                                                dy.parameter(self._utterance_attention_w),
                                                self._dropout if training else 0.)
        attention_vecs['utterance'] = utterance_dist

        # Key for state and history attention.
        attn_key = dy.concatenate([utterance_attn, rnn_state.h()[-1]])
        if training:
            attn_key = dy.dropout(attn_key, self._dropout)

        # Attend on history using current state and utterance attention.
        history_attn, history_dist = attend(encoded_history,
                                            attn_key,
                                            dy.parameter(self._history_attention_w),
                                            self._dropout if training else 0.)
        attention_vecs['history'] = history_dist

        # Attend on state.
        state_attn, state_dist = attend(enc_state,
                                        attn_key,
                                        dy.parameter(self._state_attention_w),
                                        self._dropout if training else 0.)
        state_attn2, state_dist2 = attend(enc_state,
                                          attn_key,
                                          dy.parameter(self._state_attention_w2),
                                          self._dropout if training else 0.)
        attention_vecs['state_1'] = state_dist
        attention_vecs['state_2'] = state_dist2

        # Compute previous embedding
        prev_emb = self._embed_predicted_triple(previous_triple)

        # Concatenate with history and state, and mix with a feed-forward
        # layer.
        situated_embedding = dy.concatenate([utterance_attn,
                                             history_attn,
                                             state_attn,
                                             state_attn2,
                                             prev_emb])

        # Attend on initial state (if provided)
        if self.args.feed_updated_state and self.args.always_initial_state:
            if not initial_state:
                raise ValueError("Encoding the initial state but it was not provided.")
            initial_attn, initial_dist = attend(initial_state,
                                                attn_key,
                                                dy.parameter(self._state_attention_winitial),
                                                self._dropout if training else 0.)
            initial_attn2, initial_dist2 = attend(initial_state,
                                                  attn_key,
                                                  dy.parameter(self._state_attention_winitial2),
                                                  self._dropout if training else 0.)
            attention_vecs['initial_1'] = initial_dist
            attention_vecs['initial_2'] = initial_dist2

            situated_embedding = dy.concatenate([situated_embedding,
                                                 initial_attn,
                                                 initial_attn2])

        # Situated embedding mixing parameters.
        weights = dy.parameter(self._situated_w)
        biases = dy.parameter(self._situated_b)

        situated_embedding = dy.tanh(weights * situated_embedding + biases)

        return situated_embedding, attention_vecs

    def get_losses(
            self,
            utterance,
            output_seq,
            state,
            history,
            fsa=None,
            training=False):
        """Gets the losses of a gold sequence.

        Args:
            utterance (list of str): Represents the current utterance.
            output_seq (list of triple of str): Represents the gold output sequence.
            state (WorldState): Represents the state of the environment.
            history (list of list of str): Represents the previous utterances.
            fsa (ExecutableFSA, optional): An FSA builder object.
            training (bool, optional): Whether or not you are training right now.

        Returns:
            list of dy.Expression, where each corresponds to the loss at each
                gold output prediction.

        """
        enc_utterance, enc_history, enc_state = self._encode_inputs(
            utterance, state, history)
        initial_encoded_state = enc_state

        output_seq = self.group_tokens(output_seq + [EOS])

        # Run the decoder (forced decoding).
        rnn_state = self._init_decoder()
        losses = []
        prev_token_ints = (self.output_action_vocabulary.lookup_index(BEG),
                           self.output_location_vocabulary.lookup_index(BEG),
                           self.output_argument_vocabulary.lookup_index(BEG))
        for i, output_token in enumerate(output_seq):
            if self.args.feed_updated_state:
                if not fsa:
                    raise ValueError("Attempting to feed the updated state " \
                                     + "no FSA was provided")
                enc_state = self._state_encoder.encode(fsa.state())
            # Compute the decoder input.
            situated_embedding, _ = self._decoder_input_embedding(
                rnn_state,
                prev_token_ints,
                enc_utterance,
                enc_state,
                enc_history,
                training,
                initial_state=initial_encoded_state if self.args.always_initial_state else None)
            if training:
                situated_embedding = dy.dropout(
                    situated_embedding, self._dropout)

            # Weird choice -- not adding previous token generated token
            # embedding. TODO: fix
            rnn_state = rnn_state.add_input(situated_embedding)

            gold_index = self._all_output_vocabulary.lookup_index(tuple(output_token))
            log_prob_token = dy.log(self._get_probs(rnn_state.output())[gold_index])

            if self.args.feed_updated_state and output_token != (EOS, NO_ARG, NO_ARG) and output_token != [EOS, NO_ARG, NO_ARG]:
                fsa.feed_complete_action(*output_token)

            # Loss of labeled token.
            losses.append(-log_prob_token)

            prev_token_ints = (self.output_action_vocabulary.lookup_index(output_token[0]),
                               self.output_location_vocabulary.lookup_index(output_token[1]),
                               self.output_argument_vocabulary.lookup_index(output_token[2]))

        return losses

    def _update_rnn_state(self,
                          encoded_states,
                          fsa,
                          rnn_state,
                          previous_token,
                          initial_state=None,
                          training=False):
        """ Generates a single token given a state.
        """
        # Generate only if at the beginning of the sequence or the
        # previously generated token was EOS.
        utterance = encoded_states[0]
        history = encoded_states[1]
        world_state = encoded_states[2]

        if self.args.feed_updated_state:
            if not fsa:
                raise ValueError("Attempting to feed the updated state " \
                                 + "no FSA was provided")
            if not fsa.state():
                raise ValueError("Attempting to feed the updated state " \
                                 + "FSA state was None")
            world_state = self._state_encoder.encode(fsa.state())
        situated_embedding, attentions = self._decoder_input_embedding(
            rnn_state,
            previous_token,
            utterance,
            world_state,
            history,
            initial_state=initial_state,
            training=training)
        if training:
            situated_embedding = dy.dropout(situated_embedding, self._dropout)
        return rnn_state.add_input(situated_embedding), attentions

    def _policy_shape_probs(self,
                            prob_dist):
        # TODO: this is specific to Alchemy
        num_actions = len(self.output_action_vocabulary) - 1
        num_locations = len(self.output_location_vocabulary) - 1
        num_arguments = len(self.output_argument_vocabulary) - 1
        new_probdist = dy.zeros(prob_dist.dim()[0])
        zeroes = numpy.zeros(num_locations * num_arguments)
        ones = numpy.ones(num_locations * num_arguments)
        eos_prob = prob_dist[self._all_output_vocabulary.lookup_index((EOS, NO_ARG, NO_ARG))]
        action_idx = 0
        for action in self.output_action_vocabulary:
            masks = numpy.concatenate(
                        (numpy.repeat(zeroes, action_idx),
                         ones,
                         numpy.repeat(zeroes, num_actions - action_idx - 1)))
            actions_masks = dy.reshape(dy.inputTensor(masks),
                                       (num_actions * num_locations * num_arguments, 1))
            if action == EOS:
                new_probdist += dy.cmult(actions_masks, prob_dist) / 2.
            elif action == "push":
                new_probdist += dy.cmult(actions_masks, prob_dist) + eos_prob / (2. * 56.)
            elif action == "pop":
                new_probdist += dy.cmult(actions_masks, prob_dist)

        if self.args.syntax_restricted:
            return dy.exp(dy.log_softmax(dy.cmult(new_probdist, prob_dist),
                                         restrict = self._valid_action_indices))
        else:
            return dy.softmax(dy.cmult(new_probdist, prob_dist))

    def sample_sequences(self,
                         batch,
                         length=LEN_LIMIT,
                         training=False,
                         fsa_builder=None):
        """Rolls out using a policy (the probability distribution.

        Args:
            batch (list of examples): The batch that is being used to roll
                out.
            length (int, optional): The maximum length of the roll out.
            training (bool, optional): Whether or not training.
            fsa_builder (ExecutableFSA): An FSA that can be used to constrain.

        Returns:

        Todo:
            * Docstring.
            * No use of 'filter'.
            * Make returned value more clear.
            * Fewer branches.
            * Shorter (i.e. refactor).
        """
        sample_start = time.time()
        batch_states = []
        batch_initial_states = []
        batch_prob_sequences = [[] for example in batch]
        batch_sequences = [[] for example in batch]
        finished_seqs = [False for example in batch]
        batch_encoded_states = []

        
        for example in batch:
            encoded_inputs = self._encode_inputs(
                example.utterance,
                example.initial_state,
                example.history)
            batch_encoded_states.append(encoded_inputs)
            batch_initial_states.append(encoded_inputs[2])
            initial_state = None
            if self.args.feed_updated_state:
                if not fsa_builder:
                    raise ValueError("Need an FSA builder when feeding the "\
                                     + " updated state during sampling")
                initial_state = fsa_builder(example.initial_state)
            batch_states.append( \
                (initial_state,
                 self._init_decoder(),
                 (self.output_action_vocabulary.lookup_index(BEG),
                  self.output_location_vocabulary.lookup_index(BEG),
                  self.output_argument_vocabulary.lookup_index(BEG))))

        for _ in range(length):
            # Generate probabilities for this step.
            batch_probs = []

            batch_rnn_states = []

            assert len(batch) == len(batch_encoded_states)
            assert len(batch) == len(batch_states)

            for j, (example, encoded_states, state, initial_state) in \
                    enumerate(zip(batch, batch_encoded_states, batch_states, batch_initial_states)):

                if not finished_seqs[j]:
                    rnn_state, _ = self._update_rnn_state(encoded_states,
                                                          state[0],
                                                          state[1],
                                                          state[2],
                                                          initial_state,
                                                          training=training)
                    probs = self._get_probs(rnn_state.output())
                else:
                    probs = None
                    rnn_state = None
                batch_probs.append(probs)
                batch_rnn_states.append(rnn_state)

            # Do a forward pass on the entire batch.
            if [prob for prob in batch_probs if prob]:
                dy.esum([dy.concatenate(list(prob))
                         for prob in batch_probs if prob]).value()

                # Update the batch states and keep track of probability distribution
                # and generated sequences.
                new_states = []

                assert len(batch) == len(batch_states)
                assert len(batch) == len(batch_probs)
                assert len(batch) == len(batch_rnn_states)
                for j, (example, old_state, prob_dist, rnn_state) in enumerate(
                        zip(batch, batch_states, batch_probs, batch_rnn_states)):
                    if not finished_seqs[j]:
                        # Get the predicted token by sampling.
                        sampling_policy = prob_dist
                        if self.args.policy_shaping:
                            sampling_policy = self._policy_shape_probs(prob_dist)
                        predicted_token, token_prob = sample_any_tok(
                            sampling_policy, self._all_output_vocabulary)

                        # Update the FSA.
                        fsa = None
                        if self.args.feed_updated_state and predicted_token != (EOS, NO_ARG, NO_ARG):
                            fsa = old_state[0]
                            peek_state = fsa.peek_complete_action(*predicted_token)
                            if peek_state and predicted_token != (EOS, NO_ARG, NO_ARG):
                                fsa.feed_complete_action(*predicted_token)

                        # Only update batch states if you don't predict EOS. Otherwise,
                        # no point in continuing to generate for this example.
                        if predicted_token == (EOS, NO_ARG, NO_ARG):
                            finished_seqs[j] = True
                            new_states.append((None, None, None))
                        else:
                            predicted_token_idxs = \
                                (self.output_action_vocabulary.lookup_index(predicted_token[0]),
                                 self.output_location_vocabulary.lookup_index(predicted_token[1]),
                                 self.output_argument_vocabulary.lookup_index(predicted_token[2]))
                            new_states.append(
                                (fsa, rnn_state, predicted_token_idxs))

                        # Update probability expressions and samples.
                        batch_sequences[j].append(
                            (predicted_token, token_prob))
                        batch_prob_sequences[j].append(prob_dist)
                    else:
                        new_states.append((None, None, None))
                batch_states = new_states
            else:
                break

        return batch_prob_sequences, batch_sequences

    def generate_probs(self, utterance, state, history, fsa=None, fsa_restricted=False):
        """Gets predictions (by argmax) and their probabilities.


        Args:
            utterance (list of str): The current utterance.
            state (WorldState): The world state.
            history (list of list of str): The previous utterances.
            fsa (ExecutableFSA, optional): The FSA builder object, if using
                constrained decoding.

        Returns:
            list of (str, float), representing the predicted sequence, where
                each string is the predicted token and the float is the
                probability of the token.
        """
        dy.renew_cg()

        encoded_states = self._encode_inputs(utterance, state, history)
        initial_state = encoded_states[2]

        # Run the decoder.
        rnn_state = self._init_decoder()
        output_seq_probs = []
        attentions = []
        predicted_token_ints = [self.output_action_vocabulary.lookup_index(BEG),
                                self.output_location_vocabulary.lookup_index(BEG),
                                self.output_argument_vocabulary.lookup_index(BEG)]
        while len(output_seq_probs) <= LEN_LIMIT:
            # Compute the decoder input.
            rnn_state, attention = self._update_rnn_state(
                encoded_states,
                fsa,
                rnn_state,
                predicted_token_ints,
                initial_state if self.args.always_initial_state else None)
            attentions.append(attention)

            if self.args.fsa_restricted:
                raise ValueError("FSA generation is not implemented " \
                                 + "jointly predicting all three things")
            else:
                predicted_token, prob = self._predict(rnn_state.output(),
                                                      fsa_restricted,
                                                      fsa)

            output_seq_probs.append((predicted_token, prob))
            predicted_token_ints = \
                [self.output_action_vocabulary.lookup_index(predicted_token[0]),
                 self.output_location_vocabulary.lookup_index(predicted_token[1]),
                 self.output_argument_vocabulary.lookup_index(predicted_token[2])]
            if predicted_token == (EOS, NO_ARG, NO_ARG):
                return output_seq_probs, attentions
            if self.args.feed_updated_state:
                peek_state = fsa.peek_complete_action(*predicted_token)
                if peek_state:
                    fsa.feed_complete_action(*predicted_token)
        return output_seq_probs, attentions

    def generate(self, utterance, state, history, fsa, fsa_restricted=False):
        """Generates a sequence of predicted tokens for an input.

        Args:
            utterance (list of str): The current utterance.
            state (WorldState): The world state.
            history (list of list of str): The previous utterances.
            fsa (ExecutableFSA): The FSA, for constrained decoding.

        Returns:
            list of str, representing the predicted sequence.

        Todo:
            * Don't use map.
        """
        preds_and_probs, attentions = self.generate_probs(utterance,
                                                          state,
                                                          history,
                                                          fsa,
                                                          fsa_restricted)

        # Get only the tokens and remove the EOS token at the end.
        preds = [p[0] for p in preds_and_probs]
        if list(preds[-1]) == [EOS, NO_ARG, NO_ARG]:
            preds = preds[:-1]
        return preds, attentions
