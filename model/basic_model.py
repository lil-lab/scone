""" Generic scone model."""
import functools
import os

import dynet as dy

from util import run_rnn
from vocabulary import EOS, SEP, CURRENT_SEP, Vocabulary

class SconeModel():
    """ Generic scone model.

    Attributes:
    """
    def __init__(self,
                 state_encoder_builder,
                 input_vocabulary,
                 embeddings_size,
                 enc_layers,
                 enc_state_size,
                 dec_state_size,
                 RNNBuilder):
        self._pc = dy.ParameterCollection()

        # State encoder: specific to the domain.
        self._state_encoder = state_encoder_builder(self._pc)

        # Input vocabulary: used the same way for all models.
        self._input_vocabulary = Vocabulary(input_vocabulary, [EOS, SEP, CURRENT_SEP])
        self._input_embeddings = self._pc.add_lookup_parameters( \
            (len(self._input_vocabulary), embeddings_size), name="input-embeddings")

        # Encoder
        self._encoder_fwd = RNNBuilder(
            enc_layers, embeddings_size, enc_state_size, self._pc)
        self._encoder_bwd = RNNBuilder(
            enc_layers, embeddings_size, enc_state_size, self._pc)

        # Attention
        self._utterance_attention_w = self._pc.add_parameters(
            (dec_state_size, enc_state_size * 2),
            name="utterance-attention-w")

        # Attention on state.
        self._state_attention_w = self._pc.add_parameters(
            (enc_state_size * 2 + dec_state_size,
             self._state_encoder.item_size()),
            name="state-attention-w")
        self._state_attention_w2 = self._pc.add_parameters(
            (enc_state_size * 2 + dec_state_size,
             self._state_encoder.item_size()),
            name="state-attention-w2")

        # Attention on history.
        self._history_attention_w = self._pc.add_parameters(
            (enc_state_size * 2 + dec_state_size,
             enc_state_size * 2),
            name="history-attention-w")

        # Not included:
        #  -- output embeddings or vocabulary -- this differs for the methods
        #  -- decoder params -- input/output may be different
        #  -- final layers -- different prediction depending on the layers


    def load_params(self, filename):
        """Loads the parameters from a filename into the ParameterCollection.

        Args:
            filename (str): The name of the file to load the parameters from.

        Raises:
            OSError: If the file is not found.
        """
        if not os.path.exists(filename):
            raise OSError("File not found: " + str(filename))
        print("loading from filename " + str(filename))
        self._pc.populate(filename)
        dy.renew_cg()

    def save_params(self, filename):
        """Saves the parameters in ParameterCollection to a specified file.

        Args:
            filename (str): The name of the file to save the parameters to.
        """
        self._pc.save(filename)
        dy.renew_cg()

    def get_params(self):
        """Gets the ParameterCollection.

        Returns:
            self._pc (dy.ParameterCollection)
        """
        return self._pc

    def _in_to_int(self, string, use_eos=True):
        string = list(string)
        if use_eos:
            string += [EOS]
        return [self._input_vocabulary.lookup_index(token) for token in string]

    def _embed_in_string(self, string):
        return [self._input_embeddings[token]for token in string]

    def _process_history(self, history):
        if history:
#            history = [history[-1]]
            history = functools.reduce(lambda a, b: a + [SEP] + b, history)

        # in_to_int adds an EOS to history. So it separates each with SEP and
        # also appends EOS.
        return self._embed_in_string(self._in_to_int(history, use_eos=False))

    def _encode_utterance(self, embedded_string):
        # Forward RNN.
        initial_state_fwd = self._encoder_fwd.initial_state()
        hidden_states_fwd = run_rnn(initial_state_fwd, embedded_string)

        # Backward RNN.
        initial_state_bwd = self._encoder_bwd.initial_state()
        hidden_states_bwd = run_rnn(
            initial_state_bwd, embedded_string[::-1])[::-1]

        birnn_states = [dy.concatenate([f, b]) for f, b in
                        zip(hidden_states_fwd, hidden_states_bwd)]

        return birnn_states

    def _encode_inputs(self, utterance, state, history):
        # Embed text.
        utterance = self._in_to_int(utterance)
        embedded_string = self._embed_in_string(utterance)

        embedded_history = self._process_history(history)

        # Encode current state.
        enc_state = self._state_encoder.encode(state)

        # Encode all the text (current utterance and history).
        # Separate the current utterance (at the end) with a special marker.
        enc_utterance = self._encode_utterance(
            embedded_history
            + [self._input_embeddings[self._input_vocabulary.lookup_index(CURRENT_SEP)]]
            + embedded_string)

        # Separate history from current utterance. They will be attended
        # separately.

        # Before my change (just now), the following line was:
        #enc_history = enc_utterance[:-len(embedded_string) + 1]
        # which includes the first token of embedded_string.
        enc_history = enc_utterance[:-len(embedded_string)]

        enc_utterance = enc_utterance[-len(embedded_string):]

        return enc_utterance, enc_history, enc_state
