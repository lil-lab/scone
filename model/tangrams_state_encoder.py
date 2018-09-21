import dynet as dy

from tangrams_fsa import SHAPES, NUM_POSITIONS

RNN_BUILDER = dy.LSTMBuilder

class TangramsStateEncoder:
    def __init__(
            self,
            pc,
            shape_embedding_size=10,
            pos_embedding_size=10,
            rnn_dim=5):
        self._pc = pc
        self._position_dim = shape_embedding_size + pos_embedding_size + 2 * rnn_dim

        # Shape indexing.
        self._shape_int = dict(zip(SHAPES, range(len(SHAPES))))

        # Embeddings.
        self._shape_embeddings = self._pc.add_lookup_parameters(
            (len(SHAPES), shape_embedding_size))
        self._pos_embeddings = self._pc.add_lookup_parameters(
            (NUM_POSITIONS, pos_embedding_size))

        # BiRNN for context-dependent encoding.
        self._encoder_fwd = RNN_BUILDER(
            1,
            shape_embedding_size +
            pos_embedding_size,
            rnn_dim,
            self._pc)
        self._encoder_bwd = RNN_BUILDER(
            1,
            shape_embedding_size +
            pos_embedding_size,
            rnn_dim,
            self._pc)

        # The state may be empty (if the list is empty) -- so we use a
        # placeolder.
        self._empty_state_placeholder = self._pc.add_parameters(
            (self.item_size()))

    def _run_rnn(self, init_state, input_vecs):
        s = init_state
        states = s.add_inputs(input_vecs)
        rnn_outputs = [s.output() for s in states]
        return rnn_outputs

    def item_size(self):
        return self._position_dim

    def encode(self, state):
        if len(state) == 0:
            return [dy.parameter(self._empty_state_placeholder)]

        state_embedding = []
        for i, shape in zip(range(len(state)), state):
            state_embedding.append(dy.concatenate(
                [self._pos_embeddings[i], self._shape_embeddings[self._shape_int[shape]]]))

        # Forward RNN.
        initial_state_fwd = self._encoder_fwd.initial_state()
        hidden_states_fwd = self._run_rnn(initial_state_fwd, state_embedding)

        # Backward RNN.
        initial_state_bwd = self._encoder_bwd.initial_state()
        hidden_states_bwd = self._run_rnn(
            initial_state_bwd, state_embedding[::-1])[::-1]

        state_embedding = [dy.concatenate([f, b, s]) for f, b, s in zip(
            hidden_states_fwd, hidden_states_bwd, state_embedding)]

        return state_embedding
