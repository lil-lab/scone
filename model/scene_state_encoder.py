from scene_fsa import COLORS, NUM_POSITIONS
from scene_state import EMPTY_COLOR
import dynet as dy
from util import run_rnn

COLOR_VOCAB = COLORS + [EMPTY_COLOR]

RNNBuilder = dy.LSTMBuilder

class StateEncoder:
    def __init__(self, pc, color_embedding_size=10, pos_embedding_size=10):
        self._pc = pc
        self._position_dim = color_embedding_size * 2 + pos_embedding_size

        self._color_embeddings_hat = self._pc.add_lookup_parameters(
            (len(COLOR_VOCAB), color_embedding_size))
        self._color_embeddings_person = self._pc.add_lookup_parameters(
            (len(COLOR_VOCAB), color_embedding_size))
        self._pos_embeddings = self._pc.add_lookup_parameters(
            (NUM_POSITIONS, pos_embedding_size))

        # Color indexing.
        self._color_vocab_int = dict(zip(COLOR_VOCAB, range(len(COLOR_VOCAB))))

    def item_size(self):
        return self._position_dim

    def encode(self, state):
        state_embedding = []
        for i, (person, hat) in zip(range(len(state)), state):
            state_embedding.append(dy.concatenate([self._pos_embeddings[i],
                                                   self._color_embeddings_person[self._color_vocab_int[person]],
                                                   self._color_embeddings_hat[self._color_vocab_int[hat]]]))
        return state_embedding


class StateEncoderEmptyPlaceholders:
    def __init__(
            self,
            pc,
            color_embedding_size=10,
            pos_embedding_size=10,
            presence_embedding_size=2):
        self._pc = pc
        self._position_dim = color_embedding_size * 2 + \
            pos_embedding_size + presence_embedding_size * 2

        self._color_embeddings = self._pc.add_lookup_parameters(
            (len(COLOR_VOCAB), color_embedding_size))
        self._pos_embeddings = self._pc.add_lookup_parameters(
            (NUM_POSITIONS, pos_embedding_size))

        # Presence embeddings for person and hat. Two options: present or not.
        self._person_presence_embeddings = self._pc.add_lookup_parameters(
            (2, presence_embedding_size))
        self._hat_presence_embeddings = self._pc.add_lookup_parameters(
            (2, presence_embedding_size))

        # Color indexing.
        self._color_vocab_int = dict(zip(COLOR_VOCAB, range(len(COLOR_VOCAB))))

    def item_size(self):
        return self._position_dim

    def encode(self, state):
        state_embedding = []
        for i, (person, hat) in zip(range(len(state)), state):
            state_embedding.append(dy.concatenate([self._pos_embeddings[i],
                                                   self._color_embeddings[self._color_vocab_int[person]],
                                                   self._color_embeddings[self._color_vocab_int[hat]],
                                                   self._person_presence_embeddings[0 if person == EMPTY_COLOR else 1],
                                                   self._hat_presence_embeddings[0 if hat == EMPTY_COLOR else 1]]))
        return state_embedding


class ContextDepStateEncoder:
    def __init__(
            self,
            pc,
            color_embedding_size=10,
            pos_embedding_size=10,
            rnn_dim=5):
        self._pc = pc
        self._position_dim = color_embedding_size * 2 + pos_embedding_size + 2 * rnn_dim

        self._color_embeddings = self._pc.add_lookup_parameters(
            (len(COLOR_VOCAB), color_embedding_size))
        self._pos_embeddings = self._pc.add_lookup_parameters(
            (NUM_POSITIONS, pos_embedding_size))

        # Color indexing.
        self._color_vocab_int = dict(zip(COLOR_VOCAB, range(len(COLOR_VOCAB))))

        # BiRNN for context-dependent encoding.
        self._encoder_fwd = RNNBuilder(
            1,
            color_embedding_size *
            2 +
            pos_embedding_size,
            rnn_dim,
            self._pc)
        self._encoder_bwd = RNNBuilder(
            1,
            color_embedding_size *
            2 +
            pos_embedding_size,
            rnn_dim,
            self._pc)

    def item_size(self):
        return self._position_dim

    def encode(self, state):
        state_embedding = []
        for i, (person, hat) in zip(range(len(state)), state):
            state_embedding.append(dy.concatenate([self._pos_embeddings[i],
                                                   self._color_embeddings[self._color_vocab_int[person]],
                                                   self._color_embeddings[self._color_vocab_int[hat]]]))

        # Forward RNN.
        initial_state_fwd = self._encoder_fwd.initial_state()
        hidden_states_fwd = run_rnn(initial_state_fwd, state_embedding)

        # Backward RNN.
        initial_state_bwd = self._encoder_bwd.initial_state()
        hidden_states_bwd = run_rnn(
            initial_state_bwd, state_embedding[::-1])[::-1]

        state_embedding = [dy.concatenate([f, b, s]) for f, b, s in zip(
            hidden_states_fwd, hidden_states_bwd, state_embedding)]

        return state_embedding

