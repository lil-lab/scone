"""
RNNStateEncoder for the Alchemy domain.
"""

import dynet as dy
from vocabulary import EOS
from util import run_rnn

RNNBuilder = dy.LSTMBuilder
LOG_DIR = 'logs-alchemy'

# Alchemy-specific methods.
COLORS = ['y', 'o', 'r', 'g', 'b', 'p']
ACTION_POP = 'pop'
ACTION_PUSH = 'push'
ACTIONS = [ACTION_POP, ACTION_PUSH]

BEAKER_EOS = EOS
COLORS_VOCAB = COLORS + [BEAKER_EOS]
NUM_BEAKERS = 7

class AlchemyStateEncoder:
    """
    RNN State encoder for the Alchemy domain.

    Attributes:
        _beaker_dim (int): The size of the beaker embeddings.
        _pos_embedding_size (int): The size of the positional embeddings.
        _pc (dy.ParameterCollection): The parameter collection of the model.
        _color_embeddings (dy.LookupParameters): Word embeddings for colors.
        _beaker_pos_embeddings (dy.LookupParameters): Positional embeddings
            of the beakers.
        _color_vocab_int (dictionary): Vocabulary for colors.
        _beaker_encoder (dy._RNNBuiler): RNN for encoding the beakers.
    """
    def __init__(
            self,
            pc,
            enc_layers=1,
            beaker_dim=20,
            color_embedding_size=10,
            pos_embedding_size=10):
        self._beaker_dim = beaker_dim
        self._pos_embedding_size = pos_embedding_size
        self._pc = pc
        self._color_embeddings = self._pc.add_lookup_parameters(
            (len(COLORS_VOCAB), color_embedding_size), name="color-embeddings")
        self._beaker_pos_embeddings = self._pc.add_lookup_parameters(
            (NUM_BEAKERS, pos_embedding_size), name="beaker-embeddings")

        # Color indexing.
        self._color_vocab_int = dict(
            zip(COLORS_VOCAB, range(len(COLORS_VOCAB))))

        # Beaker RNN.
        self._beaker_encoder = RNNBuilder(
            enc_layers, color_embedding_size, beaker_dim, self._pc)

    def item_size(self):
        """ Returns the size of the embedding for the state encoder.

        Returns:
            beaker dimensions + positional embeddings size.
        """
        return self._beaker_dim + self._pos_embedding_size

    def _colors_to_int(self, colors):
        colors = list(colors) + [BEAKER_EOS]
        return [self._color_vocab_int[token] for token in colors]

    def _embed_colors(self, colors):
        return [self._color_embeddings[c] for c in colors]

    def encode(self, state):
        """Encodes an AlchemyState.

        Inputs:
            state (AlchemyState): The state to encode.

        Returns:
            dy.Expression representing the encoded state.
        """
        state_embedding = []
        for i, beaker in zip(range(len(state)), state):
            # Embed the colors.
            embedded_colors = self._embed_colors(self._colors_to_int(beaker))
            initial_state = self._beaker_encoder.initial_state()
            beaker_embedding = run_rnn(initial_state, embedded_colors)[-1]
            state_embedding.append(dy.concatenate(
                [beaker_embedding, self._beaker_pos_embeddings[i]]))
        return state_embedding
