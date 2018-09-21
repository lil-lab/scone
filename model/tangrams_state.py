from fsa import WorldState
from nltk.metrics import edit_distance
from tangrams_fsa import NUM_POSITIONS, TangramsFSA
from util import levenshtein_d

class TangramsState(WorldState):
    def __init__(self, string=None):
        self._tangrams = []
        if string:
            self._tangrams = [position.split(':')[1]
                              for position in string.split()]

    def __eq__(self, other):
        return isinstance(other, TangramsState) and self._tangrams == other._tangrams

    def __ne__(self, other):
        return not self.__eq__(other)

    def tangrams(self):
        return self._tangrams

    def components(self):
        return self.tangrams()

    def __str__(self):
        return ' '.join([str(i) + ':' + str(x)
                         for i, x in zip(range(1, len(self) + 1), self._tangrams)])

    def remove(self, index):
        if index > len(self._tangrams):
            return None
        new_state = TangramsState()
        new_state._tangrams = self._tangrams[:]
        del new_state._tangrams[index - 1]
        return new_state

    def insert(self, index, shape):
        if shape in self._tangrams:
            return None
        if index > len(self._tangrams) + 1:
            return None
        new_state = TangramsState()
        new_state._tangrams = self._tangrams[:]
        new_state._tangrams.insert(index - 1, shape)
        if (len(new_state)) > NUM_POSITIONS:
            return None
        return new_state

    def __len__(self):
        return len(self._tangrams)

    def __iter__(self):
        return self._tangrams.__iter__()

    # def execute(self, action):
    #     if len(action) == 2 and action[0] == 'remove' and is_digit_between(action[1], 1, len(self)):
    #         new_tangrams = self._tangrams[:]
    #         del new_tangrams[int(action[1]) - 1]
    #         new_state = State()
    #         new_state._tangrams = new_tangrams
    #         return new_state

    #     if len(action) == 3 and action[0] == 'insert' and is_digit_between(action[1], 1, 5) and action[2] in shapes:
    #         new_tangrams = self._tangrams[:]
    #         new_tangrams.insert(int(action[1]) - 1, action[2])
    #         # Avoid creating states that the encoder can't handle.
    #         if len(new_tangrams) > NUM_POSITIONS:
    #             return None
    #         new_state = State()
    #         new_state._tangrams = new_tangrams
    #         return new_state

    def execute_seq(self, actions):
        fsa = TangramsFSA(self)
        for action in actions:
            peek_state = fsa.peek_complete_action(*action)
            if peek_state:
                fsa.feed_complete_action(*action)
        return fsa.state()

    def distance(self, other_state):
        return edit_distance(self.tangrams(),
                             other_state.tangrams(),
                             substitution_cost=2,
                             transpositions=False)
