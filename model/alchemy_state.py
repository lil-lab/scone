"""Contains class for the Alchemy world state."""

from fsa import WorldState
from alchemy_fsa import AlchemyFSA
from util import edit_distance

# Alchemy-specific methods.
COLORS = ['y', 'o', 'r', 'g', 'b', 'p']
ACTION_POP = 'pop'
ACTION_PUSH = 'push'
ACTIONS = [ACTION_POP, ACTION_PUSH]


# Immutable world state. Action execution returns a new state.
class AlchemyState(WorldState):
    """ The Alchemy world state definition.

    Attributes:
        _beakers (list of list of str): Beakers in the state.
    """
    def __init__(self, string=None):
        self._beakers = [[]] * 7
        if string:
            string = [beaker.split(':')[1] for beaker in string.split()]
            self._beakers = []
            for beaker in string:
                if beaker == '_':
                    self._beakers.append([])
                else:
                    self._beakers.append(list(beaker))
        else:
            self._beakers = [[]] * 7

    def __eq__(self, other):
        return isinstance(other, AlchemyState) and self._beakers == other.beakers()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return ' '.join([str(i) +
                         ':' +
                         ''.join(beaker) if beaker else str(i) +
                         ':_' for i, beaker in zip(range(1, 8), self._beakers)])

    def __len__(self):
        return len(self._beakers)

    def __iter__(self):
        return self._beakers.__iter__()

    def beakers(self):
        """ Returns the beakers for the state. """
        return self._beakers

    def components(self):
        return self.beakers()

    def set_beakers(self, beakers):
        """ Sets the beakers of this class to something else.

        Inputs:
            beakers (list of list of str): The beakers to set.
        """
        self._beakers = beakers

    def set_beaker(self, index, new_value):
        """ Resets the units for a specific beaker.

        Inputs:
            index (int): The beaker to reset.
            new_value (list of str): The new values for the beaker.
        """
        self._beakers[index] = new_value

    def pop(self, beaker):
        """ Removes a unit from a beaker.

        Inputs:
            beaker (int): The beaker to pop from.

        Returns:
            AlchemyState, representing the state after popping.
        """
        beaker -= 1
        if self._beakers[beaker]:
            new_state = AlchemyState()
            new_state.set_beakers(self._beakers[:])
            new_state.set_beaker(beaker, self._beakers[beaker][:-1])
            return new_state
        return None

    def push(self, beaker, color):
        """ Adds a new unit to a beaker.

        Inputs:
            beaker (int): The beaker to add to.
            color (str): The color to add.
        Returns:
            AlchemyState, representing the state after pushing.
        """
        beaker -= 1
        new_state = AlchemyState()
        new_state.set_beakers(self._beakers[:])
        new_state.set_beaker(beaker, self._beakers[beaker] + [color])
        return new_state

    def execute_seq(self, actions):
        fsa = AlchemyFSA(self)
        for action in actions:
            peek_state = fsa.peek_complete_action(*action)
            if peek_state:
                fsa.feed_complete_action(*action)
        return fsa.state()

    def distance(self, other_state):
        """ Returns the distance between two WorldStates.

        Inputs:
            other_state (AlchemyState): The other alchemy state to compute the
                distance from.

        Returns:
            float representing the distance.
        """
        delta = 0
        for this_beaker, that_beaker in zip(
                self._beakers, other_state.beakers()):
            delta += edit_distance(this_beaker, that_beaker)
        return delta
