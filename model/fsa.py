""" Abstract FSA methods and classes """

import copy
from abc import abstractmethod

from vocabulary import EOS, NO_ARG

class WorldState():
    """Abstract class for a world state."""
    @abstractmethod
    def execute_seq(self, actions):
        """Execute a sequence of actions on a state.

        Args:
            actions (list of any): The sequence of actions to execute.
        """
        pass

    @abstractmethod
    def distance(self, other_state):
        """ Computes a distance between itself and another state of the same type.

        Args:
            other_state (WorldState): the state to compare with.

        Returns:
            float, representing the distance.
        """
        pass


class ExecutionFSA():
    """Abstract class for an FSA that can execute various actions."""
    @abstractmethod
    def is_valid(self):
        """Returns whether the current FSA state is valid."""
        pass

    @abstractmethod
    def is_in_action(self):
        """Returns whether the current FSA state is in an action."""
        pass

    @abstractmethod
    def state(self):
        """Returns the current FSA state."""
        pass

    @abstractmethod
    def valid_feeds(self):
        """Returns the valid actions that can be executed."""
        pass

    @abstractmethod
    def peek_complete_action(self, action, arg1, arg2):
        pass

    @abstractmethod
    def feed_complete_action(self, action, arg1, arg2):
        pass


