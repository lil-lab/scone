"""Script for training on the alchemy domain."""

import copy

from enum import Enum
from fsa import WorldState, ExecutionFSA
from vocabulary import EOS, ACTION_SEP, NO_ARG

import dynet as dy

# Alchemy-specific methods.

COLORS = ['y', 'o', 'r', 'g', 'b', 'p']
ACTION_POP = 'pop'
ACTION_PUSH = 'push'
ACTIONS = [ACTION_POP, ACTION_PUSH]

# States of the execution FSA.
class FSAStates(Enum):
    """Contains the possible states the Alchemy FSA can be in."""
    NO_ACTION = 1
    PUSH = 2
    PUSH_BEAKER = 3
    PUSH_BEAKER_COLOR = 4
    POP = 5
    POP_BEAKER = 6
    INVALID = 7

def token_is_beaker(token):
    """Returns whether a token represents a beaker.

    Inputs:
        token (str): The token.

    Returns:
        Boolean if token represents a beaker.
    """
    return token.isdigit() and 1 <= int(token) <= 7

def valid_feeds_push_beaker():
    """Returns all colors -- only can pass these after pushing a location."""
    # Can push all colors.
    return COLORS

def valid_feeds_push_beaker_color():
    """Returns valid actions after pushing a color to a beaker -- action sep."""
    # Must complete action.
    return [ACTION_SEP]

def valid_feeds_pop_beaker():
    """After popping, must return the action separator."""
    # Must complete action.
    return [ACTION_SEP]

def valid_feeds_invalid():
    """If invalid, just push the EOS."""
    # Nothing is valid, just wrap it up.
    return [EOS]

# Execution FSA.
class AlchemyFSA(ExecutionFSA):
    """FSA for the Alchemy domain.

    Attributes:
        _state (AlchemyState): The current world state.
        _fsa_state (FSAStates): The current FSA state.
        _current_beaker (int or None): The current beaker being used.
        _current_color (char or None): The current color being pushed.
    """
    def __init__(self, state):
        self._state = state
        self._fsa_state = FSAStates.NO_ACTION
        self._current_beaker = None
        self._current_color = None

    def is_in_action(self):
        return self._fsa_state == FSAStates.NO_ACTION

    def is_valid(self):
        return self._fsa_state != FSAStates.INVALID

    def state(self):
        return self._state

    def _valid_feeds_no_action(self):
        # If all beakers are empty, only 'push' is possible, or wrap it up.
        if all([len(b) == 0 for b in self._state.beakers()]):
            return [ACTION_PUSH, EOS]
        else:
            return ACTIONS + [EOS]

    def _valid_feeds_push(self):
        # Can push to all beakers. Beaker IDs start at 1.
        return map(str, range(1, len(self._state.beakers()) + 1))

    def _valid_feeds_pop(self):
        # Can pop from all beakers that have something in them.
        return map(lambda i: str(i + 1),
                   filter(lambda i: len(self._state.beakers()[i]) != 0,
                          range(len(self._state.beakers()))))


    def valid_feeds(self):
        valid_funcs = {
            FSAStates.NO_ACTION: self._valid_feeds_no_action,
            FSAStates.PUSH: self._valid_feeds_push,
            FSAStates.PUSH_BEAKER: valid_feeds_push_beaker,
            FSAStates.PUSH_BEAKER_COLOR: valid_feeds_push_beaker_color,
            FSAStates.POP: self._valid_feeds_pop,
            FSAStates.POP_BEAKER: valid_feeds_pop_beaker,
            FSAStates.INVALID: valid_feeds_invalid
        }
        return valid_funcs[self._fsa_state]()

    def valid_actions(self, all_action_vocabulary):
        valid_actions = [ (EOS, NO_ARG, NO_ARG) ]
        for location in range(7):
            for color in COLORS:
                valid_actions.append(("push", str(location + 1), color))
                if self._state.beakers()[location]:
                    valid_actions.append(("pop", str(location + 1), color))
            

    def peek_complete_action(self, act, arg1, arg2):
        if self._fsa_state != FSAStates.NO_ACTION:
            return None

        if act == ACTION_POP and token_is_beaker(arg1) and arg2 == NO_ARG:
            s = self._state.pop(int(arg1))
            return s

        if act == ACTION_PUSH and token_is_beaker(arg1) and arg2 in COLORS:
            s = self._state.push(
                int(arg1), arg2)
            return s

        raise Exception('should never happen')


    def feed_complete_action(self, act, arg1, arg2):
        if self._fsa_state != FSAStates.NO_ACTION:
            self._fsa_state = FSAStates.INVALID
            return None

        if act == ACTION_POP and token_is_beaker(arg1) and arg2 == NO_ARG:
            self._state = self._state.pop(int(arg1))
            if self._state is None:
                self._fsa_state = FSAStates.INVALID
            else:
                self._fsa_state = FSAStates.NO_ACTION
            return self._state

        if act == ACTION_PUSH and token_is_beaker(arg1) and arg2 in COLORS:
            self._state = self._state.push(
                int(arg1), arg2)
            if self._state is None:
                self._fsa_state = FSAStates.INVALID
            else:
                self._fsa_state = FSAStates.NO_ACTION
            return self._state

        raise Exception('should never happen')


#    def feed(self, token):
#        if self._fsa_state == FSAStates.NO_ACTION and token == ACTION_POP:
#            self._fsa_state = FSAStates.POP
#        elif self._fsa_state == FSAStates.NO_ACTION and token == ACTION_PUSH:
#            self._fsa_state = FSAStates.PUSH
#        elif self._fsa_state == FSAStates.PUSH and token_is_beaker(token):
#            self._current_beaker = int(token)
#            self._fsa_state = FSAStates.PUSH_BEAKER
#        elif self._fsa_state == FSAStates.PUSH_BEAKER and token in COLORS:
#            self._current_color = token
#            self._fsa_state = FSAStates.PUSH_BEAKER_COLOR
#        elif self._fsa_state == FSAStates.POP and token_is_beaker(token):
#            self._current_beaker = int(token)
#            self._fsa_state = FSAStates.POP_BEAKER
#        elif self._fsa_state == FSAStates.PUSH_BEAKER_COLOR and token == ACTION_SEP:
#            self._state = self._state.push(
#                self._current_beaker, self._current_color)
#            if self._state is None:
#                self._fsa_state = FSAStates.INVALID
#            else:
#                self._fsa_state = FSAStates.NO_ACTION
#            return self._state
#        elif self._fsa_state == FSAStates.POP_BEAKER and token == ACTION_SEP:
#            self._state = self._state.pop(self._current_beaker)
#            if self._state is None:
#                self._fsa_state = FSAStates.INVALID
#            else:
#                self._fsa_state = FSAStates.NO_ACTION
#            return self._state
#        else:
#            self._fsa_state = FSAStates.INVALID
#
#        return None
