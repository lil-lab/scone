"""Tangrams FSA"""

from enum import Enum
from fsa import ExecutionFSA
from vocabulary import NO_ARG

ACT_INSERT = "insert"
ACT_REMOVE = "remove"
SHAPES = ['A', 'B', 'C', 'D', 'E']
NUM_POSITIONS = 6


# States of the execution FSA.
class FSAStates(Enum):
    NO_ACTION = 1
    REMOVE = 2
    REMOVE_INDEX = 3
    INSERT = 4
    INSERT_INDEX = 5
    INSERT_INDEX_SHAPE = 6
    INVALID = 7

def token_is_spot(token):
    return token.isdigit() and 1 <= int(token) <= NUM_POSITIONS

class TangramsFSA(ExecutionFSA):
    def __init__(self, state):
        self._state = state
        self._fsa_state = FSAStates.NO_ACTION
        self._current_index = None
        self._current_shape = None

    def is_valid(self):
        return self._fsa_state != FSAStates.INVALID

    def is_in_action(self):
        return self._fsa_state == FSAStates.NO_ACTION

    def state(self):
        return self._state

    def _valid_feeds_no_action(self):
        if len(self._state) == 0:
            return [ACT_INSERT, EOS]
        elif len(filter(lambda s: s not in self._state.tangrams(), shapes)) == 0:
            return [ACT_REMOVE, EOS]
        else:
            return [ACT_INSERT, ACT_REMOVE, EOS]

    def _valid_feeds_remove(self):
        return [str(i + 1) for i in range(len(self._state))]

    def _valid_feeds_remove_index(self):
        return [ACTION_SEP]

    def _valid_feeds_insert(self):
        # Can insert anywhere in the list and at its end.
        return [str(i + 1) for i in range(len(self._state) + 1)]

    def _valid_feeds_insert_index(self):
        # Can only insert shapes that are not present.
        return filter(lambda s: s not in self._state.tangrams(), shapes)

    def _valid_feeds_insert_index_shape(self):
        return [ACTION_SEP]

    def _valid_feeds_invalid(self):
        return [EOS]

    def valid_feeds(self):
        valid_funcs = {
            FSAStates.NO_ACTION: self._valid_feeds_no_action,
            FSAStates.REMOVE: self._valid_feeds_remove,
            FSAStates.REMOVE_INDEX: self._valid_feeds_remove_index,
            FSAStates.INSERT: self._valid_feeds_insert,
            FSAStates.INSERT_INDEX: self._valid_feeds_insert_index,
            FSAStates.INSERT_INDEX_SHAPE: self._valid_feeds_insert_index_shape,
            FSAStates.INVALID: self._valid_feeds_invalid,
        }

        return valid_funcs[self._fsa_state]()

    def peek_complete_action(self, act, arg1, arg2):
        if self._fsa_state != FSAStates.NO_ACTION:
            return None

        # Can only remove an 
        if act == ACT_REMOVE and token_is_spot(arg1) and arg2 == NO_ARG:
            s = self._state.remove(int(arg1))
            return s
        if act == ACT_INSERT and token_is_spot(arg1) and arg2 in SHAPES:
            s = self._state.insert(int(arg1), arg2)
            return s
        raise Exception('SHOULD NEVER HAPPEN')

    def feed_complete_action(self, act, arg1, arg2):
        if self._fsa_state != FSAStates.NO_ACTION:
            self._fsa_state = FSAStates.INVALID
            return None

        if act == ACT_REMOVE and token_is_spot(arg1) and arg2 == NO_ARG:
            self._state = self._state.remove(int(arg1))
            if self._state is None:
                self._fsa_state = FSAStates.INVALID
            else:
                self._fsa_state = FSAStates.NO_ACTION
            return self._state

        if act == ACT_INSERT and token_is_spot(arg1) and arg2 in SHAPES:
            self._state = self._state.insert(int(arg1), arg2)
            if self._state is None:
                self._fsa_state = FSAStates.INVALID
            else:
                self._fsa_state = FSAStates.NO_ACTION
            return self._state
        raise Exception('should never happen!')

#    def feed(self, token):
#        s = self._fsa_state
#        if s is FSAStates.NO_ACTION and token == ACT_INSERT:
#            self._fsa_state = FSAStates.INSERT
#        elif s is FSAStates.NO_ACTION and token == ACT_REMOVE:
#            self._fsa_state = FSAStates.REMOVE
#        elif s is FSAStates.REMOVE and token in self._valid_feeds_remove():
#            self._current_index = int(token)
#            self._fsa_state = FSAStates.REMOVE_INDEX
#        elif s is FSAStates.REMOVE_INDEX and token in self._valid_feeds_remove_index():
#            self._state = self._state.remove(self._current_index)
#            self._fsa_state = FSAStates.NO_ACTION
#            return self._state
#        elif s is FSAStates.INSERT and token in self._valid_feeds_insert():
#            self._current_index = int(token)
#            self._fsa_state = FSAStates.INSERT_INDEX
#        elif s is FSAStates.INSERT_INDEX and token in self._valid_feeds_insert_index():
#            self._current_shape = token
#            self._fsa_state = FSAStates.INSERT_INDEX_SHAPE
#        elif s is FSAStates.INSERT_INDEX_SHAPE and token in self._valid_feeds_insert_index_shape():
#            self._state = self._state.insert(
#                self._current_index, self._current_shape)
#            self._fsa_state = FSAStates.NO_ACTION
#            return self._state
#        else:
#            self._fsa_state = FSAStates.INVALID
#
#        return None
