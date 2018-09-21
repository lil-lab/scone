from enum import Enum
from fsa import ExecutionFSA
from vocabulary import ACTION_SEP, NO_ARG


NUM_POSITIONS = 10
COLORS = ['y', 'o', 'r', 'g', 'b', 'p']

# Actions.
ACT_APPEAR_PERSON = 'appear_person'
ACT_APPEAR_HAT = 'appear_hat'
ACT_REMOVE_PERSON = 'remove_person'
ACT_REMOVE_HAT = 'remove_hat'

# States of the execution FSA.


class FSAStates(Enum):
    NO_ACTION = 1
    INVALID = 2

# Execution FSA.


class SceneFSA(ExecutionFSA):
    def __init__(self, state):
        self._state = state
        self._fsa_state = FSAStates.NO_ACTION
        self._current_pos = None
        self._current_color = None

    def is_in_action(self):
        return self._fsa_state == FSAStates.NO_ACTION

    def is_valid(self):
        return self._fsa_state != FSAStates.INVALID

    def state(self):
        return self._state

    def _is_pos_index(self, token):
        return token.isdigit() and 1 <= int(token) <= 10

    def peek_complete_action(self, act, arg1, arg2):
        if self._fsa_state != FSAStates.NO_ACTION:
            return None

        if self._is_pos_index(arg1):
            if act == ACT_APPEAR_PERSON and arg2 in COLORS:
                s = self._state.appear_person(
                    int(arg1) - 1, arg2)
                return s
            if act == ACT_REMOVE_PERSON and arg2 == NO_ARG:
                s = self._state.remove_person(int(arg1) - 1)
                return s
            if act == ACT_APPEAR_HAT and arg2 in COLORS:
                s = self._state.appear_hat(
                    int(arg1) - 1, arg2)
                return s
            if act == ACT_REMOVE_HAT and arg2 == NO_ARG:
                s = self._state.remove_hat(int(arg1) - 1)
                return s

        raise Exception('should never happen')

    def feed_complete_action(self, act, arg1, arg2):
        if self._fsa_state != FSAStates.NO_ACTION:
            self._fsa_state = FSAStates.INVALID
            return None

        if self._is_pos_index(arg1):
            if act == ACT_APPEAR_PERSON and arg2 in COLORS:
                self._state = self._state.appear_person(
                    int(arg1) - 1, arg2)
            elif act == ACT_REMOVE_PERSON and arg2 == NO_ARG:
                self._state = self._state.remove_person(int(arg1) - 1)
            elif act == ACT_APPEAR_HAT and arg2 in COLORS:
                self._state = self._state.appear_hat(
                    int(arg1) - 1, arg2)
            elif act == ACT_REMOVE_HAT and self._is_pos_index(arg1):
                self._state = self._state.remove_hat(int(arg1) - 1)

            if self._state:
                self._fsa_state = FSAStates.NO_ACTION
            else:
                self._fsa_state = FSAStates.INVALID
            return self._state

        raise Exception('should never happen')
