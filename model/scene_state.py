from fsa import WorldState
from scene_fsa import SceneFSA, COLORS

EMPTY_COLOR = '_'

class SceneState(WorldState):
    def __init__(self, string=None):
        self._people = ['_'] * 10
        self._hats = ['_'] * 10
        if string:
            string = [beaker.split(':')[1] for beaker in string.split()]
            self._people, self._hats = zip(*[(x[0], x[1]) for x in string])
            self._people = list(self._people)
            self._hats = list(self._hats)

    def __eq__(self, other):
        return isinstance(
            other, SceneState) and self._people == other._people and self._hats == other._hats

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return ' '.join([str(i) + ':' + p + h for i, p,
                         h in zip(range(1, len(self) + 1), self._people, self._hats)])

    def __len__(self):
        return len(self._people)

    def __iter__(self):
        return zip(self._people, self._hats).__iter__()

    def people_indices(self):
        return map(
            itemgetter(0), filter(
                lambda x: x[1] != EMPTY_COLOR, zip(
                    range(
                        1, len(
                            self._people) + 1), self._people)))

    def empty_indices(self):
        return map(
            itemgetter(0), filter(
                lambda x: x[1] == EMPTY_COLOR, zip(
                    range(
                        1, len(
                            self._people) + 1), self._people)))

    def hat_indices(self):
        return map(
            itemgetter(0), filter(
                lambda x: x[1] != EMPTY_COLOR, zip(
                    range(
                        1, len(
                            self._hats) + 1), self._hats)))

    def empty_hat_indices(self):
        return map(
            itemgetter(0), filter(
                lambda x: x[1] == EMPTY_COLOR, zip(
                    range(
                        1, len(
                            self._hats) + 1), self._hats)))

    def people(self):
        return self._people

    def hats(self):
        return self._hats

    def components(self):
        return [(person, hat) for person, hat in zip(self.people(), self.hats())]

    def remove_hat(self, index):
        new_state = SceneState()
        new_state._people = self._people[:]
        new_state._hats = self._hats[:]
        current_color = new_state._hats[index]
        
        # Can't remove a nonexistent hat.
        if current_color == EMPTY_COLOR:
            return None
        else:
            new_state._hats[index] = EMPTY_COLOR
            return new_state


    def remove_person(self, index):
        new_state = SceneState()
        new_state._people = self._people[:]
        new_state._hats = self._hats[:]
        current_color = new_state._people[index]
        
        # Can't remove a nonexistent person.
        if current_color == EMPTY_COLOR:
            return None
        else:
            new_state._people[index] = EMPTY_COLOR
            return new_state

    def appear_hat(self, index, color):
        new_state = SceneState()
        new_state._people = self._people[:]
        new_state._hats = self._hats[:]
        current_color = new_state._hats[index]

        # Can only put a hat where there isn't already one.
        if current_color != EMPTY_COLOR:
            return None
        else:
            new_state._hats[index] = color
            return new_state

    def appear_person(self, index, color):
        new_state = SceneState()
        new_state._people = self._people[:]
        new_state._hats = self._hats[:]
        current_color = new_state._people[index]

        # Can only put a person where there isn't already one.
        if current_color != EMPTY_COLOR:
            return None
        else:
            new_state._people[index] = color
            return new_state

    def execute_seq(self, actions):
        fsa = SceneFSA(self)
        for action in actions:
            peek_state = fsa.peek_complete_action(*action)
            if peek_state:
                fsa.feed_complete_action(*action)
        return fsa.state()

    def distance(self, other_state):
        delta = 0
        for this_people, that_people in zip(self._people, other_state.people()):
            if this_people != that_people:
                # Only one if one of the two is empty.
                if this_people == EMPTY_COLOR or that_people == EMPTY_COLOR:
                    delta += 1
                else:
                    delta += 2
        for this_hats, that_hats in zip(self._hats, other_state.hats()):
            if this_hats != that_hats:
                # Only one if one of the two is empty.
                if this_hats == EMPTY_COLOR or that_hats == EMPTY_COLOR:
                    delta += 1
                else:
                    delta += 2
        return delta


