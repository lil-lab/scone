""" Functions and classes for storing data (interactions and utterances)."""
from collections import namedtuple

import json
import random
import vocabulary

def strip_unicode(text):
    """ Removes unicode buggy text.

    Inputs:
        text (list of str): The input text.

    Returns:
        list of str where all characters are unicode.

    """
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


# Methods to read and prepare data.
class Example(namedtuple('Example',
                         ('id',
                          'turn',
                          'utterance',
                          'history',
                          'actions',
                          'initial_state',
                          'final_state'))):
    """ Contains an Example in the dataset.

    Attributes:
        id (str): Unique identifier for the example.
        turn (int): The turn where this example occurs in the interaction.
        utterance (list of str): The utterance for the example.
        history (list of list of str): The previous utterances.
        actions (list of str): The gold actions mapped from the utterance.
        initial_state (AlchemyState): The alchemy state it begins in.
        final_state (AlchemyState): The gold final state.
    """
    __slots__ = ()

    def __str__(self):
        return '%s:%d\n%s\n%s\n%s --> %s' % (self.id,
                                             self.turn,
                                             ' '.join(self.utterance),
                                             ' '.join(self.actions),
                                             self.initial_state,
                                             self.final_state)


class Interaction():
    """ Contains an Interaction in the dataset.

    Attributes:
        _id (str): The unique identifier for the interaction.
        _init_state (AlchemyState): The alchemy state at the beginning.
        _utterances (list of list of str): The utterances used.
        _action_seqs (list of list of str): The gold action sequences.
        _final_states (list of AlchemyState): The state after each utterance.
    """
    def __init__(self, unique_id, utterances, action_seqs, states):
        self._id = unique_id
        self._init_state = states[0]
        self._utterances = utterances
        self._action_seqs = action_seqs
        self._final_states = states[1:]

    def get_init_state(self):
        """ Gets the initial state of the interaction.

        Returns:
            WorldState representing the initial state in the interaction.
        """
        return self._init_state

    def get_utterances(self):
        """ Gets all of the utterances in the interaction.

        Returns:
            list of list of str, representing all of the utterances used.
        """
        return self._utterances

    def get_final_state(self, turn=None):
        """ Gets the final state at a specific turn (or none).

        Inputs:
            turn (int, optional): The turn to get the state for.

        Returns:
            WorldState representing the state at the turn, or the final state in
                the interaction (if turn is None).
        """
        return self._final_states[turn] if turn else self._final_states[-1]

    def __str__(self):
        return '%s\t%s\n%s' % (self._id, str(self._init_state),
                               '\n'.join(map(lambda x: '%s\t%s\t-->\t%s' %
                                             (' '.join(x[0]),
                                              str(x[1]),
                                              str(x[2])),
                                             zip(self._utterances,
                                                 self._action_seqs,
                                                 self._final_states))))

    def get_examples(self):
        """ Gets all of the examples (per turn) in the Interaction.

        Returns:
            list of Example, representing all the examples in the interaction.
        """
        examples = []
        history = []
        state = self._init_state
        for turn, utterance, action_seq, final_state in zip(range(len(self._utterances)),
                                                            self._utterances,
                                                            self._action_seqs,
                                                            self._final_states):
            examples.append(Example(self._id + str(turn),
                                    turn,
                                    utterance,
                                    history[:],
                                    action_seq,
                                    state,
                                    final_state))
            history.append(utterance)
            state = final_state
        return examples


def prepare_data(data, state_builder, sort=False):
    """ Prepares data by creating Interactions.

    Inputs:
        data (list of dict): Contains all of the data.
        state_builder (lambda dict: WorldState): Maps from a dict world state
            to an actual WorldState.
        sort (bool): Whether or not to sort the data.

    Returns:
        list of Interaction and list of Example
    """
    interactions = []
    for interaction in data:
        init_state = state_builder(interaction['initial_env'])
        unique_id = interaction['identifier']

        # Get data from json.
        utterances, actions_seqs, final_states = zip(*[(x['instruction'],
                                                        x['actions'],
                                                        x['after_env']) for x in
                                                       interaction['utterances']])

        # Process data.
        utterances = [strip_unicode(x).split() for x in utterances]
        final_states = [state_builder(x) for x in final_states]
        if sort:
            actions_seqs = [[token for action in sorted(seq) for token in action.split(
            ) + [vocabulary.ACTION_SEP]] for seq in actions_seqs]
        else:
            actions_seqs = [[token for action in seq for token in action.split(
            ) + [vocabulary.ACTION_SEP]] for seq in actions_seqs]

        # Create interaction object.
        interactions.append(
            Interaction(
                unique_id,
                utterances,
                actions_seqs,
                [init_state] + final_states))

    return interactions, [
        example for i in interactions for example in i.get_examples()]


def chunks(input_list, number):
    """Create equally sized (n) chunks from a list (l).

    Inputs:
        input_list (list): List to chunk.
        number (int): Number of chunks to make.

    """
    return [input_list[i:i + number] for i in range(0, len(input_list), number)]

def load_data(train_file,
              dev_file,
              test_file,
              state_type,
              args,
              sort=True):
    """Loads the train, dev, and test data, and gets the vocabularies."""
    # Read and prepare data.
    train = json.load(open(train_file))
    dev = json.load(open(dev_file))
    test = json.load(open(test_file))

    train_interactions, train = prepare_data(train, state_type, sort)
    dev_interactions, dev = prepare_data(dev, state_type, sort)
    test_interactions, test = prepare_data(test, state_type, sort)

    # Get intput and output vocabularies. Using test as well to avoid UNKs --
    # tokens in test are not seen during training though.
    # These NEED to be sorted or you can't load the model files!!
    in_vocab = sorted(
        set([token for example in train + dev + test for token in example.utterance]))
    out_vocab = sorted(
        set([token for example in train + dev + test for token in example.actions]))

    # Take validation out of training data.

    random.seed(1)
    random.shuffle(train_interactions)
    val_interactions = train_interactions[:int(
        len(train_interactions) * args.validation_ratio)]
    val = [example for i in val_interactions for example in i.get_examples()]
    train_interactions = train_interactions[int(
        len(train_interactions) * args.validation_ratio):]
    train = [example for i in train_interactions for example in i.get_examples()]

    # Comment to enable testing.
    if not args.enable_testing:
        test_interactions = test = None

    print('Training: %d interactions, %d examples' %
          (len(train_interactions), len(train)))
    print('Validation: %d interactions, %d examples' %
          (len(val_interactions), len(val)))
    print('Development: %d interactions, %d examples' %
          (len(dev_interactions), len(dev)))
    if test:
        print('Test: %d interactions, %d examples' %
              (len(test_interactions), len(test)))

    print('Input vocabulary size: ', len(in_vocab))
    print('Output vocabulary size: ', len(out_vocab))

    return (train, train_interactions), \
           (dev, dev_interactions), \
           (val, val_interactions), \
           (test, test_interactions), \
           in_vocab, \
           out_vocab
