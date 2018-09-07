"""Contains the vocabulary class."""

BEG = "<BEG>"
EOS = "<EOS>"
NO_ARG = "<NONE>"
SEP = "<SEP>"
CURRENT_SEP = "<CURRENT_SEP>"
ACTION_SEP = ";"

class Vocabulary():
    """ Stores vocabulary.

    Attributes:
        _vocabulary_list (list): List of the vocabulary.
        vocabulary_ints (dict type->int): Maps from word types to integers.
    """
    def __init__(self, raw_vocabulary, extra_tokens):
        self._vocabulary_list = list(raw_vocabulary) + extra_tokens
        self.vocabulary_ints = dict(zip(self._vocabulary_list,
                                        range(len(self._vocabulary_list))))

    def __len__(self):
        return len(self._vocabulary_list)

    def __iter__(self):
        return iter(self._vocabulary_list)

    def __contains__(self, token):
        return token in self._vocabulary_list

    def lookup_index(self, token):
        """ Looks up the index of a specific token.

        Inputs:
            token (word type): The token to look up.

        Returns:
            integer representing its index.
        """
        return self.vocabulary_ints[token]

    def lookup_token(self, index):
        """ Looks up a token at a specific index.

        Inputs:
            index (integer)

        Returns:
            word type at that index
        """
        return self._vocabulary_list[index]
