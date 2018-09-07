""" Contains various utility functions for multihead predictions. """

import dynet as dy

from vocabulary import NO_ARG

def flatten_triple(action_scores, location_scores, argument_scores):
    """ Flattens three scores vectors by summing over all possibilities. """
    num_actions = action_scores.dim()[0][0]
    num_locations = location_scores.dim()[0][0]
    num_arguments = argument_scores.dim()[0][0]

    expanded_arguments = dy.reshape(argument_scores, (num_arguments, 1)) \
        * dy.ones((1, num_locations))
    expanded_locations = dy.ones((num_arguments, 1)) \
        * dy.reshape(location_scores, (1, num_locations))

    # num_locations x num_arguments
    location_and_argument_scores = expanded_locations + expanded_arguments
    location_and_argument_expanded = dy.reshape(location_and_argument_scores,
                                                (num_locations * num_arguments, 1)) \
        * dy.ones((1, num_actions))

    expanded_actions = dy.ones((num_arguments * num_locations, 1)) \
        * dy.reshape(action_scores, (1, num_actions))

    final_scores = location_and_argument_expanded + expanded_actions

    # num_actions * num_locations x num_arguments
    final_scores = dy.reshape(final_scores, (num_actions * num_locations * num_arguments, 1))

    return final_scores

def readable_action(action, location, argument):
    """Constructs a string out of a tripled action.

    Inputs:
        action (str): The action.
        location (str): The location of the action.
        argument (str): The argument of the action.

    Returns:
        human-readable string representing the action.
    """
    string = action
    if location != NO_ARG:
        string += " " + location
    if argument != NO_ARG:
        string += " " + argument
    string += " ;"
    return string
