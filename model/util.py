"""Contains various useful functions for Dynet models and training."""

import dynet as dy

def run_rnn(init_state, input_vecs):
    """Gets RNN outputs for a sequence of vectors.

    Inputs:
        init_state (dy.RNNState): An RNN state indicating the beginning of the
            sequence.
        input_vecs (list of dy.Expression): List of vectors used to update the
            RNN state.

    Returns:
        list of dy.Expression, representing the hidden state at each point in
            encoding the sequence.
    """
    state = init_state
    states = state.add_inputs(input_vecs)
    rnn_outputs = [state.output() for state in states]

    return rnn_outputs


def attend(input_vectors, state, params, dropout_amount=0.):
    """Attends on some input vectors given a state and attention parameters.

    Inputs:
        input_vectors (list of dy.Expression): Vectors to attend on.
        state (dy.Expression): A state (query).
        params (dy.Expression): Attentional weights to transform the state before
            computing attention.
        dropout_amount (float, optional): The amount of dropout to apply after
            transforming the state.

    Returns:
        dy.Expression representing the weighted sum of input vectors given the
            computed attentional weights.
    """
    projected_state = dy.transpose(dy.reshape(
        state, (1, state.dim()[0][0])) * params)
    projected_state = dy.dropout(projected_state, dropout_amount)
    attention_weights = dy.select_rows(
        dy.transpose(projected_state) *
        dy.concatenate_cols(input_vectors),
        [0])[0]
    context = dy.concatenate_cols(
        input_vectors) * dy.softmax(attention_weights)
    return context, dy.softmax(attention_weights)

def shaping(previous_state,
            current_state,
            final_state,
            potential_function,
            default_reward):
    """Computes the shaping reward for a set of states and a potential function.

    Inputs:
        previous_state (WorldState): The previous world state.
        current_state (WorldState): The current world state.
        final_state (WorldState): The goal state.
        potential_function ((WorldState, WorldState) -> numeric)): A function
            that computes the potential from one state to another.
        default_reward (float): A reward to return if the current state is None.

    Returns:
        float, representing the shaping reward.
    """
    if current_state is not None:
        return potential_function(previous_state, final_state) \
            - potential_function(current_state, final_state)
    return default_reward

def levenshtein_d(v_1, v_2):
    """Levenshtein distance.

    Inputs:
        v_1: First vector.
        v_2: Second vector.

    Returns:
        float, representing the distance.
    """
    cost = 0
    if not v_1:
        return len(v_2)
    elif not v_2:
        return len(v_1)

    if v_1[-1] == v_2[-1]:
        cost = 0
    else:
        cost = 1
    return min([levenshtein_d(v_1[:-1], v_2) + 1,
                levenshtein_d(v_1, v_2[:-1]) + 1,
                levenshtein_d(v_1[:-1], v_2[:-1]) + cost])

def edit_distance(v_1, v_2):
    """ Edit distance between two vectors (from the end).

    Inputs:
        v_1: First vector.
        v_2: Second vector.

    Returns:
        float, representing the distance.
    """
    assert not '_' in v_1
    assert not '_' in v_2

    prefix = ''
    for c_1, c_2 in zip(v_1, v_2):
        if c_1 != c_2:
            break
        prefix += c_1

    return len(v_1) + len(v_2) - 2 * len(prefix)
