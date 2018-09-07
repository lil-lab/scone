"""
Functions for evaluating data: loss over a set, accuracy over utterance
predictions, and accuracy over entire interactions.
"""
from collections import defaultdict
from operator import itemgetter

import dynet as dy
import tqdm

from data import chunks
from model_util import readable_action
from visualize_attention import AttentionGraph
from vocabulary import EOS, SEP, CURRENT_SEP
from model import LEN_LIMIT

# Procedure to compute loss data set.
def get_set_loss(model,
                 dataset,
                 fsa_builder,
                 batch_size=1):
    """ Computes average loss over examples in a dataset.

    Inputs:
        model (SconeModel): The model used to compute loss.
        dataset (list of examples): List of labeled SCONE examples.
        fsa_builder (ExecutableFSA): FSA for this domain.
        batch_size (int, optional): Batch size to use for the examples.
    
    Returns:
        float, the average loss over examples in the dataset.
    """
    losses = []
    for batch in tqdm.tqdm(
            chunks(
                dataset,
                batch_size),
            desc='Computing loss',
            ncols=80):
        dy.renew_cg()
        token_losses = [
            model.get_losses(
                example.utterance,
                example.actions,
                example.initial_state,
                example.history,
                fsa_builder(example.initial_state)) for example in batch]
        token_losses = [token_loss for l in token_losses for token_loss in l]
        dy.esum(token_losses).value()
        losses += map(lambda x: x.value(), token_losses)
    return sum(losses) / len(losses)

def utterance_accuracy(model,
                       dataset,
                       fsa_builder,
                       logfile=None,
                       args=None,
                       reward_function = None):
    """ Computes per-utterance accuracy over a dataset.

    Inputs:
        model (SconeModel): The trained model.
        dataset (list of examples): List of labeled SCONE examples.
        fsa_builder (ExecutableFSA): FSA for this domain.
        logfile (str, optional): Filename to log results to.

    Returns:
        float, representing the mean execution accuracy for the dataset.
    """
    logfile = open(logfile, 'w') if logfile else None
    exec_correct_count = defaultdict(float)
    correct_count = defaultdict(float)
    count = defaultdict(float)
    token_accuracy = 0.
    reward_sum = 0.
    for _, example in tqdm.tqdm(
            enumerate(dataset), desc='Computing accuracy', ncols=80):
        count[example.turn] += 1
        output, _ = model.generate(example.utterance,
                                   example.initial_state,
                                   example.history,
                                   fsa_builder(example.initial_state))
        grouped_actions = [tuple(tok) for tok in model.group_tokens(example.actions)]
        readable_output = " ".join([readable_action(*action) for action in output])
        split_output = readable_output.split(" ")
        tok_correct = 0.
        if output:
            for i, token in enumerate(output):
                if i < len(grouped_actions) and token == grouped_actions[i]:
                    tok_correct += 1.
            token_accuracy += tok_correct / len(output)
        else:
            token_accuracy += 0.

        if split_output == example.actions:
            correct_count[example.turn] += 1

        fsa = fsa_builder(example.initial_state)
        reward = 0.
        prev_state = fsa.state()
        for i, action in enumerate(output):
            peek_state = fsa.peek_complete_action(*action)
            if reward_function:
                reward += reward_function(example.final_state,
                                          prev_state,
                                          peek_state,
                                          action,
                                          i == LEN_LIMIT,
                                          args.verbosity_penalty)
            if peek_state:
                fsa.feed_complete_action(*action)
            prev_state = fsa.state()
        exec_result = fsa.state()
        reward_sum += reward / float(len(output) + 1)

        if exec_result == example.final_state:
            exec_correct_count[example.turn] += 1
        if logfile:
            logfile.write(
                '%s\t%s\t%s\n' %
                ('CORRECT_STRING' if output == example.actions else 'WRONG_STRING',
                 'CORRECT_EXEC' if exec_result == example.final_state else 'WRONG_EXEC',
                 example))
            readable_output = " ".join([readable_action(*action) for action in output])
            if readable_output != example.actions:
                logfile.write('%s\n' % readable_output)
    accuracies = dict([(k, correct_count[k] / v) for k, v in count.items()])
    exec_accuracies = dict([(k, exec_correct_count[k] / v)
                            for k, v in count.items()])
    mean_accuracy = sum(accuracies.values()) / len(accuracies)
    exec_mean_accuracy = sum(exec_accuracies.values()) / len(exec_accuracies)
    if logfile:
        for key, value in sorted(accuracies.items(), key=itemgetter(0)):
            logfile.write(
                'Accuracy-%d: %.4f (string), %.4f (exec)\n' %
                (key, value, exec_accuracies[key]))
        logfile.write(
            'Mean accuracy: %.4f (string), %.4f (exec)' %
            (mean_accuracy, exec_mean_accuracy))
        logfile.close()
    return exec_mean_accuracy, reward_sum / len(dataset), token_accuracy / len(dataset)

def attention_analysis(model, example, fsa_builder, name = ""):
    output, attentions = model.generate(example.utterance,
                                       example.initial_state,
                                       example.history,
                                       fsa_builder(example.initial_state))
    attention_graphs = { }
    fsa = fsa_builder(example.initial_state)
    for key in attentions[0]:
        attention_keys = []
        if key == 'history':
            if example.history:
                for utterance in example.history:
                    attention_keys.extend(utterance)
                    attention_keys.append(SEP)
                attention_keys = attention_keys[:-1]
                attention_keys.append(CURRENT_SEP)
            else:
                continue
        elif key == 'utterance':
            attention_keys = example.utterance + [EOS]
        elif key.startswith('initial'):
            attention_keys = example.initial_state.components()
        else:
            # TODO: Deal with current/initial state as different (must pass in args)
            # TODO: showing attention over current state will be difficult given
            # that we change state over time
            continue 
        try:
            attention_graphs[key] = AttentionGraph(attention_keys)
        except ValueError as e:
            print(e)
            print("Couldn't create graph for key " + str(key))
            exit()
    for output_token, attention in zip(output, attentions):
        for key, distribution in attention.items():
            # TODO: make sure the key is associated with an attention graph
            # (currently not used because not all keys are supported ATM)
            if key in attention_graphs:
                graph = attention_graphs[key]
                try:
                    graph.add_attention(readable_action(*output_token), distribution.value())
                except ValueError as e:
                    print(e)
                    print("Probability distribution of size " + str(len(distribution.value())) \
                          + " but expected one for each key (" + str(graph.keys) + ")")
                    exit()
        peek_state = fsa.peek_complete_action(*output_token)
        if peek_state:
            fsa.feed_complete_action(*output_token)

    if fsa.state() == example.final_state:
        for key, graph in attention_graphs.items():
            graph.render(name + "-" + key + ".png")
    

def interaction_accuracy(model, dataset, fsa_builder, logfile=None):
    """ Computes the interaction-level accuracy over a dataset.

    Inputs:
        model (SconeModel): The trained model.
        dataset (list of examples): List of labeled SCONE examples.
        fsa_builder (ExecutableFSA): FSA for this domain.
        logfile (str, optional): Filename to log results to.

    Returns:
        float, representing the mean execution accuracy for the dataset.
    """
    logfile = open(logfile, 'w') if logfile else None
    utt3_correct_count = 0.0
    correct_count = 0.0
    count = 0.0
    for _, interaction in tqdm.tqdm(
            enumerate(dataset), desc='Computing accuracy', ncols=80):
        count += 1
        state = interaction.get_init_state()
        history = []
        intermediate_states = [state]
        action_seqs = []
        for i, utterance in enumerate(interaction.get_utterances()):
            output, _ = model.generate(
                utterance, state, history, fsa_builder(state))
            action_seqs.append(output)
            state = state.execute_seq(output)
            if i == 2:
                if state == interaction._final_states[2]:
                    utt3_correct_count += 1.
            if state is None:
                break
            intermediate_states.append(state)
            history.append(utterance)
        if state == interaction.get_final_state():
            correct_count += 1.0
        if logfile:
            logfile.write(
                '%s\t%s\n' %
                ('CORRECT' if state == interaction.get_final_state() else 'WRONG', interaction))
            logfile.write('Predicated actions: %s\n' %
                          (' / '.join([' '.join([readable_action(*tok) for tok in x]) for x in action_seqs])))
            logfile.write('Predicated states: %s\n' %
                          (' --> '.join([str(x) for x in intermediate_states])))
    accuracy = correct_count / count
    utt3_acc = utt3_correct_count / count
    if logfile:
        logfile.write('Accuracy: %.4f (exec)' % (accuracy))
        logfile.write('Utt3 Accuracy: " %.4f (exec) ' % (utt3_acc))
        logfile.close()
    print("utt3: " + str(utt3_acc))
    return accuracy

