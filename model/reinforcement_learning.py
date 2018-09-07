"""Contains methods for performing reinforcement learning on SCONE."""

import copy
import dynet as dy
from collections import namedtuple, defaultdict
from enum import Enum
from operator import itemgetter
import numpy
import os
import progressbar
import time
import tqdm
import random

from data import prepare_data, chunks
from evaluation import get_set_loss, utterance_accuracy, interaction_accuracy
from pycrayon import CrayonClient
from util import shaping
from model_util import readable_action
from vocabulary import EOS, NO_ARG, BEG, ACTION_SEP
from scone_logging import log_metrics

GAMMA = 0.5

REWARD_CORRECT = 1.0
REWARD_INCORRECT = -1.0
REWARD_OTHER = 0.


def distance_potential(state, final_state):
    """Distance potential function for a state and a final state.

    Inputs:
        state (AlchemyState): The current world state.
        final_state (AlchemyState): The goal state.

    Returns:
        int, representing the distance between the current and goal states.
    """
    return final_state.distance(state)


def reward_function(goal_state,
                    current_state,
                    action,
                    is_at_end,
                    verbosity_penalty):
    """ Reward function for non-supervised RL.

    Inputs:
        goal_state (AlchemyState): The goal state for the example.
        previous_state (AlchemyState): The previous state.
        current_state (AlchemyState): The current state.
        fsa (AlchemyFSA): The Alchemy FSA currently being updated.
        action (string): The action just taken.
        is_at_end (bool): Whether or not the current index is at the length limit.

    Returns:
        float, representing the reward for (s, a, s').
    """
    reward = REWARD_OTHER - verbosity_penalty
    if action == EOS or action == (EOS, NO_ARG, NO_ARG) and current_state:
        if current_state == goal_state:
            reward = REWARD_CORRECT
        else:
            reward = REWARD_INCORRECT
    elif is_at_end:
        reward = REWARD_INCORRECT
    elif not current_state:
        reward = REWARD_INCORRECT - verbosity_penalty
    return reward


def reward_with_shaping(goal_state,
                        previous_state,
                        current_state,
                        action,
                        is_at_end,
                        verbosity_penalty):
    """Returns the shaped reward for non-supervised RL.

    Inputs:
        goal_state (AlchemyState): The goal state for the example.
        previous_state (AlchemyState): The previous state.
        current_state (AlchemyState): The current state.
        fsa (AlchemyFSA): The Alchemy FSA currently being updated.
        action (string): The action just taken.
        is_at_end (bool): Whether or not the current index is at the length limit.

    Returns:
        float, representing the reward for (s, a, s').
    """
    problem_reward = reward_function(goal_state,
                                     current_state,
                                     action,
                                     is_at_end,
                                     verbosity_penalty)
    distance_term = shaping(previous_state,
                            current_state,
                            goal_state,
                            distance_potential,
                            REWARD_OTHER)

    return problem_reward + distance_term


class Prediction(namedtuple('Prediction',
                            ('example',
                             'predicted_sequence',
                             'token_probabilities',
                             'all_probabilities',
                             'rewards',
                             'entropies',
                             'distances',
                             'reward_expressions',
                             'final_fsa'))):
    """Contains all of the information about a sampled prediction.

    Attributes:
        example (Example): The original example.
        predicted_sequence (list of str): The predicted action sequence.
        token_probabilities (list of dy.Expression): Expressions representing the
            probability of the sampled token.
        all_probabilities (list of dy.Expression): Probability distributions for
            all timesteps.
        rewards (list of float): The rewards achieved at each timestep.
        entropies (list of dy.Expression): Expressions representing the probability
            distribution entropy at each timestep.
        distances (list of float): The distance from the goal state after each
            action.
        reward_expressions (list of dy.Expression): Actual reward expressions
            (log (p) * R) for each timestep.
        final_fsa (ExecutionFSA): The FSA representing the final state of the
            world after execution.
    """
    __slots__ = ()

    def __str__(self):
        string = ""
        string += str(self.example) + "\n"
        assert len(self.predicted_sequence) == len(self.token_probabilities)
        assert len(self.predicted_sequence) == len(self.rewards)
        assert len(self.predicted_sequence) == len(self.entropies)
        assert len(self.predicted_sequence) == len(self.distances)
        assert len(self.predicted_sequence) == len(self.reward_expressions)
        for token, prob, reward, entropy, distance, reward_exp in zip(self.predicted_sequence,
                                                                      self.token_probabilities,
                                                                      self.rewards,
                                                                      self.entropies,
                                                                      self.distances,
                                                                      self.reward_expressions):
            string += str(token) + "\t" \
                + str(prob.npvalue()) + "\t" \
                + str(reward) + "\t" \
                + str(distance) + "\t" \
                + str(reward_exp.value()) + "\t" \
                + str(entropy.npvalue()) + "\n"
        return string


class RLModes(Enum):
    """The possible RL types: Policy gradient and contextual bandit."""
    POLICY_GRADIENT = 1
    CONTEXTUAL_BANDIT = 2
    SUPERVISED_BASELINE = 3


def get_rl_mode(string_val):
    """ Gets the RL mode enum from a string.

    Inputs:
        string_val (str): The name of the mode.

    Returns:
        RLModes, the mode specified.

    Raises:
        ValueError, if the name is not recognized.
    """
    if string_val == "CONTEXTUAL_BANDIT":
        return RLModes.CONTEXTUAL_BANDIT
    if string_val == "POLICY_GRADIENT":
        return RLModes.POLICY_GRADIENT
    if string_val == "SUPERVISED_BASELINE":
        raise ValueError("Supervised baseline not supported anymore.")
    raise ValueError("RL Mode " + string_val + " not recognized.")


def get_rewards(final_state, predicted_sequence, fsa, reward_fn, args):
    """ Gets a sequence of rewards for some predictions.

    Inputs:
        example (Example): The original example for the data.
        predictions (list of tok): The actual predictions made.
        fsa_builder (lambda x : ExecutionFSA): Gets the FSA for the example.
        args (kwargs): Arguments for computing the reward.
    """
    previous_state = fsa.state()

    distances = []
    rewards = []

    stop_tok = (EOS if args.single_head else (EOS, NO_ARG, NO_ARG))

    for i, token in enumerate(predicted_sequence):
        state = fsa.state()
        if token != stop_tok:
            state = fsa.peek_complete_action(*token)
            if state:
                distances.append(final_state.distance(state))
                fsa.feed_complete_action(*token)
            else:
                distances.append(-1)
        else:
            distances.append(final_state.distance(state))

        rewards.append(reward_fn(final_state,
                                 previous_state,
                                 state,
                                 token,
                                 i == args.sample_length_limit - 1,
                                 args.verbosity_penalty))

        previous_state = fsa.state()

        if token == stop_tok:
            break

    return distances, rewards, fsa


def get_reward_expressions(token_probabilities, rewards, entropies, args):
    """ Computes the reward expressinos for the prediction.

    Inputs:
        precited_sequence (list of tok): The predicted sequence.
        token_probabilities (list of dy.Expression): Expressions representing the
            probability of selecting each token.
        rewards (list of float): Rewards for each token.
        entropies (list of dy.Expression): Entropy for each token.
        args (kwards): Arguments for computing the reward expressions.

    Returns:
        list of dy.Expression representing R(s,a)log(p(a)) for each action taken.
    """
    if len(token_probabilities) != len(rewards) or len(
            token_probabilities) != len(entropies):
        raise ValueError("Length of token probabilities, rewards, and entropies" +
                         " must be the same, but was the following: " +
                         " token probabilities: " +
                         str(len(token_probabilities)) +
                         ";  rewards: " +
                         str(len(rewards)) +
                         "; entropies: " +
                         str(len(entropies)))
    reward_expressions = []
    rl_mode = get_rl_mode(args.rl_mode)
    if rl_mode == RLModes.CONTEXTUAL_BANDIT:
        for token_prob, reward, entropy in zip(
                token_probabilities, rewards, entropies):
            # Contextual bandit: R(i) * p(i)
            entropy_add = 0.
            if args.entropy_coefficient > 0:
                entropy_add = args.entropy_coefficient * entropy.value()
            reward_expressions.append((reward + entropy_add)
                                      * dy.log(token_prob))
    elif rl_mode == RLModes.POLICY_GRADIENT:
        prev_reward = 0.
        for token_prob, reward, entropy in reversed(list(enumerate(zip(
                token_probabilities, rewards, entropies)))):
            # Starts with the /final/ probability first.
            timestep_reward = reward + GAMMA * prev_reward
            reward_expressions.append(timestep_reward * dy.log(token_prob))
            prev_reward = timestep_reward

    return reward_expressions


def get_sestra_rewards(sequence,
                       probability_distributions,
                       reward_fn,
                       model,
                       example,
                       args,
                       fsa_builder):
    """Performs SESTRA exploration.

    Inputs:
        sequence (list of tok): Predicted sequence.
        probability_distributions (list of dy.Expression): Probability distributions
            from the policy.
        reward_fn (lambda): Maps from (state/action/state) to a reward.
        example (Example): Original example with data.
        args (namespace): Arguments for training.
        fsa_builder (FSABuilder): For initializing the environment.
    """
    reward_expressions = []
    fsa = fsa_builder(example.initial_state)

    num_actions = len(model.output_action_vocabulary) - 1
    num_locations = len(model.output_location_vocabulary) - 1
    num_arguments = len(model.output_argument_vocabulary) - 1
    valid_mask = numpy.zeros((num_actions * num_locations * num_arguments))
    for index in model.valid_action_indices:
        valid_mask[index] = 1.0
    valid_mask = dy.inputTensor(valid_mask)

    for i, (token, distribution) in enumerate(
            zip(sequence, probability_distributions)):
        # Try feeding all possible actions, and get a reward for each.
        prev_state = fsa.state()
        total_num_actions = len(model.all_output_vocabulary)
        action_rewards = numpy.zeros((total_num_actions,))
        index = 0

        indices = model.valid_action_indices[:]
        random.shuffle(indices)
        if args.sample_ratio < 1:
            num_samples = int(len(indices) * args.sample_ratio)
            indices = indices[:num_samples]
        sampled_action_index = model.all_output_vocabulary.lookup_index(
            tuple(token))
        if sampled_action_index not in indices:
            indices.append(sampled_action_index)

        for action in model.output_action_vocabulary:
            for location in model.output_location_vocabulary:
                for argument in model.output_argument_vocabulary:
                    if action != BEG and location != BEG and argument != BEG:
                        if model.all_output_vocabulary.lookup_index(
                                (action, location, argument)) in indices:
                            if (action, location, argument) != (
                                    EOS, NO_ARG, NO_ARG):
                                peek_state = fsa.peek_complete_action(
                                    action, location, argument)
                            else:
                                peek_state = fsa.state()
                            reward = reward_fn(
                                example.final_state,
                                prev_state,
                                peek_state,
                                action,
                                i == args.sample_length_limit,
                                args.verbosity_penalty)
                            action_rewards[index] = reward
                        index += 1
        valid_probs = dy.cmult(valid_mask, distribution)
        invalid_probs = 1. - valid_mask
        masked_dist = valid_probs + invalid_probs
        reward_expressions.append(
            dy.sum_elems(
                dy.cmult(
                    dy.inputTensor(action_rewards),
                    masked_dist)))

        # Then feed the sampled token.
        if token != (EOS, NO_ARG, NO_ARG):
            peek_state = fsa.peek_complete_action(*token)
            if peek_state:
                fsa.feed_complete_action(*token)
    return reward_expressions


def process_example(example,
                    predictions,
                    probability_distributions,
                    reward_fn,
                    entropy_function,
                    model,
                    args,
                    fsa_builder):
    """ Creates a Prediction for an example. """
    predicted_sequence = [prediction[0] for prediction in predictions]
    token_probabilities = [prediction[1] for prediction in predictions]
    entropies = [entropy_function(distribution)
                 for distribution in probability_distributions]

    fsa = fsa_builder(example.initial_state)
    distances, rewards, fsa = get_rewards(example.final_state,
                                          predicted_sequence,
                                          fsa,
                                          reward_fn,
                                          args)

    if args.use_sestra:
        reward_expressions = get_sestra_rewards(
            predicted_sequence,
            probability_distributions,
            reward_fn,
            model,
            example,
            args,
            fsa_builder)
    else:
        reward_expressions = get_reward_expressions(token_probabilities,
                                                    rewards,
                                                    entropies,
                                                    args)

    return Prediction(example,
                      predicted_sequence,
                      token_probabilities,
                      probability_distributions,
                      rewards,
                      entropies,
                      distances,
                      reward_expressions,
                      fsa)


def compute_metrics(predictions,
                    divisor,
                    metrics,
                    args,
                    model=None):
    """ Computes metrics for a set of predictions.

    Inputs:
        predictions (list of Prediction): Predictions.
        divisor (float): Number to divide the results (for averages).
        metrics (list of str): The metrics to compute.

    Returns:
        dict (str to float) representing the computed metrics.

    Raises:
        ValueError if a metric name is not recognized.
    """
    computed_metrics = {}

    stop_tok = (EOS if args.single_head else (EOS, NO_ARG, NO_ARG))
    for metric in metrics:
        if metric == "entropy":
            computed_metrics[metric] = dy.esum(
                [dy.esum(prediction.entropies) for prediction in predictions]).value() / divisor
        elif metric == "reward":
            computed_metrics[metric] = sum(
                [sum(prediction.rewards) for prediction in predictions]) / divisor
        elif metric == "gold_probability":
            prob_sum = dy.zeros((1,))
            if not stop_tok:
                raise ValueError(
                    "Stop token should not be none if computing " +
                    str(metric))
            for prediction in predictions:
                gold_actions = prediction.example.actions + [stop_tok]
                if not args.single_head:
                    gold_actions = model.group_tokens(gold_actions)
                for i, token in enumerate(gold_actions):
                    if i < len(prediction.predicted_sequence):
                        prob_sum += model.probability_of_token(
                            token,
                            prediction.all_probabilities[i])
            computed_metrics[metric] = prob_sum.npvalue()[0] / divisor
        elif metric == "distance":
            distance = 0.
            for prediction in predictions:
                gold_state = prediction.example.final_state
                initial_state = prediction.example.initial_state
                original_distance = gold_state.distance(initial_state)
                if prediction.final_fsa.state():
                    final_distance = gold_state.distance(
                        prediction.final_fsa.state())
                else:
                    final_distance = original_distance
                distance += final_distance / original_distance
            computed_metrics[metric] = distance / divisor
        elif metric == "invalid":
            num_invalid = 0.
            if not stop_tok:
                raise ValueError(
                    "Stop token should not be none if computing " +
                    str(metric))
            for prediction in predictions:
                if not (prediction.final_fsa.is_valid()
                        and prediction.predicted_sequence[-1] == stop_tok):
                    num_invalid += 1
            computed_metrics[metric] = num_invalid / divisor
        elif metric == "num_tokens":
            computed_metrics[metric] = sum([len(prediction.predicted_sequence)
                                            for prediction in predictions]) / divisor
        elif metric == "prefix_length":
            prefix_sum = 0.
            if not stop_tok:
                raise ValueError(
                    "Stop token should not be none if computing " +
                    str(metric))
            if not model:
                raise ValueError(
                    "Model should not be none if computing " +
                    str(metric))
            for prediction in predictions:
                gold_actions = prediction.example.actions
                if not args.single_head:
                    gold_actions = model.group_tokens(gold_actions)
                for pred_token, gold_token in zip(
                        prediction.predicted_sequence, gold_actions):
                    if pred_token == tuple(gold_token):
                        prefix_sum += 1
                if prediction.predicted_sequence[-1] == stop_tok:
                    prefix_sum += 1
            computed_metrics[metric] = prefix_sum / divisor
        elif metric == "completion":
            completion_sum = 0.
            if not stop_tok:
                raise ValueError(
                    "Stop token should not be none if computing " +
                    str(metric))
            for prediction in predictions:
                if prediction.final_fsa.state() == prediction.example.final_state \
                        and prediction.predicted_sequence[-1] == stop_tok:
                    completion_sum += 1
            computed_metrics[metric] = completion_sum / divisor
        else:
            raise ValueError("Unrecognized metric name " + metric)

    return computed_metrics


def reinforcement_learning(model,
                           train_set,
                           val_set,
                           val_interactions,
                           log_dir,
                           fsa_builder,
                           reward_function,
                           entropy_fn,
                           args,
                           batch_size=1,
                           epochs=20,
                           single_head=True,
                           explore_with_fsa=False):
    """Performs training with exploration.

    Inputs:
        model (Model): Model to train.
        train_set (list of Examples): The set of training examples.
        val_set (list of Examples): The set of validation examples.
        val_interactions (list of Interactions): Full interactions for validation.
        log_dir (str): Location to log.

    """
    trainer = dy.RMSPropTrainer(model.get_params())
    trainer.set_clip_threshold(1)

    mode = get_rl_mode(args.rl_mode)

    best_val_accuracy = 0.0
    best_val_reward = -float('inf')
    best_model = None

    crayon = CrayonClient(hostname="localhost")
    experiment = crayon.create_experiment(log_dir)

    num_batches = 0
    train_file = open(os.path.join(log_dir, "train.log"), "w")

    patience = args.patience
    countdown = patience

    for epoch in range(epochs):
        random.shuffle(train_set)
        batches = chunks(train_set, batch_size)

        num_examples = 0
        num_tokens = 0
        num_tokens_zero = 0
        progbar = progressbar.ProgressBar(
            maxval=len(batches),
            widgets=[
                "Epoch " + str(epoch),
                progressbar.Bar(
                    '=',
                    '[',
                    ']'),
                ' ',
                progressbar.Percentage(),
                ' ',
                progressbar.ETA()])
        progbar.start()
        for i, batch in enumerate(batches):
            dy.renew_cg()

            prob_seqs, predictions = model.sample_sequences(
                batch,
                length=args.sample_length_limit,
                training=True,
                fsa_builder=fsa_builder)

            batch_entropy_sum = dy.inputTensor([0.])
            batch_rewards = []
            processed_predictions = []

            train_file.write("--- NEW BATCH # " + str(num_batches) + " ---\n")
            action_probabilities = {}
            for action in model._output_action_vocabulary:
                if action != BEG:
                    action_probabilities[action] = []

            for example, prob_seq, prediction in zip(
                    batch, prob_seqs, predictions):
                # Get reward (and other evaluation information)
                prediction = process_example(example,
                                             prediction,
                                             prob_seq,
                                             reward_fn,
                                             entropy_function,
                                             model,
                                             args,
                                             fsa_builder)
                for distribution in prob_seq:
                    action_probability = model.action_probabilities(
                        distribution)
                    for action, prob_exp in action_probability.items():
                        action_probabilities[action].append(prob_exp)

                batch_rewards.extend(prediction.reward_expressions)
                batch_entropy_sum += dy.esum(prediction.entropies)
                processed_predictions.append(prediction)

                num_examples += 1

            # Now backpropagate given these rewards
            batch_action_probabilities = {}
            for action, prob_exps in action_probabilities.items():
                batch_action_probabilities[action] = dy.esum(
                    prob_exps) / len(batch_rewards)

            num_reward_exps = len(batch_rewards)
            loss = dy.esum(batch_rewards)
            if args.entropy_coefficient > 0:
                loss += args.entropy_coefficient * batch_entropy_sum
            loss = -loss / num_reward_exps
            loss.backward()
            try:
                trainer.update()
            except RuntimeError as r:
                print(loss.npvalue())
                for lookup_param in model._pc.lookup_parameters_list():
                    print(lookup_param.name())
                    print(lookup_param.grad_as_array())
                for param in model._pc.parameters_list():
                    print(param.name())
                    print(param.grad_as_array())
                print(r)
                exit()

            # Calculate metrics
            stop_tok = (EOS if single_head else (EOS, NO_ARG, NO_ARG))
            per_token_metrics = compute_metrics(processed_predictions,
                                                num_reward_exps,
                                                ["entropy",
                                                 "reward"],
                                                args)
            gold_token_metrics = compute_metrics(
                processed_predictions,
                sum([len(ex.actions) for ex in batch]) + len(batch),
                ["gold_probability"],
                args,
                model=model)
            per_example_metrics = compute_metrics(processed_predictions,
                                                  len(batch),
                                                  ["distance",
                                                   "completion",
                                                   "invalid",
                                                   "num_tokens",
                                                   "prefix_length"],
                                                  args,
                                                  model=model)

            for prediction in processed_predictions:
                train_file.write(str(prediction) + "\n")
            train_file.write("=====\n")
            log_metrics({"loss": loss.npvalue()[0]},
                        train_file, experiment, num_batches)
            log_metrics(per_token_metrics, train_file, experiment, num_batches)
            log_metrics(
                gold_token_metrics,
                train_file,
                experiment,
                num_batches)
            log_metrics(
                per_example_metrics,
                train_file,
                experiment,
                num_batches)
            train_file.flush()

            num_batches += 1
            progbar.update(i)

        progbar.finish()
        train_acc, _, _ = utterance_accuracy(model,
                                             train_set,
                                             fsa_builder=fsa_builder,
                                             logfile=log_dir + "/rl-train" + str(epoch) + ".log")
        val_acc, val_reward, _ = utterance_accuracy(
            model,
            val_set,
            fsa_builder=fsa_builder,
            logfile=log_dir + "/rl-val-" + str(epoch) + ".log",
            args=args,
            reward_fn=reward_fn)

        val_int_acc = interaction_accuracy(
            model,
            val_interactions,
            fsa_builder=fsa_builder,
            logfile=log_dir +
            "/rl-val-int-" +
            str(epoch) +
            ".log")

        log_metrics({"train_accuracy": train_acc,
                     "validation_accuracy": val_acc,
                     "validation_int_acc": val_int_acc,
                     "validation_reward": val_reward,
                     "countdown": countdown},
                    train_file,
                    experiment,
                    num_batches)
        experiment.to_zip(
            os.path.join(
                log_dir,
                "crayon-" +
                str(epoch) +
                ".zip"))
        model_file_name = log_dir + "/model-rl-epoch" + str(epoch) + ".dy"
        model.save_params(model_file_name)
        if val_int_acc > best_val_accuracy or best_model is None:
            best_model = model_file_name
            best_val_accuracy = val_int_acc

        if val_reward > best_val_reward:
            patience *= 1.005
            countdown = patience
            best_val_reward = val_reward
        else:
            countdown -= 1
        if countdown <= 0:
            print("Patience ran out -- stopping")
            break
    train_file.close()
    print('Loading parameters from best model: %s' % (best_model))
    model.load_params(best_model)
    model.save_params(log_dir + "/best_rl_model.dy")
    print(train_set[0])
    print(
        model.generate(
            train_set[0].utterance,
            train_set[0].initial_state,
            train_set[0].history)[0])
