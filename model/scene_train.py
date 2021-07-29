import json
import sys
import random
import os
from itertools import groupby
from enum import Enum
from args import interpret_args

import dynet_config
dynet_config.set(mem=2048,
                 autobatch="--dynet-autobatch" in sys.argv \
                    or "--dynet-autobatch=1" in sys.argv)
import dynet as dy
from data import load_data
from scene_state import SceneState
from model import ConstrainedContextSeq2SeqEmbeddings
from scene_fsa import COLORS, NUM_POSITIONS, SceneFSA
from scene_state_encoder import ContextDepStateEncoder
from supervised_learning import train_and_evaluate
from reinforcement_learning import reinforcement_learning, reward_with_shaping
from vocabulary import NO_ARG, EOS
from evaluation import utterance_accuracy, interaction_accuracy


SCENE_ACTION_DIST = {EOS: 0.2,
                     "appear_person": 0.2,
                     "appear_hat": 0.2,
                     "remove_person": 0.2,
                     "remove_hat": 0.2}

def valid_action_fn(action, location, argument):
    invalid = False
    if action == "appear_person" or action == "appear_hat":
        invalid = location == NO_ARG or argument == NO_ARG
    elif action == "remove_person" or action == "remove_hat":
        invalid = location == NO_ARG or argument != NO_ARG
    elif action == EOS:
        invalid = location != NO_ARG or argument != NO_ARG
    return not invalid

def evaluate(args, utterance_data, interaction_data, model):
    accuracy, _, _ = utterance_accuracy(
        model,
        utterance_data,
        fsa_builder=SceneFSA,
        logfile=args.results_file)
    print(accuracy)
    accuracy = interaction_accuracy(
        model,
        interaction_data,
        fsa_builder=SceneFSA,
        logfile=args.results_file + "_int.log")
    print(accuracy)

def main():
    args = interpret_args()
    train_data, dev_data, val_data, test_data, in_vocab, _ = \
        load_data('../scene/train_sequences.json',
                  '../scene/dev_sequences.json',
                  '../scene/test_sequences.json',
                  SceneState,
                  args,
                  sort=False)
    train, train_interactions = train_data
    dev, dev_interactions = dev_data
    val, val_interactions = val_data
    test, test_interactions = test_data

    model = ConstrainedContextSeq2SeqEmbeddings(in_vocab,
                                                (["appear_person", "appear_hat", "remove_person", "remove_hat"],
                                                 [str(x+1) for x in range(NUM_POSITIONS)],
                                                 list(COLORS)),
                                                ContextDepStateEncoder,
                                                valid_action_fn,
                                                args)
    train.sort(key=lambda x: (x.turn, len(x.utterance)))

    if args.evaluate:
        model.load_params(args.saved_model)
        split_to_eval = None
        interactions_to_eval = None
        if args.evaluate_split == "train":
            split_to_eval = train
            interactions_to_eval = train_interactions
        elif args.evaluate_split == "val":
            split_to_eval = val
            interactions_to_eval = val_interactions
        elif args.evaluate_split == "dev":
            split_to_eval = dev
            interactions_to_eval = dev_interactions
        elif args.evaluate_split == "test":
            split_to_eval = test
            interactions_to_eval = test_interactions
        else:
            raise ValueError("Unexpected split name " + args.evaluate_split)
        evaluate(args, split_to_eval, interactions_to_eval, model)
    if args.evaluate_attention:
        print("TODO: need to implement evaluate attention for scene")
    if args.supervised:
        train_and_evaluate(model,
                           train,
                           val,
                           dev,
                           dev_interactions,
                           args,
                           fsa_builder=SceneFSA)
    if args.rl:
        if args.pretrained:
            model.load_params(args.logdir + "/supervised_model.dy")
        model.set_dropout(args.rl_dropout)
        multihead_entropy = lambda x: dy.esum(list(get_entropy(x)))
        reinforcement_learning(model,
                               train,
                               val,
                               val_interactions,
                               args.logdir,
                               SceneFSA,
                               reward_with_shaping,
                               model.compute_entropy if args.syntax_restricted else multihead_entropy,
                               SCENE_ACTION_DIST,
                               args,
                               epochs=200,
                               batch_size=20,
                               single_head=False,
                               explore_with_fsa=False)


if __name__ == "__main__":
    main()

