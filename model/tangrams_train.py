"""Script for training on the tangrams domain.

Attributes:
    RNNBuilder (dy._RNNBuilder): The kind of RNN to use.
"""

import sys
import os

import dynet_config
dynet_config.set(mem=2048,
                 autobatch="--dynet-autobatch" in sys.argv \
                    or "--dynet-autobatch=1" in sys.argv)
import dynet as dy

import argparse
from data import load_data
from tangrams_fsa import TangramsFSA, NUM_POSITIONS, SHAPES
from tangrams_state import TangramsState
from tangrams_state_encoder import TangramsStateEncoder
from evaluation import utterance_accuracy, interaction_accuracy, attention_analysis
from constrained_model import ConstrainedContextSeq2SeqEmbeddings
from reinforcement_learning import reinforcement_learning, reward_with_shaping
from supervised_learning import train_and_evaluate
from train import interpret_args
from vocabulary import EOS, NO_ARG
from util import shaping


# Tangrams-specific methods.
TANGRAMS_ACTION_DIST = {EOS: 0.3, "remove": 0.35, "insert": 0.35}

def valid_action_fn(action, location, argument):
    invalid = False
    if action == "insert":
        invalid = location == NO_ARG or argument == NO_ARG
    elif action == "remove":
        invalid = location == NO_ARG or argument != NO_ARG
    elif action == EOS:
        invalid = location != NO_ARG or argument != NO_ARG
    return not invalid

def evaluate(args, utterance_data, interaction_data, model):
    accuracy, _, _ = utterance_accuracy(
        model,
        utterance_data,
        fsa_builder=TangramsFSA,
        logfile=args.results_file)
    print(accuracy)
    accuracy = interaction_accuracy(
        model,
        interaction_data,
        fsa_builder=TangramsFSA)
    print(accuracy)

def main():
    """Actually does the training."""
    args = interpret_args()
    train_data, dev_data, val_data, test_data, in_vocab, _ = \
        load_data('../tangram/train_sequences.json',
                  '../tangram/dev_sequences.json',
                  '../tangram/test_sequences.json',
                  TangramsState,
                  args,
                  sort=False)
    train, train_interactions = train_data
    dev, dev_interactions = dev_data
    val, val_interactions = val_data
    test, test_interactions = test_data

    model = ConstrainedContextSeq2SeqEmbeddings(in_vocab,
                                                (["insert", "remove"],
                                                 [str(x+1) for x in range(NUM_POSITIONS)],
                                                 list(SHAPES)),
                                                TangramsStateEncoder,
                                                valid_action_fn,
                                                args)

    train.sort(key=lambda x: (x.turn, len(x.utterance)))

    # Train.
    # Using development set for validation -- currently only to pick the best
    # model.
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
        model.load_params(args.saved_model)
        for example in dev:
            attention_analysis(model, example, TangramsFSA, name = args.logdir + "/attention/" + example.id)
    if args.supervised:
        train_and_evaluate(model,
                           train,
                           val,
                           val_interactions,
                           dev,
                           dev_interactions,
                           args,
                           fsa_builder=TangramsFSA)
    if args.rl:
        if args.pretrained:
            model.load_params(args.logdir + "/supervised_model.dy")

        model.set_dropout(args.rl_dropout)
        reinforcement_learning(model,
                               train,
                               val,
                               val_interactions,
                               args.logdir,
                               TangramsFSA,
                               reward_with_shaping,
                               model.compute_entropy,
                               TANGRAMS_ACTION_DIST,
                               args,
                               epochs=200,
                               batch_size=20,
                               single_head=False,
                               explore_with_fsa=False)

        # Save final model.
        model.save_params(args.logdir + '/model-final.dy')

    # Test results.
    if test:
        test_accuracy, _, _ = utterance_accuracy(
            model, test, TangramsFSA, args.logdir + '/test.log')
        test_interaction_accuracy = interaction_accuracy(
            model, test_interactions, TangramsFSA, args.logdir + '/test.interactions.log')
        print('Test accuracy: %.4f (single), %.4f (interaction)' %
              (test_accuracy, test_interaction_accuracy))


if __name__ == "__main__":
    main()
