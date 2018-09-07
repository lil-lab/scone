"""Script for training on the alchemy domain.

Attributes:
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
from alchemy_fsa import AlchemyFSA
from alchemy_state import AlchemyState
from alchemy_state_encoder import AlchemyStateEncoder
from evaluation import utterance_accuracy, interaction_accuracy, attention_analysis
from model import ConstrainedContextSeq2SeqEmbeddings
from reinforcement_learning import reinforcement_learning, reward_with_shaping
from supervised_learning import train_and_evaluate
from args import interpret_args
from vocabulary import EOS, NO_ARG
from util import shaping


LOG_DIR = 'logs-alchemy'

# Alchemy-specific methods.
COLORS = ['y', 'o', 'r', 'g', 'b', 'p']
ACTION_POP = 'pop'
ACTION_PUSH = 'push'
ACTIONS = [ACTION_POP, ACTION_PUSH]

NUM_BEAKERS = 7
BEAKER_EOS = "<EOS>"
COLORS_VOCAB = COLORS + [BEAKER_EOS]

def valid_action_fn(action, location, argument):
    invalid = False
    if action == "push":
        invalid = location == NO_ARG or argument == NO_ARG
    elif action == "pop":
        invalid = location == NO_ARG or argument != NO_ARG
    elif action == EOS:
        invalid = location != NO_ARG or argument != NO_ARG
    return not invalid

def evaluate(args, utterance_data, interaction_data, model):
    accuracy, _, _ = utterance_accuracy(
        model,
        utterance_data,
        fsa_builder=AlchemyFSA,
        logfile=args.results_file + "_utt.log")
    print(accuracy)
    accuracy = interaction_accuracy(
        model,
        interaction_data,
        fsa_builder=AlchemyFSA,
        logfile=args.results_file + "_int.log")
    print(accuracy)

def main():
    """Actually does the training."""
    args = interpret_args()
    train_data, dev_data, val_data, test_data, in_vocab, _ = \
        load_data('../data/alchemy/train_sequences.json',
                  '../data/alchemy/dev_sequences.json',
                  '../data/alchemy/test_sequences.json',
                  AlchemyState,
                  args)
    train, train_interactions = train_data
    dev, dev_interactions = dev_data
    val, val_interactions = val_data
    test, test_interactions = test_data

    model = ConstrainedContextSeq2SeqEmbeddings(in_vocab,
                                                (["push", "pop"],
                                                 [str(x+1) for x in range(NUM_BEAKERS)],
                                                 list(COLORS)),
                                                AlchemyStateEncoder,
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
            attention_analysis(model, example, AlchemyFSA, name = args.logdir + "/attention/" + example.id)
    if args.supervised:
        train_and_evaluate(model,
                           train,
                           val,
                           val_interactions,
                           dev,
                           dev_interactions,
                           args,
                           fsa_builder=AlchemyFSA)
    if args.rl:
        if args.pretrained:
            model.load_params(args.logdir + "/supervised_model.dy")
#            train_acc = utterance_accuracy(model,
#                                           train,
#                                           fsa_builder=AlchemyFSA,
#                                           syntax_restricted=args.syntax_restricted,
#                                           logfile=args.logdir + "supervised_train.log")
#            print('Training accuracy: ' + str(train_acc) + " (single)")

        model.set_dropout(args.rl_dropout)
        reinforcement_learning(model,
                               train,
                               val,
                               val_interactions,
                               args.logdir,
                               AlchemyFSA,
                               reward_with_shaping,
                               model.compute_entropy,
                               args,
                               epochs=args.max_epochs,
                               batch_size=20,
                               single_head=False,
                               explore_with_fsa=False)

        # Save final model.
        model.save_params(args.logdir + '/model-final.dy')

    # Test results.
    if test:
        test_accuracy, _, _ = utterance_accuracy(
            model, test, AlchemyFSA, args.logdir + '/test.log')
        test_interaction_accuracy = interaction_accuracy(
            model, test_interactions, AlchemyFSA, args.logdir + '/test.interactions.log')
        print('Test accuracy: %.4f (single), %.4f (interaction)' %
              (test_accuracy, test_interaction_accuracy))


if __name__ == "__main__":
    main()
