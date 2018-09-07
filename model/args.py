import argparse
import os

DROPOUT_AMOUNT = 0.5
WORD_EMBEDDING_SIZE = 50
ENCODER_STATE_SIZE = 100
DECODER_STATE_SIZE = 100

LEN_LIMIT = 10

def interpret_args():
    """Interprets the command line arguments.

    Returns:
        string, representing the logging directory.
    """
    parser = argparse.ArgumentParser()

    ### Logging
    parser.add_argument("--logdir", default="logs-alchemy")
    parser.add_argument("--dynet-autobatch", type=bool, default=True)

    ### Data
    parser.add_argument("--supervised_ratio", type=float, default=1.0)
    parser.add_argument("--validation_ratio", type=float, default=0.07)
    parser.add_argument("--supervised_amount", type=int, default=0)
    parser.add_argument("--enable_testing", type=bool, default=False)

    ### Training
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=400)
    parser.add_argument("--patience", type=float, default=50.)

    # Supervised
    parser.add_argument("--supervised", type=bool, default=False)

    # RL
    parser.add_argument("--entropy_coefficient", type=float, default=0.)
    parser.add_argument("--rl", type=bool, default=False)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--rl_mode", default="CONTEXTUAL_BANDIT")
    parser.add_argument("--sample_length_limit", type=int, default=LEN_LIMIT)
    parser.add_argument("--supervised_reward", type=bool, default=False)
    parser.add_argument("--verbosity_penalty", type=float, default=0.)
    parser.add_argument("--policy_shaping", type=bool, default=False)
    parser.add_argument("--use_sestra", type=bool, default=False)
    parser.add_argument("--sample_ratio", type=float, default=1.0)

    ### Evaluation
    parser.add_argument("--saved_model", default="")
    parser.add_argument("--results_file", default="")
    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--evaluate_split", default="val")
    parser.add_argument("--evaluate_attention", type=bool, default=False)
    parser.add_argument("--fsa_restricted_evaluation", type=bool, default=False)

    ### Architecture
    # Interaction with model
    parser.add_argument("--single_head", type=bool, default=False)
    parser.add_argument("--syntax_restricted", type=bool, default=True)
    parser.add_argument("--fsa_restricted", type=bool, default=False)
    parser.add_argument("--feed_updated_state", type=bool, default=True)
    parser.add_argument("--always_initial_state", type=bool, default=True)

    # Dropout, other minor details
    parser.add_argument("--supervised_dropout", type=float, default=DROPOUT_AMOUNT)
    parser.add_argument("--rl_dropout", type=float, default=0.)
    parser.add_argument("--final_nonlinearity", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    # Sizes of things
    parser.add_argument("--num_enc_layers", type=int, default=1)
    parser.add_argument("--num_dec_layers", type=int, default=1)
    parser.add_argument("--embeddings_size", type=int, default=WORD_EMBEDDING_SIZE)
    parser.add_argument("--encoder_size", type=int, default=ENCODER_STATE_SIZE)
    parser.add_argument("--decoder_size", type=int, default=DECODER_STATE_SIZE)

    args = parser.parse_args()

    # Out directory
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # Making sure things are consistent
    if not args.supervised and not args.rl and not args.evaluate and not args.evaluate_attention:
        raise ValueError("You need to be supervised, RL, or evaluation")
    if args.pretrained and not args.rl:
        raise ValueError("If pretraining, you need to then do RL.")
    if args.evaluate and not (args.saved_model and args.results_file):
        raise ValueError("If evaluating, you need to provide a trained model file.")
    if args.evaluate_attention and not args.saved_model:
        raise ValueError("If evaluating attention, you need to provied a trained model file.")

    # Warnings for not implemented options
    if args.fsa_restricted:
        raise ValueError("Restricting with FSA is not implemented")
    if args.supervised_amount > 0 and args.supervised_ratio < 1.0:
        raise ValueError("At most one of supervised ratio and amount can be specified " \
                         + "ratio: " + str(args.supervised_ratio) + "; " \
                         + "amount: " + str(args.supervised_amount))
    if args.always_initial_state and not args.feed_updated_state:
        raise ValueError("Setting always initial state should only be done " \
                         + " when feeding the updated state")
    if args.single_head:
        raise ValueError("Need to implement code to do single-head prediction again")
    if args.supervised_reward:
        raise ValueError("Need to reimplement supervised reward code")

    if args.rl or args.supervised:
        args_file = args.logdir + "/args.log"
        if os.path.exists(args_file):
            print("Warning: arguments already exist in " + str(args_file))
            exit()
        with open(args_file, "w") as infile:
            infile.write(str(args))

    return args

