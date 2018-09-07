"""
Functions for supervised training on SCONE.
"""
import dynet as dy
import math
import random
import tqdm

from data import chunks
from evaluation import get_set_loss, utterance_accuracy, interaction_accuracy

def do_one_batch(batch, model, trainer, fsa_builder):
    """ Updates model with one batch of data.

    Inputs:
        batch (list of Example): The batch to use.
        model (SconeModel): Model that's being updated.
        trainer (dy.Trainer): Trainer that updates the model.
        fsa_builder (lambda x : ExecutionFSA): Execution FSA builder.

    Returns:
        float representing the loss computed during the batch.
    """
    dy.renew_cg()
    losses = []
    for example in batch:
        losses.extend(
            model.get_losses(
                example.utterance,
                example.actions,
                example.initial_state,
                example.history,
                fsa_builder(example.initial_state),
                True))
    loss = dy.esum(losses) / len(losses)
    loss.forward()
    loss.backward()
    trainer.update()
    
    return loss.value()

def do_one_epoch(model,
                 train_set,
                 val_set,
                 val_ints,
                 fsa_builder,
                 args,
                 epoch,
                 trainer):
    """ Performs one epoch of update to the model.

    Inputs:
        model (SconeModel): Model to update.
        train_set (list of Example): Examples to train on.
        val_set (list of Example): Examples to compute held out accuracy on.
        fsa_builder (lambda WorldState : ExecutionFSA): Creates an FSA from a
            world state.
        args (kwargs)

    Returns:
        float, the validation accuracy computed on the val_set after the epoch
    """
    epoch_loss = 0.0
    batches = chunks(train_set, args.batch_size)
    for batch in tqdm.tqdm(batches,
                           desc='Training epoch %d' % (epoch),
                           ncols=80):
        epoch_loss += do_one_batch(batch, model, trainer, fsa_builder)
    print('Epoch mean loss: %.4f' % (epoch_loss / len(batches)))
    # At the end of each epoch, run on the validation data.
    val_loss = get_set_loss(model, val_set, fsa_builder, args.batch_size)
    val_accuracy, _, val_token_accuracy = utterance_accuracy(
        model, val_set, fsa_builder, '%s/val-epoch%d.log' %
        (args.logdir, epoch))
    val_int_acc = interaction_accuracy(model, val_ints, fsa_builder,
                    '%s/val-int-epoch-%d.log' % (args.logdir, epoch))
    print(
        'Validation: loss=%.4f, accuracy=%.4f, int_acc=%.4f, token_acc=%.4f' %
        (val_loss, val_accuracy, val_int_acc, val_token_accuracy))
    return val_token_accuracy, val_int_acc


# Training procedure.
def training(model,
             train_set,
             val_set,
             val_ints,
             args,
             fsa_builder=None):
    """ Training function given a training and validation set.

    Inputs:
        train_set (list of examples): List of training examples.
        val_set (list of examples): List of validation examples.
        fsa_builder (ExecutableFSA): Builder for the FSA.
    """
    trainer = dy.RMSPropTrainer(model.get_params())
    trainer.set_clip_threshold(1)

    best_val_accuracy = 0.0
    best_token_accuracy = 0.0
    best_model = None
    patience = args.patience
    countdown = patience

    for epoch in range(args.max_epochs):
        token_acc, int_acc = do_one_epoch(model, train_set, val_set, val_ints, fsa_builder, args, epoch, trainer)

        # Save model.
        model_file_name = '%s/model-epoch%d.dy' % (args.logdir, epoch)
        model.save_params(model_file_name)

        # Stopping.
        if token_acc > best_token_accuracy:
            best_token_accuracy = token_acc
            patience *= 1.005
            countdown = patience
            print('Validation token accuracy increased to ' + str(token_acc))
            print('Countdown reset and patience set to %f' % (patience))
        else:
            countdown -= 1

        if int_acc > best_val_accuracy or best_model is None:
            best_model = model_file_name
            best_val_accuracy = int_acc
            print('Interaction accuracy increased to ' + str(int_acc))

        if countdown <= 0:
            print('Patience ran out -- stopping')
            break

    print('Loading parameters from best model: %s' % (best_model))
    model.load_params(best_model)

def train_and_evaluate(model,
                       train_set,
                       val_set,
                       val_interactions,
                       dev_set,
                       dev_interactions,
                       args,
                       fsa_builder=None):
    model.set_dropout(args.supervised_dropout)
    training_subset = train_set
    if args.supervised_ratio < 1.0 or args.supervised_amount > 0:
        train_ids = [example.id[:-1] for example in train_set]
        random.shuffle(train_ids)
        if args.supervised_ratio < 1.0:
            ids_to_use = train_ids[:int(len(train_ids) * args.supervised_ratio)]
        elif args.supervised_amount > 0:
            ids_to_use = train_ids[:args.supervised_amount]
        training_subset = [example for example in train_set if example.id[:-1] in ids_to_use]
    print("Training with " + str(len(training_subset)) + " examples")
    training(model,
             training_subset,
             val_set,
             val_interactions,
             args,
             fsa_builder=fsa_builder)
    model.save_params(args.logdir + "/supervised_model.dy")
    dev_accuracy, _, _ = utterance_accuracy(
        model,
        dev_set,
        fsa_builder=fsa_builder,
        logfile=args.logdir + "/dev_utterances.log")
    dev_interaction_accuracy = interaction_accuracy(
        model,
        dev_interactions,
        fsa_builder=fsa_builder,
        logfile=args.logdir + "/dev_interactions.log")
    print("Development accuracy: %.4f (single), %.4f (interaction)" %
          (dev_accuracy, dev_interaction_accuracy))
