"""GitClone optimization loop and its validation pass; preserves the original protocol."""

import torch

import torch.optim as optim

import numpy as np

import torch.nn as nn

import torch.nn.functional as F

import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau

import tqdm

import copy
from .losses import load_loss_criterion


# i am deprecating this in favor of a function argument. -jl

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_ds, val_ds, config, writer, device, verbosity=1):

    

    """Optimize one model with the original GitClone training and early-stopping rules.
    
    Args:
        model: Fresh EEGNet instance already on device.
        train_ds: Training DataLoader including augmented windows.
        val_ds: Validation DataLoader containing original windows only.
        config: Original training settings (loss, learning rate, maximum epochs).
        writer: Object supporting add_scalar; typically a TensorBoard SummaryWriter.
        device: torch device for labels and model execution.
        verbosity: Positive to print batch/epoch progress.
    Returns:
        Best model, accuracy history, zero-based best epoch, train loss history,
        validation loss history, and per-epoch predicted/true validation labels.
    
    Scheduler uses validation accuracy; best model and early stopping use loss."""
    min_val_loss = np.inf   # will be used for early stopping

    epochs_no_improve = 0   # will be used for early stopping

    best_model_index = 0 # will be used to find train and val acc of best model

    train_losses_epochs = []

    val_losses_epochs = []

    

    # use label distribution for class_weights

    class_weight = None

    if 'class_weight' in config:

        if config['loss_func'] != 'OneHotMSE':

            raise NotImplementedError('class_weight only implemented for OneHotMSE')

        if config['class_weight'] == 'balanced':

            labels_all = np.concatenate([labels.detach().cpu().numpy() for i, (inputs, labels) in enumerate(train_ds)])

            labels_count = np.unique(labels_all, return_counts=True)[1]

            class_weight = torch.FloatTensor(labels_count.mean()/labels_count)

            print('labels_count:', labels_count)

            print('class_weight:', class_weight)

        else:

            class_weight = torch.FloatTensor(config['class_weight'])

            


    #Get loss criterion

    criterion = load_loss_criterion(config['loss_func'], weight=class_weight)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])  

    scheduler = ReduceLROnPlateau(optimizer, 'max')  # default patience is 10 and factor is 0.1

    


    train_accuracy = []

    val_accuracy = []

    pred_labels, true_labels = [], []

    for epoch in range(config['max_epochs']):

        train_losses = []

        num_correct = 0

        num_samples = 0

        model.train()

        if verbosity > 0:

            ds_iterator = tqdm.tqdm(train_ds)

        else:

            ds_iterator = train_ds

        for i, (inputs, labels) in enumerate(ds_iterator):


            labels = labels.to(device)

            

            #for input_i in inputs:

            #    if not input_i.detach().cpu().numpy().any():

            #        raise Exception('zero input')

            #    else:

            #        print(input_i.shape, input_i.detach().cpu().numpy().sum())


            # zero the parameter gradients

            optimizer.zero_grad()

            outputs = model(inputs)

            try:

                loss = criterion(outputs, labels)

            except Exception as e:

                print(outputs)

                print(labels)

                print('Exception:', e)

            writer.add_scalar('Training Loss', loss.item(), epoch)

            train_losses.append(loss.item())

            loss.backward()

            optimizer.step()


            _, softmax_indices = outputs.max(1)

            num_correct += (softmax_indices == labels).sum()

            num_samples += softmax_indices.size(0)


        accuracy = 100 * num_correct / num_samples

        if verbosity > 0:

            print("epoch #", epoch, " training accuracy (%):", accuracy.item())

        train_accuracy.append(accuracy.item())

        train_losses_epochs.append(sum(train_losses) / len(train_losses))


        # test model on validation data

        val_loss, correct_list, e_pred_labels, e_true_labels = test(model,

                                      val_ds,

                                      criterion=criterion)

        pred_labels.append(e_pred_labels)

        true_labels.append(e_true_labels)

        

        writer.add_scalar('Validation Loss', val_loss, epoch)

        val_losses_epochs.append(val_loss.item() / len(val_ds))


        num_correct = sum(correct_list)

        num_samples = len(correct_list)

        accuracy = 100 * num_correct / num_samples

        if verbosity > 0:

            print("epoch #", epoch, " validation accuracy (%):", accuracy)

        scheduler.step(accuracy)

        val_accuracy.append(accuracy)


        writer.add_scalar('validation loss', val_loss, epoch * len(train_ds))


        # check for early stopping

        if val_loss < min_val_loss:

            epochs_no_improve = 0

            min_val_loss = val_loss

            best_model = copy.deepcopy(model)

            best_model_index = epoch

        else:

            epochs_no_improve += 1

        if epoch > 5 and epochs_no_improve == 10:

            if verbosity > 0:

                print("Training terminated early!")

            break

        

    return (best_model, 

           {'acc' : train_accuracy, 'val_acc' : val_accuracy},

           best_model_index,

           train_losses_epochs,

           val_losses_epochs,

           pred_labels,

           true_labels)


def test(model, val_ds, criterion):

    """Evaluate all loader batches without gradients.
    
    Returns accumulated criterion loss, per-window correctness, predicted labels
    and true labels. This is the original engine validation pass, not an independent
    test-set protocol; final metadata-aligned inference lives in evaluation.py."""

    model.eval()

    for p in model.parameters():

        device = p.device

        break

    

    pred_labels, true_labels = [], []

    

    with torch.no_grad():

        correct_list = []

        val_loss = 0

        for x, y in val_ds:

            x = x.to(device)

            y = y.to(device)


            scores = model(x)

            val_loss += criterion(scores, y)


            _, softmax_indices = scores.max(1)

            correct_list = correct_list + (softmax_indices == y).tolist()


            pred_labels.extend(softmax_indices)

            true_labels.extend(y)


    pred_labels = [pred_label.item() for pred_label in pred_labels]

    true_labels = [true_label.item() for true_label in true_labels]


    return val_loss, correct_list, pred_labels, true_labels


def collect_outputs(model, dset, criterion, return_numpy=False):

    '''This function collects the outputs and labels of dsets.'''

    model.eval()

    #out = [[] for dset in dsets]

    for p in model.parameters():

        device = p.device

        break

    logits_collect = []

    hidden_states_collect = []

    labels_collect = []

    loss_collect = []

    with torch.no_grad():

        for x, y in dset:

            x = x.to(device)

            y = y.to(device)


            out = model(x, return_logits=False, return_dataclass=True) # Misnomer. Instead outputs tuple (probs, logits, hidden_state)

            out_i = {'probs': out[0], 'logits': out[1], 'hidden_state': out[2]}

            loss_i = criterion(out_i['logits'], y)


            if return_numpy:

                logits_collect.extend(out_i['hidden_state'].detach().cpu().numpy())

                hidden_states_collect.extend(out_i['hidden_state'].detach().cpu().numpy())

                labels_collect.extend(y.detach().cpu().numpy())

                try:

                    loss_collect.extend(loss_i.detach().cpu().numpy())

                except:

                    loss_collect.append(loss_i.item())

            else:

                logits_collect.extend(out_i['hidden_state'].detach())

                hidden_states_collect.extend(out_i['hidden_state'].detach())

                labels_collect.extend(y.detach())

                try:

                    loss_collect.extend(loss_i)

                except:

                    loss_collect.append(loss_i.item())

        pass

    print(len(hidden_states_collect), len(labels_collect))

    return logits_collect, hidden_states_collect, labels_collect, loss_collect

