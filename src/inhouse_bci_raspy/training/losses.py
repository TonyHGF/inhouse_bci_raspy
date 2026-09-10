"""GitClone loss implementations and configuration-to-criterion selection."""

import torch
import torch.nn as nn
import torch.nn.functional as F

def modified_cross_entropy(beta=0.5):

    '''

    this is a wrapper to allow specification of the

    regularization parameter beta

    '''

    def modified_cross_entropy_loss(y_pred, y_true):

        '''

        Special cross entropy loss that adds penalty to incorrect 

        confident predictions



        beta controls how much incorrect confident predictions get 

        penalized



        y_true is (N,)

        y_pred is (N, C)

        where N is the batch size and C are the number of classes

        '''

        ids = y_true.view(-1,1).long()

        numerator = torch.exp(y_pred.gather(-1, ids))        # (N,)

        denominator = torch.sum(torch.exp(y_pred), axis=-1).view(-1,1)  # (N,)



        # mean reduction is used by default for PyTorch Cross-Entropy Loss

        cross_entropy = torch.log(numerator / denominator)



        # confidence_penalty

        # indicates what samples are misclassified

        indicator = torch.argmax(y_pred, axis=-1) != y_true

        logits = torch.max(torch.exp(y_pred), axis=-1)[0]

        penalty = indicator * logits

        penalty = torch.sum(penalty) / torch.sum(indicator)

        #pdb.set_trace()



        loss = -torch.mean(cross_entropy) + beta*penalty

        return loss

    return modified_cross_entropy_loss

class OneHotMSE():

    '''Loss function class for mapping classes to one-hot vectors.'''

    def __init__(self, gamma=10.0, reduction='mean', weight=None):

        self.gamma_sq = gamma**2

        if weight is None:

            self.mseloss = nn.MSELoss(reduction=reduction)

            self.weight = weight

        else:

            self.weight = torch.FloatTensor(weight)

            self.mseloss = nn.MSELoss(reduction='none')

        return

    def __call__(self, inputs, targets):

        '''

        inputs: shape (N, D)

        targets: shape (N,) of integers with 0 <= value < D

        Truncates or expands the one-hot of targets if they do not match inputs!

        Returns (gamma**2) * (mse between inputs and one-hot targets)

        '''

        targets_vec = F.one_hot(targets, num_classes=inputs.shape[1]).float()

        if self.weight is None:

            out = self.gamma_sq*self.mseloss(inputs, targets_vec)

        else:

            mses = self.mseloss(inputs, targets_vec).mean(-1)

            sample_weight = self.weight.to(inputs.device)[targets]

            sample_weight = sample_weight/(sample_weight.mean())

            out = self.gamma_sq*(sample_weight*mses).mean()

        return out

def load_loss_criterion(loss_name='CEL', weight=None):

    '''Used to translate loss names into pyTorch criterion'''

    if loss_name == "CEL":

        # use cross entropy loss

        criterion = nn.CrossEntropyLoss()

    elif loss_name == "FL":

        # use focal loss

        pass

    elif loss_name == "smoothed":

        # use label smoothing

        criterion = nn.CrossEntropyLoss(label_smoothing=0.25)

    elif loss_name == "modified":

        # use modified cross entropy loss

        criterion = modified_cross_entropy()

    elif loss_name[:8] == "modified":

        criterion = modified_cross_entropy(beta=float(loss_name[8:]))

    elif loss_name == "NLL":

        criterion = nn.NLLLoss()

    elif loss_name == "OneHotMSE":

        criterion = OneHotMSE(weight=weight)

    else:

        raise ValueError(f'loss_name {loss_name} is not valid!')

    return criterion
