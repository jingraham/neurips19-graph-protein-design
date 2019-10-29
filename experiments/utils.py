from __future__ import print_function
import json, time, os, sys

from matplotlib import pyplot as plt
plt.switch_backend('agg')

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

# Library code
sys.path.insert(0, '..')
from struct2seq import *

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='Structure to sequence modeling')
    parser.add_argument('--hidden', type=int, default=128, help='number of hidden dimensions')
    parser.add_argument('--k_neighbors', type=int, default=30, help='Neighborhood size for k-NN')
    parser.add_argument('--vocab_size', type=int, default=20, help='Alphabet size')
    parser.add_argument('--features', type=str, default='full', help='Protein graph features')
    parser.add_argument('--model_type', type=str, default='structure', help='Enrich with alignments')
    parser.add_argument('--mpnn', action='store_true', help='Use MPNN updates instead of attention')

    parser.add_argument('--restore', type=str, default='', help='Checkpoint file for restoration')
    parser.add_argument('--name', type=str, default='', help='Experiment name for logging')
    parser.add_argument('--file_data', type=str, default='../data/cath/chain_set.jsonl', help='input chain file')
    parser.add_argument('--file_splits', type=str, default='../data/cath/chain_set_splits.json', help='input chain file')
    parser.add_argument('--batch_tokens', type=int, default=2500, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--seed', type=int, default=1111, help='random seed for reproducibility')
    parser.add_argument('--cuda', action='store_true', help='whether to use CUDA for computation')
    parser.add_argument('--augment', action='store_true', help='Enrich with alignments')
    
    parser.add_argument('--shuffle', type=float, default=0., help='Shuffle for training a background model')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing rate')
    args = parser.parse_args()
    return args

def setup_device_rng(args):
    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # CUDA device handling.
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    return device

def setup_model(hyperparams, device):
    # Build the model
    if hyperparams['model_type'] == 'structure':
        model = struct2seq.Struct2Seq(
            num_letters=hyperparams['vocab_size'], 
            node_features=hyperparams['hidden'],
            edge_features=hyperparams['hidden'], 
            hidden_dim=hyperparams['hidden'],
            k_neighbors=hyperparams['k_neighbors'],
            protein_features=hyperparams['features'],
            dropout=hyperparams['dropout'],
            use_mpnn=hyperparams['mpnn']
        ).to(device)
    elif hyperparams['model_type'] == 'sequence':
        model = seq_model.SequenceModel(
            num_letters=hyperparams['vocab_size'],
            hidden_dim=hyperparams['hidden'],
            top_k=hyperparams['k_neighbors']
        ).to(device)
    elif hyperparams['model_type'] == 'rnn':
        model = seq_model.LanguageRNN(
            num_letters=hyperparams['vocab_size'],
            hidden_dim=hyperparams['hidden']
        ).to(device)

    print('Number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    return model

def setup_cli_model():
    args = get_args()
    device = setup_device_rng(args)
    model = setup_model(vars(args), device)
    if args.restore is not '':
        load_checkpoint(args.restore, model)
    return args, device, model

def load_checkpoint(checkpoint_path, model):
    print('Loading checkpoint from {}'.format(checkpoint_path))
    state_dicts = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dicts['model_state_dict'])
    print('\tEpoch {}'.format(state_dicts['epoch']))
    return

def featurize(batch, device, shuffle_fraction=0.):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)

    def shuffle_subset(n, p):
        n_shuffle = np.random.binomial(n, p)
        ix = np.arange(n)
        ix_subset = np.random.choice(ix, size=n_shuffle, replace=False)
        ix_subset_shuffled = np.copy(ix_subset)
        np.random.shuffle(ix_subset_shuffled)
        ix[ix_subset] = ix_subset_shuffled
        return ix

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b['coords'][c] for c in ['N', 'CA', 'C', 'O']], 1)
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        if shuffle_fraction > 0.:
            idx_shuffle = shuffle_subset(l, shuffle_fraction)
            S[i, :l] = indices[idx_shuffle]
        else:
            S[i, :l] = indices

    # Mask
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    return X, S, mask, lengths

def plot_log_probs(log_probs, total_step, folder=''):
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    reorder = 'DEKRHQNSTPGAVILMCFWY'
    permute_ix = np.array([alphabet.index(c) for c in reorder])
    plt.close()
    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(111)
    P = np.exp(log_probs.cpu().data.numpy())[0].T
    plt.imshow(P[permute_ix])
    plt.clim(0,1)
    plt.colorbar()
    plt.yticks(np.arange(20), [a for a in reorder])
    ax.tick_params(
        axis=u'both', which=u'both',length=0, labelsize=5
    )
    plt.tight_layout()
    plt.savefig(folder + 'probs{}.pdf'.format(total_step))
    return

def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

def loss_smoothed_reweight(S, log_probs, mask, weight=0.1, factor=10.):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    # Upweight the examples with worse performance
    loss = -(S_onehot * log_probs).sum(-1)
    
    # Compute an error-weighted average
    loss_av_per_example = torch.sum(loss * mask, -1, keepdim=True) / torch.sum(mask, -1, keepdim=True)
    reweights = torch.nn.functional.softmax(factor * loss_av_per_example, 0)
    mask_reweight = mask * reweights
    loss_av = torch.sum(loss * mask_reweight) / torch.sum(mask_reweight)
    return loss, loss_av
