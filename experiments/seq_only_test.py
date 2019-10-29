from __future__ import print_function
import json, time, os, sys, glob

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

# Library code
sys.path.insert(0, '..')
from struct2seq import *

import matplotlib
from matplotlib import pyplot as plt
plt.switch_backend('agg')


# sys.argv = [sys.argv[0], '--rnn', '--restore', 'log/19Mar14_1216AM_RNN_128/checkpoints/epoch18_step3798.pt', '--hidden', '128']

from argparse import ArgumentParser
parser = ArgumentParser(description='Structure to sequence modeling')
parser.add_argument('--hidden', type=int, default=256, help='number of hidden dimensions')
parser.add_argument('--file_data', type=str, default='../data/cath/chain_set.jsonl', help='input chain file')
parser.add_argument('--file_splits', type=str, default='../data/cath/chain_set_splits.json', help='input chain file')
parser.add_argument('--restore', type=str, default='', help='Restore from checkpoint')
parser.add_argument('--batch_tokens', type=int, default=2500, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--seed', type=int, default=1111, help='random seed for reproducibility')
parser.add_argument('--cuda', action='store_true', help='whether to use CUDA for computation')
parser.add_argument('--augment', action='store_true', help='Enrich with alignments')
parser.add_argument('--rnn', action='store_true', help='RNN model')
parser.add_argument('--split_random', action='store_true', help='Split randomly')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Hyperparameters
hyperparams = {
    'batch_size': args.batch_tokens,
    'hidden':  args.hidden,
    'letters': 20
}

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Build the model
MODEL = seq_model.LanguageRNN if args.rnn else seq_model.SequenceModel
model = MODEL(
    num_letters=20, 
    hidden_dim=args.hidden
).to(device)
print('Number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))

if args.restore is not '':
    model_state = torch.load(args.restore, map_location='cpu')['model_state_dict']
    model.load_state_dict(model_state)
criterion = torch.nn.NLLLoss(reduction='none')

# Our test
with open(args.file_splits) as f:
    dataset_splits = json.load(f)
test_names = dataset_splits['test']
print(test_names)

# Load the dataset
dataset = data.StructureDataset(args.file_data, truncate=None, max_length=500)

# Split the dataset
dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
test_indices = [dataset_indices[name] for name in test_names]
test_set = Subset(dataset, test_indices)
loader_test = data.StructureLoader(test_set, batch_size=args.batch_tokens)

print('Testing {} domains'.format(len(test_set)))

def _plot_log_probs(log_probs, total_step):
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
    plt.savefig(base_folder + 'probs{}.pdf'.format(total_step))
    return

def _featurize(batch, hyperparams=hyperparams):
    """ Represent structure as an attributed graph """
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 3, 3])
    S = np.zeros([B, L_max], dtype=np.int32)

    # Build the batch
    for i, b in enumerate(batch):
        x_n = b['coords']['N']
        x_ca = b['coords']['CA']
        x_c = b['coords']['C']
        x = np.stack([x_n, x_ca, x_c], 1)
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
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

def _loss(S, log_probs, mask):
    """ Negative log probabilities """
    loss = criterion(
        log_probs.contiguous().view(-1,hyperparams['letters']),
        S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

total_step = 0
# Validation epoch
model.eval()
with torch.no_grad():
    test_sum, test_weights = 0., 0.
    for ix, batch in enumerate(loader_test):
        X, S, mask, lengths = _featurize(batch)
        log_probs = model(S, lengths, mask)
        loss, loss_av = _loss(S, log_probs, mask)

        print(ix, len(batch),[b['name'] for b in batch], len(test_set))

        # Accumulate
        test_sum += torch.sum(loss * mask).cpu().data.numpy()
        test_weights += torch.sum(mask).cpu().data.numpy()

test_loss = test_sum / test_weights
test_perplexity = np.exp(test_loss)
print('Perplexity: {}'.format(test_perplexity))
