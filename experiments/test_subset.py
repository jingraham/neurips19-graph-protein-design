from __future__ import print_function
import json, time, os, sys, glob

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

## Library code
sys.path.insert(0, '..')
from struct2seq import *

# Debug plotting
import matplotlib
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from utils import setup_cli_model, load_checkpoint, featurize


# Simulate extra arguments
sys.argv = [sys.argv[0], '--features', 'full']

args, device, model = setup_cli_model()
load_checkpoint(checkpoint, model)

optimizer = noam_opt.get_std_opt(model.parameters(), args.hidden)
criterion = torch.nn.NLLLoss(reduction='none')

# Our test
with open(args.file_splits) as f:
    dataset_splits = json.load(f)
test_names = dataset_splits['test']

# Load the dataset
if args.augment:
    alignments = data.AlignmentDataset(args.file_alignments)
dataset = data.StructureDataset(args.file_data, truncate=None, max_length=500)

# Split the dataset
dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
test_indices = [dataset_indices[name] for name in test_names]
test_set = Subset(dataset, test_indices)
loader_test = data.StructureLoader(
    test_set, batch_size=args.batch_tokens
)
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

def _loss(S, log_probs, mask):
    """ Negative log probabilities """
    loss = criterion(
        log_probs.contiguous().view(-1,args.vocab_size),
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
        X, S, mask, lengths = featurize(batch, device)

        # Perplexity
        log_probs = model(X, S, lengths, mask)
        loss, loss_av = _loss(S, log_probs, mask)

        loss_per_datum = torch.sum(mask * loss, 1) / torch.sum(mask, 1)
        print(loss_per_datum.cpu().data.numpy(), loss_per_datum.std())
        print(ix, len(batch),[b['name'] for b in batch])

        # Accumulate
        test_sum += torch.sum(loss * mask).cpu().data.numpy()
        test_weights += torch.sum(mask).cpu().data.numpy()

test_loss = test_sum / test_weights
test_perplexity = np.exp(test_loss)
print('Perplexity: {}'.format(test_perplexity))
