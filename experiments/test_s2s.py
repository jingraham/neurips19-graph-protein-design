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
from utils import *


# Simulate extra arguments
# sys.argv = [sys.argv[0], '--features', 'full', '--restore', 'log/h128_full/epoch90_step61740.pt', '--file_splits', '../data/SPIN2/test_split_sc.json']

args, device, model = setup_cli_model()

optimizer = noam_opt.get_std_opt(model.parameters(), args.hidden)
criterion = torch.nn.NLLLoss(reduction='none')

# Load the test set from a splits file
with open(args.file_splits) as f:
    dataset_splits = json.load(f)
test_names = dataset_splits['test']
# Load the dataset
dataset = data.StructureDataset(args.file_data, truncate=None, max_length=500)
# Split the dataset
dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
test_set = Subset(dataset, [dataset_indices[name] for name in test_names])
loader_test = data.StructureLoader(test_set, batch_size=args.batch_tokens)
print('Testing {} domains'.format(len(test_set)))


total_step = 0
# Validation epoch
model.eval()
with torch.no_grad():
    test_sum, test_weights = 0., 0.
    for ix, batch in enumerate(loader_test):
        X, S, mask, lengths = featurize(batch, device)

        # Perplexity
        log_probs = model(X, S, lengths, mask)
        loss, loss_av = loss_nll(S, log_probs, mask)

        loss_per_datum = torch.sum(mask * loss, 1) / torch.sum(mask, 1)
        print(loss_per_datum.cpu().data.numpy(), loss_per_datum.std())
        print(ix, len(batch),[b['name'] for b in batch])

        # Accumulate
        test_sum += torch.sum(loss * mask).cpu().data.numpy()
        test_weights += torch.sum(mask).cpu().data.numpy()

test_loss = test_sum / test_weights
test_perplexity = np.exp(test_loss)
print('Perplexity: {}'.format(test_perplexity))
