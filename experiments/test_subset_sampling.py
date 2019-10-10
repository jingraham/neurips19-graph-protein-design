from __future__ import print_function
import json, time, os, sys, glob, copy

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

from utils import *


args, device, model = setup_cli_model()
load_checkpoint(checkpoint, model)

optimizer = noam_opt.get_std_opt(model.parameters(), args.hidden)
criterion = torch.nn.NLLLoss(reduction='none')


dataset_path = '../../../dataset/ollikainen/ollikainen_set.json.txt'
test_set = data.StructureDataset(dataset_path, truncate=None, max_length=500)
loader_test =  data.StructureLoader(test_set, batch_size=args.batch_tokens)


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
    ax.tick_params(axis=u'both', which=u'both',length=0, labelsize=5)
    plt.tight_layout()
    plt.savefig(base_folder + 'probs{}.pdf'.format(total_step))
    return

def _loss(S, log_probs, mask):
    """ Negative log probabilities """
    loss = criterion(
        log_probs.contiguous().view(-1,log_probs.size(-1)),
        S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av


def _scores(S, log_probs, mask):
    """ Negative log probabilities """
    loss = criterion(
        log_probs.contiguous().view(-1,log_probs.size(-1)),
        S.contiguous().view(-1)
    ).view(S.size())
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores

def _S_to_seq(S):
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    seq = ''.join([alphabet[c] for c in S.tolist()])
    return seq


# Build paths for experiment
base_folder = time.strftime("test/%y%b%d_%I%M%p/", time.localtime())
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
for subfolder in ['alignments']:
    if not os.path.exists(base_folder + subfolder):
        os.makedirs(base_folder + subfolder)
logfile = base_folder + '/log.txt'
with open(base_folder + '/hyperparams.json', 'w') as f:
    json.dump(vars(args), f)


BATCH_COPIES = 100
NUM_BATCHES = 1
# temperatures = [1.0, 0.5, 0.1, 0.01]
temperatures = [0.01]

total_step = 0
# Validation epoch
model.eval()
with torch.no_grad():
    test_sum, test_weights = 0., 0.
    for ix, protein in enumerate(test_set):
        
        batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
        X, S, mask, lengths = featurize(batch_clones, device)

        # log_probs = model(X, S, lengths, mask)
        # log_probs_sequential = model.forward_sequential(X, S, lengths, mask)
        # # _plot_log_probs(log_probs, 'normal.pdf')
        # # _plot_log_probs(log_probs_sequential, 'sequential.pdf')
        # loss, loss_av = _loss(S, log_probs, mask)
        # loss_seq, loss_av_seq = _loss(S, log_probs_sequential, mask)
        # print('Loss from normal:', loss_av.cpu().data.numpy())
        # print('Loss from sequential:', loss_av_seq.cpu().data.numpy())

        log_probs = model(X, S, lengths, mask)
        _plot_log_probs(log_probs, 'sequential.pdf')

        # Generate some sequences
        ali_file = base_folder + 'alignments/' + batch_clones[0]['name'] + '.fa'
        
        with open(ali_file, 'w') as f:
            f.write('>Native\n{}\n'.format(_S_to_seq(S[0])))
            for temp in temperatures:
                for j in range(NUM_BATCHES):
                    S_sample = model.sample(X, lengths, mask, temperature=temp)

                    # Compute scores
                    log_probs = model(X, S_sample, lengths, mask)
                    scores = _scores(S_sample, log_probs, mask)
                    scores = scores.cpu().data.numpy()

                    # print('Protein {}, batch {}'.format(protein['name'], j))
                    for b_ix in range(BATCH_COPIES):
                        seq = _S_to_seq(S_sample[b_ix])
                        score = scores[b_ix]
                        f.write('>T={}, sample={}, score={}\n{}\n'.format(temp,b_ix,score,seq))

                frac_recovery = torch.sum(mask * (S.eq(S_sample).float())) / torch.sum(mask)
                frac_recovery = frac_recovery.cpu().data.numpy()
                # print(mask)
                # print(frac_recovery, torch.numel(mask), torch.sum(mask).cpu().data.numpy(), batch_clones[0]['name'])
                print(frac_recovery)
