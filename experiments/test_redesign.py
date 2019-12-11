from __future__ import print_function
import json, time, os, sys, glob, copy

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import random

## Library code
sys.path.insert(0, '..')
from struct2seq import *

# Debug plotting
import matplotlib
import glob
import pandas as pd

from matplotlib import pyplot as plt
plt.switch_backend('agg')

from utils import *

# Simulate extra arguments
# sys.argv = [
#     sys.argv[0], '--features', 'full',
#     '--restore', 'log/h128_full_ollikainen/epoch48_step22896.pt',
#     '--file_data', '../data/ollikainen/ollikainen_set.jsonl',
#     '--file_splits', '../data/ollikainen/ollikainen_benchmark.json'
# ]
# --cuda --features full --restore log/h128_full_ollikainen/epoch48_step22896.pt --file_data ../data/ollikainen/ollikainen_set.jsonl --file_splits ../data/ollikainen/ollikainen_benchmark.json

args, device, model = setup_cli_model()
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

def _S_to_seq(S, mask):
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    seq = ''.join([alphabet[c] for c, m in zip(S.tolist(), mask.tolist()) if m > 0])
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


BATCH_COPIES = 50
NUM_BATCHES = 1
# temperatures = [1.0, 0.33, 0.1, 0.033, 0.01]
temperatures = [0.1] * 2

# Timing
start_time = time.time()
total_residues = 0

total_step = 0
# Validation epoch
model.eval()
with torch.no_grad():
    test_sum, test_weights = 0., 0.
    for ix, protein in enumerate(test_set):
        
        batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
        X, S, mask, lengths = featurize(batch_clones, device)

        log_probs = model(X, S, lengths, mask)
        scores = _scores(S, log_probs, mask)
        native_score = scores.cpu().data.numpy()[0]
        print(scores)

        # Generate some sequences
        ali_file = base_folder + 'alignments/' + batch_clones[0]['name'] + '.fa'
        
        with open(ali_file, 'w') as f:
            native_seq = _S_to_seq(S[0], mask[0])
            f.write('>Native, score={}\n{}\n'.format(native_score, native_seq))
            for temp in temperatures:
                for j in range(NUM_BATCHES):
                    S_sample = model.sample(X, lengths, mask, temperature=temp)

                    # Compute scores
                    log_probs = model(X, S_sample, lengths, mask)
                    scores = _scores(S_sample, log_probs, mask)
                    scores = scores.cpu().data.numpy()

                    for b_ix in range(BATCH_COPIES):
                        seq = _S_to_seq(S_sample[b_ix], mask[0])
                        score = scores[b_ix]
                        f.write('>T={}, sample={}, score={}\n{}\n'.format(temp,b_ix,score,seq))

                    total_residues += torch.sum(mask).cpu().data.numpy()
                    elapsed = time.time() - start_time
                    residues_per_second = float(total_residues) / float(elapsed)
                    print('{} residues / s'.format(residues_per_second))

                frac_recovery = torch.sum(mask * (S.eq(S_sample).float())) / torch.sum(mask)
                frac_recovery = frac_recovery.cpu().data.numpy()
                # print(mask)
                # print(frac_recovery, torch.numel(mask), torch.sum(mask).cpu().data.numpy(), batch_clones[0]['name'])
                print(frac_recovery)

# Plot the results
files = glob.glob(base_folder + 'alignments/*.fa')
df = pd.DataFrame(columns=['name', 'T', 'score', 'similarity'])

def similarity(seq1, seq2):
    matches = sum([c1==c2 for c1, c2 in zip(seq1,seq2)])
    return float(matches) / len(seq1)

for file in files:
    with open(file, 'r') as f:
        # Skip over native
        entries = f.read().split('>')[1:]
        entries = [e.strip().split('\n') for e in entries]

        # Get native information
        native_header = entries[0][0]
        native_score = float(native_header.split(', ')[1].split('=')[1])
        native_seq = entries[0][1]
        print(entries[0])
        print(native_score)

        for header, seq in entries[1:]:
            T, sample, score = [float(s.split('=')[1]) for s in header.split(', ')]
            pdb, chain = file.split('/')[-1].split('.')[0:2]

            df = df.append({
                'name': pdb + '.' + chain,
                'T': T, 'score': score,
                'native': native_score,
                'similarity': similarity(native_seq, seq)
                },  ignore_index=True
            )

df['diff'] = -(df['score'] - df['native'])

boxplot = df.boxplot(column='diff', by= 'T')
plt.xlabel('Decoding temperature')
plt.ylabel('log P(sample) - log P(native)')
boxplot.get_figure().gca().set_title('')
boxplot.get_figure().suptitle('')
plt.tight_layout()
plt.savefig(base_folder + 'decoding.pdf')

boxplot = df.boxplot(column='similarity', by= 'T')
plt.xlabel('Decoding temperature')
plt.ylabel('Native sequence recovery')
boxplot.get_figure().gca().set_title('')
boxplot.get_figure().suptitle('')
plt.tight_layout()
plt.savefig(base_folder + 'recovery.pdf')

# Store the results
df_mean = df.groupby(['name', 'T'], as_index=False).mean()
df_mean.to_csv(base_folder + 'results.csv')

print('Speed total: {} residues / s'.format(residues_per_second))
print('Median', df_mean['similarity'].median())
