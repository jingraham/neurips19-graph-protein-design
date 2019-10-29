from __future__ import print_function
import json, time, os, sys

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

from argparse import ArgumentParser
parser = ArgumentParser(description='Structure to sequence modeling')
parser.add_argument('--hidden', type=int, default=256, help='number of hidden dimensions')
parser.add_argument('--file_data', type=str, default='../data/cath/chain_set.jsonl', help='input chain file')
parser.add_argument('--file_splits', type=str, default='../data/cath/chain_set_splits.json', help='input chain file')
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
    'rnn': args.rnn,
    'batch_size': args.batch_tokens,
    'hidden': args.hidden,
    'letters': 20
}

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Build the model
MODEL = seq_model.LanguageRNN if args.rnn else seq_model.LanguageRNN
model = MODEL(
    num_letters=hyperparams['letters'], 
    hidden_dim=hyperparams['hidden']
).to(device)
print('Number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))

# DEBUG load the checkpoint
# checkpoint_path = 'log/19Mar04_0533PM_h256_suspicious/checkpoints/epoch18_step21960.pt'
# checkpoint_path = 'log/19Mar04_1118PM/checkpoints/epoch50_step60000.pt'
# model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

optimizer = noam_opt.get_std_opt(model.parameters(), hyperparams['hidden'])
criterion = torch.nn.NLLLoss(reduction='none')

# Load the dataset
dataset = data.StructureDataset(args.file_data, truncate=None, max_length=500)


if args.split_random:
    # Split 
    print('Random split')
    split_frac = 0.1
    num_test = int(len(dataset) * split_frac)
    train_set, test_set = random_split(dataset, [len(dataset) - num_test, num_test])
    train_set, validation_set = random_split(train_set, [len(train_set) - num_test,  num_test])
    loader_train, loader_validation, loader_test = [data.StructureLoader(
        d, batch_size=hyperparams['batch_size']
    ) for d in [train_set, validation_set, test_set]]
else:
    # Split the dataset
    print('Structural split')
    dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
    with open(args.file_splits) as f:
        dataset_splits = json.load(f)
    train_set, validation_set, test_set = [
        Subset(dataset, [dataset_indices[chain_name] for chain_name in dataset_splits[key]])
        for key in ['train', 'validation', 'test']
    ]
    loader_train, loader_validation, loader_test = [data.StructureLoader(
        d, batch_size=hyperparams['batch_size']
    ) for d in [train_set, validation_set, test_set]]


print('Training:{}, Validation:{}, Test:{}'.format(len(train_set),len(validation_set),len(test_set)))

# Build basepath for experiment
base_folder = time.strftime("log/%y%b%d_%I%M%p/", time.localtime())
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
for subfolder in ['checkpoints']:
    if not os.path.exists(base_folder + subfolder):
        os.makedirs(base_folder + subfolder)

# Log files
logfile = base_folder + 'log.txt'
with open(logfile, 'w') as f:
    f.write('Epoch\tTrain\tValidation\n')
with open(base_folder + 'hyperparams.json', 'w') as f:
    json.dump(hyperparams, f)

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

start_train = time.time()
total_step = 0
for e in range(args.epochs):
    # Training epoch
    model.train()
    train_sum, train_weights = 0., 0.
    for train_i, batch in enumerate(loader_train):

        # Augment the data
        if args.augment:
            batch = alignments.augment(batch)

        start_batch = time.time()
        # Get a batch
        X, S, mask, lengths = _featurize(batch)
        elapsed_featurize = time.time() - start_batch

        optimizer.zero_grad()
        log_probs = model(S, lengths, mask)
        loss, loss_av = _loss(S, log_probs, mask)
        loss_av.backward()
        optimizer.step()

        # Timing
        elapsed_batch = time.time() - start_batch
        elapsed_train = time.time() - start_train
        total_step += 1
        print(total_step, elapsed_train, np.exp(loss_av.cpu().data.numpy()))

        # Accumulate
        train_sum += torch.sum(loss * mask).cpu().data.numpy()
        train_weights += torch.sum(mask).cpu().data.numpy()

        # DEBUG UTILIZATION Stats
        if args.cuda:
            utilize_mask = 100. * mask.sum().cpu().data.numpy() / float(mask.numel())
            utilize_gpu = float(torch.cuda.max_memory_allocated(device=device)) / 1024.**3
            tps = mask.cpu().data.numpy().sum() / elapsed_batch
            print('Tokens per second: {:.2f}, Mask efficiency: {:.2f}, GPU max allocated: {:.2f}'.format(tps, utilize_mask, utilize_gpu))

        if total_step % 5000 == 0:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict()
            }, base_folder + 'checkpoints/epoch{}_step{}.pt'.format(e+1, total_step))

    # Test image
    _plot_log_probs(log_probs, total_step)

    # Validation epoch
    model.eval()
    with torch.no_grad():
        validation_sum, validation_weights = 0., 0.
        for _, batch in enumerate(loader_validation):
            X, S, mask, lengths = _featurize(batch)
            log_probs = model(S, lengths, mask)
            loss, loss_av = _loss(S, log_probs, mask)

            # Accumulate
            validation_sum += torch.sum(loss * mask).cpu().data.numpy()
            validation_weights += torch.sum(mask).cpu().data.numpy()

    train_loss = train_sum / train_weights
    train_perplexity = np.exp(train_loss)
    validation_loss = validation_sum / validation_weights
    validation_perplexity = np.exp(validation_loss)
    print('Perplexity\tTrain:{}\t\tValidation:{}'.format(train_perplexity, validation_perplexity))

    with open(logfile, 'a') as f:
        f.write('{}\t{}\t{}\n'.format(e, train_perplexity, validation_perplexity))

    # Save the model
    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.optimizer.state_dict()
    }, base_folder + 'checkpoints/epoch{}_step{}.pt'.format(e+1, total_step))
