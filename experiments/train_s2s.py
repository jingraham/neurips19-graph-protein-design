from __future__ import print_function
import json, time, os, sys, glob
import shutil

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset

# Library code
sys.path.insert(0, '..')
from struct2seq import *
from utils import *

args, device, model = setup_cli_model()
optimizer = noam_opt.get_std_opt(model.parameters(), args.hidden)
criterion = torch.nn.NLLLoss(reduction='none')

# Load the dataset
dataset = data.StructureDataset(args.file_data, truncate=None, max_length=500)

# Split the dataset
dataset_indices = {d['name']:i for i,d in enumerate(dataset)}
with open(args.file_splits) as f:
    dataset_splits = json.load(f)
train_set, validation_set, test_set = [
    Subset(dataset, [
        dataset_indices[chain_name] for chain_name in dataset_splits[key] 
        if chain_name in dataset_indices
    ])
    for key in ['train', 'validation', 'test']
]
loader_train, loader_validation, loader_test = [data.StructureLoader(
    d, batch_size=args.batch_tokens
) for d in [train_set, validation_set, test_set]]
print('Training:{}, Validation:{}, Test:{}'.format(len(train_set),len(validation_set),len(test_set)))

# Build basepath for experiment
if args.name != '':
    base_folder = 'log/' + args.name + '/'
else:
    base_folder = time.strftime('log/%y%b%d_%I%M%p/', time.localtime())
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
subfolders = ['checkpoints', 'plots']
for subfolder in subfolders:
    if not os.path.exists(base_folder + subfolder):
        os.makedirs(base_folder + subfolder)

# Log files
logfile = base_folder + 'log.txt'
with open(logfile, 'w') as f:
    f.write('Epoch\tTrain\tValidation\n')
with open(base_folder + 'args.json', 'w') as f:
    json.dump(vars(args), f)

start_train = time.time()
epoch_losses_train, epoch_losses_valid = [], []
epoch_checkpoints = []
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
        X, S, mask, lengths = featurize(batch, device, shuffle_fraction=args.shuffle)
        elapsed_featurize = time.time() - start_batch

        optimizer.zero_grad()
        log_probs = model(X, S, lengths, mask)
        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask, weight=args.smoothing)
        loss_av_smoothed.backward()
        optimizer.step()

        loss, loss_av = loss_nll(S, log_probs, mask)

        # Timing
        elapsed_batch = time.time() - start_batch
        elapsed_train = time.time() - start_train
        total_step += 1
        print(total_step, elapsed_train, np.exp(loss_av.cpu().data.numpy()), np.exp(loss_av_smoothed.cpu().data.numpy()))

        if False:
            # Test reproducibility
            log_probs_sequential = model.forward_sequential(X, S, lengths, mask)
            loss_sequential, loss_av_sequential = loss_nll(S, log_probs_sequential, mask)
            log_probs = model(X, S, lengths, mask)
            loss, loss_av = loss_nll(S, log_probs, mask)
            print(loss_av, loss_av_sequential)

        # Accumulate true loss
        train_sum += torch.sum(loss * mask).cpu().data.numpy()
        train_weights += torch.sum(mask).cpu().data.numpy()

        # DEBUG UTILIZATION Stats
        if args.cuda:
            utilize_mask = 100. * mask.sum().cpu().data.numpy() / float(mask.numel())
            utilize_gpu = float(torch.cuda.max_memory_allocated(device=device)) / 1024.**3
            tps_train = mask.cpu().data.numpy().sum() / elapsed_batch
            tps_features = mask.cpu().data.numpy().sum() / elapsed_featurize
            print('Tokens/s (train): {:.2f}, Tokens/s (features): {:.2f}, Mask efficiency: {:.2f}, GPU max allocated: {:.2f}'.format(tps_train, tps_features, utilize_mask, utilize_gpu))

        if total_step % 5000 == 0:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.optimizer.state_dict()
            }, base_folder + 'checkpoints/epoch{}_step{}.pt'.format(e+1, total_step))

    # Train image
    plot_log_probs(log_probs, total_step, folder='{}plots/train_{}_'.format(base_folder, batch[0]['name']))

    # Validation epoch
    model.eval()
    with torch.no_grad():
        validation_sum, validation_weights = 0., 0.
        for _, batch in enumerate(loader_validation):
            X, S, mask, lengths = featurize(batch, device, shuffle_fraction=args.shuffle)
            log_probs = model(X, S, lengths, mask)
            loss, loss_av = loss_nll(S, log_probs, mask)

            # Accumulate
            validation_sum += torch.sum(loss * mask).cpu().data.numpy()
            validation_weights += torch.sum(mask).cpu().data.numpy()

    train_loss = train_sum / train_weights
    train_perplexity = np.exp(train_loss)
    validation_loss = validation_sum / validation_weights
    validation_perplexity = np.exp(validation_loss)
    print('Perplexity\tTrain:{}\t\tValidation:{}'.format(train_perplexity, validation_perplexity))

    # Validation image
    plot_log_probs(log_probs, total_step, folder='{}plots/valid_{}_'.format(base_folder, batch[0]['name']))

    with open(logfile, 'a') as f:
        f.write('{}\t{}\t{}\n'.format(e, train_perplexity, validation_perplexity))

    # Save the model
    checkpoint_filename = base_folder + 'checkpoints/epoch{}_step{}.pt'.format(e+1, total_step)
    torch.save({
        'epoch': e,
        'hyperparams': vars(args),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.optimizer.state_dict()
    }, checkpoint_filename)

    epoch_losses_valid.append(validation_perplexity)
    epoch_losses_train.append(train_perplexity)
    epoch_checkpoints.append(checkpoint_filename)

# Determine best model via early stopping on validation
best_model_idx = np.argmin(epoch_losses_valid).item()
best_checkpoint = epoch_checkpoints[best_model_idx]
train_perplexity = epoch_losses_train[best_model_idx]
validation_perplexity = epoch_losses_valid[best_model_idx]
best_checkpoint_copy = base_folder + 'best_checkpoint_epoch{}.pt'.format(best_model_idx + 1)
shutil.copy(best_checkpoint, best_checkpoint_copy)
load_checkpoint(best_checkpoint_copy, model)


# Test epoch
model.eval()
with torch.no_grad():
    test_sum, test_weights = 0., 0.
    for _, batch in enumerate(loader_test):
        X, S, mask, lengths = featurize(batch, device)
        log_probs = model(X, S, lengths, mask)
        loss, loss_av = loss_nll(S, log_probs, mask)
        # Accumulate
        test_sum += torch.sum(loss * mask).cpu().data.numpy()
        test_weights += torch.sum(mask).cpu().data.numpy()

test_loss = test_sum / test_weights
test_perplexity = np.exp(test_loss)
print('Perplexity\tTest:{}'.format(test_perplexity))

with open(base_folder + 'results.txt', 'w') as f:
    f.write('Best epoch: {}\nPerplexities:\n\tTrain: {}\n\tValidation: {}\n\tTest: {}'.format(
        best_model_idx+1, train_perplexity, validation_perplexity, test_perplexity
    ))

