from __future__ import print_function
import json, time, os, sys
import copy

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt

# Library code
sys.path.insert(0, '..')
from struct2seq import *
from torch.utils.data.dataset import random_split
from utils import setup_cli_model, load_checkpoint, featurize

# SIMULATE
sys.argv = [sys.argv[0], '--features', 'full', '--restore', 'log/h128_full/epoch90_step61740.pt']
sys.argv = [sys.argv[0], '--features', 'hbonds', '--restore', 'log/h128_hbonds/epoch40_step27440.pt']

args, device, model = setup_cli_model()

# dataset_path = '../../../dataset/rocklin/rocklin_mini_exemplars.json.txt'
# dataset_path = '../../../dataset/rocklin/rocklin_full.json.txt'
dataset_path = '../data/rocklin/output/rocklin_mutations.jsonl'

# Load the dataset
dataset = data.StructureDataset(dataset_path, truncate=None, max_length=500)

criterion = torch.nn.NLLLoss(reduction='none')
def _loss(S, log_probs, mask, num_letters=20):
    """ Negative log probabilities """
    loss = criterion(
        log_probs.contiguous().view(-1,num_letters),
        S.contiguous().view(-1)
    ).view(S.size())
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av

start_test = time.time()
total_step = 0
# Training epoch
model.eval()
with torch.no_grad():
    train_sum, train_weights = 0., 0.
    for fold_ix, fold in enumerate(dataset):
        start_batch = time.time()

        batch = [fold]
        fold_mut = copy.deepcopy(fold)
        print(len(fold['mutation_data']))

        num_mutations = len(fold['mutation_data'])
        rocklin_df = pd.DataFrame(columns=['seq', 'stabilityscore', 'neglogp'])
        try:

            for mut_ix, (mut_seq, effect) in enumerate(fold['mutation_data']):
                fold_mut['seq'] = mut_seq

                # Get a batch
                X, S, mask, lengths = featurize([fold_mut], device)
                elapsed_featurize = time.time() - start_batch

                log_probs = model(X, S, lengths, mask)
                loss, loss_av = _loss(S, log_probs, mask)
                neglogp = torch.sum(loss * mask, dim=1) / torch.sum(mask, dim=1)
                neglogp = neglogp.cpu().data.numpy().tolist()[0]
                print(fold['name'], neglogp, effect, mut_ix, num_mutations)
                rocklin_df.loc[mut_ix] = [mut_seq, effect, neglogp]

            rocklin_df.to_csv('rocklin/mutations/' + fold['name'] + '_' + args.features + '.tsv', sep='\t')

            plt.clf()
            plt.scatter(rocklin_df['neglogp'], rocklin_df['stabilityscore'], s=5)
            plt.xlabel('Transformer Neglogp')
            plt.ylabel('Stability score in experiment')
            plt.savefig('rocklin/mutations/' + fold['name'] + '_' + args.features + '.pdf')
        except:
            print('failed on ' + fold['name'])

        # # Designs
        # designs = ['HHH', 'EHEE', 'HEEH', 'EEHEE']
        # for design in designs:
        #     idx = rocklin_df['name'].str.startswith(design + '_')
        #     design_data = rocklin_df[idx]
        #     plt.clf()
        #     plt.scatter(design_data['total_score'], design_data['stabilityscore'],s=5)
        #     plt.xlabel('Rosetta score')
        #     plt.ylabel('Stability score')
        #     plt.savefig('rocklin/' + design + '_rosetta.pdf')
        #     plt.clf()
        #     plt.scatter(design_data['neglogp'], design_data['stabilityscore'],s=5)
        #     plt.xlabel('Transformer neglogp')
        #     plt.ylabel('Stability score')
        #     plt.savefig('rocklin/' + design + '_model.pdf')    

        # Mess it up
        # S[0,0:10] = 9
        # log_probs = model(X, S, lengths, mask)
        # loss, loss_av = _loss(S, log_probs, mask)
        # neglogp = torch.sum(loss * mask, dim=1) / torch.sum(mask, dim=1)
        # neglogp = neglogp.cpu().data.numpy().tolist()
        # print(fold['name'], neglogp)



        # # Collect per protein log-probabilities
        # data_dict = {
        #     key: [entry[key] for entry in batch]
        #     for key in ['name', 'stabilityscore', 'total_score', 'total_score_talaris']
        # }
        # batch_df = pd.DataFrame(data=data_dict)
        
        # batch_df['neglogp'] = pd.Series(neglogp)
        # rocklin_df = rocklin_df.append(batch_df, ignore_index=True, sort=False)

        # print(lengths)
        # batch_df['total_score'] = batch_df['total_score'] / lengths
        # batch_df['total_score_talaris'] = batch_df['total_score_talaris'] / lengths

        # # Timing
        # elapsed_batch = time.time() - start_batch
        # elapsed_test = time.time() - start_test
        # total_step += 1
        # print(total_step, elapsed_test, np.exp(loss_av.cpu().data.numpy()))

        # # Accumulate
        # train_sum += torch.sum(loss * mask).cpu().data.numpy()
        # train_weights += torch.sum(mask).cpu().data.numpy()

        # # DEBUG UTILIZATION Stats
        # if args.cuda:
        #     utilize_mask = 100. * mask.sum().cpu().data.numpy() / float(mask.numel())
        #     utilize_gpu = float(torch.cuda.max_memory_allocated(device=device)) / 1024.**3
        #     tps = mask.cpu().data.numpy().sum() / elapsed_batch
        #     print('Tokens per second: {:.2f}, Mask efficiency: {:.2f}, GPU max allocated: {:.2f}'.format(tps, utilize_mask, utilize_gpu))


# rocklin_df.to_csv('rocklin/rocklin_results.tsv', sep='\t')
# # Designs
# designs = ['HHH', 'EHEE', 'HEEH', 'EEHEE']
# for design in designs:
#     idx = rocklin_df['name'].str.startswith(design + '_')
#     design_data = rocklin_df[idx]
#     plt.clf()
#     plt.scatter(design_data['total_score'], design_data['stabilityscore'],s=5)
#     plt.xlabel('Rosetta score')
#     plt.ylabel('Stability score')
#     plt.savefig('rocklin/' + design + '_rosetta.pdf')
#     plt.clf()
#     plt.scatter(design_data['neglogp'], design_data['stabilityscore'],s=5)
#     plt.xlabel('Transformer neglogp')
#     plt.ylabel('Stability score')
#     plt.savefig('rocklin/' + design + '_model.pdf')

        
