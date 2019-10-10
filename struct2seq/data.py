from __future__ import print_function
from torch.utils.data import Dataset
import numpy as np
import json, time, copy
import random

class StructureDataset():
    def __init__(self, json_txt_file, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWY'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
        }

        with open(json_txt_file) as f:
            self.data = []

            lines = f.readlines()
            start = time.time()
            for i, line in enumerate(lines):
                name, entry = line.split('\t')
                entry = json.loads(entry)
                entry['name'] = name
                seq = entry['seq']

                # Convert raw coords to np arrays
                for key, val in entry['coords'].items():
                    entry['coords'][key] = np.asarray(val)

                # Check if in alphabet
                bad_chars = set([s for s in seq]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry['seq']) <= max_length:
                        self.data.append(entry)
                    else:
                        discard_count['too_long'] += 1
                else:
                    print(name, bad_chars, entry['seq'])
                    discard_count['bad_chars'] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

            print('Discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SequenceDataset():
    def __init__(self, json_txt_file, verbose=True, truncate=None, max_length=100,
        alphabet='ACDEFGHIKLMNPQRSTVWY'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
        }

        with open(json_txt_file) as f:
            self.data = []
            start = time.time()
            for i, line in enumerate(f):
                name, seq = line.strip().split('\t')
                entry = {'name': name, 'seq': seq}
                self.data.append(entry)

                # # Check if in alphabet
                # bad_chars = set([s for s in seq]).difference(alphabet_set)
                # if len(bad_chars) == 0:
                #     if len(entry['seq']) <= max_length:
                #         self.data.append(entry)
                #     else:
                #         discard_count['too_long'] += 1
                # else:
                #     discard_count['bad_chars'] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 100000 == 0:
                    elapsed = time.time() - start
                    print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

            print('Discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class AlignmentDataset():
    def __init__(self, jsonl_file, verbose=True, alphabet='ACDEFGHIKLMNPQRSTVWY'):
        self.data = {}
        self.alphabet = alphabet
        with open(jsonl_file) as f:
            start = time.time()
            for i, line in enumerate(f):
                json_dict = json.loads(line)
                num_seqs = len(json_dict['ali'])
                self.data[json_dict['name']] = json_dict['ali']

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print('{} alignments ({} loaded) in {:.1f} s'.format(len(self.data), i+1, elapsed))

    def augment(self, batch):
        batch_augment = copy.deepcopy(batch)

        for b1, b2 in zip(batch, batch_augment):
            name = b1['name']
            if name in self.data:
                seq = b1['seq']
                seq_new = random.choice(self.data[name])
                b2['seq'] = ''.join([c2 if c2 in self.alphabet else c1 for c1, c2 in zip(seq, seq_new)])

        # for b1, b2 in zip(batch, batch_augment):
        #     s1, s2 = b1['seq'], b2['seq']
        #     print(b1['seq'])
        #     print(''.join(['|' if c1==c2 else ' ' for c1,c2 in zip(s1,s2)]))
        #     print(b2['seq'])
        return batch_augment

class StructureLoader():
    def __init__(self, dataset, batch_size=100, shuffle=True,
        collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch
        