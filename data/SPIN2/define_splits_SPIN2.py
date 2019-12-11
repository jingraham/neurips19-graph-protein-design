import glob, os
import json


for suffix in ['sc', 'L100']:
    results_files = glob.glob('results_'+ suffix +'/*.spin2')
    chains_spin2 = [file.split('/')[-1][:6] for file in results_files]

    with open('../cath/chain_set_splits_revise.json') as f:
        splits = json.load(f)

    overlap = set(splits['test']).intersection(set(chains_spin2))

    splits_new = {
        'test': sorted(list(overlap))
    }
    print(len(splits['test']), len(chains_spin2), len(splits_new['test']))
    with open('test_split_'+ suffix +'.json', 'w') as f:
        json.dump(splits_new, f)
        print(len(splits_new['test']))