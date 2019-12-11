import glob, json
import numpy as np
import pandas as pd


suffix = 'sc'
test_set = 'test_split_' + suffix + '.json'
result_folder = 'results_' + suffix + '/'

with open(test_set, 'r') as f:
    test_names = json.load(f)['test']

spin_files = [result_folder + name + '.spin2' for name in test_names]
spin_data = {}
for spin_file in spin_files:
    name = spin_file.split('/')[1][:6]
    spin_data[name] = pd.read_table(spin_file, sep=r"\s+", engine='python')
    print(spin_file, name)

def _perplexity(seq, table):
    logprobs = []
    probs = []
    correct = []
    for i,c in enumerate(seq):
        # Table is listed to 100 and offset by one
        prob = table.loc[i+1,:]
        prob = prob / prob.sum()

        correct.append(prob.idxmax() == c)
        probs.append(prob[c])
        logp = np.log(prob[c])
        logprobs.append(logp)
    return logprobs, probs, correct

# Perplexity time
logprobs_total = []
probs_total = []
correct_total = []
chains = {}
chain_file = '../../../../../dataset/pfam32/chain_set.json.txt'
with open(chain_file,'r') as f:
    for line in f:
        chain_name, jsons = line.split('\t')
        if chain_name in spin_data:
            domain = json.loads(jsons)
            logprobs, probs, correct = _perplexity(domain['seq'], spin_data[chain_name])
            probs_total.extend(probs)
            logprobs_total.extend(logprobs)
            correct_total.extend(correct)

    logprobs_total = np.array(logprobs_total)
    perplex = np.exp(-np.mean(logprobs_total))
    accuracy = np.mean(np.array(correct_total))
    avgprob = np.mean(probs)

    print(perplex)
    print(accuracy)
    print(avgprob)