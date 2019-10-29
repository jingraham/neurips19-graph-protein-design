import os, json, random
from collections import defaultdict

# Get SIFTS annotations from
# pdb_chain_cath_uniprot.tsv.gz


def load_chain_set(chain_file, verbose=True):
    chain_set = {}
    with open(chain_file,'r') as f:
        for i, line in enumerate(f):
            chain_name, jsons = line.split('\t')
            chain_set[chain_name] = json.loads(jsons)
            if verbose and (i + 1) % 1000 == 0:
                print('Loaded {} chains'.format(i+1))
    return chain_set

def load_cath_topologies(cath_domain_file = '../pfam32/cath/cath-domain-list-v4_2_0.txt'):
    print('Loading CATH domain nodes')
    cath_nodes = defaultdict(list)
    with open(cath_domain_file,'r') as f:
        lines = [line.strip() for line in f if not line.startswith('#')]
        for line in lines:
            entries = line.split()
            cath_id, cath_node = entries[0], '.'.join(entries[1:4])
            chain_name = cath_id[:4].lower() + '.' + cath_id[4]
            cath_nodes[chain_name].append(cath_node)
    # Uniquify the list
    cath_nodes = {key:list(set(val)) for key,val in cath_nodes.iteritems()}
    return cath_nodes


if __name__ == '__main__':
    valid_fraction = 0.1

    # We will make splits
    splits = {key:set() for key in ['test', 'train', 'validation']}
    
    # What topologies are in the Ollikainen set?
    cath_nodes = load_cath_topologies()
    ollikainen_set = load_chain_set('ollikainen_set.json.txt')
    topo_test = [t for key in ollikainen_set for t in cath_nodes[key]]
    topo_test = sorted(list(set(topo_test)))
    print(topo_test)

    # Build the test set from all topologies that appeared in Ollikainen 40 PDBs
    dataset = load_chain_set('../pfam32/chain_set.json.txt')
    dataset_names = dataset.keys()
    dataset_values = dataset.values()
    test_topology_set = set(topo_test)
    splits['test'] = [
        name for name in dataset_names
        if not set(cath_nodes[name]).isdisjoint(test_topology_set)
    ]

    # Set aside a small portion of the train topologies for validation
    splits['train'] = set(dataset_names).difference(set(splits['test']))
    topo_train = list(set([t for name in splits['train'] for t in cath_nodes[name]]))
    num_topo_validation = int(valid_fraction * len(topo_train))
    random.seed(42)
    random.shuffle(topo_train)
    topo_validation = set(topo_train[:num_topo_validation])
    splits['validation'] = [
        name for name in splits['train']
        if not set(cath_nodes[name]).isdisjoint(topo_validation)
    ]

    # Everything else is train
    splits['train'] = set(splits['train']).difference(set(splits['validation']))
    splits['train'] = set(splits['train']).difference(set(splits['test']))
    splits['train'] = list(splits['train'])

    print(len(splits['test']), len(splits['train']), len(splits['validation']))
    print(len(dataset_names))


    with open('splits_ollikainen.json', 'w') as f:
        json.dump(splits, f)
