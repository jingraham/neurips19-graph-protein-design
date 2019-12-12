#!/bin/bash

python3 train_s2s.py --cuda --file_data ../data/chain_set.jsonl --file_splits ../data/chain_set_splits.json --batch_tokens=6000 --features full --name h128_full

python3 train_s2s.py --cuda --file_data ../data/chain_set.jsonl --file_splits ../data/chain_set_splits.json --batch_tokens=6000 --features hbonds --name h128_hbonds

python3 train_s2s.py --cuda --file_data ../data/chain_set.jsonl --file_splits ../data/chain_set_splits.json --batch_tokens=6000 --features dist --name h128_dist

python3 train_s2s.py --cuda --file_data ../data/chain_set.jsonl --file_splits ../data/chain_set_splits.json --batch_tokens=6000 --features coarse --name h128_coarse

python3 train_s2s.py --cuda --file_data ../data/chain_set.jsonl --file_splits ../data/chain_set_splits.json --batch_tokens=6000 --features full --name h128_mpnn --mpnn

python3 train_s2s.py --cuda --file_data ../data/cath/chain_set.jsonl --file_splits ../data/ollikainen/splits_ollikainen.json --batch_tokens=6000 --features full  --mpnn --name h128_full_mpnn_ollikainen

python3 train_s2s.py --cuda --file_data ../data/cath/chain_set.jsonl --file_splits ../data/ollikainen/splits_ollikainen.json --batch_tokens=6000 --features full --name h128_full_ollikainen