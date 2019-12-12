#!/bin/bash

CHECKPOINT=log/h128_full/best*.pt
ARGS="--cuda --features full --restore ${CHECKPOINT} --file_data ../data/cath/chain_set.jsonl --file_splits ../data/SPIN2/test_split_sc.json"
python3 test_redesign.py --seed 111 ${ARGS} > test/sc_full_1.log
python3 test_redesign.py --seed 222 ${ARGS} > test/sc_full_2.log
python3 test_redesign.py --seed 333 ${ARGS} > test/sc_full_3.log
python3 test_redesign.py --seed 444 ${ARGS} > test/sc_full_4.log

CHECKPOINT=log/h128_full_mpnn/best*.pt
ARGS="--cuda --features full --mpnn --restore ${CHECKPOINT} --file_data ../data/cath/chain_set.jsonl --file_splits ../data/SPIN2/test_split_sc.json"
python3 test_redesign.py --seed 111 ${ARGS} > test/sc_full_mpnn_1.log
python3 test_redesign.py --seed 222 ${ARGS} > test/sc_full_mpnn_2.log
python3 test_redesign.py --seed 333 ${ARGS} > test/sc_full_mpnn_3.log
python3 test_redesign.py --seed 444 ${ARGS} > test/sc_full_mpnn_4.log

CHECKPOINT=log/h128_full_ollikainen/best*.pt
ARGS="--cuda --features full --restore ${CHECKPOINT} --file_data ../data/ollikainen/ollikainen_set.jsonl --file_splits ../data/ollikainen/ollikainen_benchmark.json"
python3 test_redesign.py --seed 111 ${ARGS} > test/ollikainen_full_1.log
python3 test_redesign.py --seed 222 ${ARGS} > test/ollikainen_full_2.log
python3 test_redesign.py --seed 333 ${ARGS} > test/ollikainen_full_3.log
python3 test_redesign.py --seed 444 ${ARGS} > test/ollikainen_full_4.log

CHECKPOINT=log/h128_full_mpnn_ollikainen/best*.pt
ARGS="--cuda --features full --mpnn --restore ${CHECKPOINT} --file_data ../data/ollikainen/ollikainen_set.jsonl --file_splits ../data/ollikainen/ollikainen_benchmark.json"
python3 test_redesign.py --seed 111 ${ARGS} > test/ollikainen_full_mpnn_1.log
python3 test_redesign.py --seed 222 ${ARGS} > test/ollikainen_full_mpnn_2.log
python3 test_redesign.py --seed 333 ${ARGS} > test/ollikainen_full_mpnn_3.log
python3 test_redesign.py --seed 444 ${ARGS} > test/ollikainen_full_mpnn_4.log