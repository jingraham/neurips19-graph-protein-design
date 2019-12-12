#!/bin/bash

CHECKPOINT=log/h128_full/best_*.pt
TYPE=full
python3 test_s2s.py --cuda --features ${TYPE} --file_splits ../data/cath/chain_set_splits.json --restore ${CHECKPOINT} > log/test_all_h128_${TYPE}.log
python3 test_s2s.py --cuda --features ${TYPE} --file_splits ../data/SPIN2/test_split_sc.json --restore ${CHECKPOINT} > log/test_sc_h128_${TYPE}.log
python3 test_s2s.py --cuda --features ${TYPE} --file_splits ../data/SPIN2/test_split_L100.json --restore ${CHECKPOINT} > log/test_L100_h128_${TYPE}.log

CHECKPOINT=checkpoints/h128_coarse/best_*.pt
TYPE=coarse
python3 test_s2s.py --cuda --features ${TYPE} --file_splits ../data/cath/chain_set_splits.json --restore ${CHECKPOINT} > log/test_all_h128_${TYPE}.log
python3 test_s2s.py --cuda --features ${TYPE} --file_splits ../data/SPIN2/test_split_sc.json --restore ${CHECKPOINT} > log/test_sc_h128_${TYPE}.log
python3 test_s2s.py --cuda --features ${TYPE} --file_splits ../data/SPIN2/test_split_L100.json --restore ${CHECKPOINT} > log/test_L100_h128_${TYPE}.log

CHECKPOINT=checkpoints/h128_dist/best_*.pt
TYPE=dist
python3 test_s2s.py --cuda --features ${TYPE} --file_splits ../data/cath/chain_set_splits.json --restore ${CHECKPOINT} > log/test_all_h128_${TYPE}.log
python3 test_s2s.py --cuda --features ${TYPE} --file_splits ../data/SPIN2/test_split_sc.json --restore ${CHECKPOINT} > log/test_sc_h128_${TYPE}.log
python3 test_s2s.py --cuda --features ${TYPE} --file_splits ../data/SPIN2/test_split_L100.json --restore ${CHECKPOINT} > log/test_L100_h128_${TYPE}.log

CHECKPOINT=checkpoints/h128_hbonds/best_*.pt
TYPE=hbonds
python3 test_s2s.py --cuda --features ${TYPE} --file_splits ../data/cath/chain_set_splits.json --restore ${CHECKPOINT} > log/test_all_h128_${TYPE}.log
python3 test_s2s.py --cuda --features ${TYPE} --file_splits ../data/SPIN2/test_split_sc.json --restore ${CHECKPOINT} > log/test_sc_h128_${TYPE}.log
python3 test_s2s.py --cuda --features ${TYPE} --file_splits ../data/SPIN2/test_split_L100.json --restore ${CHECKPOINT} > log/test_L100_h128_${TYPE}.log

CHECKPOINT=checkpoints/h128_mpnn/best_*.pt
TYPE=full
python3 test_s2s.py --cuda --features ${TYPE} --mpnn --file_splits ../data/cath/chain_set_splits.json --restore ${CHECKPOINT} > log/test_all_h128_mpnn.log
python3 test_s2s.py --cuda --features ${TYPE} --mpnn --file_splits ../data/SPIN2/test_split_sc.json --restore ${CHECKPOINT} > log/test_sc_h128_mpnn.log
python3 test_s2s.py --cuda --features ${TYPE} --mpnn --file_splits ../data/SPIN2/test_split_L100.json --restore ${CHECKPOINT} > log/test_L100_h128_mpnn.log