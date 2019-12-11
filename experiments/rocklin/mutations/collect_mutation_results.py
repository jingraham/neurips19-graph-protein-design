import glob
import pandas as pd

import numpy as np
import scipy.stats

results_files = glob.glob('*.tsv')

print(results_files)

for suffix in ['_full', '_hbonds']:
    for file in sorted(glob.glob('*' + suffix + '.tsv')):
        df = pd.read_csv(file, sep='\t')
        pearson = np.corrcoef(df['stabilityscore'], df['neglogp'])[0,1]
        spearman = scipy.stats.spearmanr(df['stabilityscore'], df['neglogp'])[0]
        print('{}\t{}\t{}'.format(file, spearman, pearson))


    # print(df.corr()[2,1])
    # print(df)
    # .corr()

# for file in glob.glob('*_hbonds.tsv'):
#     df = pd.read_csv(file, sep='\t')
#     pearson = np.corrcoef(df['stabilityscore'], df['neglogp'])[0,1]
#     spearman = scipy.stats.spearmanr(df['stabilityscore'], df['neglogp'])
#     print(file, pearson, spearman)

# ['HHH_rd2_0134.pdb_full.tsv', 'HEEH_rd3_0872.pdb_full.tsv', 'EEHEE_rd3_0037.pdb_full.tsv', 'HEEH_rd3_0726.pdb_full.tsv', 'HHH_rd3_0138.pdb_full.tsv', 'HEEH_rd2_0779.pdb_full.tsv', 'EEHEE_rd3_1702.pdb_full.tsv', 'EEHEE_rd3_1716.pdb_full.tsv', 'HEEH_rd3_0223.pdb_full.tsv', 'EEHEE_rd3_1498.pdb_full.tsv']
# EEHEE_rd3_0037.pdb_full.tsv -0.4740502402013245
# EEHEE_rd3_1498.pdb_full.tsv -0.44535192689340797
# EEHEE_rd3_1702.pdb_full.tsv -0.11836715933901011
# EEHEE_rd3_1716.pdb_full.tsv -0.47101633147337535
# HEEH_rd2_0779.pdb_full.tsv  -0.5690828207436367

# HEEH_rd3_0223.pdb_full.tsv  -0.3627209793518538
# HEEH_rd3_0726.pdb_full.tsv  -0.11399479742582153
# HEEH_rd3_0872.pdb_full.tsv  -0.20699664141096993
# HHH_rd2_0134.pdb_full.tsv   -0.23959563270818668
# HHH_rd3_0138.pdb_full.tsv   -0.3287112683057721