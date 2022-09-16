import numpy as np
from kmerpapa.CV_tools import make_all_folds, make_all_folds_contextD_kmers, make_all_folds_contextD_patterns
from kmerpapa.pattern_utils import pattern_max, PatternEnumeration

def test_make_all_folds():
    kmer_table = np.array([[1,100,200],[10,1000,2000]])
    nf = 10
    prng = np.random.RandomState(0)
    table_w_folds = make_all_folds(kmer_table, nf, 1, prng)
    assert(table_w_folds.shape == (1, 10, 2, 3))
    assert(np.all(table_w_folds.sum(axis=(0,1)) == kmer_table))


def test_make_all_folds_contextD_patterns():
    contextD = {'AAA':(10,100), 'CAA': (200,1000), 'GAA':(500,2000), 'TAA':(300,1000)}
    general_pattern = 'NAA'
    npat = pattern_max(general_pattern)
    PE = PatternEnumeration(general_pattern)
    nf = 10
    U_mem = np.zeros((npat,nf), dtype=np.uint64)
    M_mem = np.zeros((npat,nf), dtype=np.uint64)
    prng = np.random.RandomState(0)
    make_all_folds_contextD_patterns(contextD, U_mem, M_mem, general_pattern, prng)
    for i in range(5):
        pattern = PE.num2pattern(i)
        assert(i == PE.pattern2num(pattern))
        if pattern not in contextD:
            count = 0
        else:
            count = contextD[pattern][1]
        assert(U_mem[i].sum() == count)
    for i in range(5):
        pattern = PE.num2pattern(i)
        assert(i == PE.pattern2num(pattern))
        if pattern not in contextD:
            count = 0
        else:
            count = contextD[pattern][0]
        assert(M_mem[i].sum() == count)
