import numpy as np
from kmerpapa.CV_tools import make_all_folds, make_all_folds_contextD_kmers, make_all_folds_contextD_patterns
from kmerpapa.pattern_utils import pattern_max, PatternEnumeration

def test_make_all_folds():
    n_mut = np.array([1,100,200])# counts of unmutated sites for each of npat
    n_unmut = np.array([10,1000,2000]) # counts of unmutated sites for each of npat
    npat = len(n_mut)
    nf = 10
    U_mem = np.zeros((npat,nf), dtype=np.uint64)
    M_mem = np.zeros((npat,nf), dtype=np.uint64)
    make_all_folds(n_mut, n_unmut, U_mem, M_mem)
    for i in range(3):
        assert(U_mem[i].sum() == n_unmut[i])
    for i in range(3):
        assert(M_mem[i].sum() == n_mut[i])

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
