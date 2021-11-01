import numpy as np
from kmerpapa.CV_tools import make_all_folds, make_all_folds_contextD
from kmerpapa.pattern_utils import num2pattern, pattern_max, pattern2num

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

def test_make_all_folds_context():
    contextD = {'AAA':(10,100), 'CAA': (200,1000), 'GAA':(500,2000), 'TAA':(300,1000)}
    general_pattern = 'NAA'
    npat = pattern_max(general_pattern)
    nf = 10
    U_mem = np.zeros((npat,nf), dtype=np.uint64)
    M_mem = np.zeros((npat,nf), dtype=np.uint64)
    prng = np.random.RandomState(0)
    make_all_folds_contextD(contextD, U_mem, M_mem, general_pattern, prng)
    for i in range(5):
        pattern = num2pattern(general_pattern, i)
        assert(i == pattern2num(general_pattern, pattern))
        if pattern not in contextD:
            count = 0
        else:
            count = contextD[pattern][1]
        assert(U_mem[i].sum() == count)
    for i in range(5):
        pattern = num2pattern(general_pattern, i)
        assert(i == pattern2num(general_pattern, pattern))
        if pattern not in contextD:
            count = 0
        else:
            count = contextD[pattern][0]
        assert(M_mem[i].sum() == count)
