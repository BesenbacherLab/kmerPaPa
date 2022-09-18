from kmerpapa.pattern_utils import *
from kmerpapa.CV_tools import make_all_folds
from kmerpapa.score_utils import get_betas, get_betas_kmer_table
import sys
import numpy
from scipy.special import xlogy, xlog1py


def all_kmers(kmer_table, alphas, n_folds, nit, seed, verbosity):
    n_kmer, n_mut_types = kmer_table.size
    test_loss = {a_i:[] for a_i in range(len(alphas))}
    prng = np.random.RandomState(seed)
    for iteration in range(nit):
        folds_kmer_table = make_all_folds(kmer_table, n_folds, 1, prng)[0]
        table_sum_test = folds_kmer_table.sum(axis=0) # counts for each fold
        table_sum_train = sum(table_sum_test) - table_sum_test
        for a_i in range(len(alphas)):            
            alpha = alphas[a_i]
            test_LL = 0
            for i in range(n_folds):
                train_table = kmer_table - folds_kmer_table[0]
                betas = get_betas_kmer_table(alpha, train_table)
                p = (train_table + alpha)/(train_table.sum(axis=1)[:,np.newaxis] + alpha + betas)
                p = p/p.sum(axis=1)[:,np.newaxis]
                test_LL += -2*xlogy(folds_kmer_table[0], p).sum()
            test_loss[a_i].append(test_LL)

    best_test_loss = 1e100
    best_alpha = None

    for a_i in range(len(alphas)):
        alpha = alphas[a_i]
        test = sum(test_loss[a_i])/nit
        if verbosity > 0:
            print(f'alpha={alpha} test_loss={test}', file = sys.stderr)
        if test < best_test_loss:
            best_alpha = alpha
            best_test_loss = test

    return best_alpha, best_test_loss


      
