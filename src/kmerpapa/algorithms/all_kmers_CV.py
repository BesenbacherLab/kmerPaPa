from kmerpapa.pattern_utils import *
from kmerpapa.CV_tools import make_all_folds_contextD_kmers
from kmerpapa.score_utils import get_betas
import sys
import numpy
from scipy.special import xlogy, xlog1py

def test_folds(trainM, trainU, testM, testU, alphas, betas):
    '''
    likelihood of test_data on training set
    '''
    p = (trainM + alphas)/(trainM + trainU + alphas + betas)
    return -2 *(xlogy(testM, p) + xlog1py(testU, -p))

def all_kmers(gen_pat, contextD, alphas, args, nmut, nunmut, index_mut=0):
    nf = args.nfolds
    nit = args.iterations
    npat = generality(gen_pat)
    U_mem = numpy.zeros((npat,nf), dtype=numpy.uint64)
    M_mem = numpy.zeros((npat,nf), dtype=numpy.uint64)
    M_sum_test = numpy.zeros(nf)
    U_sum_test = numpy.zeros(nf)
    test_loss = {a_i:[] for a_i in range(len(alphas))}
    train_loss = {a_i:[] for a_i in range(len(alphas))}
    prng = np.random.RandomState(args.seed)
    for iteration in range(nit):
        make_all_folds_contextD_kmers(contextD, U_mem, M_mem, gen_pat, prng)
        M_sum_test = M_mem.sum(axis=0) # n_mut for each fold
        U_sum_test = U_mem.sum(axis=0) # n_unmut for each fold 
        M_sum_train = sum(M_sum_test) - M_sum_test
        U_sum_train = sum(U_sum_test) - U_sum_test

        for a_i in range(len(alphas)):
            alpha = alphas[a_i]
            betas = get_betas(alpha, M_sum_train, U_sum_train)
            sum_test = numpy.zeros(nf)
            sum_train = numpy.zeros(nf)
            for i, context in enumerate(matches(gen_pat)):
                M_test = M_mem[i]
                U_test = U_mem[i]
                M_train = sum(M_test) - M_test
                U_train = sum(U_test) - U_test
                sum_train += test_folds(M_train, U_train, M_train, U_train, alpha, betas)
                sum_test += test_folds(M_train, U_train, M_test, U_test, alpha, betas)
            train_loss[a_i].extend(list(sum_train))
            test_loss[a_i].extend(list(sum_test))

    best_test_loss = 1e100
    best_alpha = None

    for a_i in range(len(alphas)):
        alpha = alphas[a_i]
        test = sum(test_loss[a_i])/nit
        if args.verbosity > 0:
            print(f'alpha={alpha} test_loss={test}', file = sys.stderr)
        if test < best_test_loss:
            best_alpha = alpha
            best_test_loss = test

    return best_alpha, best_test_loss

            
      
