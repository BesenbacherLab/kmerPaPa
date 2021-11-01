from kmerpapa.pattern_utils import *
from kmerpapa.CV_tools import make_all_folds_contextD_kmers
from kmerpapa.score_utils import get_betas
import sys
import numpy
from scipy.special import xlogy, xlog1py
#from numba import jit

'''
Handles cross validation.
Should model-selection be determined based on cross validation?
Do I need backtrack for the Cross-validation?
Maybe i just need CV to estimate best meta-par value
And then return best papa for all data using that meta-par.
'''

def score_folds(trainM, trainU, alphas, betas, penalty):
    p = (trainM + alphas)/(trainM + trainU + alphas + betas)
    res = -2*(xlogy(trainM, p) + xlog1py(trainU,-p)) + penalty
    if res<0:
        return 1e100
    return res

#@jit
def test_folds(trainM, trainU, testM, testU, alphas, betas):
    '''
    likelihood of test_data on training set
    Burde jeg lave Binomial deviance?
    '''
    #p = (trainM + alphas -1)/(trainM + trainU + alphas + betas -2)
    p = (trainM + alphas)/(trainM + trainU + alphas + betas)
    return -2 *(xlogy(testM, p) + xlog1py(testU, -p))


def get_score(pattern, contextD, alpha, beta, penalty):
    M,U = get_M_U(pattern, contextD)
    return score_folds(M, U, alpha, beta, penalty)

def get_test_train(pattern, trainD, testD,  alphas, betas):
    M_train,U_train = get_M_U(pattern, trainD)
    M_test, U_test = get_M_U(pattern, testD)
    train_score = test_folds(M_train, U_train, M_train, U_train, alphas, betas)
    test_score = test_folds(M_train, U_train, M_test, U_test, alphas, betas)
    return train_score, test_score

def greedy_res(pattern, contextD, alphas, betas, penalty):
    if pattern in contextD:
        return get_score(pattern, contextD, alphas, betas, penalty), [pattern]
    else:
        best_score = get_score(pattern, contextD, alphas, betas, penalty)
        best_papa = [pattern]
        for i in range(len(pattern)):#[::-1]:
            if pattern[i] not in complements:
                continue
            for c1, c2 in complements[pattern[i]]:
                pat1 = pattern[:i] + c1 + pattern[i+1:]
                pat2 = pattern[:i] + c2 + pattern[i+1:]

                score1 = get_score(pat1, contextD, alphas, betas, penalty)
                score2 = get_score(pat2, contextD, alphas, betas, penalty)

                s = score1 + score2
                if s < best_score:
                    best_score = s
                    best_papa = [pat1, pat2]

        if len(best_papa) > 1:
            pat1, pat2 = best_papa
            #print('split', pat1, pat2)
            s1, papa1 = greedy_res(pat1, contextD, alphas, betas, penalty)
            s2, papa2 = greedy_res(pat2, contextD, alphas, betas, penalty)
            return s1+s2, papa1+papa2
        else:
            return best_score, best_papa

def greedy_partition_CV(gen_pat, contextD, alphas, args, nmut, nunmut, penalties, index_mut=0):
    resD = {}
    nf = args.nfolds
    nit = args.iterations
    npat = generality(gen_pat)
    U_mem = numpy.zeros((npat,nf), dtype=numpy.uint64)
    M_mem = numpy.zeros((npat,nf), dtype=numpy.uint64)
    M_sum_test = numpy.zeros(nf)
    U_sum_test = numpy.zeros(nf)
    test_loss = {(a_i, p_i):[] for a_i in range(len(alphas)) for p_i in range(len(penalties))}
    train_loss = {(a_i, p_i):[] for a_i in range(len(alphas)) for p_i in range(len(penalties))}
    #{(a_i, p_i):[] for a in alphas for p in penalties}
    prng = np.random.RandomState(args.seed)
    for iteration in range(nit):
        if args.verbose:
            print(f'CV iteration {iteration}', file=sys.stderr)
        make_all_folds_contextD_kmers(contextD, U_mem, M_mem, gen_pat, prng)
        M_sum_test = M_mem.sum(axis=0) # n_mut for each fold
        U_sum_test = U_mem.sum(axis=0) # n_unmut for each fold 
        M_sum_train = sum(M_sum_test) - M_sum_test
        U_sum_train = sum(U_sum_test) - U_sum_test
        p_all = M_sum_test/(M_sum_test + U_sum_test)
        LL_intercept = -2*(xlogy(M_sum_test, p_all) + xlog1py(U_sum_test, -p_all))

        trainDs = []
        testDs = []
        for fold in range(nf):
            trainDs.append({})
            testDs.append({})

        for i, context in enumerate(matches(gen_pat)):
            M_test = M_mem[i]
            U_test = U_mem[i]
            M_train = sum(M_test) - M_test
            U_train = sum(U_test) - U_test
            for fold in range(nf):
                trainDs[fold][context] = (M_train[fold], U_train[fold])
                testDs[fold][context] = (M_test[fold], U_test[fold])

        for a_i in range(len(alphas)):
            alpha = alphas[a_i]
            for p_i in range(len(penalties)):
                penalty = penalties[p_i]
                betas = get_betas(alpha, M_sum_train, U_sum_train)
                test_score = [0] * nf
                train_score = [0] * nf
                for fold in range(nf):
                    best_score, papa = greedy_res(gen_pat, trainDs[fold], alpha, betas[fold], penalty)
                    for pattern in papa:
                        train_s, test_s = get_test_train(pattern, trainDs[fold], testDs[fold],  alpha, betas[fold])
                        train_score[fold] += train_s
                        test_score[fold] += test_s
                train_loss[(a_i, p_i)].extend(train_score)
                test_loss[(a_i, p_i)].extend(test_score)
    best_test_loss = 1e100
    best_values = (None, None)
    for a_i in range(len(alphas)):
        alpha = alphas[a_i]
        for p_i in range(len(penalties)):
            penalty = penalties[p_i]
            test = sum(test_loss[(a_i, p_i)])/nit
            if args.verbose:
                print(alpha, penalty, test, file = sys.stderr)
            if test < best_test_loss:
                best_values = (alpha, penalty)
                best_test_loss = test
    return best_values[0], best_values[1], best_test_loss

def greedy_partition(gen_pat, contextD, alpha, beta, penalty, args):
    best_score, papa = greedy_res(gen_pat, contextD, alpha, beta, penalty)
    M, U = get_M_U(gen_pat, contextD)
    return best_score, M, U, papa

