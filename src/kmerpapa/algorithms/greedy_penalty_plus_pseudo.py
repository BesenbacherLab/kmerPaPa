from kmerpapa.pattern_utils import *
from kmerpapa.CV_tools import make_all_folds, make_all_folds_contextD_kmers
from kmerpapa.score_utils import get_betas
import sys
import numpy
from scipy.special import xlogy, xlog1py
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from numba import njit


'''
Handles cross validation.
Should model-selection be determined based on cross validation?
Do I need backtrack for the Cross-validation?
Maybe i just need CV to estimate best meta-par value
And then return best papa for all data using that meta-par.
'''

# @njit
# def score_folds(trainM, trainU, alphas, betas, penalty):
#     p = (trainM + alphas)/(trainM + trainU + alphas + betas)
#     res = -2*(xlogy(trainM, p) + xlog1py(trainU,-p)) + penalty
#     if res<0:
#         return 1e100
#     return res

@njit
def score_folds(trainM, trainU, alphas, betas, penalty):
    p = (trainM + alphas)/(trainM + trainU + alphas + betas)
    s = penalty
    if trainM > 0:
        s += -2.0 * trainM*np.log(p)
    if trainU > 0:
        s += -2.0 * trainU*np.log(1-p)
    return s

@njit
def test_folds(trainM, trainU, testM, testU, alphas, betas):
    p = (trainM + alphas)/(trainM + trainU + alphas + betas)
    s = 0.0
    if testM > 0:
        s += -2.0 * testM*np.log(p)
    if testU > 0:
        s += -2.0 * testU*np.log(1-p)
    return s

# def test_folds(trainM, trainU, testM, testU, alphas, betas):
#     '''
#     likelihood of test_data
#     '''
#     #p = (trainM + alphas -1)/(trainM + trainU + alphas + betas -2)
#     p = (trainM + alphas)/(trainM + trainU + alphas + betas)
#     return -2 *(xlogy(testM, p) + xlog1py(testU, -p))


def get_score(pattern, contextD, alpha, beta, penalty):
    M,U = get_M_U(pattern, contextD)
    return score_folds(M, U, alpha, beta, penalty)

def get_test_train(pattern, trainD, testD,  alphas, betas):
    M_train,U_train = get_M_U(pattern, trainD)
    M_test, U_test = get_M_U(pattern, testD)
    train_score = test_folds(M_train, U_train, M_train, U_train, alphas, betas)
    test_score = test_folds(M_train, U_train, M_test, U_test, alphas, betas)
    return train_score, test_score

def get_test_logLik(pattern, trainD, testD,  alphas, betas):
    M_train,U_train = get_M_U(pattern, trainD)
    M_test, U_test = get_M_U(pattern, testD)
    test_score = test_folds(M_train, U_train, M_test, U_test, alphas, betas)
    return test_score

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

@njit
def get_M_U_kmer_table(pattern, kmer_table, matches_num):
    M_U = numpy.zeros(2)
    for i in matches_num(pattern):
        M_U += kmer_table[i]
    return M_U

@njit
def get_score_kmer_table(pattern, kmer_table, alpha, beta, penalty, matches_num):
    M_U = get_M_U_kmer_table(pattern, kmer_table, matches_num)
    return score_folds(M_U[0], M_U[1], alpha, beta, penalty)

@njit
def get_test_logLik_kmer_table(pattern, train_kmer_table, test_kmer_table,  alpha, beta, matches_num):
    M_U_train = get_M_U_kmer_table(pattern, train_kmer_table, matches_num)
    M_U_test = get_M_U_kmer_table(pattern, test_kmer_table, matches_num)
    return test_folds(M_U_train[0], M_U_train[1], M_U_test[0], M_U_test[1], alpha, beta)

def greedy_res_kmer_table(pattern, kmer_table, alpha, beta, penalty, matches_num):
    if generality(pattern) == 1:
        return get_score_kmer_table(pattern, kmer_table, alpha, beta, penalty, matches_num), [pattern]
    else:
        best_score = get_score_kmer_table(pattern, kmer_table, alpha, beta, penalty, matches_num)
        best_papa = [pattern]
        for i in range(len(pattern)):#[::-1]:
            if pattern[i] not in complements:
                continue
            for c1, c2 in complements[pattern[i]]:
                pat1 = pattern[:i] + c1 + pattern[i+1:]
                pat2 = pattern[:i] + c2 + pattern[i+1:]

                score1 = get_score_kmer_table(pat1, kmer_table, alpha, beta, penalty, matches_num)
                score2 = get_score_kmer_table(pat2, kmer_table, alpha, beta, penalty, matches_num)

                s = score1 + score2
                if s < best_score:
                    best_score = s
                    best_papa = [pat1, pat2]

        if len(best_papa) > 1:
            pat1, pat2 = best_papa
            s1, papa1 = greedy_res_kmer_table(pat1, kmer_table, alpha, beta, penalty, matches_num)
            s2, papa2 = greedy_res_kmer_table(pat2, kmer_table, alpha, beta, penalty, matches_num)
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
        if args.verbosity > 0:
            print(f'greedy CV iteration {iteration}', file=sys.stderr)
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
                if args.verbosity > 0:
                    print(f'CV on k={len(gen_pat)} alpha={alpha} penalty={penalty} i={iteration} test_LL={sum(test_score)}', file=sys.stderr)
                if args.verbosity > 1:
                    print(f'test LL for each fold: {test_score}', file=sys.stderr)
                train_loss[(a_i, p_i)].extend(train_score)
                test_loss[(a_i, p_i)].extend(test_score)
    best_test_loss = 1e100
    best_values = (None, None)
    for a_i in range(len(alphas)):
        alpha = alphas[a_i]
        for p_i in range(len(penalties)):
            penalty = penalties[p_i]
            test = sum(test_loss[(a_i, p_i)])/nit
            if not args.CVfile is None:    
                print(len(gen_pat), alpha, penalty, test, file = args.CVfile)
            if test < best_test_loss:
                best_values = (alpha, penalty)
                best_test_loss = test
    return best_values[0], best_values[1], best_test_loss


def greedy_partition(gen_pat, contextD, alpha, beta, penalty, args):
    best_score, papa = greedy_res(gen_pat, contextD, alpha, beta, penalty)
    M, U = get_M_U(gen_pat, contextD)
    return best_score, M, U, papa


class CrossValidation:
    def __init__(self, genpat, contextD, nfolds=2, nit=1, seed=None, verbosity=1):
        self.nfolds = nfolds
        self.nit = nit
        self.seed = seed
        self.genpat = genpat
        self.KE = KmerEnumeration(genpat)
        self.npat = generality(genpat)
        self.kmer_table = numpy.zeros((self.npat,2), dtype=numpy.uint64)
        prng = np.random.RandomState(seed)
        self.matches_num = self.KE.get_matches_num()
        for i, context in enumerate(matches(self.genpat)):
            n_mut, n_unmut = contextD[context]
            self.kmer_table[i,0] = n_mut
            self.kmer_table[i,1] = n_unmut
        self.fold_kmer_table = make_all_folds(self.kmer_table, nfolds, nit, prng)


    def loglik(self, alpha, penalty):
        """Calculate test Log Likelihood
        Args:
            penalty (float): complexity penalty
            alpha (float): pseudo count
        Returns:
            Sum of Log Likelihoods of each test fold
        """
        ll_list = []
        for repeat in range(self.nit):
            test_ll = 0.0
            for fold in range(self.nfolds):
                train_kmer_table = self.kmer_table - self.fold_kmer_table[repeat][fold]
                train_MU = train_kmer_table.sum(axis=0)
                beta = get_betas(alpha, train_MU[0],  train_MU[1])
                best_score, papa = greedy_res_kmer_table(self.genpat, train_kmer_table, alpha, beta, penalty, self.matches_num)
                for pattern in papa:
                    this_ll = get_test_logLik_kmer_table(pattern, train_kmer_table, self.fold_kmer_table[repeat][fold], alpha, beta, self.matches_num)
                    test_ll +=  this_ll            
            ll_list.append(test_ll)        
        #print(penalty, alpha, sum(ll_list)/len(ll_list))
        return sum(ll_list)/len(ll_list)

class GridSearchCV(CrossValidation):
    def __init__(self, genpat, contextD, penalties, pseudo_counts, nfolds=2, nit=1, seed=None, verbosity=1):
        super().__init__(genpat, contextD, nfolds=nfolds, nit=nit, seed=seed)
        self.penalties = penalties
        self.pseudo_counts = pseudo_counts

    def get_best_a_c(self):
        best_combo = (None, None)
        best_ll = 1e100
        for a in self.pseudo_counts:
            for c in self.penalties:
                ll = self.loglik(a,c)
                if ll < best_ll:
                    best_ll = ll
                    best_combo = (a,c)
        return best_combo + (best_ll,)
                

class BaysianOptimizationCV(CrossValidation):
    def __init__(self, genpat, contextD, nfolds=2, nit=1, seed=None):
        super().__init__(genpat, contextD, nfolds=nfolds, nit=nit, seed=seed)
        self.search_space = [
            Real(name='pseudo', low=0.01, high=100),
            Real(name='penalty', low=0.1, high=30),
        ]    
        @use_named_args(self.search_space)
        def f(pseudo, penalty):
            return self.loglik(pseudo, penalty)
        self.func = f

    def get_best_a_c(self):
        result = gp_minimize(self.func, self.search_space, n_calls=50)
        #print(result)
        return (result.x[0], result.x[1], result.fun)