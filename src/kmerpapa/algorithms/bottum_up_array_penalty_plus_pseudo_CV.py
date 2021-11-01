from kmerpapa.pattern_utils import *
from kmerpapa.score_utils import get_betas
import kmerpapa.CV_tools
import sys
from numba import njit
import numpy as np
import scipy.special

'''
Handles cross validation. Used to find best meta-parameter values.
Back-track is not implemented for CV. After best meta-parameters are found kmerpapa.bottom_up_array_w_numba 
should be used to train model on all data and to backtrack to find best solution
'''

def score_test_folds(trainM, trainU, testM, testU, alphas, betas, penalty):
    p = (trainM + alphas)/(trainM + trainU + alphas + betas)
    #TODO: log p udregnes to gange er det hurtigere at udregne det selv end at bruge xlogy?
    train_score = -2.0*(scipy.special.xlogy(trainM, p) + scipy.special.xlog1py(trainU,-p)) + penalty
    test_ll =  -2.0*(scipy.special.xlogy(testM, p) + scipy.special.xlog1py(testU, -p))
    return train_score, test_ll

@njit
def get_train(test):
    return test.sum() - test

@njit
def handle_pattern(score_mem, test_score_mem, pattern, M_mem, U_mem, alpha, betas, penalty):
    """Fill in dynamic programming tables for a pattern
    """
    pat_num = 0
    for i in range(gen_pat_len):
        pat_num += perm_code_no_np[gpo[i]][pattern[i]] * cgppl[i]
    res_train = score_mem[pat_num]
    res_test = test_score_mem[pat_num]

    first = True
    for i in range(gen_pat_len):
        if has_complement[pattern[i]]:
            pcn_row = perm_code_no_np[gpo[i]]
            pat_num_diff  = pat_num - pcn_row[pattern[i]] * cgppl[i]
            for j in range(n_complements[pattern[i]]):
                c1 = complements_tab_1[pattern[i]][j]
                c2 = complements_tab_2[pattern[i]][j]
                pat1_num = pat_num_diff + pcn_row[c1] * cgppl[i]
                pat2_num = pat_num_diff + pcn_row[c2] * cgppl[i]
                new_train = score_mem[pat1_num]+score_mem[pat2_num] 
                new_test = test_score_mem[pat1_num] + test_score_mem[pat2_num]
                for k in range(nf):
                    if new_train[k] < res_train[k]:
                        res_train[k] = new_train[k]
                        res_test[k] = new_test[k]
                if first:
                    M_mem[pat_num] = M_mem[pat1_num] + M_mem[pat2_num]
                    U_mem[pat_num] = U_mem[pat1_num] + U_mem[pat2_num]
                    first = False
    M_test = M_mem[pat_num]
    U_test = U_mem[pat_num]
    M_train = M_test.sum() - M_test 
    U_train = U_test.sum() - U_test 
    ps = (M_train + alpha)/(M_train + U_train + alpha + betas)
    logps = np.log(ps)
    log1mps = np.log(1.0-ps)
    for i in range(nf):
        logp = logps[i]
        log1mp = log1mps[i]
        s = penalty
        if M_train[i] > 0:
            s += -2.0 * M_train[i]*logp
        if U_train[i] > 0:
            s += -2.0 * U_train[i]*log1mp
        if s < res_train[i]:
            res_train[i] = s
            s = 0.0
            if M_test[i] > 0:
                s += -2.0 * M_test[i]*logp
            if U_test[i] > 0:
                s += -2.0 * U_test[i]*log1mp
            res_test[i] = s


def pattern_partition_bottom_up(gen_pat, contextD, alphas, args, nmut, nunmut, penalties, index_mut=0):
    global cgppl
    global gppl
    global gpo
    global gen_pat_len
    global has_complement
    global nf
    global gen_pat_num
    ftype = np.float32
    npat = pattern_max(gen_pat)
    nf = args.nfolds
    nit = args.iterations
    test_score_mem = np.empty((npat,nf), dtype=ftype)
    if nmut+nunmut > np.iinfo(np.uint32).max:
        itype = np.uint64
    else:
        itype = np.uint32

    PE = PatternEnumeration(gen_pat)

    U_mem = np.empty((npat,nf), dtype=itype)
    M_mem = np.empty((npat,nf), dtype=itype)
    #U_mem[2,1] indeholder antal test sampels i pat_no 2 foerste fold
    #test kan fåes ved at trække fra sum: sum(U_mem[pat])-U_mem[pat]

    cgppl = np.array(get_cum_genpat_pos_level(gen_pat), dtype=np.uint32)
    gppl = np.array(get_genpat_pos_level(gen_pat), dtype=np.uint32)
    gpo = np.array([ord(x) for x in gen_pat], dtype=np.uint32)

    gen_pat_level = pattern_level(gen_pat)
    gpot = tuple(ord(x) for x in gen_pat)

    test_loss = {(a_i, p_i):[] for a_i in range(len(alphas)) for p_i in range(len(penalties))}
    train_loss = {(a_i, p_i):[] for a_i in range(len(alphas)) for p_i in range(len(penalties))}
    gen_pat_len = len(gen_pat)
    gen_pat_num = PE.pattern2num(gen_pat)

    has_complement = np.full(100,True)
    has_complement[ord('A')] = False
    has_complement[ord('C')] = False
    has_complement[ord('G')] = False
    has_complement[ord('T')] = False

    prng = np.random.RandomState(args.seed)

    # nit is number of itereations if we do repeated CV
    for iteration in range(nit):
        if args.verbosity > 0 and nit > 1:
            print('CV Iteration', iteration, file=sys.stderr)
        kmerpapa.CV_tools.make_all_folds_contextD_patterns(contextD, U_mem, M_mem, gen_pat, prng, itype)
        if args.verbosity > 0:
            print('CV sampling DONE', file=sys.stderr)
    
        M_sum_test = M_mem.sum(axis=0) # n_mut for each fold
        U_sum_test = U_mem.sum(axis=0) # n_unmut for each fold 
        M_sum_train = M_sum_test.sum() - M_sum_test
        U_sum_train = U_sum_test.sum() - U_sum_test

        for a_i in range(len(alphas)):
            alpha = alphas[a_i]
            betas = get_betas(alpha, M_sum_train, U_sum_train)
            for p_i in range(len(penalties)):
                score_mem = np.full((npat,nf), 1e100, dtype=ftype)
                penalty = penalties[p_i]
                for pattern in subpatterns_level(gen_pat, 0):
                    pat_num = PE.pattern2num(pattern)
                    M_test = M_mem[pat_num]
                    U_test = U_mem[pat_num]
                    M_train = get_train(M_test)
                    U_train = get_train(U_test)
                    score_mem[pat_num], test_score_mem[pat_num] = score_test_folds(M_train, U_train, M_test, U_test, alpha, betas, penalty)

                for level in range(1, gen_pat_level+1):
                    if args.verbosity > 1:    
                        print(f'level {level} of {gen_pat_level}', file=sys.stderr)
                    for pattern in subpatterns_level_ord_np(gpot, gen_pat_level, level):
                        handle_pattern(score_mem, test_score_mem, pattern, M_mem, U_mem, alpha, betas, penalty)
                if args.verbosity > 0:
                    print(f'CV on k={len(gen_pat)} alpha={alpha} penalty={penalty} i={iteration} test_LL={sum(test_score_mem[gen_pat_num])}', file=sys.stderr)
                if args.verbosity > 1:
                    print(f'test LL for each fold: {test_score_mem[gen_pat_num]}', file=sys.stderr)
                train_loss[(a_i, p_i)].extend(list(score_mem[gen_pat_num]))
                test_loss[(a_i, p_i)].extend(list(test_score_mem[gen_pat_num]))

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

