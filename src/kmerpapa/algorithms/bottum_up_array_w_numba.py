from kmerpapa.pattern_utils import *
import sys
from numba import njit
import math
import numpy as np
from scipy.special import xlogy, xlog1py

def get_right(super_pat, left):
    right = ''
    for i in range(len(super_pat)):
        if super_pat[i] == left[i]:
            right += super_pat[i]
        else:
            right += minus_set[super_pat[i]][left[i]]
    return right

def backtrack(pattern, backtrack_mem, PE):
    pat_num = PE.pattern2num(pattern)
    left_num = backtrack_mem[pat_num]
    if left_num == pat_num:
        return [pattern]
    left_pat = PE.num2pattern(left_num)
    right_pat = get_right(pattern, left_pat)
    return backtrack(left_pat, backtrack_mem, PE) + backtrack(right_pat, backtrack_mem, PE)

def score(M, U):
    p = (M + alpha)/(M + U + alpha + beta)
    res = -2*(xlogy(M, p) + xlog1py(U,-p)) + penalty
    return res

@njit
def handle_pattern(pattern, score_mem, backtrack_mem, M_mem, U_mem):
    pat_num = 0
    for i in range(gen_pat_len):
        pat_num += perm_code_no_np[gpo[i]][pattern[i]] * cgppl[i]
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
                new_score = score_mem[pat1_num]+score_mem[pat2_num] 
                if new_score < score_mem[pat_num]:
                    score_mem[pat_num] = new_score
                    backtrack_mem[pat_num] = pat1_num
                if first:
                    M_mem[pat_num] = M_mem[pat1_num] + M_mem[pat2_num]
                    U_mem[pat_num] = U_mem[pat1_num] + U_mem[pat2_num]
                    first = False
    M = M_mem[pat_num]
    U = U_mem[pat_num]
    p = (M + alpha)/(M + U + alpha + beta)
    s = penalty
    if M > 0:
        s += -2.0 * M*math.log(p)
    if U > 0:
        s += -2.0 * U*math.log(1.0-p)
    if s < score_mem[pat_num]:
        score_mem[pat_num] = s
        backtrack_mem[pat_num] = pat_num


def pattern_partition_bottom_up(gen_pat, contextD, alpha_, beta_, penalty_, args, nmut, nunmut, index_mut=0):
    global cgppl
    global gppl
    global gpo
    global gen_pat_len
    global has_complement
    global alpha
    global beta
    global penalty
    alpha = alpha_
    beta = beta_
    penalty = penalty_
    ftype = np.float32
    npat = pattern_max(gen_pat)
    score_mem = np.full(npat, 1e100, dtype=ftype)
    if nmut+nunmut > np.iinfo(np.uint32).max:
        itype = np.uint64
    else:
        itype = np.uint32
    U_mem = np.empty(npat, dtype=itype)
    M_mem = np.empty(npat, dtype=itype)
    backtrack_mem = np.empty(npat, dtype=np.uint64)

    PE = PatternEnumeration(gen_pat)

    cgppl = np.array(get_cum_genpat_pos_level(gen_pat), dtype=np.uint32)
    gppl = np.array(get_genpat_pos_level(gen_pat), dtype=np.uint32)
    gpo = np.array([ord(x) for x in gen_pat], dtype=np.uint32)
    gen_pat_level = pattern_level(gen_pat)
    gpot = tuple(ord(x) for x in gen_pat)

    gen_pat_len = len(gen_pat)

    has_complement = np.full(100,True)
    has_complement[ord('A')] = False
    has_complement[ord('C')] = False
    has_complement[ord('G')] = False
    has_complement[ord('T')] = False

    for pattern in subpatterns_level(gen_pat, 0):
        pat_num = PE.pattern2num(pattern)
        tup = contextD[pattern]
        nm = tup[index_mut]
        nu = tup[-1]
        M_mem[pat_num] = nm
        U_mem[pat_num] = nu
        score_mem[pat_num] = score(nm, nu)
        backtrack_mem[pat_num] = pat_num

    for level in range(1, gen_pat_level+1):
        if args.verbosity > 1:
            print(f'level {level} of {gen_pat_level}', file=sys.stderr)
        for pattern in subpatterns_level_ord_np(gpot, gen_pat_level, level):
            handle_pattern(pattern, score_mem, backtrack_mem, M_mem, U_mem)
    
    pat_num = PE.pattern2num(gen_pat)
    names = backtrack(gen_pat, backtrack_mem, PE)
    return score_mem[pat_num], M_mem[pat_num], U_mem[pat_num], names


