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

def backtrack(pattern, backtrack_mem, pattern2num, PE):
    pat_num = pattern2num(pattern)
    left_num = backtrack_mem[pat_num]
    if left_num == pat_num:
        return [pattern]
    left_pat = PE.num2pattern(left_num)
    right_pat = get_right(pattern, left_pat)
    return backtrack(left_pat, backtrack_mem, pattern2num, PE) + backtrack(right_pat, backtrack_mem, pattern2num, PE)

def score(M, U):
    p = (M + alpha)/(M + U + alpha + beta)
    res = -2*(xlogy(M, p) + xlog1py(U,-p)) + penalty
    return res

def score_array(A):
    p = (A + alpha)/(A.sum() + alpha + beta)
    p = p/p.sum()
    res = -2*xlogy(A, p).sum()  + penalty
    return res

@njit
def score_array_pair(A, alpha, my):
    _two, n_types  = A.shape
    s = 0.0
    for i in range(n_types):
        M = A[0][i]
        U = A[1][i]
        p = (M + alpha)/(M + U + (alpha/my[i]))
        if M > 0:
            s += M*np.log(p)
        if U > 0:
            s += U*np.log(1.0-p)
    s = penalty -2.0 *s
    return s

@njit
def score_array_3D(A, alpha, my):
    n_bintypes, n_muttypes  = A.shape
    s = penalty

    for i in range(n_bintypes):
        p = (A[i] + alpha)/(A[i].sum() + alpha/my[i])
        p = p/p.sum()
        assert len(p) == n_muttypes
        for j in range(len(p)):
            if A[i][j] > 0:
                s += -2.0 * A[i][j]*np.log(p[j])
    return s


@njit
def rate_array_pair(A, alpha, my):
    _two, n_types  = A.shape
    s = 0.0
    r = np.empty(n_types, np.float32)
    for i in range(n_types):
        M = A[0][i]
        U = A[1][i]
        r[i] = (M + alpha)/(M + U + (alpha/my[i]))
    return r

@njit
def rate_array_3D(A, alpha, my):
    n_bintypes, n_muttypes  = A.shape
    s = penalty
    r = np.empty((n_bintypes, n_muttypes), np.float32)
    for i in range(n_bintypes):
        r[i] = (A[i] + alpha)/(A[i].sum() + alpha/my[i])
        r[i] = r[i]/r[i].sum()
    return r


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


@njit
def handle_pattern_multi(pattern, score_mem, backtrack_mem, pattern_table):
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
                new_score = score_mem[pat1_num] + score_mem[pat2_num] 
                if new_score < score_mem[pat_num]:
                    score_mem[pat_num] = new_score
                    backtrack_mem[pat_num] = pat1_num
                if first:
                    pattern_table[pat_num] = pattern_table[pat1_num] + pattern_table[pat2_num]
                    first = False
    T = pattern_table[pat_num]
    p = (T + alpha)/(T.sum() + alpha + beta)
    p = p/p.sum()
    #s = penalty -2.0 * xlogy(T, p).sum()
    s = penalty
    #s += np.sum(-2.0*T*np.log(p))
    for i in range(len(p)):
        if T[i] > 0:
            s += -2.0 * T[i]*np.log(p[i])
    
    if s < score_mem[pat_num]:
        score_mem[pat_num] = s
        backtrack_mem[pat_num] = pat_num


def pattern_partition_bottom_up_kmer_table(KE, kmer_table, alpha_, beta_, penalty_, verbosity):
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
    gen_pat = KE.genpat
    npat = pattern_max(gen_pat)
    score_mem = np.full(npat, 1e100, dtype=ftype)

    if kmer_table.sum() > np.iinfo(np.uint32).max:
       itype = np.uint64
    else:
       itype = np.uint32

    n_kmers, n_types = kmer_table.shape    
    pattern_table = np.empty((npat, n_types), dtype=itype)
    backtrack_mem = np.empty(npat, dtype=itype)

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

    kmer2num  = KE.get_kmer2num()
    pattern2num = PE.get_pattern2num()
    for pattern in subpatterns_level(gen_pat, 0):
        #pe_pat_num = PE.pattern2num(pattern)
        pe_pat_num = pattern2num(pattern)
        ke_pat_num = kmer2num(pattern)
        pattern_table[pe_pat_num] = kmer_table[ke_pat_num]
        score_mem[pe_pat_num] = score_array(pattern_table[pe_pat_num])
        backtrack_mem[pe_pat_num] = pe_pat_num

    for level in range(1, gen_pat_level+1):
        if verbosity > 1:
            print(f'level {level} of {gen_pat_level}', file=sys.stderr)
        for pattern in subpatterns_level_ord_np(gpot, gen_pat_level, level):
            handle_pattern_multi(pattern, score_mem, backtrack_mem, pattern_table)
    
    pat_num = pattern2num(gen_pat)
    names = backtrack(gen_pat, backtrack_mem, pattern2num, PE)
    assert(all(pattern_table[pat_num] == kmer_table.sum(axis=0)))
    counts = np.empty((len(names), n_types), itype)
    for i in range(len(names)):
        counts[i] = pattern_table[pattern2num(names[i])]
    return score_mem[pat_num], names, counts



@njit
def handle_pattern_multi_pair(pattern, score_mem, backtrack_mem, pattern_table):
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
                new_score = score_mem[pat1_num] + score_mem[pat2_num] 
                if new_score < score_mem[pat_num]:
                    score_mem[pat_num] = new_score
                    backtrack_mem[pat_num] = pat1_num
                if first:
                    pattern_table[pat_num] = pattern_table[pat1_num] + pattern_table[pat2_num]
                    first = False
    T = pattern_table[pat_num]
    _two, n_types  = T.shape
    s = 0.0
    for i in range(n_types):
        M = T[0][i]
        U = T[1][i]
        p = (M + alpha)/(M + U + (alpha/my[i]))
        if M > 0:
            s += M*np.log(p)
        if U > 0:
            s += U*np.log(1.0-p)
    s = penalty -2.0 *s
    if s < score_mem[pat_num]:
        score_mem[pat_num] = s
        backtrack_mem[pat_num] = pat_num


def pattern_partition_bottom_up_kmer_table_pair(KE, kmer_table, alpha_, penalty_, verbosity):
    global cgppl
    global gppl
    global gpo
    global gen_pat_len
    global has_complement
    global alpha
    global my
    global penalty
    alpha = alpha_
    #beta = beta_.sum(axis=0)
    penalty = penalty_
    ftype = np.float32
    gen_pat = KE.genpat
    npat = pattern_max(gen_pat)
    score_mem = np.full(npat, 1e100, dtype=ftype)

    if kmer_table.sum() > np.iinfo(np.uint32).max:
       itype = np.uint64
    else:
       itype = np.uint32

    n_kmers, _two , n_types = kmer_table.shape    
    assert _two == 2
    pattern_table = np.empty((npat, 2, n_types), dtype=itype)
    backtrack_mem = np.empty(npat, dtype=itype)

    sums = kmer_table.sum(axis=0)
    my = sums[0]/(sums[0]+sums[1])

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

    kmer2num  = KE.get_kmer2num()
    pattern2num = PE.get_pattern2num()
    for pattern in subpatterns_level(gen_pat, 0):
        #pe_pat_num = PE.pattern2num(pattern)
        pe_pat_num = pattern2num(pattern)
        ke_pat_num = kmer2num(pattern)
        pattern_table[pe_pat_num] = kmer_table[ke_pat_num]
        score_mem[pe_pat_num] = score_array_pair(pattern_table[pe_pat_num], alpha, my)
        backtrack_mem[pe_pat_num] = pe_pat_num

    for level in range(1, gen_pat_level+1):
        if verbosity > 1:
            print(f'level {level} of {gen_pat_level}', file=sys.stderr)
        for pattern in subpatterns_level_ord_np(gpot, gen_pat_level, level):
            handle_pattern_multi_pair(pattern, score_mem, backtrack_mem, pattern_table)
    
    pat_num = pattern2num(gen_pat)
    names = backtrack(gen_pat, backtrack_mem, pattern2num, PE)
    assert(all(pattern_table[pat_num][0] == kmer_table.sum(axis=0)[0]))
    assert(all(pattern_table[pat_num][1] == kmer_table.sum(axis=0)[1]))
    counts = np.empty((len(names), 2, n_types), itype)
    rates = np.empty((len(names), n_types), np.float32)
    for i in range(len(names)):
        counts[i] = pattern_table[pattern2num(names[i])]
        rates[i] = rate_array_pair(counts[i], alpha, my)
    return score_mem[pat_num], names, counts, rates




@njit
def handle_pattern_multi_multi(pattern, score_mem, backtrack_mem, pattern_table):
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
                new_score = score_mem[pat1_num] + score_mem[pat2_num] 
                if new_score < score_mem[pat_num]:
                    score_mem[pat_num] = new_score
                    backtrack_mem[pat_num] = pat1_num
                if first:
                    pattern_table[pat_num] = pattern_table[pat1_num] + pattern_table[pat2_num]
                    first = False
    
    #print(pattern_table.shape)
    #T = pattern_table[pat_num]
    #print(T.shape)
    #print([0][0])
    n_pat, n_bintypes, n_muttypes = pattern_table.shape
    #n_muttypes, n_bintypes  = T.shape
    s = penalty

    for i in range(n_bintypes):
        T = pattern_table[pat_num][i]
        p = (T + alpha)/(T.sum() + alpha/my[i])
        p = p/p.sum()
        assert(len(p)==n_muttypes)
        #s += np.sum(-2.0*T[i]*np.log(p))
        for j in range(n_muttypes):
            if T[j] > 0:
                s += -2.0 * T[j]*np.log(p[j])

    if s < score_mem[pat_num]:
        score_mem[pat_num] = s
        backtrack_mem[pat_num] = pat_num


def pattern_partition_bottom_up_kmer_table_multi(KE, kmer_table, alpha_, penalty_, verbosity):
    global cgppl
    global gppl
    global gpo
    global gen_pat_len
    global has_complement
    global alpha
    global my
    global penalty
    alpha = alpha_
    #beta = beta_.sum(axis=0)
    penalty = penalty_
    ftype = np.float32
    gen_pat = KE.genpat
    npat = pattern_max(gen_pat)
    score_mem = np.full(npat, 1e100, dtype=ftype)

    if kmer_table.sum() > np.iinfo(np.uint32).max:
       itype = np.uint64
    else:
       itype = np.uint32

    # TODO input reader skal skrives om saa rækkefølgen her passer!!
    n_kmers, n_bins, n_types = kmer_table.shape    
    
    pattern_table = np.empty((npat, n_bins, n_types), dtype=itype)
    backtrack_mem = np.empty(npat, dtype=itype)

    #sums = kmer_table.sum(axis=0)
    #my = sums[0]/(sums[0]+sums[1])
    ## Antager at jeg skal have my for hver bin?
    my = (kmer_table.sum(axis=(0)).T/kmer_table.sum(axis=(0,2))).T
    # hvis fælles my for alle bins:
    #my = kmer_table.sum(axis=(0,1))/kmer_table.sum(axis=(0,1,2))

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

    kmer2num  = KE.get_kmer2num()
    pattern2num = PE.get_pattern2num()
    for pattern in subpatterns_level(gen_pat, 0):
        #pe_pat_num = PE.pattern2num(pattern)
        pe_pat_num = pattern2num(pattern)
        ke_pat_num = kmer2num(pattern)
        pattern_table[pe_pat_num] = kmer_table[ke_pat_num]
        # TODO: lav score_array_multi_multi
        score_mem[pe_pat_num] = score_array_3D(pattern_table[pe_pat_num], alpha, my)
        backtrack_mem[pe_pat_num] = pe_pat_num

    for level in range(1, gen_pat_level+1):
        if verbosity > 1:
            print(f'level {level} of {gen_pat_level}', file=sys.stderr)
        for pattern in subpatterns_level_ord_np(gpot, gen_pat_level, level):
            handle_pattern_multi_multi(pattern, score_mem, backtrack_mem, pattern_table)
    
    pat_num = pattern2num(gen_pat)
    names = backtrack(gen_pat, backtrack_mem, pattern2num, PE)
    assert(all(pattern_table[pat_num][0] == kmer_table.sum(axis=0)[0]))
    assert(all(pattern_table[pat_num][1] == kmer_table.sum(axis=0)[1]))
    counts = np.empty((len(names), n_bins, n_types), itype)
    rates = np.empty((len(names), n_bins, n_types), np.float32)
    for i in range(len(names)):
        counts[i] = pattern_table[pattern2num(names[i])]
        rates[i] = rate_array_3D(counts[i], alpha, my)
    return score_mem[pat_num], names, counts, rates

