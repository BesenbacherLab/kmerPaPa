import numpy as np
import sys
from kmerpapa.pattern_utils import generality, KmerEnumeration
from kmerpapa.pattern_utils import LCA_pattern_of_kmers, inv_code
from kmerpapa.papa import Pattern

nucleotides = ['A','C','G','T']

def peek_first(f):
    """
    Return the first line and a generator that yields all lines including the first.
    """
    first_line = f.readline()
    def line_generator():
        yield first_line
        for line in f:
            yield line
    return first_line, line_generator


def read_joint_kmer_counts_table_no_sp(f, last_background = False,  dtype=np.uint32, n_scale=1):
    """Read kmer counts from a file with kmer counts.
    First columns is kmer as string. Next columns are counts.
    Should be at least two columns with counts.

    Args:
        f (file object): input file
        general_pattern (str): General Pattern
        last_background: Is the last column backround counts (includes counts from other columns)?
        n_scale (int, optional): Option to scale the background counts. Defaults to 1. Only works if last_background==True

    Returns:
        numpy.array with counts for each kmer and KmerEnumeration
    """
    line, line_generator = peek_first(f)
    n_cols = len(line.split())-1
    k = len(line.split()[0])
    lines = []
    sets = [set() for x in range(k)]
    for line in line_generator():
        kmer, *counts = line.split()
        kmer = kmer.upper()
        assert all(n in nucleotides for n in kmer)
        for i in range(k):
            sets[i].add(kmer[i])
        lines.append((kmer, counts))
    pat_L = []
    for s in sets:
        pat_L.append(inv_code[frozenset(s)])
    general_pattern =  ''.join(pat_L)
    max_val = np.iinfo(dtype).max
    n_rows = generality(general_pattern)
    kmer_table = np.zeros((n_rows, n_cols), dtype)
    KE = KmerEnumeration(general_pattern)
    kmer2num = KE.get_kmer2num()
    for kmer, counts in lines:
        try:
            counts = [int(x) for x in counts]
        except:
            #Floating points will be rounded to ints
            counts = [int(float(x)) for x in counts]
            assert len(counts) == n_cols, f'Too few counts in row with kmer:{kmer}'

        if last_background:
            counts[-1] = counts[-1]-sum(counts[:-1])
            assert counts[-1] >= 0, '''
                background counts should be larger than the positive counts
                so that a negative set can be created by subtraction the positive count
                from the background count. Problematic k-mer: {kmer}
                '''
            #counts[-1] *= n_scale
        kmer_table[kmer2num(kmer)] = counts
        assert(all(c <= max_val for c in counts))
    return kmer_table, KE


def read_joint_kmer_counts_table(f, general_pattern, last_background = False,  dtype=np.uint32, n_scale=1):
    """Read kmer counts from a file with kmer counts.
    First columns is kmer as string. Next columns are counts.
    Should be at least two columns with counts.

    Args:
        f (file object): input file
        general_pattern (str): General Pattern
        last_background: Is the last column backround counts (includes counts from other columns)?
        n_scale (int, optional): Option to scale the background counts. Defaults to 1. Only works if last_background==True

    Returns:
        numpy.array with counts for each kmer and KmerEnumeration
    """
    line, line_generator = peek_first(f)
    n_cols = len(line.split())-1
    max_val = np.iinfo(dtype).max
    n_rows = generality(general_pattern)
    kmer_table = np.zeros((n_rows, n_cols), dtype)
    KE = KmerEnumeration(general_pattern)
    kmer2num = KE.get_kmer2num()
    for line in line_generator():
        kmer, *counts = line.split()
        kmer = kmer.upper()
        assert all(n in nucleotides for n in kmer)
        #if not all(n in nucleotides for n in kmer):
        #    #Ignore kmers with unusual nucleotides
        #    #TODO: should add warning
        #    continue
        try:
            counts = [int(x) for x in counts]
        except:
            #Floating points will be rounded to ints
            counts = [int(float(x)) for x in counts]

        assert len(counts) == n_cols, f'Too few counts in row with kmer:{kmer}'

        if last_background:
            counts[-1] = counts[-1]-sum(counts[:-1])
            assert counts[-1] >= 0, '''
                background counts should be larger than the positive counts
                so that a negative set can be created by subtraction the positive count
                from the background count. Problematic k-mer: {kmer}
                '''
            #counts[-1] *= n_scale
        kmer_table[kmer2num(kmer)] = counts
        assert(all(c <= max_val for c in counts))
    return kmer_table, KE


def read_joint_kmer_counts_table_pair(f, general_pattern, last_background = False,  dtype=np.uint32, n_scale=1):
    """Read pairwise kmer counts from a file with kmer counts.
    First columns is kmer as string. Next columns are counts.
    The columns are paired with equal col numbers corresponding to mutated counts and
    unequal numbers ot unmutated counts.
    If col i is mutated of type x then col i+1 is unmutated of type x.
    Should be at least two columns with counts. And an equal number of count columns.

    Args:
        f (file object): input file
        general_pattern (str): General Pattern
        last_background: Is the last column backround counts (includes counts from other columns)?
        n_scale (int, optional): Option to scale the background counts. Defaults to 1. Only works if last_background==True

    Returns:
        numpy.array with counts for each kmer and KmerEnumeration
    """
    line, line_generator = peek_first(f)
    n_cols = len(line.split())-1
    assert n_cols %2 ==0, "ERROR: should be equalt number of columns in pairwise mode"
    n_cols = n_cols // 2
    max_val = np.iinfo(dtype).max
    n_rows = generality(general_pattern)
    kmer_table = np.zeros((n_rows, 2, n_cols), dtype)
    print(kmer_table.shape)
    KE = KmerEnumeration(general_pattern)
    kmer2num = KE.get_kmer2num()
    for line in line_generator():
        kmer, *counts = line.split()
        kmer = kmer.upper()
        assert all(n in nucleotides for n in kmer)
        #if not all(n in nucleotides for n in kmer):
        #    #Ignore kmers with unusual nucleotides
        #    #TODO: should add warning
        #    continue
        try:
            counts_M = [int(x) for x in counts[0::2]]
            counts_U = [int(x) for x in counts[1::2]]

        except:
            #Floating points will be rounded to ints
            counts_M = [int(float(x)) for x in counts[0::2]]
            counts_U = [int(float(x)) for x in counts[1::2]]

        assert len(counts_M) == n_cols, f'Too few counts in row with kmer:{kmer}'
        assert len(counts_U) == n_cols, f'Too few counts in row with kmer:{kmer}'

        if last_background:
            for i in range(n_cols):
                counts_U[i] -= counts_M[i]
                assert counts_U[i] >= 0, '''
                    background counts should be larger than the positive counts
                    so that a negative set can be created by subtraction the positive count
                    from the background count. Problematic k-mer: {kmer}
                    '''
            #counts[-1] *= n_scale
        kmer_table[kmer2num(kmer)][0] = counts_M
        kmer_table[kmer2num(kmer)][1] = counts_U
        assert(all(c <= max_val for c in counts_M))
        assert(all(c <= max_val for c in counts_U))
    return kmer_table, KE



def contextD2kmer_table(contextD, general_pattern, dtype=np.uint32):
    if general_pattern is None:
        general_pattern = LCA_pattern_of_kmers(contextD.keys())
    first_context = next(iter(contextD.keys()))
    n_cols = len(contextD[first_context])
    max_val = np.iinfo(dtype).max
    n_rows = generality(general_pattern)
    kmer_table = np.zeros((n_rows, n_cols), dtype)
    KE = KmerEnumeration(general_pattern)
    kmer2num = KE.get_kmer2num()
    for kmer in contextD:
        counts = contextD[kmer]
        assert(all(c <= max_val for c in counts))
        kmer_table[kmer2num(kmer)] = counts
    return kmer_table, KE


def read_joint_kmer_counts(f, super_pattern, n_scale=1):
    """Read kmer counts from a file with three columns: kmer, count_mutated count_background

    Args:
        f (file object): input file
        super_pattern: General pattern
        n_scale (int, optional): Option to scale the background counts. Defaults to 1.

    Returns:
        tuple of: 
            dictionary with kmer counts
            total number of unmutated counts
            total number of mutated counts
    """
    contextD = {}
    n_sites = 0
    n_mut = 0
    for line in f:
        kmer, count_mut, count_denominator = line.split()

        if not all(n in nucleotides for n in kmer):
            continue

        try:
            count_denominator = int(count_denominator)
        except:
            count_denominator = int(float(count_denominator))
        try:
            count_mut = int(count_mut)
        except:
            count_mut = int(float(count_mut))

        assert n_scale*count_denominator - count_mut >= 0, f'''
            background counts should be larger than the positive counts
            so that a negative set can be created by subtraction the positive count
            from the background count. Problematic kmer: {kmer}'''

        if not super_pattern is None:
            if kmer not in super_pattern:
                continue
        n_sites += n_scale*count_denominator
        n_mut += count_mut
        contextD[kmer] = (count_mut, n_scale*count_denominator - count_mut)
    f.close()
    return contextD, n_sites-n_mut, n_mut


def downsize_contextD(D, general_pattern, length):
    """Change a dictionary of kmer counts to a dictinary with kmer counts
    for a smaller value of k.

    Args:
        D (dict): kmer count dictonary
        general_patttern (str): the general pattern
        length (int): The new value of k

    Returns:
        tuple with downsized dictionary and downsized general pattern
    """
    res = {}
    start = None
    end = None
    for context in D:
        if start is None:
            assert not length is None
            assert len(context) > length, f'k-mer:{context} cannot be reduced to length {length}'
            radius1 = (len(context)//2)
            radius2 = length//2
            start = radius1-radius2
            end = (radius1-radius2)+length
        counts = D[context]
        context = context[start:end]
        if context not in res:
            res[context] = [0]*len(counts)
        for i in range(len(counts)):
            res[context][i] += counts[i]
    return res, general_pattern[start:end]

def downsize_kmer_table(kmer_table, KE, length):
    """Change kmer_table to a kmer_table for a smaller value of k.

    Args:
        kmer_table (np.array): kmer counts
        KE : KmerEnumeration
        general_patttern (str): the general pattern
        length (int): The new value of k

    Returns:
        tuple with downsized kmer_table and new KE
    """
    current_length  = len(KE.genpat)
    radius1 = (current_length//2)
    radius2 = length//2
    start = radius1-radius2
    end = (radius1-radius2)+length
    new_genpat = KE.genpat[start:end]
    n_cols = kmer_table.shape[1]
    n_rows = generality(new_genpat)
    dtype= kmer_table.dtype
    new_kmer_table = np.zeros((n_rows, n_cols), dtype)
    max_val = np.iinfo(dtype).max
    new_KE = KmerEnumeration(new_genpat)
    new_kmer2num = new_KE.get_kmer2num()
    n_kmers = kmer_table.shape[0]
    for i in range(n_kmers):
        old_kmer = KE.num2kmer(i)
        new_kmer = old_kmer[start:end]
        new_num = new_kmer2num(new_kmer)
        new_kmer_table[new_num] += kmer_table[i]
    return new_kmer_table, new_KE


def read_dict(f, super_pattern, length=None):
    """Read kmer counts from a file with three columns: kmer, count

    Args:
        f (file object): input file
        super_pattern: General pattern
        length (int, optional): downscale kmers to this length


    Returns:
        tuple of: 
            dictionary with kmer counts
            total number of counts
    """
    D = {}
    all_counts = 0
    start = None
    end = None
    for line in f:
        context, count = line.split()

        if not all(n in nucleotides for n in context):
            continue

        try:
            count = int(count)
        except:
            count = int(float(count))
        assert count>=0, f'negative counts are not allowed, bad line:\n{line.strip()}'

        if start is None:
            if not length is None and length!=len(context):
                assert len(context) > length
                radius1 = (len(context)//2)
                radius2 = length//2
                start = radius1-radius2
                end = (radius1-radius2)+length
            else:
                start = 0
                end = len(context)
        context = context[start:end]


        if not super_pattern is None:
            assert len(super_pattern) == len(context)
            if context not in super_pattern:
                continue
        
        all_counts += count
        if context not in D:
            D[context] = 0
        D[context] += count
    return D, all_counts


def read_postive_and_other(fpos, fother, super_pattern, n_scale=1, background=True):
    """Read input data from two input files

    Args:
        fpos (file obj): Input file with columns: kmer, count
        fother (file obj): Input file with columns: kmer, count
        super_pattern (str): General Pattern
        n_scale (int, optional): Option to scale the background counts. Defaults to 1.
        background (bool, optional): Is the other dictionary based on background counts or
            positive counts? Defaults to True.

    Returns:
        tuple of: 
            dictionary with kmer counts
            total number of unmutated counts
            total number of mutated counts
    """
    posD, allpos = read_dict(fpos, super_pattern)

    otherD, allother = read_dict(fother, super_pattern, length=len(next(iter(posD.keys()))))
    all_contexts = list(set([*posD.keys(),*otherD.keys()]))
    resD = {}
    for context in all_contexts:
        if context in posD:
            count_mut = posD[context]
        else:
            count_mut = 0

        if context in otherD:
            count_denominator = n_scale * otherD[context]
        else:
            count_denominator  = 0

        if background:
            assert count_denominator >= count_mut, '''
                background counts should be larger than the positive counts
                so that a negative set can be created by subtraction the positive count
                from the background count. Problematic k-mer: {context}
                '''
            count_denominator = count_denominator - count_mut

        resD[context] = (count_mut, count_denominator)
    if background:
        allother = allother - allpos

    return resD, allother, allpos


# def read_input(args, super_pattern):
#     """Read input file and get dictionary with kmer counts

#     Args:
#         args: Parsed command line arguments
#         super_pattern (str): General Pattern

#     Returns:
#         tuple of: 
#             dictionary with kmer counts
#             total number of unmutated counts
#             total number of mutated counts
#     """
    
#     assert (args.positive is None) != (args.joint_context_counts is None), '''
#         Either the --positive option or the --join_context_counts option (but not both)
#         must be used to provide input data.
#         '''
#     if not args.positive is None:
#         assert (args.negative is None) != (args.background is None), '''
#             If the --joint_context_counts option is not used then either the --negative or the
#             --background option (but not both) must be used.
#             '''
#         if not args.negative is None:
#             contextD, n_unmut, n_mut = read_postive_and_other(args.positive, args.negative, super_pattern, n_scale = 1, background=False)
#         else:
#             contextD, n_unmut, n_mut = read_postive_and_other(args.positive, args.background, super_pattern, n_scale = 1, background=True)
#     else:
#             contextD, n_unmut, n_mut = read_joint_kmer_counts(args.joint_context_counts, super_pattern, n_scale = 1)
        
#     return contextD, n_unmut, n_mut

def read_input(args):
    """Read input file and get dictionary with kmer counts

    Args:
        args: Parsed command line arguments
    Returns:
        kmer_table
    """
    
    assert (args.positive is None) != (args.joint_context_counts is None), '''
        Either the --positive option or the --join_context_counts option (but not both)
        must be used to provide input data.
        '''
    if not args.positive is None:
        assert (args.negative is None) != (args.background is None), '''
            If the --joint_context_counts option is not used then either the --negative or the
            --background option (but not both) must be used.
            '''
        if args.super_pattern is None:
            sp = None
        else:
            sp = Pattern(args.super_pattern)
        if not args.negative is None:
            contextD, n_unmut, n_mut = read_postive_and_other(args.positive, args.negative, sp, n_scale = 1, background=False)
        else:
            contextD, n_unmut, n_mut = read_postive_and_other(args.positive, args.background, sp, n_scale = 1, background=True)
        kmer_table, KE = contextD2kmer_table(contextD, sp)
        assert (all((n_mut, n_unmut)==kmer_table.sum(axis=0)))
    else:
        if args.pairwise:
            kmer_table, KE = read_joint_kmer_counts_table_pair(args.joint_context_counts, args.super_pattern)
        else:
            if args.super_pattern is None:
                if args.verbosity > 0:
                    print(f'Superpattern not provided. Inferring it.', file=sys.stderr)
                kmer_table, KE = read_joint_kmer_counts_table_no_sp(args.joint_context_counts)
            else:
                kmer_table, KE = read_joint_kmer_counts_table(args.joint_context_counts, args.super_pattern)
        
    return kmer_table, KE
