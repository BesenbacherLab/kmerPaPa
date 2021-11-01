import numpy as np
from kmerpapa.pattern_utils import matches, PatternEnumeration


def sample(m, colors, itype, prng):
    """ Sample from multivariate hypergeometric distribution
    colors should be n_kmers*2 array (unmutated and mutated are seperate colors)
    # from: https://stackoverflow.com/questions/35734026/numpy-drawing-from-urn
    Args:
        m : number balls to draw from the urn
        colors : one-dimensional array of number balls of each color in the urn
        itype : integer type should be either np.uint32 or np.uint64
        prng : numpy random number generator

    Returns:
        One-dimensional array with the same length as `colors` containing the
        number of balls of each color in a random sample.
    """
    remaining = np.cumsum(colors[::-1])[::-1]
    result = np.zeros(len(colors), dtype=itype)
    for i in range(len(colors)-1):
        if m < 1:
            break
        result[i] = prng.hypergeometric(colors[i], remaining[i+1], m)
        m -= result[i]
    result[-1] = m
    return result


def make_all_folds_contextD_patterns(contextD, U_mem, M_mem, general_pattern, prng, itype = np.uint64):
    """Divide counts from contextD into U_mem and M_mem by sampling from multivariate hypergeometric distribution

    Args:
        contextD ({kmer:(n_mut, n_unmut)}): Dictionary with counts for kmer
        U_mem ((n_folds x n_patterns) np.array): Array to be filled with unmutated counts for each kmer for each fold
        M_mem ((n_folds x n_pattern) np.array): Array to be filled with mutated counts for each kmer for each fold
        general_pattern (str): The general pattern. fx "NNN".
        prng : numpy random number generator
        itype : integer type should be either np.uint32 or np.uint64
    """
    PE = PatternEnumeration(general_pattern)
    contexts = list(contextD.keys())
    contexts.sort()
    colors = np.empty(2*len(contexts), dtype=itype)
    for i in range(len(contexts)):
        nm, nu = contextD[contexts[i]]
        colors[i]  = nm
        colors[len(contexts)+i] = nu
    n = colors.sum()
    n_folds = U_mem.shape[1]
    n_samples = n//n_folds
    samples = np.empty((2*len(contexts), n_folds), dtype=itype)
    for i in range(n_folds-1):
        s = sample(n_samples, colors, itype, prng)
        samples[:,i] = s
        colors -= s
    samples[:,n_folds-1] = colors
    for i in range(len(contexts)):
        context = contexts[i]
        pat_num = PE.pattern2num(context)
        M_mem[pat_num] = samples[i]
        U_mem[pat_num] = samples[i+len(contexts)]


def make_all_folds_contextD_kmers(contextD, U_mem, M_mem, general_pattern, prng):
    """Divide counts from contextD into U_mem and M_mem by sampling from multivariate hypergeometric distribution

    Args:
        contextD ({kmer:(n_mut, n_unmut)}): Dictionary with counts for kmer
        U_mem ((n_folds x n_kmers) np.array): Array to be filled with unmutated counts for each kmer for each fold
        M_mem ((n_folds x n_kmers) np.array): Array to be filled with mutated counts for each kmer for each fold
        general_pattern (str): The general pattern. fx "NNN".
        prng : numpy random number generator
    """
    contexts = list(matches(general_pattern))
    itype = np.uint64
    colors = np.zeros(2*len(contexts), dtype=np.uint64)
    for i in range(len(contexts)):
        nm, nu = contextD[contexts[i]]
        colors[i]  = nm
        colors[len(contexts)+i] = nu
    n = colors.sum()
    n_folds = U_mem.shape[1]
    n_samples = n//n_folds
    samples = np.zeros((2*len(contexts), n_folds), dtype=np.uint64)
    #Fill the first n_folds-1 folds:
    for i in range(n_folds-1):
        s = sample(n_samples, colors, itype, prng)
        samples[:,i] = s
        colors -= s
    #Fill the last fold with the remaining counts:
    samples[:,n_folds-1] = colors
    for i in range(len(contexts)):
        M_mem[i] = samples[i]
        U_mem[i] = samples[i+len(contexts)]
    

def make_all_folds(n_mut, n_unmut, U_mem, M_mem):
    """Divide counts from n_mut and n_unmut into U_mem and M_mem by sampling from multivariate hypergeometric distribution

    Args:
        n_mut (int np.array): number of mutated sites for each kmer
        n_unmut (int np.array): number of unmutated sites for each kmer
        U_mem ((n_folds x n_kmers) np.array): Array to be filled with unmutated counts for each kmer for each fold
        M_mem ((n_folds x n_kmers) np.array): Array to be filled with mutated counts for each kmer for each fold
        prng: numpy random number generator
    """
    itype = n_mut.dtype
    prng = np.random.RandomState()
    n_context = len(n_mut)
    colors = np.concatenate((n_unmut, n_mut))
    n = n_mut.sum() + n_unmut.sum()
    n_folds = U_mem.shape[1]
    n_samples = n//n_folds
    for i in range(n_folds-1):
        samples = sample(n_samples, colors, itype, prng)
        colors -= samples
        U_mem[:n_context,i] = samples[:n_context]
        M_mem[:n_context,i] = samples[n_context:]
    U_mem[:n_context,n_folds-1] = colors[:n_context]
    M_mem[:n_context,n_folds-1] = colors[n_context:]



