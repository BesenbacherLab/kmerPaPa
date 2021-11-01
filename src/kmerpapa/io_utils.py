nucleotides = ['A','C','G','T']

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
                from the background count. Problematic context: {context}
                '''
            count_denominator = count_denominator - count_mut

        resD[context] = (count_mut, count_denominator)
    if background:
        allother = allother - allpos

    return resD, allother, allpos


def read_input(args, super_pattern):
    """Read input file and get dictionary with kmer counts

    Args:
        args: Parsed command line arguments
        super_pattern (str): General Pattern

    Returns:
        tuple of: 
            dictionary with kmer counts
            total number of unmutated counts
            total number of mutated counts
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
        if not args.negative is None:
            contextD, n_unmut, n_mut = read_postive_and_other(args.positive, args.negative, super_pattern, n_scale = args.scale_factor, background=False)
        else:
            contextD, n_unmut, n_mut = read_postive_and_other(args.positive, args.background, super_pattern, n_scale = args.scale_factor, background=True)
    else:
            contextD, n_unmut, n_mut = read_joint_kmer_counts(args.joint_context_counts, super_pattern, n_scale = args.scale_factor)
        
    return contextD, n_unmut, n_mut
