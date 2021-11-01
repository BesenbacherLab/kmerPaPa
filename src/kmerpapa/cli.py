"""Module that contains the command line application."""

import argparse
import sys
from kmerpapa.pattern_utils import LCA_pattern_of_kmers, get_M_U
from kmerpapa.score_utils import get_loss
from kmerpapa.papa import *
from kmerpapa.io_utils import read_input, downsize_contextD
from math import log    
import kmerpapa.algorithms.all_kmers_CV
from kmerpapa.algorithms import bottum_up_array_penalty_plus_pseudo_CV
from kmerpapa.algorithms import greedy_penalty_plus_pseudo
from kmerpapa.algorithms import bottum_up_array_w_numba

def get_parser():
    """
    Return the CLI argument parser.

    Returns:
        An argparse parser.
    """
    parser = argparse.ArgumentParser(
        prog="kmerpapa",
        description='Finds optimal k-mer pattern partition in fx. mutation data')
    parser.add_argument(
        '-p', '--positive', type=argparse.FileType('r'),
        help='File with k-mer counts in positive set')
    parser.add_argument(
        '-n', '--negative', type=argparse.FileType('r'),
        help='File with k-mer counts in negative set. '
        'If the negative set is created with a larger k than the positive set then the k-mers '
        'will be collapsed so that they have the same length.')
    parser.add_argument(
        '-b', '--background', type=argparse.FileType('r'),
        help='File with k-mer counts in backgound set (includes both positive and negative regions). '
        'If the background set is created with a larger k than the positive set then the k-mers '
        'will be collapsed so that they have the same length.')
    parser.add_argument(
        '-j', '--joint_context_counts', type=argparse.FileType('r'),
        help='File with k-mer counts in positive set and background set. '
        'This option can be used instead of having positive and '
        'negative counts in seperate files.')
    parser.add_argument(
        '-m', '--scale_factor', default='1', type=float,
        help='All background (or negative) counts will be multiplied by this number. '
        'If the positive set is based on observations from n genomes and the bacgkround '
        'is based on the number of occurances in the reference genome then the scale factor '
        'should be 2*n.')
    parser.add_argument(
        '-f', '--CVfile', type=argparse.FileType('w'),
        help='File with training and test likelihood values from cross validation.')
    parser.add_argument(
        '--verbosity', type = int, default=1,
        help="Amount of info printed to stderr during execution. 0:silent, 1:default, 2:verbose")
    parser.add_argument(
        '--CV_only', action='store_true',
        help="Only run crossvalidation. Do not run on whole data set using best values afterwards.")
    parser.add_argument(
        '--greedy', action='store_true',
        help="Do not calculate the optimal partition by use a greedy heuristic")
    parser.add_argument(
        '--greedyCV', action='store_true',
        help="Use a greedy heuristic during CV but use optimal algorithm afterwards")
    parser.add_argument(
        '-l', '--long_output', action='store_true',
        help="Print all contexts in output format.")
    parser.add_argument(
        '-s', '--super_pattern', type=str,
        help='If a super-pattern is provided the program will only consider k-mers that match that pattern. '
        'The total n will still be used when calculating BIC or HQ.')
    parser.add_argument(
        '--score', type=str, default='penalty_and_pseudo', choices=['penalty_and_pseudo', 'all_kmers', 'BIC', 'AIC', 'HQ', 'LL'],
        help='Type of score function. Default is "penalty_and_pseudo". '
             '"all_kmers" will calculate a rate for each k-mer.')
    parser.add_argument(
        '-N', '--nfolds', type=int, metavar='N',
        help='Perform cross validation with N folds. '
             'If more than one value of pseudo_count and penalty is given then default is 2. '
             'Otherwise default is not to run cross validation if --nfolds option is not set.')
    parser.add_argument(
        '-i', '--iterations', type=int, default=1, metavar='i',
        help='Repeat cross validation i times')
    parser.add_argument(
        '-a', '--pseudo_counts', type=float, metavar='a', nargs='+', default = [0.8],
        help='Different pseudo count (alpha) values to test using cross validation')
    parser.add_argument(
        '-c', '--penalty_values', type=float, metavar='c', nargs='+',
        help='Different penalty values to test using cross validation. '
             'If no value is set for the default scoring function then '
             'log(#k-mers) will be used.')
    parser.add_argument(
        '--test_smaller_k', action='store_true',
        help='By standard k is the width of the k-mers in the input data. '
        'If this option is supplied it will test all odd numbern up to the width using CV '
        'and use the best.')
    parser.add_argument(
        '--seed', type=int,
        help='seed for numpy.random')
    parser.add_argument(
        '-V', '--version', action='store_true',
        help="Print version number and return")
    return parser


def main(args = None):
    """
    Run the main program.

    This function is executed when you type `kmerpapa` or `python -m kmerpapa`.

    Arguments:
        args: Arguments passed from the command line.

    Returns:
        An exit code.
    """
    parser = get_parser()
    args = parser.parse_args(args=args)

    if args.version:
        from kmerpapa import __version__
        print("version:", __version__)
        print()
        return 0

    if not args.super_pattern is None:
        super_pattern = Pattern(args.super_pattern)
    else:
        super_pattern = None

    try:
        contextD, n_unmut, n_mut = read_input(args, super_pattern)
    except Exception as e:
        parser.print_help()
        print('='*80)
        print("input error:")
        print(e)
        print('='*80)

        return 0

    if args.verbosity > 0:
        print(f'Input data read. {n_mut} positive k-mers and {n_unmut} negative k-mers', file=sys.stderr)

    if not args.penalty_values is None:
        assert args.score == 'penalty_and_pseudo', f'you cannot specify penalty values when using the {args.score} score function'
    else:
        if args.score == "BIC":
            args.penalty_values = [log(n_mut)]
        elif args.score == "AIC":
            args.penalty_values = [2.0]
        elif args.score == "HQ":
            args.penalty_values = [log(log(n_mut))]
        elif args.score == "LL":
            args.penalty_values = [0.0]
        elif args.score == 'all_kmers':
            pass
        elif args.score == 'penalty_and_pseudo':
            args.penalty_values = [log(len(contextD))]
            if args.verbosity > 0:
                print(f'penalty values not set. Using {args.penalty_values[0]}', file=sys.stderr)
        else:
            assert False, f"illegal score option {args.score}"

 
    gen_pat = LCA_pattern_of_kmers(list(contextD.keys()))

    if not args.super_pattern is None:
        assert gen_pat == args.super_pattern

    for context in matches(gen_pat):
        if context not in contextD:
            contextD[context] = (0,0)

    if args.verbosity > 0:
        print(f'General pattern: {gen_pat}', file=sys.stderr)

    if not args.CVfile is None:
        print('k alpha P LL_test', file=args.CVfile)

    best_alpha = None
    best_penalty = None
    best_k = None

    if args.test_smaller_k:
        ks = range(len(gen_pat),1,-2)
    else:
        ks = [len(gen_pat)]

    this_contextD = contextD
    this_gen_pat = gen_pat
    best_score = 1e100

    if args.nfolds is None and (len(ks) > 1 or len(args.pseudo_counts) > 1 or len(args.penalty_values) > 1):
        args.nfolds = 2            

    if not args.nfolds is None and args.nfolds > 1:
        for k in ks:
            if args.verbosity > 0:
                print(f'Running {args.nfolds}-fold cross validation on {k}-mers', file=sys.stderr)
            if k != len(this_gen_pat):
                this_contextD, this_gen_pat = downsize_contextD(this_contextD, this_gen_pat, k)
            if args.greedy or args.greedyCV:
                assert args.score != 'all_kmers', 'greedy option cannot be used wil all-kmers'
                this_alpha, this_penalty, test_score = greedy_penalty_plus_pseudo.greedy_partition_CV(this_gen_pat, this_contextD, args.pseudo_counts, args, n_mut, n_unmut, args.penalty_values)
            elif args.score == 'all_kmers':
                this_alpha, test_score =  kmerpapa.algorithms.all_kmers_CV.all_kmers(this_gen_pat, this_contextD, args.pseudo_counts, args, n_mut, n_unmut)
                this_penalty = None
            else:
                this_alpha, this_penalty, test_score = bottum_up_array_penalty_plus_pseudo_CV.pattern_partition_bottom_up(this_gen_pat, this_contextD, args.pseudo_counts, args, n_mut, n_unmut, args.penalty_values)
            if test_score < best_score:
                best_score = test_score
                best_k = k
                best_alpha = this_alpha
                best_penalty = this_penalty
        if args.verbosity > 0:
            print(f'CV DONE. best_k={best_k}, best_alpha={best_alpha}, best_penalty={best_penalty}, best_test_LL={best_score}', file=sys.stderr)

    if not args.CVfile is None:
        args.CVfile.close()

    if args.CV_only:
        return 0
    
    if best_alpha is None:
        assert len(args.pseudo_counts)==1
        best_alpha = args.pseudo_counts[0]

    if args.score != 'all_kmers' and best_penalty is None:
        assert len(args.penalty_values)==1
        best_penalty = args.penalty_values[0]

    if best_k is None:
        best_k = len(gen_pat)

    if best_k != len(gen_pat):
        contextD, gen_pat = downsize_contextD(contextD, gen_pat, best_k)

    my=n_mut/(n_mut+n_unmut)
    best_beta = (best_alpha*(1.0-my))/my

    if args.verbosity >0:
        print(f'Training on whole data set with k={best_k} alpha={best_alpha} penalty={best_penalty}' ,file=sys.stderr)

    if args.greedy:
        if args.score == 'PPCV':
            best_score, M, U, names = \
                greedy_penalty_plus_pseudo.greedy_partition(gen_pat, contextD, best_alpha, best_beta, best_penalty, args)
    elif args.score == 'all_kmers':
        #TODO: maybe I should calculate best score also for all_kmers
        best_score = 0
        M = n_mut
        U = n_unmut
        names = list(matches(gen_pat))
    else:
        best_score, M, U, names = \
            bottum_up_array_w_numba.pattern_partition_bottom_up(gen_pat, contextD, best_alpha, best_beta, best_penalty, args, n_mut, n_unmut)
        
    counts = []
    for pat in names:
        counts.append(get_M_U(pat, contextD))

    # Just to check that it is a partition
    #sp = PatternPartition(list(names), superPattern=gen_pat)
    # Check removed because it takes too long on large k datasets with all_kmers.
    print(M, n_mut)
    assert M == n_mut
    assert U == n_unmut
    assert n_mut == sum(x[0] for x in counts)
    assert n_unmut == sum(x[1] for x in counts)

    total_rate = float(n_mut)/(n_mut+n_unmut)

    if args.verbosity>0:
        print(f'Optimal k-mer pattern partition contains {len(names)} patterns.', file=sys.stderr)
        print(f'loss={best_score}', file=sys.stderr)
        print(f'LL={get_loss(counts, best_alpha, best_beta)}', file=sys.stderr)

    if args.long_output:
        print('context', 'c_neg', 'c_pos', 'c_rate',
                'pattern', 'p_neg', 'p_pos', 'p_rate')
    else:
        print('pattern', 'p_neg', 'p_pos', 'p_rate')

    for i in range(len(names)):
        pat = names[i]
        M, U = counts[i]
        p = (M + best_alpha)/(M + U + best_alpha + best_beta)
        if args.long_output:
            for context in matches(pat):
                nm, ns = contextD[context]
                print(context, ns, nm, float(nm)/(nm+ns), pat, U, M, p)
        else:
            print(pat, U, M, p)

    return 0
