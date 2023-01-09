"""Module that contains the command line application."""

import argparse
import sys
from kmerpapa.pattern_utils import LCA_pattern_of_kmers, get_M_U
from kmerpapa.score_utils import get_loss, get_betas_kmer_table
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
    ## It is probably best not to use the --scale_factor option but instead just scale the
    ## rates afterwards. So I am commenting it out for now. If I want to use it again I need
    ## to use args.scale_factor in io_utils.read_input again.
    #parser.add_argument(
    #    '-m', '--scale_factor', default='1', type=float,
    #    help='All background (or negative) counts will be multiplied by this number. '
    #    'If the positive set is based on observations from n genomes and the bacgkround '
    #    'is based on the number of occurances in the reference genome then the scale factor '
    #    'should be 2*n.')
    parser.add_argument(
        '-o', '--output', type=argparse.FileType('w'), default='-',
        metavar='PATH',
        help="Output file (default: standard output)")
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
        help="Use a fast greedy heuristic to find a (hopefully) good but not necessarily optimal pattern partition.")
    parser.add_argument(
        '--BayesOpt', action='store_true',
        help="Using Bayesian Optimization to fit pseudo_count and penalty with Cross Validation. Sofar only works in combination with --greedy")
    parser.add_argument(
        '--greedyCV', action='store_true',
        help="Use a greedy heuristic during CV but use optimal algorithm afterwards")
    parser.add_argument(
        '-l', '--long_output', action='store_true',
        help="Print all k-mers in output format.")
    parser.add_argument(
        '--pairwise', action='store_true',
        help="Multi-class counts are pairwise counts.")
    parser.add_argument(
        '-s', '--super_pattern', type=str,
        help='If a super-pattern is provided the program will only consider k-mers that match that pattern. '
             'If for instance the "--positive" file contain all 5-mers at A->T mutated sites but the "--background" '
             'file contains 5-mers from all sites in the genome. Then "--super_pattern NNANN" should be specified to '
             'ignore 5-mers where A->T mutations cannot happen.')
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

    try:
        kmer_table, KE = read_input(args)
        gen_pat = KE.genpat
    except Exception as e:
        parser.print_help()
        print('='*80, file=sys.stderr)
        print("input error:", file=sys.stderr)
        print(e, file=sys.stderr)
        print('='*80, file=sys.stderr)
        return 0
    
    col_sums = kmer_table.sum(axis=0)
    if args.verbosity > 0:
        print(f'Input data read. {col_sums}', file=sys.stderr)
        #print(f'Input data read. {n_mut} positive k-mers and {n_unmut} negative k-mers', file=sys.stderr)

    if args.pairwise:
        n_kmers, _two, n_muttype = kmer_table.shape
        assert _two == 2
    else:
        n_kmers, n_muttype = kmer_table.shape        
    

    if not args.penalty_values is None:
        assert args.score == 'penalty_and_pseudo', f'you cannot specify penalty values when using the {args.score} score function'
    else:
        if args.score == "BIC":
            args.penalty_values = [log(col_sums.sum())]
        elif args.score == "AIC":
            args.penalty_values = [2.0]
        elif args.score == "HQ":
            args.penalty_values = [log(log(col_sums.sum()))]
        elif args.score == "LL":
            args.penalty_values = [0.0]
        elif args.score == 'all_kmers':
            pass
        elif args.score == 'penalty_and_pseudo':
            if not args.BayesOpt:
                args.penalty_values = [log(n_kmers*n_muttype)]
                if args.verbosity > 0:
                    print(f'penalty values not set. Using {args.penalty_values[0]}', file=sys.stderr)
        else:
            assert False, f"illegal score option {args.score}"

 
    #gen_pat = LCA_pattern_of_kmers(list(contextD.keys()))

    #assert(not args.super_pattern is None)
    #gen_pat = args.super_pattern

    #if not args.super_pattern is None:
    #    assert gen_pat == args.super_pattern

    #for context in matches(gen_pat):
    #    if context not in contextD:
    #        contextD[context] = (0,0)

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

    this_kmer_table = kmer_table
    this_gen_pat = gen_pat
    best_score = 1e100

    # if args.nfolds is None and (len(ks) > 1 or len(args.pseudo_counts) > 1 or len(args.penalty_values) > 1 or args.CV_only):
    #     args.nfolds = 2            
    # if not args.nfolds is None and args.nfolds > 1:
    #     for k in ks:
    #         if args.verbosity > 0:
    #             print(f'Running {args.nfolds}-fold cross validation on {k}-mers', file=sys.stderr)
    #         if k != len(this_gen_pat):
    #             this_contextD, this_gen_pat = downsize_contextD(this_contextD, this_gen_pat, k)
    #         if args.greedy or args.greedyCV:
    #             assert args.score != 'all_kmers', 'greedy option cannot be used wil all-kmers'
    #             if args.BayesOpt:
    #                 CV = greedy_penalty_plus_pseudo.BaysianOptimizationCV(gen_pat, contextD, args.nfolds, args.iterations, args.seed)
    #                 this_alpha, this_penalty, test_score = CV.get_best_a_c()
    #             else:
    #                 #Grid Search
    #                 CV = greedy_penalty_plus_pseudo.GridSearchCV(gen_pat, contextD, args.penalty_values, args.pseudo_counts, args.nfolds, args.iterations, args.seed)
    #                 this_alpha, this_penalty, test_score = CV.get_best_a_c()
    #             #this_alpha2, this_penalty2, test_score2 = greedy_penalty_plus_pseudo.greedy_partition_CV(this_gen_pat, this_contextD, [this_alpha], args, n_mut, n_unmut, [this_penalty])
    #             #print(test_score, test_score3, test_score2)
    #             #this_alpha, this_penalty, test_score = greedy_penalty_plus_pseudo.greedy_partition_CV(this_gen_pat, this_contextD, args.pseudo_counts, args, n_mut, n_unmut, args.penalty_values)
    #         elif args.score == 'all_kmers':
    #             this_alpha, test_score =  kmerpapa.algorithms.all_kmers_CV.all_kmers(this_gen_pat, this_contextD, args.pseudo_counts, args, n_mut, n_unmut)
    #             this_penalty = None
    #         else:
    #             this_alpha, this_penalty, test_score = bottum_up_array_penalty_plus_pseudo_CV.pattern_partition_bottom_up(this_gen_pat, this_contextD, args.pseudo_counts, args, n_mut, n_unmut, args.penalty_values)
    #         if test_score < best_score:
    #             best_score = test_score
    #             best_k = k
    #             best_alpha = this_alpha
    #             best_penalty = this_penalty
    #     if args.verbosity > 0:
    #         print(f'CV DONE. best_k={best_k}, best_alpha={best_alpha}, best_penalty={best_penalty}, best_test_LL={best_score}', file=sys.stderr)

    # if not args.CVfile is None:
    #     args.CVfile.close()

    # if args.CV_only:
    #     return 0
    
    if best_alpha is None:
        assert len(args.pseudo_counts)==1
        best_alpha = args.pseudo_counts[0]

    if args.score != 'all_kmers' and best_penalty is None:
        assert len(args.penalty_values)==1
        best_penalty = args.penalty_values[0]

    if best_k is None:
        best_k = len(gen_pat)

    if best_k != len(gen_pat):
        assert(False)
        #contextD, gen_pat = downsize_contextD(contextD, gen_pat, best_k)


    if args.pairwise:
        n_kmers, _two, n_muttype = kmer_table.shape
        assert _two == 2
    else:
        n_kmers, n_muttype = kmer_table.shape        
        

    #my=n_mut/(n_mut+n_unmut)
    best_betas = get_betas_kmer_table(best_alpha, kmer_table)

    if args.verbosity >0:
        print(f'Training on whole data set with k={best_k} alpha={best_alpha} penalty={best_penalty}' ,file=sys.stderr)

    # if args.score == 'all_kmers':
    #     #TODO: maybe I should calculate best score also for all_kmers
    #     best_score = 0
    #     M = n_mut
    #     U = n_unmut
    #     names = list(matches(gen_pat))
    # elif args.greedy:
    #     best_score, M, U, names = \
    #         greedy_penalty_plus_pseudo.greedy_partition(gen_pat, contextD, best_alpha, best_beta, best_penalty, args)

    #else:
    if args.pairwise:
        best_score, names, counts, rates = \
            bottum_up_array_w_numba.pattern_partition_bottom_up_kmer_table_pair(KE, kmer_table, best_alpha, best_penalty, args.verbosity)
    else:
        best_score, names, counts = \
            bottum_up_array_w_numba.pattern_partition_bottom_up_kmer_table(KE, kmer_table, best_alpha, best_betas, best_penalty, args.verbosity)
        
        #(gen_pat, contextD, best_alpha, best_beta, best_penalty, args, n_mut, n_unmut)
        
    #counts = []
    #for pat in names:
    #    counts.append(get_M_U(pat, contextD))

    # Just to check that it is a partition
    #sp = PatternPartition(list(names), superPattern=gen_pat)
    # Check removed because it takes too long on large k datasets with all_kmers.

    #total_rate = float(n_mut)/(n_mut+n_unmut)

    if args.verbosity>0:
        print(f'Optimal k-mer pattern partition contains {len(names)} patterns.', file=sys.stderr)
        print(f'loss={best_score}', file=sys.stderr)
        #print(f'LL={get_loss(counts, best_alpha, best_betas)}', file=sys.stderr)

    if args.long_output:
        print('context', 'c_neg', 'c_pos', 'c_rate',
                'pattern', 'p_neg', 'p_pos', 'p_rate', file=args.output)
    elif args.pairwise:
        count_head = ' '.join(f'positive{i+1}_count negative{i+1}_count' for i in range(n_muttype)) 
        #count1_head = ' '.join(f'positive{i+1}_count' for i in range(n_muttype)) 
        #count2_head = ' '.join(f'negative{i+1}_count' for i in range(n_muttype)) 
        rate_head = ' '.join(f'type{i+1}_rate' for i in range(n_muttype))
        print('pattern', count_head, rate_head, file=args.output) 
    else:
        count_head = ' '.join(f'type{i+1}_count' for i in range(n_muttype)) 
        rate_head = ' '.join(f'type{i+1}_rate' for i in range(n_muttype)) 

        print('pattern', count_head, rate_head, file=args.output)

    print(counts[0])
    for i in range(len(names)):
        pat = names[i]
        #if args.long_output:
        #    for context in matches(pat):
        #        nm, ns = contextD[context]
        #        print(context, ns, nm, float(nm)/(nm+ns), pat, U, M, p, file=args.output)
        #else:

        if args.pairwise:
            #count_list = [str(x) for x in counts[i].flatten()]
            count_list = [f'{counts[i][0][j]} {counts[i][1][j]}' for j in range(n_muttype)]
            rate_list = [str(x) for x in rates[i]]
            print(pat, " ".join(count_list), " ".join(rate_list),file=args.output)
        else:
            count_list = [str(x) for x in list(counts[i])]
            p = (counts[i] + best_alpha)/(counts[i].sum() + best_alpha + best_betas)
            p = p/p.sum()
            p_list = [str(x) for x in list(p)]
            print(pat, " ".join(count_list), " ".join(p_list), file=args.output)

    return 0
