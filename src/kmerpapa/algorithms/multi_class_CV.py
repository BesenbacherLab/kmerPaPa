import numpy as np
from kmerpapa.pattern_utils import *
from kmerpapa.CV_tools import make_all_folds
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

class CrossValidationMultPair:
    def __init__(self, KE, kmer_table, nfolds=2, nit=1, seed=None, verbosity=2):
        self.nfolds = nfolds
        self.nit = nit
        self.seed = seed
        self.KE = KE
        self.genpat =  KE.genpat
        self.verbosity = verbosity
        prng = np.random.RandomState(seed)

        if kmer_table.sum() > np.iinfo(np.uint32).max:
            itype = np.uint64
        else:
            itype = np.uint32

        self.matches_num_ord = self.KE.get_matches_num_ord()

        self.fold_kmer_table = make_all_folds(self.kmer_table, nfolds, nit, prng)
        self.genpat_ord = np.array([ord(x) for x in self.genpat])

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
                #beta = get_betas(alpha, train_MU[0],  train_MU[1])
                score, names, counts, rates = pattern_partition_bottom_up_kmer_table_pair()
                best_score, papa = greedy_res_kmer_table_ord(self.genpat_ord, train_kmer_table, alpha, beta, penalty, self.matches_num_ord)
                for pattern in papa:
                    this_ll = get_test_logLik_kmer_table(pattern, train_kmer_table, self.fold_kmer_table[repeat][fold], alpha, beta, self.matches_num_ord)
                    test_ll +=  this_ll        
            ll_list.append(test_ll)
        if(self.verbosity>1):
            print(penalty, alpha, sum(ll_list)/len(ll_list), file=sys.stderr)
        return sum(ll_list)/len(ll_list)

class GridSearchCV(CrossValidationMulti):
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

class BaysianOptimizationCV(CrossValidationMulti):
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