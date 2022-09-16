from kmerpapa.score_utils import get_betas_kmer_table
from kmerpapa.algorithms.bottum_up_array_w_numba import pattern_partition_bottom_up_kmer_table
from kmerpapa.io_utils import read_joint_kmer_counts_table
import pytest

def test_num_bottum_up_array_w_numba():
    f = open("./test_data/joint_5mers.txt")
    genpat = "NNMNN"
    kmer_table, KE = read_joint_kmer_counts_table(f, genpat)
    f.close()
    betas = get_betas_kmer_table(0.8, kmer_table)
    best_score, names, counts = \
        pattern_partition_bottom_up_kmer_table(KE, kmer_table, 0.8, betas, 3.0, 0)
    assert len(names) == 153
    assert best_score == pytest.approx(1324873.0)