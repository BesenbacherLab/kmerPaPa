
from kmerpapa.io_utils import read_joint_kmer_counts_table, downsize_kmer_table, read_joint_kmer_counts_table_no_sp

def test_read_joint_kmer_counts_table():
    D = {}
    genpat = 'NNMNN'
    f = open("./test_data/joint_5mers.txt")
    for line in f:
        kmer, *counts = line.split()
        kmer = kmer.upper()
        D[kmer] = counts
    f.close()
    f = open("./test_data/joint_5mers.txt")
    kmer_table, KE = read_joint_kmer_counts_table(f, genpat)
    f.close()
    f = open("./test_data/joint_5mers.txt")
    kmer_table_no_sp, KE_no_sp = read_joint_kmer_counts_table_no_sp(f)
    f.close()
    assert((kmer_table_no_sp==kmer_table).all())
    for i in range(len(D)):
        counts1 = kmer_table[i][0]
        counts2 = kmer_table[i][1]
        kmer = KE.num2kmer(i)
        counts = D[kmer]
        assert(int(counts1) == int(counts[0]))
        assert(int(counts2) == int(counts[1]))


def test_downsize_kmer_counts():
    f = open("./test_data/joint_5mers.txt")
    genpat = "NNMNN"
    kmer_table, KE = read_joint_kmer_counts_table(f, genpat)
    new_kmer_table, new_KE = downsize_kmer_table(kmer_table, KE, 3)
    assert(all(kmer_table.sum(axis=0)==new_kmer_table.sum(axis=0)))

