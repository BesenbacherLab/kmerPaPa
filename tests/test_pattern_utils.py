from kmerpapa.pattern_utils import *


def test_pattern_enumeration():
    for general_pattern in ["NNMNN", "SWSW"]:
        npat = pattern_max(general_pattern)
        gppl = get_genpat_pos_level(general_pattern)
        cgppl = get_cum_genpat_pos_level(general_pattern)
        gen_pat_level = pattern_level(general_pattern)
        PE = PatternEnumeration(general_pattern)
        n_seen_pat = 0
        for level in (range(gen_pat_level+1)):
            for pattern in subpatterns_level(general_pattern, level):
                assert pattern_level(pattern) == level
                n_seen_pat +=1
                #assert num2pattern(general_pattern, pattern2num(general_pattern, pattern))== pattern
                assert PE.num2pattern(PE.pattern2num(pattern)) == pattern
        assert n_seen_pat == npat, f"mismatch n_seen_pat:{n_seen_pat}, npat:{npat}"

        ord_general_pattern = tuple(ord(x) for x in general_pattern)
        n_seen_pat = 0
        for level in (range(gen_pat_level+1)):
            for ord_pattern in subpatterns_level_ord_np(ord_general_pattern, gen_pat_level, level):
                pattern = ''.join([chr(x) for x in ord_pattern])
                assert pattern_level(pattern) == level
                n_seen_pat +=1 
                assert PE.num2pattern(pattern2num_new_ord(cgppl, ord_general_pattern, ord_pattern)) == pattern
        assert n_seen_pat == npat, f"mismatch n_seen_pat:{n_seen_pat}, npat:{npat}"
        