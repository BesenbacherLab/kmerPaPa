from kmerpapa.pattern_utils import code, matches, set_code, set_perm_code, inv_code

class Pattern():
    """Class representing a k-mer pattern
    """
    def __init__(self, pattern_string):
        self.pattern = pattern_string

    def __contains__(self, context):
        '''Is context string covered by the pattern?'''
        for (p, c) in zip(self.pattern, context):
            if c not in code[p]:
                return False
        return True

    def __str__(self):
        return self.pattern

    def __repr__(self):
        return self.pattern

    def __len__(self):
        return len(self.pattern)

    def __iter__(self):
       return matches(self.pattern)

    def __and__(self, other):
        '''intersection between two patterns'''
        res = []
        for (c1, c2) in zip(self.pattern, other.pattern):
            scode = set_code[c1] & set_code[c2]
            if len(scode) == 0:
                return None
            res.append(inv_code[scode])
        return Pattern(''.join(res))

    def __le__(self, other):
        '''Is other pattern a subpattern of this'''
        for (x, y) in zip(self.pattern, other.pattern):
            if x not in set_perm_code[y]:
                return False
        return True

    def cardinality(self):
        '''number of contexts matching pattern'''
        res = 1
        for x in self.pattern:
            res *= len(code[x])
        return res


class PatternPartition():
    """Class representing a k-mer Pattern Partition
    """
    def __init__(self, patterns, superPattern=None, strandSymmetry=True):
        ## validate that it is a partition ##
        patterns.sort()
        self.patterns = [Pattern(p) for p in patterns]
        if superPattern is None:
            sp = 'N' * len(patterns[0])
            if strandSymmetry:
                radius = len(patterns[0])//2
                sp =  'N'*radius + 'M' + 'N'*radius
            self.superPattern = Pattern(sp)
        else:
            self.superPattern = Pattern(superPattern)
        n_matches = 0

        for i in range(len(self.patterns)):
            n_matches += self.patterns[i].cardinality()

            #assert that pattern i is a subpattern of the superPattern
            assert self.patterns[i] <= self.superPattern, \
              "pattern #%r (%r) is not a subpattern of the superPattern (%r)" \
              %(i, self.patterns[i], self.superPattern)

            # Out-commentet because it is slow and most errors should be caught by n_matches assertion
            #for j in range(i+1,len(self.patterns)):
            #    #assert that common set between pattern i and pattern j is the empty set:
            #    assert (self.patterns[i] & self.patterns[j]) is None, \
            #      "the common set between pattern #%r (%r) and pattern #%r (%r) is not empty (%r)" \
            #      %(i, self.patterns[i], j, self.patterns[j], self.patterns[i] & self.patterns[j])

        #assert that the patterns cover the super pattern:
        assert n_matches == self.superPattern.cardinality(), \
          "the patterns does not cover the superPattern (%r)" %(self.superPattern)

    def __len__(self):
        return len(self.patterns)

    def pattern_length(self):
        return len(self.patterns[0])

    def __getitem__(self, context):
        for p in self.patterns:
            if context in p:
                return p
        return None

    def __str__(self):
        res = ["[PatternPartition:"]
        for pattern in self.patterns:
            res.append(str(pattern) + ' ' + str(pattern.cardinality()))
        res.append('-' * len(self.patterns[0]))
        res.append(str(self.superPattern) + ' ' + str(self.superPattern.cardinality()) + ']')
        return '\n '.join(res)
