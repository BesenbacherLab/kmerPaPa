import numpy as np
from numba import njit

code = {'A':['A'],
        'C':['C'],
        'G':['G'],
        'T':['T'],
        'R':['A', 'G'],
        'Y':['C', 'T'],
        'S':['G', 'C'],
        'W':['A', 'T'],
        'K':['G', 'T'],
        'M':['A', 'C'],
        'B':['C', 'G', 'T'],
        'D':['A', 'G', 'T'],
        'H':['A', 'C', 'T'],
        'V':['A', 'C', 'G'],
        'N':['A', 'C', 'G', 'T']}

# dictionary from IUPAC character to nucleotide set
set_code = {}
for x in code:
    set_code[x] = frozenset(code[x])

# dictionary from IUPAC character to all subset encoded as a set of IUPAC characters
subset_code = {}


# dictionary from nucleotide set to IUPAC character
inv_code = {}
for key, val in code.items():
    inv_code[frozenset(val)] = key

# minus_set[nucleotide set][nucleotide set] = IUPAC character
minus_set = {}
for x in code:
    minus_set[x] = {}
    for y in code:
        minus = frozenset(set(code[x])-set(code[y]))
        if len(minus)>0:
            minus_set[x][y] = inv_code[minus]

# dictionary from IUPAC character to list of tuples with all two-partitions written with IUPAC characters
complements = {}
for x in code:
    if len(code[x]) == 2:
        complements[x] = [code[x]]

complements['V'] = [('A', 'S'), ('C', 'R'), ('G', 'M')]
complements['H'] = [('A', 'Y'), ('C', 'W'), ('T', 'M')]
complements['D'] = [('A', 'K'), ('G', 'W'), ('T', 'R')]
complements['B'] = [('C', 'K'), ('G', 'Y'), ('T', 'S')]
complements['N'] = [('S','W'),('K','M'),('R','Y'), ('A','B'),('C','D'),('G','H'),('T','V')]


# complements dictionary but with the output tuples IUPAC characters are encoded as ord(x)
complements_ord = {}
for char in complements:
    complements_ord[char] = [(ord(x),ord(y)) for x,y in complements[char]]

# complements dictionary but with the input and output IUPAC characters are encoded as ord(x)
complements_ord2ord = [()]*100

# n_complements[ord(x)] is number of complements of IUPAC character x
n_complements  = np.zeros(90, dtype=int)
 
# (complements_tab_1[ord(x)][i], complements_tab_1[ord(x)][i]) are the i'th complements pair of the IUPAC character x 
complements_tab_1 = np.zeros((90,7), dtype=int)
complements_tab_2 = np.zeros((90,7), dtype=int)

for x in complements:
    complements_ord2ord[ord(x)] = tuple(complements_ord[x])
    n_complements[ord(x)] = len(complements_ord[x])
    for i in range(len(complements_ord[x])):
        c1,c2 = complements_ord[x][i]
        complements_tab_1[ord(x)][i] = c1
        complements_tab_2[ord(x)][i] = c2

complements_ord2ord = tuple(complements_ord2ord)

set_code2 = {'A':set(['A']),
             'C':set(['C']),
             'G':set(['G']),
             'T':set(['T']),
             'R':set(['A', 'G', 'R']),
             'Y':set(['C', 'T', 'Y']),
             'S':set(['G', 'C', 'S']),
             'W':set(['A', 'T', 'W']),
             'K':set(['G', 'T', 'K']),
             'M':set(['A', 'C', 'M']),
             'B':set(['C', 'G', 'T', 'S', 'Y', 'K', 'B']),
             'D':set(['A', 'G', 'T', 'R', 'W', 'K', 'D']),
             'H':set(['A', 'C', 'T', 'M', 'W', 'Y', 'H']),
             'V':set(['A', 'C', 'G', 'M', 'R', 'S', 'V']),
             'N':set(['A', 'C', 'G', 'T', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V','N'])}

# sub_code = {'A':['A'],
#             'C':['C'],
#             'G':['G'],
#             'T':['T'],
#             'R':['A', 'G'],
#             'Y':['C', 'T'],
#             'S':['G', 'C'],
#             'W':['A', 'T'],
#             'K':['G', 'T'],
#             'M':['A', 'C'],
#             'B':['C', 'G', 'T', 'S', 'Y', 'K'],
#             'D':['A', 'G', 'T', 'R', 'W', 'K'],
#             'H':['A', 'C', 'T', 'M', 'W', 'Y'],
#             'V':['A', 'C', 'G', 'M', 'R', 'S'],
#             'N':['A', 'C', 'G', 'T', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V']}

perm_code = {'A':['A'],
            'C':['C'],
            'G':['G'],
            'T':['T'],
            'R':['A', 'G', 'R'],
            'Y':['C', 'T', 'Y'],
            'S':['G', 'C', 'S'],
            'W':['A', 'T', 'W'],
            'K':['G', 'T', 'K'],
            'M':['A', 'C', 'M'],
            'B':['C', 'G', 'T', 'S', 'Y', 'K', 'B'],
            'D':['A', 'G', 'T', 'R', 'W', 'K', 'D'],
            'H':['A', 'C', 'T', 'M', 'W', 'Y', 'H'],
            'V':['A', 'C', 'G', 'M', 'R', 'S', 'V'],
            'N':['A', 'C', 'G', 'T', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V', 'N']}

# x,y are IUPAC patterns
# y is sub pattern of x
# perm_code_no[x][y] is the number of y in the perm_code list for x
perm_code_no = {}
for x in perm_code:
    perm_code_no[x] = {}
    for i in range(len(perm_code[x])):
        perm_code_no[x][perm_code[x][i]] = i

# same as perm_code_no just in np.array where x and y are encoded as ord(X) and ord(Y)
perm_code_no_np = np.full((90,90), -100, dtype=int)
for x in perm_code:
    for i in range(len(perm_code[x])):
        perm_code_no_np[ord(x),ord(perm_code[x][i])] = i

# perm_code_max[i][x] is the subset of perm_code x where the cardinality is at most i
perm_code_max = [dict() for x in range(4)]
for i in range(4):
    for x in perm_code:
        perm_code_max[i][x] = []
        for y in perm_code[x]:
            if len(code[y])-1 <= i:
                perm_code_max[i][x].append(y)

# perm_code_min_max[i][j][x] is the subset of perm_code x where the cardinality is at least i and at most j
perm_code_min_max = [[dict() for x in range(4)] for y in range(4)]
for i in range(4):
    for j in range(4):
        for x in perm_code:
            perm_code_min_max[i][j][x] = []
            for y in perm_code[x]:
                if len(code[y])-1 >= i and len(code[y])-1 <= j:
                    perm_code_min_max[i][j][x].append(y)

# same as perm_code_min_max[i][j][x] but x is encoded as ord(x)
perm_code_min_max_ord = [[dict() for x in range(4)] for y in range(4)]
for i in range(4):
    for j in range(4):
        for x in perm_code:
            perm_code_min_max_ord[i][j][x] = []
            for y in perm_code[x]:
                if len(code[y])-1 >= i and len(code[y])-1 <= j:
                    perm_code_min_max_ord[i][j][x].append(ord(y))


#same as perm_code_min_max_ord but in a np.array
perm_code_min_max_ord_np = np.full((4,4,100,15),-100, dtype=int)

#number of subpatterns in perm_code_min_max_ord[i][j][x]
perm_code_min_max_ord_np_n = np.empty((4,4,100), dtype=int)
for i in range(4):
    for j in range(4):
        for x in perm_code:
            perm_code_min_max_ord_np_n[i,j,ord(x)] = len(perm_code_min_max_ord[i][j][x])
            for k in range(len(perm_code_min_max_ord[i][j][x])):
                perm_code_min_max_ord_np[i,j,ord(x),k] = perm_code_min_max_ord[i][j][x][k]


# minus_set[x][y] is what is left in x after removing y
# minus_set[x][x] is not set
minus_set = {}
for c in complements:
    minus_set[c] = {}
    for x,y in complements[c]:
        minus_set[c][x] = y
        minus_set[c][y] = x

def get_M_U(pattern, contextD, index_mut=0):
    """calculates the number of mutated and unmutated sites that match a IUPAC pattern

    Args:
        pattern (str): IUPAC pattern
        contextD (dict): The number of mutated and unmutated sites for each k-mer
        index_mut (int, optional): To be used for multi class version. Defaults to 0.

    Returns:
        (M,U): number of mutated (M) and unmutated (U) sites
    """
    M = None
    U = None
    for context in matches(pattern):
        tup = contextD[context]
        nm = tup[index_mut]
        nu = tup[-1]
        if M is None:
            M = nm
            U = nu
        else:
            M += nm
            U += nu
    return M, U


def pattern_level(pattern):
    """caculates the level of a pattern
       all k-mers a level 0
       a pattern that needs to be split x times with complements to have only k-mers left has level x 

    Args:
        pattern (str): IUPAC pattern

    Returns:
        int: the level
    """
    return sum(len(code[x])-1 for x in pattern)

# TODO: rename to index_cardinality_table
def get_genpat_pos_level(general_pattern):
    return [len(perm_code[x]) for x in general_pattern]

# TODO: rename to prefix_cardinality_table
def get_cum_genpat_pos_level(general_pattern):
    """Create list where the i'th position is the cardinality of general_pattern[:i]

    Args:
        genpat ([type]): [description]

    Returns:
        list with prefix cardinality table
    """
    gppl = get_genpat_pos_level(general_pattern)
    L = []
    s = 1
    for x in gppl:
        L.append(s)
        s *= x
    return L

# TODO: rename
# TODO: Can jeg samle funktionerne i en klasse
# hvor cgppl og gen_pat er sat i __init__?
def pattern2num_new(cgppl, genpat, pat):
    """[summary]

    Args:
        cgppl ([type]): [description]
        genpat ([type]): [description]
        pat ([type]): [description]

    Returns:
        [type]: [description]
    """
    s = 0
    for i in range(len(genpat)):
        s += perm_code_no[genpat[i]][pat[i]] * cgppl[i]
    return s

def pattern2num_new_ord(cgppl, genpat, pat):
    s = 0
    for i in range(len(genpat)):
        s += perm_code_no_np[genpat[i]][pat[i]] * cgppl[i]
    return s

# TODO: Can jeg samle funktionerne i en klasse
# hvor cgppl og gen_pat er sat i __init__?
def num2pattern_new(gppl, genpat, num):
    pat = ''
    for i in range(len(genpat)):
        k = num % gppl[i]
        num = num // gppl[i]
        pat = pat + perm_code[genpat[i]][k]
    return pat

def LCA_pattern(contexts):
    '''Least common ancestor (LCA) pattern af liste af contexts'''
    pat_L = []
    for i in range(len(contexts[0])):
        s = frozenset(x[i] for x in contexts)
        pat_L.append(inv_code[s])
    return ''.join(pat_L)

def LCA_pattern2(patterns):
    '''Least common ancestor (LCA) pattern af liste af contexts'''
    pat_L = []
    for i in range(len(patterns[0])):
        s = frozenset(list(chain(*[code[x[i]] for x in patterns])))
        pat_L.append(inv_code[s])
    return ''.join(pat_L)

# def consensus(oldpat, notpat):
#     n_identical = 0
#     index = -1
#     for i in range(len(oldpat)):
#         if oldpat[i]==notpat[i]:
#             n_identical+=1
#         else:
#             index = i
#     if n_identical == len(oldpat)-1:
#         return oldpat[:index] + minus_set[oldpat[index]][notpat[index]] + oldpat[index+1:]
#     else:
#         return oldpat


def match(pattern, context):
    """check if a k-mer matches a pattern

    Args:
        pattern (str): IUPAC pattern
        context (str): k-mer

    Returns:
        bool: True if context matches pattern
    """
    for (p, c) in zip(pattern, context):
        if c not in code[p]:
            return False
    return True

def matches(pattern):
    """generate all k-mers that match a pattern

    Args:
        pattern (str): IUPAC pattern

    Yields:
        str: k-mer
    """
    if len(pattern) == 0:
        yield ''
    else:
        for y in matches(pattern[1:]):
            for x in code[pattern[0]]:
                yield x+y

def subpatterns_level(pattern, level):
    cur_level = pattern_level(pattern)
    rest_level = cur_level - (len(code[pattern[0]])-1)
    for x in perm_code_min_max[max(0,level - rest_level)][min(level,3)][pattern[0]]:#perm_code[pattern[0]]:
        new_level = cur_level - (len(code[pattern[0]]) - len(code[x]))
        if len(pattern)>1 and (new_level >= level):
            for y in subpatterns_level(pattern[1:], level-(len(code[x])-1)):
                yield x+y
        elif len(pattern)==1 and new_level == level:
            yield x

def subpatterns_level_ord(pattern, level):
    cur_level = pattern_level(pattern)
    rest_level = cur_level - (len(code[pattern[0]])-1)
    for x in perm_code_min_max[max(0,level - rest_level)][min(level,3)][pattern[0]]:#perm_code[pattern[0]]:
        new_level = cur_level - (len(code[pattern[0]]) - len(code[x]))
        if len(pattern)>1 and (new_level >= level):
            for y in subpatterns_level_ord(pattern[1:], level-(len(code[x])-1)):
                yield [ord(x)]+y
        elif len(pattern)==1 and new_level == level:
            yield [ord(x)]

code_lev = {}
code_lev_ord = {}
for x in code:
    code_lev[x] = len(code[x]) -1
    code_lev_ord[ord(x)] = len(code[x]) -1

def subpatterns_level_ord_new(pattern, cur_level, level):
    rest_level = cur_level - code_lev[pattern[0]]
    for x in perm_code_min_max_ord[max(0,level - rest_level)][min(level,3)][pattern[0]]:#perm_code[pattern[0]]:
        new_level = cur_level - (code_lev[pattern[0]] - code_lev_ord[x])
        if len(pattern)>1 and (new_level >= level):
            for y in subpatterns_level_ord_new(pattern[1:],cur_level-code_lev[pattern[0]], level-code_lev_ord[x]):
                yield (x,)+y
        elif len(pattern)==1 and new_level == level:
            yield (x,)

code_lev_ord_np = np.full(90,-100,dtype=int)
for x in code_lev_ord:
    code_lev_ord_np[x] = code_lev_ord[x]

@njit
def subpatterns_level_ord_np(pattern, cur_level, level):
    """Fast solution to generate all subpattern at a specific level

    Args:
        pattern: general_pattern encoded as tuple of ord(x) where x is a IUPAC character
        cur_level: level of general pattern
        level ([type]): [description]

    Yields:
        patterns encoded as tuples of ord(x) where x is a IUPAC character
    """
    #print(pattern, cur_level, level)
    rest_level = cur_level - code_lev_ord_np[pattern[0]]
    imin = max(0,level - rest_level)
    imax = min(level,3)
    for i in range(perm_code_min_max_ord_np_n[imin,imax,pattern[0]]):
        x = perm_code_min_max_ord_np[imin,imax,pattern[0],i]
        new_level = cur_level - (code_lev_ord_np[pattern[0]] - code_lev_ord_np[x])
        if len(pattern)>1 and (new_level >= level):
            for y in subpatterns_level_ord_np(pattern[1:],cur_level-code_lev_ord_np[pattern[0]], level-code_lev_ord_np[x]):
                yield (x,)+y
        elif len(pattern)==1 and new_level == level:
            yield (x,)


def subpatterns2(pattern):
    if len(pattern) == 0:
        yield ''
    else:
        #TODO har aendret nedenstaende linje ved at tilfoeje to-tallet.
        #Er det rigtigt?
        for y in subpatterns2(pattern[1:]):
            for x in perm_code[pattern[0]]:
                yield x+y

def subpatterns(pattern):
    if len(pattern) == 0:
        yield ''
    else:
        for x in perm_code[pattern[0]]:
            for y in subpatterns(pattern[1:]):
                yield x+y


def generality(pat):
    res = 1
    for x in pat:
        res *= len(code[x])
    return res

# def IsSubpattern(pattern, sub_pattern):
#     for (p, c) in zip(pattern, sub_pattern):
#         if c not in set_code2[p]:
#             return False
#     return True

# def IsMutualExlusive(pattern1, pattern2):
#     for (c1, c2) in zip(pattern1, pattern2):
#         if len(set_code[c1]&set_code[c2])==0:
#             #print pattern1, pattern2, perm_code[c1], perm_code[c2]
#             return True
#     return False

def num2pattern(general_pattern, num):
    pat = ''
    for i in range(len(general_pattern)-1,-1,-1):
        k = int(num) % len(perm_code[general_pattern[i]])
        num = num // len(perm_code[general_pattern[i]])
        pat = perm_code[general_pattern[i]][k] + pat
    return pat

def num2pattern2(general_pattern, num):
    pat = ['']
    for i in range(len(general_pattern)-1,-1,-1):
        k = int(num) % len(perm_code[general_pattern[i]])
        num = num // len(perm_code[general_pattern[i]])
        pat = [perm_code[general_pattern[i]][k]] + pat
    return ''.join(pat)


def pattern2num(general_pattern, pattern):
    num = 0
    k = 1
    for i in range(len(general_pattern)-1,-1,-1):
        num += perm_code_no[general_pattern[i]][pattern[i]] * k
        k *= (len(perm_code[general_pattern[i]]))
    return num


def pattern_max(general_pattern):
    res = 1
    for i in range(len(general_pattern)):
        res *= len(perm_code[general_pattern[i]])
    return res

print(code_lev_ord_np)
print(perm_code)
print(get_cum_genpat_pos_level('NNMNN'))
print(perm_code_min_max_ord_np)
for i in range(4):
    for j in range(4):
        print(perm_code_min_max_ord_np[i][j][ord("N")])

print(set_code)
