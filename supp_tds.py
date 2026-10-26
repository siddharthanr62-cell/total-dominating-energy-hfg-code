# ---- Listing 3: minimum total dominating set (self-contained) ----------------
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def open_masks(n, arcs):                 # bitmask of the OPEN neighbourhood N(v)
    nb = [0] * n
    for u, v in arcs:
        nb[u] |= 1 << v; nb[v] |= 1 << u
    return nb

def exact_tds(nb):                       # Algorithm 2: sizes 2, 3, ... ; ALL minimum sets
    n, full = len(nb), (1 << len(nb)) - 1
    if 0 in nb:                          # isolated vertex: no TDS exists
        return None, []
    for k in range(2, n + 1):
        sets = []
        for D in itertools.combinations(range(n), k):
            m = 0
            for v in D:
                m |= nb[v]
            if m == full:
                sets.append(D)
        if sets:
            return k, sets

def ip_tds(nb):                          # min sum x_v  s.t.  sum_{u in N(v)} x_u >= 1
    n = len(nb)
    A = np.array([[(nb[v] >> u) & 1 for u in range(n)] for v in range(n)], float)
    r = milp(np.ones(n), constraints=LinearConstraint(A, lb=1), integrality=np.ones(n),
             bounds=Bounds(0, 1))
    return tuple(int(i) for i in np.flatnonzero(r.x > .5))

def greedy_tds(nb):                      # Algorithm 3 (heuristic): greedy set cover on N(v)
    unc, D = (1 << len(nb)) - 1, []
    while unc:
        u = max(range(len(nb)), key=lambda x: bin(nb[x] & unc).count('1'))
        D.append(u); unc &= ~nb[u]
    return tuple(sorted(D))

arcs5 = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]      # five-vertex example
nb5 = open_masks(5, arcs5)
k, S = exact_tds(nb5)
print('example: gamma_t =', k, 'minimum sets', [[v + 1 for v in D] for D in S])
print('IP optimum', [v + 1 for v in ip_tds(nb5)], '| greedy', [v + 1 for v in greedy_tds(nb5)])

arcs8 = [(0, 1), (0, 5), (1, 7), (2, 4), (2, 6), (3, 5), (3, 6), (3, 7)]   # counterexample
nb8 = open_masks(8, arcs8)
k, S = exact_tds(nb8)
g = greedy_tds(nb8)
print('8-vertex graph: gamma_t =', k, [[v + 1 for v in D] for D in S],
      '| greedy', [v + 1 for v in g], 'size', len(g))
