"""Core routines for Step 3 (strong-arc graph, minimum total dominating sets, heuristics).
Vertices are 0..n-1.  A graph is given by neighbourhood bitmasks nb[v] (OPEN neighbourhoods)."""
import itertools
import numpy as np

TOL = 1e-9


# ------------------------------------------------------------------ strong-arc graph
def strong_arcs_definition(n, edges):
    """Literal Definition 2.3: compare each edge with all simple u-v paths of G-e.
    edges: dict {(u,v): (mu2,gamma2,beta2)}, u<v.  Exponential; used only as reference."""
    adj = {v: [] for v in range(n)}
    for (u, v), w in edges.items():
        adj[u].append((v, w)); adj[v].append((u, w))
    strong = set()
    for e, (m, g, b) in edges.items():
        paths = []

        def dfs(x, seen, ws):
            if x == e[1]:
                paths.append(ws); return
            for y, w in adj[x]:
                if y not in seen and not ({x, y} == set(e)):
                    dfs(y, seen | {y}, ws + [w])
        dfs(e[0], {e[0]}, [])
        mu_i = max((min(w[0] for w in p) for p in paths), default=0.0)
        ga_i = min((max(w[1] for w in p) for p in paths), default=1.0)
        be_i = max((min(w[2] for w in p) for p in paths), default=0.0)
        if m >= mu_i - TOL and g <= ga_i + TOL and b >= be_i - TOL:
            strong.add(e)
    return strong


def _tree_bottleneck(n, W, maximize):
    """Spanning forest by Prim (dense, O(n^2)) and all-pairs bottleneck value on it (O(n^2)).
    W[i][j] = weight or None for a non-edge.  Returns B with B[i][j] = max-min (maximize=True)
    or min-max (maximize=False) path value in G between i and j (None if disconnected)."""
    sign = 1.0 if maximize else -1.0
    in_t = [False] * n
    tree = {v: [] for v in range(n)}
    for root in range(n):
        if in_t[root]:
            continue
        best = {root: (-np.inf, None)}
        in_t[root] = True
        key = [(-np.inf, None)] * n
        for j in range(n):
            if not in_t[j] and W[root][j] is not None:
                key[j] = (sign * W[root][j], root)
        while True:
            cand = [(key[j][0], j) for j in range(n) if not in_t[j] and key[j][1] is not None]
            if not cand:
                break
            _, j = max(cand)
            p = key[j][1]
            in_t[j] = True
            tree[p].append((j, W[p][j])); tree[j].append((p, W[p][j]))
            for k in range(n):
                if not in_t[k] and W[j][k] is not None and (key[k][1] is None or sign * W[j][k] > key[k][0]):
                    key[k] = (sign * W[j][k], j)
    B = [[None] * n for _ in range(n)]
    for s in range(n):
        B[s][s] = np.inf if maximize else -np.inf
        stack = [s]
        while stack:
            x = stack.pop()
            for y, w in tree[x]:
                if B[s][y] is None:
                    B[s][y] = min(B[s][x], w) if maximize else max(B[s][x], w)
                    stack.append(y)
    return B


def strong_arcs_fast(n, edges):
    """O(n^2) strong-arc graph.  Edge e=uv is strong in the mu-component iff mu2(e) equals the
    strength of connectedness CONN_mu(u,v) of u,v in G (max-min path value), because
    CONN(u,v) = max(mu2(e), mu_inf_e).  Same for beta (max-min) and gamma (min-max)."""
    Wm = [[None] * n for _ in range(n)]
    Wg = [[None] * n for _ in range(n)]
    Wb = [[None] * n for _ in range(n)]
    for (u, v), (m, g, b) in edges.items():
        Wm[u][v] = Wm[v][u] = m
        Wg[u][v] = Wg[v][u] = g
        Wb[u][v] = Wb[v][u] = b
    Bm = _tree_bottleneck(n, Wm, True)
    Bg = _tree_bottleneck(n, Wg, False)
    Bb = _tree_bottleneck(n, Wb, True)
    strong = set()
    for (u, v), (m, g, b) in edges.items():
        if m >= Bm[u][v] - TOL and g <= Bg[u][v] + TOL and b >= Bb[u][v] - TOL:
            strong.add((u, v))
    return strong


def open_masks(n, arcs):
    nb = [0] * n
    for u, v in arcs:
        nb[u] |= 1 << v
        nb[v] |= 1 << u
    return nb


# ------------------------------------------------------------------ existence
def has_total_dominating_set(nb):
    return all(m != 0 for m in nb)


# ------------------------------------------------------------------ exact: exhaustive bitmask
def exhaustive_min_tds(nb):
    """Subsets in order of increasing size; first size that works; ALL minimum sets returned.
    Returns (gamma_t, [sorted tuples])."""
    n = len(nb)
    full = (1 << n) - 1
    if not has_total_dominating_set(nb):
        return None, []
    for k in range(2, n + 1):
        found = []
        for comb in itertools.combinations(range(n), k):
            m = 0
            for v in comb:
                m |= nb[v]
            if m == full:
                found.append(comb)
        if found:
            return k, found
    raise AssertionError('unreachable: V is always a TDS when no vertex is isolated')


# ------------------------------------------------------------------ exact: integer program
def ip_min_tds(nb, extra_cuts=()):
    """min sum x_v  s.t. sum_{u in N(v)} x_u >= 1 for all v, x binary (scipy.optimize.milp)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    n = len(nb)
    A = np.array([[1.0 if (nb[v] >> u) & 1 else 0.0 for u in range(n)] for v in range(n)])
    cons = [LinearConstraint(A, lb=np.ones(n), ub=np.inf)]
    for D in extra_cuts:          # no-good cut excluding the set D
        row = np.array([-1.0 if u in D else 1.0 for u in range(n)])
        cons.append(LinearConstraint(row.reshape(1, -1), lb=1.0 - len(D), ub=np.inf))
    res = milp(c=np.ones(n), constraints=cons, integrality=np.ones(n), bounds=Bounds(0, 1))
    if res.status != 0:
        return None, None
    D = tuple(int(i) for i in np.flatnonzero(res.x > 0.5))
    return len(D), D


def ip_all_min_tds(nb):
    """All minimum sets by re-solving with no-good cuts until the optimum value increases."""
    k0, D = ip_min_tds(nb)
    if k0 is None:
        return None, []
    out, cuts = [D], [D]
    while True:
        k, D = ip_min_tds(nb, cuts)
        if k is None or k > k0:
            return k0, sorted(out)
        out.append(D); cuts.append(D)


# ------------------------------------------------------------------ heuristics
def greedy_open(nb):
    """Corrected greedy = greedy set cover with the OPEN neighbourhoods N(v) over the universe V:
    repeatedly add the vertex u whose open neighbourhood covers most uncovered vertices
    (ties: smallest index).  Every vertex is covered by a CHOSEN neighbour, so the output is a
    total dominating set by construction.  Guarantee: |D| <= H(Delta) * gamma_t, Delta = max degree."""
    n = len(nb)
    unc = (1 << n) - 1
    D = []
    while unc:
        best, bu = 0, None
        for u in range(n):
            c = bin(nb[u] & unc).count('1')
            if c > best:
                best, bu = c, u
        D.append(bu)
        unc &= ~nb[bu]
    return tuple(sorted(D))


def greedy_closed_naive(nb):
    """The rule of the ORIGINAL Algorithm 1 read literally: a chosen vertex counts as covered itself
    (closed neighbourhood N[v]).  Output is a dominating set, in general not a total one."""
    n = len(nb)
    unc = (1 << n) - 1
    D = []
    while unc:
        best, bu = 0, None
        for u in range(n):
            c = bin((nb[u] | (1 << u)) & unc).count('1')
            if c > best:
                best, bu = c, u
        D.append(bu)
        unc &= ~(nb[bu] | (1 << bu))
    return tuple(sorted(D))


def repair(nb, D):
    """Repair step: every chosen vertex without a chosen neighbour gets one neighbour added."""
    D = set(D)
    dm = sum(1 << d for d in D)
    for v in sorted(D):
        if nb[v] & dm == 0:
            u = min(i for i in range(len(nb)) if (nb[v] >> i) & 1)
            D.add(u); dm |= 1 << u
    return tuple(sorted(D))


def greedy_closed_repaired(nb):
    return repair(nb, greedy_closed_naive(nb))


def is_dominating(nb, D):
    n = len(nb)
    dm = sum(1 << d for d in D)
    return all((dm >> v) & 1 or nb[v] & dm for v in range(n))


def is_total_dominating(nb, D):
    dm = sum(1 << d for d in D)
    return all(nb[v] & dm for v in range(len(nb)))


def all_greedy_outcomes(nb, closed=False):
    """Set of sizes of the outputs of the greedy rule over ALL tie-breakings."""
    n = len(nb)
    masks = [(nb[u] | (1 << u)) if closed else nb[u] for u in range(n)]
    sizes, seen = set(), set()

    def rec(unc, chosen):
        if unc == 0:
            sizes.add(len(chosen)); return
        key = (unc, chosen)
        if key in seen:
            return
        seen.add(key)
        cs = [bin(masks[u] & unc).count('1') for u in range(n)]
        mx = max(cs)
        for u in range(n):
            if cs[u] == mx:
                rec(unc & ~masks[u], tuple(sorted(chosen + (u,))))
    rec((1 << n) - 1, ())
    return sizes
