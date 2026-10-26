"""Supplementary script 1: five-vertex supply chain HFG (Sections 4 and 5).
Checks the HFG conditions, strong arcs, minimum total dominating sets, the six energy indices
and their bounds, the properties of TDE (Theorem 3.9) and the three minimum sets (Section 4.7)."""
import itertools
import numpy as np

n = 5
# vertices (mu1, gamma1, beta1) and edges (mu2, gamma2, beta2), 0-based
vert = {0: (.5, .3, .2), 1: (.5, .3, .2), 2: (.5, .3, .2),
        3: (.4, .4, .2), 4: (.4, .4, .2)}
edges = {(0, 1): (.4, .3, .2), (0, 2): (.4, .3, .2), (1, 2): (.4, .3, .2),
         (1, 3): (.3, .3, .2), (2, 4): (.3, .3, .2), (3, 4): (.3, .3, .2)}
tol = 1e-9

def check_hfg():
    """Vertex rule (0) and edge conditions (1)-(4)."""
    ok = all(abs(sum(t) - 1) < tol for t in vert.values())
    for (u, v), (m, g, b) in edges.items():
        a, c = vert[u], vert[v]
        ok &= (m <= min(a[0], c[0]) + tol and g <= max(a[1], c[1]) + tol
               and b <= min(a[2], c[2]) + tol and m + g + b <= 1 + tol)
    return ok

def alt_paths(e):
    """Weights along all paths between the ends of e in G - e."""
    adj = {v: [] for v in range(n)}
    for f, w in edges.items():
        if f != e:
            adj[f[0]].append((f[1], w))
            adj[f[1]].append((f[0], w))
    out = []
    def dfs(x, seen, ws):
        if x == e[1]:
            out.append(ws)
            return
        for y, w in adj[x]:
            if y not in seen:
                dfs(y, seen | {y}, ws + [w])
    dfs(e[0], {e[0]}, [])
    return out

def is_strong(e):
    """Strong arc: strength of connectedness in G - e (Definition 2.3)."""
    P = alt_paths(e)
    mu_i = max((min(w[0] for w in p) for p in P), default=0.0)
    ga_i = min((max(w[1] for w in p) for p in P), default=1.0)
    be_i = max((min(w[2] for w in p) for p in P), default=0.0)
    m, g, b = edges[e]
    return m >= mu_i - tol and g <= ga_i + tol and b >= be_i - tol

def min_total_dominating_sets():
    nb = {v: set() for v in range(n)}
    for e in edges:
        if is_strong(e):
            nb[e[0]].add(e[1])
            nb[e[1]].add(e[0])
    for k in range(2, n + 1):
        S = [D for D in itertools.combinations(range(n), k)
             if all(nb[v] & set(D) for v in range(n))]
        if S:
            return k, S

def matrix(c):
    A = np.zeros((n, n))
    for (u, v), w in edges.items():
        A[u, v] = A[v, u] = w[c]
    return A

def energies(A, DT):
    W = A.sum() / 2
    avg = 2 * W / n
    D = np.diag(A.sum(1))
    T = A + np.diag([1.0 if i in DT else 0.0 for i in range(n)])
    DTm = np.diag(T.sum(1))
    ev = np.linalg.eigvalsh
    assert np.allclose(DTm - T, D - A)      # L_T = L, hence TDLE = LE
    dev = abs(ev(DTm + T) - avg)
    tr = np.trace(T @ T)
    return {'E': abs(ev(A)).sum(),
            'LE': abs(ev(D - A) - avg).sum(),
            'SLE': abs(ev(D + A) - avg).sum(),
            'TDE': abs(ev(T)).sum(),
            'STDLE': dev.sum(),
            'tr': tr,
            'bT': (np.sqrt(tr), np.sqrt(n * tr)),
            'bS': (dev.max(), np.sqrt(n * (dev ** 2).sum()))}

print('HFG conditions (0)-(4) hold:', check_hfg())
print('Strong arcs:', sum(map(is_strong, edges)), 'of', len(edges))
k, S = min_total_dominating_sets()
print('gamma_t =', k, '; minimum sets:', [[v + 1 for v in D] for D in S])

DT = (1, 2)                                  # D_T = {v2, v3}, 0-based
R = [energies(matrix(c), DT) for c in range(3)]
line = '=' * 44
print(line)
print(f"{'Index':8s}{'mu':>12s}{'gamma':>12s}{'beta':>12s}")
print(line)
for key in ['E', 'LE', 'SLE', 'TDE', 'STDLE']:
    print(f"{key:8s}" + ''.join(f"{r[key]:12.4f}" for r in R))
print(line)
for c, name in enumerate(['mu', 'gamma', 'beta']):
    r = R[c]
    print(f"{name:6s} tr(T^2)={r['tr']:.4f}  "
          f"{r['bT'][0]:.4f} <= TDE={r['TDE']:.4f} <= {r['bT'][1]:.4f}")
for c, name in enumerate(['mu', 'gamma', 'beta']):
    r = R[c]
    print(f"{name:6s} {r['bS'][0]:.4f} <= STDLE={r['STDLE']:.4f} "
          f"<= {r['bS'][1]:.4f}")
print('STDLE/TDE (mu):', round(R[0]['STDLE'] / R[0]['TDE'], 4))
# ---- Listing 2 (continues Listing 1): properties of TDE, non-uniqueness
def T_of(A, D):
    return A + np.diag([1.0 if i in D else 0.0 for i in range(n)])

print('Properties of TDE for D_T = {v2, v3}')
for c, name in enumerate(['mu', 'gamma', 'beta']):
    A = matrix(c)
    T = T_of(A, DT)
    la, lt = np.linalg.eigvalsh(A), np.linalg.eigvalsh(T)
    sw2 = (A ** 2).sum() / 2                   # sum of squared edge weights
    E, TDE = abs(la).sum(), abs(lt).sum()
    print(f"{name:6s} tr(T)={np.trace(T):.0f}  "
          f"shifts in [{(lt - la).min():.4f}, {(lt - la).max():.4f}]  "
          f"TDE-E={TDE - E:.4f}")
    print(f"       tr(T^2)={np.trace(T @ T):.4f} = 2*{sw2:.2f} + 2;  "
          f"TDE >= 2: {TDE >= 2}")

print('All minimum total dominating sets (omega = mu-weight of')
print('strong arcs incident with D):')
for D in S:
    omega = sum(edges[e][0] for e in edges if e[0] in D or e[1] in D)
    R = [energies(matrix(c), D) for c in range(3)]
    print(f"D={[v + 1 for v in D]}  omega={omega:.1f}")
    print('  TDE  ', ' '.join(f"{r['TDE']:.4f}" for r in R))
    print('  STDLE', ' '.join(f"{r['STDLE']:.4f}" for r in R))
