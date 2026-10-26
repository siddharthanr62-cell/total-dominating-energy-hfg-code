"""Independent verification of the supply-chain HFG example under Option B
(mu_2(v2,v3) = 0.4).  Two eigenvalue methods are compared: numpy eigvalsh and
mpmath eigsy (50 digits).  Everything the paper reports is recomputed here."""
import itertools, json
import numpy as np
import mpmath as mp

mp.mp.dps = 50
n = 5
vert = {0: (.5, .3, .2), 1: (.5, .3, .2), 2: (.5, .3, .2), 3: (.4, .4, .2), 4: (.4, .4, .2)}
edges = {(0, 1): (.4, .3, .2), (0, 2): (.4, .3, .2), (1, 2): (.4, .3, .2),
         (1, 3): (.3, .3, .2), (2, 4): (.3, .3, .2), (3, 4): (.3, .3, .2)}
TOL = 1e-9
res = {}

# ---- (0) vertex sums, edge conditions (1)-(4)
res['vertex_sums'] = {v + 1: round(sum(t), 12) for v, t in vert.items()}
chk = {}
for (u, v), (m, g, b) in edges.items():
    a, c = vert[u], vert[v]
    chk[(u + 1, v + 1)] = dict(
        e1=m <= min(a[0], c[0]) + TOL, e2=g <= max(a[1], c[1]) + TOL,
        e3=b <= min(a[2], c[2]) + TOL, e4=m + g + b <= 1 + TOL, sum=round(m + g + b, 6))
res['edge_checks'] = {str(k): v for k, v in chk.items()}
assert all(all(v[k] for k in ('e1', 'e2', 'e3', 'e4')) for v in chk.values())
assert all(abs(s - 1) < 1e-12 for s in res['vertex_sums'].values())

# ---- (2) strong arcs: strength of connectedness in G - e (Definition 2.3)
def adjacency(E, skip=None):
    d = {v: {} for v in range(n)}
    for e, w in E.items():
        if e != skip:
            d[e[0]][e[1]] = w
            d[e[1]][e[0]] = w
    return d

def all_paths(E, s, t, skip):
    d = adjacency(E, skip)
    out = []
    def dfs(x, seen, ws):
        if x == t:
            out.append(list(ws)); return
        for y, w in d[x].items():
            if y not in seen:
                seen.add(y); ws.append(w); dfs(y, seen, ws); ws.pop(); seen.discard(y)
    dfs(s, {s}, [])
    return out

strong = {}
for e, (m, g, b) in edges.items():
    P = all_paths(edges, e[0], e[1], e)
    if P:
        mu_i = max(min(w[0] for w in p) for p in P)
        ga_i = min(max(w[1] for w in p) for p in P)
        be_i = max(min(w[2] for w in p) for p in P)
    else:
        mu_i, ga_i, be_i = 0.0, 1.0, 0.0
    ok = (m >= mu_i - TOL, g <= ga_i + TOL, b >= be_i - TOL)
    # equivalent formulation: equal to strength of connectedness in G itself
    mu_G = max(m, mu_i); ga_G = min(g, ga_i); be_G = max(b, be_i)
    eq = (abs(m - mu_G) < TOL, abs(g - ga_G) < TOL, abs(b - be_G) < TOL)
    strong[(e[0] + 1, e[1] + 1)] = dict(mu_inf=mu_i, ga_inf=ga_i, be_inf=be_i, ok=ok, eq_in_G=eq)
res['strong'] = {str(k): v for k, v in strong.items()}
assert all(all(v['ok']) and all(v['eq_in_G']) for v in strong.values())

# ---- (3) total domination on the strong-arc graph (all arcs strong here)
S = [e for e in edges if all(strong[(e[0] + 1, e[1] + 1)]['ok'])]
nb = {v: set() for v in range(n)}
for u, v in S:
    nb[u].add(v); nb[v].add(u)
mins = None
for k in range(1, n + 1):
    sets = [D for D in itertools.combinations(range(n), k)
            if all(any(u in nb[v] for u in D) for v in range(n))]
    if sets:
        mins = (k, sets); break
res['gamma_t'] = mins[0]
res['min_TDS'] = [[v + 1 for v in D] for D in mins[1]]
assert mins[0] == 2 and (1, 2) in mins[1]
DT = (1, 2)  # zero-based {v2, v3}

# ---- (4) matrices and energies
def mats(c):
    A = np.zeros((n, n))
    for (u, v), w in edges.items():
        A[u, v] = A[v, u] = w[c]
    return A

def eig_np(M):
    return np.linalg.eigvalsh(M)

def eig_mp(M):
    E = mp.eigsy(mp.matrix(M.tolist()), eigvals_only=True)
    return np.array(sorted(float(x) for x in E))

out = {}
names = ['mu', 'ga', 'be']
for c, name in enumerate(names):
    A = mats(c)
    W = A.sum() / 2
    avg = 2 * W / n
    D = np.diag(A.sum(1))
    T = A.copy()
    for i in DT:
        T[i, i] = 1.0
    DTm = np.diag(T.sum(1))
    Q = D + A
    QT = DTm + T
    L = D - A
    LT = DTm - T
    assert np.allclose(L, LT)  # Theorem 3.7 (TDLE = LE) holds exactly
    ev = {'A': eig_np(A), 'L': eig_np(L), 'Q': eig_np(Q), 'T': eig_np(T), 'QT': eig_np(QT)}
    evm = {'A': eig_mp(A), 'L': eig_mp(L), 'Q': eig_mp(Q), 'T': eig_mp(T), 'QT': eig_mp(QT)}
    for k in ev:
        assert np.allclose(ev[k], evm[k], atol=1e-12), (name, k)
    # characteristic polynomial roots as a third check on T and QT
    for k, M in (('T', T), ('QT', QT)):
        r = np.sort(np.roots(np.poly(M)).real)
        assert np.allclose(r, ev[k], atol=1e-8), (name, k)
    assert abs(ev['T'].sum() - len(DT)) < 1e-12  # trace(T) = |D_T|
    energies = dict(
        E=float(abs(ev['A']).sum()),
        LE=float(abs(ev['L'] - avg).sum()),
        SLE=float(abs(ev['Q'] - avg).sum()),
        TDE=float(abs(ev['T']).sum()),
        STDLE=float(abs(ev['QT'] - avg).sum()))
    energies['TDLE'] = energies['LE']
    trT2 = float(np.trace(T @ T))
    dev = np.abs(ev['QT'] - avg)
    out[name] = dict(
        W=W, avg=avg, deg=A.sum(1).tolist(), degT=T.sum(1).tolist(),
        spec={k: ev[k].tolist() for k in ev}, energies=energies, trT2=trT2,
        tde_lo=float(np.sqrt(trT2)), tde_hi=float(np.sqrt(n * trT2)),
        stdle_lo=float(dev.max()), stdle_hi=float(np.sqrt(n * (dev ** 2).sum())),
        QT=QT.tolist(), A=A.tolist(), T=T.tolist())
res['comp'] = out
res['ratio_mu'] = out['mu']['energies']['STDLE'] / out['mu']['energies']['TDE']
res['ratio_all'] = {k: out[k]['energies']['STDLE'] / out[k]['energies']['TDE'] for k in names}

# ---- ordering claims
order = {}
for k in names:
    e = out[k]['energies']
    order[k] = dict(E_lt_SLE=e['E'] < e['SLE'], SLE_lt_LE=e['SLE'] < e['LE'],
                    LE_lt_TDE=e['LE'] < e['TDE'], TDE_lt_STDLE=e['TDE'] < e['STDLE'],
                    gapE_SLE=e['SLE'] - e['E'])
res['order'] = order
res['hierarchy_mu_gt_ga_gt_be'] = {
    key: out['mu']['energies'][key] > out['ga']['energies'][key] > out['be']['energies'][key]
    for key in ['E', 'LE', 'SLE', 'TDE', 'STDLE']}

json.dump(res, open('results_optionB.json', 'w'), indent=1, default=str)

f = lambda x: f"{x:.4f}"
print('gamma_t =', res['gamma_t'], ' minimum total dominating sets:', res['min_TDS'])
for k in names:
    c = out[k]
    print(f"--- {k}: W={c['W']:.2f} 2W/n={c['avg']:.2f} deg={c['deg']} degT={c['degT']} trT2={c['trT2']:.4f}")
    for s in ['A', 'L', 'Q', 'T', 'QT']:
        print('   spec', s, [f(x) for x in c['spec'][s]])
    print('   energies', {a: f(b) for a, b in c['energies'].items()})
    print('   TDE bounds', f(c['tde_lo']), f(c['tde_hi']), ' STDLE bounds', f(c['stdle_lo']), f(c['stdle_hi']))
print('ratio STDLE/TDE', {k: round(v, 4) for k, v in res['ratio_all'].items()})
print('order', json.dumps(order, default=str))
print('hierarchy mu>ga>be', res['hierarchy_mu_gt_ga_gt_be'])
print('strong arcs:', {k: (v['mu_inf'], v['ga_inf'], v['be_inf']) for k, v in strong.items()})
print('ALL ASSERTIONS PASSED')
