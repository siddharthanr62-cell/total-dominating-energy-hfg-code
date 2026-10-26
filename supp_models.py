"""Step 5 (final revision): reviewer-2 items Q2 (effect of n, edges, hesitancy) and Q7 (comparison with
fuzzy / intuitionistic fuzzy graphs).  Self-contained apart from tds_lib.py.
Run: python3 supp_models.py     (about 1 min; writes results_models.json)"""
import itertools, json, math, random
import numpy as np
from tds_lib import (strong_arcs_fast, _tree_bottleneck, open_masks, has_total_dominating_set,
                     exhaustive_min_tds)

EIG = np.linalg.eigvalsh
TOL = 1e-9
R = {}


def hdr(s):
    print('\n' + '=' * 78 + '\n' + s + '\n' + '=' * 78)


def mat(n, edges, c):
    A = np.zeros((n, n))
    for (u, v), w in edges.items():
        A[u, v] = A[v, u] = w[c]
    return A


def TDE_STDLE(A, D):
    n = A.shape[0]
    delta = np.zeros(n)
    delta[list(D)] = 1.0
    T = A + np.diag(delta)
    c = A.sum() / n                      # 2W/n
    QT = np.diag(T.sum(1)) + T
    return abs(EIG(T)).sum(), abs(EIG(QT) - c).sum()


def strong_model(n, edges, model):
    """Strong arcs when only the first k components are used: FG (mu), IFG (mu,gamma), HFG (mu,gamma,beta)."""
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
    S = set()
    for (u, v), (m, g, b) in edges.items():
        ok = m >= Bm[u][v] - TOL
        if model >= 2:
            ok = ok and g <= Bg[u][v] + TOL
        if model >= 3:
            ok = ok and b >= Bb[u][v] - TOL
        if ok:
            S.add((u, v))
    return S


def rule_S(n, arcs, edges, S_sets):
    """Rule (S): maximise omega(D)=sum of mu2 over strong arcs meeting D; first maximiser (ties -> smallest tuple)."""
    best, bw = None, -1
    for D in S_sets:
        w = sum(edges[e][0] for e in arcs if e[0] in D or e[1] in D)
        if w > bw + 1e-12:
            best, bw = D, w
    return best


# ============================================================ 0. the example
hdr('0. Example')
edges5 = {(0, 1): (.4, .3, .2), (0, 2): (.4, .3, .2), (1, 2): (.4, .3, .2),
          (1, 3): (.3, .3, .2), (2, 4): (.3, .3, .2), (3, 4): (.3, .3, .2)}
n5 = 5
for model, nm in [(1, 'FG'), (2, 'IFG'), (3, 'HFG')]:
    arcs = strong_model(n5, edges5, model)
    nb = open_masks(n5, arcs)
    g, S = exhaustive_min_tds(nb)
    D = rule_S(n5, arcs, edges5, S)
    te, st = TDE_STDLE(mat(n5, edges5, 0), D)
    print(f'{nm:4s} strong arcs={len(arcs)}  gamma_t={g}  min sets={S}  Rule S -> {D}  TDE_mu={te:.4f}  STDLE_mu={st:.4f}')
    assert len(arcs) == 6 and g == 2 and D == (1, 2) and abs(te - 3.2236) < 6e-5 and abs(st - 5.4960) < 6e-5
R['example_models'] = 'FG, IFG and HFG give the same strong arcs (all 6) and the same D_T on the example'

# ============================================================ 1. Q2 (a): lower bounds that grow with n
hdr('1. Q2: gamma_t >= n/Delta(G_s); hence TDE >= n/Delta, STDLE >= 2n/Delta (random weighted graphs)')
rg = random.Random(2026)
cnt = bad = 0
for _ in range(3000):
    n = rg.randint(4, 13)
    p = rg.uniform(.2, .9)
    arcs = [e for e in itertools.combinations(range(n), 2) if rg.random() < p]
    nb = open_masks(n, arcs)
    if not has_total_dominating_set(nb):
        continue
    g, S = exhaustive_min_tds(nb)
    Dmax = max(bin(m).count('1') for m in nb)
    A = np.zeros((n, n))
    for u, v in arcs:
        A[u, v] = A[v, u] = rg.uniform(.01, 1)
    te, st = TDE_STDLE(A, S[0])
    cnt += 1
    ok = g >= n / Dmax - 1e-12 and te >= g - 1e-9 and st >= 2 * g - 1e-9 and te >= n / Dmax - 1e-9 and st >= 2 * n / Dmax - 1e-9
    bad += (not ok)
print(f'{cnt} instances, violations: {bad}')
assert bad == 0
R['lower_bound_trials'] = cnt

# ============================================================ 2. Q2 (b): scaling a component (hesitancy values)
hdr('2. Q2: scaling the beta weights by s in (0,1]  (strong arcs, hence D_T, do not change; limits TDE->gamma_t, STDLE->2 gamma_t)')
rows = []
for s in [1.0, 0.75, 0.5, 0.25, 0.1, 0.01, 1e-4]:
    e2 = {k: (m, g, b * s) for k, (m, g, b) in edges5.items()}
    assert strong_model(n5, e2, 3) == strong_model(n5, edges5, 3)       # scaling preserves strong arcs
    te, st = TDE_STDLE(mat(n5, e2, 2), (1, 2))
    rows.append((s, te, st))
    print(f's={s:<8} TDE_beta={te:.4f}  STDLE_beta={st:.4f}')
assert abs(rows[-1][1] - 2) < 1e-3 and abs(rows[-1][2] - 4) < 1e-3
# monotone in s on this example?
R['scale_beta'] = [dict(s=s, TDE=t, STDLE=q) for s, t, q in rows]
mono_te = all(rows[i][1] >= rows[i + 1][1] - 1e-12 for i in range(len(rows) - 1))
mono_st = all(rows[i][2] >= rows[i + 1][2] - 1e-12 for i in range(len(rows) - 1))
print('monotone decreasing as s decreases on the example: TDE', mono_te, ' STDLE', mono_st)
# components are independent: changing beta leaves mu and gamma indices unchanged when strong arcs are unchanged
e3 = {k: (m, g, b * .5) for k, (m, g, b) in edges5.items()}
for c in (0, 1):
    assert np.allclose(TDE_STDLE(mat(n5, e3, c), (1, 2)), TDE_STDLE(mat(n5, edges5, c), (1, 2)))
print('mu and gamma indices unchanged when only beta is rescaled (strong arcs unchanged): OK')

# random check of the limits and of the monotonicity question
rg = random.Random(7)
nonmono = trials = 0
for _ in range(500):
    n = rg.randint(5, 10)
    arcs = [e for e in itertools.combinations(range(n), 2) if rg.random() < .5]
    nb = open_masks(n, arcs)
    if not has_total_dominating_set(nb):
        continue
    g, S = exhaustive_min_tds(nb)
    A0 = np.zeros((n, n))
    for u, v in arcs:
        A0[u, v] = A0[v, u] = rg.uniform(.1, 1)
    prev = None
    ok = True
    for s in [1, .75, .5, .25, .1]:
        te, st = TDE_STDLE(A0 * s, S[0])
        if prev and (te > prev[0] + 1e-12 or st > prev[1] + 1e-12):
            ok = False
        prev = (te, st)
    te0, st0 = TDE_STDLE(A0 * 1e-6, S[0])
    assert abs(te0 - g) < 1e-3 and abs(st0 - 2 * g) < 1e-3
    trials += 1
    nonmono += (not ok)
print(f'limit s->0 gives gamma_t and 2 gamma_t in all {trials} random cases; '
      f'cases where TDE or STDLE is not monotone in s: {nonmono}')
R['scale_trials'] = dict(trials=trials, nonmonotone=nonmono)

# ============================================================ 3. Q2 (c): adding edges to the example
hdr('3. Q2: adding edges to the five-vertex example (new edges (0.3,0.3,0.2); strong arcs and D_T recomputed)')
extra = [(0, 3), (1, 4), (2, 3), (0, 4)]
cur = dict(edges5)
rows = []
for k in range(len(extra) + 1):
    if k:
        cur[extra[k - 1]] = (.3, .3, .2)
    arcs = strong_model(n5, cur, 3)
    nb = open_masks(n5, arcs)
    g, S = exhaustive_min_tds(nb)
    D = rule_S(n5, arcs, cur, S)
    te, st = TDE_STDLE(mat(n5, cur, 0), D)
    # also the Laplacian-type energies for context
    A = mat(n5, cur, 0)
    c = A.sum() / n5
    Dg = np.diag(A.sum(1))
    LE = abs(EIG(Dg - A) - c).sum()
    rows.append(dict(edges=len(cur), strong=len(arcs), gamma_t=g, D=[x + 1 for x in D], nsets=len(S), TDE=te, STDLE=st, LE=LE))
    print(f"|E|={len(cur)} strong={len(arcs)} gamma_t={g} D_T={[x+1 for x in D]} (#min sets {len(S)}) TDE_mu={te:.4f} STDLE_mu={st:.4f} LE_mu={LE:.4f}")
R['add_edges'] = rows

# ============================================================ 4. Q2 (d): growth with n on random uniform HFGs
hdr('4. Q2: growth with n (uniform-weight HFGs, every edge strong, w = 0.4; Rule S irrelevant for uniform weights up to symmetry)')
rg = random.Random(99)
tab = []
for d in (3, 4):
    for n in (6, 8, 10, 12, 14, 16):
        vals = []
        tries = 0
        while len(vals) < 100 and tries < 5000:
            tries += 1
            p = min(1.0, d / (n - 1))
            arcs = [e for e in itertools.combinations(range(n), 2) if rg.random() < p]
            nb = open_masks(n, arcs)
            if not has_total_dominating_set(nb):
                continue
            g, S = exhaustive_min_tds(nb)
            A = np.zeros((n, n))
            for u, v in arcs:
                A[u, v] = A[v, u] = 0.4
            te, st = TDE_STDLE(A, S[0])
            vals.append((g, te, st, len(arcs)))
        v = np.array(vals)
        tab.append(dict(d=d, n=n, m=float(v[:, 3].mean()), gamma_t=float(v[:, 0].mean()), TDE=float(v[:, 1].mean()),
                        STDLE=float(v[:, 2].mean())))
        print(f"d={d} n={n:2d} mean|E|={v[:,3].mean():5.1f} mean gamma_t={v[:,0].mean():.2f} mean TDE_mu={v[:,1].mean():.3f} mean STDLE_mu={v[:,2].mean():.3f}")
R['growth_n'] = tab

# ============================================================ 5. Q7: comparison with FG and IFG
hdr('5. Q7: FG (mu), IFG (mu,gamma) and HFG (mu,gamma,beta) on random HFGs')
rg = random.Random(31337)
M = [.2, .4, .6]
GB = [.1, .2]
res = {}
for n in (6, 8, 10):
    N = 2000
    ex_ok = [0, 0, 0]
    nested = 0
    smaller_ifg_hfg = smaller_fg_ifg = 0
    both = 0
    same_sets = 0
    diff_te = []
    gt_diff = 0
    for _ in range(N):
        edges = {}
        for e in itertools.combinations(range(n), 2):
            if rg.random() < .5:
                edges[e] = (rg.choice(M), rg.choice(GB), rg.choice(GB))
        if not edges:
            continue
        # vertex triples (0.6,0.2,0.2): mu2<=.6, gamma2<=.2, beta2<=.2 -> valid HFG
        S3 = [strong_model(n, edges, k) for k in (1, 2, 3)]
        assert S3[2] <= S3[1] <= S3[0]
        nested += 1
        smaller_fg_ifg += (len(S3[1]) < len(S3[0]))
        smaller_ifg_hfg += (len(S3[2]) < len(S3[1]))
        info = []
        for k in range(3):
            nb = open_masks(n, S3[k])
            if has_total_dominating_set(nb):
                ex_ok[k] += 1
                info.append(exhaustive_min_tds(nb))
            else:
                info.append(None)
        if all(i is not None for i in info):
            both += 1
            gt = [i[0] for i in info]
            gt_diff += (len(set(gt)) > 1)
            same_sets += (info[0][1] == info[1][1] == info[2][1])
            A = mat(n, edges, 0)
            tes = [TDE_STDLE(A, rule_S(n, S3[k], edges, info[k][1]))[0] for k in range(3)]
            diff_te.append(max(tes) - min(tes))
    res[n] = dict(N=nested, tds_FG=ex_ok[0], tds_IFG=ex_ok[1], tds_HFG=ex_ok[2], fewer_arcs_IFG_vs_FG=smaller_fg_ifg,
                  fewer_arcs_HFG_vs_IFG=smaller_ifg_hfg, solvable_all=both, same_min_sets=same_sets, gamma_t_differs=gt_diff,
                  mean_TDE_mu_spread=float(np.mean(diff_te)) if diff_te else None)
    r = res[n]
    print(f"n={n}: {r['N']} HFGs | TDS exists: FG {r['tds_FG']}, IFG {r['tds_IFG']}, HFG {r['tds_HFG']} | "
          f"IFG has fewer strong arcs than FG in {r['fewer_arcs_IFG_vs_FG']}, HFG fewer than IFG in {r['fewer_arcs_HFG_vs_IFG']} | "
          f"solvable in all three: {r['solvable_all']}, identical minimum sets: {r['same_min_sets']}, gamma_t differs: {r['gamma_t_differs']}, "
          f"mean spread of TDE_mu: {r['mean_TDE_mu_spread']}")
R['models_random'] = res

# a small illustration where the hesitancy condition changes the strong arcs of the example
hdr('5b. Q7: the example with beta(v1v2) lowered to 0.1  (IFG ignores beta, HFG does not)')
ex2 = dict(edges5)
ex2[(0, 1)] = (.4, .3, .1)
ill = {}
for model, nm in [(1, 'FG'), (2, 'IFG'), (3, 'HFG')]:
    arcs = strong_model(n5, ex2, model)
    nb = open_masks(n5, arcs)
    if has_total_dominating_set(nb):
        g, S = exhaustive_min_tds(nb)
        D = rule_S(n5, arcs, ex2, S)
        te, st = TDE_STDLE(mat(n5, ex2, 0), D)
        ill[nm] = dict(arcs=sorted((u + 1, v + 1) for u, v in arcs), gamma_t=g, sets=[[x + 1 for x in s] for s in S],
                       D=[x + 1 for x in D], TDE_mu=te, STDLE_mu=st)
    else:
        ill[nm] = dict(arcs=sorted((u + 1, v + 1) for u, v in arcs), gamma_t=None)
    print(nm, ill[nm])
R['illustration'] = ill


hdr('5c. Q7: the triangle of Lemma 5.1 (v1 isolated in the HFG strong-arc graph)')
tri = {(0, 1): (.1, .2, .3), (0, 2): (.1, .1, .1), (1, 2): (.4, .1, .2)}
ill3 = {}
for model, nm in [(1, 'FG'), (2, 'IFG'), (3, 'HFG')]:
    arcs = strong_model(3, tri, model)
    nb = open_masks(3, arcs)
    if has_total_dominating_set(nb):
        g, S = exhaustive_min_tds(nb)
        D = rule_S(3, arcs, tri, S)
        te, st = TDE_STDLE(mat(3, tri, 0), D)
        ill3[nm] = dict(arcs=sorted((u + 1, v + 1) for u, v in arcs), gamma_t=g, sets=[[x + 1 for x in s_] for s_ in S],
                        D=[x + 1 for x in D], TDE_mu=float(te), STDLE_mu=float(st))
    else:
        ill3[nm] = dict(arcs=sorted((u + 1, v + 1) for u, v in arcs), gamma_t=None)
    print(nm, ill3[nm])
R['triangle'] = ill3
assert ill3['HFG']['gamma_t'] is None and ill3['FG']['gamma_t'] == 2 and ill3['IFG']['gamma_t'] == 2

json.dump(R, open('results_models.json', 'w'), indent=1, default=str)
print('\nall assertions passed; results_models.json written')
