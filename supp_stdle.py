"""Supplementary script: structure of Q_T, centring, STDLE bounds, sufficient condition, counterexample search, figures.
Run: python3 supp_stdle.py   (needs tds_lib.py; writes figs/ and results_step4.json)"""
import itertools, json, math, os, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tds_lib import *

EIG = np.linalg.eigvalsh
OUT = 'figs'
os.makedirs(OUT, exist_ok=True)
R = {}
TOLN = 1e-9


def hdr(s):
    print('\n' + '=' * 78 + '\n' + s + '\n' + '=' * 78)


# ------------------------------------------------------------------------ matrices
def parts(A, D):
    """All matrices of one component for a weight matrix A and total dominating set D."""
    n = A.shape[0]
    delta = np.array([1.0 if i in D else 0.0 for i in range(n)])
    Dg = np.diag(A.sum(1))
    Dl = np.diag(delta)
    T = A + Dl
    DT = np.diag(T.sum(1))
    return dict(n=n, A=A, delta=delta, Dg=Dg, L=Dg - A, Q=Dg + A, T=T, DT=DT, LT=DT - T, QT=DT + T,
                W=A.sum() / 2.0, gt=len(D))


def indices(P):
    n, c = P['n'], 2 * P['W'] / P['n']
    ev = EIG
    return dict(E=abs(ev(P['A'])).sum(), LE=abs(ev(P['L']) - c).sum(), SLE=abs(ev(P['Q']) - c).sum(),
                TDE=abs(ev(P['T'])).sum(), TDLE=abs(ev(P['LT']) - c).sum(), STDLE=abs(ev(P['QT']) - c).sum(),
                STDLE_c=abs(ev(P['QT']) - (2 * P['W'] + 2 * P['gt']) / n).sum())


def LE_bound(P):
    """B = sqrt(n (sum d^2 + 2 sum w^2 - 4W^2/n)) : Cauchy-Schwarz upper bound for LE and SLE."""
    d = P['A'].sum(1)
    S2 = (P['A'] ** 2).sum() / 2
    return math.sqrt(P['n'] * ((d ** 2).sum() + 2 * S2 - 4 * P['W'] ** 2 / P['n']))


def structure_checks(P):
    """Part A and B identities / inequalities.  Returns dict of booleans and numbers."""
    n, g, W = P['n'], P['gt'], P['W']
    c, cT = 2 * W / n, (2 * W + 2 * g) / n
    rho, tau = EIG(P['Q']), EIG(P['QT'])
    d = P['A'].sum(1)
    S2 = (P['A'] ** 2).sum() / 2
    dl = P['delta']
    trQT2 = np.trace(P['QT'] @ P['QT'])
    form = ((d + 2 * dl) ** 2).sum() + 2 * S2
    sq = ((tau - c) ** 2).sum()
    sq_form = trQT2 - 2 * c * np.trace(P['QT']) + n * c * c
    sq_closed = ((d + 2 * dl) ** 2).sum() + 2 * S2 - 4 * W ** 2 / n - 8 * W * g / n
    ST = abs(tau - c).sum()
    ok = dict(
        Q_T_eq_Q_plus_2Delta=np.allclose(P['QT'], P['Q'] + 2 * np.diag(dl)),
        L_T_eq_L=np.allclose(P['LT'], P['L']),
        Q_psd=rho.min() > -TOLN, QT_psd=tau.min() > -TOLN,
        interlace=bool(np.all(tau >= rho - TOLN) and np.all(tau <= rho + 2 + TOLN)),
        shift_sum=abs((tau - rho).sum() - 2 * g) < TOLN,
        tr_QT=abs(tau.sum() - (2 * W + 2 * g)) < TOLN,
        tr_QT2=abs(trQT2 - form) < TOLN,
        mean_Q=abs(rho.mean() - c) < TOLN, mean_QT=abs(tau.mean() - cT) < TOLN,
        sq_identity=abs(sq - sq_form) < TOLN and abs(sq - sq_closed) < TOLN,
        STDLE_ge_2g=ST >= 2 * g - TOLN,
        STDLE_ge_l2=ST >= math.sqrt(sq) - TOLN, STDLE_le_up=ST <= math.sqrt(n * sq) + TOLN,
        l2_ge_max=math.sqrt(sq) >= abs(tau - c).max() - TOLN,
        eq_cond=(abs(ST - 2 * g) < 1e-7) == (tau.min() >= c - 1e-9),
        LE_le_B=abs(EIG(P['L']) - c).sum() <= LE_bound(P) + TOLN,
        SLE_le_B=abs(rho - c).sum() <= LE_bound(P) + TOLN,
        trL2_eq=abs(np.trace(P['L'] @ P['L']) - (((d ** 2).sum()) + 2 * S2)) < TOLN)
    return ok, dict(c=c, cT=cT, trQT=float(tau.sum()), trQT2=float(trQT2), sq=float(sq), STDLE=float(ST),
                    lo_l2=math.sqrt(sq), lo_2g=2 * g, up=math.sqrt(n * sq), B=LE_bound(P))


# ========================================================================== 0. the example
hdr('0. Supply chain example (Option B data), D_T = {v2, v3}')
edges5 = {(0, 1): (.4, .3, .2), (0, 2): (.4, .3, .2), (1, 2): (.4, .3, .2),
          (1, 3): (.3, .3, .2), (2, 4): (.3, .3, .2), (3, 4): (.3, .3, .2)}
DT5 = (1, 2)


def mat(n, edges, c):
    A = np.zeros((n, n))
    for (u, v), w in edges.items():
        A[u, v] = A[v, u] = w[c]
    return A


names = ['mu', 'gamma', 'beta']
ex = {}
for c, nm in enumerate(names):
    P = parts(mat(5, edges5, c), DT5)
    ok, num = structure_checks(P)
    assert all(ok.values()), (nm, [k for k, v in ok.items() if not v])
    idx = indices(P)
    ex[nm] = dict(ok=True, num=num, idx=idx, rho=EIG(P['Q']).tolist(), tau=EIG(P['QT']).tolist(),
                  eta=EIG(P['L']).tolist(), W=P['W'])
    print(f"{nm:6s} W={P['W']:.2f} c=2W/n={num['c']:.4f} c_T={num['cT']:.4f} tr(Q_T)={num['trQT']:.4f}=2W+2g={2 * P['W'] + 4:.4f} "
          f"tr(Q_T^2)={num['trQT2']:.4f} sum(tau-c)^2={num['sq']:.4f}")
    print(f"       bounds: max(2g, sqrt(sum))={max(num['lo_2g'], num['lo_l2']):.4f} (2g={num['lo_2g']:.0f}, sqrt={num['lo_l2']:.4f}) "
          f"<= STDLE={num['STDLE']:.4f} <= {num['up']:.4f};  old lower bound max|tau-c|={abs(np.array(ex[nm]['tau']) - num['c']).max():.4f};"
          f"  B(LE,SLE)={num['B']:.4f} < 2g=4: {num['B'] < 4}")
    assert abs(num['STDLE'] - [5.4960, 5.3049, 4.8267][c]) < 6e-5
assert abs(ex['mu']['num']['trQT2'] - 22.08) < 1e-9
print('all identities of Part A and B hold on the example (assertions passed)')
R['example'] = ex

# the figures quoted in the task (20.72, 7.4565, 2.757) come from the data before Step 1
old = {(0, 1): (.4,), (0, 2): (.4,), (1, 2): (.3,), (1, 3): (.3,), (2, 4): (.3,), (3, 4): (.3,)}
Aold = np.zeros((5, 5))
for (u, v), w in old.items():
    Aold[u, v] = Aold[v, u] = w[0]
Po = parts(Aold, DT5)
_, no = structure_checks(Po)
print(f"old data (mu2(v2,v3)=0.3): tr(Q_T^2)={no['trQT2']:.4f}, upper bound={no['up']:.4f}, B={no['B']:.4f}"
      "  -> the 20.72, 7.4565 and 2.757 of the task text refer to this pre-Step-1 example")
assert abs(no['trQT2'] - 20.72) < 1e-9 and abs(no['up'] - 7.4565) < 5e-5 and abs(no['B'] - 2.757) < 5e-4
R['old_data'] = no

# ------------------------------------------------------------------------ centring and sensitivity
hdr('Centring: mean eigenvalue of Q is c, of Q_T is c_T; mean-centred variant (sensitivity only)')
print(f"{'':7s}{'c':>7s}{'c_T':>7s}{'LE':>9s}{'SLE':>9s}{'TDE':>9s}{'STDLE':>9s}{'STDLE^c':>9s}")
for nm in names:
    i, nu = ex[nm]['idx'], ex[nm]['num']
    print(f"{nm:7s}{nu['c']:7.2f}{nu['cT']:7.2f}{i['LE']:9.4f}{i['SLE']:9.4f}{i['TDE']:9.4f}{i['STDLE']:9.4f}{i['STDLE_c']:9.4f}")
    assert abs(np.mean(ex[nm]['rho']) - nu['c']) < 1e-9 and abs(np.mean(ex[nm]['tau']) - nu['cT']) < 1e-9
    assert abs(i['TDLE'] - i['LE']) < 1e-12
R['centring'] = {nm: dict(c=ex[nm]['num']['c'], cT=ex[nm]['num']['cT'], STDLE=ex[nm]['idx']['STDLE'],
                          STDLE_c=ex[nm]['idx']['STDLE_c']) for nm in names}
# is the centring offset responsible?  STDLE = sum|tau - c|, tau - c = (tau - c_T) + 2g/n
for nm in names:
    tau = np.array(ex[nm]['tau']); c, cT, g = ex[nm]['num']['c'], ex[nm]['num']['cT'], 2
    assert abs(abs(tau - c).sum() - abs(tau - cT + 2 * g / 5).sum()) < 1e-12

# ========================================================================== 1. random structure checks
hdr('1. Structure checks on random weighted graphs (every identity / inequality of Parts A and B)')
rg = random.Random(4242)
fails = {}
NS = 4000
for _ in range(NS):
    n = rg.randint(4, 12)
    p = rg.uniform(.25, 1)
    arcs = [e for e in itertools.combinations(range(n), 2) if rg.random() < p]
    nb = open_masks(n, arcs)
    if not has_total_dominating_set(nb):
        continue
    k, S = exhaustive_min_tds(nb)
    A = np.zeros((n, n))
    for u, v in arcs:
        A[u, v] = A[v, u] = rg.uniform(.01, 1)
    for D in S[:3]:
        ok, _ = structure_checks(parts(A, D))
        for key, v in ok.items():
            if not v:
                fails.setdefault(key, 0); fails[key] += 1
print('failures:', fails if fails else 'none')
assert not fails
R['structure_random_trials'] = NS

# ========================================================================== 2. counterexample search
hdr('2. Search: how often do STDLE > LE, STDLE > SLE, STDLE > TDE hold?')


def gen_weights(n, arcs, rg, mode):
    A = np.zeros((n, n))
    for u, v in arcs:
        A[u, v] = A[v, u] = (rg.uniform(.01, 1.0) if mode == 'uniform' else rg.choice([.25, .5, .75, 1.0]))
    if mode == 'uniformw':
        w0 = rg.uniform(.05, 1.0)
        A = w0 * (A > 0)
    return A


def graph_for_regime(n, rg, regime):
    """regime 'matrix': any graph, all edges count as dominating arcs.
       regime 'hfg'   : random weights on G*, strong arcs by Lemma 5.2, D_T from G_s (faithful to the method)."""
    while True:
        p = rg.uniform(.2, .9)
        arcs = [e for e in itertools.combinations(range(n), 2) if rg.random() < p]
        if not arcs:
            continue
        if regime == 'matrix':
            nb = open_masks(n, arcs)
            if has_total_dominating_set(nb):
                return arcs, arcs, nb
        else:
            return arcs, None, None


rows = []
cex = {'LE': None, 'SLE': None, 'TDE': None}
TRIALS_PER_N = 800
regimes = {'matrix-uniform': ('matrix', 'uniform'), 'matrix-discrete': ('matrix', 'discrete'), 'uniform-weight': ('matrix', 'uniformw'),
           'hfg-strong': ('hfg', 'discrete')}
cexU = {'LE': None, 'SLE': None, 'TDE': None}
for reg, (kind, wmode) in regimes.items():
    rg = random.Random(hash(reg) % 1000 + 7)
    rg.seed({'matrix-uniform': 11, 'matrix-discrete': 12, 'hfg-strong': 13, 'uniform-weight': 14}[reg])
    for n in range(5, 16):
        done = 0; tries = 0
        while done < TRIALS_PER_N and tries < 50 * TRIALS_PER_N:
            tries += 1
            arcs, _, nb = graph_for_regime(n, rg, kind)
            if kind == 'matrix':
                A = gen_weights(n, arcs, rg, wmode)
                k, S = exhaustive_min_tds(nb)
            else:
                # random HFG edge weights (three components); strong arcs by Lemma 5.2
                ed = {e: (rg.choice([.2, .4, .6]), rg.choice([.1, .2]), rg.choice([.1, .2])) for e in arcs}
                sa = strong_arcs_fast(n, ed)
                nb = open_masks(n, sa)
                if not has_total_dominating_set(nb):
                    continue
                k, S = exhaustive_min_tds(nb)
                A = np.zeros((n, n))
                for (u, v), w in ed.items():
                    A[u, v] = A[v, u] = w[0]                       # the mu-component
            done += 1
            arcs_s = arcs if kind == 'matrix' else sorted(sa)
            om = [sum(A[u, v] for u, v in arcs_s if u in D or v in D) for D in S]
            selS = [D for D, o in zip(S, om) if o >= max(om) - 1e-12]
            for D in selS[:5]:
                P = parts(A, D)
                ok, num = structure_checks(P)
                assert all(ok.values())
                i = indices(P)
                rows.append(dict(reg=reg, n=n, k=k, nsets=len(S), LE=i['LE'], SLE=i['SLE'], TDE=i['TDE'], ST=i['STDLE'],
                                 STc=i['STDLE_c'], B=num['B'], cond=2 * k > num['B'],
                                 gLE=i['STDLE'] > i['LE'], gSLE=i['STDLE'] > i['SLE'], gTDE=i['STDLE'] > i['TDE'],
                                 gLEc=i['STDLE_c'] > i['LE']))
                for key, flag in (('LE', 'gLE'), ('SLE', 'gSLE'), ('TDE', 'gTDE')):
                    if not rows[-1][flag] and (cex[key] is None or P['n'] < cex[key]['n']):
                        cex[key] = dict(n=n, D=[v + 1 for v in D], A=np.round(A, 4).tolist(), vals=i, B=num['B'], k=k)
                    if reg == 'uniform-weight' and not rows[-1][flag] and (cexU[key] is None or (n, len(arcs)) < (cexU[key]['n'], len(cexU[key]['arcs']))):
                        cexU[key] = dict(n=n, arcs=[(u + 1, v + 1) for u, v in arcs], w=float(A.max()), D=[v + 1 for v in D], vals=i, B=num['B'], k=k)
    sel = [r for r in rows if r['reg'] == reg]
    print(f"{reg:16s} {len(sel)} (instance, Rule-S set) pairs: "
          f"STDLE>LE {100 * np.mean([r['gLE'] for r in sel]):6.2f}%  STDLE>SLE {100 * np.mean([r['gSLE'] for r in sel]):6.2f}%  "
          f"STDLE>TDE {100 * np.mean([r['gTDE'] for r in sel]):6.2f}%  | sufficient condition holds in {100 * np.mean([r['cond'] for r in sel]):6.2f}%  "
          f"(then claim holds in {100 * np.mean([r['gLE'] and r['gSLE'] for r in sel if r['cond']]):.2f}%)")
    # THE THEOREM: whenever the condition holds, STDLE > max(LE, SLE)
    assert all(r['gLE'] and r['gSLE'] for r in sel if r['cond']), 'sufficient condition violated!'
R['search_rows'] = len(rows)

summary = {}
for reg in regimes:
    sel = [r for r in rows if r['reg'] == reg]
    summary[reg] = dict(pairs=len(sel), LE=float(np.mean([r['gLE'] for r in sel])), SLE=float(np.mean([r['gSLE'] for r in sel])),
                        TDE=float(np.mean([r['gTDE'] for r in sel])), cond=float(np.mean([r['cond'] for r in sel])),
                        LEc=float(np.mean([r['gLEc'] for r in sel])),
                        by_n={n: dict(LE=float(np.mean([r['gLE'] for r in sel if r['n'] == n])),
                                      SLE=float(np.mean([r['gSLE'] for r in sel if r['n'] == n])),
                                      TDE=float(np.mean([r['gTDE'] for r in sel if r['n'] == n])),
                                      cond=float(np.mean([r['cond'] for r in sel if r['n'] == n]))) for n in range(5, 16)})
R['search'] = summary
print('\ncounterexamples found (smallest n):')
for key, v in cex.items():
    print(f"  STDLE <= {key}:", 'none' if v is None else f"n={v['n']}, D={v['D']}, gamma_t={v['k']}, values { {a: round(b, 4) for a, b in v['vals'].items()} }, B={v['B']:.3f}")
print('\nsmallest counterexamples among uniform-weight HFGs (every edge strong, so valid for the method):')
for key, v in cexU.items():
    print(f"  STDLE <= {key}:", 'none' if v is None else f"n={v['n']}, w={v['w']:.4f}, arcs={v['arcs']}, D={v['D']}, gamma_t={v['k']}, "
          f"LE={v['vals']['LE']:.4f} SLE={v['vals']['SLE']:.4f} TDE={v['vals']['TDE']:.4f} STDLE={v['vals']['STDLE']:.4f}")
R['counterexamples'] = {k: v for k, v in cex.items()}
R['counterexamples_uniform_weight'] = cexU
if cex['LE']:
    print('  explicit weight matrix (STDLE <= LE):', cex['LE']['A'])

# ------------------------------------------------------------------------ explicit counterexamples (valid HFGs, w = 0.9)
hdr('2b. Smallest explicit counterexamples: uniform HFG (vertices (1,0,0), edges (0.9,0,0)); violation for EVERY minimum set')
from networkx.generators.atlas import graph_atlas_g
w0 = 0.9
found = {k: None for k in ('LE', 'SLE', 'TDE')}
for Gx in graph_atlas_g():
    n = Gx.number_of_nodes()
    if n < 4:
        continue
    arcs = [(u, v) for u, v in Gx.edges()]
    nb = open_masks(n, arcs)
    if not has_total_dominating_set(nb):
        continue
    k, S = exhaustive_min_tds(nb)
    A = np.zeros((n, n))
    for u, v in arcs:
        A[u, v] = A[v, u] = w0
    res = [indices(parts(A, D)) for D in S]
    for key in found:
        if all(r['STDLE'] < r[key] for r in res):
            cand = (n, len(arcs))
            if found[key] is None or cand < found[key][0]:
                found[key] = (cand, sorted(arcs), k, S, res)
R['explicit_cex'] = {}
for key, v in found.items():
    if v is None:
        print(f'no graph with <= 7 vertices where STDLE < {key} for every minimum set (w=0.9)')
        continue
    (n, m), arcs, k, S, res = v
    A = np.zeros((n, n))
    for u, vv in arcs:
        A[u, vv] = A[vv, u] = w0
    edgesU = {(u, vv): (w0, 0.0, 0.0) for u, vv in arcs}
    assert strong_arcs_definition(n, edgesU) == set(edgesU) == strong_arcs_fast(n, edgesU)
    r0 = res[0]
    P = parts(A, S[0]); num = structure_checks(P)[1]
    print(f"STDLE < {key} for all {len(S)} minimum set(s): n={n}, {m} arcs {[(u + 1, vv + 1) for u, vv in arcs]}, gamma_t={k}")
    for D, r in zip(S, res):
        print(f"    D={[x + 1 for x in D]}: LE={r['LE']:.4f} SLE={r['SLE']:.4f} TDE={r['TDE']:.4f} STDLE={r['STDLE']:.4f}")
    print(f"    2*gamma_t={2 * k}, B={num['B']:.4f}")
    R['explicit_cex'][key] = dict(n=n, arcs=[(u + 1, vv + 1) for u, vv in arcs], k=k, sets=[[x + 1 for x in D] for D in S],
                                 vals=[{a: float(b) for a, b in r.items()} for r in res], B=num['B'])


# connected counterexample for STDLE < LE (every minimum set) among all graphs with at most 7 vertices
import networkx as nx
bestc = None
for Gx in graph_atlas_g():
    n = Gx.number_of_nodes()
    if n < 4 or not nx.is_connected(Gx):
        continue
    arcs = sorted(Gx.edges())
    nb = open_masks(n, arcs)
    if not has_total_dominating_set(nb):
        continue
    k, S = exhaustive_min_tds(nb)
    A = np.zeros((n, n))
    for u, v in arcs:
        A[u, v] = A[v, u] = w0
    res = [indices(parts(A, D)) for D in S]
    if all(r['STDLE'] < r['LE'] for r in res):
        if bestc is None or (n, len(arcs)) < (bestc[0], bestc[1]):
            bestc = (n, len(arcs), arcs, k, S, res)
if bestc:
    n, m, arcs, k, S, res = bestc
    print(f"connected: STDLE < LE for all {len(S)} minimum set(s): n={n}, {m} arcs {[(u + 1, v + 1) for u, v in arcs]}, gamma_t={k}")
    for D, r in zip(S, res):
        print(f"    D={[x + 1 for x in D]}: LE={r['LE']:.4f} SLE={r['SLE']:.4f} TDE={r['TDE']:.4f} STDLE={r['STDLE']:.4f}")
    A = np.zeros((n, n))
    for u, v in arcs:
        A[u, v] = A[v, u] = w0
    edgesU = {(u, v): (w0, 0.0, 0.0) for u, v in arcs}
    assert strong_arcs_definition(n, edgesU) == set(edgesU) == strong_arcs_fast(n, edgesU)
    R['explicit_cex_connected_LE'] = dict(n=n, arcs=[(u + 1, v + 1) for u, v in arcs], k=k, sets=[[x + 1 for x in D] for D in S],
                                         vals=[{a: float(b) for a, b in r.items()} for r in res], B=structure_checks(parts(A, S[0]))[1]['B'])
else:
    print('no connected graph with <= 7 vertices where STDLE < LE for every minimum set (w=0.9)')

# violations for SOME (non-selected) minimum set only: STDLE < SLE (n=8) and STDLE < TDE (n=7)
print('\nviolations for some minimum set, not for the set selected by Rule (S):')
CX = {'SLE': ([(1, 3), (1, 4), (1, 5), (1, 8), (2, 6), (3, 4), (3, 7), (3, 8), (4, 5), (4, 7), (4, 8), (5, 7), (5, 8), (7, 8)], [1, 2, 3, 6], 8),
      'TDE': ([(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 3), (2, 6), (3, 4), (3, 5), (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7)], [2, 6], 7)}
R['some_set_cex'] = {}
for key, (arcs, D, n) in CX.items():
    A = np.zeros((n, n))
    for u, v in arcs:
        A[u - 1, v - 1] = A[v - 1, u - 1] = w0
    arcs0 = [(u - 1, v - 1) for u, v in arcs]
    k, S = exhaustive_min_tds(open_masks(n, arcs0))
    om = [sum(A[u, v] for u, v in arcs0 if u in Dd or v in Dd) for Dd in S]
    selS = [Dd for Dd, o in zip(S, om) if o >= max(om) - 1e-12]
    Dz = tuple(d - 1 for d in D)
    i = indices(parts(A, Dz))
    assert Dz in S and i['STDLE'] < i[key]
    viol = [d for d in S if indices(parts(A, d))['STDLE'] < indices(parts(A, d))[key]]
    assert all(indices(parts(A, d))['STDLE'] > indices(parts(A, d))[key] for d in selS)
    print(f"  STDLE < {key}: n={n}, gamma_t={k}, {len(S)} minimum sets, {len(viol)} violate; D={D}: {key}={i[key]:.4f} > STDLE={i['STDLE']:.4f};"
          f" Rule (S) selects {[[x + 1 for x in d] for d in selS]} for which STDLE > {key}")
    R['some_set_cex'][key] = dict(n=n, arcs=arcs, k=k, nsets=len(S), nviol=len(viol), D=D, vals={a: float(b) for a, b in i.items()},
                                  selected=[[x + 1 for x in d] for d in selS],
                                  vals_sel=[{a: float(b) for a, b in indices(parts(A, d)).items()} for d in selS])

# ========================================================================== 3. figures
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
                     'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
BL, OR, GR, RD, PU = '#0072B2', '#E69F00', '#009E73', '#D55E00', '#7F3C8D'

# --- Figure 1: spectra of L, Q, Q_T (mu-component)
eta, rho, tau = ex['mu']['eta'], ex['mu']['rho'], ex['mu']['tau']
c, cT = ex['mu']['num']['c'], ex['mu']['num']['cT']
fig, ax = plt.subplots(figsize=(7.4, 3.9))
ys = {'$L$': 2, '$Q$': 1, '$Q_T$': 0}
for lab, vals, col in (('$L$', eta, BL), ('$Q$', rho, GR), ('$Q_T$', tau, RD)):
    y = ys[lab]
    ax.hlines(y, -0.2, 4.0, color='#cccccc', lw=0.8, zorder=0)
    ax.vlines(vals, y - 0.0, y + 0.0, color=col)
    ax.scatter(vals, [y] * 5, s=46, color=col, zorder=3)
for r_, t_ in zip(rho, tau):
    ax.annotate('', xy=(t_, 0.08), xytext=(r_, 0.92), arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0, shrinkA=3, shrinkB=3))
ax.axvline(c, color='k', ls='--', lw=1.1, ymin=0.0, ymax=1.0)
ax.axvline(cT, color=RD, ls=':', lw=1.4)
ax.text(c - 0.04, 2.45, '$c=2W/n=0.84$', ha='right', va='center', fontsize=10)
ax.text(cT + 0.04, 2.45, '$c_T=(2W+2\\gamma_t)/n=1.64$', ha='left', va='center', fontsize=10, color=RD)
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(['$Q_T$', '$Q$', '$L$'])
ax.set_ylim(-0.45, 2.75); ax.set_xlim(-0.2, 4.0)
ax.set_xlabel('eigenvalue ($\\mu$-component, $D_T=\\{v_2,v_3\\}$)')
ax.text(3.95, 0.52, 'shifts $\\tau_i-\\rho_i\\geq0$, total $2\\gamma_t=4$', ha='right', va='center', fontsize=9.5, color='#444444')
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/fig_spectra.{ext}', dpi=300)
plt.close(fig)

# --- Figure 2: six indices x three components
labels = ['$E$', '$LE$', '$SLE$', '$TDE$', '$TDLE$', '$STDLE$']
keys = ['E', 'LE', 'SLE', 'TDE', 'TDLE', 'STDLE']
fig, ax = plt.subplots(figsize=(8.0, 4.0))
w = 0.26
for j, (nm, col, lab) in enumerate(zip(names, (BL, OR, GR), ('$\\mu$', '$\\gamma$', '$\\beta$'))):
    vals = [ex[nm]['idx'][k] for k in keys]
    bars = ax.bar(np.arange(6) + (j - 1) * w, vals, w, color=col, label=lab, edgecolor='white', lw=.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f'{v:.2f}', ha='center', va='bottom', fontsize=7.5, rotation=90)
ax.set_xticks(range(6)); ax.set_xticklabels(labels)
ax.set_ylabel('index value'); ax.set_ylim(0, 6.6)
ax.annotate('$TDLE=LE$', xy=(4, 2.9), xytext=(4.35, 4.5), fontsize=10, ha='left',
            arrowprops=dict(arrowstyle='->', color='#444444'))
ax.legend(frameon=False, ncol=3, loc='upper left', title='component', title_fontsize=9)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/fig_indices.{ext}', dpi=300)
plt.close(fig)

# --- Figure 3: proportion of instances in which each inequality holds, by n
fig, axs2 = plt.subplots(2, 2, figsize=(8.8, 6.0), sharey=True, sharex=True)
axs = axs2.ravel()
for ax, reg, ttl in zip(axs, regimes, ('(a) random weights in $(0,1]$, all edges count', '(b) weights in $\\{.25,.5,.75,1\\}$, all edges count',
                                       '(c) one common weight $w$ (valid HFG, all arcs strong)', '(d) random HFG, strong arcs only')):
    nn = list(range(5, 16))
    for key, lab, col, mk in (('LE', 'STDLE > LE', BL, 'o'), ('SLE', 'STDLE > SLE', GR, 's'), ('TDE', 'STDLE > TDE', OR, '^'),
                              ('cond', 'sufficient condition', PU, 'x')):
        ax.plot(nn, [100 * summary[reg]['by_n'][n][key] for n in nn], marker=mk, color=col, lw=1.5, ms=4.5, label=lab,
                ls='--' if key == 'cond' else '-')
    ax.set_title(ttl, fontsize=9.5); ax.set_xticks(nn[::2])
for ax in axs2[1]: ax.set_xlabel('number of vertices $n$')
for ax in axs2[:, 0]: ax.set_ylabel('share of instances (%)')
axs[0].set_ylim(-3, 103)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.07, 1, 1))
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/fig_search.{ext}', dpi=300)
plt.close(fig)
print('\nfigures written')

json.dump(R, open('results_step4.json', 'w'), indent=1, default=float)
print('ALL ASSERTIONS PASSED')
