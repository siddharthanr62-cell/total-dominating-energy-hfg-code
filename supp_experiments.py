"""Step 3: minimum total dominating set -- verification, counterexamples, experiments, figures.
Run:  python3 step3_experiments.py        (needs tds_lib.py next to it; writes figs/ and results_step3.json)"""
import itertools, json, math, os, random, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tds_lib import *

OUT = 'figs'
os.makedirs(OUT, exist_ok=True)
R = {}                                    # everything that goes into results_step3.json
H = lambda m: sum(1.0 / i for i in range(1, m + 1))      # harmonic number


def hdr(s):
    print('\n' + '=' * 78 + '\n' + s + '\n' + '=' * 78)


# ====================================================================================
# 0. five-vertex supply-chain example (Options B data of Step 1)
# ====================================================================================
hdr('0. Five-vertex example (corrected strong arcs of Step 1, Rule S of Step 2)')
n5 = 5
vert5 = {0: (.5, .3, .2), 1: (.5, .3, .2), 2: (.5, .3, .2), 3: (.4, .4, .2), 4: (.4, .4, .2)}
edges5 = {(0, 1): (.4, .3, .2), (0, 2): (.4, .3, .2), (1, 2): (.4, .3, .2),
          (1, 3): (.3, .3, .2), (2, 4): (.3, .3, .2), (3, 4): (.3, .3, .2)}


def check_hfg(vert, edges):
    ok = all(abs(sum(t) - 1) < TOL for t in vert.values())
    for (u, v), (m, g, b) in edges.items():
        a, c = vert[u], vert[v]
        ok &= (m <= min(a[0], c[0]) + TOL and g <= max(a[1], c[1]) + TOL
               and b <= min(a[2], c[2]) + TOL and m + g + b <= 1 + TOL)
    return bool(ok)


def energies(A, DT):
    n = A.shape[0]
    W = A.sum() / 2
    avg = 2 * W / n
    T = A + np.diag([1.0 if i in DT else 0.0 for i in range(n)])
    ev = np.linalg.eigvalsh
    return {'E': abs(ev(A)).sum(), 'TDE': abs(ev(T)).sum(),
            'STDLE': abs(ev(np.diag(T.sum(1)) + T) - avg).sum()}


def matrix(n, edges, c):
    A = np.zeros((n, n))
    for (u, v), w in edges.items():
        A[u, v] = A[v, u] = w[c]
    return A


assert check_hfg(vert5, edges5)
arcs5_def = strong_arcs_definition(n5, edges5)
arcs5 = strong_arcs_fast(n5, edges5)
assert arcs5 == arcs5_def == set(edges5)
nb5 = open_masks(n5, arcs5)
g5, S5 = exhaustive_min_tds(nb5)
gi5, Si5 = ip_all_min_tds(nb5)
print('strong arcs:', len(arcs5), 'of', len(edges5), '(definition == O(n^2) test: True)')
print('exhaustive: gamma_t =', g5, ' minimum sets', [[v + 1 for v in D] for D in S5])
print('IP        : gamma_t =', gi5, ' minimum sets', [[v + 1 for v in D] for D in Si5])
assert g5 == gi5 == 2 and S5 == Si5 == [(1, 2), (1, 3), (2, 4)]
omega = {D: sum(edges5[e][0] for e in arcs5 if e[0] in D or e[1] in D) for D in S5}
best = max(omega.values())
chosen = [D for D in S5 if abs(omega[D] - best) < 1e-12]
print('Rule (S): omega =', {tuple(v + 1 for v in D): round(w, 2) for D, w in omega.items()},
      '-> selected', [[v + 1 for v in D] for D in chosen])
assert chosen == [(1, 2)]
res_prev = json.load(open('results_step2.json'))
print(f"{'D_T':12s}{'TDE_mu':>9s}{'TDE_ga':>9s}{'TDE_be':>9s}{'STDLE_mu':>10s}{'STDLE_ga':>10s}{'STDLE_be':>10s}")
tab5 = []
for D in S5:
    en = [energies(matrix(n5, edges5, c), D) for c in range(3)]
    row = [e['TDE'] for e in en] + [e['STDLE'] for e in en]
    print(f"{str([v + 1 for v in D]):12s}" + ''.join(f"{x:9.4f}" if i < 3 else f"{x:10.4f}" for i, x in enumerate(row)))
    ref = [r for r in res_prev['partC'] if r['D'] == [v + 1 for v in D]][0]
    for c, nm in enumerate(['mu', 'ga', 'be']):
        assert abs(en[c]['TDE'] - ref['TDE_' + nm]) < 1e-9 and abs(en[c]['STDLE'] - ref['STDLE_' + nm]) < 1e-9
    tab5.append(dict(D=[v + 1 for v in D], vals=row))
print('values agree with Step 2 (results_step2.json): True')
R['example'] = dict(gamma_t=g5, sets=[[v + 1 for v in D] for D in S5], table=tab5)
# original greedy on the example
print('original (closed-nbhd) greedy on the example:', [v + 1 for v in greedy_closed_naive(nb5)],
      ' open-nbhd greedy:', [v + 1 for v in greedy_open(nb5)])
assert is_total_dominating(nb5, greedy_closed_naive(nb5)) and len(greedy_open(nb5)) == 2

# ====================================================================================
# 1. Part A: counterexamples and existence
# ====================================================================================
hdr('1. Part A -- audit of the greedy step')
# A2: dominating but not total (P3)
nbP3 = open_masks(3, [(0, 1), (1, 2)])
Dn = greedy_closed_naive(nbP3)
print('A2  path v1-v2-v3, original greedy (closed neighbourhoods) returns', [v + 1 for v in Dn],
      '| dominating:', is_dominating(nbP3, Dn), '| total dominating:', is_total_dominating(nbP3, Dn))
print('    exact: gamma_t =', exhaustive_min_tds(nbP3)[0], 'sets', [[v + 1 for v in D] for D in exhaustive_min_tds(nbP3)[1]],
      '| repaired greedy', [v + 1 for v in repair(nbP3, Dn)])
assert is_dominating(nbP3, Dn) and not is_total_dominating(nbP3, Dn)
R['A2'] = dict(graph='P3', naive=[v + 1 for v in Dn])

# A1: greedy suboptimal for every tie-breaking: n = 8
arcsC = [(0, 1), (0, 5), (1, 7), (2, 4), (2, 6), (3, 5), (3, 6), (3, 7)]
nbC = open_masks(8, arcsC)
kC, SC = exhaustive_min_tds(nbC)
print('A1  graph on v1..v8 with arcs', [(u + 1, v + 1) for u, v in arcsC])
print('    exact gamma_t =', kC, ' minimum sets', [[v + 1 for v in D] for D in SC])
print('    greedy (open neighbourhoods), all tie-breakings give sizes', sorted(all_greedy_outcomes(nbC)))
print('    greedy (closed + repair) sizes ->', len(greedy_closed_repaired(nbC)), '; plain closed greedy total?',
      is_total_dominating(nbC, greedy_closed_naive(nbC)))
assert kC == 4 and min(all_greedy_outcomes(nbC)) == 5
# trace of the deterministic run
unc = (1 << 8) - 1
trace = []
while unc:
    cs = [bin(nbC[u] & unc).count('1') for u in range(8)]
    mx = max(cs)
    u = cs.index(mx)
    trace.append((u + 1, mx, [i + 1 for i in range(8) if cs[i] == mx]))
    unc &= ~nbC[u]
print('    trace (chosen vertex, newly covered, vertices attaining the maximum):', trace)
Dg = greedy_open(nbC)
print('    greedy output', [v + 1 for v in Dg], 'size', len(Dg), '; optimum size', kC)
# smallest order: exhaustive over all graphs up to isomorphism with <= 7 vertices
import networkx as nx
from networkx.generators.atlas import graph_atlas_g
cnt = 0
for Gx in graph_atlas_g():
    nn = Gx.number_of_nodes()
    if nn < 4:
        continue
    nbx = open_masks(nn, list(Gx.edges()))
    if not has_total_dominating_set(nbx):
        continue
    k, _ = exhaustive_min_tds(nbx)
    if min(all_greedy_outcomes(nbx)) > k:
        cnt += 1
print('    graphs with <= 7 vertices (all isomorphism classes) where every tie-breaking is suboptimal:', cnt)
assert cnt == 0
R['A1'] = dict(arcs=[(u + 1, v + 1) for u, v in arcsC], gamma_t=kC, greedy=len(Dg), trace=trace,
               n_le7_counterexamples=cnt)
# realisation as an HFG: uniform weights make every edge a strong arc
vertU = {v: (.5, .3, .2) for v in range(8)}
edgesU = {e: (.4, .3, .2) for e in arcsC}
assert check_hfg(vertU, edgesU) and strong_arcs_definition(8, edgesU) == set(edgesU) == strong_arcs_fast(8, edgesU)
print('    realised as an HFG (all vertices (.5,.3,.2), all edges (.4,.3,.2)): HFG conditions hold and all arcs strong')

# A3: existence
vertX = {0: (.5, .2, .3), 1: (.5, .2, .3), 2: (.5, .2, .3)}
edgesX = {(0, 1): (.1, .2, .3), (0, 2): (.1, .1, .1), (1, 2): (.4, .1, .2)}
assert check_hfg(vertX, edgesX)
sX = strong_arcs_fast(3, edgesX)
assert sX == strong_arcs_definition(3, edgesX) == {(1, 2)}
print('A3  HFG on a triangle: strong arcs =', [(u + 1, v + 1) for u, v in sX],
      '-> v1 isolated in the strong-arc graph; total dominating set exists:',
      has_total_dominating_set(open_masks(3, sX)))
R['A3'] = dict(strong=[(u + 1, v + 1) for u, v in sX])

# ====================================================================================
# 2. Part B/C verification: equivalence and correctness of the exact methods
# ====================================================================================
hdr('2. Verification of the building blocks')
rng = random.Random(2024)
# 2a. O(n^2) strong-arc test == literal definition
tr, bad = 0, 0
for _ in range(1500):
    n = rng.randint(3, 8)
    levels = rng.choice([2, 3, 5, 100])
    edges = {}
    for u, v in itertools.combinations(range(n), 2):
        if rng.random() < rng.uniform(.3, 1):
            edges[(u, v)] = tuple(rng.randint(1, levels) / levels * .5 for _ in range(3))
    if edges:
        tr += 1
        bad += strong_arcs_definition(n, edges) != strong_arcs_fast(n, edges)
print(f'2a  strong-arc test (spanning-tree bottleneck) vs literal definition: {tr} random weighted graphs, {bad} mismatches')
assert bad == 0
R['equiv'] = dict(trials=tr, mismatches=bad)
# quadratic growth of the strong-arc step
tt = []
for n in (40, 80, 160, 320):
    edges = {(u, v): (rng.random() * .5, rng.random() * .5, rng.random() * .5) for u, v in itertools.combinations(range(n), 2)}
    t0 = time.perf_counter(); strong_arcs_fast(n, edges); tt.append((n, time.perf_counter() - t0))
print('    strong-arc graph on complete graphs, seconds:', [(n, round(t, 3)) for n, t in tt],
      ' successive ratios:', [round(tt[i + 1][1] / tt[i][1], 2) for i in range(3)], '(quadratic: 4)')
R['strong_timing'] = tt

# 2b. exhaustive == IP == independent numpy brute force
def brute(n, arcs):
    A = np.zeros((n, n), int)
    for u, v in arcs: A[u, v] = A[v, u] = 1
    for k in range(2, n + 1):
        out = []
        for D in itertools.combinations(range(n), k):
            x = np.zeros(n, int); x[list(D)] = 1
            if (A @ x >= 1).all(): out.append(D)
        if out: return k, out
tc, ok = 0, True
for _ in range(300):
    n = rng.randint(4, 11)
    p = rng.uniform(.2, .6)
    arcs = [e for e in itertools.combinations(range(n), 2) if rng.random() < p]
    nb = open_masks(n, arcs)
    if not has_total_dominating_set(nb):
        continue
    k1, S1 = exhaustive_min_tds(nb); k2, S2 = ip_all_min_tds(nb); k3, S3 = brute(n, arcs)
    tc += 1
    ok &= (k1 == k2 == k3 and S1 == S2 == S3)
print(f'2b  exhaustive == integer program (all minimum sets, no-good cuts) == numpy brute force on {tc} random graphs: {ok}')
assert ok
R['exact_crosscheck'] = dict(instances=tc, agree=bool(ok))

# ====================================================================================
# 3. Part D: experiments
# ====================================================================================
hdr('3. Random experiments: exact vs greedy')


def er_graph(n, d, rg):
    p = d / (n - 1)
    while True:
        arcs = [e for e in itertools.combinations(range(n), 2) if rg.random() < p]
        nb = open_masks(n, arcs)
        if has_total_dominating_set(nb):
            return arcs, nb


def timed(f, *a):
    t0 = time.perf_counter(); out = f(*a); return out, time.perf_counter() - t0


NS = list(range(8, 21))
DEG = (3, 4, 6)                    # mean degree of the strong-arc graph
TRIALS = 50
rg = random.Random(99)
rows = []
for n in NS:
    for d in DEG:
        for t in range(TRIALS):
            arcs, nb = er_graph(n, d, rg)
            (k, Sx), t_ex = timed(exhaustive_min_tds, nb)
            (ki, _), t_ip = timed(ip_min_tds, nb)
            assert ki == k
            Do, t_go = timed(greedy_open, nb)
            Dc = greedy_closed_naive(nb)
            Dr = repair(nb, Dc)
            t_gr = timed(greedy_closed_repaired, nb)[1]
            Delta = max(bin(m).count('1') for m in nb)
            assert is_total_dominating(nb, Do) and is_total_dominating(nb, Dr)
            assert len(Do) <= H(Delta) * k + 1e-9, (len(Do), k, Delta)          # Chvatal bound
            assert len(Dr) <= 2 * H(Delta + 1) * k + 1e-9
            rows.append(dict(n=n, d=d, k=k, nsets=len(Sx), Delta=Delta, open=len(Do), rep=len(Dr), naive=len(Dc),
                             naive_total=bool(is_total_dominating(nb, Dc)),
                             t_ex=t_ex, t_ip=t_ip, t_go=t_go, t_gr=t_gr))
print(f'{len(rows)} instances; both greedy variants always total dominating; Chvatal bound H(Delta) respected in all')
R['exp_rows'] = len(rows)


def summarize(sel):
    k = np.array([r['k'] for r in sel])
    out = {}
    for key in ('open', 'rep'):
        s = np.array([r[key] for r in sel])
        out[key] = dict(opt=float(np.mean(s == k)), mean_gap=float(np.mean(s - k)),
                        mean_rel=float(np.mean((s - k) / k)), max_gap=int(np.max(s - k)),
                        max_ratio=float(np.max(s / k)))
    out['naive_not_total'] = float(np.mean([not r['naive_total'] for r in sel]))
    out['naive_total_size_le_opt'] = int(sum(r['naive'] < r['k'] for r in sel))
    out['mean_nsets'] = float(np.mean([r['nsets'] for r in sel]))
    out['mean_k'] = float(k.mean())
    return out


allsum = summarize(rows)
print('ALL  :', json.dumps(allsum, indent=1))
bydeg = {d: summarize([r for r in rows if r['d'] == d]) for d in DEG}
for d in DEG:
    s = bydeg[d]
    print(f"mean degree {d}: open-greedy optimal {100 * s['open']['opt']:.1f}% (mean gap {s['open']['mean_gap']:.3f}, max {s['open']['max_gap']}), "
          f"closed+repair optimal {100 * s['rep']['opt']:.1f}% (mean gap {s['rep']['mean_gap']:.3f}, max {s['rep']['max_gap']}), "
          f"plain closed greedy NOT total in {100 * s['naive_not_total']:.1f}%")
R['summary_all'] = allsum
R['summary_bydeg'] = {str(d): v for d, v in bydeg.items()}
print(f"plain closed greedy returns fewer vertices than gamma_t in {allsum['naive_total_size_le_opt']} of {len(rows)} instances (it only dominates)")

byn = {}
for n in NS:
    sel = [r for r in rows if r['n'] == n]
    byn[n] = dict(
        t_ex=np.percentile([r['t_ex'] for r in sel], [10, 50, 90]).tolist(),
        t_ip=np.percentile([r['t_ip'] for r in sel], [10, 50, 90]).tolist(),
        t_go=np.percentile([r['t_go'] for r in sel], [10, 50, 90]).tolist(),
        gap_open=[float(np.mean([r['open'] - r['k'] == g for r in sel])) for g in (0, 1, 2)] +
                 [float(np.mean([r['open'] - r['k'] >= 3 for r in sel]))],
        gap_rep=[float(np.mean([r['rep'] - r['k'] == g for r in sel])) for g in (0, 1, 2)] +
                [float(np.mean([r['rep'] - r['k'] >= 3 for r in sel]))],
        k=float(np.mean([r['k'] for r in sel])))
R['by_n'] = {str(n): v for n, v in byn.items()}
print('\nmedian runtimes (ms)   n: exhaustive / IP(HiGHS) / greedy')
for n in NS:
    print(f"  n={n:2d}: {1e3 * byn[n]['t_ex'][1]:9.2f} / {1e3 * byn[n]['t_ip'][1]:7.2f} / {1e3 * byn[n]['t_go'][1]:6.3f}   mean gamma_t={byn[n]['k']:.2f}")

# --- HFG-derived strong-arc graphs (full pipeline, discrete levels so that ties occur)
print('\nHFG-derived instances (random HFG -> strong-arc graph -> exact vs greedy):')
rgh = random.Random(5)
hf = []
for cfg, (p, mus) in {'fine (p=0.3, 3 mu-levels)': (0.3, [.2, .4, .6]), 'coarse (p=0.5, 2 mu-levels)': (0.5, [.4, .6])}.items():
    for n in (8, 12, 16, 20):
        ne, fracs, gaps = 0, [], []
        for t in range(100):
            edges = {}
            for u, v in itertools.combinations(range(n), 2):
                if rgh.random() < p:
                    edges[(u, v)] = (rgh.choice(mus), rgh.choice([.1, .2]), rgh.choice([.1, .2]))
            vert = {v: (.6, .2, .2) for v in range(n)}
            assert check_hfg(vert, edges)
            sa = strong_arcs_fast(n, edges)
            fracs.append(len(sa) / max(1, len(edges)))
            nb = open_masks(n, sa)
            if not has_total_dominating_set(nb):
                ne += 1
                continue
            k, _ = exhaustive_min_tds(nb)
            gaps.append(len(greedy_open(nb)) - k)
        opt = float(np.mean(np.array(gaps) == 0)) if gaps else float('nan')
        hf.append(dict(cfg=cfg, n=n, no_tds=ne, frac_strong=float(np.mean(fracs)), optimal=opt, max_gap=int(max(gaps)) if gaps else None, solved=len(gaps)))
        print(f"  {cfg:28s} n={n:2d}: strong arcs {100 * np.mean(fracs):5.1f}% of edges; no TDS (isolated vertex in strong-arc graph) in {ne}/100; "
              f"open-greedy optimal in {100 * opt:.1f}% of the {len(gaps)} solvable (max gap {max(gaps) if gaps else '-'})")
R['hfg_derived'] = hf

# ====================================================================================
# 4. Figures
# ====================================================================================
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
                     'font.size': 11, 'axes.labelsize': 12, 'legend.fontsize': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
C = {'ex': '#0072B2', 'ip': '#009E73', 'gr': '#D55E00'}

fig, ax = plt.subplots(figsize=(6.2, 4.0))
for key, lab, col, mk in (('t_ex', 'Exhaustive (bitmask)', C['ex'], 'o'), ('t_ip', 'Integer program (HiGHS)', C['ip'], 's'),
                          ('t_go', 'Greedy heuristic', C['gr'], '^')):
    med = np.array([1e3 * byn[n][key][1] for n in NS]); lo = np.array([1e3 * byn[n][key][0] for n in NS])
    hi = np.array([1e3 * byn[n][key][2] for n in NS])
    ax.plot(NS, med, marker=mk, color=col, lw=1.8, ms=5, label=lab)
    ax.fill_between(NS, lo, hi, color=col, alpha=.15, lw=0)
ax.set_yscale('log'); ax.set_xlabel('number of vertices $n$'); ax.set_ylabel('runtime (ms)')
ax.set_xticks(NS[::2]); ax.grid(True, which='major', alpha=.3)
ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9, handlelength=1.6, columnspacing=1.2)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/fig_runtime.{ext}', dpi=300)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
cols = ['#0072B2', '#E69F00', '#D55E00', '#7F3C8D']
labs = ['optimal', '+1', '+2', '$\\geq$ +3']
for ax, key, ttl in ((axs[0], 'gap_open', '(a) greedy on open neighbourhoods'), (axs[1], 'gap_rep', '(b) closed-neighbourhood greedy + repair')):
    bottom = np.zeros(len(NS))
    for j in range(4):
        v = np.array([100 * byn[n][key][j] for n in NS])
        ax.bar(NS, v, bottom=bottom, color=cols[j], width=.8, label=labs[j], edgecolor='white', lw=.4)
        bottom += v
    ax.set_title(ttl, fontsize=11); ax.set_xlabel('number of vertices $n$'); ax.set_xticks(NS[::2])
axs[0].set_ylabel('share of instances (%)')
h_, l_ = axs[1].get_legend_handles_labels()
fig.legend(h_, l_, title='$|D|-\\gamma_t$', frameon=False, loc='lower center', ncol=4, fontsize=9, title_fontsize=9)
fig.tight_layout(rect=(0, 0.1, 1, 1))
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/fig_gap.{ext}', dpi=300)
plt.close(fig)
print('\nfigures written to', OUT)

json.dump(R, open('results_step3.json', 'w'), indent=1, default=float)
print('ALL ASSERTIONS PASSED')
