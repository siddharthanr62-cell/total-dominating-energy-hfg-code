"""Step 2 verification: every identity/inequality claimed for TDHFAM / TDE is tested
(i) on the Option-B supply-chain HFG and (ii) on thousands of random weighted graphs."""
import itertools, json, math
import numpy as np

rng = np.random.default_rng(12345)
EIG = np.linalg.eigvalsh
TOL = 1e-9
report = {}

def Tmat(A, D):
    n = A.shape[0]
    return A + np.diag([1.0 if i in D else 0.0 for i in range(n)])

def N_neg(M):
    ev = EIG(M)
    return float(-ev[ev < 0].sum())

def props(A, D):
    """Return dict of booleans for properties (a)-(f) for matrix A and vertex set D."""
    n = A.shape[0]
    T = Tmat(A, D)
    g = len(D)
    la, lt = EIG(A), EIG(T)
    E, TDE = abs(la).sum(), abs(lt).sum()
    w2 = (A ** 2).sum() / 2.0                 # sum over edges of w(e)^2
    theta = np.trace(T @ T)
    d = lt - la
    R = N_neg(A) - N_neg(T)
    out = dict(
        a_trace=abs(np.trace(T) - g) < TOL and abs(lt.sum() - g) < TOL,
        b_weyl=(d.min() > -TOL) and (d.max() < 1 + TOL),
        b_sum_shift=abs(d.sum() - g) < TOL,
        c_energy=abs(TDE - E) <= g + TOL,
        d_lower_gamma=TDE >= g - TOL,
        e_trace_sq=abs(theta - (2 * w2 + g)) < TOL,
        f_identity=abs(TDE - (g + 2 * N_neg(T))) < TOL,
        E_identity=abs(E - 2 * N_neg(A)) < TOL,
        g_R=abs((TDE - E) - (g - 2 * R)) < TOL and -TOL <= R <= g + TOL,
        bounds_lo=TDE >= max(g, math.sqrt(theta)) - TOL,
        bounds_hi=TDE <= math.sqrt(n * theta) + TOL,
    )
    # strict right inequality whenever some edge touches D
    touches = any(A[i, j] > 0 for i in D for j in range(n) if j != i)
    out['strict_hi'] = (TDE < math.sqrt(n * theta) - 1e-12) if touches else True
    return out, dict(E=E, TDE=TDE, theta=theta, R=R, g=g)

# ---------------------------------------------------------------- 1. the example (Option B)
edges = {(0, 1): (.4, .3, .2), (0, 2): (.4, .3, .2), (1, 2): (.4, .3, .2),
         (1, 3): (.3, .3, .2), (2, 4): (.3, .3, .2), (3, 4): (.3, .3, .2)}
def mat(c):
    A = np.zeros((5, 5))
    for (u, v), w in edges.items():
        A[u, v] = A[v, u] = w[c]
    return A
ex = {}
for c, name in enumerate(['mu', 'ga', 'be']):
    p, v = props(mat(c), (1, 2))
    assert all(p.values()), (name, p)
    la, lt = EIG(mat(c)), EIG(Tmat(mat(c), (1, 2)))
    ex[name] = dict(v, shifts=(lt - la).tolist(), sumw2=float((mat(c) ** 2).sum() / 2),
                    lo=max(2, math.sqrt(v['theta'])), hi=math.sqrt(5 * v['theta']))
report['example'] = ex

# ---------------------------------------------------------------- 2. random graphs
fails = {}
N_TRIALS = 30000
for t in range(N_TRIALS):
    n = int(rng.integers(2, 11))
    dens = rng.uniform(0.2, 1.0)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < dens:
                A[i, j] = A[j, i] = rng.uniform(0.01, 1.0)
    k = int(rng.integers(1, n + 1))
    D = tuple(sorted(rng.choice(n, size=k, replace=False).tolist()))
    p, _ = props(A, D)
    for key, ok in p.items():
        if not ok:
            fails.setdefault(key, []).append((n, D))
report['random_trials'] = N_TRIALS
report['random_failures'] = {k: len(v) for k, v in fails.items()}
assert not fails, {k: v[:2] for k, v in fails.items()}

# left bound equality only for K2 with unit weight; check K2 numerically
A = np.array([[0, 1.0], [1.0, 0]])
p, v = props(A, (0, 1))
report['K2_w1'] = dict(TDE=v['TDE'], sqrt_theta=math.sqrt(v['theta']), gamma_t=v['g'])
assert abs(v['TDE'] - 2) < 1e-12 and abs(math.sqrt(v['theta']) - 2) < 1e-12

# ---------------------------------------------------------------- 3. closed forms
cf = {}
# K2 with D=V : TDE = 2 (w<=1)
for w in np.linspace(0.05, 1, 20):
    A = np.array([[0, w], [w, 0]])
    assert abs(abs(EIG(Tmat(A, (0, 1)))).sum() - 2) < 1e-12
cf['K2'] = 'TDE = 2 for all 0<w<=1'
# P4 uniform w, D_T={v2,v3}
maxerr = 0
for w in np.linspace(0.05, 1, 20):
    A = np.zeros((4, 4))
    for i in range(3):
        A[i, i + 1] = A[i + 1, i] = w
    num = abs(EIG(Tmat(A, (1, 2)))).sum()
    form = math.sqrt(1 + 2 * w + 5 * w * w) + math.sqrt(1 - 2 * w + 5 * w * w)
    maxerr = max(maxerr, abs(num - form))
cf['P4_max_err'] = maxerr
assert maxerr < 1e-12
# K_n uniform w, D_T = any pair
maxerr = 0
for n in range(3, 10):
    for w in np.linspace(0.05, 1, 20):
        A = w * (np.ones((n, n)) - np.eye(n))
        num = abs(EIG(Tmat(A, (0, 1)))).sum()
        r = n - 2
        disc = (1 + w * r) ** 2 - 4 * w * (r - 1) + 4 * w * w * (r + 1)
        form = 2 + 2 * (n - 3) * w + max(0.0, math.sqrt(disc) - (1 + w * r))
        maxerr = max(maxerr, abs(num - form))
cf['Kn_max_err'] = maxerr
assert maxerr < 1e-12, maxerr
# star K_{1,m}, D_T={centre, one leaf}: TDE = 2 - 2*lambda_minus, cubic
maxerr = 0
for m in range(2, 9):
    for w in np.linspace(0.05, 1, 20):
        n = m + 1
        A = np.zeros((n, n))
        for l in range(1, n):
            A[0, l] = A[l, 0] = w
        num = abs(EIG(Tmat(A, (0, 1)))).sum()
        roots = np.roots([1, -2, 1 - m * w * w, (m - 1) * w * w])
        assert np.allclose(roots.imag, 0, atol=1e-9)
        roots = np.sort(roots.real)
        neg = [x for x in roots if x < -1e-12]
        assert len(neg) == 1
        maxerr = max(maxerr, abs(num - (2 - 2 * neg[0])))
cf['star_max_err'] = maxerr
assert maxerr < 1e-10
report['closed_forms'] = cf

# sample values for the paper (w = 0.5)
w = 0.5
A = np.zeros((4, 4))
for i in range(3):
    A[i, i + 1] = A[i + 1, i] = w
report['P4_w0.5'] = dict(E=float(abs(EIG(A)).sum()), TDE=float(abs(EIG(Tmat(A, (1, 2)))).sum()))
A = w * (np.ones((3, 3)) - np.eye(3))
report['K3_w0.5'] = dict(E=float(abs(EIG(A)).sum()), TDE=float(abs(EIG(Tmat(A, (0, 1)))).sum()),
                         formula=1 - w + math.sqrt(1 + 2 * w + 9 * w * w))
A = np.zeros((5, 5))
for l in range(1, 5):
    A[0, l] = A[l, 0] = w
roots = np.sort(np.roots([1, -2, 1 - 4 * w * w, 3 * w * w]).real)
report['star_m4_w0.5'] = dict(E=float(abs(EIG(A)).sum()), TDE=float(abs(EIG(Tmat(A, (0, 1)))).sum()),
                              lam_minus=float(roots[0]))

# ---------------------------------------------------------------- 4. Part C
def strong_edges():
    return list(edges)  # all six arcs are strong under Option B (verified in verify_optionB.py)
n = 5
nb = {v: set() for v in range(n)}
for u, v in strong_edges():
    nb[u].add(v); nb[v].add(u)
mins = [D for D in itertools.combinations(range(n), 2) if all(nb[v] & set(D) for v in range(n))]
assert [tuple(x + 1 for x in D) for D in mins] == [(2, 3), (2, 4), (3, 5)]

def stdle(A, D):
    nn = A.shape[0]
    T = Tmat(A, D)
    avg = A.sum() / nn          # = 2W/n
    QT = np.diag(T.sum(1)) + T
    return float(abs(EIG(QT) - avg).sum())

partC = []
for D in mins:
    row = dict(D=[v + 1 for v in D])
    for c, name in enumerate(['mu', 'ga', 'be']):
        A = mat(c)
        _, v = props(A, D)
        row['TDE_' + name] = v['TDE']
        row['STDLE_' + name] = stdle(A, D)
        row['theta_' + name] = v['theta']
        row['R_' + name] = v['R']
    omega = {}
    for c, name in enumerate(['mu', 'ga', 'be']):
        omega[name] = sum(edges[e][c] for e in strong_edges() if (e[0] in D or e[1] in D))
    omega['total'] = sum(omega[k] for k in ('mu', 'ga', 'be'))
    row['omega'] = omega
    partC.append(row)
report['partC'] = partC
# robustness of the selection rule: argmax of omega identical for every component and for the total
for key in ('mu', 'ga', 'be', 'total'):
    vals = [r['omega'][key] for r in partC]
    report.setdefault('argmax_omega', {})[key] = [partC[i]['D'] for i, x in enumerate(vals) if abs(x - max(vals)) < 1e-12]
# invariance of tr(T^2) across minimum sets
for name in ('mu', 'ga', 'be'):
    th = [r['theta_' + name] for r in partC]
    assert max(th) - min(th) < 1e-12
report['ranges'] = {}
for name in ('mu', 'ga', 'be'):
    report['ranges'][name] = dict(
        TDE=(min(r['TDE_' + name] for r in partC), max(r['TDE_' + name] for r in partC)),
        STDLE=(min(r['STDLE_' + name] for r in partC), max(r['STDLE_' + name] for r in partC)))

json.dump(report, open('results_step2.json', 'w'), indent=1, default=float)

f = lambda x: f'{x:.4f}'
print('EXAMPLE (Option B, D_T={v2,v3}):')
for k, v in ex.items():
    print(f"  {k}: E={f(v['E'])} TDE={f(v['TDE'])} TDE-E={f(v['TDE']-v['E'])} R={f(v['R'])} "
          f"sumw2={f(v['sumw2'])} theta={f(v['theta'])} lo={f(v['lo'])} hi={f(v['hi'])}")
    print('     shifts lambda_i(T)-lambda_i(A):', [f(x) for x in v['shifts']], 'sum=', f(sum(v['shifts'])))
print('random trials:', N_TRIALS, 'failures:', report['random_failures'])
print('K2 w=1:', report['K2_w1'])
print('closed forms:', cf)
print('P4 w=.5:', report['P4_w0.5'], ' formula', math.sqrt(1 + 1 + 1.25) + math.sqrt(1 - 1 + 1.25))
print('K3 w=.5:', report['K3_w0.5'])
print('star m=4 w=.5:', report['star_m4_w0.5'])
print('PART C:')
for r in partC:
    print('  D=', r['D'], {k: f(v) for k, v in r.items() if k.startswith(('TDE', 'STDLE'))}, 'omega=', {k: f(v) for k, v in r['omega'].items()})
print('argmax omega:', report['argmax_omega'])
print('ranges:', {k: {a: (f(b[0]), f(b[1])) for a, b in v.items()} for k, v in report['ranges'].items()})
print('ALL ASSERTIONS PASSED')
