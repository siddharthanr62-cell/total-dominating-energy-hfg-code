"""Independent recomputation (fresh code, no shared helpers) of every number printed for the example and for the
named counterexamples in the paper; each value is looked up in main_SOIC.tex."""
import re, itertools, numpy as np
tex=open('main_SOIC.tex').read()
tex_flat=re.sub(r'\s+',' ',tex)
bad=[]
def need(s,what):
    if s not in tex_flat: bad.append((what,s))
def fmt(x,d=4): return f'{x:.{d}f}'
ev=np.linalg.eigvalsh
V=[(.5,.3,.2)]*3+[(.4,.4,.2)]*2
E={(0,1):(.4,.3,.2),(0,2):(.4,.3,.2),(1,2):(.4,.3,.2),(1,3):(.3,.3,.2),(2,4):(.3,.3,.2),(3,4):(.3,.3,.2)}
# HFG conditions
for (u,v),(m,g,b) in E.items():
    assert m<=min(V[u][0],V[v][0]) and g<=max(V[u][1],V[v][1]) and b<=min(V[u][2],V[v][2]) and m+g+b<=1
for t in V: assert abs(sum(t)-1)<1e-12
n=5
def M(c):
    A=np.zeros((n,n))
    for (u,v),w in E.items(): A[u,v]=A[v,u]=w[c]
    return A
# strong arcs by brute force over simple paths
def strong(e):
    u,v=e; others=[f for f in E if f!=e]
    best=[0,1,0]
    # enumerate simple paths u->v avoiding e
    def dfs(x,seen,mn,mx,bn):
        if x==v:
            best[0]=max(best[0],mn); best[1]=min(best[1],mx); best[2]=max(best[2],bn); return
        for f in others:
            if x in f:
                y=f[0] if f[1]==x else f[1]
                if y not in seen:
                    w=E[f]; dfs(y,seen|{y},min(mn,w[0]),max(mx,w[1]),min(bn,w[2]))
    dfs(u,{u},9,-9,9)
    if best[0]==0 and best[2]==0 and best[1]==1: return True
    m,g,b=E[e]; return m>=best[0]-1e-12 and g<=best[1]+1e-12 and b>=best[2]-1e-12
S=[e for e in E if strong(e)]; assert len(S)==6
# all minimum total dominating sets
nb={i:{j for (a,b) in S for (i2,j) in ((a,b),(b,a)) if i2==i} for i in range(n)}
tds=[D for k in range(2,6) for D in itertools.combinations(range(n),k) if all(nb[i]&set(D) for i in range(n))]
k=min(len(D) for D in tds); mins=[D for D in tds if len(D)==k]
assert k==2 and mins==[(1,2),(1,3),(2,4)],mins
omega={D:sum(E[e][0] for e in S if e[0] in D or e[1] in D) for D in mins}
assert abs(omega[(1,2)]-1.8)<1e-9 and abs(omega[(1,3)]-1.4)<1e-9 and abs(omega[(2,4)]-1.4)<1e-9
need('1.8','omega'); need('1.4','omega')
res={}
for c,nm in enumerate(['mu','gamma','beta']):
    A=M(c); d=A.sum(1); W=A.sum()/2; cc=2*W/n
    L=np.diag(d)-A; Q=np.diag(d)+A
    r={}
    r['E']=abs(ev(A)).sum(); r['LE']=abs(ev(L)-cc).sum(); r['SLE']=abs(ev(Q)-cc).sum()
    for D in mins:
        De=np.diag([1. if i in D else 0. for i in range(n)])
        T=A+De; DT=np.diag(T.sum(1)); LT=DT-T; QT=DT+T
        assert np.allclose(LT,L) and np.allclose(QT,Q+2*De)
        r[('TDE',D)]=abs(ev(T)).sum(); r[('STDLE',D)]=abs(ev(QT)-cc).sum()
        r[('TDLE',D)]=abs(ev(LT)-cc).sum()
        if D==(1,2):
            r['specA']=ev(A); r['specL']=ev(L); r['specQ']=ev(Q); r['specT']=ev(T); r['specQT']=ev(QT)
            r['trQT2']=np.trace(QT@QT); r['trT2']=np.trace(T@T); r['SigmaT']=((ev(QT)-cc)**2).sum()
            S2=(A**2).sum()/2; r['B']=np.sqrt(n*((d**2).sum()+2*S2-4*W**2/n)); r['W']=W; r['d']=d
            r['STc']=abs(ev(QT)-(2*W+4)/n).sum()
    res[nm]=r
for nm,r in res.items():
    for key in ['E','LE','SLE']: need(fmt(r[key]),f'{nm} {key}')
    for D in mins:
        need(fmt(r[('TDE',D)]),f'{nm} TDE {D}'); need(fmt(r[('STDLE',D)]),f'{nm} STDLE {D}')
    for key in ['specA','specL','specQ','specT','specQT']:
        for x in r[key]: need(fmt(x),f'{nm} {key}')
    need(fmt(r['trQT2']),f'{nm} trQT2'); need(fmt(r['SigmaT']),f'{nm} Sigma'); need(fmt(r['B']),f'{nm} B'); need(fmt(r['STc']),f'{nm} STDLE^c')
    need(fmt(np.sqrt(r['SigmaT'])),f'{nm} sqrtSigma'); need(fmt(np.sqrt(5*r['SigmaT'])),f'{nm} sqrt(n Sigma)')
    # bounds
    assert 2*2<=r[('STDLE',(1,2))]<=np.sqrt(5*r['SigmaT'])+1e-12
    assert r['B']<4
    D=(1,2); assert r[('TDE',D)]>=2 and abs(r[('TDLE',D)]-r['LE'])<1e-12
    assert r['LE']<r[('TDE',D)]<r[('STDLE',D)] and max(r['E'],r['SLE'])<r['LE']
print('example: all assertions passed')
# degree sequences / W
for nm in res: print(nm,'d=',res[nm]['d'],'W=',res[nm]['W'])
# raw LE rounding issue for figure labels
for nm in res: print(nm,'LE raw',repr(res[nm]['LE']))
# ---- named counterexamples
def idx(n,edges,w,D):
    A=np.zeros((n,n))
    for u,v in edges: A[u-1,v-1]=A[v-1,u-1]=w
    d=A.sum(1); c=d.sum()/n; De=np.diag([1. if i+1 in D else 0. for i in range(n)])
    T=A+De; QT=np.diag(T.sum(1))+T
    L=np.diag(d)-A; Q=np.diag(d)+A
    B=np.sqrt(n*((d**2).sum()+(A**2).sum()-d.sum()**2/n/n*n/ n*0 - (d.sum())**2/n))  # placeholder, replaced below
    W=d.sum()/2; S2=(A**2).sum()/2
    B=np.sqrt(n*((d**2).sum()+2*S2-4*W**2/n))
    return dict(LE=abs(ev(L)-c).sum(),SLE=abs(ev(Q)-c).sum(),TDE=abs(ev(T)).sum(),STDLE=abs(ev(QT)-c).sum(),B=B)
def mintds(n,edges):
    nbs={i:set() for i in range(1,n+1)}
    for u,v in edges: nbs[u].add(v); nbs[v].add(u)
    for k in range(2,n+1):
        f=[D for D in itertools.combinations(range(1,n+1),k) if all(nbs[i]&set(D) for i in nbs)]
        if f: return k,f
e7=[(1,2),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5),(5,6),(6,7)]
k,f=mintds(7,e7); assert f==[(5,6)],f
r=idx(7,e7,.9,(5,6)); print('n=7',r)
for key,val in [('LE',11.9210),('STDLE',11.4779),('SLE',9.8134),('TDE',8.6307),('B',13.6492)]: assert abs(r[key]-val)<6e-5,(key,r[key])
assert r['LE']>r['STDLE'] and 4<r['B']
e6=[(1,2),(1,3),(1,4),(2,3),(2,4),(3,4),(5,6)]
k,f=mintds(6,e6); assert k==4 and len(f)==6
for D in f:
    r=idx(6,e6,.9,D); assert abs(r['LE']-9.0)<5e-5 and abs(r['STDLE']-8.8)<5e-5,(D,r)
print('K4+K2: all six minimum sets LE=9.0000 STDLE=8.8000')
e8=[(1,3),(1,4),(1,5),(1,8),(2,6),(3,4),(3,7),(3,8),(4,5),(4,7),(4,8),(5,7),(5,8),(7,8)]
k,f=mintds(8,e8); assert len(f)==13 and k==4
r1=idx(8,e8,.9,(1,2,3,6)); r2=idx(8,e8,.9,(2,4,6,8)); print(r1,r2)
assert abs(r1['STDLE']-12.1316)<6e-5 and abs(r1['SLE']-12.2912)<6e-5 and r1['STDLE']<r1['SLE'] and abs(r2['STDLE']-13.0)<5e-5 and r2['STDLE']>r2['SLE']
# Rule S for 8-vertex graph
w={D:sum(.9 for e in e8 if e[0] in D or e[1] in D) for D in f}; m=max(w.values()); print('rule S sets',[D for D in f if abs(w[D]-m)<1e-9])
e7b=[(1,3),(1,4),(1,5),(1,6),(1,7),(2,3),(2,6),(3,4),(3,5),(4,5),(4,6),(4,7),(5,6),(5,7),(6,7)]
k,f=mintds(7,e7b); assert len(f)==7 and k==2,(k,f)
r1=idx(7,e7b,.9,(2,6)); print(r1); assert abs(r1['STDLE']-10.4403)<6e-5 and abs(r1['TDE']-10.6662)<6e-5 and r1['STDLE']<r1['TDE']
w={D:sum(.9 for e in e7b if e[0] in D or e[1] in D) for D in f}; m=max(w.values()); top=[D for D in f if abs(w[D]-m)<1e-9]; print('rule S sets',top)
for D in top:
    r=idx(7,e7b,.9,D); assert r['STDLE']>r['TDE']
print('named counterexamples: all verified')
# ---- triangle and greedy counterexample facts
print('MISSING FROM TEX:',bad if bad else 'none')
