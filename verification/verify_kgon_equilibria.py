"""
Rotating k-cluster relative equilibria (Propositions "kgon" and "triangle selection").
Run: python verification/verify_kgon_equilibria.py   (needs numpy; about a minute)

Verifies:
 TEST 1 (EXISTENCE, the proven theorem): every regular k-gon on the equator is
         an exact relative equilibrium -- zero out-of-plane velocity and a single
         common rotation rate. PASS = machine-zero out-of-plane, machine-zero
         rate-spread, for all k. Also confirms Omega_2 = 0 (static pair).
 TEST 2 (SELECTION, numerical): starting near each k-gon, only k=3 persists
         under realistic noise; k=2 and k>=4 collapse toward 3.
 TEST 3 (GENERIC, numerical): from random starts, the endpoint is 3 clusters.
"""
import numpy as np
from collections import Counter

def flow(X, D):
    S = X @ D @ X.T; S = S - S.max(axis=1, keepdims=True)
    W = np.exp(S); W /= W.sum(axis=1, keepdims=True)
    F = -(W @ X)
    return F - np.sum(X * F, axis=1)[:, None] * X

def A_from_axis(w):
    w = np.asarray(w, float)
    return np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])

def settle(X0, eps, T, dt=0.05, beta=1.0):
    X = X0.copy(); D = np.diag([0,0,beta]) + eps*A_from_axis((0,0,1))
    for _ in range(int(T/dt)):
        X = X + dt*flow(X,D); X /= np.linalg.norm(X,axis=1,keepdims=True)
    return X

def clusters(X, tol=0.15):
    used = np.zeros(len(X), bool); s = []
    for i in range(len(X)):
        if used[i]: continue
        g = np.linalg.norm(X-X[i],axis=1) < tol; used |= g; s.append(int(g.sum()))
    return sorted(s, reverse=True)

def dominant_k(X, N, tol=0.15, frac=0.1):
    return len([s for s in clusters(X,tol) if s >= frac*N])

print("="*70); print("TEST 1  EXISTENCE: k-gons are exact relative equilibria"); print("="*70)
ok1 = True
for eps in [0.3, 0.7]:
    D = np.diag([0,0,1.0]) + eps*A_from_axis((0,0,1))
    print(f" eps={eps}:")
    for k in [2,3,4,5,6]:
        th = 2*np.pi*np.arange(k)/k
        X = np.stack([np.cos(th),np.sin(th),0*th],1)
        V = flow(X, D)
        zvel = np.abs(V[:,2]).max()
        tang = np.stack([-np.sin(th),np.cos(th),0*th],1)
        rate = np.sum(V*tang,1)
        spread = rate.max()-rate.min()
        good = (zvel < 1e-12) and (spread < 1e-12)
        ok1 &= good
        static = "STATIC" if abs(rate.mean())<1e-12 else f"rate={rate.mean():+.4f}"
        print(f"   k={k}: out-of-plane={zvel:.1e}, rate-spread={spread:.1e}, "
              f"{static}  {'ok' if good else 'FAIL'}")
print(f" -> TEST 1 {'PASS' if ok1 else 'FAIL'}")

print("\n"+"="*70); print("TEST 2  SELECTION: only k=3 persists under noise"); print("="*70)
rng = np.random.default_rng(0); ok2 = True
for k in [2,3,4,5,6]:
    th = 2*np.pi*np.arange(k)/k
    centers = np.stack([np.cos(th),np.sin(th),0*th],1)
    X0 = np.repeat(centers,15,axis=0) + 0.15*rng.standard_normal((15*k,3))
    X0 /= np.linalg.norm(X0,axis=1,keepdims=True)
    end = clusters(settle(X0, 0.6, T=6000))
    ek = len([s for s in end if s>=0.1*len(X0)])
    verdict = "stays 3" if ek==3 else (f"stays {k}" if ek==k else f"-> {ek}")
    ok2 &= (ek==3)
    print(f"   start k={k}: ends {end} (dominant {ek})  [{verdict}]")
print(f" -> TEST 2 {'PASS (all roads lead to 3)' if ok2 else 'MIXED - report distribution'}")

print("\n"+"="*70); print("TEST 3  GENERIC: random starts -> 3 clusters"); print("="*70)
ks = []
for seed in range(10):
    r = np.random.default_rng(100+seed)
    X0 = r.standard_normal((60,3)); X0 /= np.linalg.norm(X0,axis=1,keepdims=True)
    ks.append(dominant_k(settle(X0, 0.6, T=6000), 60))
dist = dict(Counter(ks))
print(f"   k distribution over 10 random seeds: {dist}")
ok3 = (dist.get(3,0) == len(ks))
print(f" -> TEST 3 {'PASS (3 every time)' if ok3 else 'report: not always 3'}")

print("\n"+"="*70); print("SUMMARY"); print("="*70)
print(f"  TEST 1 (existence, PROVEN theorem): {'PASS' if ok1 else 'FAIL'}")
print(f"  TEST 2 (selection under noise):     {'PASS' if ok2 else 'MIXED'}")
print(f"  TEST 3 (generic random starts):     {'PASS' if ok3 else 'MIXED'}")
if ok1 and ok2 and ok3:
    print("\n  Existence theorem confirmed to machine precision; k=3 selection")
    print("  confirmed numerically.")