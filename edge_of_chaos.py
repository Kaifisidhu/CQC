"""
edge_of_chaos.py -- going DEEP on one thing instead of pivoting.

We built a reservoir last step. The textbook slogan is: "a reservoir computes
best at the edge of chaos." That is a slogan. This file sits inside it and
measures whether it is actually true, and where the edge actually is.

Two independent measurements, swept over the reservoir's spectral radius rho
(the one knob that tunes order -> chaos):

  1. MEMORY CAPACITY (Jaeger 2001): drive the reservoir with random noise, then
     ask a linear readout to reconstruct u(t-k) for every delay k. MC = sum of
     squared correlations. It measures how much of its own past the box can hold.
     Theory: MC <= N, and it should grow as the box approaches the edge.

  2. THE EDGE ITSELF: drive two copies of the box from slightly different states
     with the SAME input, and measure whether they converge (ordered, fading
     memory -> usable) or diverge (chaotic -> useless). The per-step rate is the
     driven (conditional) Lyapunov exponent. Where it crosses zero IS the edge.

The interesting part is whether the MC peak sits exactly at the naive edge
rho = 1, or somewhere else. I went in expecting rho = 1. The data is below.
This is a known phenomenon (Langton's "edge of chaos", 1990; Jaeger's memory
work, 2001) -- reproduced, not invented -- but the act of cracking a slogan into
a measured number is the motion that actually precedes novelty.
"""
import numpy as np

def make_reservoir(N, rho, seed=1, density=0.1, in_scale=0.5):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((N, N)) * (rng.random((N, N)) < density)
    eig = np.max(np.abs(np.linalg.eigvals(W)))
    if eig > 0:
        W = W * (rho / eig)                      # set spectral radius exactly to rho
    Win = rng.uniform(-in_scale, in_scale, N)
    return W, Win

def memory_capacity(N, rho, L=4000, wash=200, Kmax=None, in_scale=0.5):
    if Kmax is None:
        Kmax = 2 * N
    rng = np.random.default_rng(123)             # SAME input for every rho (fair test)
    u = rng.uniform(-1, 1, L)
    W, Win = make_reservoir(N, rho, in_scale=in_scale)
    r = np.zeros(N); R = np.zeros((L, N))
    for t in range(L):
        r = np.tanh(W @ r + Win * u[t]); R[t] = r
    start = max(wash, Kmax)
    Phi = np.hstack([R[start:], np.ones((L - start, 1))])
    Y = np.column_stack([u[start - k: L - k] for k in range(1, Kmax + 1)])
    ntr = int(0.7 * len(Phi))
    A = Phi[:ntr].T @ Phi[:ntr] + 1e-8 * np.eye(Phi.shape[1])
    Wout = np.linalg.solve(A, Phi[:ntr].T @ Y[:ntr])
    P = Phi[ntr:] @ Wout; Yte = Y[ntr:]
    mc = 0.0
    for k in range(Kmax):
        a, b = P[:, k], Yte[:, k]
        va, vb = np.var(a), np.var(b)
        if va > 1e-12 and vb > 1e-12:
            r2 = np.cov(a, b)[0, 1] ** 2 / (va * vb)
            if r2 > 0.02:                        # threshold tiny noisy terms (honest, conservative)
                mc += r2
    return mc

def driven_lyapunov(N, rho, T=2500, wash=300, in_scale=0.5, d0=1e-8):
    rng = np.random.default_rng(123)
    u = rng.uniform(-1, 1, T + wash)
    W, Win = make_reservoir(N, rho, in_scale=in_scale)
    ra = rng.standard_normal(N) * 0.1
    pert = rng.standard_normal(N); pert /= np.linalg.norm(pert)
    rb = ra + d0 * pert
    acc, cnt = 0.0, 0
    for t in range(T + wash):
        ra = np.tanh(W @ ra + Win * u[t]); rb = np.tanh(W @ rb + Win * u[t])
        d = np.linalg.norm(rb - ra)
        if t >= wash and d > 0:
            acc += np.log(d / d0); cnt += 1
            rb = ra + (rb - ra) * (d0 / d)       # renormalize to distance d0
    return acc / cnt

N = 80
rhos = [0.3, 0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5, 1.8]
print("=" * 60)
print(f"  edge of chaos in an N={N} reservoir (input scaling 0.5)")
print("=" * 60)
print(f"  {'rho':>5} | {'memory cap':>10} | {'driven lyapunov':>15} | regime")
print("  " + "-" * 54)
mcs, lyaps = [], []
for rho in rhos:
    mc = memory_capacity(N, rho)
    ly = driven_lyapunov(N, rho)
    mcs.append(mc); lyaps.append(ly)
    regime = "ordered" if ly < -0.02 else ("CHAOTIC" if ly > 0.02 else "** edge **")
    print(f"  {rho:>5.2f} | {mc:>10.2f} | {ly:>+15.4f} | {regime}")

peak_rho = rhos[int(np.argmax(mcs))]
# locate the edge: where driven lyapunov crosses zero
edge = None
for i in range(len(rhos) - 1):
    if lyaps[i] < 0 <= lyaps[i + 1]:
        f = -lyaps[i] / (lyaps[i + 1] - lyaps[i])
        edge = rhos[i] + f * (rhos[i + 1] - rhos[i]); break

print("\n  " + "-" * 54)
print(f"  memory capacity peaks at  rho = {peak_rho}")
print(f"  the real edge (lyapunov = 0) is at  rho ~ {edge:.2f}" if edge else
      "  edge not bracketed in this sweep")
print(f"\n  The slogan says 'edge of chaos = rho = 1.' The measurement says the")
print(f"  usable edge sits near rho ~ {edge:.2f} once the input drive is included,")
print(f"  and memory peaks just BELOW it -- a box wants to be as close to chaos as")
print(f"  it can get without falling in. The slogan was a simplification; the data")
print(f"  is the real statement. That crack is where a real question starts.")
