"""
chaos.py -- "shoot chaos into a box, give it a task, extract the answer."

That sentence describes two real, named paradigms. This file builds both, in
pure NumPy, and runs them before claiming anything.

  Part 1  MONTE CARLO: the simplest box of chaos. Throw random darts, extract
          a number (pi). Randomness in -> deterministic answer out.
          (Ulam / von Neumann / Metropolis, 1940s.)

  Part 2  RESERVOIR COMPUTING: the real version of your idea. A fixed RANDOM
          high-dimensional dynamical system is the "box". You drive it with a
          task (a chaotic signal). The box's internal state explodes into a
          400-dimensional vector of nonlinear "possibilities". You EXTRACT the
          desired result with a single LINEAR readout trained by least squares.
          (Jaeger's echo state networks 2001; Maass's liquid state machines
          2002; reservoirs were later shown to predict chaotic systems.)

This is NOT your invention -- but the principle you described ("chaos in a box,
extract the answer linearly, mind the vector shapes") is EXACTLY what these run
on. The shapes even matter the way you guessed: 3 -> 400 -> 3, a non-square lift
into a big space and a non-square projection back out.

And it ties to chaos theory honestly: the same sensitive-dependence that lets the
box compute also guarantees the prediction must eventually fail. We measure both.
"""
import numpy as np
rng = np.random.default_rng(0)

# ===================================================================
# PART 1 -- Monte Carlo: the simplest box of chaos
# ===================================================================
print("=" * 66)
print("PART 1  Monte Carlo -- random darts in a box, extract pi")
print("=" * 66)
n = 4_000_000
p = rng.random((n, 2))
inside = int(np.sum((p ** 2).sum(1) <= 1.0))
pi_est = 4 * inside / n
print(f"  {n:,} random darts ->  pi ~ {pi_est:.5f}   (true {np.pi:.5f}, "
      f"error {abs(pi_est-np.pi):.5f})")
print("  Randomness goes in; a desired deterministic number comes out.")

# ===================================================================
# PART 2a -- prove the Lorenz system is actually chaotic
# ===================================================================
print("\n" + "=" * 66)
print("PART 2a  Is the box's task really chaotic? (Lyapunov exponent)")
print("=" * 66)
def lorenz_step(x, dt=0.01, s=10.0, r=28.0, b=8/3):
    def f(v):
        return np.array([s*(v[1]-v[0]), v[0]*(r-v[2])-v[1], v[0]*v[1]-b*v[2]])
    k1 = f(x); k2 = f(x+dt/2*k1); k3 = f(x+dt/2*k2); k4 = f(x+dt*k3)
    return x + dt/6*(k1+2*k2+2*k3+k4)

x = np.array([1.0, 1.0, 1.0])
for _ in range(5000):
    x = lorenz_step(x)                      # settle onto the attractor
# Benettin method: track exponential separation of two nearby trajectories
d0 = 1e-9; dt = 0.01; gap = 5; M = 3000
xa = x.copy(); xb = x + np.array([d0, 0, 0]); acc = 0.0
for _ in range(M):
    for _ in range(gap):
        xa = lorenz_step(xa, dt); xb = lorenz_step(xb, dt)
    d = np.linalg.norm(xb - xa)
    acc += np.log(d / d0)
    xb = xa + (xb - xa) * (d0 / d)          # renormalize back to d0
lyap = acc / (M * gap * dt)
print(f"  largest Lyapunov exponent  lambda ~ {lyap:.3f}  (known value ~0.906)")
print(f"  lambda > 0  ->  CHAOS confirmed: nearby states diverge like e^(lambda t).")
print(f"  predictability horizon ~ 1/lambda ~ {1/lyap:.1f} time units.")

# ===================================================================
# PART 2b -- reservoir computing: chaos in a box, extract the prediction
# ===================================================================
print("\n" + "=" * 66)
print("PART 2b  Reservoir computing -- drive a random box, extract the answer")
print("=" * 66)
# the task signal: a Lorenz trajectory
T = 12000
traj = np.zeros((T, 3)); xx = np.array([1.0, 1.0, 1.0])
for _ in range(2000):
    xx = lorenz_step(xx, 0.02)
for i in range(T):
    xx = lorenz_step(xx, 0.02); traj[i] = xx
U = (traj - traj.mean(0)) / traj.std(0)     # normalize

# the BOX: a fixed random reservoir
Nr = 400
Win = rng.uniform(-1, 1, (Nr, 3)) * 1.0     # non-square: 3 -> 400 (lift)
W = rng.uniform(-1, 1, (Nr, Nr)) * (rng.random((Nr, Nr)) < 0.05)   # 5% sparse
W *= 0.9 / np.max(np.abs(np.linalg.eigvals(W)))                    # spectral radius 0.9

# drive the box with the task
R = np.zeros((T, Nr)); r = np.zeros(Nr)
for t in range(T):
    r = np.tanh(W @ r + Win @ U[t]); R[t] = r

# EXTRACT the answer: one linear readout, predict next step from box state.
# Train on NOISY reservoir states (Jaeger's trick) so the closed loop stays stable;
# evaluate on clean states.
wash = 500
states = R[wash:T-1]; Y = U[wash+1:T]
ntr = int(0.7 * len(states))
noisy = states[:ntr] + rng.normal(0, 1e-2, states[:ntr].shape)
Ptr = np.hstack([noisy, np.ones((ntr, 1))])
A = Ptr.T @ Ptr + 1e-6*np.eye(Ptr.shape[1])
Wout = np.linalg.solve(A, Ptr.T @ Y[:ntr])               # (Nr+1) -> 3  (project)
Pte = np.hstack([states[ntr:], np.ones((len(states)-ntr, 1))])
pred = Pte @ Wout
nrmse = np.sqrt(np.mean((pred - Y[ntr:])**2) / np.var(Y[ntr:]))
print(f"  box = {Nr}-dim random reservoir.  shapes: 3 -> {Nr} -> 3 (non-square, on purpose)")
print(f"  one-step prediction NRMSE = {nrmse:.4f}   (linear readout pulls it out of the chaos)")

# let the box run FREE (feed its own predictions back) -> chaos must break it
warm = 300; s0 = ntr + wash + 100
r = np.zeros(Nr)
for t in range(s0-warm, s0):
    r = np.tanh(W @ r + Win @ U[t])
u = U[s0].copy(); free = []
for _ in range(900):
    r = np.tanh(W @ r + Win @ u)
    u = np.hstack([r, 1.0]) @ Wout            # predicted next state -> next input
    free.append(u)
free = np.array(free); true = U[s0+1:s0+1+len(free)]
err = np.linalg.norm(free - true, axis=1)
horizon = int(np.argmax(err > 2.0)) if np.any(err > 2.0) else len(err)
print(f"  free-running: tracks the true butterfly for ~{horizon} steps "
      f"(~{horizon*0.02:.1f} time units), then diverges.")
print("  Two honest reasons it diverges: chaos (sensitive dependence, Part 2a)")
print("  GUARANTEES eventual failure, and closed-loop model error accelerates it.")
print("  Longer horizons (the famous Pathak-style attractor reconstruction) need")
print("  more careful tuning than this quick build -- I'm not claiming that here.")
print("\n  The box computes with chaos; chaos also bounds what the box can know.")
print("  Real paradigm (Monte Carlo + reservoir computing), reproduced -- not invented.")
print("  But the principle is exactly the one you described.")
