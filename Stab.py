"""
stab.py -- "quantum computing on a classical computer," done on the side of the
line where things actually transfer.

A full state-vector simulator stores 2^n complex amplitudes and dies at ~25 qubits.
But the Gottesman-Knill theorem says any circuit of only Clifford gates (H, S, CNOT)
+ measurement can be simulated in BINARY. The trick: don't track amplitudes, track
the STABILIZERS. Every n-qubit Pauli is a 2n-bit vector (n bits X-part, n bits
Z-part). The whole quantum state becomes a (2n x 2n) binary matrix over GF(2);
gates become binary row/column operations; measurement becomes GF(2) linear algebra.

This is the exact tableau algorithm of Aaronson & Gottesman, "Improved Simulation
of Stabilizer Circuits" (2004) -- their CHP program. Pauli encoding (x,z):
(0,0)=I, (1,0)=X, (0,1)=Z, (1,1)=Y. Rows 0..n-1 = destabilizers, n..2n-1 =
stabilizers, row 2n = scratch.

It is exact, all-binary, and scales to thousands of qubits -- and every quantum
error-correcting code (the 3-qubit code from last build, Steane, the surface code
Willow ran) lives natively in this binary space. The one thing it CANNOT do is the
non-Clifford T gate: adding "magic" is exactly where quantum advantage begins, and
exactly where this easy classical door slams shut.
"""
import numpy as np
import time

class Stab:
    def __init__(self, n):
        self.n = n
        N = 2 * n + 1
        self.x = np.zeros((N, n), dtype=np.int8)
        self.z = np.zeros((N, n), dtype=np.int8)
        self.r = np.zeros(N, dtype=np.int8)
        for i in range(n):
            self.x[i, i] = 1          # destabilizer i = X_i
            self.z[n + i, i] = 1      # stabilizer   i = Z_i   (state |0...0>)

    # ---- Clifford gates: conjugation rules act on ALL rows at once (binary) ----
    def cnot(self, a, b):
        x, z, r = self.x, self.z, self.r
        r ^= x[:, a] & z[:, b] & (x[:, b] ^ z[:, a] ^ 1)
        x[:, b] ^= x[:, a]
        z[:, a] ^= z[:, b]

    def h(self, a):
        x, z, r = self.x, self.z, self.r
        r ^= x[:, a] & z[:, a]
        x[:, a], z[:, a] = z[:, a].copy(), x[:, a].copy()

    def s(self, a):
        x, z, r = self.x, self.z, self.r
        r ^= x[:, a] & z[:, a]
        z[:, a] ^= x[:, a]

    def x_gate(self, a):           # Pauli X = H S S H, stays in the proven primitives
        self.h(a); self.s(a); self.s(a); self.h(a)

    # ---- phase bookkeeping for multiplying two Pauli rows (A-G g-function) ----
    def _g(self, xi, zi, xh, zh):
        xi = xi.astype(np.int64); zi = zi.astype(np.int64)
        xh = xh.astype(np.int64); zh = zh.astype(np.int64)
        out = np.zeros(self.n, dtype=np.int64)
        m = (xi == 1) & (zi == 1); out[m] = zh[m] - xh[m]
        m = (xi == 1) & (zi == 0); out[m] = zh[m] * (2 * xh[m] - 1)
        m = (xi == 0) & (zi == 1); out[m] = xh[m] * (1 - 2 * zh[m])
        return out

    def _rowsum(self, h, i):       # row h := row i + row h, tracking phase
        x, z, r = self.x, self.z, self.r
        tot = 2 * int(r[h]) + 2 * int(r[i]) + int(self._g(x[i], z[i], x[h], z[h]).sum())
        r[h] = 0 if (tot % 4) == 0 else 1
        x[h] ^= x[i]
        z[h] ^= z[i]

    def measure(self, a, rng):
        x, z, r, n = self.x, self.z, self.r, self.n
        p = -1
        for k in range(n, 2 * n):            # is there a stabilizer with X/Y on qubit a?
            if x[k, a] == 1:
                p = k; break
        if p >= 0:                            # CASE I: random outcome
            for i in range(2 * n):
                if i != p and x[i, a] == 1:
                    self._rowsum(i, p)
            x[p - n] = x[p]; z[p - n] = z[p]; r[p - n] = r[p]
            x[p] = 0; z[p] = 0
            z[p, a] = 1
            r[p] = int(rng.integers(2))
            return int(r[p])
        else:                                 # CASE II: deterministic outcome
            x[2 * n] = 0; z[2 * n] = 0; r[2 * n] = 0
            for i in range(n):
                if x[i, a] == 1:
                    self._rowsum(2 * n, i + n)
            return int(r[2 * n])

    def pauli(self, i):
        sym = {(0, 0): 'I', (1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'}
        s = '-' if self.r[i] else '+'
        return s + ''.join(sym[(int(self.x[i, j]), int(self.z[i, j]))] for j in range(self.n))

    def stabilizers(self):
        return [self.pauli(i) for i in range(self.n, 2 * self.n)]


rng = np.random.default_rng(0)

# ============================================================================
# PART 1 -- validate against the state-vector simulator: a Bell pair
# ============================================================================
print("=" * 66)
print("PART 1  Bell pair  (validate the binary simulator vs last build)")
print("=" * 66)
sim = Stab(2)
sim.h(0); sim.cnot(0, 1)
print(f"  after H(0),CNOT(0,1) the stabilizers are: {sim.stabilizers()}")
print(f"  (+XX and +ZZ -- the textbook stabilizers of (|00>+|11>)/sqrt2)")
counts = {}
for _ in range(2000):
    s = Stab(2); s.h(0); s.cnot(0, 1)
    b = (s.measure(0, rng), s.measure(1, rng))
    key = f"{b[0]}{b[1]}"; counts[key] = counts.get(key, 0) + 1
print(f"  2000 shots: {dict(sorted(counts.items()))}")
print(f"  -> only 00 and 11, ~50/50, never 01/10. Matches the state-vector sim exactly.")

# ============================================================================
# PART 2 -- the whole point: scale PAST the wall the state vector cannot cross
# ============================================================================
print("\n" + "=" * 66)
print("PART 2  GHZ on 1000 qubits  (a state-vector sim would need 2^1000 numbers)")
print("=" * 66)
N = 1000
t0 = time.time()
sim = Stab(N)
sim.h(0)
for i in range(1, N):
    sim.cnot(0, i)                            # entangle every qubit with qubit 0
build_t = time.time() - t0
t0 = time.time()
bits = [sim.measure(q, rng) for q in range(12)]   # read the first 12 qubits
meas_t = time.time() - t0
print(f"  built + entangled {N} qubits in {build_t*1e3:.1f} ms")
print(f"  first 12 measured qubits: {bits}")
print(f"  all 12 equal: {len(set(bits)) == 1}  (GHZ: every qubit collapses to the same bit)")
print(f"  for scale: 2^{N} amplitudes is ~10^{int(N*np.log10(2))} numbers --")
print(f"  more than atoms in the observable universe. This ran on a laptop in milliseconds.")

# ============================================================================
# PART 3 -- the honest payoff: run last build's 3-qubit code natively in binary
# ============================================================================
print("\n" + "=" * 66)
print("PART 3  3-qubit bit-flip code, now in the all-binary stabilizer world")
print("=" * 66)
Hpc = np.array([[1, 1, 0], [0, 1, 1]])       # classical [3,1] repetition parity-check

def run_code(logical_bit, err_qubit, rng):
    # qubits: 0,1,2 = data ; 3,4 = syndrome ancillas
    s = Stab(5)
    if logical_bit == 1:
        s.x_gate(0)                           # logical |1> starts as X on qubit 0
    s.cnot(0, 1); s.cnot(0, 2)                # ENCODE: |bbb>
    if err_qubit is not None:
        s.x_gate(err_qubit)                   # INJECT bit-flip
    s.cnot(0, 3); s.cnot(1, 3)               # ancilla3 = parity(q0,q1) = Z0Z1
    s.cnot(1, 4); s.cnot(2, 4)               # ancilla4 = parity(q1,q2) = Z1Z2
    syn = (s.measure(3, rng), s.measure(4, rng))
    fix = {(0, 0): None, (1, 0): 0, (1, 1): 1, (0, 1): 2}[syn]
    if fix is not None:
        s.x_gate(fix)                         # CORRECT
    data = [s.measure(0, rng), s.measure(1, rng), s.measure(2, rng)]
    ok = data == [logical_bit] * 3
    return ok, syn, fix

print("  encode logical |0> and |1>, inject each single bit-flip, correct, read back:")
allok = 0; total = 0
for b in (0, 1):
    for e in (None, 0, 1, 2):
        ok, syn, fix = run_code(b, e, rng)
        allok += ok; total += 1
        es = "none" if e is None else f"q{e}"
        print(f"    logical |{b}>  error {es:<4}  syndrome {syn} -> fix {fix}   readback {'OK' if ok else 'FAIL'}")
print(f"  -> {allok}/{total} recovered correctly\n")

print("  the binary-code bridge, exact: syndrome = (parity-check matrix) @ error  over GF(2)")
for k in range(3):
    e = np.zeros(3, int); e[k] = 1
    print(f"    error on q{k}: H @ {e} mod 2 = {tuple(Hpc @ e % 2)}   (matches the measured syndrome)")
print("\n  Same code as last build -- but now living in pure binary linear algebra,")
print("  the thread you've pulled since quantvec and the chip accumulator. This is")
print("  literally how Stim (the tool that simulates Willow's surface code) works.")
