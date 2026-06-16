"""
quantum1.py -- a minimal quantum computer in pure NumPy, then real error correction.

A quantum state of n qubits is a 2^n complex vector. Gates are unitary matrices.
Measurement samples an outcome with probability |amplitude|^2. That's the whole
machine -- the same linear algebra (vectors, unitary/orthogonal matrices, SVD-era
tools) used everywhere in this project.

Then we put the 3-QUBIT BIT-FLIP CODE on top: encode one logical qubit into three
physical qubits, inject a bit-flip error, read PARITY CHECKS (the syndrome) on two
ancilla qubits, correct, and confirm the logical state survived.

Honest framing: this is the conceptual ancestor of the SURFACE CODE that Google's
Willow chip ran "below threshold" in 2024. It corrects ONE bit-flip and no phase
errors; the surface code generalizes it to a 2D grid catching both. NumPy state
vectors die around ~20-25 qubits (2^25 complex = 0.5 GB), so this is a teaching
simulator, not hardware.
"""
import numpy as np

rng = np.random.default_rng(0)

# ---- gates ----
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

def zero_state(n):
    s = np.zeros(2 ** n, dtype=complex)
    s[0] = 1.0
    return s

def apply_1q(state, U, q, n):
    t = state.reshape([2] * n)
    t = np.tensordot(U, t, axes=([1], [q]))
    return np.moveaxis(t, 0, q).reshape(-1)

def apply_2q(state, U4, qa, qb, n):
    t = state.reshape([2] * n)
    U = U4.reshape(2, 2, 2, 2)                 # (a_out, b_out, a_in, b_in)
    t = np.tensordot(U, t, axes=([2, 3], [qa, qb]))
    return np.moveaxis(t, [0, 1], [qa, qb]).reshape(-1)

def sample(state, n, shots, rng):
    p = np.abs(state) ** 2
    p /= p.sum()
    idx = rng.choice(2 ** n, size=shots, p=p)
    out = {}
    for i in idx:
        b = format(i, f"0{n}b")
        out[b] = out.get(b, 0) + 1
    return dict(sorted(out.items()))

# ============================================================================
# PART 1 -- sanity check: a Bell pair must give only 00 and 11, ~50/50
# ============================================================================
print("=" * 64)
print("PART 1  Bell pair  (validate the simulator)")
print("=" * 64)
n = 2
s = zero_state(n)
s = apply_1q(s, H, 0, n)        # superpose qubit 0
s = apply_2q(s, CNOT, 0, 1, n)  # entangle: (|00> + |11>)/sqrt(2)
amp = {format(i, "02b"): np.round(s[i], 3) for i in range(4) if abs(s[i]) > 1e-9}
print(f"  amplitudes: {amp}")
print(f"  10000 shots: {sample(s, n, 10000, rng)}")
print(f"  -> only 00 and 11 appear (entangled); never 01 or 10.")

# ============================================================================
# PART 2 -- 3-qubit bit-flip code: encode, error, syndrome, correct, verify
# ============================================================================
print("\n" + "=" * 64)
print("PART 2  3-qubit bit-flip code  (detect + correct a real error)")
print("=" * 64)

# parity-check matrix of the classical [3,1] repetition code -- the honest bridge
# to the binary-code thread: the quantum syndrome is exactly H_classical @ error.
Hpc = np.array([[1, 1, 0], [0, 1, 1]])   # checks Z0Z1 and Z1Z2

def run_code(alpha, beta, err_qubits, rng, verbose=False):
    n = 5                                  # 3 data (0,1,2) + 2 ancilla (3,4)
    s = np.zeros(2 ** n, dtype=complex)
    s[0] = alpha; s[1 << (n - 1)] = 0       # |00000>
    # load logical qubit on data qubit 0:  (alpha|0> + beta|1>) on q0
    s = zero_state(n) * 0
    s[0b00000] = alpha                      # |0> logical part
    s[0b10000] = beta                       # q0=1, rest 0
    # ENCODE: copy q0 -> q1, q2  (alpha|000> + beta|111>) on data qubits
    s = apply_2q(s, CNOT, 0, 1, n)
    s = apply_2q(s, CNOT, 0, 2, n)
    ideal = np.zeros(8, dtype=complex); ideal[0b000] = alpha; ideal[0b111] = beta

    # INJECT ERROR(S): bit-flip X on the chosen data qubit(s)
    for q in err_qubits:
        s = apply_1q(s, X, q, n)

    # SYNDROME: ancilla3 = parity(q0,q1), ancilla4 = parity(q1,q2)
    s = apply_2q(s, CNOT, 0, 3, n)
    s = apply_2q(s, CNOT, 1, 3, n)
    s = apply_2q(s, CNOT, 1, 4, n)
    s = apply_2q(s, CNOT, 2, 4, n)

    # measure the two ancillas (collapse), read syndrome bits
    t = s.reshape([2] * n)
    probs = np.array([[np.abs(t[:, :, :, a, b]).sum() ** 2 if False else
                       (np.abs(t[:, :, :, a, b]) ** 2).sum() for b in (0, 1)] for a in (0, 1)])
    probs /= probs.sum()
    flat = probs.ravel(); pick = rng.choice(4, p=flat)
    s1, s2 = pick // 2, pick % 2
    data = t[:, :, :, s1, s2].reshape(-1)
    data = data / np.linalg.norm(data)

    # DECODE the syndrome -> which qubit to fix (classical lookup)
    table = {(0, 0): None, (1, 0): 0, (1, 1): 1, (0, 1): 2}
    fix = table[(s1, s2)]
    if fix is not None:
        data = apply_1q(data, X, fix, 3)       # correct on the 3-qubit data space

    fid = abs(np.vdot(ideal, data)) ** 2
    if verbose:
        lbl = str(err_qubits) if err_qubits else "none"
        print(f"    error on {lbl:<6}  syndrome ({s1},{s2})  "
              f"-> fix qubit {fix}   fidelity {fid:.3f}")
    return fid, (s1, s2), fix

# logical state to protect: a nontrivial superposition
alpha, beta = np.sqrt(0.7), np.sqrt(0.3)
print(f"  logical qubit: {alpha:.3f}|0> + {beta:.3f}|1>  encoded as alpha|000> + beta|111>\n")
print("  single bit-flip errors (what a distance-3 code is built to fix):")
ok = 0
for q in [[], [0], [1], [2]]:
    fid, syn, fix = run_code(alpha, beta, q, rng, verbose=True)
    ok += fid > 0.999
print(f"  -> {ok}/4 recovered to fidelity 1.000\n")

print("  syndrome = (classical parity-check matrix) @ (error vector), over GF(2):")
for e_idx, name in [(0, "q0"), (1, "q1"), (2, "q2")]:
    e = np.zeros(3, int); e[e_idx] = 1
    print(f"    error {e} -> syndrome {tuple(Hpc @ e % 2)}   (matches measured above)")

print("\n  the code's LIMIT (honest): a 2-bit error is mis-corrected ->")
fid2, syn2, fix2 = run_code(alpha, beta, [0, 1], rng, verbose=True)
print(f"    two flips give fidelity {fid2:.3f}: distance-3 corrects ONE error, not two,")
print(f"    and it catches no phase (Z) errors at all -- that is what the surface code fixes.")
