"""
css.py -- the ONE place everything actually combines.

You said "combine everything." Most of it does not combine, and I'll say so
plainly at the bottom. But there is exactly one real, load-bearing bridge
running through the spine of this whole body of work -- the binary / GF(2)
thread -- and it has a name worth earning:

    classical error-correcting codes  ==(CSS construction)==  quantum codes

Both are the SAME linear algebra over GF(2). This file proves it by running it:
one GF(2) kernel builds the classical [7,4,3] Hamming code, then the IDENTICAL
kernel turns that classical code into the quantum [[7,1,3]] Steane code via the
Calderbank-Shor-Steane construction (1996). The payoff: quantum error
correction here reduces to running classical Hamming decoding twice -- once for
bit-flips (X), once for phase-flips (Z).

This is the honest answer to "combine everything": the classical-coding work and
the quantum-coding work were never two things. CSS is the bridge; one kernel
walks across it.
"""
import numpy as np

# ===== the one shared kernel: linear algebra over GF(2) ===================
def gf2_rref(M):
    A = (np.array(M) & 1).astype(np.int8).copy()
    rows, cols = A.shape; r = 0; pivots = []
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c]), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        pivots.append(c); r += 1
        if r == rows:
            break
    return A, pivots

def gf2_rank(M):
    return len(gf2_rref(M)[1])

def gf2_nullspace(M):
    A = (np.array(M) & 1).astype(np.int8)
    R, piv = gf2_rref(A); cols = A.shape[1]
    free = [c for c in range(cols) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(cols, np.int8); v[f] = 1
        for ri, pc in enumerate(piv):
            if R[ri, f]:
                v[pc] = 1
        basis.append(v)
    return np.array(basis, np.int8) if basis else np.zeros((0, cols), np.int8)

# ===== PART 1: classical [7,4,3] Hamming code, from the kernel ============
print("=" * 64)
print("PART 1  classical [7,4,3] Hamming code   (syndrome = H . e^T)")
print("=" * 64)
H = np.array([[(c >> b) & 1 for c in range(1, 8)] for b in range(3)], np.int8)
print("  parity-check H (3x7), column c = binary(c):")
for row in H:
    print("     ", row)
G = gf2_nullspace(H)                       # codewords = null space of H
print(f"  code dimension k = {G.shape[0]}  ->  [7,{G.shape[0]},3], {2**G.shape[0]} codewords")
ok = 0
for pos in range(7):
    e = np.zeros(7, np.int8); e[pos] = 1
    syn = (H @ e) & 1
    decoded = sum(int(b) << i for i, b in enumerate(syn)) - 1   # syndrome = binary(pos+1)
    ok += (decoded == pos)
print(f"  single-bit-error correction: {ok}/7   (the syndrome IS the bad bit's address)")

# ===== PART 2: the CSS condition -- the entire bridge in one equation =====
print("\n" + "=" * 64)
print("PART 2  CSS condition:   H . H^T = 0  over GF(2)   (dual-containing)")
print("=" * 64)
HHt = (H @ H.T) & 1
for row in HHt:
    print("     ", row)
assert np.all(HHt == 0), "Hamming code is not dual-containing!"
print("  All zero -> the Hamming code contains its own dual. That single fact is")
print("  what lets ONE classical matrix serve as both the X- and Z-checks of a")
print("  quantum code without the checks fighting each other.")

# ===== PART 3: CSS -> quantum [[7,1,3]] Steane code, SAME kernel ===========
print("\n" + "=" * 64)
print("PART 3  CSS:  classical Hamming  ->  quantum Steane [[7,1,3]]")
print("=" * 64)
n = 7
Sx = np.hstack([H, np.zeros((3, n), np.int8)])      # X-type stabilizers [x|0]
Sz = np.hstack([np.zeros((3, n), np.int8), H])      # Z-type stabilizers [0|z]
S = np.vstack([Sx, Sz])                              # 6 x 14 symplectic check matrix
X, Z = S[:, :n], S[:, n:]
commute = np.all(((X @ Z.T + Z @ X.T) & 1) == 0)     # symplectic product
print(f"  all 6 stabilizers commute? {commute}   (this IS H.H^T = 0 from Part 2)")
rank = gf2_rank(S)
print(f"  logical qubits k = {n} - {rank} = {n - rank}   ->  [[7,{n - rank},3]]")

# single-qubit-error correction = TWO classical Hamming decodes
paulis = {"X": (1, 0), "Z": (0, 1), "Y": (1, 1)}
seen = {}; total = 0; nonzero = 0; distinct = True
for q in range(n):
    for name, (xb, zb) in paulis.items():
        xe = np.zeros(n, np.int8); ze = np.zeros(n, np.int8)
        if xb: xe[q] = 1
        if zb: ze[q] = 1
        s = (tuple(((H @ ze) & 1)), tuple(((H @ xe) & 1)))   # (Z-err synd, X-err synd)
        total += 1
        if s != ((0, 0, 0), (0, 0, 0)):
            nonzero += 1
        if s in seen and seen[s] != (name, q):
            distinct = False
        seen[s] = (name, q)
print(f"  single-qubit errors with nonzero syndrome: {nonzero}/{total}")
print(f"  all {total} single-qubit errors have distinct syndromes? {distinct}")
print("  Each correction = classical Hamming syndrome on the X-part, and the SAME")
print("  decode on the Z-part. The quantum decoder IS the classical one, run twice.")

# ===== honest map =========================================================
print("\n" + "=" * 64)
print("HONEST MAP -- what combines, what does not")
print("=" * 64)
print("""  LOAD-BEARING  (one GF(2) kernel; theorems + algorithms transfer and run):
      classical Hamming  <->  Steane / surface stabilizer codes  <->  the CHP
      stabilizer simulator.  CSS is the literal bridge; GF(2) syndrome decoding
      is the shared algorithm. Every line above ran from a single kernel.

  NOT LOAD-BEARING  (connected only by metaphor -- said honestly):
      - chaos / reservoir computing: continuous nonlinear dynamics over R.
        A different mathematical world. Does not join this spine.
      - binary quantization / ITQ hashing: same alphabet (bits), but it is
        vector quantization for search, not a linear code. No syndrome to
        transfer. Same letters, different language.
      - RNS accumulator (CRT over Z/mZ): adjacent modular arithmetic, a cousin
        of finite-field algebra -- not the same family.

  "Combine everything" contained exactly one true combination. This was it,
  and it was load-bearing all along.""")
