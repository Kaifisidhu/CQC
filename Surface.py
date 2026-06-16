"""
surface.py -- the distance-3 rotated surface code ("surface-17"), the real object
Google's Willow ran, built on the same binary linear algebra as stab.py.

This is NOT a new invention. The surface code is Kitaev (1997); this rotated
[[9,1,3]] layout is textbook. What this file does is build it correctly from
scratch and verify, by running, every claim about it:

  - the 8 stabilizer generators mutually commute (it is a valid code),
  - it encodes exactly 1 logical qubit in 9 physical qubits,
  - its code distance is 3,
  - it corrects EVERY single-qubit error -- X, Y, and Z -- which is the real
    advance over last build's 3-qubit code (that caught only X, in 1-D),
  - and it fails, exactly as the distance predicts, on weight-2 errors.

Everything is Pauli-as-binary-vector over GF(2): a 9-qubit Pauli is an 18-bit
vector [x(9) | z(9)]. The syndrome of an error e is which stabilizers it
ANTICOMMUTES with -- and "anticommute" is just the symplectic inner product
e^T J s over GF(2). The measured syndrome on real hardware EQUALS this product
(a stabilizer measurement returns +1/-1 = whether the state stayed in the +1
eigenspace), so this is exact, not a model.
"""
import numpy as np
from itertools import combinations

# ---- GF(2) linear algebra ----
def gf2_rref(M):
    M = (np.array(M, dtype=np.int8) % 2).copy()
    m, n = M.shape
    piv = []
    r = 0
    for c in range(n):
        sel = None
        for i in range(r, m):
            if M[i, c]:
                sel = i; break
        if sel is None:
            continue
        M[[r, sel]] = M[[sel, r]]
        for i in range(m):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        piv.append(c); r += 1
        if r == m:
            break
    return M, piv

def gf2_rank(M):
    if len(M) == 0:
        return 0
    _, piv = gf2_rref(np.array(M, dtype=np.int8))
    return len(piv)

def gf2_nullspace(A):
    A = np.array(A, dtype=np.int8) % 2
    R, piv = gf2_rref(A)
    m, n = A.shape
    free = [c for c in range(n) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(n, dtype=np.int8); v[f] = 1
        for i, pc in enumerate(piv):
            if R[i, f]:
                v[pc] = 1
        basis.append(v % 2)
    return basis

def in_span(rows, v):
    rows = [np.array(r, dtype=np.int8) % 2 for r in rows]
    base = gf2_rank(rows)
    return gf2_rank(rows + [np.array(v, dtype=np.int8) % 2]) == base

N = 9
J = np.zeros((2 * N, 2 * N), dtype=np.int8)
J[:N, N:] = np.eye(N); J[N:, :N] = np.eye(N)

def pauli_vec(xs=(), zs=()):
    v = np.zeros(2 * N, dtype=np.int8)
    for q in xs: v[q] = 1
    for q in zs: v[N + q] = 1
    return v

def pstr(v):
    sym = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
    return "".join(sym[(int(v[j]), int(v[N + j]))] for j in range(N))

def weight(v):
    return int(sum((v[j] | v[N + j]) > 0 for j in range(N)))

def symp(a, b):
    return int((a @ (J @ b)) % 2)

# ---- the 8 stabilizer generators (qubits 0..8), from the standard surface-17 set ----
Sgen = [
    pauli_vec(zs=[0, 1, 4, 5]),   # Z1 Z2 Z5 Z6
    pauli_vec(zs=[3, 4, 7, 8]),   # Z4 Z5 Z8 Z9
    pauli_vec(zs=[2, 3]),         # Z3 Z4
    pauli_vec(zs=[5, 6]),         # Z6 Z7
    pauli_vec(xs=[1, 2, 3, 4]),   # X2 X3 X4 X5
    pauli_vec(xs=[4, 5, 6, 7]),   # X5 X6 X7 X8
    pauli_vec(xs=[0, 1]),         # X1 X2
    pauli_vec(xs=[7, 8]),         # X8 X9
]
Smat = np.array(Sgen, dtype=np.int8)

print("=" * 68)
print("distance-3 rotated surface code  [[9,1,3]]  (the 'surface-17' object)")
print("=" * 68)
print("  8 stabilizer generators on 9 data qubits:")
for i, s in enumerate(Sgen):
    print(f"    S{i+1} = {pstr(s)}")

# 1) valid code? all generators must commute
ok = all(symp(a, b) == 0 for a, b in combinations(Sgen, 2))
print(f"\n  all generators mutually commute: {ok}")
k = N - gf2_rank(Sgen)
print(f"  independent generators: {gf2_rank(Sgen)}  ->  logical qubits k = {k}")

# 2) logical operators = commute with every stabilizer, but not in the stabilizer group
A_syn = (Smat @ J) % 2                       # 8 x 18: row i flips if error anticommutes with S_i
null = gf2_nullspace(A_syn)                  # everything that commutes with all stabilizers
cur = [s for s in Sgen]
logicals = []
for v in null:
    if gf2_rank(cur + [v]) > gf2_rank(cur):
        cur.append(v); logicals.append(v)
    if len(logicals) == 2:
        break
XL, ZL = logicals
assert symp(XL, ZL) == 1                     # a valid logical pair anticommutes
print(f"\n  logical operators (commute w/ all stabilizers, outside the group):")
print(f"    L_a = {pstr(XL)}   weight {weight(XL)}")
print(f"    L_b = {pstr(ZL)}   weight {weight(ZL)}   (the two anticommute: {symp(XL,ZL)==1})")

# 3) code distance = min weight of any nontrivial logical (over all stabilizer-equivalent forms)
reps = [XL, ZL, (XL ^ ZL) % 2]
dmin = 99
for rep in reps:
    for combo in range(2 ** len(Sgen)):
        v = rep.copy()
        for b in range(len(Sgen)):
            if (combo >> b) & 1:
                v = v ^ Sgen[b]
        dmin = min(dmin, weight(v % 2))
print(f"\n  code distance d = {dmin}  (min weight of an undetectable logical error)")

# ---- syndrome + lookup decoder ----
def syndrome(e):
    return tuple(int(x) for x in (A_syn @ (e % 2)) % 2)

errs_1q = []
for q in range(N):
    errs_1q.append((q, "X", pauli_vec(xs=[q])))
    errs_1q.append((q, "Z", pauli_vec(zs=[q])))
    errs_1q.append((q, "Y", pauli_vec(xs=[q], zs=[q])))

lookup = {}
for q, kind, v in errs_1q:
    s = syndrome(v)
    if s not in lookup:        # minimal-weight representative (all weight 1 here)
        lookup[s] = v

# 4) correct every single-qubit error
print("\n" + "-" * 68)
print("  correcting ALL single-qubit errors (X, Y, Z on each of 9 qubits):")
by_kind = {"X": [0, 0], "Y": [0, 0], "Z": [0, 0]}   # [success, total]
for q, kind, v in errs_1q:
    s = syndrome(v)
    corr = lookup[s]
    residual = (v ^ corr) % 2
    success = in_span(Sgen, residual)          # residual is a stabilizer => logical intact
    by_kind[kind][1] += 1
    by_kind[kind][0] += success
tot_ok = sum(b[0] for b in by_kind.values()); tot = sum(b[1] for b in by_kind.values())
for kind in ("X", "Y", "Z"):
    print(f"    {kind}-type errors: {by_kind[kind][0]}/{by_kind[kind][1]} corrected")
print(f"  -> {tot_ok}/{tot} single-qubit errors corrected "
      f"(X AND Z AND Y -- last build's 3-qubit code caught only X)")

# 5) the honest limit, exactly as the distance predicts
print("\n" + "-" * 68)
print("  the limit (honest): distance 3 corrects ANY 1 error but not 2.")
# undetectable logical: zero syndrome yet flips the logical qubit
best = min(((rep ^ 0) for rep in reps), key=weight)
bestv = min(
    (rep ^ (np.bitwise_xor.reduce([Sgen[b] for b in range(8) if (c >> b) & 1] + [np.zeros(2*N, np.int8)]) % 2)
     for rep in reps for c in range(256)), key=weight)
print(f"    a weight-{weight(bestv)} logical like {pstr(bestv)} has syndrome "
      f"{syndrome(bestv)} -- looks clean, but silently flips the logical qubit.")
# a weight-2 error the decoder mis-corrects into a logical failure
shown = False
for (q1, k1, v1), (q2, k2, v2) in combinations(errs_1q, 2):
    if q1 == q2:
        continue
    e = (v1 ^ v2) % 2
    corr = lookup.get(syndrome(e))
    if corr is None:
        continue
    residual = (e ^ corr) % 2
    if not in_span(Sgen, residual):
        print(f"    weight-2 error {k1}{q1+1}+{k2}{q2+1}: decoder guesses one qubit, "
              f"residual is a LOGICAL error -> uncorrected.")
        shown = True
        break
if not shown:
    print("    (no weight-2 logical failure found in scan)")
print("\n  Same binary-linear-algebra thread as quantvec and the chip accumulator,")
print("  now realized as the actual code on Google's Willow. Reproduced, not invented.")
