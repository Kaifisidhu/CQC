"""
shapes.py -- your two ideas, built and filtered honestly.

You said two things:
  (1) "make your own gates / a new type of code"
  (2) "make a tensor in a non-square shape -- circle, pentagram, hexagram"

Here is the honest version of both, on the same binary machinery as surface.py.

IDEA 1 is real with a catch: ANY set of commuting Pauli generators IS a valid
stabilizer code, and ANY 2x2 unitary IS a valid gate. You can absolutely invent
them. The catch is that almost all of them are WORSE than known ones -- so the
useful thing is a machine that tells you, immediately and honestly, whether your
invention is any good. That is what this file is.

IDEA 2 has a category error wrapped around a correct instinct. A NumPy array is
indexed by an integer grid -- there is no "circular ndarray", that is just not
what an array is. BUT the shape that actually matters in this field is not the
array's shape, it is the shape of the LATTICE the code lives on:
    square lattice   -> surface code   (what we built last time)
    hexagonal lattice -> COLOR codes    (the smallest is the Steane [[7,1,3]] code)
You aimed "hexagram shape code" almost exactly at a real, named, active object.
(Web-confirmed: the Steane code is a 2D color code on a 7-qubit patch of the
6.6.6 hexagonal tiling; color codes are Bombin-Martin-Delgado 2006.)
And "pentagram": 5-fold symmetry CANNOT tile the plane periodically (Penrose /
quasicrystals), so there is no periodic pentagonal code the way there is a
hexagonal one. The wall is itself the interesting true fact.
"""
import numpy as np
from itertools import combinations

# ---------- GF(2) machinery ----------
def gf2_rref(M):
    M = (np.array(M, dtype=np.int8) % 2).copy()
    m, n = M.shape; piv = []; r = 0
    for c in range(n):
        sel = next((i for i in range(r, m) if M[i, c]), None)
        if sel is None: continue
        M[[r, sel]] = M[[sel, r]]
        for i in range(m):
            if i != r and M[i, c]: M[i] ^= M[r]
        piv.append(c); r += 1
        if r == m: break
    return M, piv

def gf2_rank(M):
    M = np.array(M, dtype=np.int8)
    if M.size == 0: return 0
    return len(gf2_rref(M)[1])

def gf2_nullspace(A):
    A = np.array(A, dtype=np.int8) % 2
    R, piv = gf2_rref(A); m, n = A.shape
    free = [c for c in range(n) if c not in piv]; basis = []
    for f in free:
        v = np.zeros(n, np.int8); v[f] = 1
        for i, pc in enumerate(piv):
            if R[i, f]: v[pc] = 1
        basis.append(v % 2)
    return basis

def in_span(rows, v):
    rows = [np.array(r, np.int8) % 2 for r in rows]
    if len(rows) == 0: return not np.any(v % 2)
    return gf2_rank(rows + [np.array(v, np.int8) % 2]) == gf2_rank(rows)

def J(n):
    M = np.zeros((2 * n, 2 * n), np.int8); M[:n, n:] = np.eye(n); M[n:, :n] = np.eye(n)
    return M

def symp(a, b, n): return int((a @ (J(n) @ b)) % 2)
def weight(v, n):  return int(sum((v[j] | v[n + j]) > 0 for j in range(n)))
def pstr(v, n):
    s = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
    return "".join(s[(int(v[j]), int(v[n + j]))] for j in range(n))
def xz(n, xs=(), zs=()):
    v = np.zeros(2 * n, np.int8)
    for q in xs: v[q] = 1
    for q in zs: v[n + q] = 1
    return v

# ---------- the quality filter: hand it a code, it tells you if it's any good ----------
def code_report(name, gens, n):
    print(f"\n[{name}]  n={n} data qubits, {len(gens)} generators")
    valid = all(symp(a, b, n) == 0 for a, b in combinations(gens, 2))
    print(f"  valid code (all generators commute)? {valid}")
    if not valid:
        print("  -> NOT a code. (You invented something; the math rejected it.)")
        return
    k = n - gf2_rank(gens)
    A = (np.array(gens) @ J(n)) % 2
    null = gf2_nullspace(A)
    # distance = min weight of a nonzero normalizer element outside the stabilizer group
    d = 10 ** 9
    if len(null) <= 16:
        for mask in range(1, 2 ** len(null)):
            v = np.zeros(2 * n, np.int8)
            for i in range(len(null)):
                if (mask >> i) & 1: v ^= null[i]
            v %= 2
            if not in_span(gens, v):
                d = min(d, weight(v, n))
    print(f"  encodes k={k} logical qubit(s);  code distance d={d}   ->  [[{n},{k},{d}]]")
    # single-qubit error correction
    errs = []
    for q in range(n):
        errs += [(q, "X", xz(n, xs=[q])), (q, "Z", xz(n, zs=[q])), (q, "Y", xz(n, xs=[q], zs=[q]))]
    look = {}
    for q, kind, v in errs:
        s = tuple(int(x) for x in (A @ v) % 2)
        if s not in look: look[s] = v
    ok = 0
    for q, kind, v in errs:
        s = tuple(int(x) for x in (A @ v) % 2)
        if in_span(gens, (v ^ look[s]) % 2): ok += 1
    print(f"  corrects {ok}/{len(errs)} single-qubit errors (X, Y, Z on every qubit)")

print("=" * 70)
print("INVENT YOUR OWN CODE  --  with an honest quality filter")
print("=" * 70)

# the surface code (square lattice) -- last build, re-checked through the analyzer
surf = [xz(9, zs=[0,1,4,5]), xz(9, zs=[3,4,7,8]), xz(9, zs=[2,3]), xz(9, zs=[5,6]),
        xz(9, xs=[1,2,3,4]), xz(9, xs=[4,5,6,7]), xz(9, xs=[0,1]), xz(9, xs=[7,8])]
code_report("surface-17  (SQUARE lattice)", surf, 9)

# the Steane code = 2D color code on a HEXAGONAL patch (your 'hexagram' instinct, made real)
# X- and Z-stabilizers from the [7,4,3] Hamming parity check
H = [[3,4,5,6], [1,2,5,6], [0,2,4,6]]          # 0-indexed qubit supports
steane = [xz(7, xs=s) for s in H] + [xz(7, zs=s) for s in H]
code_report("Steane color code  (HEXAGONAL lattice)", steane, 7)

# a code you might invent first -- and why it disappointed us two builds ago
bitflip = [xz(3, zs=[0,1]), xz(3, zs=[1,2])]    # the 3-qubit "bit-flip" code
code_report("3-qubit code you'd invent first", bitflip, 3)
print("  ^ distance 1: a single Z is invisible. THIS is exactly why last week's")
print("    3-qubit code caught only X. The analyzer shows the reason as a number.")

# an invalid invention -- the filter catches it
bad = [xz(2, xs=[0,1]), xz(2, zs=[0])]          # X0X1 and Z0 do not commute
code_report("a broken invention", bad, 2)

# ---------- INVENT YOUR OWN GATE -- classify it Clifford (free) vs magic (hard) ----------
print("\n" + "=" * 70)
print("INVENT YOUR OWN GATE  --  is it free, or is it magic?")
print("=" * 70)
I2 = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], complex); Yp = np.array([[0,-1j],[1j,0]], complex)
Z = np.array([[1,0],[0,-1]], complex)
def is_clifford(U):
    for P in (X, Yp, Z):
        C = U @ P @ U.conj().T
        if not any(np.allclose(C, ph * Q, atol=1e-9)
                   for Q in (I2, X, Yp, Z) for ph in (1, -1, 1j, -1j)):
            return False
    return True
def phase(theta): return np.array([[1, 0], [0, np.exp(1j * theta)]], complex)
Hd = np.array([[1,1],[1,-1]], complex) / np.sqrt(2)

gates = [("H", Hd), ("S = phase(pi/2)", phase(np.pi/2)), ("Z = phase(pi)", phase(np.pi)),
         ("T = phase(pi/4)", phase(np.pi/4)), ("phase(pi/8)", phase(np.pi/8)),
         ("phase(0.3) custom", phase(0.3))]
for name, U in gates:
    c = is_clifford(U)
    tag = "Clifford  -> FREE (stab.py scales to thousands of qubits)" if c \
          else "non-Clifford -> MAGIC (cost ~2^t, the hard resource)"
    print(f"  {name:20s}: {tag}")
print("\n  You can write down any gate. The math sorts it into 'free' or 'magic'.")
print("  And a deeper truth: only dyadic angles (pi/2^k) sit in the Clifford")
print("  hierarchy and can be done EXACTLY fault-tolerantly. phase(0.3) cannot be")
print("  -- it can only be approximated (Solovay-Kitaev). Inventing a gate is easy;")
print("  inventing a USEFUL, implementable one is the constraint that bites.")
