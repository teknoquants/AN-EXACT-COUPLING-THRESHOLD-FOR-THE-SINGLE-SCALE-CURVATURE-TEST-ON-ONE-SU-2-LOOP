# MIT License
#
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Shared tools: SU(2) class functions in the character basis, the flow of the paper, su(N) algebra, reporting.

Companion code to "A Conditional Multiscale Horizontal Curvature Criterion on a Finite Gauge Lattice"
(R. Cooperman and A.E. Petrunina V.V., XISOSIS, Vienna, 11 November 2026).

Conventions (one SU(2) loop, Definition 8): x = Tr(U)/2 = cos(theta); metric on the radial line 2 dtheta^2;
Haar measure on class functions (2/pi) sin^2(theta) dtheta; for class functions f, V:
    Laplacian f = (f'' + 2 cot(theta) f')/2,   |grad f|^2 = f'^2/2,   <grad V, grad f> = V' f'/2,
    2 Hess V (unit radial) = V'',   2 Hess V (unit tangential) = cot(theta) V',   Ric = 1.
"""
import math

import numpy as np
from scipy.integrate import cumulative_simpson, simpson
from scipy.linalg import eigh

# ---------------------------------------------------------------- grid and character transform
K = 400
k_ = np.arange(1, K + 1)
TH = k_ * np.pi / (K + 1)
X = np.cos(TH)
SN = np.sin(TH)
WQ = 2 * SN ** 2 / (K + 1)                  # Haar quadrature weights (exact for polynomials in x of degree < 2K)
NN = np.arange(K)                           # index n <-> character chi_{n+1}, a polynomial of degree n in x
LAM = NN * (NN + 2) / 2.0                   # -Laplacian eigenvalue of chi_{n+1}
_S = np.sin(np.outer(NN + 1, TH))
_C = np.cos(np.outer(NN + 1, TH))
_SGN = (-1.0) ** NN


def to_coef(f):
    return (2.0 / (K + 1)) * (_S @ (f * SN))


def from_coef(c):
    return (c @ _S) / SN


def heat(f, t):
    """exp(t Laplacian) f for a class function given on the grid."""
    return from_coef(to_coef(f) * np.exp(-t * LAM))


def lap(f):
    return from_coef(-LAM * to_coef(f))


def d1(f):
    """f' (derivative in theta) of a class function."""
    c = to_coef(f)
    return (((c * (NN + 1)) @ _C) * SN - (c @ _S) * np.cos(TH)) / SN ** 2


def d2(f):
    return 2 * lap(f) - 2 * np.cos(TH) / SN * d1(f)


def end_values(f):
    """(f(0), f(pi), f''(0), f''(pi)) from the character expansion (f' vanishes at both ends)."""
    c = to_coef(f)
    f0, fp = np.sum(c * (NN + 1)), np.sum(c * (NN + 1) * _SGN)
    l0, lp = np.sum(-LAM * c * (NN + 1)), np.sum(-LAM * c * (NN + 1) * _SGN)
    return f0, fp, (2.0 / 3.0) * l0, (2.0 / 3.0) * lp      # Laplacian f = (3/2) f'' at the poles


def integ(f):
    return float(WQ @ f)


def grad2(f):
    return d1(f) ** 2 / 2.0


# ---------------------------------------------------------------- the flow of the paper on one loop
class Flow:
    """Renormalization flow for a class action S on one SU(2) loop."""

    def __init__(self, S):
        self.S = S
        self.z0 = np.exp(-S)
        self.Z = integ(self.z0)

    def z(self, t):
        return heat(self.z0, t)

    def V(self, t):
        return -np.log(self.z(t))

    def Vp(self, t):
        z = self.z(t)
        return -d1(z) / z

    def Vpp(self, t):
        z = self.z(t)
        return -d2(z) / z + (d1(z) / z) ** 2

    def Q(self, f, s, t):
        return heat(f * self.z(s), t - s) / self.z(t)

    def L(self, f, t):
        return lap(f) - self.Vp(t) * d1(f)

    def mu(self, f, t=0.0):
        return integ(f * self.z(t)) / self.Z

    def kappa(self, t, full=False):
        """Optimal lower bound in (H): 1 + inf V_t'' (poles included). full=True also uses tangential directions."""
        z = self.z(t)
        vpp = self.Vpp(t)
        z0, zp, z0pp, zppp = end_values(z)
        m = min(vpp.min(), -z0pp / z0, -zppp / zp)
        if full:
            m = min(m, (np.cos(TH) / SN * self.Vp(t)).min())
        return 1.0 + m

    def constant(self, full=False, T=30.0, nt=901):
        """C_kappa = 2 int_0^infty exp(-2 Lambda(t)) dt, with Simpson's rule in log t (plus the pieces near 0 and infinity)."""
        t0 = 1e-7
        u = np.linspace(math.log(t0), math.log(T), nt)
        tl = np.exp(u)
        kl = np.array([self.kappa(t, full) for t in tl])
        k0 = self.kappa(0.0, full)
        Laml = k0 * t0 + cumulative_simpson(kl * tl, x=u, initial=0.0)
        g = np.exp(-2 * Laml)
        C = 2 * (t0 + simpson(g * tl, x=u) + g[-1] / (2 * kl[-1]))
        ts = np.concatenate([[0.0], tl])
        return C, ts, np.concatenate([[k0], kl]), np.concatenate([[0.0], Laml])


def optimal_constant(S, deg=120):
    """Reciprocal of the smallest positive eigenvalue of the Dirichlet form on class functions (numerical)."""
    w = np.exp(-S) * WQ
    w = w / w.sum()
    U = _S[:deg + 1] / SN
    dU = ((NN[:deg + 1, None] + 1) * _C[:deg + 1] * SN - _S[:deg + 1] * np.cos(TH)) / SN ** 2
    Kmat = (dU * w) @ dU.T / 2.0
    M = (U * w) @ U.T
    m = U @ w
    Mc = M - np.outer(m, m)
    ev = eigh(Kmat[1:, 1:], Mc[1:, 1:], eigvals_only=True)
    return 1.0 / ev[0]


def random_class_function(rng, deg=6, scale=1.0):
    c = scale * rng.normal(size=deg + 1) / (1 + np.arange(deg + 1)) ** 1.5
    return sum(ci * X ** i for i, ci in enumerate(c))


ACTIONS = {
    "free": lambda: 0 * X,
    "1+0.6x": lambda: -np.log(1 + 0.6 * X),
    "1-0.9x": lambda: -np.log(1 - 0.9 * X),
    "Wilson beta=1": lambda: -2 * 1.0 * X,
    "Wilson beta=2": lambda: -2 * 2.0 * X,
    "double well 2x^2": lambda: 2.0 * X ** 2,
    "cubic": lambda: 0.8 * X - 1.2 * X ** 2 + 0.5 * X ** 3,
}


# ---------------------------------------------------------------- su(N), SU(N)
def su_basis(N):
    Hs = []
    for j in range(N):
        for k in range(j + 1, N):
            A = np.zeros((N, N), complex); A[j, k] = A[k, j] = 1 / math.sqrt(2); Hs.append(A)
            B = np.zeros((N, N), complex); B[j, k] = -1j / math.sqrt(2); B[k, j] = 1j / math.sqrt(2); Hs.append(B)
    for m in range(1, N):
        D = np.zeros((N, N), complex)
        D[:m, :m] = np.eye(m); D[m, m] = -m
        Hs.append(D / math.sqrt(m * m + m))
    return [1j * H for H in Hs]


def ip(Xm, Ym):
    return float(np.real(-np.trace(Xm @ Ym)))


def haar_sun(n, N, rng):
    Xr = (rng.normal(size=(n, N, N)) + 1j * rng.normal(size=(n, N, N))) / np.sqrt(2)
    Q, R = np.linalg.qr(Xr)
    d = np.diagonal(R, axis1=1, axis2=2)
    Q = Q * (d / np.abs(d))[:, None, :]
    Q[:, :, 0] /= np.linalg.det(Q)[:, None]
    return Q


def expm_skew(Xm):
    """exp of an anti-Hermitian matrix."""
    w, v = np.linalg.eigh(1j * Xm)
    return (v * np.exp(-1j * w)) @ v.conj().T


# ---------------------------------------------------------------- reporting
class Report:
    def __init__(self, name):
        self.name = name
        self.rows = []

    def check(self, label, ok, detail=""):
        self.rows.append((label, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
        return ok

    def summary(self):
        p = sum(r[1] for r in self.rows)
        print(f"  -> {self.name}: {p}/{len(self.rows)} checks passed")
        return p == len(self.rows)


# ---------------------------------------------------------------- additions for the threshold paper
def exp_su2(A):
    """exp of A in su(2): cos(|A|/sqrt2) 1 + sqrt2 sin(|A|/sqrt2) A/|A|, with |A|^2 = -Tr(A^2)."""
    n = math.sqrt(max(ip(A, A), 0.0))
    if n < 1e-300:
        return np.eye(2, dtype=complex)
    return math.cos(n / math.sqrt(2)) * np.eye(2) + math.sqrt(2) * math.sin(n / math.sqrt(2)) * A / n


def random_su2(rng, unit=True):
    B = su_basis(2)
    c = rng.normal(size=3)
    if unit:
        c /= np.linalg.norm(c)
    return sum(ci * T for ci, T in zip(c, B))


def x_of(U):
    return float(np.real(np.trace(U)) / 2)


def second_diff(f, h=1e-3):
    """Fourth-order central second difference of a scalar function at 0."""
    return (-f(2 * h) + 16 * f(h) - 30 * f(0.0) + 16 * f(-h) - f(-2 * h)) / (12 * h * h)


def sector_constant(S_of_theta, ell, nth=2000):
    """Optimal Poincare constant of exp(-S) dHaar on SU(2) restricted to functions f(theta) Y_ell (class weight S).

    ell = 0 gives class functions (constants removed); ell >= 1 the non-invariant sectors. Finite differences."""
    t = np.linspace(0, np.pi, nth)[1:-1]
    d = t[1] - t[0]
    tm = 0.5 * (t[1:] + t[:-1])
    w = np.sin(t) ** 2 * np.exp(-S_of_theta(t))
    wm = np.sin(tm) ** 2 * np.exp(-S_of_theta(tm))
    n = len(t)
    Km = np.zeros((n, n))
    idx = np.arange(n - 1)
    c = 0.5 * wm / d
    Km[idx, idx] += c; Km[idx + 1, idx + 1] += c; Km[idx, idx + 1] -= c; Km[idx + 1, idx] -= c
    Km += np.diag(ell * (ell + 1) / (2 * np.sin(t) ** 2) * w * d)
    ev = eigh(Km, np.diag(w * d), eigvals_only=True, subset_by_index=[0, 1])
    return 1.0 / (ev[1] if ell == 0 else ev[0])


def cubical_box(dims, periodic=False):
    """Open (or periodic) cubical complex with dims[k] plaquette-widths in direction k.

    Returns (n_vertices, edges, plaquettes, d0, d1): d0 is E x V, d1 is P x E (signs of the oriented boundary)."""
    import itertools
    D = len(dims)
    if periodic:
        verts = list(itertools.product(*[range(n) for n in dims]))
        step = lambda v, k: tuple((v[i] + (1 if i == k else 0)) % dims[i] for i in range(D))
        ok = lambda v: True
    else:
        verts = list(itertools.product(*[range(n + 1) for n in dims]))
        step = lambda v, k: tuple(v[i] + (1 if i == k else 0) for i in range(D))
        ok = lambda v: all(0 <= v[i] <= dims[i] for i in range(D))
    vi = {v: i for i, v in enumerate(verts)}
    edges = [(v, k) for v in verts for k in range(D) if ok(step(v, k))]
    ei = {e: i for i, e in enumerate(edges)}
    plaqs = []
    for v in verts:
        for i in range(D):
            for j in range(i + 1, D):
                if ok(step(v, i)) and ok(step(v, j)) and ok(step(step(v, i), j)):
                    plaqs.append([(ei[(v, i)], 1), (ei[(step(v, i), j)], 1), (ei[(step(v, j), i)], -1), (ei[(v, j)], -1)])
    d0 = np.zeros((len(edges), len(verts)))
    for (v, k), e in ei.items():
        d0[e, vi[v]] += 1; d0[e, vi[step(v, k)]] -= 1
    d1 = np.zeros((len(plaqs), len(edges)))
    for p, bd in enumerate(plaqs):
        for e, o in bd:
            d1[p, e] += o
    return len(verts), edges, plaqs, d0, d1


def solve_gf2(Amat, b):
    """One solution of A x = b over GF(2), or None."""
    A = (np.array(Amat, dtype=np.int64) % 2).copy()
    b = (np.array(b, dtype=np.int64) % 2).copy()
    rows, cols = A.shape
    piv, r = [], 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if A[i, c]), None)
        if p is None:
            continue
        A[[r, p]] = A[[p, r]]; b[[r, p]] = b[[p, r]]
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] ^= A[r]; b[i] ^= b[r]
        piv.append(c); r += 1
        if r == rows:
            break
    if b[r:].any():
        return None
    xsol = np.zeros(cols, dtype=np.int64)
    for k, c in enumerate(piv):
        xsol[c] = b[k]
    return xsol


def plaquette_log_weight(U, plaqs, g):
    tot = 0.0
    for bd in plaqs:
        M = np.eye(2, dtype=complex)
        for e, o in bd:
            M = M @ (U[e] if o > 0 else U[e].conj().T)
        tot += x_of(M)
    return tot / g ** 2
