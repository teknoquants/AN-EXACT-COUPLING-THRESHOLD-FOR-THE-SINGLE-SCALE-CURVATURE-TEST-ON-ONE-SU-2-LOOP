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
"""Validation of Proposition 8 (one-sided threshold on a finite lattice) and every step of its proof.

Lattices: open boxes 1x1, 2x1, 2x2, 3x3 (d = 2) and 1x1x1, 2x1x1, 2x2x1 (d = 3), gauge group SU(2).
What is checked
  (a) Hypothesis (Z): a Z_2 edge labelling eta with every plaquette sum odd exists on each open box (solved over
      GF(2)); the configuration U0_e = (-1)^eta_e 1 has every plaquette holonomy equal to -1.
      Control: (Z) FAILS on the periodic 3x3 torus (nine plaquettes, every edge in two of them), so it is a real
      hypothesis.
  (b) For W in su(2), (1/2) Tr exp(W) = cos(|W|/sqrt2); along the geodesic s -> (exp(s A_e) U0_e) the plaquette
      value is x(U_p(s)) = -1 + (s^2/4)|(d_1 A)_p|^2 + O(s^3) (the Baker-Campbell-Hausdorff step).
  (c) Hessian at U0: Hess log Omega_g(A,A) = (1/(2 g^2)) sum_p |(d_1 A)_p|^2 and
      T_0(A,A) = |A|^2 - g^-2 sum_p |(d_1 A)_p|^2  (finite differences, random directions, g = 0.7, 1, 2).
  (d) Gauge orbit at U0: the tangent of g.U0 is d_0 X (finite differences); d_1 d_0 = 0.
  (e) An eigenvector a of d_1^T d_1 for lambda_max is orthogonal to the image of d_0 (horizontal), and A = a (x) X_0
      gives T_0(A,A) = 1 - lambda_max/g^2 (finite differences). Hence kappa_0(g) <= 1 - lambda_max/g^2, and the test
      fails for g^2 <= lambda_max (checked at g^2 = lambda_max and below).
  (f) 4 <= lambda_max <= 8(d-1): diagonal of d_1 d_1^T equals 4, absolute row sums of d_1^T d_1 are <= 8(d-1).
  (g) Values quoted in the paper for 2D boxes: 4, 5, 6, 6.83, 7.24 (4x4) and 7.84 (10x10).
  (h) One loop (d_1 = (1)): lambda_max = 1, reproducing the threshold g <= 1 of Theorem 3.
"""
import math
import sys

import numpy as np

from common import (Report, su_basis, ip, exp_su2, random_su2, x_of, second_diff, cubical_box, solve_gf2,
                    plaquette_log_weight, expm_skew)

BOXES = [(1, 1), (2, 1), (2, 2), (3, 3), (1, 1, 1), (2, 1, 1), (2, 2, 1)]


def main():
    rep = Report("Proposition 8")
    rng = np.random.default_rng(1808)
    B = su_basis(2)

    data = {}
    ok = True
    for dims in BOXES:
        nv, edges, plaqs, d0, d1 = cubical_box(dims)
        eta = solve_gf2(np.abs(d1), np.ones(len(plaqs)))
        ok &= eta is not None
        U0 = [(-1) ** int(e) * np.eye(2, dtype=complex) for e in eta]
        ok &= abs(plaquette_log_weight(U0, plaqs, 1.0) + len(plaqs)) < 1e-12
        data[dims] = (nv, edges, plaqs, d0, d1, U0)
    _, _, _, _, d1t = cubical_box((3, 3), periodic=True)
    fails_on_torus = solve_gf2(np.abs(d1t), np.ones(d1t.shape[0])) is None
    rep.check("(a) (Z) holds on open boxes (all plaquettes = -1); control: (Z) fails on the 3x3 torus", ok and fails_on_torus,
              f"{len(BOXES)} boxes; torus solvable: {not fails_on_torus}")

    ok = True
    for _ in range(100):
        W = random_su2(rng, unit=False) * rng.uniform(0.1, 3)
        ok &= abs(x_of(expm_skew(W)) - math.cos(math.sqrt(ip(W, W)) / math.sqrt(2))) < 1e-12
    worst = 0
    nv, edges, plaqs, d0, d1, U0 = data[(2, 2)]
    for _ in range(10):
        A = [random_su2(rng, unit=False) for _ in edges]
        for p, bd in enumerate(plaqs):
            Y = sum(o * A[e] for e, o in bd)
            def xp(s):
                M = np.eye(2, dtype=complex)
                for e, o in bd:
                    Ue = exp_su2(s * A[e]) @ U0[e]
                    M = M @ (Ue if o > 0 else Ue.conj().T)
                return x_of(M)
            for s in (1e-2, 2e-2):
                worst = max(worst, abs(xp(s) - (-1 + s * s / 4 * ip(Y, Y))) / s ** 3)
    rep.check("(b) (1/2)Tr exp(W) = cos(|W|/sqrt2); x(U_p(s)) = -1 + (s^2/4)|(d_1A)_p|^2 + O(s^3)", ok and worst < 50,
              f"max |remainder|/s^3 = {worst:.2f}")

    worst_c = worst_e = worst_d = 0
    ok_d = ok_e = ok_f = True
    lam = {}
    for dims in BOXES:
        nv, edges, plaqs, d0, d1, U0 = data[dims]
        D = len(dims)
        for g in (0.7, 1.0, 2.0):
            for _ in range(3):
                A = [random_su2(rng, unit=False) for _ in edges]
                f = lambda s: plaquette_log_weight([exp_su2(s * A[e]) @ U0[e] for e in range(len(edges))], plaqs, g)
                hess = second_diff(f)
                curl2 = sum(ip(Y, Y) for Y in (sum(o * A[e] for e, o in bd) for bd in plaqs))
                norm2 = sum(ip(a, a) for a in A)
                worst_c = max(worst_c, abs(hess - curl2 / (2 * g ** 2)) / max(1, curl2 / g ** 2),
                              abs((norm2 - 2 * hess) - (norm2 - curl2 / g ** 2)) / max(1, curl2 / g ** 2))
        # (d) gauge orbit tangent = d0 X
        ok_d &= np.abs(d1 @ d0).max() == 0
        Xv = [random_su2(rng, unit=False) for _ in range(nv)]
        h = 1e-6
        verts_of = lambda e: (int(np.argmax(d0[e] > 0)), int(np.argmax(d0[e] < 0)))
        for e in range(len(edges)):
            sv, tv = verts_of(e)
            Ug = exp_su2(h * Xv[sv]) @ U0[e] @ exp_su2(-h * Xv[tv])
            tangent = (Ug @ U0[e].conj().T - np.eye(2)) / h
            worst_d = max(worst_d, np.abs(tangent - (Xv[sv] - Xv[tv])).max())
        # (e) top eigenvector
        w, v = np.linalg.eigh(d1.T @ d1)
        lmax, a = w[-1], v[:, -1]
        lam[dims] = lmax
        ok_e &= np.abs(d0.T @ a).max() < 1e-10 and abs(np.linalg.norm(a) - 1) < 1e-12
        X0 = random_su2(rng)
        for g in (math.sqrt(lmax), 0.8 * math.sqrt(lmax), 1.5 * math.sqrt(lmax)):
            A = [a[e] * X0 for e in range(len(edges))]
            f = lambda s: plaquette_log_weight([exp_su2(s * A[e]) @ U0[e] for e in range(len(edges))], plaqs, g)
            T0 = 1 - 2 * second_diff(f)
            worst_e = max(worst_e, abs(T0 - (1 - lmax / g ** 2)))
            ok_e &= (T0 <= 1e-7) == (g ** 2 <= lmax * (1 + 1e-9))
        # (f)
        ok_f &= np.allclose(np.diag(d1 @ d1.T), 4) and np.abs(d1.T @ d1).sum(axis=1).max() <= 8 * (D - 1) + 1e-12
        ok_f &= 4 - 1e-12 <= lmax <= 8 * (D - 1) + 1e-12
    rep.check("(c) Hess log Omega_g and T_0 at U0 (finite differences)", worst_c < 1e-6, f"max rel error {worst_c:.1e}")
    rep.check("(d) orbit tangent at U0 is d_0 X; d_1 d_0 = 0", ok_d and worst_d < 1e-5, f"max error {worst_d:.1e}")
    rep.check("(e) top eigenvector is horizontal and T_0(A,A) = 1 - lambda_max/g^2; test fails iff g^2 <= lambda_max there",
              ok_e and worst_e < 1e-6, f"max error {worst_e:.1e}")
    rep.check("(f) 4 <= lambda_max <= 8(d-1)", ok_f, "; ".join(f"{'x'.join(map(str, k))}: {v:.3f}" for k, v in lam.items()))

    quoted = {(1, 1): 4, (2, 1): 5, (2, 2): 6, (3, 3): 6.83, (4, 4): 7.24, (10, 10): 7.84}
    ok = True
    for dims, val in quoted.items():
        d1 = cubical_box(dims)[4]
        ok &= abs(np.linalg.eigvalsh(d1 @ d1.T).max() - val) < 5e-3
    rep.check("(g) quoted values 4, 5, 6, 6.83, 7.24, 7.84 for 2D boxes", ok)
    rep.check("(h) one loop: d_1 = (1), lambda_max = 1, threshold g <= 1", float(np.linalg.eigvalsh(np.array([[1.0]])).max()) == 1.0)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
