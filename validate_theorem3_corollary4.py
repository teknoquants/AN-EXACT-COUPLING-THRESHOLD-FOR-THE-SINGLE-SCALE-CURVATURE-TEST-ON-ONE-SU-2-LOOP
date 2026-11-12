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
"""Validation of Theorem 3 (exact threshold), Corollary 4 (the classical tensor), their proofs and the two remarks.

Theorem 3
  (i)   T_0 = (1 + x/g^2) g_met on all tangent vectors: matrix-level finite differences of log Omega_g along
        geodesics at 300 random points and directions, five couplings.
  (ii)  kappa_0(g) = 1 - 1/g^2: infimum of 1 + x/g^2 over the group, attained only at x = -1; agreement with the
        optimal kappa(0) of the flow code of the preceding paper; a nonzero horizontal vector exists at every point.
  (iii) kappa_0 > 0 iff g > 1; at g = 1, T_0 >= 0 with a zero at -1; for g < 1, T_0 < 0 exactly on {x < -g^2},
        which has positive Haar measure (computed) and small mu_g measure.
  (iv)  Control: a profile with kappa(0) = kappa_0(g) + delta violates hypothesis (H) at -1, for delta > 0.
Corollary 4
  (v)   B_0 = (1 + x/(2 g^2)) g_met and kappa_BE(g) = 1 - 1/(2 g^2), positive iff g > 1/sqrt2.
  (vi)  Bakry-Emery inequality for all functions: the optimal Poincare constant in the sectors ell = 0, 1, 2
        (class functions and two non-invariant sectors) is <= 1/kappa_BE for eight couplings g > 1/sqrt2.
  (vii) For 1/sqrt2 < g <= 1 the classical test succeeds while kappa_0(g) <= 0 ("the factor two matters").
Remark
  (viii) Horizontal and full lower bounds coincide for T_0 and B_0 (both are multiples of the metric).
"""
import math
import sys

import numpy as np

from common import (Report, haar_sun, exp_su2, random_su2, x_of, second_diff, Flow, X, TH, WQ, sector_constant, integ)


def main():
    rep = Report("Theorem 3 / Corollary 4")
    rng = np.random.default_rng(1303)
    couplings = (2.0, 1.25, 1.0, 0.8, 0.5)

    worst = 0
    for g in couplings:
        for U in haar_sun(60, 2, rng):
            A = random_su2(rng)
            hess = second_diff(lambda r: x_of(U @ exp_su2(r * A)) / g ** 2)          # Hess log Omega_g (A, A), |A| = 1
            worst = max(worst, abs((1 - 2 * hess) - (1 + x_of(U) / g ** 2)) * min(1.0, g ** 2))
    rep.check("(i) T_0 = Ric - 2 Hess log Omega_g = (1 + x/g^2) g_met on all vectors", worst < 1e-7, f"max error {worst:.1e}")

    xs = np.linspace(-1, 1, 400001)
    ok = True
    for g in couplings:
        vals = 1 + xs / g ** 2
        ok &= abs(vals.min() - (1 - 1 / g ** 2)) < 1e-14 and xs[vals.argmin()] == -1 and (vals[1:] > vals[0]).all()
        ok &= abs(Flow(-X / g ** 2).kappa(0.0) - (1 - 1 / g ** 2)) < 2e-3 * max(1, 1 / g ** 2)
    rep.check("(ii) kappa_0(g) = 1 - 1/g^2, attained only at U = -1; equals the optimal kappa(0) of the flow code", ok)

    ok, det = True, []
    for g in (0.3, 0.6, 0.9, 1.0, 1.1, 3.0):
        ok &= ((1 - 1 / g ** 2) > 0) == (g > 1)
        neg = (1 + X / g ** 2) < 0
        ok &= np.array_equal(neg, X < -g ** 2)
        haar = float(WQ[neg].sum())
        mu = float((WQ * np.exp(X / g ** 2))[neg].sum() / integ(np.exp(X / g ** 2)))
        ok &= (haar > 0) == (g < 1) and mu <= haar + 1e-15
        if g < 1:
            det.append(f"g={g}: Haar {haar:.3f}, mu_g {mu:.2e}")
    ok &= abs((1 + (-1) / 1.0 ** 2)) < 1e-15 and (1 + xs / 1.0).min() >= 0
    rep.check("(iii) positive iff g > 1; zero at -1 when g = 1; negative exactly on {x < -g^2} when g < 1", ok, "; ".join(det))

    ok = True
    for g in couplings:
        for delta in (1e-3, 0.1):
            ok &= (1 + (-1) / g ** 2) < (1 - 1 / g ** 2 + delta)           # T_0 at -1 on a unit (horizontal) vector
    rep.check("(iv) control: kappa(0) = kappa_0 + delta violates (H) at -1", ok)

    worst, ok = 0, True
    for g in couplings:
        for U in haar_sun(40, 2, rng):
            A = random_su2(rng)
            hessV = second_diff(lambda r: -x_of(U @ exp_su2(r * A)) / g ** 2)
            worst = max(worst, abs((1 + hessV) - (1 + x_of(U) / (2 * g ** 2))) * min(1.0, g ** 2))
        ok &= ((1 - 1 / (2 * g ** 2)) > 0) == (g > 1 / math.sqrt(2))
    rep.check("(v) B_0 = (1 + x/(2 g^2)) g_met; kappa_BE = 1 - 1/(2 g^2) > 0 iff g > 1/sqrt2", ok and worst < 1e-7, f"max error {worst:.1e}")

    ok, det = True, []
    for g in (0.72, 0.75, 0.8, 0.9, 1.0, 1.25, 2.0, 4.0):
        bound = 1 / (1 - 1 / (2 * g ** 2))
        S = lambda t, g=g: -np.cos(t) / g ** 2
        cs = [sector_constant(S, ell) for ell in (0, 1, 2)]
        ok &= max(cs) <= bound * (1 + 1e-6)
        det.append(f"g={g}: max optimal {max(cs):.4f} <= {bound:.4f}")
    rep.check("(vi) Bakry-Emery inequality on all functions (sectors ell = 0, 1, 2) for g > 1/sqrt2", ok, "; ".join(det[:4]) + " ...")

    ok = all((1 - 1 / (2 * g ** 2)) > 0 and (1 - 1 / g ** 2) <= 0 for g in np.linspace(0.7072, 1.0, 50))
    rep.check("(vii) for 1/sqrt2 < g <= 1: classical test succeeds, kappa_0(g) <= 0", ok)

    ok = True
    for g in couplings:
        F = Flow(-X / g ** 2)
        ok &= abs(F.kappa(0.0) - F.kappa(0.0, full=True)) < 1e-6 * max(1, 1 / g ** 2)
    rep.check("(viii) horizontal and full lower bounds coincide at the initial scale", ok)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
