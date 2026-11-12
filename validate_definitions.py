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
"""Validation of Definitions 1-5.

What is checked
  D1  Orthonormal basis of su(2) for -Tr(XY); Ric = g_met, i.e. (1/4) sum_a |[X,T_a]|^2 = |X|^2.
  D2  Conjugation orbits: at U = +-1 the orbit is a point (zero-dimensional tangent space, every vector horizontal);
      at regular U the orbit tangent space has dimension 2 and the horizontal space is the radial line.
  D3  -1 <= x <= 1; Z_g = 2 g^2 I_1(1/g^2) (modified Bessel function); mu_g is a probability measure;
      the Wilson parameter of the preceding paper is beta = 1/(2 g^2).
  D4  T_0 and B_0 on one loop: in the radial calculus of the preceding paper, 2 Hess V_0 = (x/g^2) on radial and
      tangential unit vectors alike; kappa_0 agrees with the optimal kappa(0) computed by the flow code of [1].
  D5  The constant C_kappa of the multiscale criterion is infinite when kappa <= 0 identically and equals 1/kappa
      for constant kappa > 0.
"""
import math
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import iv

from common import Report, su_basis, ip, haar_sun, Flow, X, TH, SN, WQ, integ, d1, d2


def main():
    rep = Report("Definitions")
    rng = np.random.default_rng(1201)
    B = su_basis(2)
    G = np.array([[ip(a, b) for b in B] for a in B])
    ok = np.allclose(G, np.eye(3))
    for _ in range(20):
        Xm = sum(c * T for c, T in zip(rng.normal(size=3), B))
        ok &= abs(0.25 * sum(ip(Xm @ T - T @ Xm, Xm @ T - T @ Xm) for T in B) - ip(Xm, Xm)) < 1e-12
    rep.check("D1 orthonormal basis; Ric = g_met on su(2)", ok)

    def orbit_rank(U):
        vecs = [T - U.conj().T @ T @ U for T in B]                 # d/ds of exp(sT) U exp(-sT), left-trivialized at U
        M = np.array([[ip(v, T) for T in B] for v in vecs])
        return np.linalg.matrix_rank(M, tol=1e-9), M
    r_plus, _ = orbit_rank(np.eye(2, dtype=complex))
    r_minus, _ = orbit_rank(-np.eye(2, dtype=complex))
    ok = r_plus == 0 and r_minus == 0
    for U in haar_sun(30, 2, rng):
        r, M = orbit_rank(U)
        ok &= r == 2
        # the radial direction (the Lie algebra element commuting with U) is orthogonal to the orbit
        Xr = (U - U.conj().T) / 2
        ok &= np.abs(M @ np.array([ip(Xr, T) for T in B])).max() < 1e-12
    rep.check("D2 orbit of +-1 is a point; regular orbits are 2-dimensional with radial horizontal line", ok)

    ok, det = True, []
    for g in (2.0, 1.0, 0.6):
        a = 1 / g ** 2
        Z = integ(np.exp(X * a))
        ok &= abs(Z - 2 * iv(1, a) / a) / Z < 1e-12
        F = Flow(-X * a)
        ok &= abs(F.mu(1 + 0 * X) - 1) < 1e-13
        det.append(f"g={g}: Z_g={Z:.6f}")
    ok &= X.min() >= -1 and X.max() <= 1
    ok &= all(abs(-(1 / (2 * g ** 2)) * 2 * xx + xx / g ** 2) < 1e-15 for g in (0.5, 1.3) for xx in (-1, 0.3))   # -beta Tr U = -x/g^2
    rep.check("D3 Z_g = 2 g^2 I_1(1/g^2); mu_g probability; beta = 1/(2 g^2)", ok, "; ".join(det))

    ok, worst = True, 0
    for g in (2.0, 1.0, 0.8, 0.5):
        V0 = -X / g ** 2
        rad, tan = d2(V0), np.cos(TH) / SN * d1(V0)
        worst = max(worst, np.abs(rad - X / g ** 2).max(), np.abs(tan - X / g ** 2).max())
        ok &= abs(Flow(V0).kappa(0.0) - (1 - 1 / g ** 2)) < 2e-3 * max(1, 1 / g ** 2)
    rep.check("D4 2 Hess V_0 = x/g^2 in radial and tangential directions; kappa_0 = optimal kappa(0) of the flow code", ok and worst < 1e-6,
              f"max deviation {worst:.1e}")

    C = lambda kap, T: 2 * quad(lambda t: math.exp(-2 * kap * t), 0, T)[0]
    ok = abs(C(0.7, np.inf) - 1 / 0.7) < 1e-10 and abs(C(0.0, 50) - 100) < 1e-9 and C(-0.1, 50) > C(0.0, 50)
    rep.check("D5 C_kappa = 1/kappa for constant kappa > 0; unbounded for kappa <= 0", ok)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
