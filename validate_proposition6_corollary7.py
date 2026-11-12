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
"""Validation of Proposition 6 (the multiscale criterion succeeds for every g), Corollary 7 (two meanings of
"necessary"), and their proofs.

Proposition 6 (couplings g = 2, 1, 0.8, 0.6, 0.5)
  (a) z_t = exp(t Laplacian) Omega_g is strictly positive and converges to Z_g in C^2: sup|z_t - Z_g|, sup|z_t'| and
      sup|z_t''| decay at least like exp(-3t/2) (the first non-constant eigenvalue);
  (b) the optimal kappa_g(t) is continuous (increments shrink with the step in t), kappa_g(0) = 1 - 1/g^2, and
      kappa_g(t) -> 1;
  (c) C_kappa is finite and stable under refinement of the scale grid and extension of the cut-off.
Corollary 7 (g <= 1)
  (d) kappa_g(0) <= 0; with a profile that is <= 0 on [0, T] the partial constant is >= 2T (it diverges with T);
  (e) the actual profile becomes positive: the set {t : kappa_g(t) > 0} has positive measure, and the first sign
      change is reported; for g = 1 the profile is positive for every t > 0;
  (f) absolute necessity is false: for each g <= 1 the optimal Poincare constant is finite and
      <= (2/3) e^{2/g^2} (all sectors ell = 0, 1, 2), a bound obtained without any multiscale analysis;
  (g) relative necessity: the multiscale constant is finite although kappa_g(0) <= 0, and it bounds the optimal
      constant on gauge-invariant functions.
"""
import math
import sys

import numpy as np

from common import Report, Flow, X, d1, d2, integ, optimal_constant, sector_constant


def main():
    rep = Report("Proposition 6 / Corollary 7")
    couplings = (2.0, 1.0, 0.8, 0.6, 0.5)

    ok, det = True, []
    for g in couplings:
        F = Flow(-X / g ** 2)
        Z = F.Z
        t1, t2 = 4.0, 6.0
        for fn in (lambda z: np.abs(z - Z).max(), lambda z: np.abs(d1(z)).max(), lambda z: np.abs(d2(z)).max()):
            a, b = fn(F.z(t1)), fn(F.z(t2))
            rate = math.log(a / b) / (t2 - t1)
            ok &= rate > 1.49 and b < 1e-3 * Z
        ok &= all(F.z(t).min() > 0 for t in (0.0, 0.01, 0.3, 2.0, 10.0))
        det.append(f"g={g}: rate {rate:.3f}")
    rep.check("(a) z_t > 0 and z_t -> Z_g in C^2 at rate >= 3/2", ok, "; ".join(det))

    ok, det = True, []
    for g in couplings:
        F = Flow(-X / g ** 2)
        k = lambda t: F.kappa(t)
        ok &= abs(k(0.0) - (1 - 1 / g ** 2)) < 2e-3 * max(1, 1 / g ** 2)
        ok &= abs(k(12.0) - 1) < 1e-4 and abs(k(20.0) - 1) < 1e-6
        for t0 in (0.0, 0.05, 0.4, 1.5):
            incs = [abs(k(t0 + h) - k(t0)) for h in (1e-2, 1e-3, 1e-4)]
            ok &= incs[2] <= incs[1] + 1e-6 <= incs[0] + 2e-6 and incs[2] < 5e-2 * max(1, 1 / g ** 4)
        det.append(f"g={g}: kappa(0)={k(0.0):+.3f}, kappa(12)={k(12.0):.5f}")
    rep.check("(b) kappa_g continuous, kappa_g(0) = 1 - 1/g^2, kappa_g(t) -> 1", ok, "; ".join(det))

    ok, det = True, []
    for g in couplings:
        F = Flow(-X / g ** 2)
        C1, C2, C3 = F.constant()[0], F.constant(nt=1801)[0], F.constant(T=45.0, nt=1201)[0]
        ok &= math.isfinite(C1) and abs(C1 - C2) / C1 < 1e-4 and abs(C1 - C3) / C1 < 1e-4
        det.append(f"g={g}: C={C1:.5g}")
    rep.check("(c) C_kappa finite and stable under refinement", ok, "; ".join(det))

    ok = all((1 - 1 / g ** 2) <= 0 for g in (1.0, 0.8, 0.6, 0.5))
    for T in (5.0, 50.0, 500.0):
        ts = np.linspace(0, T, 20001)
        for kap in (0.0, -0.2):
            part = 2 * np.trapezoid(np.exp(-2 * kap * ts), ts)
            ok &= part >= 2 * T * (1 - 1e-9)
    rep.check("(d) g <= 1: kappa(0) <= 0; a profile <= 0 on [0,T] gives a partial constant >= 2T", ok)

    ok, det = True, []
    for g in (1.0, 0.8, 0.6, 0.5):
        F = Flow(-X / g ** 2)
        _, ts, kap, _ = F.constant()
        pos = kap > 1e-6
        first = ts[np.argmax(pos)]
        meas = float(np.sum(np.diff(ts)[pos[1:]]))
        ok &= pos.any() and meas > 1.0
        if g == 1.0:
            ok &= bool((kap[1:] > 0).all())
        det.append(f"g={g}: kappa > 0 for all t > 0" if g == 1.0 else f"g={g}: kappa > 0 from t = {first:.3g}")
    rep.check("(e) the profile becomes positive on a set of scales of positive measure", ok, "; ".join(det))

    ok, det = True, []
    for g in (1.0, 0.8, 0.6, 0.5):
        S = lambda t, g=g: -np.cos(t) / g ** 2
        best = max(sector_constant(S, ell) for ell in (0, 1, 2))
        bound = (2 / 3) * math.exp(2 / g ** 2)
        ok &= math.isfinite(best) and best <= bound
        det.append(f"g={g}: optimal {best:.4f} <= {bound:.4g}")
    rep.check("(f) absolute necessity false: explicit finite Poincare constant without multiscale analysis", ok, "; ".join(det))

    ok, det = True, []
    for g in (1.0, 0.8, 0.6, 0.5):
        F = Flow(-X / g ** 2)
        C = F.constant()[0]
        o = optimal_constant(F.S)
        ok &= math.isfinite(C) and o <= C and F.kappa(0.0) <= 2e-3
        det.append(f"g={g}: optimal {o:.4f} <= C_kappa {C:.4g}")
    rep.check("(g) relative necessity: finite multiscale constant with kappa(0) <= 0, bounding the optimal constant", ok, "; ".join(det))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
