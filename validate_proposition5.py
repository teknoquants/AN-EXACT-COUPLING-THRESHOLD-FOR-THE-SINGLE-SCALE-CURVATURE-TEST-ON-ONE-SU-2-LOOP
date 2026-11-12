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
"""Validation of Proposition 5 (direct comparison) and every step of its proof.

What is checked (couplings g = 2, 1.25, 1, 0.8, 0.6, 0.5; the last four are in the range where kappa_0(g) <= 0)
  (a) e^{-1/g^2} <= Omega_g <= e^{1/g^2} and max/min = e^{2/g^2};
  (b) the variational identity Var_mu(f) = inf_a int (f - a)^2 dmu (the minimum is at a = mean, checked on a grid);
  (c) the Haar Poincare constant: smallest positive eigenvalue of -Laplacian on SU(2) is 3/2, so C_H = 2/3
      (class sector and non-invariant sectors ell = 1, 2), and the weaker Bakry-Emery value 1;
  (d) each inequality of the chain, for 100 random class functions per coupling:
        Var <= int (f-c)^2 dmu_g <= (max/Z) int (f-c)^2 dU <= (max/Z) C_H int |grad f|^2 dU <= (max/min) C_H int |grad f|^2 dmu_g;
  (e) the conclusion for ALL functions: optimal constants in the sectors ell = 0, 1, 2 are <= (2/3) e^{2/g^2} <= e^{2/g^2};
  (f) the constant is finite for every g and increases as g decreases, while kappa_0(g) <= 0 for g <= 1.
"""
import math
import sys

import numpy as np

from common import Report, X, WQ, integ, d1, random_class_function, sector_constant, optimal_constant

COUPLINGS = (2.0, 1.25, 1.0, 0.8, 0.6, 0.5)


def main():
    rep = Report("Proposition 5")
    rng = np.random.default_rng(1505)

    xs = np.linspace(-1, 1, 100001)
    ok = True
    for g in COUPLINGS:
        Om = np.exp(xs / g ** 2)
        ok &= abs(Om.min() - math.exp(-1 / g ** 2)) < 1e-12 * Om.min() and abs(Om.max() - math.exp(1 / g ** 2)) < 1e-12 * Om.max()
        ok &= abs(Om.max() / Om.min() - math.exp(2 / g ** 2)) < 1e-9 * math.exp(2 / g ** 2)
    rep.check("(a) bounds on Omega_g and max/min = e^{2/g^2}", ok)

    ok = True
    for g in (1.0, 0.6):
        w = WQ * np.exp(X / g ** 2); w = w / w.sum()
        f = random_class_function(rng)
        mean = w @ f
        var = w @ (f - mean) ** 2
        grid = mean + np.linspace(-1, 1, 2001)
        vals = np.array([w @ (f - a) ** 2 for a in grid])
        ok &= abs(vals.min() - var) < 1e-12 and abs(grid[vals.argmin()] - mean) < 1e-12 and (vals >= var - 1e-14).all()
    rep.check("(b) Var_mu(f) = inf_a int (f - a)^2 dmu", ok)

    haar = [sector_constant(lambda t: 0 * t, ell) for ell in (0, 1, 2)]
    rep.check("(c) Haar Poincare constant 2/3 (first eigenvalue 3/2); weaker value 1 also valid",
              abs(optimal_constant(0 * X) - 2 / 3) < 1e-9 and abs(max(haar) - 2 / 3) < 1e-4 and max(haar) <= 1,
              "sectors ell=0,1,2: " + ", ".join(f"{h:.4f}" for h in haar))

    ok, slack = True, []
    for g in COUPLINGS:
        Om = np.exp(X / g ** 2)
        Z, mx, mn = integ(Om), math.exp(1 / g ** 2), math.exp(-1 / g ** 2)
        worst = 0
        for _ in range(100):
            f = random_class_function(rng, deg=int(rng.integers(1, 9)))
            c = integ(f)
            mean = integ(f * Om) / Z
            var = integ((f - mean) ** 2 * Om) / Z
            t1 = integ((f - c) ** 2 * Om) / Z
            t2 = mx / Z * integ((f - c) ** 2)
            t3 = mx / Z * (2 / 3) * integ(d1(f) ** 2 / 2)
            t4 = mx / mn * (2 / 3) * integ(d1(f) ** 2 / 2 * Om) / Z
            ok &= var <= t1 * (1 + 1e-12) and t1 <= t2 * (1 + 1e-12) and t2 <= t3 * (1 + 1e-9) and t3 <= t4 * (1 + 1e-12)
            worst = max(worst, var / t4)
        slack.append(f"g={g}: max Var/bound {worst:.3g}")
    rep.check("(d) every inequality of the chain, 100 random functions per coupling", ok, "; ".join(slack))

    ok, det = True, []
    for g in COUPLINGS:
        S = lambda t, g=g: -np.cos(t) / g ** 2
        cs = [sector_constant(S, ell) for ell in (0, 1, 2)]
        b = (2 / 3) * math.exp(2 / g ** 2)
        ok &= max(cs) <= b <= math.exp(2 / g ** 2)
        det.append(f"g={g}: {max(cs):.4f} <= {b:.4g}")
    rep.check("(e) optimal constants on class and non-invariant sectors <= (2/3) e^{2/g^2} <= e^{2/g^2}", ok, "; ".join(det))

    bs = [(2 / 3) * math.exp(2 / g ** 2) for g in COUPLINGS]
    ok = all(math.isfinite(b) for b in bs) and all(bs[i] < bs[i + 1] for i in range(len(bs) - 1))
    ok &= all((1 - 1 / g ** 2) <= 0 for g in COUPLINGS if g <= 1)
    rep.check("(f) finite for every g, deteriorating as g decreases, including where kappa_0(g) <= 0", ok)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
