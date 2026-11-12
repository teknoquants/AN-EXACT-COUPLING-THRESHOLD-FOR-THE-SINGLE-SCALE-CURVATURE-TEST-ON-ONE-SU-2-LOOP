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
"""Validation of Remark 9 (numerical comparison on one loop).

Remark 9 is a table of numbers, so validation means re-computation.
What is checked
  (a) every printed entry of the table: kappa_0(g) = 1 - 1/g^2, the classical single-scale constant 1/(1 - 1/(2g^2))
      (or "fails"), the direct-comparison constant (2/3) e^{2/g^2}, the multiscale constant C_kappa (flow code of the
      preceding paper, Simpson quadrature in log-scale), and the optimal constant on class functions (two
      independent solvers: character basis and finite differences);
  (b) the three observations in the text: for g <= 1 both the direct comparison and the multiscale criterion are
      finite and the multiscale constant is the smaller; no bound is close to the optimal constant, which decreases
      as g decreases while the bounds grow; at g = 1 the classical test gives the constant 2;
  (c) consistency with the Wilson table of the preceding paper under beta = 1/(2 g^2).
"""
import math
import sys

import numpy as np

from common import Report, Flow, X, optimal_constant, sector_constant

R2 = math.sqrt(2)
TABLE = [  # g, kappa_0, classical, direct comparison, multiscale, optimal   (as printed in the paper)
    (2.0, 0.75, 1.143, 1.099, 1.164, 0.664), (R2, 0.50, 1.333, 1.812, 1.380, 0.656), (1.0, 0.00, 2.000, 4.926, 2.078, 0.625),
    (1 / R2, -1.00, None, 36.4, 6.55, 0.527), (0.5, -3.00, None, 1987.0, 222.0, 0.328)]
PREVIOUS = {0.25: (1.38, 0.656), 0.5: (2.08, 0.625), 1.0: (6.55, 0.527), 2.0: (222.0, 0.328)}     # beta: (multiscale, optimal) in [1]


def main():
    rep = Report("Remark 9")
    ok_a = ok_b = ok_c = True
    det, opts, cms = [], [], []
    for g, k0, cl, dc, ms, opt in TABLE:
        F = Flow(-X / g ** 2)
        C = F.constant()[0]
        o1 = optimal_constant(F.S)
        o2 = sector_constant(lambda t, g=g: -np.cos(t) / g ** 2, 0)
        direct = (2 / 3) * math.exp(2 / g ** 2)
        kbe = 1 - 1 / (2 * g ** 2)
        ok_a &= abs((1 - 1 / g ** 2) - k0) < 5e-3 and abs(direct - dc) / dc < 1e-3
        ok_a &= abs(C - ms) / ms < 2e-3 and abs(o1 - opt) < 6e-4 and abs(o1 - o2) < 2e-4
        if cl is None:
            ok_a &= kbe <= 1e-12
        else:
            ok_a &= abs(1 / kbe - cl) < 6e-4
        if g <= 1:
            ok_b &= math.isfinite(direct) and math.isfinite(C) and C < direct
        ok_b &= o1 < min(C, direct)
        opts.append(o1); cms.append(C)
        beta = 1 / (2 * g ** 2)
        for b, (pm, po) in PREVIOUS.items():
            if abs(beta - b) < 1e-9:
                ok_c &= abs(C - pm) / pm < 5e-3 and abs(o1 - po) < 1e-3
        det.append(f"g={g:.4f}: C={C:.5g}, direct={direct:.5g}, optimal={o1:.4f}")
    for d in det:
        print("      " + d)
    ok_b &= all(opts[i + 1] < opts[i] for i in range(4)) and all(cms[i + 1] > cms[i] for i in range(4))
    ok_b &= abs(1 / (1 - 1 / 2) - 2) < 1e-15 and abs(1 - 1 / 1.0) < 1e-15
    rep.check("(a) all 30 printed table entries reproduced (optimal constant by two independent solvers)", ok_a)
    rep.check("(b) the three observations in the text", ok_b)
    rep.check("(c) consistent with the Wilson table of the preceding paper (beta = 1/(2 g^2))", ok_c)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
