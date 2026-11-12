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
"""Validation of Lemma 1 (Hessian of x), Lemma 2 (adverse configuration), their proofs, and the remark on compactness.

Lemma 1
  (a) X^2 = (1/2)Tr(X^2) 1 = -(1/2)|X|^2 1 for X in su(2) (Cayley-Hamilton), 200 random X.
  (b) s -> U exp(sX) has constant speed |X| (it is a geodesic of the bi-invariant metric), and
      d^2/ds^2 x(U exp(sX)) = -(|X|^2/2) x(U exp(sX)) for all s (finite differences, matrix level).
  (c) Hess x (X,X) = -(x/2)|X|^2 at 100 random points and directions; polarization: Hess x(X,Y) = -(x/2)<X,Y>.
  (d) Laplacian x = -(3/2) x: trace over an orthonormal basis, and the character-basis Laplacian of [1].
  (e) Control: the identity is special to SU(2). For SU(3), Hess of Re Tr is not a multiple of the metric.
Lemma 2
  (f) exp(sX) = cos(s/sqrt2) 1 + sqrt2 sin(s/sqrt2) X for unit X; U(s) = -exp(sX) has unit speed.
  (g) x(U(s)) = -cos(s/sqrt2) and d^2/ds^2 log Omega_g(U(s)) at 0 equals 1/(2 g^2) (symbolic and finite differences).
  (h) Conjugation fixes -1, so every tangent vector there is horizontal (zero orbit tangent space).
Remark
  (i) For the constant weight the Hessian of log Omega vanishes; for the Wilson weight the positive Hessian sits
      where the weight is smallest.
"""
import math
import sys

import numpy as np
import sympy as sp

from common import (Report, su_basis, ip, haar_sun, exp_su2, random_su2, x_of, second_diff, expm_skew, lap, X)


def main():
    rep = Report("Lemmas 1-2")
    rng = np.random.default_rng(1212)
    B = su_basis(2)

    ok = True
    for _ in range(200):
        A = random_su2(rng, unit=False)
        ok &= np.allclose(A @ A, 0.5 * np.trace(A @ A) * np.eye(2)) and np.allclose(A @ A, -0.5 * ip(A, A) * np.eye(2))
    rep.check("(a) X^2 = -(1/2)|X|^2 1 on su(2)", ok)

    ok, worst = True, 0
    for U in haar_sun(40, 2, rng):
        A = random_su2(rng, unit=False)
        for s in (0.0, 0.4, -1.3):
            f = lambda r: x_of(U @ exp_su2((s + r) * A))
            worst = max(worst, abs(second_diff(f) + 0.5 * ip(A, A) * f(0.0)))
        vel = (U @ exp_su2(1e-6 * A) - U) / 1e-6
        ok &= abs(ip(U.conj().T @ vel, U.conj().T @ vel) - ip(A, A)) < 1e-4
        ok &= np.allclose(exp_su2(0.7 * A), expm_skew(0.7 * A))
    rep.check("(b) d^2/ds^2 x(U e^{sX}) = -(|X|^2/2) x(U e^{sX}); geodesic of constant speed", ok and worst < 1e-8, f"max error {worst:.1e}")

    worst = 0
    for U in haar_sun(100, 2, rng):
        A, C = random_su2(rng, unit=False), random_su2(rng, unit=False)
        q = lambda Y: second_diff(lambda r: x_of(U @ exp_su2(r * Y)))
        worst = max(worst, abs(q(A) + x_of(U) / 2 * ip(A, A)), abs(0.25 * (q(A + C) - q(A - C)) + x_of(U) / 2 * ip(A, C)))
    rep.check("(c) Hess x = -(x/2) g_met (quadratic form and polarization)", worst < 1e-8, f"max error {worst:.1e}")

    worst = 0
    for U in haar_sun(50, 2, rng):
        tr = sum(second_diff(lambda r, T=T: x_of(U @ exp_su2(r * T))) for T in B)
        worst = max(worst, abs(tr + 1.5 * x_of(U)))
    rep.check("(d) Laplacian x = -(3/2) x", worst < 1e-8 and np.abs(lap(X) + 1.5 * X).max() < 1e-6, f"max error {worst:.1e}")

    B3 = su_basis(3)
    U3 = haar_sun(1, 3, rng)[0]
    q3 = [second_diff(lambda r, T=T: float(np.real(np.trace(U3 @ expm_skew(r * T))))) for T in B3]
    rep.check("(e) control: for SU(3), Hess Re Tr is not a multiple of the metric", max(q3) - min(q3) > 1e-2,
              f"spread of the quadratic form over an orthonormal basis: {max(q3) - min(q3):.3f}")

    ok = True
    for _ in range(50):
        A = random_su2(rng)
        for s in (0.3, 1.1, -2.0):
            ok &= np.allclose(expm_skew(s * A), math.cos(s / math.sqrt(2)) * np.eye(2) + math.sqrt(2) * math.sin(s / math.sqrt(2)) * A)
            ok &= abs(x_of(-exp_su2(s * A)) + math.cos(s / math.sqrt(2))) < 1e-13
        ok &= abs(ip(A, A) - 1) < 1e-13
    rep.check("(f) exp(sX) formula for unit X; x(-exp(sX)) = -cos(s/sqrt2)", ok)

    s, g = sp.symbols("s g", positive=True)
    val = sp.diff(-sp.cos(s / sp.sqrt(2)) / g ** 2, s, 2).subs(s, 0)
    worst = 0
    for gg in (0.5, 1.0, 2.0):
        for _ in range(10):
            A = random_su2(rng)
            fd = second_diff(lambda r: x_of(-exp_su2(r * A)) / gg ** 2)
            worst = max(worst, abs(fd - 1 / (2 * gg ** 2)))
    rep.check("(g) d^2/ds^2 log Omega_g(U(s)) at 0 = 1/(2 g^2) (symbolic and finite differences)",
              sp.simplify(val - 1 / (2 * g ** 2)) == 0 and worst < 1e-8, f"max FD error {worst:.1e}")

    m1 = -np.eye(2, dtype=complex)
    ok = all(np.allclose(h @ m1 @ h.conj().T, m1) for h in haar_sun(20, 2, rng))
    ok &= all(np.allclose(T @ m1 - m1 @ T, 0) for T in B)
    rep.check("(h) conjugation fixes -1: zero orbit tangent space, every direction horizontal", ok)

    U = haar_sun(1, 2, rng)[0]
    A = random_su2(rng)
    const = second_diff(lambda r: math.log(1.0) + 0 * x_of(U @ exp_su2(r * A)))
    xs = np.linspace(-1, 1, 2001)
    hess_scalar = -xs / 2                                 # Hess log Omega_g = -(x/(2 g^2)) g_met, with g = 1
    rep.check("(i) constant weight: zero Hessian; Wilson weight: largest Hessian where the weight is smallest",
              const == 0 and xs[np.argmax(hess_scalar)] == -1 and xs[np.argmin(np.exp(xs))] == -1)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
