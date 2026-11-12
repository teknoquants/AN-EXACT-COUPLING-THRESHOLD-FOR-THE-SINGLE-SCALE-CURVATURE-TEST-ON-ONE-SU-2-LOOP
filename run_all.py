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
"""Run every validation script for the white paper
"An Exact Coupling Threshold for the Single-Scale Curvature Test on One SU(2) Loop".

Usage:  python run_all.py      (exit code 0 if every check passes)
"""
import sys
import time

import validate_definitions
import validate_lemma1_2
import validate_theorem3_corollary4
import validate_proposition5
import validate_proposition6_corollary7
import validate_proposition8
import validate_remark9

SUITES = [
    ("Definitions 1-5", validate_definitions),
    ("Lemma 1 and Lemma 2 - Hessian of x; adverse configuration", validate_lemma1_2),
    ("Theorem 3 and Corollary 4 - exact threshold; classical tensor", validate_theorem3_corollary4),
    ("Proposition 5 - direct comparison", validate_proposition5),
    ("Proposition 6 and Corollary 7 - the criterion succeeds; two meanings of 'necessary'", validate_proposition6_corollary7),
    ("Proposition 8 - finite lattices", validate_proposition8),
    ("Remark 9 - numerical comparison", validate_remark9),
]


def main():
    results = []
    for title, mod in SUITES:
        print(f"\n=== {title} ===")
        t = time.time()
        ok = mod.main()
        results.append((title, ok, time.time() - t))
    print("\n=== Summary ===")
    for title, ok, dt in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {title}  ({dt:.1f} s)")
    print("\nNot validated: the exact threshold of the single-scale test on a lattice with more than one loop\n"
          "(Proposition 8 is one-sided: only one configuration is examined); any bound uniform in the lattice; the size\n"
          "of the multiscale constant beyond its numerical value. Statements for all functions are checked on class\n"
          "functions and on the non-invariant sectors ell = 1, 2. Passing these checks supports the proofs; it does not\n"
          "replace them.")
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
