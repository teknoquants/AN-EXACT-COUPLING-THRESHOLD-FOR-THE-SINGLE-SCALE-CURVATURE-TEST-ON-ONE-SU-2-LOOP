# Validation scripts: An Exact Coupling Threshold for the Single-Scale Curvature Test on One SU(2) Loop

Companion code to the white paper by R. Cooperman and A.E. Petrunina V.V. (XISOSIS, Vienna, 12 November 2026).
Released under the MIT License (see `LICENSE`; the licence is also reproduced at the top of every script).

GitHub: https://github.com/teknoquants/AN-EXACT-COUPLING-THRESHOLD-FOR-THE-SINGLE-SCALE-CURVATURE-TEST-ON-ONE-SU-2-LOOP · DOI: https://doi.org/10.5281/zenodo.24168254

## Run

```
pip install numpy scipy sympy mpmath
python run_all.py
```

Python 3.10+, NumPy ≥ 2.0, SciPy ≥ 1.12, SymPy. Runtime is about 75 seconds.
`run_log.txt` is the output of a full run: 46 checks, all passing.

## Coverage

| Script | Paper item | Checks |
|---|---|---|
| `validate_definitions.py` | Definitions 1–5 | Ric = metric on su(2); conjugation orbits (a point at ±𝟙, two-dimensional elsewhere, radial horizontal line); Z_g = 2g²I₁(1/g²); β = 1/(2g²); the initial tensors in the radial calculus of the preceding paper; C_κ for constant profiles |
| `validate_lemma1_2.py` | Lemma 1, Lemma 2, remark on compactness | X² = −½\|X\|²𝟙; **∇²x = −(x/2)·metric at matrix level** (quadratic form and polarization); Δx = −(3/2)x; **control: the identity fails for SU(3)**; the geodesic through −𝟙, x = −cos(s/√2), Hessian 1/(2g²) (symbolic and numerical); every direction at −𝟙 is horizontal |
| `validate_theorem3_corollary4.py` | Theorem 3, Corollary 4, both remarks | **T₀ = (1 + x/g²)·metric on all vectors** by finite differences; κ₀(g) = 1 − 1/g² attained only at −𝟙; the sign pattern in g and the set {x < −g²}; a control violating (H); B₀ and the threshold 1/√2; **Bakry–Émery inequality on class and non-invariant sectors**; the window 1/√2 < g ≤ 1; horizontal = full |
| `validate_proposition5.py` | Proposition 5, all proof steps | Bounds on Ω_g; the variational identity; Haar constant 2/3; **each inequality of the chain** for 100 random functions per coupling; the conclusion on class and non-invariant sectors, including g ≤ 1 |
| `validate_proposition6_corollary7.py` | Proposition 6, Corollary 7 | C² convergence of the heat-smoothed weight at rate 3/2; continuity and limits of κ_g(t); finiteness and stability of the multiscale constant; divergence for non-positive profiles; the sign change of the actual profile; finite constants without multiscale analysis |
| `validate_proposition8.py` | Proposition 8, all proof steps | Hypothesis (Z) on seven open boxes in 2 and 3 dimensions, **and its failure on a torus**; the Baker–Campbell–Hausdorff step; **Hessian and T₀ at the all-minus-one configuration by finite differences**; gauge tangent = d₀X; the top eigenvector is horizontal and gives 1 − λ_max/g²; the bounds 4 ≤ λ_max ≤ 8(d−1); the quoted values; the one-loop case |
| `validate_remark9.py` | Remark 9 | All 30 table entries; the optimal constant by two independent solvers; the three observations; consistency with the preceding paper's Wilson table |

`common.py` holds the character-basis tools and the flow of the preceding paper, plus SU(2) matrix tools, a
finite-difference sector solver, cubical complexes and a GF(2) solver.

## What is not validated

The exact threshold of the single-scale test on a lattice with more than one loop: Proposition 8 examines one
configuration and is one-sided. Any bound uniform in the lattice. The size of the multiscale constant beyond its
numerical value. Statements about all functions are checked on class functions and on the non-invariant sectors
ℓ = 1, 2, not on every sector. Passing these checks supports the proofs; it does not replace them, and it says nothing
about a continuum limit or a mass gap.
