# OPTALG

[![Build Status](https://travis-ci.org/ttinoco/OPTALG.svg?branch=master)](https://travis-ci.org/ttinoco/OPTALG)
[![Documentation Status](https://readthedocs.org/projects/optalg/badge/?version=latest)](http://optalg.readthedocs.io/en/latest/?badge=latest)

## Overview

OPTALG is a Python package that provides algorithms, wrappers, and tools for solving large and sparse optimization problems. Currently, it contains the following:
* Newton-Raphson algorithm for solving systems of equations.
* Interior-point algorithm for solving convex quadratic problems.
* Augmented Lagrangian algorithm for solving problems with convex objective.
* Interface for the interior-point solver [Ipopt](https://projects.coin-or.org/Ipopt).
* Interface for the linear programming solver [Clp](https://projects.coin-or.org/Clp).
* Interface for mixed integer programming solver [Cbc](https://projects.coin-or.org/Cbc).
* Common interface for linear solvers ([SuperLU](http://crd-legacy.lbl.gov/~xiaoye/SuperLU/), [MUMPS](http://mumps-solver.org)).

## License

BSD 2-clause license.

## Documentation

The documentation for this package can be found in <http://optalg.readthedocs.io/>.
