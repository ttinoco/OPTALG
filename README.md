# OPTALG

[![Build Status](https://travis-ci.org/ttinoco/OPTALG.svg?branch=master)](https://travis-ci.org/ttinoco/OPTALG)
[![Documentation Status](https://readthedocs.org/projects/optalg/badge/?version=latest)](http://optalg.readthedocs.io/en/latest/?badge=latest)

## Overview

OPTALG is a Python package that provides algorithms, wrappers, and tools for solving optimization problems. Currently, it contains the following:
* Newton-Raphson algorithm for solving systems of equations.
* Primal-dual interior-point algorithms for solving convex problems.
* Augmented Lagrangian algorithm for solving problems with convex objective.
* Interface for the interior-point solver [Ipopt](https://projects.coin-or.org/Ipopt) (via cython).
* Interface for the linear programming solver [Clp](https://projects.coin-or.org/Clp) (via command-line or cython).
* Interface for the mixed-integer linear programming solver [Cbc](https://projects.coin-or.org/Cbc) (via command-line or cython).
* Interface for the mixed-integer linear programming solver CPLEX (via command-line).
* Common interface for linear solvers ([SuperLU](http://crd-legacy.lbl.gov/~xiaoye/SuperLU/), [MUMPS](http://mumps-solver.org), [UMFPACK](https://directory.fsf.org/wiki/UMFPACK)) (via cython).

This package is meant to be used by other Python packages and not by users directly. Currently, it is used by:
* [GRIDOPT](https://github.com/ttinoco/GRIDOPT)
* [OPTMOD](https://github.com/ttinoco/OPTMOD)

## Documentation

The documentation for this package can be found in <http://optalg.readthedocs.io/>.

## License

BSD 2-clause license.
