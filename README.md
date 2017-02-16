# OPTALG #

## Overview ##

OPTALG is a Python package that provides algorithms, wrappers, and tools for solving large and sparse optimization problems. Currently, it contains the following:

* Newton-Raphson algorithm for solving systems of equations.
* Interior-point algorithm for solving convex quadratic problems.
* Augmented Lagrangian algorithm for solving problems with convex objective.
* Interface for the interior-point solver [Ipopt](https://projects.coin-or.org/Ipopt).
* Common interface for linear solvers ([SuperLU](http://crd-legacy.lbl.gov/~xiaoye/SuperLU/), [MUMPS](http://mumps-solver.org)).

## License ##

BSD 2-clause license.

## Documentation ##

The documentation for this package can be found in http://ttinoco.github.io/OPTALG.

## Download ##

The latest version of the package can be obtained from https://github.com/ttinoco/OPTALG.

## Dependencies ##

* [Numpy](http://www.numpy.org) (>=1.8.2)
* [Scipy](http://www.scipy.org) (>=0.13.3)
* [Dill](https://pypi.python.org/pypi/dill) (>=0.2.5)
* [MUMPS](http://mumps-solver.org) (==4.10) (optional)
* [Ipopt](https://projects.coin-or.org/Ipopt) (>=3.12.6) (optional)
 
## Contributors ##

* [Tomas Tinoco De Rubira](http://n.ethz.ch/~tomast/)
