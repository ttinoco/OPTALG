Targets
-------
* ECOS interface.
* SNOPT interface.
* CPLEX interace.

Unreleased
----------
* Added initial dual update to AugL when initial point is primal feasible (done).
* Exposed max_iter option for IPOPT (done).
* Made maxiter of line seach an optionalo param (done).
* Removed linesearch from INLP, improved performance (done).

Version 1.1.4
-------------
* Improved error checks in ipopt wrapper, and added derivative_test and hessian approx option.
* Added linear_solver and print_level options for ipopt.
* IQP allows general problem as input, forms QP approximation.
* Naive, but hopefully efficient interior-point nonlinear programming solver (inlp) based on taking newton steps on the KKT conditions.
* Improved obj scaling of IQP (to match that of INLP).
* Improved bound stretching in AugL for problems with non-empty interior.

Version 1.1.3
-------------
* Portable setup.py (--with argument).
* Linear problem class.
* Mixed integer linear problem class.
* Coin-OR Clp interface.
* Coin-OR Cbc interface.
* Auto objective scaling in AugL solver.
* IPOPT wrapper extracts number of iterations.

Version 1.1.2
-------------
* AugL solver handles variable bounds using a barrier.

Version 1.1.1
-------------
* IPOPT interface.
* Unittests comparing IQP, AugL and IPOPT on QPs.

Version 1.1
-----------
* Auto objective function scaling in IQP.
* Basic IQP untitests.
* Multistage stochastic solvers (SH, SDDP).
* Scenario tree class.
* Python 2 and 3 compatibility.
* Made dill a requirement.

Version 1.0
-----------
* Initial version.
