Version 1.1.8
-------------
* Simplified barrier parameters in augl.
* Added problem properties and methods for checking whether solvers support certain problem properties.
* Added methods to convert general problem into linear, mixed integer linear, and quadratic problems.
* Added methods to write lp files from linear and mixed integer linear problems.
* Added command-line interfaces for cbc and clp.
* Added command-line interface for cplex.
* Added acceleration factor to nr.
* Added recovery strategy for augl linear system factorization error.
* Lagrange multiplier extraction from cbc/clp/cplex solutions.

Version 1.1.7
-------------
- Got CBC interface working again on Linux.
- Improved location of inlp termination condition in order not to miss output of last iter.
- Added version file.

Version 1.1.6
-------------
* Fixed bug with Linux installation with OPTALG_IPOPT=true.
* Made AugL/INLP/Ipopt work with empty problems.
* Added line_search option to INLP as well as line search maxiter param.
* Added UMFPACK linear solver interface (can be used if scikit-umfpack is installed).
* Changed 'linsolver' param value of nr to 'default'.
* INLP improvements: separate primal and dual steps, better var initializations.
* Fixed IPOPT windows dll handling.
* Seperated feasibility and optimality tolerances in inlp.

Version 1.1.5
-------------
* Added initial dual update to AugL when initial point is primal feasible.
* Exposed max_iter option for IPOPT.
* Made maxiter of line seach an optional param.
* Removed linesearch from INLP (improved performance) until KKT modifications are introduced.
* Added automatic ipopt/mumps building and linking, controlled by env variable OPTAlG_IPOPT.
* Added automatic clp building and linking, controlled by env variable OPTALG_CLP.
* Uploaded OPTALG to pypi.
* Updated documentation.

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
