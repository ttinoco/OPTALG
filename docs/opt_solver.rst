.. _opt_solver:

********************
Optimization Solvers
********************

In OPTALG, optimization solvers are objects of type :class:`OptSolver <optalg.opt_solver.opt_solver.OptSolver>`, and optimization problems are objects of type :class:`OptProblem <optalg.opt_solver.problem.OptProblem>`, which represents general problems of the form 

.. math:: 
   :nowrap:

   \begin{alignat*}{3}
   & \mbox{minimize}   \quad && \varphi(x)       \ && \\
   & \mbox{subject to} \quad && Ax = b           \ && : \lambda \\
   &                   \quad && f(x) = 0         \ && : \nu \\
   &                   \quad && l \le x \le u    \ && : \pi, \mu \\
   &                   \quad && Px \in \{0,1\}^m,\ &&
   \end{alignat*}

where :math:`P` is a matrix that extracts a sub-vector of :math:`x`. 

Before solving a :class:`problem <optalg.opt_solver.problem.OptProblem>` with a specific solver, the solver parameters can be configured using the method :func:`set_parameters() <optalg.opt_solver.opt_solver.OptSolver.set_parameters>`. Then, the :func:`solve() <optalg.opt_solver.opt_solver.OptSolver.solve>` method can be invoked with the :class:`problem <optalg.opt_solver.problem.OptProblem>` to be solved as its argument. The status, optimal primal variables, and optimal dual variables can be extracted using the class methods :func:`get_status() <optalg.opt_solver.opt_solver.OptSolver.get_status>`, :func:`get_primal_variables() <optalg.opt_solver.opt_solver.OptSolver.get_primal_variables>`, and :func:`get_dual_variables() <optalg.opt_solver.opt_solver.OptSolver.get_dual_variables>`, respectively.

.. _opt_solver_nr:

NR
==

This solver, which corresponds to the class :class:`OptSolverNR <optalg.opt_solver.nr.OptSolverNR>`, solves problems of the form

.. math:: 
   :nowrap:

   \begin{alignat*}{2}
   & \mbox{find}       \quad && x \\
   & \mbox{subject to} \quad && Ax = b \\
   &                   \quad && f(x) = 0
   \end{alignat*}

using the Newton-Raphson algorithm. It requires the number of variables to be equal to the number of constraints.

.. _opt_solver_clp:

Clp
===

This is a wrapper of the solver `Clp`_ from COIN-OR. It corresponds to the class :class:`OptSolverClp <optalg.opt_solver.clp.OptSolverClp>`, and solves problems of the form 

.. math:: 
   :nowrap:

   \begin{alignat*}{3}
   & \mbox{minimize}   \quad && c^Tx           \ && \\
   & \mbox{subject to} \quad && Ax = b         \ && : \lambda \\
   &                   \quad && l \le x \le u  \ && : \pi, \mu.
   \end{alignat*}

Linear optimization problems solved with this solver must be instances of the class :class:`LinProblem <optalg.opt_solver.problem_lin.LinProblem>`, which is a subclass of :class:`OptProblem <optalg.opt_solver.problem.OptProblem>`.

.. _opt_solver_cbc:

Cbc
===

This is a wrapper of the solver `Cbc`_ from COIN-OR. It corresponds to the class :class:`OptSolverCbc <optalg.opt_solver.cbc.OptSolverCbc>`, and solves problems of the form 

.. math:: 
   :nowrap:

   \begin{alignat*}{3}
   & \mbox{minimize}   \quad && c^Tx              \\
   & \mbox{subject to} \quad && Ax = b            \\
   &                   \quad && l \le x \le u     \\
   &                   \quad && Px \in \{0,1\}^m.
   \end{alignat*}

Mixed-integer linear optimization problems solved with this solver must be instances of the class :class:`MixIntLinProblem <optalg.opt_solver.problem_mixintlin.MixIntLinProblem>`, which is a subclass of :class:`OptProblem <optalg.opt_solver.problem.OptProblem>`.

.. note:: Currently, this wrapper is not working and will be updated soon.

.. _opt_solver_iqp:

IQP
===

This solver, which corresponds to the class :class:`OptSolverIQP <optalg.opt_solver.iqp.OptSolverIQP>`, solves convex quadratic problems of the form

.. math:: 
   :nowrap:

   \begin{alignat*}{3}
   & \mbox{minimize}   \quad && \frac{1}{2}x^THx + g^Tx \ && \\
   & \mbox{subject to} \quad && Ax = b                  \ && : \lambda \\
   &                   \quad && l \le x \le u           \ && : \pi, \mu
   \end{alignat*}

using a primal-dual interior-point algorithm. Quadratic problems solved with this solver must be instances of the class :class:`QuadProblem <optalg.opt_solver.problem_quad.QuadProblem>`, which is a subclass of :class:`OptProblem <optalg.opt_solver.problem.OptProblem>`. The following example shows how to solve the quadratic problem

.. math:: 
   :nowrap:

   \begin{alignat*}{2}
   & \mbox{minimize}   \quad && 3x_1-6x_2 + 5x_1^2 - 2x_1x_2 + 5x_2^2 \\
   & \mbox{subject to} \quad && x_1 + x_2 = 1 \\
   &                   \quad && 0.2 \le x_1 \le 0.8 \\
   &                   \quad && 0.2 \le x_2 \le 0.8
   \end{alignat*}

using :class:`OptSolverIQP <optalg.opt_solver.iqp.OptSolverIQP>`::

  >>> import numpy as np
  >>> from optalg.opt_solver import OptSolverIQP, QuadProblem

  >>> g = np.array([3.,-6.])
  >>> H = np.array([[10.,-2],
  ...               [-2.,10]])

  >>> A = np.array([[1.,1.]])
  >>> b = np.array([1.])

  >>> u = np.array([0.8,0.8])
  >>> l = np.array([0.2,0.2])

  >>> problem = QuadProblem(H,g,A,b,l,u)

  >>> solver = OptSolverIQP()

  >>> solver.set_parameters({'quiet': True,
  ...                        'tol': 1e-6})

  >>> solver.solve(problem)

  >>> print solver.get_status()
  solved

Then, the optimal primal and dual variables can be extracted, and feasibility and optimality can be checked as follows::

  >>> x = solver.get_primal_variables()
  >>> lam,nu,mu,pi = solver.get_dual_variables()

  >>> print x
  [ 0.20  0.80 ]

  >>> print x[0] + x[1]
  1.00

  >>> print l <= x
  [ True  True ]

  >>> print x <= u
  [ True  True ]

  >>> print pi
  [ 9.00e-01  1.80e-06 ]

  >>> print mu
  [ 1.80e-06  9.00e-01 ]

  >>> print np.linalg.norm(g+np.dot(H,x)-np.dot(A.T,lam)+mu-pi)
  1.25e-15

  >>> print np.dot(mu,u-x)
  2.16e-06

  >>> print np.dot(pi,x-l)
  2.16e-06

.. _opt_solver_inlp:

INLP
====

This solver, which corresponds to the class :class:`OptSolverINLP <optalg.opt_solver.inlp.OptSolverINLP>`, solves general nonlinear optimization problems of the form

.. math:: 
   :nowrap:

   \begin{alignat*}{3}
   & \mbox{minimize}   \quad && \varphi(x)     \ && \\
   & \mbox{subject to} \quad && Ax = b         \ && : \lambda \\
   &                   \quad && f(x) = 0       \ && : \nu \\
   &                   \quad && l \le x \le u  \ && : \pi, \mu
   \end{alignat*}

using a primal-dual interior-point algorithm. It computes Newton steps for solving modified KKT conditions and it is not globally convergent (yet!).

.. _opt_solver_augl:

AugL
====

This solver, which corresponds to the class :class:`OptSolverAugL <optalg.opt_solver.augl.OptSolverAugL>`, solves optimization problems of the form

.. math:: 
   :nowrap:

   \begin{alignat*}{3}
   & \mbox{minimize}   \quad && \varphi(x)     \ && \\
   & \mbox{subject to} \quad && Ax = b         \ && : \lambda \\
   &                   \quad && f(x) = 0       \ && : \nu \\
   &                   \quad && l \le x \le u  \ && : \pi, \mu 
   \end{alignat*}

using an Augmented Lagrangian algorithm. It requires the objective function :math:`\varphi` to be convex.

.. _opt_solver_ipopt:

Ipopt
=====

This is a wrapper of the solver `IPOPT`_ from COIN-OR. It corresponds to the class :class:`OptSolverIpopt <optalg.opt_solver.ipopt.OptSolverIpopt>`, and solves optimization problems of the form

.. math:: 
   :nowrap:

   \begin{alignat*}{3}
   & \mbox{minimize}   \quad && \varphi(x)     \ && \\
   & \mbox{subject to} \quad && Ax = b         \ && : \lambda \\
   &                   \quad && f(x) = 0       \ && : \nu \\
   &                   \quad && l \le x \le u  \ && : \pi, \mu.
   \end{alignat*}

.. _IPOPT: https://projects.coin-or.org/Ipopt
.. _CLP: https://projects.coin-or.org/Clp
.. _CBC: https://projects.coin-or.org/Cbc
