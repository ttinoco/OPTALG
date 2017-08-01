.. _reference:

*************
API Reference
*************

.. _ref_lin_solver:

Linear Solvers
==============

.. autofunction:: optalg.lin_solver.new_linsolver

.. autoclass:: optalg.lin_solver.lin_solver.LinSolver
   :members:

.. autoclass:: optalg.lin_solver.mumps.LinSolverMUMPS

.. autoclass:: optalg.lin_solver.superlu.LinSolverSUPERLU

Optimization Problems
=====================

.. _ref_opt_problems:

.. autoclass:: optalg.opt_solver.problem.OptProblem
   :members:

.. autoclass:: optalg.opt_solver.problem_lin.LinProblem

.. autoclass:: optalg.opt_solver.problem_mixintlin.MixIntLinProblem

.. autoclass:: optalg.opt_solver.problem_quad.QuadProblem

.. _ref_opt_solvers:

Optimization Solvers
====================

.. autoclass:: optalg.opt_solver.opt_solver.OptSolver
   :members:

.. autoclass:: optalg.opt_solver.nr.OptSolverNR

.. autoclass:: optalg.opt_solver.iqp.OptSolverIQP

.. autoclass:: optalg.opt_solver.inlp.OptSolverINLP

.. autoclass:: optalg.opt_solver.augl.OptSolverAugL

.. autoclass:: optalg.opt_solver.ipopt.OptSolverIpopt

.. autoclass:: optalg.opt_solver.clp.OptSolverClp

.. autoclass:: optalg.opt_solver.cbc.OptSolverCbc


