#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2019, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

from __future__ import print_function
import os
import numpy as np
import tempfile
import subprocess
from . import utils
from .opt_solver_error import *
from .opt_solver import OptSolver
from .problem import OptProblem

class OptSolverCbcCMD(OptSolver):

    parameters = {'quiet' : False, 'debug': False}

    def __init__(self):
        """
        Mixed integer linear "branch and cut" solver from COIN-OR (via command-line interface, version 2.8.5).
        """

        # Check
        if not utils.cmd_exists('cbc'):
            raise ImportError('cbc cmd not available')
        
        OptSolver.__init__(self)
        self.parameters = OptSolverCbcCMD.parameters.copy()

    def supports_properties(self, properties):

        for p in properties:
            if p not in [OptProblem.PROP_CURV_LINEAR,
                         OptProblem.PROP_VAR_CONTINUOUS,
                         OptProblem.PROP_VAR_INTEGER,
                         OptProblem.PROP_TYPE_FEASIBILITY,
                         OptProblem.PROP_TYPE_OPTIMIZATION]:
                return False
        return True

    def read_solution(self, filename, problem):

        f = open(filename, 'r')
        
        l = f.readline().split()
        status = l[0]    
        
        x = np.zeros(problem.c.size)
        for l in f:
            l = l.split()
            name = l[1]
            i = int(name.split('_')[1])
            val = float(l[2])
            x[i] = val
        f.close()
        return status, x

        
    def solve(self, problem):

        # Local vars
        params = self.parameters

        # Parameters
        quiet = params['quiet']
        debug = params['debug']

        # Problem
        try:
            self.problem = problem.to_mixintlin()
        except:
            raise OptSolverError_BadProblemType(self)

        # Solve
        status = ''
        try:
            base_name = next(tempfile._get_candidate_names())
            input_filename = base_name+'.lp'
            output_filename = base_name+'.sol'
            self.problem.write_to_lp_file(input_filename)
            cmd = ['cbc', input_filename, 'solve', 'solution', output_filename]
            if not quiet:
                code = subprocess.call(cmd)
            else:
                code = subprocess.call(cmd,
                                       stdout=open(os.devnull, 'w'),
                                       stderr=subprocess.STDOUT)
            assert(code == 0)
            status, self.x = self.read_solution(output_filename, self.problem)
        except Exception as e:
            raise OptSolverError_CbcCMDCall(self)
        finally:
            if os.path.isfile(input_filename) and not debug:
                os.remove(input_filename)
            if os.path.isfile(output_filename) and not debug:
                os.remove(output_filename)

        if status == 'Optimal':
            self.set_status(self.STATUS_SOLVED)
            self.set_error_msg('')
        else:
            raise OptSolverError_CbcCMD(self)

        
