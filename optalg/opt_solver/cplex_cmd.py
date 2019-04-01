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

class OptSolverCplexCMD(OptSolver):

    parameters = {'quiet' : False}

    def __init__(self):
        """
        CPLEX solver interface (via command-line interface).
        """

        # Check
        if not utils.cmd_exists('cplex'):
            raise ImportError('cplex cmd not available')
        
        OptSolver.__init__(self)
        self.parameters = OptSolverCplexCMD.parameters.copy()

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

        import xml.etree.ElementTree as ET

        x = np.zeros(problem.c.size)

        tree = ET.parse(filename)
        root = tree.getroot()

        header = root.find('header')
        status = header.get('solutionStatusString')

        for var in root.find('variables'):
            index = int(var.get('index'))
            value = float(var.get('value'))
            x[index] = value

        return status, x

    def solve(self, problem):

        # Local vars
        params = self.parameters

        # Parameters
        quiet = params['quiet']
        
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
            cmd = ['cplex', '-c', 'read', input_filename, 'optimize', 'write', output_filename, 'quit']
            if not quiet:
                code = subprocess.call(cmd)
            else:
                code = subprocess.call(cmd,
                                       stdout=open(os.devnull, 'w'),
                                       stderr=subprocess.STDOUT)
            assert(code == 0)
            status, self.x = self.read_solution(output_filename, self.problem)
        except Exception as e:
            raise OptSolverError_CplexCMDCall(self)
        finally:
            if os.path.isfile(input_filename):
                os.remove(input_filename)
            if os.path.isfile(output_filename):
                os.remove(output_filename)
            if os.path.isfile('cplex.log'):
                os.remove('cplex.log')

        if 'optimal' in status.lower():
            self.set_status(self.STATUS_SOLVED)
            self.set_error_msg('')
        else:
            raise OptSolverError_CplexCMD(self)
