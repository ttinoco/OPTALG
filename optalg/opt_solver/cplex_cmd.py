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
from multiprocessing import cpu_count

class OptSolverCplexCMD(OptSolver):
    
    parameters = {'quiet' : False,
                  'mipgap': None,
                  'feasibility': None,
                  'debug': False}                  

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
        lam = np.zeros(problem.A.shape[0])
        nu = np.zeros(0)
        mu = np.zeros(x.size)
        pi = np.zeros(x.size)

        tree = ET.parse(filename)
        root = tree.getroot()

        header = root.find('header')
        status = header.get('solutionStatusString')

        for var in root.find('variables'):
            name = var.get('name')
            value = float(var.get('value'))
            index = int(name.split('_')[1])
            x[index] = value
            rcost = var.get('reducedCost')
            if rcost is not None:
                if float(rcost) > 0.:
                    pi[index] = float(rcost)
                else:
                    mu[index] = -float(rcost)

        for c in root.find('linearConstraints'):
            name = c.get('name')
            index = int(name.split('_')[1])
            dual = c.get('dual')
            if dual is not None:
                lam[index] = float(dual)
            
        return status, x, lam, nu, mu, pi

    def solve(self, problem):

        # Local vars
        params = self.parameters

        # Parameters
        quiet = params['quiet']
        mipgap = params['mipgap']
        feasibility = params['feasibility']
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
            cmd = ['cplex']
            cmd += ['-c', 'read', input_filename]
            if mipgap is not None:
                cmd += ['set mip tolerances mipgap %.2e' %mipgap]
            if feasibility is not None:
                cmd += ['set simplex tolerances feasibility %.2e' %feasibility]
            cmd += ['optimize']
            cmd += ['write', output_filename]
            cmd += ['quit']
            if not quiet:
                code = subprocess.call(cmd)
            else:
                code = subprocess.call(cmd,
                                       stdout=open(os.devnull, 'w'),
                                       stderr=subprocess.STDOUT)
            assert(code == 0)
            status, self.x, self.lam, self.nu, self.mu, self.pi = self.read_solution(output_filename, self.problem)
        except Exception as e:
            raise OptSolverError_CplexCMDCall(self)
        finally:
            if os.path.isfile(input_filename) and not debug:
                os.remove(input_filename)
            if os.path.isfile(output_filename) and not debug:
                os.remove(output_filename)
            if os.path.isfile('cplex.log') and not debug:
                os.remove('cplex.log')
            for i in range(cpu_count()):
                if os.path.isfile('clone%d.log' %i) and not debug:
                    os.remove('clone%d.log' %i)

        if 'optimal' in status.lower():
            self.set_status(self.STATUS_SOLVED)
            self.set_error_msg('')
        else:
            raise OptSolverError_CplexCMD(self)
