#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2016, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

from __future__ import print_function
import time
import numpy as np
from .stoch_solver import StochSolver

class StochGradientPD(StochSolver):

    parameters = {'maxiters': 1000,
                  'maxtime': 600,
                  'period': 60,
                  'quiet' : True,
                  'theta_lam': 1.,
                  'theta_x': 1.,
                  'k0': 0,
                  'no_G': False}

    name = 'Primal-Dual Stochastic Gradient'

    def __init__(self):
        """
        Primal-Dual Stochastic Gradient Algorithm.
        """
        
        # Init
        StochSolver.__init__(self)
        self.parameters = StochGradientPD.parameters.copy()

        self.results = []

    def solve(self,problem):

        # Local vars
        params = self.parameters

        # Parameters
        maxiters = params['maxiters']
        maxtime = params['maxtime']
        period = params['period']
        quiet = params['quiet']
        theta_lam = params['theta_lam']
        theta_x = params['theta_x']
        k0 = params['k0']
        no_G = params['no_G']

        # Header
        if not quiet:
            print('\nPrimal-Dual Stochastic Gradient')
            print('-------------------------------')
            print('{0:^8s}'.format('iter'), end=' ')
            print('{0:^10s}'.format('time(s)'), end=' ')
            print('{0:^10s}'.format('alpha'), end=' ')
            print('{0:^12s}'.format('prop'), end=' ')
            print('{0:^12s}'.format('lmax'), end=' ')
            print('{0:^12s}'.format('EF_run'), end=' ')
            print('{0:^12s}'.format('EGmax_run'), end=' ')
            print('{0:^12s}'.format('saved'))

        # Init
        k = 0
        t1 = 0
        t0 = time.time()
        self.x = problem.get_init_x()
        lam = np.zeros(problem.get_size_lam())
        self.results = []
        
        # Loop
        while True:

            # Steplength
            alpha_lam = theta_lam/(k0+k+1.)
            alpha_x = theta_x/(k0+k+1.)
             
            # Save
            if time.time()-t0 > t1:
                self.results.append((k,time.time()-t0,self.x,np.max(lam)))
                t1 += period

            # Iters
            if k >= maxiters:
                break
                
            # Maxtime
            if time.time()-t0 >= maxtime:
                break
            
            # Sample
            w = problem.sample_w()
            
            # Eval
            F,gF,G,JG = problem.eval_FG(self.x,w)
            
            # Lagrangian subgradient
            gL = gF + JG.T*lam

            # Running
            if k == 0:
                EF_run = F
                EG_run = G.copy()
            else:
                EF_run += alpha_x*(F-EF_run)
                EG_run += alpha_lam*(G-EG_run)
            
            # Show progress
            if not quiet:
                print('{0:^8d}'.format(k), end=' ')
                print('{0:^10.2f}'.format(time.time()-t0), end=' ')
                print('{0:^10.2e}'.format(np.maximum(alpha_x,alpha_lam)), end=' ')
                print('{0:^12.5e}'.format(problem.get_prop_x(self.x)), end=' ')
                print('{0:^12.5e}'.format(np.max(lam)), end=' ')
                print('{0:^12.5e}'.format(EF_run), end=' ')
                print('{0:^12.5e}'.format(np.max(EG_run)), end=' ')
                print('{0:^12d}'.format(len(self.results)))
            
            # Update
            self.x = problem.project_x(self.x - alpha_x*gL)
            if not no_G:
                lam = problem.project_lam(lam + alpha_lam*G)
            k += 1

    def get_results(self):

        return self.results
        
    
            
