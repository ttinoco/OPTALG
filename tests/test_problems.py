#*****************************************************#
# This file is part of OPTALG.                        #
#                                                     #
# Copyright (c) 2019, Tomas Tinoco De Rubira.         #
#                                                     #
# OPTALG is released under the BSD 2-clause license.  #
#*****************************************************#

import os
import unittest
import numpy as np
import optalg as opt

class TestProblems(unittest.TestCase):

    def test_lin_to_lp_file(self):

        A = np.array([[-2.,2. ,1.,0.],
                      [-8.,10.,0.,1.]])
        b = np.array([1.,13.])
        
        l = np.array([-1e8,-1e8,-1e8,0.])
        u = np.array([1e8,1e8,0.,1e8])
        
        c = np.array([-1.,-1.,0.,0.])
                
        problem = opt.opt_solver.LinProblem(c,A,b,l,u)

        try:
            
            problem.write_to_lp_file('foo.lp')

        finally:

            if os.path.isfile('foo.lp'):
                os.remove('foo.lp')
   
    def test_mixintlin_to_lp_file(self):

        A = np.array([[-2.,2. ,1.,0.],
                      [-8.,10.,0.,1.]])
        b = np.array([1.,13.])
        
        l = np.array([-1e8,-1e8,-1e8,0.])
        u = np.array([1e8,1e8,0.,1e8])
        
        c = np.array([-1.,-1.,0.,0.])
        
        P = np.array([True,True,False,False])
        
        problem = opt.opt_solver.MixIntLinProblem(c,A,b,l,u,P)

        try:
            
            problem.write_to_lp_file('foo.lp')

        finally:

            if os.path.isfile('foo.lp'):
                os.remove('foo.lp')
        
