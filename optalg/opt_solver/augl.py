#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

from __future__ import print_function
import numpy as np
from .opt_solver_error import *
from .opt_solver import OptSolver
from scipy.sparse import bmat,eye,coo_matrix,tril
from optalg.lin_solver import new_linsolver
from functools import reduce

class OptSolverAugL(OptSolver):
    
    parameters = {'beta_large':0.9,      # for decreasing penalty when progress
                  'beta_med':0.5,        # for decreasing miu
                  'beta_small':0.1,      # for decreasing miu
                  'feastol':1e-4,        # feasibility tolerance
                  'optol':1e-4,          # optimality tolerance
                  'subtol':1e-5,         # for solving subproblem
                  'gamma':0.1,           # for determining required decrease in ||f||
                  'tau':0.1,             # for reductions in ||GradF||
                  'kappa':1e-4,          # for initializing miu
                  'maxiter':300,         # maximum iterations
                  'miu_min':1e-12,       # lowest mi
                  'miu_init_min':1e-6,   # lowest initial miu
                  'miu_init_max':1e6,    # largest initial miu
                  'lam_reg':1e-2,        # eta/miu ratio for regularization of first order dual update
                  'subprob_force':10,    # for periodic miu decrease
                  'linsolver':'default', # linear solver
                  'quiet':False}         # flag for omitting output
    
    def __init__(self):
        """
        Augmented Lagrangian algorithm.
        """
        
        OptSolver.__init__(self)
        self.parameters = OptSolverAugL.parameters.copy()
        self.linsolver1 = None 
        self.linsolver2 = None
        self.problem = None

    def compute_search_direction(self,useH):

        problem = self.problem
        fdata = self.fdata

        miu = np.maximum(self.miu,1e-8)
        
        if useH:
            problem.combine_H(-miu*self.nu+problem.f,False) # exact Hessian
            self.code[0] = 'h'
        else:
            problem.combine_H(-miu*self.nu+problem.f,True) # ensure pos semidef
            self.code[0] = 'g'

        Hfmiu = problem.H_combined/miu
        Hphi = problem.Hphi
        G = coo_matrix((np.concatenate((Hphi.data,Hfmiu.data)),
                        (np.concatenate((Hphi.row,Hfmiu.row)),
                         np.concatenate((Hphi.col,Hfmiu.col)))))

        if problem.A.size:
            W = bmat([[G,None,None],
                      [problem.J,-miu*self.Iff,None],
                      [problem.A,None,-miu*self.Iaa]])
        else:
            W = bmat([[G,None],
                      [problem.J,-miu*self.Iff]])
        b = np.hstack((-fdata.GradF/self.miu,
                       self.of,
                       self.oa))

        if not self.linsolver1.is_analyzed():
            self.linsolver1.analyze(W)

        return self.linsolver1.factorize_and_solve(W,b)[:self.x.size]
        
    def func(self,x):
         
        lam = self.lam      # linear eq
        nu = self.nu        # nonlinear eq
        miu = self.miu      # need to change to sigma
        fdata = self.fdata
        p = self.problem
        
        p.eval(x)
        
        phi = p.phi
        gphi = p.gphi
        f = p.f
        J = p.J
        A = p.A
        r = A*x-p.b
            
        nuTf = np.dot(nu,f)
        y = (miu*nu-f)
        JT = J.T
        JTnu = JT*nu
        JTy = JT*y

        lamTr = np.dot(lam,r)
        z = (miu*lam-r)
        AT = A.T
        ATlam = AT*lam
        ATz = AT*z

        pres = np.hstack((r,f))
        dres = gphi-ATlam-JTnu
        
        fdata.phi = phi
        fdata.gphi = gphi
       
        fdata.ATlam = ATlam
        fdata.JTnu = JTnu
 
        fdata.f = f
        fdata.r = r
        
        fdata.F = miu*phi - miu*(nuTf+lamTr) + 0.5*np.dot(pres,pres)
        fdata.GradF = miu*gphi - JTy - ATz
        
        fdata.pres = pres
        fdata.dres = dres
        
        return fdata        

    def print_header(self):
        
        # Local vars
        params = self.parameters
        quiet = params['quiet']

        if not quiet:
            if self.k == 0:
                print('\nSolver: augL')
                print('------------')
                print('{0:^3}'.format('k'), end=' ')
                print('{0:^9}'.format('phi'), end=' ')
                print('{0:^9}'.format('pres'), end=' ')
                print('{0:^9}'.format('dres'), end=' ')
                print('{0:^9}'.format('gLmax'), end=' ')
                print('{0:^8}'.format('dmax'), end=' ')
                print('{0:^8}'.format('alpha'), end=' ')
                print('{0:^7}'.format('miu'), end=' ')
                print('{0:^8}'.format('code'), end=' ')
                if self.info_printer:
                    self.info_printer(self,True)
                else:
                    print('')
            else:
                print('')

    def solve(self,problem):
        
        # Local vars
        norm2 = self.norm2
        norminf = self.norminf
        params = self.parameters
        
        # Parameters
        tau = params['tau']
        gamma = params['gamma']
        kappa = params['kappa']
        subtol = params['subtol']
        feastol = params['feastol']
        beta_small = params['beta_small']
        beta_large = params['beta_large']
        miu_init_min = params['miu_init_min']
        miu_init_max = params['miu_init_max']

        # Linear solver
        self.linsolver1 = new_linsolver(params['linsolver'],'symmetric')
        self.linsolver2 = new_linsolver(params['linsolver'],'symmetric')

        # Problem
        self.problem = problem

        # Reset
        self.reset()
                
        # Init primal
        if problem.x is not None:
            self.x = problem.x.copy()
        else:
            self.x = np.zeros(problem.get_num_primal_variables())
            
        # Init dual
        if problem.lam is not None:
            self.lam = problem.lam.copy()
        else:
            self.lam = np.zeros(problem.b.size)
        if problem.nu is not None:
            self.nu = problem.nu.copy()
        else:
            self.nu = np.zeros(problem.f.size)
        
        # Constants
        self.miu = 1.
        self.code = ''
        self.nx = self.x.size
        self.na = problem.b.size
        self.nf = problem.f.size
        self.ox = np.zeros(self.nx)
        self.oa = np.zeros(self.na)
        self.of = np.zeros(self.nf)
        self.Ixx = eye(self.nx,format='coo')
        self.Iff = eye(self.nf,format='coo')
        self.Iaa = eye(self.na,format='coo')
        
        # Init eval
        fdata = self.func(self.x)
                    
        # Init penalty parameter
        self.miu = 0.5*kappa*np.dot(fdata.pres,fdata.pres)/np.maximum(np.abs(fdata.phi),1.)
        self.miu = np.minimum(np.maximum(self.miu,miu_init_min),miu_init_max)
        fdata = self.func(self.x)
        
        # Outer iterations
        self.k = 0
        self.useH = False
        self.code = list('----')
        pres_prev = norminf(fdata.pres)
        gLmax_prev = norminf(fdata.GradF)
        while True:
                
            # Solve subproblem
            self.solve_subproblem(np.maximum(tau*gLmax_prev,subtol))

            # Check done
            if self.is_status_solved():
                return
                
            # Measure progress
            pres = norminf(fdata.pres)
            gLmax = norminf(fdata.GradF)
            
            # Penaly update
            if pres < np.maximum(gamma*pres_prev,feastol):
                self.miu *= beta_large
                self.code[1] = 'p'
            else:
                self.miu *= beta_small
                self.code[1] = 'n'

            # Dual update
            self.update_multiplier_estimates()

            # Update refs
            pres_prev = pres
            gLmax_prev = gLmax
            
    def solve_subproblem(self,delta):
        
        # Local vars
        norm2 = self.norm2
        norminf = self.norminf
        params = self.parameters
        problem = self.problem
        
        # Params
        quiet = params['quiet']
        maxiter = params['maxiter']
        feastol = params['feastol']
        optol = params['optol']
        maxiter = params['maxiter']
        miu_min = params['miu_min']
        beta_med = params['beta_med']
        beta_large = params['beta_large']
        beta_small = params['beta_small']
        subprob_force = params['subprob_force']
        
        # Print header
        self.print_header()

        # Init eval
        fdata = self.func(self.x)
        
        # Inner iterations
        i = 0;
        alpha = 0.;
        while True:
            
            # Compute info
            pres = norminf(fdata.pres)
            dres = norm2(fdata.dres)/np.maximum(norm2(fdata.gphi)+norm2(fdata.ATlam)+norm2(fdata.JTnu),1.)
            dmax = np.maximum(norminf(self.lam),norminf(self.nu))
            gLmax = norminf(fdata.GradF)
            
            # Show info
            if not quiet:
                print('{0:^3d}'.format(self.k), end=' ')
                print('{0:^9.2e}'.format(fdata.phi), end=' ')
                print('{0:^9.2e}'.format(pres), end=' ')
                print('{0:^9.2e}'.format(dres), end=' ')
                print('{0:^9.2e}'.format(gLmax), end=' ')
                print('{0:^8.1e}'.format(dmax), end=' ')
                print('{0:^8.1e}'.format(alpha), end=' ')
                print('{0:^7.1e}'.format(self.miu), end=' ')
                print('{0:^8s}'.format(reduce(lambda x,y: x+y,self.code)), end=' ')
                if self.info_printer:
                    self.info_printer(self,False)
                else:
                    print('')

            # Clear code
            self.code = list('----')

            # Check solved
            if pres < feastol and dres < optol:
                self.set_status(self.STATUS_SOLVED)
                self.set_error_msg('')
                return

            # Check feasibility
            if pres < feastol and i != 0:
                return
                
            # Check subproblem solved
            if gLmax < delta and i != 0:
                return

            # Check total maxiters
            if self.k >= maxiter:
                raise OptSolverError_MaxIters(self)

            # Check penalty
            if self.miu < miu_min:
                raise OptSolverError_SmallPenalty(self)
            
            # Check custom terminations
            for t in self.terminations:
                t(self)
        
            # Search direction
            p = self.compute_search_direction(self.useH)
            
            try:

                # Line search
                alpha,fdata = self.line_search(self.x,p,fdata.F,fdata.GradF,self.func,np.inf)
                
                # Update x
                self.x += alpha*p

            except OptSolverError_LineSearch:

                # Update 
                self.miu *= beta_large
                fdata = self.func(self.x)
                self.code[3] = 'b'
                self.useH = False
                i = 0
                alpha = 0.

            # Update iter count
            self.k += 1
            i += 1
            
            # Clear H flag
            tryH = False
            
            # Maxiter
            if i >= subprob_force:
                self.miu *= beta_med
                self.update_multiplier_estimates()
                fdata = self.func(self.x)
                self.code[2] = 'f'
                self.useH = True
                i = 0
    
    def update_multiplier_estimates(self):

        # Local variables
        problem = self.problem
        params = self.parameters
        fdata = self.fdata
        
        # Parameters
        lam_reg = params['lam_reg']

        eta = np.maximum(lam_reg*self.miu,1e-8)

        A = problem.A
        J = problem.J
        AT = A.T
        JT = J.T

        t = fdata.gphi-fdata.ATlam-fdata.JTnu
        if problem.A.size:
            W = bmat([[eta*self.Iaa,None,None],
                      [None,eta*self.Iff,None],
                      [AT,JT,-self.Ixx]],format='coo')
        else:
            W = bmat([[eta*self.Iff,None],
                      [JT,-self.Ixx]],format='coo')
        b = np.hstack((A*t,
                       J*t,
                       self.ox))

        if not self.linsolver2.is_analyzed():
            self.linsolver2.analyze(W)

        sol = self.linsolver2.factorize_and_solve(W,b)
        
        self.lam += sol[:self.na]
        self.nu += sol[self.na:self.na+self.nf]

class AugLBounds:
    """
    Class for handling bounds as simple nonlinear constraints.
    """
 
    def __init__(self,n,umin=None,umax=None,eps=1e-4,inf=1e8):
 
        assert(eps > 0.)
        assert(inf > 0)

        if umin is None:
            umin = -inf*np.ones(n)
        if umax is None:
            umax = inf*np.ones(n)

        assert(umin.size == n)
        assert(umin.size == umax.size)
        assert(np.all(umin <= umax))

        self.n = n
        self.eps = eps
        self.inf = inf
        self.umin = umin
        self.umax = umax
        self.du = np.maximum(umax-umin,eps)

        self.f = np.zeros(2*self.n)

        self.Jrow = np.array(range(2*self.n))
        self.Jcol = np.concatenate((range(self.n),range(self.n)))
        self.Jdata = np.zeros(2*self.n)
        self.J = coo_matrix((self.Jdata,(self.Jrow,self.Jcol)),
                                     shape=(2*self.n,self.n))
        
        self.Hdata = np.zeros(2*self.n)
        self.Hcomb_row = np.array(range(self.n))
        self.Hcomb_col = np.array(range(self.n))
        self.Hcomb_data = np.zeros(self.n)

        self.H_combined = coo_matrix((self.Hcomb_data,(self.Hcomb_row,self.Hcomb_col)),
                                     shape=(self.n,self.n))

    def eval(self,u):

        n = self.n
        eps = self.eps
        a1 = self.umax-u
        a2 = u-self.umin
        b = eps*eps/self.du
        sqrterm1 = np.sqrt(a1*a1+b*b+eps*eps)
        sqrterm2 = np.sqrt(a2*a2+b*b+eps*eps)
        
        self.f[:n] = a1 + b - sqrterm1
        self.f[n:] = a2 + b - sqrterm2

        self.Jdata[:n] = -(1-a1/sqrterm1)
        self.Jdata[n:] = (1-a2/sqrterm2)
        
        self.Hdata[:n] = -(b*b+eps*eps)/(sqrterm1*sqrterm1*sqrterm1)
        self.Hdata[n:] = -(b*b+eps*eps)/(sqrterm2*sqrterm2*sqrterm2)
        
    def combine_H(self,coeff,ensure_psd=False):

        n = self.n
        assert(coeff.size == 2*self.n)
        coeff1 = coeff[:n]
        coeff2 = coeff[n:]

        self.Hcomb_data[:] = coeff1*self.Hdata[:n] + coeff2*self.Hdata[n:]
        
