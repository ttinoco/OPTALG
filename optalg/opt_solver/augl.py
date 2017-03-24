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
    
    parameters = {'beta_large' : 0.9,       # for decreasing sigma when progress
                  'beta_med' : 0.5,         # for decreasing sigma when forcing
                  'beta_small' : 0.1,       # for decreasing sigma
                  'feastol' : 1e-5,         # feasibility tolerance
                  'optol' : 1e-5,           # optimality tolerance
                  'gamma' : 0.1,            # for determining required decrease in ||f||
                  'tau' : 0.1,              # for reductions in ||GradF||
                  'kappa' : 1e-2,           # for initializing sigma
                  'maxiter' : 1000,         # maximum iterations
                  'sigma_min' : 1e-12,      # minimum sigma
                  'sigma_init_min' : 1e-5,  # minimum initial sigma
                  'sigma_init_max' : 1e5,   # maximum initial sigma
                  'theta_min'      : 1e-5,  # minimum barrier parameter
                  'theta_init_min' : 1e-5,  # minimum initial barrier parameter
                  'theta_init_max' : 1e0 ,  # maximum initial barrier parameter
                  'lam_reg' : 1e-2,         # eta/sigma ratio for regularization of first order dual update
                  'subprob_force' : 10,     # for periodic sigma decrease
                  'subprob_maxiter' : 150,  # maximum subproblem iterations
                  'linsolver' : 'default',  # linear solver
                  'quiet' : False}          # flag for omitting output
    
    def __init__(self):
        """
        Augmented Lagrangian algorithm.
        """
        
        OptSolver.__init__(self)
        self.parameters = OptSolverAugL.parameters.copy()
        self.linsolver1 = None 
        self.linsolver2 = None
        self.problem = None
        self.barrier = None

    def solve(self,problem):
        
        # Local vars
        norm2 = self.norm2
        norminf = self.norminf
        params = self.parameters
        
        # Parameters
        tau = params['tau']
        gamma = params['gamma']
        kappa = params['kappa']
        optol = params['optol']
        feastol = params['feastol']
        beta_small = params['beta_small']
        beta_large = params['beta_large']
        sigma_init_min = params['sigma_init_min']
        sigma_init_max = params['sigma_init_max']
        theta_init_min = params['theta_init_min']
        theta_init_max = params['theta_init_max']
        theta_min = params['theta_min']

        # Linear solver
        self.linsolver1 = new_linsolver(params['linsolver'],'symmetric')
        self.linsolver2 = new_linsolver(params['linsolver'],'symmetric')

        # Problem
        self.problem = problem

        # Reset
        self.reset()
        
        # Barrier
        self.barrier = AugLBarrier(problem.get_num_primal_variables(),
                                   problem.l,
                                   problem.u,
                                   eps=feastol)
        
        # Init primal
        if problem.x is not None:
            self.x = self.barrier.to_interior(problem.x.copy())
        else:
            self.x = (self.barrier.umax+self.barrier.umin)/2.
        assert(np.all(self.x > self.barrier.umin))
        assert(np.all(self.x < self.barrier.umax))
            
        # Init dual
        if problem.lam is not None:
            self.lam = problem.lam.copy()
        else:
            self.lam = np.zeros(problem.b.size)
        if problem.nu is not None:
                self.nu = problem.nu.copy()
        else:
            self.nu = np.zeros(problem.f.size)
        try:
            if problem.pi is not None:
                self.pi = problem.pi.copy()
            else:
                self.pi = np.zeros(self.x.size)
        except AttributeError:
            self.pi = np.zeros(self.x.size)
        try: 
            if problem.mu is not None:
                self.mu = problem.mu.copy()
            else:
                self.mu = np.zeros(self.x.size)
        except AttributeError:
            self.mu = np.zeros(self.x.size)
        
        # Constants
        self.sigma = 0.
        self.theta = 0.
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
        
        # Init penalty and barrier parameters
        self.sigma = kappa*norm2(fdata.GradF)/np.maximum(norm2(problem.gphi),1.)
        self.sigma = np.minimum(np.maximum(self.sigma,sigma_init_min),sigma_init_max)
        self.theta = kappa*norm2(fdata.GradF)/(self.sigma*np.maximum(norm2(self.barrier.gphi),1.))
        self.theta = np.minimum(np.maximum(self.theta,theta_init_min),theta_init_max)
        fdata = self.func(self.x)
        
        # Outer iterations
        self.k = 0
        self.useH = False
        self.code = list('----')
        pres_prev = norminf(fdata.pres)
        gLmax_prev = norminf(fdata.GradF)
        while True:
            
            # Solve subproblem
            self.solve_subproblem(tau*gLmax_prev)

            # Check done
            if self.is_status_solved():
                return
                
            # Measure progress
            pres = norminf(fdata.pres)
            dres = norminf(fdata.dres)
            gLmax = norminf(fdata.GradF)
            
            # Penaly update
            if pres <= np.maximum(gamma*pres_prev,feastol):
                self.sigma *= beta_large
                self.code[1] = 'p'
            else:
                self.sigma *= beta_small
                self.code[1] = 'n'

            # Dual update
            self.update_multiplier_estimates()

            # Barrier update
            self.theta = np.maximum(self.theta*beta_small,theta_min)

            # Update refs
            pres_prev = pres
            gLmax_prev = gLmax

    def solve_subproblem(self,delta):
        
        # Local vars
        norm2 = self.norm2
        norminf = self.norminf
        params = self.parameters
        problem = self.problem
        barrier = self.barrier
        
        # Params
        quiet = params['quiet']
        maxiter = params['maxiter']
        feastol = params['feastol']
        optol = params['optol']
        maxiter = params['maxiter']
        theta_min = params['theta_min']
        sigma_min = params['sigma_min']
        beta_large = params['beta_large']
        beta_med = params['beta_med']
        beta_small = params['beta_small']
        subprob_force = params['subprob_force']
        subprob_maxiter = params['subprob_maxiter']
        
        # Print header
        self.print_header()

        # Init eval
        fdata = self.func(self.x)
        
        # Inner iterations
        i = 0
        j = 0
        alpha = 0.
        while True:
            
            # Compute info
            pres = norminf(fdata.pres)
            dres = norminf(fdata.dres)
            dmax = max(map(norminf,[self.lam,self.nu,self.mu,self.pi]))
            gLmax = norminf(fdata.GradF)
            
            # Show info
            if not quiet:
                print('{0:^4d}'.format(self.k), end=' ')
                print('{0:^9.2e}'.format(problem.phi), end=' ')
                print('{0:^9.2e}'.format(pres), end=' ')
                print('{0:^9.2e}'.format(dres), end=' ')
                print('{0:^9.2e}'.format(gLmax), end=' ')
                print('{0:^8.1e}'.format(dmax), end=' ')
                print('{0:^8.1e}'.format(alpha), end=' ')
                print('{0:^7.1e}'.format(self.sigma), end=' ')
                print('{0:^7.1e}'.format(self.theta), end=' ')
                print('{0:^8s}'.format(reduce(lambda x,y: x+y,self.code)), end=' ')
                if self.info_printer:
                    self.info_printer(self,False)
                else:
                    print('')

            # Clear code
            self.code = list('----')

            # Check solved
            if pres <= feastol and dres <= optol and self.theta <= theta_min:
                self.set_status(self.STATUS_SOLVED)
                self.set_error_msg('')
                return

            # Check only theta missing
            if pres <= feastol and dres <= optol:
                return
                
            # Check subproblem solved
            if gLmax <= delta:
                return
                
            # Check total maxiters
            if self.k >= maxiter:
                raise OptSolverError_MaxIters(self)

            # Check penalty
            if self.sigma < sigma_min:
                raise OptSolverError_SmallPenalty(self)
                
            # Check custom terminations
            for t in self.terminations:
                t(self)
                
            # Search direction
            p = self.compute_search_direction(self.useH)

            # Max steplength
            ppos = p > 0
            pneg = p < 0
            a1 = np.min(((barrier.umax-self.x)[ppos])/(p[ppos])) if ppos.sum() else np.inf
            a2 = np.min(((barrier.umin-self.x)[pneg])/(p[pneg])) if pneg.sum() else np.inf
            alpha_max = 0.99*min([a1,a2])
            
            try:

                # Line search
                alpha,fdata = self.line_search(self.x,p,fdata.F,fdata.GradF,self.func,alpha_max)
                
                # Update x
                self.x += alpha*p

            except OptSolverError_LineSearch:

                # Update 
                self.sigma *= beta_large
                fdata = self.func(self.x)
                self.code[3] = 'b'
                if self.useH:
                    self.useH = False
                    i = 0
                alpha = 0.

            # Update iter count
            self.k += 1
            i += 1
            j += 1
            
            # Periodic force
            if i >= subprob_force:
                self.sigma *= beta_large
                fdata = self.func(self.x)
                self.code[2] = 'f'
                self.useH = True
                i = 0

            # Periodic maxiter
            if j >= subprob_maxiter:
                self.sigma *= beta_med
                self.update_multiplier_estimates()
                fdata = self.func(self.x)
                self.code[2] = 'm'
                j = 0 

    def compute_search_direction(self,useH):
        
        fdata = self.fdata
        problem = self.problem
        barrier = self.barrier
        
        sigma = self.sigma
        theta = self.theta

        problem.combine_H(-sigma*self.nu+problem.f,not useH)
        self.code[0] = 'h' if useH else 'g'

        Hfsigma = problem.H_combined/sigma
        Hphi = problem.Hphi
        HphiB = barrier.Hphi
        G = coo_matrix((np.concatenate((Hphi.data,theta*HphiB.data,Hfsigma.data)),
                        (np.concatenate((Hphi.row,HphiB.row,Hfsigma.row)),
                         np.concatenate((Hphi.col,HphiB.col,Hfsigma.col)))))

        if problem.A.size:
            W = bmat([[G,None,None],
                      [problem.J,-sigma*self.Iff,None],
                      [problem.A,None,-sigma*self.Iaa]])
        else:
            W = bmat([[G,None],
                      [problem.J,-sigma*self.Iff]])
        b = np.hstack((-fdata.GradF/sigma,
                       self.of,
                       self.oa))

        if not self.linsolver1.is_analyzed():
            self.linsolver1.analyze(W)

        return self.linsolver1.factorize_and_solve(W,b)[:self.x.size]
        
    def func(self,x):
        
        # Norm
        norm = self.norminf

        # Multipliers
        lam = self.lam    
        nu = self.nu

        # Penalty
        sigma = self.sigma
        theta = self.theta

        # Objects
        p = self.problem
        fdata = self.fdata
        barrier = self.barrier

        # Eval
        p.eval(x)
        barrier.eval(x)
        
        # Problem data
        phi = p.phi
        gphi = p.gphi
        f = p.f
        J = p.J
        A = p.A
        r = A*x-p.b
           
        # Barrier data
        phiB = barrier.phi
        gphiB = barrier.gphi
 
        # Intermediate
        nuTf = np.dot(nu,f)
        y = (sigma*nu-f)
        JT = J.T
        JTnu = JT*nu
        JTy = JT*y
        
        # Intermediate
        lamTr = np.dot(lam,r)
        z = (sigma*lam-r)
        AT = A.T
        ATlam = AT*lam
        ATz = AT*z
        
        pres = np.hstack((r,f))
        dres = gphi+theta*gphiB-ATlam-JTnu
        dres_den = 1.+norm(gphi)+theta*norm(gphiB)+norm(A.data)*norm(lam)+norm(J.data)*norm(nu)
               
        fdata.ATlam = ATlam
        fdata.JTnu = JTnu
 
        fdata.r = r
        fdata.f = f
        
        fdata.F = sigma*phi + sigma*theta*phiB - sigma*(nuTf+lamTr) + 0.5*np.dot(pres,pres)
        fdata.GradF = sigma*gphi + sigma*theta*gphiB - JTy - ATz
        
        fdata.pres = pres
        fdata.dres = dres/dres_den
        
        return fdata

    def print_header(self):
        
        # Local vars
        params = self.parameters
        quiet = params['quiet']

        if not quiet:
            if self.k == 0:
                print('\nSolver: augL')
                print('------------')
                print('{0:^4}'.format('k'), end=' ')
                print('{0:^9}'.format('phi'), end=' ')
                print('{0:^9}'.format('pres'), end=' ')
                print('{0:^9}'.format('dres'), end=' ')
                print('{0:^9}'.format('gLmax'), end=' ')
                print('{0:^8}'.format('dmax'), end=' ')
                print('{0:^8}'.format('alpha'), end=' ')
                print('{0:^7}'.format('sigma'), end=' ')
                print('{0:^7}'.format('theta'), end=' ')
                print('{0:^8}'.format('code'), end=' ')
                if self.info_printer:
                    self.info_printer(self,True)
                else:
                    print('')
            else:
                print('')
                
    def update_multiplier_estimates(self):

        # Local variables
        params = self.parameters
        problem = self.problem
        barrier = self.barrier
        fdata = self.fdata
        
        # Parameters
        lam_reg = params['lam_reg']
        sigma = self.sigma
        theta = self.theta
        eta = lam_reg*sigma

        # Eval
        fdata = self.func(self.x)

        A = problem.A
        J = problem.J
        AT = A.T
        JT = J.T

        t = problem.gphi+theta*barrier.gphi-fdata.ATlam-fdata.JTnu
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
        self.mu = theta/(barrier.umax-self.x)
        self.pi = theta/(self.x-barrier.umin)

class AugLBarrier:
    """
    Class for handling bounds using barrier.
    """

    def __init__(self,n,umin=None,umax=None,eps=1e-4,inf=1e8):
        
        assert(n >= 0)
        assert(inf > 0)
        assert(eps > 0)

        if umin is None or not umin.size:
            umin = -inf*np.ones(n)
        if umax is None or not umax.size:
            umax = inf*np.ones(n)

        assert(np.all(umin <= umax))

        if n > 0:
            umax += eps*np.linalg.norm(umax,np.inf)
            umin -= eps*np.linalg.norm(umin,np.inf)

        assert(umin.size == n)
        assert(umin.size == umax.size)
        assert(np.all(umin < umax))
        
        self.n = n
        self.eps = eps
        self.inf = inf
        self.umin = umin
        self.umax = umax

        self.phi = 0
        self.gphi = np.zeros(n)
        self.Hphi_row = np.array(range(n))
        self.Hphi_col = np.array(range(n))
        self.Hphi_data = np.zeros(n)
        self.Hphi = coo_matrix((self.Hphi_data,(self.Hphi_row,self.Hphi_col)),shape=(n,n))

    def eval(self,u):

        assert(u.size == self.n)

        dumax = np.maximum(self.umax-u,1e-12)
        dumin = np.maximum(u-self.umin,1e-12)

        self.phi = -np.sum(np.log(dumax)+np.log(dumin))
        self.gphi[:] = -1./dumin+1./dumax
        self.Hphi_data[:] = 1./np.square(dumin)+1./np.square(dumax)

    def to_interior(self,x):
        
        du = self.umax-self.umin
        return np.maximum(np.minimum(x,self.umax-0.01*du),self.umin+0.01*du)

class AugLBounds:
    """
    Class for handling bounds as simple nonlinear constraints.
    """
 
    def __init__(self,n,umin=None,umax=None,eps=1e-4,inf=1e8):
        """
        Class for handling bounds as simple nonlinear constraints.

        Parameters
        ----------
        n : int
        umin : ndarray
        umax : ndarray
        eps : float
        inf : float
        """
 
        assert(eps > 0.)
        assert(inf > 0)
        assert(n >= 0)

        if umin is None or not umin.size:
            umin = -inf*np.ones(n)
        if umax is None or not umax.size:
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

        assert(u.size == self.n)

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
        assert(coeff.size == 2*n)
        coeff1 = coeff[:n]
        coeff2 = coeff[n:]

        if ensure_psd:
            self.Hcomb_data[:] = np.zeros(n)
        else:
            self.Hcomb_data[:] = coeff1*self.Hdata[:n] + coeff2*self.Hdata[n:]
        
