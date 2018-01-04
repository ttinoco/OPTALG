#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np
cimport numpy as np

from libc.string cimport memcpy

cimport cipopt

np.import_array()

cdef extern from "numpy/arrayobject.h":
     void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
     void PyArray_CLEARFLAGS(np.ndarray arr, int flags)

cdef ArrayDouble(double* a, int size):
     cdef np.npy_intp shape[1]
     shape[0] = <np.npy_intp> size
     arr = np.PyArray_SimpleNewFromData(1,shape,np.NPY_DOUBLE,a)
     PyArray_CLEARFLAGS(arr,np.NPY_OWNDATA)
     return arr

class IpoptContextError(Exception):
    """
    IPOPT context error exception.
    """
    pass

cdef class IpoptContext:
    """
    IPOPT context class.
    """

    cdef int iters
    cdef int n
    cdef int m
    cdef int nnzj
    cdef int nnzh
    cdef object l
    cdef object u
    cdef object gl
    cdef object gu
    cdef object eval_f
    cdef object eval_g
    cdef object eval_grad_f
    cdef object eval_jac_g
    cdef object eval_h
    cdef cipopt.IpoptProblem problem
    
    def __init__(self,n,m,l,u,gl,gu,eval_f,eval_g,eval_grad_f,eval_jac_g,eval_h):

        self.iters = 0
        self.n = n
        self.m = m
        self.l = l.copy()
        self.u = u.copy()
        self.gl = gl.copy()
        self.gu = gu.copy()
        self.eval_f = eval_f
        self.eval_g = eval_g
        self.eval_grad_f = eval_grad_f
        self.eval_jac_g = eval_jac_g
        self.eval_h = eval_h

        self.problem = NULL
        
        Jrow,Jcol = eval_jac_g(None,True) # x, flag        
        self.nnzj = Jrow.size

        Hrow,Hcol = eval_h(None,None,None,True) # x, lam, obj_factor, flag
        self.nnzh = Hrow.size
        
        try:
            assert(l.size == n)
            assert(u.size == n)
            assert(gl.size == m)
            assert(gu.size == m)
            assert(Jrow.size == Jcol.size)
            assert(Hrow.size == Hcol.size)
        except AssertionError:
            raise IpoptContextError('invalid data dimensions')

        self.create_problem()

    def __dealloc__(self):

        if self.problem != NULL:
            cipopt.FreeIpoptProblem(<cipopt.IpoptProblem>self.problem)
        self.problem = NULL

    def add_option(self,key,val):

        if not self.n:
            return

        try:
            ckey = bytes(key,'utf-8')
        except TypeError:
            ckey = bytes(key)

        if isinstance(val,int):
            valid = cipopt.AddIpoptIntOption(self.problem,ckey,val)
        elif isinstance(val,str):
            try:
                cval = bytes(val,'utf-8')
            except TypeError:
                cval = bytes(val)
            valid = cipopt.AddIpoptStrOption(self.problem,ckey,cval)
        elif isinstance(val,float):
            valid = cipopt.AddIpoptNumOption(self.problem,ckey,val)
        else:
            raise ValueError('invalid value')

        if not valid:
            raise IpoptContextError('option %s could not be set' %key)
 
    def create_problem(self):
        
        cdef np.ndarray[double,mode='c'] nl = self.l
        cdef np.ndarray[double,mode='c'] nu = self.u
        cdef np.ndarray[double,mode='c'] ngl = self.gl
        cdef np.ndarray[double,mode='c'] ngu = self.gu
        
        self.problem = cipopt.CreateIpoptProblem(self.n,
                                                 <double*>(nl.data),
                                                 <double*>(nu.data),
                                                 self.m, 
                                                 <double*>(ngl.data),
                                                 <double*>(ngu.data), 
                                                 self.nnzj,
                                                 self.nnzh, 
                                                 0, 
                                                 eval_f_cb,
                                                 eval_g_cb,
                                                 eval_grad_f_cb,
                                                 eval_jac_g_cb,
                                                 eval_h_cb)

        if self.n:
            cipopt.SetIntermediateCallback(self.problem,intermediate_cb)
    
    def solve(self,x):

        cdef UserDataPtr cself = <UserDataPtr>self
        cdef np.ndarray[double,mode='c'] nx = x.copy()
        cdef np.ndarray[double,mode='c'] nlam = np.zeros(self.m)
        cdef np.ndarray[double,mode='c'] npi = np.zeros(self.n)
        cdef np.ndarray[double,mode='c'] nmu = np.zeros(self.n)

        if self.n:
            status = cipopt.IpoptSolve(self.problem,
                                       <double*>(nx.data),
                                       NULL,
                                       NULL,
                                       <double*>(nlam.data),
                                       <double*>(npi.data),
                                       <double*>(nmu.data),
                                       cself)
        else:
            status = 0
        
        return {'status' : status,
                'k': self.iters,
                'x': nx,
                'lam': nlam,
                'pi': npi,
                'mu': nmu}

cdef bint eval_f_cb(int n, double* x, bint new_x, double* obj_value, UserDataPtr user_data):
    cdef IpoptContext c = <IpoptContext>user_data
    obj_value[0] = c.eval_f(ArrayDouble(x,c.n))
    return True

cdef bint eval_grad_f_cb(int n, double* x, bint new_x, double* grad_f, UserDataPtr user_data):
    cdef IpoptContext c = <IpoptContext>user_data
    cdef np.ndarray[double,mode='c'] grad_f_arr = c.eval_grad_f(ArrayDouble(x,c.n))
    memcpy(grad_f,<double*>(grad_f_arr.data),sizeof(double)*c.n)
    return True

cdef bint eval_g_cb(int n, double* x, bint new_x, int m, double* g, UserDataPtr user_data):
    cdef IpoptContext c = <IpoptContext>user_data
    cdef np.ndarray[double,mode='c'] g_arr = c.eval_g(ArrayDouble(x,c.n))
    memcpy(g,<double*>(g_arr.data),sizeof(double)*c.m)
    return True

cdef bint eval_jac_g_cb(int n, double* x, bint new_x, int m, int nele_jac, 
                        int* iRow, int* jCol, double* values, UserDataPtr user_data):
    cdef IpoptContext c = <IpoptContext>user_data
    cdef np.ndarray[int,mode='c'] Jrow_arr
    cdef np.ndarray[int,mode='c'] Jcol_arr
    cdef np.ndarray[double,mode='c'] Jdata_arr
    if values == NULL:
        assert(iRow != NULL and jCol != NULL)
        Jrow_arr,Jcol_arr = c.eval_jac_g(None,True)
        assert(Jrow_arr.size == nele_jac and Jcol_arr.size == nele_jac)
        memcpy(iRow,<int*>(Jrow_arr.data),sizeof(int)*nele_jac)
        memcpy(jCol,<int*>(Jcol_arr.data),sizeof(int)*nele_jac)
    else:
        assert(x != NULL)
        Jdata_arr = c.eval_jac_g(ArrayDouble(x,c.n),False)
        assert(Jdata_arr.size == nele_jac)
        memcpy(values,<double*>(Jdata_arr.data),sizeof(double)*nele_jac)
    return True

cdef bint eval_h_cb(int n, double* x, bint new_x, double obj_factor, int m, double* lam, bint new_lam,
                    int nele_hess, int* iRow, int* jCol, double* values, UserDataPtr user_data):
    cdef IpoptContext c = <IpoptContext>user_data
    cdef np.ndarray[int,mode='c'] Hrow_arr
    cdef np.ndarray[int,mode='c'] Hcol_arr
    cdef np.ndarray[double,mode='c'] Hdata_arr
    if values == NULL:
        assert(iRow != NULL and jCol != NULL)
        Hrow_arr,Hcol_arr = c.eval_h(None,None,None,True)
        assert(Hrow_arr.size == nele_hess and Hcol_arr.size == nele_hess)
        memcpy(iRow,<int*>(Hrow_arr.data),sizeof(int)*nele_hess)
        memcpy(jCol,<int*>(Hcol_arr.data),sizeof(int)*nele_hess)
    else:
        assert(x != NULL and lam != NULL)
        Hdata_arr = c.eval_h(ArrayDouble(x,c.n),ArrayDouble(lam,c.m),obj_factor,False)
        assert(Hdata_arr.size == nele_hess)
        memcpy(values,<double*>(Hdata_arr.data),sizeof(double)*nele_hess)
    return True

cdef bint intermediate_cb(int alg_mod, int iter_count, double obj_value, double inf_pr, double inf_du,
                          double mu, double d_norm, double regularization_size, double alpha_du, double alpha_pr,
                          int ls_trials, UserDataPtr user_data):
    cdef IpoptContext c = <IpoptContext>user_data
    c.iters = iter_count
    return True
        
    
