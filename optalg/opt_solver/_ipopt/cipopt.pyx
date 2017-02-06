#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import numpy as np
cimport numpy as np

from libc.string cimport memcpy

cimport cipopt

np.import_array()

cdef ArrayDouble(double* a, int size):
     cdef np.npy_intp shape[1]
     shape[0] = <np.npy_intp> size
     arr = np.PyArray_SimpleNewFromData(1,shape,np.NPY_DOUBLE,a)
     return arr

class ProblemError(Exception):
    """
    IPOPT problem error exception.
    """
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

cdef class Problem:
    """
    IPOPT problem class.
    """

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
    
    def __init__(self,n,m,l,u,gl,gu,eval_f,eval_g,eval_grad_f,eval_jac_g,eval_h):
                
        self.n = n
        self.m = m
        self.l = l
        self.u = u
        self.gl = gl
        self.gu = gu
        self.eval_f = eval_f
        self.eval_g = eval_g
        self.eval_grad_f = eval_grad_f
        self.eval_jac_g = eval_jac_g
        self.eval_h = eval_h

        Jrow,Jcol = eval_jac_g(None,True) # x, flag
        assert(Jrow.size == Jcol.size)
        self.nnzj = Jrow.size

        Hrow,Hcol = eval_h(None,None,None,True) # x, lam, obj_factor, flag
        assert(Hrow.size == Hcol.size)
        self.nnzh = Hrow.size
    
    def solve(self,x):

        cdef UserDataPtr cself = <UserDataPtr>self
        cdef np.ndarray[double,mode='c'] nx = x
        cdef np.ndarray[double,mode='c'] nlam = np.zeros(self.m)
        cdef np.ndarray[double,mode='c'] npi = np.zeros(self.n)
        cdef np.ndarray[double,mode='c'] nmu = np.zeros(self.n)
        cdef np.ndarray[double,mode='c'] nl = self.l
        cdef np.ndarray[double,mode='c'] nu = self.u
        cdef np.ndarray[double,mode='c'] ngl = self.gl
        cdef np.ndarray[double,mode='c'] ngu = self.gu

        cdef cipopt.IpoptProblem prob = cipopt.CreateIpoptProblem(self.n,
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
        
        status = cipopt.IpoptSolve(prob,
                                   <double*>(nx.data),
                                   NULL,
                                   NULL,
                                   <double*>(nlam.data),
                                   <double*>(npi.data),
                                   <double*>(nmu.data),
                                   cself)

        cipopt.FreeIpoptProblem(prob)

        return {'status' : status,
                'x': nx,
                'lam': nlam,
                'pi': npi,
                'mu': nmu}

cdef bint eval_f_cb(int n, double* x, bint new_x, double* obj_value, UserDataPtr user_data):
    cdef Problem p = <Problem>user_data
    obj_value[0] = p.eval_f(ArrayDouble(x,p.n))
    return True

cdef bint eval_grad_f_cb(int n, double* x, bint new_x, double* grad_f, UserDataPtr user_data):
    cdef Problem p = <Problem>user_data
    cdef np.ndarray[double,mode='c'] grad_f_arr = p.eval_grad_f(ArrayDouble(x,p.n))
    memcpy(grad_f,<double*>(grad_f_arr.data),sizeof(double)*p.n)
    return True

cdef bint eval_g_cb(int n, double* x, bint new_x, int m, double* g, UserDataPtr user_data):
    cdef Problem p = <Problem>user_data
    cdef np.ndarray[double,mode='c'] g_arr = p.eval_g(ArrayDouble(x,p.n))
    memcpy(g,<double*>(g_arr.data),sizeof(double)*p.m)
    return True

cdef bint eval_jac_g_cb(int n, double* x, bint new_x, int m, int nele_jac, 
                        int* iRow, int* jCol, double* values, UserDataPtr user_data):
    cdef Problem p = <Problem>user_data
    cdef np.ndarray[int,mode='c'] Jrow_arr
    cdef np.ndarray[int,mode='c'] Jcol_arr
    cdef np.ndarray[double,mode='c'] Jdata_arr
    if x == NULL and iRow != NULL and jCol != NULL:
        Jrow_arr,Jcol_arr = p.eval_jac_g(None,True)
        if Jrow_arr.size != nele_jac or Jcol_arr.size != nele_jac:
            return False
        memcpy(iRow,<int*>(Jrow_arr.data),sizeof(int)*nele_jac)
        memcpy(jCol,<int*>(Jcol_arr.data),sizeof(int)*nele_jac)
    else:
        Jdata_arr = p.eval_jac_g(ArrayDouble(x,p.n),False)
        if Jdata_arr.size != nele_jac:
            return False
        memcpy(values,<double*>(Jdata_arr.data),sizeof(double)*nele_jac)
    return True

cdef bint eval_h_cb(int n, double* x, bint new_x, double obj_factor, int m, double* lam, bint new_lam,
                    int nele_hess, int* iRow, int* jCol, double* values, UserDataPtr user_data):
    cdef Problem p = <Problem>user_data
    cdef np.ndarray[int,mode='c'] Hrow_arr
    cdef np.ndarray[int,mode='c'] Hcol_arr
    cdef np.ndarray[double,mode='c'] Hdata_arr
    if x == NULL and iRow != NULL and jCol != NULL:
        Hrow_arr,Hcol_arr = p.eval_h(None,None,None,True)
        if Hrow_arr.size != nele_hess or Hcol_arr.size != nele_hess:
            return False
        memcpy(iRow,<int*>(Hrow_arr.data),sizeof(int)*nele_hess)
        memcpy(jCol,<int*>(Hcol_arr.data),sizeof(int)*nele_hess)
    else:
        Hdata_arr = p.eval_h(ArrayDouble(x,p.n),ArrayDouble(lam,p.m),obj_factor,False)
        if Hdata_arr.size != nele_hess:
            return False
        memcpy(values,<double*>(Hdata_arr.data),sizeof(double)*nele_hess)
    return True

        
        
    
