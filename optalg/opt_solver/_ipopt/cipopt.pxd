#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

cdef extern from "coin/IpStdCInterface.h":
    
    ctypedef struct IpoptProblemInfo    
    ctypedef IpoptProblemInfo* IpoptProblem
    ctypedef void* UserDataPtr

    ctypedef bint (*Eval_F_CB)(int n, double* x, bint new_x,
                               double* obj_value, UserDataPtr user_data)

    ctypedef bint (*Eval_Grad_F_CB)(int n, double* x, bint new_x,
                                    double* grad_f, UserDataPtr user_data)

    
    ctypedef bint (*Eval_G_CB)(int n, double* x, bint new_x,
                               int m, double* g, UserDataPtr user_data)


    ctypedef bint (*Eval_Jac_G_CB)(int n, double* x, bint new_x,
                                   int m, int nele_jac,
                                   int* iRow, int* jCol, double* values,
                                   UserDataPtr user_data)
    
    
    ctypedef bint (*Eval_H_CB)(int n, double* x, bint new_x, double obj_factor,
                               int m, double* lam, bint new_lam,
                               int nele_hess, int* iRow, int* jCol,
                               double* values, UserDataPtr user_data)
    
    ctypedef bint (*Intermediate_CB)(int alg_mod, 
                                     int iter_count, double obj_value,
                                     double inf_pr, double inf_du,
                                     double mu, double d_norm,
                                     double regularization_size,
                                     double alpha_du, double alpha_pr,
                                     int ls_trials, UserDataPtr user_data)
    
    IpoptProblem CreateIpoptProblem(int n, double* x_L, double* x_U, int m, 
                                    double* g_L, double* g_U, int nele_jac, 
                                    int nele_hess, int index_style, 
                                    Eval_F_CB eval_f,
                                    Eval_G_CB eval_g,
                                    Eval_Grad_F_CB eval_grad_f,
                                    Eval_Jac_G_CB eval_jac_g,
                                    Eval_H_CB eval_h)

    void FreeIpoptProblem(IpoptProblem problem)
    bint AddIpoptStrOption(IpoptProblem problem, char* keyword, char* val)
    bint AddIpoptNumOption(IpoptProblem problem, char* keyword, double val)
    bint AddIpoptIntOption(IpoptProblem problem, char* keyword, int val)
    bint SetIpoptProblemScaling(IpoptProblem problem, double obj_scaling, double* x_scaling, double* g_scaling)
    bint SetIntermediateCallback(IpoptProblem problem,Intermediate_CB icb)
    int IpoptSolve(IpoptProblem problem,
                   double* x,
                   double* g,
                   double* obj_val,
                   double* mult_g,
                   double* mult_x_L,
                   double* mult_x_U,
                   UserDataPtr user_data)
