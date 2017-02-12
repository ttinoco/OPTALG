#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import os
import sys
import numpy as np
from Cython.Build import cythonize
from distutils.core import setup,Extension

ext_modules = []

# mumps
if '--no_mumps' in sys.argv:
    sys.argv.remove('--no_mumps')
else:
    ext_modules += cythonize([Extension(name='optalg.lin_solver._mumps._dmumps',
                                        sources=['./optalg/lin_solver/_mumps/_dmumps.pyx'],
                                        libraries=['dmumps_seq'],
                                        library_dirs=[],
                                        include_dirs=[],
                                        extra_link_args=[],
                                        extra_compile_args=[])])

# ipopt
if '--no_ipopt' in sys.argv:
    sys.argv.remove('--no_ipopt')
else:
    ext_modules += cythonize([Extension(name='optalg.opt_solver._ipopt.cipopt',
                                        sources=['./optalg/opt_solver/_ipopt/cipopt.pyx'],
                                        libraries=['ipopt','coinmumps'],
                                        library_dirs=[os.getenv('IPOPT')+'/lib'],
                                        include_dirs=[np.get_include(),os.getenv('IPOPT')+'/include/coin'],
                                        extra_link_args=[],
                                        extra_compile_args=[])])

# augl 
ext_modules += cythonize([Extension(name='optalg.opt_solver._augl.caugl',
                                    sources=['./optalg/opt_solver/_augl/caugl.pyx'],
                                    libraries=[],
                                    library_dirs=[],
                                    include_dirs=[np.get_include()],
                                    extra_link_args=[],
                                    extra_compile_args=[])])
    
setup(name='OPTALG',
      version='1.1.1',
      description='Optimization Algorithms',
      author='Tomas Tinoco De Rubira',
      author_email='ttinoco5687@gmail.com',
      ext_modules=ext_modules,
      packages=['optalg',
                'optalg.lin_solver',
                'optalg.lin_solver._mumps',
                'optalg.opt_solver',
                'optalg.opt_solver._ipopt',
                'optalg.opt_solver._augl',
                'optalg.stoch_solver'],
      requires=['scipy',
                'numpy',
                'dill',
                'cython'])
