#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import os
import sys
import platform
import numpy as np
#from setuptools import setup,Extension
from distutils.core import setup,Extension

ext_modules = []

# mumps
if '--no_mumps' in sys.argv:
    sys.argv.remove('--no_mumps')
else:
    from Cython.Build import cythonize
    if platform.system() == 'Darwin':
        libraries=['dmumps','mumps_common']
    else:
        libraries=['dmumps_seq']
    ext_modules += cythonize([Extension(name='optalg.lin_solver._mumps._dmumps',
                                        sources=['./optalg/lin_solver/_mumps/_dmumps.pyx'],
                                        libraries=libraries,
                                        library_dirs=[],
                                        include_dirs=[],
                                        extra_link_args=[],
                                        extra_compile_args=[])])

# ipopt
if '--no_ipopt' in sys.argv:
    sys.argv.remove('--no_ipopt')
else:
    from Cython.Build import cythonize
    ext_modules += cythonize([Extension(name='optalg.opt_solver._ipopt.cipopt',
                                        sources=['./optalg/opt_solver/_ipopt/cipopt.pyx'],
                                        libraries=['ipopt','coinmumps'],
                                        library_dirs=[os.getenv('IPOPT')+'/lib'],
                                        include_dirs=[np.get_include(),os.getenv('IPOPT')+'/include/coin'],
                                        extra_link_args=[],
                                        extra_compile_args=[])])

setup(name='OPTALG',
      version='1.1.2',
      description='Optimization Algorithms',
      author='Tomas Tinoco De Rubira',
      author_email='ttinoco5687@gmail.com',
      ext_modules=ext_modules,
      packages=['optalg',
                'optalg.lin_solver',
                'optalg.lin_solver._mumps',
                'optalg.opt_solver',
                'optalg.opt_solver._ipopt',
                'optalg.stoch_solver'],
      install_requires=['scipy',
                        'numpy',
                        'dill',
                        'multiprocess'])
