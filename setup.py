#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015-2017, Tomas Tinoco De Rubira.   #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import sys
import argparse
import numpy as np
from setuptools import setup,Extension

ext_modules = []

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--with',nargs='?',dest='ext',type=str,const='',default='')
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

# mumps
if 'all' in args.ext or 'mumps' in args.ext:
    from Cython.Build import cythonize
    ext_modules += cythonize([Extension(name='optalg.lin_solver._mumps._dmumps',
                                        sources=['./optalg/lin_solver/_mumps/_dmumps.pyx'])])

# ipopt
if 'all' in args.ext or 'ipopt' in args.ext:
    from Cython.Build import cythonize
    ext_modules += cythonize([Extension(name='optalg.opt_solver._ipopt.cipopt',
                                        sources=['./optalg/opt_solver/_ipopt/cipopt.pyx'],
                                        include_dirs=[np.get_include()])])

# clp
if 'all' in args.ext or 'clp' in args.ext:
    from Cython.Build import cythonize 
    ext_modules += cythonize([Extension(name='optalg.opt_solver._clp.cclp',
                                        sources=['./optalg/opt_solver/_clp/cclp.pyx'],
                                        include_dirs=[np.get_include()])])

# cbc
if 'all' in args.ext or 'cbc' in args.ext:
    from Cython.Build import cythonize 
    ext_modules += cythonize([Extension(name='optalg.opt_solver._cbc.ccbc',
                                        sources=['./optalg/opt_solver/_cbc/ccbc.pyx'],
                                        include_dirs=[np.get_include()])])
 
setup(name='OPTALG',
      version='1.1.3',
      description='Optimization Algorithms',
      author='Tomas Tinoco De Rubira',
      author_email='ttinoco5687@gmail.com',
      ext_modules=ext_modules,
      packages=['optalg',
                'optalg.lin_solver',
                'optalg.lin_solver._mumps',
                'optalg.opt_solver',
                'optalg.opt_solver._ipopt',
                'optalg.opt_solver._clp',
                'optalg.opt_solver._cbc',
                'optalg.stoch_solver'],
      install_requires=['scipy',
                        'numpy',
                        'dill',
                        'multiprocess'])
