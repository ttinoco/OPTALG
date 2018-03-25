#****************************************************#
# This file is part of OPTALG.                       #
#                                                    #
# Copyright (c) 2015, Tomas Tinoco De Rubira.        #
#                                                    #
# OPTALG is released under the BSD 2-clause license. #
#****************************************************#

import os
import sys
import numpy as np
from subprocess import call
from Cython.Build import cythonize
from setuptools import setup, Extension

# External libraries
if 'darwin' in sys.platform.lower() or 'linux' in sys.platform.lower():
    return_code = call(["./build_lib.sh"])
else:
    return_code = call(["build_lib.bat"])
if return_code != 0:
    raise ValueError('Unable to build external library')

# Libraries and extra link args
if 'darwin' in sys.platform.lower():
    libraries_mumps = ['coinmumps']
    libraries_ipopt = ['ipopt']
    extra_link_args = ['-Wl,-rpath,@loader_path/']
elif 'linux' in sys.platform.lower():
    libraries_mumps = ['coinmumps']
    libraries_ipopt = ['ipopt']
    extra_link_args = ['-Wl,-rpath=$ORIGIN', '-Wl,-rpath=$ORIGIN/../../lin_solver/_mumps']
else:
    libraries_mumps = ['IpOptFSS']
    libraries_ipopt = ['IpOpt-vc10']
    extra_link_args = ['']
    
# Extension modules
ext_modules = []

# IPOPT and MUMPS
if os.environ.get('OPTALG_IPOPT') == 'true':

    # MUMPS
    ext_modules += cythonize([Extension(name='optalg.lin_solver._mumps._dmumps',
                                        sources=['./optalg/lin_solver/_mumps/_dmumps.pyx'],
                                        libraries=libraries_mumps,
                                        include_dirs=['./lib/ipopt/include/coin/ThirdParty'],
                                        library_dirs=['./lib/ipopt/lib'],
                                        extra_link_args=extra_link_args)])
    

    # IPOPT
    ext_modules += cythonize([Extension(name='optalg.opt_solver._ipopt.cipopt',
                                        sources=['./optalg/opt_solver/_ipopt/cipopt.pyx'],
                                        libraries=libraries_ipopt,
                                        include_dirs=[np.get_include(),'./lib/ipopt/include'],
                                        library_dirs=['./lib/ipopt/lib'],
                                        extra_link_args=extra_link_args)])
    
# CLP
if os.environ.get('OPTALG_CLP') == 'true':
    ext_modules += cythonize([Extension(name='optalg.opt_solver._clp.cclp',
                                        sources=['./optalg/opt_solver/_clp/cclp.pyx'],
                                        libraries=['Clp'],
                                        include_dirs=[np.get_include(),'./lib/clp/include'],
                                        library_dirs=['./lib/clp/lib'],
                                        extra_link_args=extra_link_args)])

# cbc (need to fix)
#if 'all' in args.ext or 'cbc' in args.ext:
#    from Cython.Build import cythonize 
#    ext_modules += cythonize([Extension(name='optalg.opt_solver._cbc.ccbc',
#                                        sources=['./optalg/opt_solver/_cbc/ccbc.pyx'],
#                                        include_dirs=[np.get_include()])])
 
setup(name='OPTALG',
      zip_safe=False,
      version='1.1.6rc1',
      description='Optimization Algorithms and Wrappers',
      url='https://github.com/ttinoco/OPTALG',
      author='Tomas Tinoco De Rubira',
      author_email='ttinoco5687@gmail.com',
      include_package_data=True,
      license='BSD 2-Clause License',
      packages=['optalg',
                'optalg.lin_solver',
                'optalg.lin_solver._mumps',
                'optalg.opt_solver',
                'optalg.opt_solver._ipopt',
                'optalg.opt_solver._clp',
                'optalg.opt_solver._cbc'],
      install_requires=['cython>=0.20.1',
                        'numpy>=1.11.2',
                        'scipy>=0.18.1',
                        'nose'],
      package_data={'optalg.lin_solver._mumps' : ['libcoinmumps*', 'IpOptFSS*'],
                    'optalg.opt_solver._ipopt' : ['libipopt*', 'IpOpt-vc10*', 'IpOptFSS*'],
                    'optalg.opt_solver._clp' : ['libClp*']},
      classifiers=['Development Status :: 5 - Production/Stable',
                   'License :: OSI Approved :: BSD License',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.5'],
      ext_modules=ext_modules)
      

