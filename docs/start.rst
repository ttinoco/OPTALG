.. _start:

***************
Getting Started
***************

This section describes how to get started with OPTALG. In particular, it covers dependencies and installation.

.. _start_dependencies:

Dependencies
============

OPTALG has the following dependencies:

* `Numpy`_ (>=1.11.2): the fundamental package for scientific computing in Python.
* `Scipy`_ (>=0.18.1): a collection of mathematical algorithms and functions built on top of Numpy.
* `Cython`_ (>=0.20.1): an optimizing static compiler for both Python and the extended Cython programming language.

.. _start_installation:

Installation
============

In order to install OPTALG, the following tools are needed:

* Linux and Mac OS X: a C compiler, `Make`_, `Python`_ and `pip`_.
* Windows : `Anaconda`_, `7-Zip`_, and `MinGW`_.

After getting these tools, the OPTALG Python module can be installed using::

  pip install numpy cython
  pip install optalg

By default, no wrappers are built for any external solvers. If the environment variable ``OPTALG_IPOPT`` has the value ``true`` during the installation, OPTALG will download and build the solver `IPOPT`_ for you, and then build its Python wrapper. Similarly, if the environment variable ``OPTALG_CLP`` has the value ``true`` during the installation, OPTLAG will download and build the solver `Clp`_ for you, and then build its Python wrapper.

.. note:: Currently, the installation with `Clp`_ does not work on Windows and Mac OS X.
  
To install the module from source, the code can be obtained from `<https://github.com/ttinoco/OPTALG>`_, and then the following commands can be executed on the terminal or Anaconda prompt from the root directory of the package::

    pip install numpy cython
    python setup.py install

Running the unit tests can be done with::

    python setup.py build_ext --inplace
    nosetests -s -v

.. _Numpy: http://www.numpy.org
.. _Scipy: http://www.scipy.org
.. _Cython: http://cython.org/
.. _IPOPT: https://projects.coin-or.org/Ipopt
.. _Clp: https://projects.coin-or.org/Clp
.. _Make: https://www.gnu.org/software/make/
.. _Python: https://www.python.org/
.. _pip: https://pip.pypa.io/en/stable/
.. _Anaconda: https://www.anaconda.com/
.. _7-zip: http://www.7-zip.org/
.. _MinGW: https://anaconda.org/carlkl/mingwpy
