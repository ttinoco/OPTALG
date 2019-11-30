.. _start:

***************
Getting Started
***************

This section describes how to get started with OPTALG.

.. _start_installation:

Installation
============

In order to install OPTALG, the following tools are needed:

* Linux and Mac OS X:

  * C compiler
  * `Make`_
  * `Python`_ (2.7 or 3.6)
  * `pip`_
  
* Windows:
      
  * `Anaconda`_ (for Python 2.7)
  * `MinGW`_ (use ``pip install -i https://pypi.anaconda.org/carlkl/simple mingwpy``)
  * `7-Zip`_ (update system path to include the 7z executable, typically in ``C:\Program Files\7-Zip``)

After getting these tools, the OPTALG Python module can be installed using::

  pip install numpy cython
  pip install optalg

By default, no wrappers are built for any external solvers. If the environment variable ``OPTALG_IPOPT`` has the value ``true`` during the installation, OPTALG will download and build the solver `IPOPT`_ for you, and then build its Python wrapper. Similarly, if the environment variables ``OPTALG_CLP`` amd ``OPTALG_CBC`` have the value ``true`` during the installation, OPTLAG will download and build the solvers `Clp`_ and `Cbc`_ for you, and then build their Python wrappers.

.. note:: Currently, the installation with `Clp`_ and `Cbc`_ does not work on Windows.
  
To install the module from source, the code can be obtained from `<https://github.com/ttinoco/OPTALG>`_, and then the following commands can be executed on the terminal or Anaconda prompt from the root directory of the package::

    pip install numpy cython
    python setup.py install

Running the unit tests can be done with::

    pip install nose
    python setup.py build_ext --inplace
    nosetests -s -v

.. _Numpy: http://www.numpy.org
.. _Scipy: http://www.scipy.org
.. _Cython: http://cython.org/
.. _IPOPT: https://projects.coin-or.org/Ipopt
.. _Clp: https://projects.coin-or.org/Clp
.. _Cbc: https://projects.coin-or.org/Cbc
.. _Make: https://www.gnu.org/software/make/
.. _Python: https://www.python.org/
.. _pip: https://pip.pypa.io/en/stable/
.. _Anaconda: https://www.anaconda.com/
.. _7-zip: http://www.7-zip.org/
.. _MinGW: https://anaconda.org/carlkl/mingwpy
