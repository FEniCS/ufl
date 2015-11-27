************
Installation
************

The source code of UFL is portable and should work on any system with
a standard Python installation.  Questions, bug reports and patches
concerning the installation should be directed to the FEniCS mailing list
at the address:

  fenics-support@fenicsproject.org

UFL must currently be installed directly from source, but Debian (Ubuntu)
packages will be available in the future, for UFL and other FEniCS
components.

Installing from source
======================

Installing UFL
--------------

UFL follows the standard installation procedure for Python packages. Enter
the source directory of UFL and issue the following command::

  python setup.py install

This will install the UFL Python package in a subdirectory called
``ufl`` in the default location for user-installed Python packages
(usually something like ``/usr/lib/python2.7/site-packages``).

In addition, the executable ``ufl-analyse`` (a Python script) will
be installed in the default directory for user-installed Python scripts
(usually in ``/usr/bin``).

To see a list of optional parameters to the installation script, type::

  python setup.py install --help

If you don't have root access to the system you are using, you can pass
the ``--prefix`` option to the installation script to install UFL in
another directory::

  python setup.py install --prefix=/opt/ufl

Running the test suite
----------------------

To verify that the installation is correct, you may run the test suite.
Enter the sub directory ``test/`` from within the UFL source tree
and run the script ``test.py``::

  python test.py

This script runs all unit tests and imports UFL in the process.
