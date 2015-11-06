************
Installation
************
\label{app:installation}
\index{installation}

The source code of UFL is portable and should work on any system with
a standard Python installation.  Questions, bug reports and patches
concerning the installation should be directed to the UFL mailing list
at the address:

  ufl-dev@fenics.org

UFL must currently be installed directly from source, but Debian (Ubuntu)
packages will be available in the future, for UFL and other FEniCS
components.

Installing from source
======================

Dependencies and requirements
-----------------------------
\index{dependencies}

UFL currently has no external dependencies apart from a working Python
installation.

%UFL depends on the Python NumPy module.
%In addition, you need to have a working Python installation on your system.

Installing Python
-----------------

UFL is developed for Python 2.5, and does not work with previous versions.
To check which version of Python you have installed, issue the command
\texttt{python~-V}::

  # python -V
  Python 2.5.1

If Python is not installed on your system, it can be downloaded from::

  http://www.python.org/

Follow the installation instructions for Python given on the Python
web page.  For Debian (Ubuntu) users, the package to install is named
\texttt{python}.

%\subsubsection{Installing NumPy}
%
%In addition to Python itself, UFL depends on the Python package NumPy,
%which is used by UFL to process multidimensional arrays (tensors).
%Python NumPy can be downloaded from
%\begin{code}
%http://www.scipy.org/
%\end{code}
%For Debian (Ubuntu) users, the package to install is \texttt{python-numpy}.

% Input section shared with DOLFIN manual
\input{chapters/installation-downloading.tex}

Installing UFL
--------------

UFL follows the standard installation procedure for Python packages. Enter
the source directory of UFL and issue the following command::

  # python setup.py install

This will install the UFL Python package in a subdirectory called
\texttt{ufl} in the default location for user-installed Python packages
(usually something like \texttt{/usr/lib/python2.5/site-packages}).

In addition, the executable \texttt{ufl-analyse} (a Python script) will
be installed in the default directory for user-installed Python scripts
(usually in \texttt{/usr/bin}).

To see a list of optional parameters to the installation script, type::

  # python setup.py install --help

If you don't have root access to the system you are using, you can pass
the \texttt{--home} option to the installation script to install UFL in
your home directory::

  # mkdir ~/local
  # python setup.py install --home ~/local

This installs the UFL package in the directory
\texttt{\~{}/local/lib/python} and the UFL executables in
\texttt{\~{}/local/bin}. If you use this option, make sure to set the
environment variable \texttt{PYTHONPATH} to \texttt{\~{}/local/lib/python}
and to add \texttt{\~{}/local/bin} to the \texttt{PATH} environment
variable.


Running the test suite
----------------------

To verify that the installation is correct, you may run the test suite.
Enter the sub directory \texttt{test/} from within the UFL source tree
and run the script \texttt{test.py}::

  # python test.py

This script runs all unit tests and imports UFL in the process.

% TODO: Add regression tests on ufl-analyse output from demos? Perhaps
% when ufl-analyse stabilises.

Debian/Ubuntu packages
======================
\index{Debian package}
\index{Ubuntu package}

In preparation.
