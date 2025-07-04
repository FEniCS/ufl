.. title:: Installation


============
Installation
============

UFL is normally installed as part of an installation of FEniCS.
If you are using UFL as part of the FEniCS software suite, it
is recommended that you follow the
`installation instructions for FEniCS
<https://fenics.readthedocs.io/en/latest/>`__.

To install UFL itself, read on below for a list of requirements
and installation instructions.


Requirements and dependencies
=============================

UFL requires Python version 3.5 or later and depends on the
following Python packages:

* NumPy

These packages will be automatically installed as part of the
installation of UFL, if not already present on your system.

Installation instructions
=========================

To install UFL, download the source code from the
`UFL GitHub repository
<https://github.com/FEniCS/ufl>`__,
and run the following command:

.. code-block:: console

    pip install .

To install to a specific location, add the ``--prefix`` flag
to the installation command:

.. code-block:: console

    pip install --prefix=<some directory> .
