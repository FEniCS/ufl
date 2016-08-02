.. title:: User manual


===========
User manual
===========

.. note:: This page is work in progress.

The global namespace of the module ufl contains the entire UFL
language::

  from ufl import *

Form compilers may want to import additional implementation details
like::

  from ufl.classes import *

and::

  from ufl.algorithms import *

Importing a ``.ufl`` file can be done easily from Python::

  from ufl.algorithms import load_ufl_file
  filedata = load_ufl_file("filename.ufl")
  forms = filedata.forms
  elements = filedata.elements

to get lists of forms and elements from the .ufl file, or::

  from ufl.algorithms import load_forms
  forms = load_forms("filename.ufl")

to get a list of forms in the .ufl file.
