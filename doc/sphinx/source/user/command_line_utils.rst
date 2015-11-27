*********************
Commandline utilities
*********************


Validation and debugging: ``ufl-analyse``
=========================================

The command ``ufl-analyse`` loads all forms found in a ``.ufl``
file, tries to discover any errors in them, and prints various kinds of
information about each form.  Basic usage is::

  # ufl-analyse myform.ufl

For more information, type::

  # ufl-analyse --help

Formatting and visualization: ``ufl-convert``
=============================================

The command ``ufl-convert`` loads all forms found in a ``.ufl``
file, compiles them into a different form or extracts some information
from them, and writes the result in a suitable file format.

To try this tool, go to the ``demo/`` directory of the UFL source
tree. Some of the features to try are basic printing of ``str`` and
``repr`` string representations of each form::

  # ufl-convert --format=str stiffness.ufl
  # ufl-convert --format=repr stiffness.ufl

compilation of forms to mathematical notation in LaTeX::

  # ufl-convert --filetype=pdf --format=tex --show=1 stiffness.ufl

LaTeX output of forms after processing with UFL compiler utilities::

  # ufl-convert -tpdf -ftex -s1 --compile=1 stiffness.ufl

and visualization of expression trees using graphviz via compilation of
forms to the dot format::

  # ufl-convert -tpdf -fdot -s1 stiffness.ufl

Type ``ufl-convert --help`` for more details.


