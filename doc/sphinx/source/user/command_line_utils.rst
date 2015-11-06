*********************
Commandline utilities
*********************


Validation and debugging: ``ufl-analyse``
=========================================
\index{\ttt{ufl-analyse}}

The command \ttt{ufl-analyse} loads all forms found in a \ttt{.ufl}
file, tries to discover any errors in them, and prints various kinds of
information about each form.  Basic usage is::

  # ufl-analyse myform.ufl

For more information, type::

  # ufl-analyse --help

Formatting and visualization: ``ufl-convert``
=============================================
\index{\ttt{ufl-convert}}

The command \ttt{ufl-convert} loads all forms found in a \ttt{.ufl}
file, compiles them into a different form or extracts some information
from them, and writes the result in a suitable file format.

To try this tool, go to the \ttt{demo/} directory of the UFL source
tree. Some of the features to try are basic printing of \ttt{str} and
\ttt{repr} string representations of each form::

  # ufl-convert --format=str stiffness.ufl
  # ufl-convert --format=repr stiffness.ufl

compilation of forms to mathematical notation in LaTeX::

  # ufl-convert --filetype=pdf --format=tex --show=1 stiffness.ufl

LaTeX output of forms after processing with UFL compiler utilities::

  # ufl-convert -tpdf -ftex -s1 --compile=1 stiffness.ufl

and visualization of expression trees using graphviz via compilation of
forms to the dot format::

  # ufl-convert -tpdf -fdot -s1 stiffness.ufl

Type \ttt{ufl-convert --help} for more details.

Conversion from FFC form files: ``form2ufl``
============================================
\index{\ttt{form2ufl}}

The command \ttt{form2ufl} can be used to convert old FFC \ttt{.form}
files to UFL format. To convert a form file named \ttt{myform.form}
to UFL format, simply type::

  # form2ufl myform.ufl

Note that although, the \ttt{form2ufl} script may be helpful as a guide
to converting old FFC \ttt{.form} files, it is not foolproof and may
not always yield valid UFL files.

