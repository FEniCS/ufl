************
Introduction
************

The Unified Form Language (UFL) is a domain specific language for
defining discrete variational forms and functionals in a notation close
to pen-and-paper formulation.

UFL is part of the FEniCS Project, and is usually used in combination
with other components from this project to compute solutions to partial
differential equations. The form compiler FFC uses UFL as its
end-user interface, producing implementations of the UFC interface as
output. See DOLFIN for more details about using UFL in an integrated
problem solving environment.

This manual is intended for different audiences.  If you are an end user
and all you want to do is to solve your PDEs with the FEniCS framework,
you should read :doc:`form_language`, and also :doc:`examples`. These two
sections explain how to use all operators available in the language and
present a number of examples to illustrate the use of the form language in applications.

The remaining chapters contain more technical details intended for developers
who need to understand what is happening behind the scenes and modify
or extend UFL in the future.

:doc:`internal_representation` describes the implementation of the language, in particular
how expressions are represented internally by UFL.  This can also be
useful knowledge to understand error messages and debug errors in your
form files.

:doc:`algorithms` explains the many algorithms available to work with UFL expressions,
mostly intended to aid developers of form compilers.  The algorithms include
helper functions for easy and efficient iteration over expression
trees, formatting tools to present expressions as text or
images of different kinds, utilities to analyse properties of expressions
or checking their validity, automatic differentiation algorithms, as
well as algorithms to work with the computational graphs of expressions.
