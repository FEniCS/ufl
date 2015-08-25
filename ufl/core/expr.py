# -*- coding: utf-8 -*-
"""This module defines the Expr class, the superclass
for all expression tree node types in UFL.

NB! A note about other operators not implemented here:

More operators (special functions) on Exprs are defined in exproperators.py,
as well as the transpose "A.T" and spatial derivative "a.dx(i)".
This is to avoid circular dependencies between Expr and its subclasses.
"""

# Copyright (C) 2008-2015 Martin Sandve Alnæs
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2008

from six.moves import xrange as range

from ufl.log import warning, error


def find_geometric_dimension(expr): # TODO: Move to corealg.analysis module
    "Find the geometric dimension of an expression."
    gdims = set(domain.geometric_dimension() for domain in expr.ufl_domains())
    if len(gdims) != 1:
        error("Cannot determine geometric dimension from expression.")
    return tuple(gdims)[0]


#--- The base object for all UFL expression tree nodes ---

class Expr(object):
    """Base class for all UFL expression types.

    *Instance properties*
        Every expression instance will have certain properties.
        Most important are the ``ufl_operands``, ``ufl_shape``,
        ``ufl_free_indices``, and ``ufl_index_dimensions`` properties.
        Expressions are immutable and hashable.

    *Type traits*
        The Expr API defines a number of type traits that each subclass
        needs to provide. Most of these are specified indirectly via
        the arguments to the ``ufl_type`` class decorator, allowing UFL
        to do some consistency checks and automate most of the traits
        for most types. The type traits are accessed via a class or
        instance object on the form obj._ufl_traitname_. See the source
        code for description of each type trait.

    *Operators*
        Some Python special functions are implemented in this class,
        some are implemented in subclasses, and some are attached to
        this class in the ``ufl_type`` class decorator.

    *Defining subclasses*
        To define a new expression class, inherit from either
        ``Terminal`` or ``Operator``, and apply the ``ufl_type`` class
        decorator with suitable arguments.  See the docstring of
        ``ufl_type`` for details on its arguments.  Looking at existing
        classes similar to the one you wish to add is a good
        idea. Looking through the comments in the ``Expr`` class and
        ``ufl_type`` to understand all the properties that may need to
        be specified is also a good idea. Note that many algorithms in
        UFL and form compilers will need handlers implemented for each
        new type.

        .. code-block:: python

            @ufl_type()
            class MyOperator(Operator):
                pass

    *Type collections*
        All ``Expr`` subclasses are collected by ``ufl_type`` in global
        variables available via ``Expr``.

    *Profiling*
        Object creation statistics can be collected by doing

        .. code-block:: python

            Expr.ufl_enable_profiling()
            # ... run some code
            initstats, delstats = Expr.ufl_disable_profiling()

        Giving a list of creation and deletion counts for each typecode.
    """

    # --- Each Expr subclass must define __slots__ or _ufl_noslots_ at the top ---
    # This is to freeze member variables for objects of this class and save memory
    # by skipping the per-instance dict.

    __slots__ = ("_hash",)
    #_ufl_noslots_ = True


    # --- Basic object behaviour ---

    def __getnewargs__(self):
        """The tuple returned here is passed to as args to cls.__new__(cls, *args).

        This implementation passes the operands, which is () for terminals.

        May be necessary to override if __new__ is implemented in a subclass.
        """
        return self.ufl_operands

    def __init__(self):
        self._hash = None

    def __del__(self):
        pass

    # This shows the principal behaviour of the hash function attached in ufl_type:
    #def __hash__(self):
    #    if self._hash is None:
    #        self._hash = self._ufl_compute_hash_()
    #    return self._hash

    # --- Type traits are added to subclasses by the ufl_type class decorator ---

    # Note: Some of these are modified after the Expr class definition
    # because Expr is not defined yet at this point.
    # Note: Boolean type traits that categorize types are mostly set to
    # None for Expr but should be True or False for any non-abstract type.

    # A reference to the UFL class itself.
    # This makes it possible to do type(f)._ufl_class_ and be sure you get
    # the actual UFL class instead of a subclass from another library.
    _ufl_class_ = None

    # The handler name.
    # This is the name of the handler function you implement for this type in a multifunction.
    _ufl_handler_name_ = "expr"

    # The integer typecode, a contiguous index different for each type.
    # This is used for fast lookup into e.g. multifunction handler tables.
    _ufl_typecode_ = 0

    # Number of operands, "varying" for some types, or None if not applicable for abstract types.
    _ufl_num_ops_ = None

    # Type trait: If the type is abstract.
    # An abstract class cannot be instantiated and does not need all properties specified.
    _ufl_is_abstract_ = True

    # Type trait: If the type is terminal.
    _ufl_is_terminal_ = None

    # Type trait: If the type is a literal.
    _ufl_is_literal_ = None

    # Type trait: If the type is classified as a 'terminal modifier', for form compiler use.
    _ufl_is_terminal_modifier_ = None

    # Type trait: If the type is a shaping operator.
    # Shaping operations include indexing, slicing, transposing,
    # i.e. not introducing computation of a new value.
    _ufl_is_shaping_ = False

    # Type trait: If the type is in reference frame.
    _ufl_is_in_reference_frame_ = None

    # Type trait: If the type is a restriction to a geometric entity.
    _ufl_is_restriction_ = None

    # Type trait: If the type is evaluation in a particular way.
    _ufl_is_evaluation_ = None

    # Type trait: If the type is a differential operator.
    _ufl_is_differential_ = None

    # Type trait: If the type is purely scalar, having no shape or indices.
    _ufl_is_scalar_ = None

    # Type trait: If the type never has free indices.
    _ufl_is_index_free_ = False


    # --- All subclasses must define these object attributes ---

    # FIXME: Enable checks in ufl_type
    _ufl_required_properties_ = (
        # A tuple of operands, all of them Expr instances.
        "ufl_operands",

        # A tuple of ints, the value shape of the expression.
        "ufl_shape",

        # A tuple of free index counts.
        "ufl_free_indices",

        # A tuple providing the int dimension for each free index.
        "ufl_index_dimensions",
        )

    # FIXME: Add more and enable all
    _ufl_required_methods_ = (
        # To compute the hash on demand, this method is called.
        "_ufl_compute_hash_",

        # The == operator must be implemented to compare for identical representation, used by set() and dict().
        "__eq__",

        # To reconstruct an object of the same type with operands or properties changed.
        "reconstruct",

        # Return whether this expression is spatially constant over each cell.
        "domains",
        "is_cellwise_constant",
        #"cell",
        #"domain",
        #"geometric_dimension",

        #"__str__",
        #"__repr__",
        #"signature_data",
        #"__repr__",
        )


    # --- Global variables for collecting all types ---

    # A global counter of the number of typecodes assigned
    _ufl_num_typecodes_ = 1

    # A global set of all handler names added
    _ufl_all_handler_names_ = set()

    # A global array of all Expr subclasses, indexed by typecode
    _ufl_all_classes_ = []

    # A global dict mapping language_operator_name to the type it produces
    _ufl_language_operators_ = {}

    # List of all terminal modifier types
    _ufl_terminal_modifiers_ = []


    # --- Mechanism for profiling object creation and deletion ---

    # A global array of the number of initialized objects for each typecode
    _ufl_obj_init_counts_ = [0]

    # A global array of the number of deleted objects for each typecode
    _ufl_obj_del_counts_ = [0]

    # Backup of default init and del
    _ufl_regular__init__ = __init__
    _ufl_regular__del__ = __del__

    def _ufl_profiling__init__(self):
        "Replacement constructor with object counting."
        Expr._ufl_regular__init__(self)
        Expr._ufl_obj_init_counts_[self._ufl_typecode_] += 1

    def _ufl_profiling__del__(self):
        "Replacement destructor with object counting."
        Expr._ufl_regular__del__(self)
        Expr._ufl_obj_del_counts_[self._ufl_typecode_] -= 1

    @staticmethod
    def ufl_enable_profiling():
        "Turn on object counting mechanism and reset counts to zero."
        Expr.__init__ = Expr._ufl_profiling__init__
        Expr.__del__ = Expr._ufl_profiling__del__
        for i in range(len(Expr._ufl_obj_init_counts_)):
            Expr._ufl_obj_init_counts_[i] = 0
            Expr._ufl_obj_del_counts_[i] = 0

    @staticmethod
    def ufl_disable_profiling():
        "Turn off object counting mechanism. Returns object init and del counts."
        Expr.__init__ = Expr._ufl_regular__init__
        Expr.__del__ = Expr._ufl_regular__del__
        return (Expr._ufl_obj_init_counts_, Expr._ufl_obj_del_counts_)


    #=== Abstract functions that must be implemented by subclasses ===

    #--- Functions for reconstructing expression ---

    # All subclasses must implement reconstruct
    def reconstruct(self, *operands):
        "Return a new object of the same type with new operands."
        raise NotImplementedError(self.__class__.reconstruct)

    #--- Functions for geometric properties of expression ---

    # All subclasses must implement ufl_domains if it is known
    def ufl_domains(self): # TODO: Deprecate this and use extract_domains
        from ufl.geometry import extract_domains # TODO: Move to corealg.analysis module
        return extract_domains(self)

    # All subclasses must implement ufl_domain if it is known
    def ufl_domain(self):
        "Return the single unique domain this expression is defined on or throw an error."
        domains = self.ufl_domains()
        if len(domains) == 1:
            domain, = domains
            return domain
        elif domains:
            error("Found multiple domains, cannot return just one.")
        else:
            return None

    # All subclasses must implement cell if it is known
    def ufl_cell(self): # TODO: Deprecate this
        "Return the cell this expression is defined on."
        domain = self.ufl_domain()
        return domain.ufl_cell() if domain is not None else None

    # This function was introduced to clarify and
    # eventually reduce direct dependencies on cells.
    def geometric_dimension(self): # TODO: Deprecate this
        "Return the geometric dimension this expression lives in."
        #from ufl.corealg.analysis import find_geometric_dimension
        return find_geometric_dimension(self)

    def is_cellwise_constant(self): # TODO: Deprecate this
        "Return whether this expression is spatially constant over each cell."
        raise NotImplementedError(self.__class__.is_cellwise_constant)

    #--- Functions for float evaluation ---

    def evaluate(self, x, mapping, component, index_values): # TODO: Move to corealg.eval module
        """Evaluate expression at given coordinate with given values for terminals."""
        error("Symbolic evaluation of %s not available." % self._ufl_class_.__name__)

    def _ufl_evaluate_scalar_(self):
        if self.ufl_shape or self.ufl_free_indices:
            raise TypeError("Cannot evaluate a nonscalar expression to a scalar value.")
        return self(()) # No known x

    def __float__(self):
        "Try to evaluate as scalar and cast to float."
        try:
            v = float(self._ufl_evaluate_scalar_())
        except:
            v = NotImplemented
        return v

    def __bool__(self):
        "By default, all Expr are nonzero/False."
        return True

    def __nonzero__(self):
        "By default, all Expr are nonzero/False."
        return self.__bool__()

    @staticmethod
    def _ufl_coerce_(value):
        "Convert any value to a UFL type."
        # Quick skip for most types
        if isinstance(value, Expr):
            return value

        # Conversion from non-ufl types
        # (the _ufl_from_*_ functions are attached to Expr by ufl_type)
        ufl_from_type = "_ufl_from_{0}_".format(value.__class__.__name__)
        return getattr(Expr, ufl_from_type)(value)

        #if hasattr(Expr, ufl_from_type):
        #    return getattr(Expr, ufl_from_type)(value)
        ## Fail gracefully if no valid type conversion found
        #raise TypeError("Cannot convert a {0.__class__.__name__} to UFL type.".format(value))

    #--- Functions for shape and index handling ---

    def shape(self):
        "Return the tensor shape of the expression."
        return self.ufl_shape

    def rank(self):
        "Return the tensor rank of the expression."
        return len(self.ufl_shape)

    #--- Deprecated functions

    def domains(self):
        deprecate("Expr.domains() is deprecated, please use .ufl_domains() instead.")
        return self.ufl_domains()

    def cell(self):
        deprecate("Expr.cell() is deprecated, please use .ufl_cell() instead.")
        return self.ufl_cell()

    def domain(self):
        deprecate("Expr.domain() is deprecated, please use .ufl_domain() instead.")
        return self.ufl_domain()

    def operands(self):
        deprecate("Expr.operands() is deprecated, please use property Expr.ufl_operands instead.")
        raise NotImplementedError(self.__class__.operands)

    def free_indices(self):
        deprecate("Expr.free_indices() is deprecated, please use property Expr.ufl_free_indices instead.")
        raise NotImplementedError(self.__class__.free_indices)

    def index_dimensions(self):
        deprecate("Expr.index_dimensions() is deprecated, please use property Expr.ufl_index_dimensions instead.")
        raise NotImplementedError(self.__class__.index_dimensions)

    #--- Special functions for string representations ---

    # All subclasses must implement signature_data
    def signature_data(self):
        "Return data that uniquely identifies form compiler relevant aspects of this object."
        raise NotImplementedError(self.__class__.signature_data)

    # All subclasses must implement __repr__
    def __repr__(self):
        "Return string representation this object can be reconstructed from."
        raise NotImplementedError(self.__class__.__repr__)

    # All subclasses must implement __str__
    def __str__(self):
        "Return pretty print string representation of this object."
        raise NotImplementedError(self.__class__.__str__)

    def _repr_latex_(self):
        from ufl.algorithms import ufl2latex
        return "$%s$" % ufl2latex(self)

    def _repr_png_(self):
        from IPython.lib.latextools import latex_to_png
        return latex_to_png(self._repr_latex_())

    #--- Special functions used for processing expressions ---

    def __eq__(self, other):
        """Checks whether the two expressions are represented the
        exact same way. This does not check if the expressions are
        mathematically equal or equivalent! Used by sets and dicts."""
        raise NotImplementedError(self.__class__.__eq__)

    def __len__(self):
        "Length of expression. Used for iteration over vector expressions."
        s = self.ufl_shape
        if len(s) == 1:
            return s[0]
        raise NotImplementedError("Cannot take length of non-vector expression.")

    def __iter__(self):
        "Iteration over vector expressions."
        for i in range(len(self)):
            yield self[i]

    def __floordiv__(self, other):
        "UFL does not support integer division."
        raise NotImplementedError(self.__class__.__floordiv__)

    def __pos__(self):
        "Unary + is a no-op."
        return self

    def __round__(self, n=None):
        "Round to nearest integer or to nearest nth decimal."
        return round(float(self), n)


# Initializing traits here because Expr is not defined in the class declaration
Expr._ufl_class_ = Expr
Expr._ufl_all_handler_names_.add(Expr)
Expr._ufl_all_classes_.append(Expr)
