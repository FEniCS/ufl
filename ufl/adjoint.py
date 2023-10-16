"""This module defines the Adjoint class."""

# Copyright (C) 2021 India Marsden
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Nacime Bouziani, 2021-2022.

from ufl.argument import Coargument
from ufl.core.ufl_type import ufl_type
from ufl.form import BaseForm, FormSum, ZeroBaseForm

# --- The Adjoint class represents the adjoint of a numerical object that
#     needs to be computed at assembly time ---


@ufl_type()
class Adjoint(BaseForm):
    """UFL base form type: represents the adjoint of an object.

    Adjoint objects will result when the adjoint of an assembled object
    (e.g. a Matrix) is taken. This delays the evaluation of the adjoint until
    assembly occurs.
    """

    __slots__ = (
        "_form",
        "_repr",
        "_arguments",
        "_coefficients",
        "_domains",
        "ufl_operands",
        "_hash")

    def __new__(cls, *args, **kw):
        """Create a new Adjoint."""
        form = args[0]
        # Check trivial case: This is not a ufl.Zero but a ZeroBaseForm!
        if form == 0:
            # Swap the arguments
            return ZeroBaseForm(form.arguments()[::-1])

        if isinstance(form, Adjoint):
            return form._form
        elif isinstance(form, FormSum):
            # Adjoint distributes over sums
            return FormSum(*[(Adjoint(component), 1)
                             for component in form.components()])
        elif isinstance(form, Coargument):
            # The adjoint of a coargument `c: V* -> V*` is the identity matrix mapping from V to V (i.e. V x V* -> R).
            # Equivalently, the adjoint of `c` is its first argument, which is a ufl.Argument defined on the
            # primal space of `c`.
            primal_arg, _ = form.arguments()
            # Returning the primal argument avoids explicit argument reconstruction, making it
            # a robust strategy for handling subclasses of `ufl.Coargument`.
            return primal_arg

        return super(Adjoint, cls).__new__(cls)

    def __init__(self, form):
        """Initialise."""
        BaseForm.__init__(self)

        if len(form.arguments()) != 2:
            raise ValueError("Can only take Adjoint of a 2-form.")

        self._form = form
        self.ufl_operands = (self._form,)
        self._domains = None
        self._hash = None
        self._repr = "Adjoint(%s)" % repr(self._form)

    def ufl_function_spaces(self):
        """Get the tuple of function spaces of the underlying form."""
        return self._form.ufl_function_spaces()

    def form(self):
        """Return the form."""
        return self._form

    def _analyze_form_arguments(self):
        """The arguments of adjoint are the reverse of the form arguments."""
        self._arguments = self._form.arguments()[::-1]
        self._coefficients = self._form.coefficients()

    def _analyze_domains(self):
        """Analyze which domains can be found in Adjoint."""
        from ufl.domain import join_domains

        # Collect unique domains
        self._domains = join_domains([e.ufl_domain() for e in self.ufl_operands])

    def equals(self, other):
        """Check if two Adjoints are equal."""
        if type(other) is not Adjoint:
            return False
        if self is other:
            return True
        # Make sure we are returning a boolean as the equality can result in a `ufl.Equation`
        # if the underlying objects are `ufl.BaseForm`.
        return bool(self._form == other._form)

    def __str__(self):
        """Format as a string."""
        return f"Adjoint({self._form})"

    def __repr__(self):
        """Representation."""
        return self._repr

    def __hash__(self):
        """Hash."""
        if self._hash is None:
            self._hash = hash(("Adjoint", hash(self._form)))
        return self._hash
