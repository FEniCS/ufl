# -*- coding: utf-8 -*-


from itertools import chain

from ufl.log import UFLException
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dag
from ufl.classes import Argument, Zero


class ArityMismatch(UFLException):
    pass


class ArityChecker(MultiFunction):
    def __init__(self, arguments):
        MultiFunction.__init__(self)
        self.arguments = arguments
        self._et = ()

    def terminal(self, o):
        return self._et

    def argument(self, o):
        return (o,)

    def nonlinear_operator(self, o):
        # Cutoff traversal by not having *ops in argument list of this handler.
        # Traverse only the terminals under here the fastest way we know of:
        for t in traverse_unique_terminals(o):
            if t._ufl_typecode_ == Argument._ufl_typecode_:
                raise ArityMismatch("Applying nonlinear operator {0} to expression depending on form argument {1}.".format(o._ufl_class_.__name__, t))
        return self._et

    expr = nonlinear_operator

    def sum(self, o, a, b):
        if a != b:
            raise ArityMismatch("Adding expressions with non-matching form arguments {0} vs {1}.".format(a, b))
        return a

    def division(self, o, a, b):
        if b:
            raise ArityMismatch("Cannot divide by form argument {0}.".format(b))
        return a

    def product(self, o, a, b):
        if a and b:
            # Check that we don't have test*test, trial*trial, even for different parts in a block system
            anumbers = set(x.number() for x in a)
            for x in b:
                if x.number() in anumbers:
                    raise ArityMismatch("Multiplying expressions with overlapping form argument number {0}, argument is {1}.".format(x.number(), x))
            # Combine argument lists
            c = tuple(sorted(set(a + b), key=lambda x: (x.number(), x.part())))
            # Check that we don't have any arguments shared between a and b
            if len(c) != len(a) + len(b):
                raise ArityMismatch("Multiplying expressions with overlapping form arguments {0} vs {1}.".format(a, b))
            # It's fine for argument parts to overlap
            return c
        elif a:
            return a
        else:
            return b

    # inner, outer and dot all behave as product
    inner = product
    outer = product
    dot = product

    def linear_operator(self, o, a):
        return a

    # Positive and negative restrictions behave as linear operators
    positive_restricted = linear_operator
    negative_restricted = linear_operator

    # Cell and facet average are linear operators
    cell_avg = linear_operator
    facet_avg = linear_operator

    # Grad is a linear operator
    grad = linear_operator
    reference_grad = linear_operator
    reference_value = linear_operator

    # Does it make sense to have a Variable(Argument)? I see no problem.
    def variable(self, o, f, l):
        return f

    # Conditional is linear on each side of the condition
    def conditional(self, o, c, a, b):
        if c:
            raise ArityMismatch("Condition cannot depend on form arguments ({0}).".format(a))
        if a and isinstance(o.ufl_operands[2], Zero):
            # Allow conditional(c, arg, 0)
            return a
        elif b and isinstance(o.ufl_operands[1], Zero):
            # Allow conditional(c, 0, arg)
            return b
        elif a == b:
            # Allow conditional(c, test, test)
            return a
        else:
            # Do not allow e.g. conditional(c, test, trial), conditional(c, test, nonzeroconstant)
            raise ArityMismatch("Conditional subexpressions with non-matching form arguments {0} vs {1}.".format(a, b))

    def linear_indexed_type(self, o, a, i):
        return a

    # All of these indexed thingies behave as a linear_indexed_type
    indexed = linear_indexed_type
    index_sum = linear_indexed_type
    component_tensor = linear_indexed_type

    def list_tensor(self, o, *ops):
        args = set(chain(*ops))
        if args:
            # Check that each list tensor component has the same argument numbers (ignoring parts)
            numbers = set(tuple(sorted(set(arg.number() for arg in op))) for op in ops)
            if () in numbers: # Allow e.g. <v[0], 0, v[1]> but not <v[0], u[0]>
                numbers.remove(())
            if len(numbers) > 1:
                raise ArityMismatch("Listtensor components must depend on the same argument numbers, found {0}.".format(numbers))

            # Allow different parts with the same number
            return tuple(sorted(args, key=lambda x: (x.number(), x.part())))
        else:
            # No argument dependencies
            return self._et

def check_integrand_arity(expr, arguments):
    arguments = tuple(sorted(set(arguments), key=lambda x: (x.number(), x.part())))
    rules = ArityChecker(arguments)
    args = map_expr_dag(rules, expr, compress=False)
    if args != arguments:
        raise ArityMismatch("Integrand arguments {0} differ from form arguments {1}.".format(args, arguments))

def check_form_arity(form, arguments):
    for itg in form.integrals():
        check_integrand_arity(itg.integrand(), arguments)
