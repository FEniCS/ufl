

from itertools import chain

from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.log import UFLException

class ArityMismatch(UFLException):
    pass

class ArityChecker(MultiFunction):
    def __init__(self, arguments):
        MultiFunction.__init__(self)
        self.arguments = arguments

    def terminal(self, o):
        return ()

    def argument(self, o):
        #identifier = (o.number(), o.part())
        #return [identifier]
        return (o,)

    def nonlinear_operator(self, o, *ops):
        if any(ops):
            raise ArityMismatch("Applying nonlinear operator to expression depending on form arguments {0}.".format(a))
        return ops[0]

    expr = nonlinear_operator

    def sum(self, o, a, b):
        if a == b:
            return a
        else:
            raise ArityMismatch("Adding expressions with non-matching form arguments {0} vs {1}.".format(a, b))

    def division(self, o, a, b):
        if b:
            raise ArityMismatch("Cannot divide by form argument {0}.".format(b))
        return a

    def product(self, o, a, b):
        if a and b:
            # Combine argument lists
            c = tuple(sorted(set(a + b), key=lambda x: (x.number(), x.part())))
            # Check that we don't have any arguments shared between a and b
            if len(c) != len(a) + len(b):
                raise ArityMismatch("Multiplying expressions with overlapping form arguments {0} vs {1}.".format(a, b))
            # Check that we don't have test*test, trial*trial, even for different parts in a block system
            anumbers = set(x.number() for x in a)
            bnumbers = set(x.number() for x in b)
            cnumbers = set(x.number() for x in c)
            if len(cnumbers) != len(anumbers) + len(bnumbers):
                raise ArityMismatch("Multiplying expressions with overlapping form argument numbers {0} vs {1}.".format(a, b))
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

    # Does it make sense to have a Variable(Argument)? I see no problem.
    def variable(self, o, f, l):
        return f

    def linear_indexed_type(self, o, a, i):
        return a

    # All of these indexed thingies behave as a linear_indexed_type
    indexed = linear_indexed_type
    index_sum = linear_indexed_type
    component_tensor = linear_indexed_type

    def list_tensor(self, o, *ops):
        numbers = set(arg.number() for op in ops for arg in op)
        if len(numbers) == 0:
            return ops[0]
        elif len(numbers) == 1:
            # Allow different parts with the same number
            return tuple(sorted(set(chain(*ops)), key=lambda x: (x.number(), x.part())))
        else:
            raise ArityMismatch("A listtensor can only hold form argument with the same numbers, found {0}.".format(ops))

def check_integrand_arity(expr, arguments):
    arguments = tuple(sorted(set(arguments), key=lambda x: (x.number(), x.part())))
    rules = ArityChecker(arguments)
    args = map_expr_dag(rules, expr, compress=False)
    if args != arguments:
        raise ArityMismatch("Integrand arguments {0} differ from form arguments {1}.".format(args, arguments))

def check_form_arity(form, arguments):
    for itg in form.integrals():
        check_integrand_arity(itg.integrand(), arguments)
