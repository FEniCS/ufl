#!/usr/bin/env python

"""
Utility algorithms for inspection, conversion or transformation
of UFL objects in various ways.

(Currently, some utility functions are located in visitor.py,
some in traversal.py, and some in transformers.py,
depending on the method of implementation.
This file should contain userfriendly front-ends
to all the utility algorithms that we want to expose.)
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-05-16"

from itertools import chain

from base import *
from visitor import *
from traversal import *
from output import *

### Utilities to deal with form files

def load_forms(filename):
    # Read form file
    code = "from ufl import *\n"
    code += "\n".join(file(filename).readlines())
    namespace = {}
    try:
        exec(code, namespace)
    except:
        tmpname = "ufl_analyse_tmp_form"
        tmpfile = tmpname + ".py"
        f = file(tmpfile, "w")
        f.write(code)
        f.close()
        ufl_info("""\
An exception occured during evaluation of form file.
To find the location of the error, a temporary script
'%s' has been created and will now be run:""" % tmpfile)
        m = __import__(tmpname)
        ufl_error("Aborting load_forms.")
    
    # Extract Form objects
    forms = []
    for k,v in namespace.iteritems():
        if isinstance(v, Form):
            forms.append((k,v))
    
    return forms

def analyse_form(f): # TODO: Analyse validity of forms any way we can
    errors = None
    return errors

### Utilities to extract information from an expression:

def extract_type(a, ufl_type):
    """Returns a set of all objects of class ufl_type found in a.
    The argument a can be a Form, Integral or UFLObject.
    """
    iter = (o for e in iter_expressions(a) \
              for o in iter_child_first(e) \
              if isinstance(o, ufl_type) )
    return set(iter)

def basisfunctions(a):
    """Build a sorted list of all basisfunctions in a,
    which can be a Form, Integral or UFLObject.
    """
    # build set of all unique basisfunctions
    s = extract_type(a, BasisFunction)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x._count, y._count))
    return l

def coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or UFLObject.
    """
    # build set of all unique coefficients
    s = extract_type(a, Function)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x._count, y._count))
    return l

# alternative implementation, kept as an example:
def _coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or UFLObject.
    """
    # build set of all unique coefficients
    s = set()
    def func(o):
        if isinstance(o, Function):
            s.add(o)
    walk(a, func)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x._count, y._count))
    return l

def elements(a):
    """Returns a sorted list of all elements used in a."""
    return [f._element for f in chain(basisfunctions(a), coefficients(a))]

def unique_elements(a):
    """Returns a sorted list of all elements used in a."""
    elements = set()
    for f in chain(basisfunctions(a), coefficients(a)):
        elements.add(f._element)
    return elements

def classes(a):
    """Returns a set of all unique classes used in a (subclasses of UFLObject)."""
    classes = set()
    for e in iter_expressions(a):
        for o in iter_child_first(e):
            classes.add(o.__class__)
    return classes

def variables(a):
    """Returns a set of all variables in a,
    which can be a Form, Integral or UFLObject.
    """
    return extract_type(a, Variable)

def duplications(a):
    """Returns a set of all repeated expressions in u."""
    handled = set()
    duplicated = set()
    for e in iter_expressions(a):
        for o in iter_parent_first(e):
            if o in handled:
                duplicated.add(o)
            handled.add(o)
    return duplicated





def integral_info(itg):
    s  = "  Integral over %s domain %d:\n" % (itg._domain_type, itg._domain_id)
    s += "    Integrand expression representation:\n"
    s += "      %s\n" % repr(itg._integrand)
    s += "    Integrand expression short form:\n"
    s += "      %s" % str(itg._integrand)
    return s

def form_info(a):
    ufl_assert(isinstance(a, Form), "Expecting a Form.")
    
    bf = basisfunctions(a)
    cf = coefficients(a)
    
    ci = a.cell_integrals()
    ei = a.exterior_facet_integrals()
    ii = a.interior_facet_integrals()
    
    s  = "Form info:\n"
    s += "  rank:                          %d\n" % len(bf)
    s += "  num_coefficients:              %d\n" % len(cf)
    s += "  num_cell_integrals:            %d\n" % len(ci)
    s += "  num_exterior_facet_integrals:  %d\n" % len(ei)
    s += "  num_interior_facet_integrals:  %d\n" % len(ii)
    
    for f in cf:
        if f._name:
            s += "\n"
            s += "  Function %d is named '%s'" % (f._count, f._name)
    s += "\n"
    
    for itg in ci:
        s += "\n"
        s += integral_info(itg)
    for itg in ei:
        s += "\n"
        s += integral_info(itg)
    for itg in ii:
        s += "\n"
        s += integral_info(itg)
    return s




### Utilities to transform expression in some way:

def transform_integrands(a, transformation):
    """Transform all integrands in a form with a transformation function.
    
    Example usage:
      b = transform_integrands(a, flatten)
    """
    
    ufl_assert(isinstance(a, Form), "Expecting a Form.")
    
    integrals = []
    for itg in a.integrals():
        integrand = transformation(itg._integrand)
        newitg = Integral(itg._domain_type, itg._domain_id, integrand)
        integrals.append(newitg)
    
    return Form(integrals)




def transformation_template(a):
    """Template function for transformations of expression tree."""
    ufl_warning("This function isn't intended for production use, only testing and instruction.")
    
    # Terminal objects are usually replaced by something fixed:
    if isinstance(a, Terminal):
        if isinstance(a, Number):
            return a
        if isinstance(a, Symbol):
            return a
        if isinstance(a, Function):
            return a
        if isinstance(a, BasisFunction):
            return a
        if isinstance(a, FacetNormal):
            return a
        if isinstance(a, MeshSize):
            return a
        ufl_error("Missing handler for Terminal subclass %s in transformation_template." % str(a.__class__))
    
    # May or may not pass the variable "barrier":
    if isinstance(a, Variable):
        return a
    
    # Handle all operands first:
    operands = []
    for o in a.operands():
        b = transformation_template(o)
        operands.append(b)
    
    
    
    # FIXME: change to dictionary lookup for speed and easyer extensibility: 
    #namespace = {
    #    Sum:                 (lambda ops: sum(ops)),
    #    Product:             (lambda ops: product(ops)),
    #    PartialDerivative:   (lambda ops: PartialDerivative(*ops)),
    #    }
    #
    #transform = namespace.get(type(a), None)
    #ufl_assert(not transform is None, "Missing handler for non-terminal class %s in transformation_template." % str(type(a)))
    #return transform(*operands)
    
    
    # FIXME: add handlers for all UFLObject subclasses here
    
    
    namespace = ufl
    
    # base.py:
    if isinstance(a, Sum):
        return sum(operands)
    if isinstance(a, Product):
        return product(operands)
    if isinstance(a, PartialDerivative):
        return namespace.partial_derivative(*operands)
    
    # tensoroperators.py:
    if isinstance(a, Inner):
        return namespace.inner(*operands)
    if isinstance(a, Dot):
        return namespace.dot(*operands)
    
    # diffoperators.py:
    if isinstance(a, Grad):
        return namespace.grad(*operands)
    if isinstance(a, Div):
        return namespace.div(*operands)
    if isinstance(a, Curl):
        return namespace.curl(*operands)
    if isinstance(a, Rot):
        return namespace.rot(*operands)
    
    ufl_error("Missing handler for non-terminal class %s in transformation_template." % str(a.__class__))
    #return a.__class__(*operands)   




def _strip_variables(a):
    "Auxilliary procedure for strip_variables."
    
    if isinstance(a, Terminal):
        return a, False
    
    if isinstance(a, Variable):
        b, changed = _strip_variables(a._expression)
        return b, changed
    
    operands = []
    changed = False
    for o in a.operands():
        b, c = _strip_variables(o)
        operands.append(b)
        if c: changed = True
    
    if changed:
        return a.__class__(*operands), True
    # else: no change, reuse object
    return a, False

def strip_variables(a):
    """Strip Variable objects from a, replacing them with their expressions."""
    ufl_assert(isinstance(a, UFLObject), "Expecting an UFLObject.")
    b, changed = _strip_variables(a)
    return b

# naive version, producing lots of extra objects:
def strip_variables2(a):
    """Strip Variable objects from a, replacing them with their expressions."""
    ufl_assert(isinstance(a, UFLObject), "Expecting an UFLObject.")
    
    if isinstance(a, Terminal):
        return a
    
    if isinstance(a, Variable):
        return strip_variables2(a._expression)
    
    operands = [strip_variables2(o) for o in a.operands()]
    
    return a.__class__(*operands)


def flatten(a):
    """Flatten (a+b)+(c+d) into a (a+b+c+d) and (a*b)*(c*d) into (a*b*c*d)."""
    ufl_assert(isinstance(a, UFLObject), "Expecting an UFLObject.")
    
    # Possible optimization:
    #     Reuse objects for subtrees with no
    #     flattened sums or products.
    #     The current procedure will create a new object
    #     for every single node in the tree.
    
    # TODO: Traverse variables or not?
    
    if isinstance(a, Terminal):
        return a
    
    myclass = a.__class__
    operands = []
    
    if isinstance(a, (Sum, Product)):
        for o in a.operands():
            b = flatten(o)
            if isinstance(b, myclass):
                operands.extend(b.operands())
            else:
                operands.append(b)
    else:
        for o in a.operands():
            b = flatten(o)
            operands.append(b)
    
    return myclass(*operands)



def renumber_indices(a):
    "Renumber indices in a contiguous count."
    
    ufl_warning("Not implemented!") # FIXME
    
    # 1) Get all indices
    # 2) Define a index number mapping
    # 3) Apply number map
    
    return a

def renumber_basisfunctions(a):
    "Renumber indices in a contiguous count."
    
    ufl_warning("Not implemented!") # FIXME
    
    # 1) Get all basisfunctions
    # 2) Define a basisfunction number mapping
    # 3) Apply number map
    
    return a

def renumber_functions(a):
    "Renumber indices in a contiguous count."
    
    ufl_warning("Not implemented!") # FIXME
    
    # 1) Get all functions
    # 2) Define a function number mapping
    # 3) Apply number map
    
    return a



def criteria_not_argument(a):
    return not isinstance(a, (Function, BasisFunction))

def criteria_not_trial_function(a):
    return not (isinstance(a, BasisFunction) and (a._count > 0 or a._count == -1))

def criteria_not_basis_function(a):
    return not isinstance(a, BasisFunction)

def _detect_argument_dependencies(a, criteria):
    """Detect edges in expression tree where subtrees
    depend on different stages of form arguments.
    A Variable object is inserted at each edge.
    
    Stage 0:  Subtrees that does not depend on any arguments.
    Stage 1:  Subtrees that does not depend on any basisfunctions (i.e., that only depend on coefficients).
    Stage 2:  Subtrees that does not depend on basisfunction 1 (i.e. trial function for a matrix)
    Stage 3:  Subtrees that does not depend on basisfunction 0 (i.e. test function)
    """
    ufl_warning("NB! Assumes renumbered basisfunctions! FIXME: Implement basisfunction renumbering.")
    
    if isinstance(a, Terminal):
        return a, criteria(a)
    
    operands = []
    crit = []
    for o in a.operands():
        b, c = _detect_argument_dependencies(o, criteria)
        operands.append(b)
        crit.append(c)
    
    # FIXME: finish this
    
    if False:
        return a   
    return a.__class__(*operands)   



def convert_diff_to_indices(u):
    "Convert differential operator objects Grad, Div, Curl and Rot to their componentwise representations"
    ufl_error("Not implemented") # FIXME: Implement
    
    return u


def substitute_indices(u, indices, values):
    "Substitute Index objects from list indices with corresponding fixed values from list values in expression u."
    ufl_error("Not implemented") # FIXME: Implement
    
    return u


def apply_summation(u):
    "Expand all repeated indices into explicit sums with fixed indices."
    ufl_error("Not implemented") # FIXME: Implement
    
    return u


def discover_indices(u):
    "Convert explicit sums into implicit sums (repeated indices)."
    ufl_error("Not implemented") # FIXME: Implement (like FFCs 'simplify' done by Marie)
    
    return u

