"""Front-end for AD routines."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-12-28 -- 2009-01-09"

from itertools import izip
from ufl.log import debug
from ufl.assertions import ufl_assert
from ufl.classes import Terminal, Expr, Derivative, Tuple, SpatialDerivative, VariableDerivative, FunctionDerivative, FiniteElement, TestFunction, Function
#from ufl.algorithms import *
from ufl.algorithms.transformations import transform_integrands, expand_compounds

from ufl.algorithms.reverse_ad import reverse_ad
from ufl.algorithms.forward_ad import forward_ad

#TODO:
#- Make basic tests for the below code!
#- Implement forward mode AD.
#- Implement reverse mode AD.
#- Nonscalar expressions have been ignored in the below code!
#- Nonscalar derivatives have been ignored in the below code!
indentation = ""
def apply_ad(expr, ad_routine):
    global indentation
    indentation += " "*4

    debug("")
    debug(indentation + "In apply_ad, expr = " + str(expr))
    debug(indentation + repr(expr))
    
    # Return terminals
    if isinstance(expr, Terminal):
        debug("indentation" + "Returning terminal" + repr(expr))
        indentation = indentation[:-4]
        return expr
    
    # Handle children first to make sure they have no derivatives themselves
    ops  = expr.operands()
    ops2 = tuple(apply_ad(o, ad_routine) for o in ops)
    
    # Reuse or reconstruct
    #if all(a is b for (a,b) in izip(ops, ops2)):
    if ops == ops2:
        debug(indentation + "Reusing " + str(expr))
        expr2 = expr
    else:
        c = expr._uflid
        debug(indentation + "Reconstructing from type and ops: ")
        debug(indentation +  "  c = " + str(c))
        debug(indentation +  "  ops2 = " + "\n".join(map(str,ops2)))
        expr2 = c(*ops2)
    
    # Evaluate derivative before returning
    if isinstance(expr, Derivative):
        debug(indentation + "AD in: " + str(expr))
        expr2 = ad_routine(expr2)
        debug(indentation + "AD out: " + str(expr2))
    
    debug(indentation + "Returning expr2 = " + str(expr2))
    debug(indentation + repr(expr2))
    debug("")
    indentation = indentation[:-4]
    return expr2

# TODO: While it works fine to propagate d/dx to terminals in
#     (a + b).dx(i) => (a.dx(i) + b.dx(i))
# this case doesn't work out as easily:
#     f[i].dx(i) => f.dx(i)[i] # rhs here doesn't make sense
# What is the best way to handle this?
# We must keep this in mind during extension
# of AD to tensor expressions!
def expand_derivatives(form):
    """Expand all derivatives of expr.
    
    NB! This functionality is not finished!
    
    In the returned expression g which is mathematically
    equivalent to expr, there are no VariableDerivative
    or FunctionDerivative objects left, and SpatialDerivative
    objects have been propagated to Terminal nodes."""
    
    cell = form.cell()
    ufl_assert(cell is not None, "Need to know the spatial dimension to compute derivatives.")
    spatial_dim = cell.dim()
    
    def _expand_derivatives(expression):
        expression = expand_compounds(expression, spatial_dim)
        return apply_ad(expression, forward_ad)
        #return apply_ad(expression, reverse_ad)
    
    if isinstance(form, Expr):
        return _expand_derivatives(form)
    return transform_integrands(form, _expand_derivatives)

if __name__ == "__main__":
    from ufl import triangle, FiniteElement, TestFunction, Function, expand_derivatives
    e = FiniteElement("CG", triangle, 1)
    v = TestFunction(e)
    f = Function(e)
    a = f*v*dx
    da = expand_derivatives(a)
    print 
    print a
    print
    print da 
    print


#v
#
#In apply_ad, expr = ([Rank 1 tensor A, such that A_{i_{9}} = (d[v_{-2}] / dx_i_{9})][i_{10}] * [Rank 1 tensor A, such that A_{i_{8}} = (d[v_{-1}] / dx_i_{8})][i_{10}])
#as_tensor(v.dx(i9), i9)[i10] * as_tensor(u.dx(i8), i8)[i10]
#
#
#    In apply_ad, expr = [Rank 1 tensor A, such that A_{i_{9}} = (d[v_{-2}] / dx_i_{9})][i_{10}]
#    Indexed(ComponentTensor(SpatialDerivative(BasisFunction(FiniteElement('Lagrange', Cell('triangle', 1), 1), -2), MultiIndex((Index(9),), 1)), MultiIndex((Index(9),), 1)), MultiIndex((Index(10),), 1))
#    as_tensor(v.dx(i9), i9)[i10]
#
#    In apply_ad, expr = [Rank 1 tensor A, such that A_{i_{9}} = (d[v_{-2}] / dx_i_{9})]
#    ComponentTensor(SpatialDerivative(BasisFunction(FiniteElement('Lagrange', Cell('triangle', 1), 1), -2), MultiIndex((Index(9),), 1)), MultiIndex((Index(9),), 1))
#    as_tensor(v.dx(i9), i9)
#
#        In apply_ad, expr = (d[v_{-2}] / dx_i_{9})
#        SpatialDerivative(BasisFunction(FiniteElement('Lagrange', Cell('triangle', 1), 1), -2), MultiIndex((Index(9),), 1))
#        v.dx(i9)
#
#            In apply_ad, expr = v_{-2}
#            BasisFunction(FiniteElement('Lagrange', Cell('triangle', 1), 1), -2)
#            v
#
#            Got terminal
#            v
#
#            In apply_ad, expr = i_{9}
#            MultiIndex((Index(9),), 1)
#            i9
#
#            Got terminal
#            i9
#        reusing
#        v.dx(i9)
#    Returning expr2 = [Zero tensor with shape () and free indices ()]
#    Zero((), (), {})
#
#
#In apply_ad, expr = i_{9}
#MultiIndex((Index(9),), 1)
#
#Got terminal
#c = <class 'ufl.tensors.ComponentTensor'>
#ops2 = [Zero tensor with shape () and free indices ()]
#i_{9}
#Missing indices set([Index(9)]) in expression [Zero tensor with shape () and free indices ()].
#

