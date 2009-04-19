"""This file is useful for external code like tests and form compilers,
since it enables the syntax "from ufl.classes import FooBar" for getting
implementation details not exposed through the default ufl namespace."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-15 -- 2009-04-19"

# Modified by Anders Logg, 2008

from ufl.assertions import ufl_assert
from ufl.expr import Expr, Operator, WrapperType, AlgebraOperator
from ufl.terminal import Terminal, FormArgument, UtilityType, Tuple
from ufl.constantvalue import ConstantValue, Zero, ScalarValue, FloatValue, IntValue, ScalarSomething, Identity
from ufl.variable import Variable, Label
from ufl.finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from ufl.basisfunction import BasisFunction, TestFunction, TrialFunction
from ufl.function import Function, ConstantBase, VectorConstant, TensorConstant, Constant
from ufl.geometry import GeometricQuantity, SpatialCoordinate, FacetNormal
from ufl.indexing import MultiIndex, Indexed, IndexBase, Index, FixedIndex, IndexSum
from ufl.tensors import ListTensor, ComponentTensor
from ufl.algebra import Sum, Product, Division, Power, Abs
from ufl.tensoralgebra import CompoundTensorOperator, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Cofactor, Inverse, Deviatoric, Skew, Sym
from ufl.mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from ufl.restriction import Restricted, PositiveRestricted, NegativeRestricted
from ufl.differentiation import Derivative, CompoundDerivative, FunctionDerivative, SpatialDerivative, VariableDerivative, Grad, Div, Curl, Rot
from ufl.conditional import Condition, EQ, NE, LE, GE, LT, GT, Conditional
from ufl.form import Form
from ufl.integral import Measure, Integral

# Make sure we import exproperators which attaches special functions to Expr
from ufl import exproperators as __exproperators

# Collect all classes in lists
__all_classes       = (c for c in locals().values() if isinstance(c, type))
all_ufl_classes     = set(c for c in __all_classes if issubclass(c, Expr))
abstract_classes    = set(s for c in all_ufl_classes for s in c.mro()[1:-1])
abstract_classes.remove(Function)
ufl_classes         = set(c for c in all_ufl_classes if c not in abstract_classes)
terminal_classes    = set(c for c in all_ufl_classes if issubclass(c, Terminal))
nonterminal_classes = set(c for c in all_ufl_classes if not issubclass(c, Terminal))

# Add _uflclass and _classid to all classes:
from ufl.common import camel2underscore as _camel2underscore
for _i, _c in enumerate(all_ufl_classes):
    _c._classid = _i
    _c._uflclass = _c
    _c._handlername = _camel2underscore(_c.__name__)


# TOOD: Finish precedence list with all necessary types!
def _build_precedence_list():
    # --- From Wikipedia, precedence of operators in C:
    #1 	()   []   ->   .   :: 	Grouping, scope, array/member access
    #2 	 !   ~   -   +   *   &   sizeof   type cast ++x   --x   	(most) unary operations, sizeof and type casts
    #3 	*   /   % 	Multiplication, division, modulo
    #4 	+   - 	Addition and subtraction
    #5 	<<   >> 	Bitwise shift left and right
    #6 	<   <=   >   >= 	Comparisons: less-than, ...
    #7 	==   != 	Comparisons: equal and not equal
    #8 	& 	Bitwise AND
    #9 	^ 	Bitwise exclusive OR
    #10 	| 	Bitwise inclusive (normal) OR
    #11 	&& 	Logical AND
    #12 	|| 	Logical OR
    #13 	 ?: 	Conditional expression (ternary operator)
    #14 	=   +=   -=   *=   /=   %=   &=   |=   ^=   <<=   >>= 	Assignment operators

    # --- From python documetation, precedence of operators in python:
    #Operator 	Description
    #lambda 	Lambda expression
    #or 	Boolean OR
    #and 	Boolean AND
    #not x 	Boolean NOT
    #in, not in, is, is not, <, <=, >, >=, <>, !=, == 	Comparisons, including membership tests and identity tests,
    precedence_list.append((Condition,))

    #| 	Bitwise OR
    #^ 	Bitwise XOR
    #& 	Bitwise AND
    #<<, >> 	Shifts

    #+, - 	Addition and subtraction

    #sum_i a + b != sum_i (a + b) -> sum_i binds more than +
    precedence_list.append((Sum,))
    precedence_list.append((IndexSum,))
    
    #*, /, //, % 	Multiplication, division, remainder
    precedence_list.append((Product, Division, Inner, Outer, Dot, Cross,))

    precedence_list.append((Div, Grad, Curl, Rot,))

    #+x, -x, ~x 	Positive, negative, bitwise NOT

    #** 	Exponentiation [8]
    precedence_list.append((Power,))

    #x[index], x[index:index], x(arguments...), x.attribute 	Subscription, slicing, call, attribute reference
    precedence_list.append((Indexed,))

    #(expressions...), [expressions...], {key:datum...}, `expressions...` 	Binding or tuple display, list display, dictionary display, string conversion

    precedence_list.append((Conditional, Abs, MathFunction)) # operands always needs parenthesis unless terminal
    precedence_list.append((Terminal,)) # Never needs parenthesis around it

    # --- Obvious:
    # Example: (terminal) + (terminal) == terminal + terminal
    # -> terminal has higher precedence than sum, so no parenthesis are needed around it
    # Example: (a*b) + (c*d) == a*b + c*d
    # -> product has higher precedence than sum, so no parenthesis are needed around it

    # --- Alternatives (choose!):
    # Example: sin(a*b) == sin a*b
    # -> product has higher precedence than sin, so no parenthesis are needed around it
    # Example: sin(a*b) != sin a*b
    # -> product has lower precedence than sin, so parenthesis _are_ needed around it

    # Example: sin(a+b) == sin a*b
    # -> sum has higher precedence than sin, so no parenthesis are needed around it
    # Example: sin(a+b) != sin a*b
    # -> sum has lower precedence than sin, so parenthesis _are_ needed around it

    precedence_list = []

    precedence_list.append((Sum,))
    precedence_list.append((IndexSum,))
    
    # TODO: What to do with these?
    precedence_list.append((ListTensor, ComponentTensor))
    precedence_list.append((Restriction,))
    precedence_list.append((Conditional,))
    precedence_list.append((Condition,))
    
    precedence_list.append((Div, Grad, Curl, Rot,
                            SpatialDerivative, VariableDerivative,
                            Determinant, Trace, Cofactor, Inverse, Deviatoric, Skew, Sym))
    precedence_list.append((Product, Division, Cross, Dot, Outer, Inner))
    precedence_list.append((Indexed, Transposed, Power))

    # If parent operator binds stronger than child, must parenthesize child

    precedence_list.append((Abs, MathFunction,)) # Abs, Sqrt, Exp, Ln, Cos, Sin
    precedence_list.append((Variable,))

    precedence_list.append((Terminal,)) # terminal_classes

    k = 0
    for p in precedence_list:
        for c in p:
            c._precedence = k
        k += 1
    
    for c in all_ufl_classes:
        if not c in abstract_classes:
            ufl_assert(hasattr(c, "_precedence") and isinstance(c._precedence, int), "No precedence assigned to %s" % c.__name__)

def build_precedence_list():

    # foo(a+b) == foo((a+b)) if pres(foo) < pres(+)

    # What do we know about other types?
    #Power <= Indexed
    #Power <= Transposed

    precedence_list = []
    precedence_list.append((Operator,)) # Default behaviour: should always add parentheses
    precedence_list.append((Sum,))
    precedence_list.append((IndexSum,)) # sum_i a + b = (sum_i a) + b != sum_i (a + b), sum_i binds stronger than +, but weaker than product
    precedence_list.append((Product, Division,))
    precedence_list.append((Power,))
    precedence_list.append((Terminal,))
    return precedence_list

def build_precedences():
    precedence_list = build_precedence_list()

    k = 0
    for p in precedence_list:
        for c in p:
            c._precedence = k
        k += 1
    
    for c in all_ufl_classes:
        if not c in abstract_classes:
            ufl_assert(hasattr(c, "_precedence") and isinstance(c._precedence, int), "No precedence assigned to %s" % c.__name__)

build_precedences()

#__all__ = all_ufl_classes
