
from all import *

def transform(expression, handlers):
    """Convert a UFLExpression according to rules defined by
    a mapping handlers = dict: class -> conversion function."""
    if isinstance(expression, Terminal):
        return d[expression.__class__](expression)
    ops = [d[o.__class__](o) for o in expression.operands()]
    return d[expression.__class__](*ops)

def ufl_handlers():
    """This function constructs a handler dict for transform
    which can be used to reconstruct a ufl expression through
    transform(...), which of course makes no sense but is
    useful for testing."""
    # Show a clear error message if we miss some types here:
    def not_implemented(x):
        ufl_error("No handler defined for %s in ufl_handlers." % x.__class__)
    d = defaultdict(not_implemented)
    # Terminal objects are simply reused:
    def this(x):
        return x
    d[Number] = this
    d[Symbol] = this
    d[BasisFunction] = this
    d[Function] = this
    d[Constant] = this
    d[FacetNormal] = this
    d[MeshSize] = this
    d[Identity] = this
    # Non-terminal objects should have appropriate constructors:
    d[Variable]  = Variable
    d[Sum]       = Sum
    d[Product]   = Product
    d[Division]  = Division
    d[Power]     = Power
    d[Mod]       = Mod
    d[Abs]       = Abs
    d[Transpose] = Transpose
    d[Indexed]   = Indexed
    d[PartialDerivative] = PartialDerivative
    d[Diff] = Diff
    d[Grad] = Grad
    d[Div]  = Div
    d[Curl] = Curl
    d[Rot]  = Rot
    d[FiniteElement] = FiniteElement
    d[MixedElement]  = MixedElement
    d[VectorElement] = VectorElement
    d[TensorElement] = TensorElement
    d[MathFunction]  = MathFunction
    d[Outer] = Outer
    d[Inner] = Inner
    d[Dot]   = Dot
    d[Cross] = Cross
    d[Trace] = Trace
    d[Determinant] = Determinant
    d[Inverse]     = Inverse
    d[Deviatoric]  = Deviatoric
    d[Cofactor]    = Cofactor
    d[ListVector]  = ListVector
    d[ListMatrix]  = ListMatrix
    d[Tensor]      = Tensor
    return d

def sfc_handlers():
    # Show a clear error message if we miss some types here:
    def not_implemented(x):
        ufl_error("No handler defined for %s in sfl_handlers." % x.__class__)
    d = defaultdict(not_implemented)
    # FIXME
    return d

def ffc_handlers():
    # Show a clear error message if we miss some types here:
    def not_implemented(x):
        ufl_error("No handler defined for %s in ffl_handlers." % x.__class__)
    d = defaultdict(not_implemented)
    # FIXME
    return d

def ufl2ufl(expression):
    handlers = ufl_handlers()
    return transform(expression, handlers)

def ufl2sfc(expression):
    handlers = sfc_handlers()
    return transform(expression, handlers)

def ufl2ffc(expression):
    handlers = ffc_handlers()
    return transform(expression, handlers)

