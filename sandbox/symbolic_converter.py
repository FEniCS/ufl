
class SymbolicConverter(Transformer):
    def __init__(self):
        Transformer.__init__(self)
        # TODO: Port framework from newsfc:
        self._component = tuple() 
        self._indices = {}
        self._lib = swiginac
        self._namespace = swiginac.__dict__
    
    # ------------ Sensible defaults are implemented in Transformer
    
    #def expr(self, o, *ops):
    #    raise RuntimeError('TODO')
    #
    #def terminal(self, o):
    #    raise RuntimeError('TODO')
    #
    #def variable(self, o):
    #    raise RuntimeError('TODO')
    
    # ------------ Constants
    
    def scalar_value(self, o):
        return o._value
    
    #def scalar_something(self, o):
    #    return o._value
    #
    #def int_value(self, o):
    #    return o._value
    #
    #def float_value(self, o):
    #    return o._value
    
    def zero(self, o):
        return 0
    
    def identity(self, o):
        return 1 if self._component[0] == self._component[1] else 0
    
    # ------------ Input terminals from form compiler
    
    def basis_function(self, o):
        raise RuntimeError('TODO')
    
    def function(self, o):
        raise RuntimeError('TODO')
    
    def constant(self, o):
        raise RuntimeError('TODO')
    
    def vector_constant(self, o):
        raise RuntimeError('TODO')

    def tensor_constant(self, o):
        raise RuntimeError('TODO')
    
    def facet_normal(self, o):
        raise RuntimeError('TODO')
    
    # ------------ Mapping through python operators
    
    def abs(self, o, a):
        return abs(a)
    
    def division(self, o, a, b):
        return a / b
    
    def power(self, o, a, b):
        return a**b

    def sum(self, o, *ops):
        return sum(ops)
    
    # ------------ Indexing stuff
    
    def product(self, o, *ops):
        return product(ops) # TODO: Add indexing stuff

    def component_tensor(self, o):
        raise RuntimeError('TODO')
        # TODO: Map current component to current indices, then call visit
    
    def indexed(self, o):
        raise RuntimeError('TODO')
        # TODO: Map current indices to current component, then call visit
    
    def list_tensor(self, o):
        raise RuntimeError('TODO')
        # TODO: Map current component to item in list tensor
    
    def multi_index(self, o):
        raise RuntimeError('TODO')
        # TODO: Map indices to tuple of ints
    
    # ------------ Basic differential operators
    
    def spatial_derivative(self, o):
        raise RuntimeError('TODO')
        # TODO: Indexing stuff
    
    def variable_derivative(self, o):
        raise RuntimeError('TODO')
        # TODO: Indexing stuff
    
    # ------------ Basic math functions
    
    def math_function(self, o, a):
        return self._namespace[a._name](a)
    
    #def cos(self, o, a):
    #    return self._lib.cos(a)
    #
    #def exp(self, o, a):
    #    return self._lib.exp(a)
    #
    #def ln(self, o, a):
    #    return self._lib.ln(a)
    #
    #def sin(self, o, a):
    #    return self._lib.sin(a)
    #
    #def sqrt(self, o, a):
    #    return self._lib.sqrt(a)
    
    # ------------ Compound tensor operators
    
    def transposed(self, o, a):
        raise RuntimeError('TODO')
        # TODO: Compound expansion
    
    def trace(self, o, a):
        raise RuntimeError('TODO')
        # TODO: Compound expansion
    
    def cross(self, o, a, b):
        raise RuntimeError('TODO')
        # TODO: Compound expansion
    
    def deviatoric(self, o, a):
        raise RuntimeError('TODO')
        # TODO: Compound expansion
    
    def dot(self, o, a, b):
        raise RuntimeError('TODO')
        # TODO: Compound expansion
    
    def inner(self, o, a, b):
        raise RuntimeError('TODO')
        # TODO: Compound expansion
        return self.visit(
    
    def outer(self, o, a, b):
        raise RuntimeError('TODO')
        # TODO: Compound expansion
    
    def determinant(self, o, a):
        raise RuntimeError('TODO')
        # TODO: Compound expansion
    
    def cofactor(self, o, a):
        raise RuntimeError('TODO')
        # TODO: Compound expansion
    
    def inverse(self, o, a):
        raise RuntimeError('TODO')
        # TODO: Compound expansion
    
    # ------------ Compound differential operators
    
    def div(self, o, a):
        raise RuntimeError('TODO')
        # TODO: Compound expansion
    
    def grad(self, o, a):
        raise RuntimeError('TODO')
        # TODO: Compound expansion
    
    def curl(self, o, a):
        raise RuntimeError('TODO')
        # TODO: Compound expansion
    
    def rot(self, o, a):
        raise RuntimeError('TODO')
        # TODO: Compound expansion
    
    # ------------ Conditionals
    
    def conditional(self, o, *ops):
        raise RuntimeError('TODO')
        # TODO: Do stuff with code structure
    
    def condition(self, o, *ops):
        raise RuntimeError('TODO')
        # TODO: Do stuff with code structure
    
    #def ge(self, o, a, b):
    #    raise RuntimeError('TODO')
    #
    #def gt(self, o, a, b):
    #    raise RuntimeError('TODO')
    #
    #def eq(self, o, a, b):
    #    raise RuntimeError('TODO')
    #
    #def le(self, o, a, b):
    #    raise RuntimeError('TODO')
    #
    #def lt(self, o, a, b):
    #    raise RuntimeError('TODO')
    #
    #def ne(self, o, a, b):
    #    raise RuntimeError('TODO')
    
    # ------------ Restrictions
    
    def restricted(self, o, a):
        raise RuntimeError('TODO')
    
    def negative_restricted(self, o, a):
        raise RuntimeError('TODO')
    
    def positive_restricted(self, o, a):
        raise RuntimeError('TODO')
