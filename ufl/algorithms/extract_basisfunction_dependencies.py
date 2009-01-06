
def transform(expression, handlers):
    """Convert a UFLExpression according to rules defined by
    the mapping handlers = dict: class -> conversion function."""
    if isinstance(expression, Terminal):
        ops = ()
    else:
        ops = [transform(o, handlers) for o in expression.operands()]
    c = expression._uflid
    h = handlers.get(c, None)
    if c is None:
        ufl_error("Didn't find class %s among handlers." % c)
    return h(expression, *ops)

class NotMultiLinearException(Exception):
    pass

def extract_basisfunction_dependencies(expression):
    "TODO: Document me."
    def not_implemented(x, *ops):
        ufl_error("No handler implemented in extract_basisfunction_dependencies for '%s'" % str(x._uflid))
    h = UFLTypeDefaultDict(not_implemented)
    
    # Default for terminals: no dependency on basis functions 
    def h_terminal(x):
        return frozenset()
    for c in terminal_classes:
        h[c] = h_terminal
    
    def h_basisfunction(x):
        return frozenset((frozenset((x,)),))
    h[BasisFunction] = h_basisfunction
    
    # Default for nonterminals: nonlinear in all arguments 
    def h_nonlinear(x, *opdeps):
        for o in opdeps:
            if o: raise NotMultiLinearException, repr(x)
        return frozenset()
    for c in nonterminal_classes:
        h[c] = h_nonlinear
    
    # Some nonterminals are linear in their single argument 
    def h_linear(x, a):
        return a
    h[Grad] = h_linear
    h[Div] = h_linear
    h[Curl] = h_linear
    h[Rot] = h_linear
    h[Transposed] = h_linear
    h[Trace] = h_linear
    h[Skew] = h_linear
    h[PositiveRestricted] = h_linear
    h[NegativeRestricted] = h_linear

    def h_indexed(x, f, i):
        if i: raise NotMultiLinearException, repr(x)
        return f
    h[Indexed] = h_indexed
    
    def h_diff(x, a, b):
        if b: raise NotMultiLinearException, repr(x)
        return a
    h[SpatialDerivative] = h_diff
    h[VariableDerivative] = h_diff

    def h_variable(x):
        return extract_basisfunction_dependencies(x._expression)
    h[Variable] = h_variable

    def h_componenttensor(x, f, i):
        return f
    h[ComponentTensor] = h_componenttensor
    
    # Require same dependencies for all listtensor entries
    def h_listtensor(x, *opdeps):
        d = opdeps[0]
        for d2 in opdeps[1:]:
            if not d == d2:
                raise NotMultiLinearException, repr(x)
        return d
    h[ListTensor] = h_listtensor
    
    # Considering EQ, NE, LE, GE, LT, GT nonlinear in this context. 
    def h_conditional(x, cond, t, f):
        if cond or (not t == f):
            raise NotMultiLinearException, repr(x)
        return t

    # Basis functions cannot be in the denominator
    def h_division(x, a, b):
        if b: raise NotMultiLinearException, repr(x)
        return a
    h[Division] = h_division

    # Sums can contain both linear and bilinear terms (we could change this to require that all operands have the same dependencies)
    def h_sum(x, *opdeps):
        deps = set(opdeps[0])
        for o in opdeps[1:]:
            # o is here a set of sets
            deps |= o
        return frozenset(deps)
    h[Sum] = h_sum
    
    # Product operands should not depend on the same basis functions
    def h_product(x, *opdeps):
        c = []
        adeps, bdeps = opdeps # TODO: Generalize to any number of operands
        for ad in adeps:
            for bd in bdeps:
                cd = ad | bd
                if not len(cd) == len(ad) + len(bd):
                    raise NotMultiLinearException, repr(x)
                c.append(cd)
        return frozenset(c)
    h[Product] = h_product
    h[Inner] = h_product
    h[Outer] = h_product
    h[Dot] = h_product
    h[Cross] = h_product
    
    return transform(expression, h)

