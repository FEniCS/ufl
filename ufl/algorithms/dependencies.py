"""This module defines utilities for working with dependencies of subexpressions."""


__authors__ = "Martin Sandve Alnes"
__date__ = "2008-10-01 -- 2008-11-21"

from collections import defaultdict
from itertools import izip, chain

from ufl.common import some_key, split_dict, or_tuples, and_tuples, UFLTypeDict
from ufl.output import ufl_assert, ufl_error, ufl_warning, ufl_debug
from ufl.permutation import compute_indices

# All classes:
from ufl.base import Expr, Terminal
from ufl.zero import Zero
from ufl.scalar import FloatValue, IntValue
from ufl.variable import Variable
from ufl.basisfunction import BasisFunction
from ufl.function import Function, Constant
from ufl.differentiation import SpatialDerivative
from ufl.geometry import FacetNormal
from ufl.tensoralgebra import Identity
from ufl.indexing import MultiIndex, Indexed, Index, FixedIndex
from ufl.tensors import ListTensor, ComponentTensor

# Lists of all Expr classes
from ufl.classes import ufl_classes, terminal_classes, nonterminal_classes

# Other algorithms:
from ufl.algorithms.analysis import extract_variables


class DependencySet:
    def __init__(self, basisfunctions, runtime=False, coordinates=False):
        
        # depends on runtime arguments (i.e. should be inside tabulate_tensor, otherwise it can be precomputed)
        self.runtime = runtime
        
        # depends on spatial coordinates (i.e. should be inside quadrature loop or integral)
        self.coordinates = coordinates
        
        # depends on basis function i (i.e. should be inside a loop where this basis function is defined)
        self.basisfunctions = tuple(basisfunctions)
    
    def iter(self):
        return chain((self.runtime, self.coordinates), self.basisfunctions)
    
    def size(self):
        return len(list(self.iter()))
    
    def covers(self, other):
        "Return True if all dependencies of other are covered by this dependency set."
        for a, o in zip(self.iter(), other.iter()):
            if o and not a:
                return False
        return True
    
    def __hash__(self):
        return hash(tuple(self.iter()))
    
    def __cmp__(self, other):
        for (a,b) in izip(self.iter(), other.iter()):
            if a < b: return -1
            if a > 1: return +1
        return 0
    
    def __or__(self, other):
        basisfunctions = or_tuples(self.basisfunctions, other.basisfunctions)
        d = DependencySet(basisfunctions,
                          self.runtime     or other.runtime,
                          self.coordinates or other.coordinates)
        return d

    def __and__(self, other):
        basisfunctions = and_tuples(self.basisfunctions, other.basisfunctions)
        d = DependencySet(basisfunctions,
                          runtime = self.runtime and other.runtime,
                          coordinates = self.coordinates and other.coordinates)
        return d

    def __str__(self):
        s = "DependencySet:\n"
        s += "{\n"
        s += "  self.runtime        = %s\n" % self.runtime
        s += "  self.coordinates    = %s\n" % self.coordinates 
        s += "  self.basisfunctions = %s\n" % str(self.basisfunctions) 
        s += "}"
        return s

class VariableInfo:
    def __init__(self, variable, deps):
        # Variable
        self.variable = variable
        # DependencySet
        self.deps = deps
        # VariableDerivativeVarSet -> VariableInfo
        self.diffcache = {}
    
    def __str__(self):
        s = "VariableInfo:\n"
        s += "{\n"
        s += "  self.variable = %s\n" % self.variable
        s += "  self.deps = %s\n" % self.deps
        s += "  self.diffcache = \n"
        s += "\n".join("    %s: %s" % (k,v) for (k,v) in self.diffcache.iteritems())
        s += "\n}"
        return s

class VariableDerivativeVarSet: # TODO: Use this?
    def __init__(self):
        self.fixed_spatial_directions = set() # Set of integer indices
        self.open_spatial_directions = set() # Set of Index objects that the stored expression uses for d/dx_i
        self.variables = set() # Set of variables we're differentiating w.r.t.

    def __hash__(self):
        return hash(tuple(d for d in self.fixed_spatial_directions) + \
                    tuple(d for d in self.open_spatial_directions) + \
                    tuple(d for d in self.variables))
    
    def __eq__(self, other):
        return self.fixed_spatial_directions == other.fixed_spatial_directions and \
               len(self.open_spatial_directions) == len(other.open_spatial_directions) and \
               self.variables == other.variables

    #def __contains__(self, other):
    #def __sub__(self, other):
    #def __add__(self, other):
    #def __cmp__(self, other):

class CodeStructure:
    def __init__(self):
        # A place to look up if a variable has been added to the stacks already
        self.variableinfo = {}   # variable count -> VariableInfo
        # One stack of variables for each dependency configuration
        self.stacks = defaultdict(list) # DependencySet -> [VariableInfo]
    
#===============================================================================
#    def __str__(self):
#        deps = DependencySet(TODO)
#        s = ""
#        s += "Variables independent of spatial coordinates:\n"
#        keys = [k for k in self.stacks.keys() if not k.coordinates]
#        for deps in keys:
#            if k.facet:
#                stack = self.stacks[k]
#                s += str(stack)
#        for deps in keys:
#            if not k.facet:
#                stack = self.stacks[k]
#                s += str(stack)
#        
#        s += "Variables dependent of spatial coordinates:\n"
#        for k in dependent:
#            s += k
#        
#        return s
# 
#    def split_stacks(self): # TODO: Remove this or change the concept. Doesn't belong here.
#        
#        # Start with all variable stacks
#        stacks = self.stacks
#        
#        # Split into precomputable and not
#        def criteria(k):
#            return not (k.spatial or any(k.coefficients) or k.facet)
#        precompute_stacks, runtime_stacks = split_dict(stacks, criteria)
#        
#        # Split into outside or inside quadrature loop
#        def criteria(k):
#            return k.spatial
#        quad_precompute_stacks, precompute_stacks = split_dict(precompute_stacks, criteria)
#        quad_runtime_stacks, runtime_stacks = split_dict(runtime_stacks, criteria)
#        
#        # Example! TODO: Make below code a function and apply to each stack group separately. 
#        stacks = quad_runtime_stacks
#        
#        # Split by basis function dependencies
#        
#        # TODO: Does this give us the order we want?
#        # Want to iterate faster over test function, i.e. (0,0), (1,0), (0,1), (1,1) 
#        keys = set(stacks.iterkeys())
#        perms = [p for p in compute_permutations(rank, 2) if p in keys] # TODO: NOT RIGHT!
#        for perm in perms:
#            def criteria(k):
#                return k.basisfunctions == perm
#            dep_stacks, stacks = split_dict(stacks, criteria)
#            
#            # TODO: For all permutations of basis function indices
#            # TODO: Input elementreps
#            sizes = [elementreps[i].local_dimension for i in range(self.rank) if perms[i]]
#            basis_function_perms = compute_indices(sizes)
#            for basis_function_perm in basis_function_perms:
#                context.update_basisfunction_permutation(basis_function_perm) # TODO: Map tuple to whole range of basisfunctions.
#                for stack in dep_stacks.itervalues():
#                    for (k,v) in stack.iteritems():
#                        s = context.variable_to_symbol(k)
#                        e = ufl_to_swiginac(v, context)
#                        context.add_token(s, e)
#===============================================================================

def _split_by_dependencies(expression, codestructure, terminal_deps):
    
    if isinstance(expression, Variable):
        c = expression._count
        info = codestructure.variableinfo.get(c, None)
        if info is not None:
            return info.variable, info.deps
        #codestructure.stacks[info.deps].append(info)
    
    if isinstance(expression, Terminal):
        h = terminal_deps[expression._uflid]
        deps = h(expression)
        if codestructure.stacks:
            ufl_assert(deps.size() == some_key(codestructure.stacks).size(),\
                       "Inconsistency in dependency definitions.")
        return expression, deps
    
    ops = expression.operands()
    ops2 = [_split_by_dependencies(o, codestructure, terminal_deps) for o in ops]
    deps = ops2[0][1]
    for o in ops2[1:]:
        deps |= o[1]
        
    ops3 = []
    for (v,vdeps) in ops2:
        if isinstance(v, Variable):
            # if this subexpression is a variable, it has already been added to the stack
            ufl_assert(v._count in codestructure.variableinfo, "")
        elif not vdeps == deps:
            # if this subexpression has other dependencies
            # than 'expression', store a variable for it 
            v = Variable(v) # FIXME: Check a variable cache to avoid duplications?
            vinfo = VariableInfo(v, vdeps)
            codestructure.variableinfo[v._count] = vinfo
            codestructure.stacks[vdeps].append(vinfo)
        ops3.append(v)
    
    if isinstance(expression, Variable):
        c = expression._count
        ufl_assert(c not in codestructure.variableinfo,
            "Shouldn't reach this point if the variable was already cached!")
        vinfo = VariableInfo(expression, deps)
        codestructure.variableinfo[c] = vinfo
        codestructure.stacks[deps].append(vinfo)
    
    # Try to reuse expression if nothing has changed:
    if any((o1 is not o3) for (o1,o3) in izip(ops, ops3)):
        e = type(expression)(*ops3)
    else:
        e = expression
    return e, deps

def split_by_dependencies(expression, formdata, basisfunction_deps, function_deps):
    """Split an expression into stacks of variables based
    on the dependencies of its subexpressions.
    
    @type expression: Expr
    @param expression: The expression to parse.
    @type basisfunction_deps: list(DependencySet)
    @param basisfunction_deps:
        A list of DependencySet objects, one for each
        BasisFunction in the Form the expression originates from.
    @type function_deps: list(DependencySet)
    @param function_deps:
        A list of DependencySet objects, one for each
        Function in the Form the expression originates from.
    @return (e, deps, codestructure):
        variableinfo: data structure with info about the final
                      variable representing input expression
        codestructure: data structure containing stacks of variables
        
    If the *_deps arguments are unknown, a safe way to invoke this function is::
    
        (variableinfo, codestructure) = split_by_dependencies(expression, formdata, [(True,True)]*rank, [(True,True)]*num_coefficients)
    """
    ufl_assert(isinstance(expression, Expr), "Expecting Expr.")
    
    # Exctract a list of all variables in expression 
    variables = extract_variables(expression)
    if isinstance(expression, Variable):
        ufl_assert(expression is variables[-1],
                   "Expecting the last result from extract_variables to be the input variable...")
    else:
        # Wrap root node in Variable for consistency below
        expression = Variable(expression)
        variables.append(expression)
    
    # Split each variable
    ds = DependencySplitter(formdata, basisfunction_deps, function_deps)
    for v in variables:
        print "Handling variable ", repr(v)
        vinfo = ds.handle(v)
        print "Done handling variable ", v
        print "Got vinfo:"
        print vinfo

    # How can I be sure we won't mess up the expressions of v and vinfo.variable, before and after splitting?
    # The answer is to never use v in the form compiler after this point,
    # and let v._count identify the variable in the code structure.
    
    return (vinfo, ds.codestructure)


class DependencySplitter:
    def __init__(self, formdata, basisfunction_deps, function_deps):
        self.formdata = formdata
        self.basisfunction_deps = basisfunction_deps
        self.function_deps = function_deps
        self.variables = []
        self.codestructure = CodeStructure()

        # First set default behaviour for dependencies
        self.handlers = UFLTypeDict()
        
        for c in terminal_classes:
            self.handlers[c] = self.no_deps
        for c in nonterminal_classes:
            self.handlers[c] = self.child_deps
        # Override with specific behaviour for some classes
        self.handlers[FacetNormal]   = self.get_facet_normal_deps
        self.handlers[BasisFunction] = self.get_basisfunction_deps
        self.handlers[Constant]      = self.get_function_deps
        self.handlers[Function]      = self.get_function_deps
        self.handlers[Variable]      = self.get_variable_deps
        self.handlers[SpatialDerivative] = self.get_spatial_derivative_deps
    
    def make_empty_deps(self):
        return DependencySet((False,)*len(self.basisfunction_deps))
    
    def no_deps(self, x):
        return x, self.make_empty_deps()
    
    def get_facet_normal_deps(self, x):
        deps = self.make_empty_deps()
        deps.runtime = True
        #deps.coordinates = True # TODO: Enable for higher order geometries.
        return x, deps
    
    def get_function_deps(self, x):
        print 
        print self.formdata.coefficient_renumbering
        print
        ufl_assert(x in self.formdata.coefficient_renumbering,
                   "Can't find function %s in renumbering dict!" % repr(x))
        i = self.formdata.coefficient_renumbering[x]
        d = self.function_deps[i]
        return x, d
    
    def get_basisfunction_deps(self, x):
        ufl_assert(x in self.formdata.basisfunction_renumbering,
                   "Can't find basis function %s in renumbering dict!" % repr(x))
        i = self.formdata.basisfunction_renumbering[x]
        d = self.basisfunction_deps[i]
        return x, d
    
    def get_variable_deps(self, x):
        vinfo = self.codestructure.variableinfo.get(x._count, None)
        ufl_assert(vinfo is not None, "Haven't handled variable in time: %s" % repr(x))
        return vinfo.variable, vinfo.deps
    
    def get_spatial_derivative_deps(self, x, f, ii):
        # BasisFunction won't normally depend on the mapping,
        # but the spatial derivatives will always do...
        # FIXME: Don't just reuse deps from f[1], the form compiler needs
        # to consider whether df/dx depends on coordinates or not!
        # I.e. for gradients of a linear basis function.
        
        deps = self.make_empty_deps()
        deps.runtime = True
        #deps.coordinates = True # TODO: Enable for higher order mappings.
        
        # Combine dependencies
        d = f[1] | deps
        
        # Reuse expression if possible
        if f[0] is x.operands()[0]:
            return x, d
        
        # Construct new expression
        return type(x)(f[0], ii[0]), d
    
    def child_deps(self, x, *ops):
        ufl_assert(ops, "Non-terminal with no ops should never occur.")
        # Combine dependencies
        d = ops[0][1]
        for o in ops[1:]:
            d |= o[1]
        
        # Make variables of all ops with differing dependencies
        if any(o[1] != d for o in ops):
            oldops = ops
            ops = []
            _skiptypes = (MultiIndex, Zero, FloatValue, IntValue)
            for o in oldops:
                if isinstance(o[0], _skiptypes):
                    ops.append(o)
                else:
                    vinfo = self.register_expression(o[0], o[1])
                    ops.append((vinfo.variable, vinfo.deps))
        
        # Reuse expression if possible
        ops = [o[0] for o in ops]
        if all((a is b) for (a, b) in zip(ops, x.operands())):
            return x, d
        # Construct new expression
        return type(x)(*ops), d

    def transform(self, x):
        c = x._uflid
        h = self.handlers[c]
        if isinstance(x, Terminal):
            return h(x)
        ops = [self.transform(o) for o in x.operands()]
        return h(x, *ops)
        
    def register_expression(self, e, deps, count=None):
        """Register expression as a variable with dependency
        data, reusing variable count if necessary.
        If the expression is already a variable, reuse it."""
        if count is None:
            if isinstance(e, Variable):
                v = e
            else:
                v = Variable(e)
        else:
            v = Variable(e, count=count)
        count = v._count
        vinfo = self.codestructure.variableinfo.get(count, None)
        if vinfo is None:
            vinfo = VariableInfo(v, deps)
            self.codestructure.variableinfo[count] = vinfo
            self.codestructure.stacks[deps].append(vinfo)
        else:
            ufl_debug("When does this happen? Need an algorithm revision to trust this fully.") # FIXME
        return vinfo

    def handle(self, v):
        ufl_assert(isinstance(v, Variable), "Expecting Variable.")
        vinfo = self.codestructure.variableinfo.get(v._count, None)
        if vinfo is None:
            # split v._expression 
            e, deps = self.transform(v._expression)
            # Register expression e as the expression of variable v
            vinfo = self.register_expression(e, deps, count=v._count)
        return vinfo


def _test_dependency_set():
    basisfunctions = (True, False)
    d1 = DependencySet(basisfunctions, runtime=True, coordinates=False)
    basisfunctions = (False, True)
    d2 = DependencySet(basisfunctions, runtime=True, coordinates=False)
    d3 = d1 | d2
    d4 = d1 & d2
    print d1
    print d2
    print d3
    print d4


def _test_split_by_dependencies():
    pass
#===============================================================================
#    def unit_tuple(i, n, true=True, false=False):
#        return tuple(true if i == j else false for j in xrange(n))
#    
#    a = ...
#    
#    from ufl.algorithms.formdata import FormData
#    formdata = FormData(a)
#    
#    basisfunction_deps = []
#    for i in range(formdata.rank):
#        bfs = unit_tuple(i, formdata.rank, True, False)
#        d = DependencySet(bfs, coordinates=True) # Disable coordinates depending on element
#        basisfunction_deps.append(d)
#    
#    function_deps = []
#    bfs = (False,)*formdata.rank
#    for i in range(num_coefficients):
#        d = DependencySet(bfs, runtime=True, coordinates=True) # Disable coordinates depending on element
#        function_deps.append(d)
#    
#    e, d, c = split_by_dependencies(integrand, formdata, basisfunction_deps, function_deps)
#    print e
#    print d
#    print c
#===============================================================================

if __name__ == "__main__":
    _test_dependency_set()
