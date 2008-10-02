"""This module defines utilities for working with dependencies of subexpressions."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-10-01 -- 2008-10-02"

from collections import defaultdict
from itertools import izip, chain

from ..common import some_key, split_dict, or_tuples, and_tuples
from ..output import ufl_assert, ufl_error, ufl_warning
from ..permutation import compute_indices

# All classes:
from ..base import UFLObject, Terminal, FloatValue, ZeroType
from ..variable import Variable
from ..basisfunction import BasisFunction
from ..function import Function, Constant
from ..geometry import FacetNormal
from ..tensoralgebra import Identity
from ..indexing import MultiIndex, Indexed, Index, FixedIndex
from ..tensors import ListVector, ListMatrix, Tensor

# Lists of all UFLObject classes
from ..classes import ufl_classes, terminal_classes, nonterminal_classes, compound_classes

# Other algorithms:
from .analysis import basisfunctions, coefficients, indices, duplications
from .transformations import transform


class DependencySet:
    def __init__(self, basisfunctions, functions, \
                 cell=False, mapping=False, facet=False, coordinates=False):
        # depends on reference cell mapping
        self.mapping = mapping
        # inside quadrature loop or integral
        self.coordinates = coordinates
        # depends on global cell
        self.cell = cell
        # depends on facet
        self.facet = facet
        # depends on basis function i
        self.basisfunctions = tuple(basisfunctions)
        # depends on function i
        self.functions = tuple(functions)
    
    def size(self):
        return len(list(self.iter()))
    
    def iter(self):
        return chain((self.mapping, self.coordinates, self.cell, self.facet),
                  self.basisfunctions, self.functions)
    
    def __hash__(self):
        return hash(tuple(self.iter()))
    
    def __cmp__(self, other):
        for (a,b) in izip(self.iter(), other.iter()):
            if a < b: return -1
            if a > 1: return +1
        return 0
    
    def __or__(self, other):
        basisfunctions = or_tuples(self.basisfunctions, other.basisfunctions)
        functions = or_tuples(self.functions, other.functions)
        d = DependencySet(basisfunctions, functions)
        d.mapping = self.mapping or other.mapping
        d.coordinates = self.coordinates or other.coordinates
        d.cell = self.cell or other.cell
        d.facet = self.facet or other.facet
        return d

    def __and__(self, other):
        basisfunctions = and_tuples(self.basisfunctions, other.basisfunctions)
        functions = and_tuples(self.functions, other.functions)
        d = DependencySet(basisfunctions, functions)
        d.mapping = self.mapping and other.mapping
        d.coordinates = self.coordinates and other.coordinates
        d.cell = self.cell and other.cell
        d.facet = self.facet and other.facet
        return d

    def __str__(self):
        s = "DependencySet:\n"
        s += "self.mapping        = %s\n" % self.mapping
        s += "self.coordinates    = %s\n" % self.coordinates 
        s += "self.cell           = %s\n" % self.cell
        s += "self.facet          = %s\n" % self.facet 
        s += "self.basisfunctions = %s\n" % str(self.basisfunctions) 
        s += "self.functions      = %s" % str(self.functions) 
        return s


class VariableInfo:
    def __init__(self, variable, deps):
        # Variable
        self.variable = variable
        # DependencySet
        self.deps = deps
        # DiffVarSet -> VariableInfo
        self.diffcache = {}
    
    def __str__(self):
        s = "VariableInfo:"
        s += "  self.variable = %s\n" % self.variable
        s += "  self.deps = %s\n" % self.deps
        s += "  self.diffcache = \n"
        s += "\n".join("    %s: %s" % (k,v) for (k,v) in self.diffcache.iteritems())
        return s

class DiffVarSet:
    def __init__(self):
        self.fixed_spatial_directions = set() # Set of integer indices
        self.open_spatial_directions = set() # Set of Index objects that the stored expression uses for d/dx_i
        self.variables = set() # Set of variables we're differentiating w.r.t.

    def __hash__(self):
        return hash(tuple(d for d in self.fixed_spatial_directions) + \
                    tuple(d for d in self.open_spatial_directions) + \
                    tuple(d for d in self.variables))
    
    #def __contains__(self, other):
    #def __sub__(self, other):
    #def __add__(self, other):
    #def __cmp__(self, other):
    def __eq__(self, other):
        return self.fixed_spatial_directions == other.fixed_spatial_directions and \
               len(self.open_spatial_directions) == len(other.open_spatial_directions) and \
               self.variables == other.variables


class CodeStructure:
    def __init__(self):
        # A place to look up if a variable has been added to the stacks already
        self.variableinfo = {}   # variable count -> VariableInfo
        # One stack of variables for each dependency configuration
        self.stacks = defaultdict(list) # DependencySet -> [VariableInfo]
    
    def __str__(self):
        deps = DependencySet(FIXME)
        s = ""
        s += "Variables independent of spatial coordinates:\n"
        keys = [k for k in self.stacks.keys() if not k.coordinates]
        for deps in keys:
            if k.facet:
                stack = self.stacks[k]
                s += str(stack)
        for deps in keys:
            if not k.facet:
                stack = self.stacks[k]
                s += str(stack)
        
        s += "Variables dependent of spatial coordinates:\n"
        for k in dependent:
            s += k
        
        return s

    def split_stacks(self):
        
        # Start with all variable stacks
        stacks = self.stacks
        
        # Split into precomputable and not
        def criteria(k):
            return not (k.spatial or any(k.coefficients) or k.facet)
        precompute_stacks, runtime_stacks = split_dict(stacks, criteria)
        
        # Split into outside or inside quadrature loop
        def criteria(k):
            return k.spatial
        quad_precompute_stacks, precompute_stacks = split_dict(precompute_stacks, criteria)
        quad_runtime_stacks, runtime_stacks = split_dict(runtime_stacks, criteria)
        
        # Example! FIXME: Make below code a function and apply to each stack group separately. 
        stacks = quad_runtime_stacks
        
        # Split by basis function dependencies
        
        # FIXME: Does this give us the order we want?
        # Want to iterate faster over test function, i.e. (0,0), (1,0), (0,1), (1,1) 
        keys = set(stacks.iterkeys())
        perms = [p for p in compute_permutations(rank, 2) if p in keys] # FIXME: NOT RIGHT!
        for perm in perms:
            def criteria(k):
                return k.basisfunctions == perm
            dep_stacks, stacks = split_dict(stacks, criteria)
            
            # TODO: For all permutations of basis function indices
            # TODO: Input elementreps
            sizes = [elementreps[i].local_dimension for i in range(self.rank) if perms[i]]
            basis_function_perms = compute_indices(sizes)
            for basis_function_perm in basis_function_perms:
                context.update_basisfunction_permutation(basis_function_perm) # FIXME: Map tuple to whole range of basisfunctions.
                for stack in dep_stacks.itervalues():
                    for (k,v) in stack.iteritems():
                        s = context.variable_to_symbol(k)
                        e = ufl_to_swiginac(v, context)
                        context.add_token(s, e)


def _split_by_dependencies(expression, codestructure, terminal_deps):
    
    if isinstance(expression, Variable):
        c = expression._count
        info = codestructure.variableinfo.get(c, None)
        if info is not None:
            return info.variable, info.deps
        #codestructure.stacks[vdeps].append(v)
    
    elif isinstance(expression, Terminal):
        h = terminal_deps[expression.__class__]
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
        ufl_assert(c not in codestructure.variableinfo, "Shouldn't reach this point if the variable was already cached!")
        vinfo = VariableInfo(expression, deps)
        codestructure.variableinfo[c] = vinfo
        codestructure.stacks[deps].append(vinfo)
    
    # Try to reuse expression if nothing has changed:
    if any((o1 is not o3) for (o1,o3) in izip(ops,ops3)):
        e = expression.__class__(*ops3)
    else:
        e = expression
    return e, deps


def split_by_dependencies(expression, formdata, basisfunction_deps, function_deps):
    """Split an expression into stacks of variables based
    on the dependencies of its subexpressions.
    
    @type expression: UFLObject
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
    ufl_assert(isinstance(expression, UFLObject), "Expecting UFLObject.")
    
    # Utility function to ensure consistent ordering of dependency tuples
    bfs = (False,)*len(basisfunction_deps)
    fs = (False,)*len(function_deps)
    def make_deps(basisfunction=None, function=None,
                  cell=False, mapping=False, facet=False, coordinates=False):
        d = DependencySet(bfs, fs, cell=cell, mapping=mapping,
                          facet=facet, coordinates=coordinates)
        if basisfunction is not None:
            d |= basisfunction_deps[basisfunction]
        if function is not None:
            d |= function_deps[function]
        return d
    
    codestructure = CodeStructure()
    
    # Define dependencies for all Terminal objects:
    terminal_dep_handlers = {}
    _no_dep = make_deps()
    def no_deps(x):
        return _no_dep
    _facet_normal_dep = make_deps(facet=True)
    def get_facet_normal_deps(x):
        return _facet_normal_dep
    def get_function_deps(x):
        return function_deps[formdata.coefficient_renumbering[x]]
    def get_basisfunction_deps(x):
        return basisfunction_deps[formdata.basisfunction_renumbering[x]]
    terminal_dep_handlers[MultiIndex] = no_deps
    terminal_dep_handlers[Identity]   = no_deps
    terminal_dep_handlers[FloatValue] = no_deps
    terminal_dep_handlers[ZeroType]   = no_deps
    terminal_dep_handlers[FacetNormal]   = get_facet_normal_deps
    terminal_dep_handlers[BasisFunction] = get_basisfunction_deps
    terminal_dep_handlers[Constant]      = get_function_deps
    terminal_dep_handlers[Function]      = get_function_deps
    
    # FIXME: Introduce dependency of mapping with gradients...
    #  BasisFunction shouldn't normally depend on the mapping, but the gradients do...
    
    # Perform! This fills in codestructure and returns
    # the final form of the expression and its dependencies
    e, deps = _split_by_dependencies(expression, codestructure, terminal_dep_handlers)
    
    # Add final e to stacks and return variable
    if not isinstance(e, Variable):
        e = Variable(e) # TODO: Check a variable cache to avoid duplications?
    vinfo = codestructure.variableinfo.get(e._count, None)
    if vinfo is None:
        vinfo = VariableInfo(e, deps)
        codestructure.variableinfo[e._count] = vinfo
        codestructure.stacks[deps].append(vinfo)
    return (vinfo, codestructure)


def _test_dependency_set():
    basisfunctions, functions = (True, False), (True, False, False, True)
    d1 = DependencySet(basisfunctions, functions, \
                 cell=False, mapping=True, facet=True, coordinates=False)
    basisfunctions, functions = (False, True), (False, True, False, True)
    d2 = DependencySet(basisfunctions, functions, \
                 cell=True, mapping=False, facet=True, coordinates=False)
    d3 = d1 | d2
    d4 = d1 & d2
    print d1
    print d2
    print d3
    print d4

def unit_tuple(i, n, true=True, false=False):
    return tuple(true if i == j else false for j in xrange(n))

def _test_split_by_dependencies():
    a = FIXME
    
    formdata = FormData(a)
    
    basisfunction_deps = []
    for i in range(formdata.rank):
        # TODO: More depending on element
        d = DependencySet(unit_tuple(i, formdata.rank, True, False), (False,)*formdata.num_coefficients)
        basisfunction_deps.append(d)
    
    function_deps = []
    for i in range(num_coefficients):
        # TODO: More depending on element
        d = DependencySet((False,)*formdata.rank, unit_tuple(i, formdata.num_coefficients, True, False))
        basisfunction_deps.append(d)
    
    e, d, c = split_by_dependencies(integrand, formdata, basisfunction_deps, function_deps)
    print e
    print d
    print c

if __name__ == "__main__":
    _test_dependency_set()
