"""This module defines the Terminal class, the superclass
for all types that are terminal nodes in the expression trees."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-02-03"

# Modified by Anders Logg, 2008

from ufl.expr import Expr
from ufl.common import lstr
from ufl.log import error, warning

#--- Base class for terminal objects ---

class Terminal(Expr):
    "A terminal node in the UFL expression tree."
    __slots__ = ()
    
    def __init__(self):
        Expr.__init__(self)
    
    def operands(self):
        "A Terminal object never has operands."
        return ()
    
    def free_indices(self):
        "A Terminal object never has free indices."
        return ()
    
    def index_dimensions(self):
        "A Terminal object never has free indices."
        return {}
    
    def evaluate(self, x, mapping, component, index_values):
        "Get self from mapping and return the component asked for."
        f = mapping.get(self, self)
        if callable(f):
            f = f(x)
        return f[component] if component else f
    
    def __eq__(self, other):
        """Checks whether the two expressions are represented the
        exact same way using repr. This does not check if the forms
        are mathematically equal or equivalent!"""
        if type(self) != type(other):
            return False
        if id(self) == other:
            return True
        return repr(self) == repr(other)

    def __iter__(self):
        return iter(())
    
    #def __getnewargs__(self): # TODO: Test pickle and copy with this. Must implement differently for Terminal objects though.
    #    "Used for pickle and copy operations."
    #    raise NotImplementedError, "Must reimplement in each Terminal, or?"

#--- Subgroups of terminals ---

class FormArgument(Terminal):
    def __init__(self):
        Terminal.__init__(self)

class UtilityType(Terminal):
    def __init__(self):
        Terminal.__init__(self)
    
    def shape(self):
        error("Calling shape on a utility type is an error.")
    
    def free_indices(self):
        error("Calling free_indices on a utility type is an error.")
    
    def index_dimensions(self):
        error("Calling free_indices on a utility type is an error.")

#--- Non-tensor terminal nodes ---

class Tuple(UtilityType):
    "For internal use, never to be created by users."
    def __init__(self, *items):
        UtilityType.__init__(self)
        if not all(isinstance(i, Expr) for i in items):
            warning("Got non-Expr in Tuple, is this intended? If so, remove this warning.")
        self._items = items
    
    def __getitem__(self, i):
        return self._items[i]
    
    def __len__(self):
        return len(self._items)
    
    def __iter__(self):
        return iter(self._items)
    
    def __str__(self):
        return "Tuple(*(%s,))" % ", ".join(str(i) for i in self._items)
    
    def __repr__(self):
        return "Tuple(*%s)" % repr(self._items)

