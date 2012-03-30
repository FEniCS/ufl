
import hashlib
from ufl.classes import Index, MultiIndex, Coefficient, Argument, Terminal, Label
from ufl.common import Counted
from ufl.log import error
from ufl.algorithms.traversal import traverse_terminals, fast_pre_traversal

def compute_multiindex_hashdata(expr, index_numbering):
    data = []
    for i in expr:
        if isinstance(i, Index):
            j = index_numbering.get(i)
            if j is None:
                # Use negative ints for Index
                j = -(len(index_numbering)+1)
                index_numbering[i] = j
            data.append(j)
        else:
            # Use nonnegative ints for FixedIndex
            data.append(int(i))
    return data

def compute_terminal_hashdata(integrand):
    # Extract a unique numbering of free indices,
    # as well as form arguments, and just take
    # repr of the rest of the terminals while
    # we're iterating over them
    terminal_hashdata = {}
    labels = {}
    index_numbering = {}
    coefficients = set()
    arguments = set()
    for expr in traverse_terminals(integrand):
        if isinstance(expr, MultiIndex):
            terminal_hashdata[expr] = compute_multiindex_hashdata(expr, index_numbering)
        elif isinstance(expr, Coefficient):
            coefficients.add(expr)
        elif isinstance(expr, Argument):
            arguments.add(expr)
        elif isinstance(expr, Label):
            data = labels.get(expr)
            if data is None:
                data = "L%d" % len(labels)
                labels[expr] = data
            terminal_hashdata[expr] = data
        elif isinstance(expr, Counted):
            error("Not implemented hashing for Counted subtype %s" % type(expr))
        else:
            terminal_hashdata[expr] = repr(expr)
    # Apply renumbering of form arguments
    coefficients = sorted(coefficients, key=lambda x: x.count())
    arguments = sorted(arguments, key=lambda x: x.count())
    for i, e in enumerate(coefficients):
        terminal_hashdata[e] = repr(e.reconstruct(count=i))
    for i, e in enumerate(arguments):
        terminal_hashdata[e] = repr(e.reconstruct(count=i))
    return terminal_hashdata

def compute_form_signature(form):
    hashdata = []
    for integral in form.integrals():
        integrand = integral.integrand()

        # Build hashdata for all terminals first
        terminal_hashdata = compute_terminal_hashdata(integrand)

        # Build hashdata for expression
        expression_hashdata = []
        for expr in fast_pre_traversal(integrand):
            if isinstance(expr, Terminal):
                data = terminal_hashdata[expr]
            else:
                data = expr._classid
            expression_hashdata.append(data)
        integral_hashdata = (repr(integral.measure()), expression_hashdata)
        hashdata.append(integral_hashdata)

    return hashlib.sha512(str(hashdata)).hexdigest()
