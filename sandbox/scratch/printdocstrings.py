#!/usr/bin/env python
import ufl

missing = {}
terminals = {}
operators = {}
formoperators = {}
other = {}
for n in ufl.__all__:
    o = getattr(ufl, n)
    d = o.__doc__
    if not d:
        missing[n] = d
    elif "UFL operator:" in d:
        operators[n] = d
    elif "UFL form operator:" in d:
        formoperators[n] = d
    elif d.startswith("UFL "):
        terminals[n] = d
    else:
        other[n] = d

def format_dict(di):
    return '\n'.join('%s: %s' % (n, di[n]) for n in sorted(di.keys()))

sep = '\n' + '='*80 + '\n'
print sep+"Terminals:"
print format_dict(terminals)
print sep+"Operators:"
print format_dict(operators)
print sep+"Form operators:"
print format_dict(formoperators)
print sep+"Other:"
print format_dict(other)
print sep+"Other names:"
print '\n'.join(sorted(other.keys()))
print sep+"Missing:"
print '\n'.join(sorted(missing.keys()))
