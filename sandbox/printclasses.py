
from ufl.classes import *

print
print "--- Terminal classes:"
for name in sorted(c.__name__ for c in terminal_classes):
    print name

print
print "--- Nonterminal classes:"
for name in sorted(c.__name__ for c in nonterminal_classes):
    print name

