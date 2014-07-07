#!/usr/bin/env python
"""Read *.py and print import statements for all
classes and functions found in these files.
Intended for __init__.py generation."""

import re
from glob import glob

files = glob("*.py")

#prefix = "." # absolute_import syntax
prefix = ""

for f in files:
    lines = open(f).readlines()
    defs = []
    for l in lines:
        for s in ("def ", "class "):
            if l.startswith(s):
                m = re.search(r"%s([^:(]*)" % s, l)
                d = m.groups()[0]
                defs.append(d)
    print()
    print(("from %s%s import %s" % (prefix, f[:-3], ", ".join(defs))))
        
