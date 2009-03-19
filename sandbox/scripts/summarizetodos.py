#!/usr/bin/env python
"""Print summary of TODO and FIXME lines in files *.py."""

import os
from glob import glob

files = sorted(glob("*.py"))
lines = []
num_todos = 0
num_fixmes = 0
for f in files:
    todos = []
    fixmes = []
    for l in open(f):
        if "TODO" in l:
            todos.append(l.strip())
        if "FIXME" in l:
            fixmes.append(l.strip())
    if todos or fixmes:
        lines += ["", f]
    if todos:
        lines += ["TODOs:"] + todos
        num_todos += len(todos)
    if fixmes:
        lines += ["FIXMEs:"] + fixmes
        num_fixmes += len(fixmes)

print "\n".join(lines)

print
print "Number of TODOs: ", num_todos
print "Number of FIXMEs:", num_fixmes
print
