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
        lines += ["", "-"*80, f]
    if todos:
        n = len(todos)
        lines += ["%d TODOs:" % n] + todos
        num_todos += n
    if fixmes:
        n = len(fixmes)
        lines += ["%d FIXMEs:" % n] + fixmes
        num_fixmes += n

print("\n".join(lines))

print()
print("-"*80)
print("Number of TODOs: ", num_todos)
print("Number of FIXMEs:", num_fixmes)
print()
