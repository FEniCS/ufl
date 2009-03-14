#!/usr/bin/env python
"""Print summary of TODO and FIXME lines in files *.py."""

import os
from glob import glob

files = sorted(glob("*.py"))
lines = []
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
    if fixmes:
        lines += ["FIXMEs:"] + fixmes

print "\n".join(lines)
