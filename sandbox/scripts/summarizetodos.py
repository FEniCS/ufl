#!/usr/bin/env python
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
    lines.append("")
    lines.append(f)
    if todos:
        lines.append("TODOs:")
        lines.extend(todos)
    if fixmes:
        lines.append("FIXMEs:")
        lines.extend(fixmes)

print "\n".join(lines)
