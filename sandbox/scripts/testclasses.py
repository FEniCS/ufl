#!/usr/bin/env python
"""Print all TestCase subclasses found in *.py."""

import os
from glob import glob

lines = []
for f in sorted(glob("*.py")):
    classes = []
    for l in open(f):
        if "class" in l and "TestCase" in l:
            c = l
            c = c.split(" ")[1]
            c = c.split("(")[0]
            classes.append(c)
    lines.append("")
    lines.append(f)
    lines.append("tests = [%s]" % ", ".join(classes))

print("\n".join(lines))
