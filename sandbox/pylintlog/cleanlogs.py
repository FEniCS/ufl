#!/usr/bin/env python
import os, sys
files = sys.argv[1:]

def delete(f):
    print(("rm", f))
    os.remove(f)

for f in files:
    lines = [l.strip() for l in open(f).readlines()]
    if not lines:
        delete(f)
    else:
        for l in lines[1:]: # Skip first line
            if l:
                break
        else:
            delete(f)

