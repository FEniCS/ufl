#!/usr/bin/env python
"""Rename .ufl files from lower_score_naming to CamelCapsNaming."""

import os
from glob import glob
files = glob("*.ufl")

files1 = [f for f in files if f.find("_") > 0]
files2 = [f for f in files if f.find("_") < 0]

jobs = []

for f in files1:
    a, b = f.split("_")
    g = a[0].upper() + a[1:] + b[0].upper() + b[1:]
    cmd = "hg rename %s %s" % (f, g)
    jobs.append(cmd)

for f in files2:
    g = f[0].upper() + f[1:]
    cmd = "hg rename %s %s" % (f, g)
    jobs.append(cmd)

for cmd in jobs:
    print(cmd) 
    os.system(cmd)
