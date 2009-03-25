#!/usr/bin/env python

from distutils.core import setup
from distutils import sysconfig
from os.path import join as pjoin, split as psplit
import sys
import platform

# Version number
major = 0
minor = 1

# Set prefix
try:
    prefix = [item for item in sys.argv[1:] \
              if "--prefix=" in item][0].split("=")[1]
except:
    try:
        prefix = sys.argv[sys.argv.index('--prefix')+1]
    except:
        prefix = sys.prefix
print "Installing UFL under %s..." % prefix

# Generate pkgconfig file
file = open("ufl-%d.pc" % major, "w")
file.write("Name: UFL\n")
file.write("Version: %d.%d\n" % (major, minor))
file.write("Description: Unified Form Language\n")
file.write("Cflags: -I%s\n" % repr(pjoin(prefix,"include"))[1:-1])
# FIXME: better way for this? ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
file.close()

scripts = [pjoin("scripts", "ufl-analyse"),
           pjoin("scripts", "ufl-convert"),
           pjoin("scripts", "form2ufl")]

if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
    # In the Windows command prompt we can't execute Python scripts 
    # without a .py extension. A solution is to create batch files
    # that runs the different scripts.
    batch_files = []
    for script in scripts:
        batch_file = script + ".bat"
        f = open(batch_file, "w")
        f.write('python "%%~dp0\%s" %%*' % psplit(script)[1])
        f.close()
        batch_files.append(batch_file)
    scripts.extend(batch_files)

setup(name = "UFL",
      version = "%d.%d" % (major, minor),
      description = "Unified Form Language",
      author = "Martin Sandve Alnaes, Anders Logg",
      author_email = "ufl-dev@fenics.org",
      url = "http://www.fenics.org/ufl/",
      scripts = scripts,
      packages = ["ufl", "ufl.algorithms"],
      package_dir = {"ufl": "ufl"},
      data_files = [(pjoin("lib", "pkgconfig"), ["ufl-%d.pc" % major])])

