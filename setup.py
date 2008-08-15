#!/usr/bin/env python

from distutils.core import setup
from distutils import sysconfig
import sys

# Version number
major = 0
minor = 1

# Set prefix
try:    prefix = [item for item in sys.argv[1:] if "--prefix=" in item][0].split("=")[1]
except: prefix = ("/").join(sysconfig.get_python_inc().split("/")[:-2])
print "Installing UFL under %s..." % prefix

# Generate pkgconfig file
file = open("ufl-%d.pc" % major, "w")
file.write("Name: UFL\n")
file.write("Version: %d.%d\n" % (major, minor))
file.write("Description: Unified Form Language\n")
file.write("Cflags: -I%s/include\n" % prefix)
file.close()

setup(name = "UFL",
      version = "%d.%d" % (major, minor),
      description = "Unified Form Language",
      author = "Martin Sandve Alnaes, Anders Logg",
      author_email = "ufl-dev@fenics.org",
      url = "http://www.fenics.org/ufl/",
      scripts = ["scripts/ufl-analyse"],
      packages = ["ufl", "ufl.algorithms"],
      package_dir = {"ufl": "ufl/"},
      data_files = [("lib/pkgconfig", ["ufl-%d.pc" % major])])

