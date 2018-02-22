# -*- coding: utf-8 -*-

from setuptools import setup
from os.path import join, split
import sys
import platform

module_name = "ufl"

if sys.version_info < (3, 5):
    print("Python 3.5 or higher required, please upgrade.")
    sys.exit(1)

version = "2018.1.0.dev0"

url = "https://bitbucket.org/fenics-project/%s/" % module_name
tarball = None
if 'dev' not in version:
    tarball = url + "downloads/fenics-%s-%s.tar.gz" % (module_name, version)

script_names = ("ufl-analyse", "ufl-convert", "ufl-version", "ufl2py")

scripts = [join("scripts", script) for script in script_names]
man_files = [join("doc", "man", "man1", "%s.1.gz" % (script,)) for script in script_names]
data_files = [(join("share", "man", "man1"), man_files)]

if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
    # In the Windows command prompt we can't execute Python scripts
    # without a .py extension. A solution is to create batch files
    # that runs the different scripts.
    batch_files = []
    for script in scripts:
        batch_file = script + ".bat"
        with open(batch_file, "w") as f:
            f.write(sys.executable + ' "%%~dp0\%s" %%*' % split(script)[1])
        batch_files.append(batch_file)
    scripts.extend(batch_files)

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Operating System :: POSIX
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Topic :: Scientific/Engineering :: Mathematics
Topic :: Software Development :: Libraries :: Python Modules
"""

setup(name="fenics-ufl",
      version=version,
      description="Unified Form Language",
      author="Martin Sandve Alnæs, Anders Logg",
      author_email="fenics-dev@googlegroups.com",
      url=url,
      download_url=tarball,
      classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
      scripts=scripts,
      packages=[
          "ufl",
          "ufl.utils",
          "ufl.finiteelement",
          "ufl.core",
          "ufl.corealg",
          "ufl.algorithms",
          "ufl.formatting",
      ],
      package_dir={"ufl": "ufl"},
      install_requires=["numpy"],
      data_files=data_files
      )
