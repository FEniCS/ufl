import setuptools

# Can be removed when pip editable user installs are fixed
# https://github.com/pypa/pip/issues/7953
import site
import sys
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

<<<<<<< HEAD
if sys.version_info < (3, 6):
    print("Python 3.6 or higher required, please upgrade.")
    sys.exit(1)

version = "2022.1.0"

url = "https://github.com/FEniCS/ufl"

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
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Topic :: Scientific/Engineering :: Mathematics
Topic :: Software Development :: Libraries :: Python Modules
"""

setup(
    name="fenics-ufl",
    version=version,
    description="Unified Form Language",
    author="FEniCS Project Team",
    author_email="fenics-dev@googlegroups.com",
    url=url,
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
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
    install_requires=["numpy"])
=======
setuptools.setup()
>>>>>>> main
