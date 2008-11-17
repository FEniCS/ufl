
import sys, os, optparse
from pprint import pprint

usage = """
Usage description.
"""

def opt(long, short, t, default, help):
    return optparse.make_option("--%s" % long, "-%s" % short, action="store", type=t, dest=long, default=default, help=help)

option_list = [ \
    # Directories:
    opt("outputdir", "o", "str", "", "Output directory."),
    opt("inputdir",  "i", "str", "", "Input directory."),
    # What to do with rendered output:
    opt("file",      "f", "int", 0,   "Write rendering strings to files."),
    opt("screen",    "n", "int", 0,   "Print rendering strings to screen."),
    opt("view",      "v", "int", 0,   "Open rendered documents in external viewer applications."),
    # Expression transformations:
    opt("compile", "c", "int", 0, "Apply expression transformations like in a quadrature based form compilation."),
    # Toggles for all possible output formats:
    opt("str",     "s", "int", 0, "Render forms as str string."),
    opt("repr",    "r", "int", 0, "Render forms as repr string."),
    opt("tree",    "t", "int", 0, "Render forms as tree format string."),
    opt("dot",     "d", "int", 0, "Render forms as dot graph visualization format."),
    opt("latex",   "l", "int", 0, "Render forms as latex document."),
    opt("all",     "a", "int", 0, "Render forms as all possible outputs."),
    # Automatic rendered file conversions:
    # TODO: Writing to pdf, ps, eps, png, etc
#   opt("name", "n", "str", "default value", "help string"),
    ]

parser = optparse.OptionParser(usage=usage, option_list=option_list)
args = sys.argv[1:]
(options, args) = parser.parse_args(args=args)

print
print "options"
pprint(options)

print
print "args"
pprint(args)

for arg in args:
    filename = os.path.join(options.inputdir, arg)
    path, name = os.path.split(filename)
    basename, ext = os.path.splitext(name)
    pdfname = os.path.join(options.outputdir, basename + ".pdf")
    print 
    print "filename =", filename
    print "path =", path
    print "name =", name
    print "basename =", basename
    print "ext =", ext
    print "pdfname =", pdfname

