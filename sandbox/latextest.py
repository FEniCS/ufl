
import os

from ufl.algorithms.latextools import testdocument
testdocument()

from ufl.algorithms.ufl2latex import ufl2pdf

basename = "hyperelasticity1D"
basename = "stiffness_ad"
basename = "mass"
uflfilename = basename + ".ufl"
texfilename = basename + ".tex"
pdffilename = basename + ".pdf"

ufl2pdf(uflfilename, texfilename, pdffilename)

