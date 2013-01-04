"This module defines basic utilities for stitching together LaTeX documents."

# Copyright (C) 2008-2013 Martin Sandve Alnes
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2008-05-07
# Last changed: 2011-06-02

# --- Basic LaTeX tools ---

documenttemplate = """\
\\documentclass{article}
\\usepackage{amsmath}

\\title{%s}

\\begin{document}
\\maketitle{}

%s
\\end{document}
"""

sectiontemplate = """\
\\section{%s}
%s
"""

subsectiontemplate = """\
\\subsection{%s}
%s
"""

subsubsectiontemplate = """\
\\subsubsection{%s}
%s
"""

itemizetemplate = """\
\\begin{itemize}
%s
\\end{itemize}
"""

aligntemplate = """\
\\begin{align}
%s
\\end{align}
"""

verbatimtemplate = """\
\\begin{verbatim}
%s
\\end{verbatim}
"""

def verbatim(string):
    return verbatimtemplate % string

def align(lines):
    # Calculate column lengths
    if isinstance(lines[0], str):
        body = " \\\\\n".join(l for l in lines)
    else:
        n = len(lines[0])
        collengths = [0]*n
        for l in lines:
            for i,s in enumerate(l):
                collengths[i] = max(collengths[i], len(s))
        def coljoin(cols):
            return " & ".join(c.ljust(collengths[i]) for (i,c) in enumerate(cols))
        body = " \\\\\n".join(coljoin(l) for l in lines)
    return aligntemplate % body

def itemize(items):
    body = "\n".join(items)
    return itemizetemplate % body

def subsubsection(s):
    if isinstance(s, str):
        return s
    if isinstance(s, tuple):
        title, body = s
        if isinstance(body, list):
            body = itemize(map(str,body))
        return subsubsectiontemplate % (title, body)

def subsection(s):
    if isinstance(s, str):
        return s
    if isinstance(s, tuple):
        title, body = s
        if isinstance(body, list):
            body = "\n".join(subsubsection(ss) for ss in body)
        return subsectiontemplate % (title, body)

def section(s):
    if isinstance(s, str):
        return s
    if isinstance(s, tuple):
        title, body = s
        if isinstance(body, list):
            body = "\n".join(subsection(ss) for ss in body)
        return sectiontemplate % (title, body)

def document(title, sections):
    body = "\n".join(section(s) for s in sections)
    return documenttemplate % (title, body)

def testdocument():
    title = "Test title 1"
    sections = ["sec1", "sec2"]
    print document(title, sections)

    title = "Test title 2"
    sections = [("sec1", "secbody1"), ("sec2", "secbody2")]
    print document(title, sections)

    title = "Test title 3"
    section1 = [("subsec1", "subsecbody1"), ("subsec2", "subsecbody2")]
    section2 = [("subsec1", "subsecbody1"), ("subsec2", "subsecbody2"), ("subsec3", "subsecbody3"), ]
    section3 = "\\section{manual sec}\ntestelest"
    sections = [("sec1", section1), ("sec2", section2), ("sec3", section3)]
    print document(title, sections)

    matrix = [ ("a(...) ", "= \\int_\\Omega foo dx0"),
               ("",        "+ \\int_\\Omega foo dx1"),
               ("",        "+ \\int_\\Omega foo dx1"),
            ]
    print align(matrix)

