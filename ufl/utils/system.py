# -*- coding: utf-8 -*-
"Various utilities accessing system io."

# Copyright (C) 2008-2014 Martin Sandve Alnæs and Johannes Ring
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

# Taken from http://ivory.idyll.org/blog/mar-07/replacing-commands-with-subprocess
from subprocess import Popen, PIPE, STDOUT
from functools import reduce
def get_status_output(cmd, input=None, cwd=None, env=None):
    pipe = Popen(cmd, shell=True, cwd=cwd, env=env, stdout=PIPE, stderr=STDOUT)

    (output, errout) = pipe.communicate(input=input)
    assert not errout

    status = pipe.returncode

    return (status, output)

def write_file(filename, text):
    with open(filename, "w") as f:
        f.write(text)

def pdflatex(latexfilename, pdffilename, flags=""): # TODO: Options for this.
    "Execute pdflatex to compile a latex file into pdf."
    flags += "-file-line-error-style -interaction=nonstopmode"
    latexcmd = "pdflatex"
    cmd = "%s %s %s %s" % (latexcmd, flags, latexfilename, pdffilename)
    s, o = get_status_output(cmd)
    return s, o

def openpdf(pdffilename):
    "Open PDF file in external pdf viewer."
    reader_cmd = "evince %s &" # TODO: Add option for which reader to use. Is there a portable way to do this? Like "get default pdf reader from os"?
    cmd = reader_cmd % pdffilename
    s, o = get_status_output(cmd)
    return s, o
