"Deprecated file."

# Copyright (C) 2008-2013 Anders Logg
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
# First added:  2009-04-05
# Last changed: 2011-06-02

from ufl.log import error
from ufl.form import Form

# TODO: Move this to form.py or some other file, or just get rid of calls to it
def as_form(form):
    "Convert to form if not a form, otherwise return form."

    # Check form Form
    if isinstance(form, Form):
        return form

    error("Unable to convert object to a UFL form: %s" % repr(form))
