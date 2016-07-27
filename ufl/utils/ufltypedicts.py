# -*- coding: utf-8 -*-
"Various utility data structures."

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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


class UFLTypeDict(dict):
    def __init__(self):
        dict.__init__(self)

    def __getitem__(self, key):
        return dict.__getitem__(self, key._ufl_class_)

    def __setitem__(self, key, value):
        return dict.__setitem__(self, key._ufl_class_, value)

    def __delitem__(self, key):
        return dict.__delitem__(self, key._ufl_class_)

    def __contains__(self, key):
        return dict.__contains__(self, key._ufl_class_)


class UFLTypeDefaultDict(dict):
    def __init__(self, default):
        dict.__init__(self)

        def make_default():
            return default
        self.setdefault(make_default)

    def __getitem__(self, key):
        return dict.__getitem__(self, key._ufl_class_)

    def __setitem__(self, key, value):
        return dict.__setitem__(self, key._ufl_class_, value)

    def __delitem__(self, key):
        return dict.__delitem__(self, key._ufl_class_)

    def __contains__(self, key):
        return dict.__contains__(self, key._ufl_class_)
