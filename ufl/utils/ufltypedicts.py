# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Various utility data structures."""


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
