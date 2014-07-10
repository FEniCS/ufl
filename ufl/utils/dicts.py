"Various dict manipulation utilities."

# Copyright (C) 2008-2014 Martin Sandve Alnes
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

from six import iterkeys, iteritems, next

def split_dict(d, criteria):
    "Split a dict d into two dicts based on a criteria on the keys."
    a = {}
    b = {}
    for (k, v) in iteritems(d):
        if criteria(k):
            a[k] = v
        else:
            b[k] = v
    return a, b

def slice_dict(dictionary, keys, default=None):
    return tuple(dictionary.get(k, default) for k in keys)

def some_key(a_dict):
    "Return an arbitrary key from a dictionary."
    return next(iterkeys(a_dict))

def mergedicts(dicts):
    d = dict(dicts[0])
    for d2 in dicts[1:]:
        d.update(d2)
    return d

def mergedicts2(d1, d2):
    d = dict(d1)
    d.update(d2)
    return d

def subdict(superdict, keys):
    return dict((k, superdict[k]) for k in keys)

def dict_sum(items):
    "Construct a dict, in between dict(items) and sum(items), by accumulating items for each key."
    d = {}
    for k, v in items:
        if k not in d:
            d[k] = v
        else:
            d[k] += v
    return d

class EmptyDictType(dict):
    def __setitem__(self, key, value):
        from ufl.log import error
        error("This is a frozen unique empty dictionary object, inserting values is an error.")
    def update(self, *args, **kwargs):
        from ufl.log import error
        error("This is a frozen unique empty dictionary object, inserting values is an error.")
EmptyDict = EmptyDictType()
