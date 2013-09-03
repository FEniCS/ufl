"This module contains a collection of common utilities."

# Copyright (C) 2008-2013 Martin Sandve Alnes and Anders Logg
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
# Modified by Kristian Oelgaard, 2009
#
# First added:  2008-08-05
# Last changed: 2011-06-02

from itertools import izip
import operator

# Taken from http://ivory.idyll.org/blog/mar-07/replacing-commands-with-subprocess
from subprocess import Popen, PIPE, STDOUT
def get_status_output(cmd, input=None, cwd=None, env=None):
    pipe = Popen(cmd, shell=True, cwd=cwd, env=env, stdout=PIPE, stderr=STDOUT)

    (output, errout) = pipe.communicate(input=input)
    assert not errout

    status = pipe.returncode

    return (status, output)

def write_file(filename, text):
    f = open(filename, "w")
    f.write(text)
    f.close()

def pdflatex(latexfilename, pdffilename, flags): # TODO: Options for this.
    "Execute pdflatex to compile a latex file into pdf."
    flags = "-file-line-error-style -interaction=nonstopmode"
    latexcmd = "pdflatex"
    cmd = "%s %s %s %s" % (latexcmd, flags, latexfilename, pdffilename)
    s, o = get_status_output(cmd)

def openpdf(pdffilename):
    "Open PDF file in external pdf viewer."
    reader_cmd = "evince %s &" # TODO: Add option for which reader to use. Is there a portable way to do this? Like "get default pdf reader from os"?
    cmd = reader_cmd % pdffilename
    s, o = get_status_output(cmd)

def product(sequence):
    "Return the product of all elements in a sequence."
    return reduce(operator.__mul__, sequence, 1)

class EmptyDictType(dict):
    def __setitem__(self, key, value):
        from ufl.log import error
        error("This is a frozen unique empty dictionary object, inserting values is an error.")
    def update(self, *args, **kwargs):
        from ufl.log import error
        error("This is a frozen unique empty dictionary object, inserting values is an error.")
EmptyDict = EmptyDictType()

def sorted_by_count(seq):
    return sorted(seq, key=lambda x: x._count)

def sorted_items(mapping):
    return sorted(mapping.iteritems(), key=lambda x: x[0])

def mergedicts(dicts):
    d = dict(dicts[0])
    for d2 in dicts[1:]:
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

def unzip(seq):
    "Inverse operation of zip: unzip(zip(a, b)) == (a, b)"
    return [s[0] for s in seq], [s[1] for s in seq]

def xor(a, b):
    return bool(a) if b else not a

def or_tuples(seqa, seqb):
    "Return 'or' of all pairs in two sequences of same length."
    return tuple(a or b for (a,b) in izip(seqa, seqb))

def and_tuples(seqa, seqb):
    "Return 'and' of all pairs in two sequences of same length."
    return tuple(a and b for (a,b) in izip(seqa, seqb))

def iter_tree(tree):
    """Iterate over all nodes in a tree represented
    by lists of lists of leaves."""
    if isinstance(tree, list):
        for node in tree:
            for i in iter_tree(node):
                yield i
    else:
        yield tree

def fast_pre_traversal(expr):
    """Yields o for each tree node o in expr, parent before child."""
    input = [expr]
    while input:
        l = input.pop()
        yield l
        input.extend(l.operands())

def unique_pre_traversal(expr, visited=None):
    """Yields o for each tree node o in expr, parent before child.

    This version only visits each node once!
    """
    input = [expr]
    visited = visited or set()
    while input:
        l = input.pop()
        if l not in visited:
            visited.add(l)
            yield l
            input.extend(l.operands())
fast_pre_traversal2 = unique_pre_traversal # TODO: Remove

def unique_post_traversal(expr, visited=None):
    """Yields o for each node o in expr, child before parent.

    Never visits a node twice."""
    stack = []
    stack.append((expr, list(expr.operands())))
    visited = visited or set()
    while stack:
        expr, ops = stack[-1]
        for i, o in enumerate(ops):
            if o is not None and o not in visited:
                stack.append((o, list(o.operands())))
                ops[i] = None
                break
        else:
            yield expr
            visited.add(expr)
            stack.pop()

def fast_post_traversal2(expr, visited=None):
    """Yields o for each tree node o in expr, child before parent."""
    stack = [expr]
    visited = visited or set()
    while stack:
        curr = stack[-1]
        for o in curr.operands():
            if o not in visited:
                stack.append(o)
                break
        else:
            yield curr
            visited.add(curr)
            stack.pop()

def fast_post_traversal(expr): # TODO: Would a non-recursive implementation save anything here?
    """Yields o for each tree node o in expr, child before parent."""
    # yield children
    for o in expr.operands():
        for i in fast_post_traversal(o):
            yield i
    # yield parent
    yield expr

def split_dict(d, criteria):
    "Split a dict d into two dicts based on a criteria on the keys."
    a = {}
    b = {}
    for (k,v) in d.iteritems():
        if criteria(k):
            a[k] = v
        else:
            b[k] = v
    return a, b

def slice_dict(dictionary, keys, default=None):
    return tuple(dictionary.get(k, default) for k in keys)

def some_key(a_dict):
    "Return an arbitrary key from a dictionary."
    return a_dict.iterkeys().next()

def camel2underscore(name):
    "Convert a CamelCaps string to underscore_syntax."
    letters = []
    lastlower = False
    for l in name:
        thislower = l.islower()
        if not thislower:
            # Don't insert _ between multiple upper case letters
            if lastlower:
                letters.append("_")
            l = l.lower()
        lastlower = thislower
        letters.append(l)
    return "".join(letters)

def lstr(l):
    "Pretty-print list or tuple, invoking str() on items instead of repr() like str() does."
    if isinstance(l, list):
        return "[" + ", ".join(lstr(item) for item in l) + "]"
    elif isinstance(l, tuple):
        return "(" + ", ".join(lstr(item) for item in l) + ")"
    return str(l)

def dstr(d, colsize=80):
    "Pretty-print dictionary of key-value pairs."
    sorted_keys = sorted(d.keys())
    return tstr([(key, d[key]) for key in sorted_keys], colsize)

def tstr(t, colsize=80):
    "Pretty-print list of tuples of key-value pairs."
    if not t:
        return ""

    # Compute maximum key length
    keylen = max([len(str(item[0])) for item in t])

    # Key-length cannot be larger than colsize
    if keylen > colsize:
        return str(t)

    # Pretty-print table
    s = ""
    for (key, value) in t:
        key = str(key)
        if isinstance(value, str):
            value = "'%s'" % value
        else:
            value = str(value)
        s += key + ":" + " "*(keylen - len(key) + 1)
        space = ""
        while len(value) > 0:
            end = min(len(value), colsize - keylen)
            s += space + value[:end] + "\n"
            value = value[end:]
            space = " "*(keylen + 2)
    return s

def sstr(s):
    "Pretty-print set."
    return ", ".join(str(x) for x in s)

def istr(o):
    "Format object as string, inserting ? for None."
    if o is None:
        return "?"
    else:
        return str(o)

def estr(elements):
    "Format list of elements for printing."
    return ", ".join(e.shortstr() for e in elements)

def recursive_chain(lists):
    for l in lists:
        if isinstance(l, str):
            yield l
        else:
            for s in recursive_chain(l):
                yield s

class ExampleCounted(object):
    """An example class for classes of objects identified by a global counter.

    The old inheritance pattern is deprecated. Mimic this class instead.
    """
    __slots__ = ("_count",)
    _globalcount = 0
    def __init__(self, count=None):
        counted_init(self, count, ExampleCounted)

    def count(self):
        return self._count

def counted_init(self, count=None, countedclass=None):
    if countedclass is None:
        countedclass = type(self)

    if count is None:
        count = countedclass._globalcount

    self._count = count

    if self._count >= countedclass._globalcount:
        countedclass._globalcount = self._count + 1


class Stack(list):
    "A stack datastructure."
    def __init__(self, *args):
        list.__init__(self, *args)

    def push(self, v):
        list.append(self, v)

    def peek(self):
        return self[-1]

class StackDict(dict):
    "A dict that can be changed incrementally with 'd.push(k,v)' and have changes rolled back with 'k,v = d.pop()'."
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self._l = []

    def push(self, k, v):
        # Store previous state for this key
        self._l.append((k, self.get(k, None)))
        if v is None:
            if k in self:
                del self[k]
        else:
            self[k] = v

    def pop(self):
        # Restore previous state for this key
        k, v = self._l.pop()
        if v is None:
            if k in self:
                del self[k]
        else:
            self[k] = v
        return k, v

class UFLTypeDict(dict):
    def __init__(self):
        dict.__init__(self)

    def __getitem__(self, key):
        return dict.__getitem__(self, key._uflclass)

    def __setitem__(self, key, value):
        return dict.__setitem__(self, key._uflclass, value)

    def __delitem__(self, key):
        return dict.__delitem__(self, key._uflclass)

    def __contains__(self, key):
        return dict.__contains__(self, key._uflclass)

class UFLTypeDefaultDict(dict):
    def __init__(self, default):
        dict.__init__(self)
        def make_default():
            return default
        self.setdefault(make_default)

    def __getitem__(self, key):
        return dict.__getitem__(self, key._uflclass)

    def __setitem__(self, key, value):
        return dict.__setitem__(self, key._uflclass, value)

    def __delitem__(self, key):
        return dict.__delitem__(self, key._uflclass)

    def __contains__(self, key):
        return dict.__contains__(self, key._uflclass)

def strides(shape):
    if not shape:
        return ()
    stride = 1
    result = [1]
    for s in shape[-1:0:-1]:
        stride *= s
        result.append(stride)
    return tuple(reversed(result))

def component_to_index(component, shape):
    i = 0
    for (c,s) in izip(component, strides(shape)):
        i += c*s
    return i

def index_to_component(index, shape):
    assert index >= 0
    component = []
    a, b = -123, -123
    for s in strides(shape):
        a = index // s
        b = index % s
        index = b
        component.append(a)
    assert all(c >= 0 for c in component)
    assert all(c < s for (c,s) in izip(component, shape))
    return tuple(component)
