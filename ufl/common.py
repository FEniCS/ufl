"This module contains a collection of common utilities."

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-08-05 -- 2009-11-18"

# Modified by Kristian Oelgaard, 2009

import os
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

# TODO: The facet dim is just a dummy for now

# Mapping from domain (cell) to dimension
domain2dim = {None: None,
              "vertex": 0,
              "interval": 1,
              "triangle": 2,
              "tetrahedron": 3,
              "quadrilateral": 2,
              "hexahedron": 3,
              "facet": 0}

# Mapping from domain (cell) to facet
domain2facet = {None: None,
                "interval": "vertex",
                "triangle": "interval",
                "tetrahedron": "triangle",
                "quadrilateral": "interval",
                "hexahedron": "quadrilateral"}

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

def mergedicts(dicts):
    d = dict(dicts[0])
    for d2 in dicts[1:]:
        d.update(d2)
    return d

def subdict(superdict, keys):
    return dict((k, superdict[k]) for k in keys)

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

def some_key(a_dict):
    "Return an arbitrary key from a dictionary."
    return zip((0,), a_dict.iterkeys())[0][1]

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
    return tstr(d.items(), colsize)

def tstr(t, colsize=80):
    "Pretty-print list of tuples of key-value pairs."
    if not t:
        return ""

    # Compute maximum key length
    keylen = max([len(str(k)) for (k,v) in t])

    # Key-length cannot be larger than colsize
    if keylen > colsize:
        return str(t)

    # Pretty-print table
    s = ""
    for (key, value) in t:
        key, value = str(key), str(value)
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

class Counted(object):
    """A superclass for classes of objects identified by a global counter.

    Intended to be inherited to provide consistent counting logic.

    Usage:
    1) Inherit this class
    2) Declare a static class _globalcount variable in your subclass:
    3) Call Counted.__init__ at initialization.

    Minimal example:

        class MyClass(Counted):
            _globalcount = 0
            def __init__(self):
                Counted.__init__(self)

    If MyClass is further inherited, each subclass may get a
    different global counter, causing problems. Therefore
    it is recommended to pass the class to hold the global
    counter as an argument to Counted.__init__ like this:

        class MyClass(Counted):
            _globalcount = 0
            def __init__(self):
                Counted.__init__(self, count=None, countedclass=MyClass)

        class OtherClass(MyClass):
            def __init__(self):
                MyClass.__init__(self)
    """
    def __init__(self, count = None, countedclass = None):
        if countedclass is None:
            countedclass = type(self)
        self._countedclass = countedclass

        if count is None:
            self._count = self._countedclass._globalcount
            self._countedclass._globalcount += 1
        else:
            self._count = count
            if count >= self._countedclass._globalcount:
                self._countedclass._globalcount = count + 1

    def count(self):
        return self._count

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
    for (c,s) in zip(component, strides(shape)):
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
    assert all(c < s for (c,s) in zip(component, shape))
    return tuple(component)

def test_component_indexing():
    print
    s = ()
    print s, strides(s)
    c = ()
    q = component_to_index(c, s)
    c2 = index_to_component(q, s)
    print c, q, c2

    print
    s = (2,)
    print s, strides(s)
    for i in range(s[0]):
        c = (i,)
        q = component_to_index(c, s)
        c2 = index_to_component(q, s)
        print c, q, c2

    print
    s = (2,3)
    print s, strides(s)
    for i in range(s[0]):
        for j in range(s[1]):
            c = (i,j)
            q = component_to_index(c, s)
            c2 = index_to_component(q, s)
            print c, q, c2

    print
    s = (2,3,4)
    print s, strides(s)
    for i in range(s[0]):
        for j in range(s[1]):
            for k in range(s[2]):
                c = (i,j,k)
                q = component_to_index(c, s)
                c2 = index_to_component(q, s)
                print c, q, c2

    # Taylor-Hood example:

    # pressure element is index 3:
    c = (3,)
    # get flat index:
    i = component_to_index(c, (4,))
    # remove offset:
    i -= 3
    # map back to component:
    c = index_to_component(i, ())
    print c

    # vector element y-component is index 1:
    c = (1,)
    # get flat index:
    i = component_to_index(c, (4,))
    # remove offset:
    i -= 0
    # map back to component:
    c = index_to_component(i, (3,))
    print c

    # Try a tensor/vector element:
    mixed_shape = (6,)
    ts = (2,2)
    vs = (2,)
    offset = 4

    # vector element y-component is index offset+1:
    c = (offset+1,)
    # get flat index:
    i = component_to_index(c, mixed_shape)
    # remove offset:
    i -= offset
    # map back to vector component:
    c = index_to_component(i, vs)
    print c

    for k in range(4):
        # tensor element (1,1)-component is index 3:
        c = (k,)
        # get flat index:
        i = component_to_index(c, mixed_shape)
        # remove offset:
        i -= 0
        # map back to vector component:
        c = index_to_component(i, ts)
        print c

def test_stackdict():
    d = StackDict(a=1)
    d.push("a", 2)
    d.push("a", 3)
    print d
    d.pop()
    print d
    d.pop()
    print d

if __name__ == "__main__":
    test_component_indexing()
    test_stackdict()

