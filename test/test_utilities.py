#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

"""
Test internal utility functions.
"""

from six.moves import xrange as range

from ufl.utils.indexflattening import (shape_to_strides, flatten_multiindex,
                                       unflatten_index)


def test_shape_to_strides():
    assert () == shape_to_strides(())
    assert (1,) == shape_to_strides((3,))
    assert (2, 1) == shape_to_strides((3, 2))
    assert (4, 1) == shape_to_strides((3, 4))
    assert (12, 4, 1) == shape_to_strides((6, 3, 4))


def test_flatten_multiindex_to_multiindex():
    sh = (2, 3, 5)
    strides = shape_to_strides(sh)
    for i in range(sh[2]):
        for j in range(sh[1]):
            for k in range(sh[0]):
                index = (k, j, i)
                c = flatten_multiindex(index, strides)
                index2 = unflatten_index(c, strides)
                assert index == index2


def test_indexing_to_component():
    assert 0 == flatten_multiindex((), shape_to_strides(()))
    assert 0 == flatten_multiindex((0,), shape_to_strides((2,)))
    assert 1 == flatten_multiindex((1,), shape_to_strides((2,)))
    assert 3 == flatten_multiindex((1, 1), shape_to_strides((2, 2)))
    for i in range(5):
        for j in range(3):
            for k in range(2):
                assert 15*k+5*j+i == flatten_multiindex((k, j, i), shape_to_strides((2, 3, 5)))


def test_component_numbering():
    from ufl.permutation import build_component_numbering
    sh = (2, 2)
    sm = {(1, 0): (0, 1)}
    v, s = build_component_numbering(sh, sm)
    assert v == {(0, 1): 1, (1, 0): 1, (0, 0): 0, (1, 1): 2}
    assert s == [(0, 0), (0, 1), (1, 1)]

    sh = (3, 3)
    sm = {(1, 0): (0, 1), (2, 0): (0, 2), (2, 1): (1, 2)}
    v, s = build_component_numbering(sh, sm)
    assert v == {(0, 1): 1, (1, 2): 4, (0, 0): 0, (2, 1): 4, (1, 1): 3,
                 (2, 0): 2, (2, 2): 5, (1, 0): 1, (0, 2): 2}
    assert s == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def test_index_flattening():
    from ufl.utils.indexflattening import (shape_to_strides,
                                           flatten_multiindex,
                                           unflatten_index)
    # Scalar shape
    s = ()
    st = shape_to_strides(s)
    assert st == ()
    c = ()
    q = flatten_multiindex(c, st)
    c2 = unflatten_index(q, st)
    assert q == 0
    assert c2 == ()

    # Vector shape
    s = (2,)
    st = shape_to_strides(s)
    assert st == (1,)
    for i in range(s[0]):
        c = (i,)
        q = flatten_multiindex(c, st)
        c2 = unflatten_index(q, st)
        assert c == c2

    # Tensor shape
    s = (2, 3)
    st = shape_to_strides(s)
    assert st == (3, 1)
    for i in range(s[0]):
        for j in range(s[1]):
            c = (i, j)
            q = flatten_multiindex(c, st)
            c2 = unflatten_index(q, st)
            assert c == c2

    # Rank 3 tensor shape
    s = (2, 3, 4)
    st = shape_to_strides(s)
    assert st == (12, 4, 1)
    for i in range(s[0]):
        for j in range(s[1]):
            for k in range(s[2]):
                c = (i, j, k)
                q = flatten_multiindex(c, st)
                c2 = unflatten_index(q, st)
                assert c == c2

    # Taylor-Hood example:

    # pressure element is index 3:
    c = (3,)
    # get flat index:
    i = flatten_multiindex(c, shape_to_strides((4,)))
    # remove offset:
    i -= 3
    # map back to scalar component:
    c2 = unflatten_index(i, shape_to_strides(()))
    assert () == c2

    # vector element y-component is index 1:
    c = (1,)
    # get flat index:
    i = flatten_multiindex(c, shape_to_strides((4,)))
    # remove offset:
    i -= 0
    # map back to vector component:
    c2 = unflatten_index(i, shape_to_strides((3,)))
    assert (1,) == c2

    # Try a tensor/vector element:
    mixed_shape = (6,)
    ts = (2, 2)
    vs = (2,)
    offset = 4  # product(ts)

    # vector element y-component is index offset+1:
    c = (offset + 1,)
    # get flat index:
    i = flatten_multiindex(c, shape_to_strides(mixed_shape))
    # remove offset:
    i -= offset
    # map back to vector component:
    c2 = unflatten_index(i, shape_to_strides(vs))
    assert (1,) == c2

    for k in range(4):
        # tensor element (1,1)-component is index 3:
        c = (k,)
        # get flat index:
        i = flatten_multiindex(c, shape_to_strides(mixed_shape))
        # remove offset:
        i -= 0
        # map back to tensor component:
        c2 = unflatten_index(i, shape_to_strides(ts))
        assert (k//2, k % 2) == c2


def test_stackdict():
    from ufl.utils.stacks import StackDict
    d = StackDict(a=1)
    assert d["a"] == 1
    d.push("a", 2)
    assert d["a"] == 2
    d.push("a", 3)
    d.push("b", 9)
    assert d["a"] == 3
    assert d["b"] == 9
    d.pop()
    assert d["a"] == 3
    assert "b" not in d
    d.pop()
    assert d["a"] == 2
    d.pop()
    assert d["a"] == 1
