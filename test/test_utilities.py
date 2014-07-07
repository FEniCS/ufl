#!/usr/bin/env python

"""
Test internal utility functions.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
from ufltestcase import UflTestCase, main

class UtilityTestCase(UflTestCase):

    def test_component_numbering(self):
        from ufl.permutation import build_component_numbering
        sh = (2, 2)
        sm = { (1, 0): (0, 1) }
        v, s = build_component_numbering(sh, sm)
        assert v == {(0, 1): 1, (1, 0): 1, (0, 0): 0, (1, 1): 2}
        assert s == [(0, 0), (0, 1), (1, 1)]

        sh = (3, 3)
        sm = { (1, 0): (0, 1), (2, 0): (0, 2), (2, 1): (1, 2) }
        v, s = build_component_numbering(sh, sm)
        assert v == {(0, 1): 1, (1, 2): 4, (0, 0): 0, (2, 1): 4, (1, 1): 3,
                     (2, 0): 2, (2, 2): 5, (1, 0): 1, (0, 2): 2}
        assert s == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    
    def test_component_indexing(self):
        from ufl.common import strides, component_to_index, index_to_component
        # Scalar shape
        s = ()
        self.assertEqual(strides(s), ())
        c = ()
        q = component_to_index(c, s)
        c2 = index_to_component(q, s)
        self.assertEqual(q, 0)
        self.assertEqual(c2, ())

        # Vector shape
        s = (2,)
        self.assertEqual(strides(s), (1,))
        for i in range(s[0]):
            c = (i,)
            q = component_to_index(c, s)
            c2 = index_to_component(q, s)
            #print c, q, c2
            #self.assertEqual(FIXME)

        # Tensor shape
        s = (2, 3)
        self.assertEqual(strides(s), (3, 1))
        for i in range(s[0]):
            for j in range(s[1]):
                c = (i, j)
                q = component_to_index(c, s)
                c2 = index_to_component(q, s)
                #print c, q, c2
                #self.assertEqual(FIXME)
    
        # Rank 3 tensor shape
        s = (2, 3, 4)
        self.assertEqual(strides(s), (12, 4, 1))
        for i in range(s[0]):
            for j in range(s[1]):
                for k in range(s[2]):
                    c = (i, j, k)
                    q = component_to_index(c, s)
                    c2 = index_to_component(q, s)
                    #print c, q, c2
                    #self.assertEqual(FIXME)
    
        # Taylor-Hood example:
    
        # pressure element is index 3:
        c = (3,)
        # get flat index:
        i = component_to_index(c, (4,))
        # remove offset:
        i -= 3
        # map back to component:
        c = index_to_component(i, ())
        #print c
        #self.assertEqual(FIXME)

        # vector element y-component is index 1:
        c = (1,)
        # get flat index:
        i = component_to_index(c, (4,))
        # remove offset:
        i -= 0
        # map back to component:
        c = index_to_component(i, (3,))
        #print c
        #self.assertEqual(FIXME)
    
        # Try a tensor/vector element:
        mixed_shape = (6,)
        ts = (2, 2)
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
        #print c
        #self.assertEqual(FIXME)

        for k in range(4):
            # tensor element (1,1)-component is index 3:
            c = (k,)
            # get flat index:
            i = component_to_index(c, mixed_shape)
            # remove offset:
            i -= 0
            # map back to vector component:
            c = index_to_component(i, ts)
            #print c
            #self.assertEqual(FIXME)

    def test_stackdict(self):
        from ufl.common import StackDict
        d = StackDict(a=1)
        self.assertEqual(d["a"], 1)
        d.push("a", 2)
        self.assertEqual(d["a"], 2)
        d.push("a", 3)
        d.push("b", 9)
        self.assertEqual(d["a"], 3)
        self.assertEqual(d["b"], 9)
        d.pop()
        self.assertEqual(d["a"], 3)
        self.assertTrue("b" not in d)
        d.pop()
        self.assertEqual(d["a"], 2)
        d.pop()
        self.assertEqual(d["a"], 1)

if __name__ == "__main__":
    main()
