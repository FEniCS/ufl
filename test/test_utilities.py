#!/usr/bin/env python

"""
Test internal utility functions.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
from ufltestcase import UflTestCase, main

class UtilityTestCase(UflTestCase):

    def test_component_numbering(self):
        from ufl.permutation import build_component_numbering
        sh = (2,2)
        sm = { (1,0): (0,1) }
        v, s = build_component_numbering(sh, sm)
        assert v == {(0, 1): 1, (1, 0): 1, (0, 0): 0, (1, 1): 2}
        assert s == [(0, 0), (0, 1), (1, 1)]

        sh = (3,3)
        sm = { (1,0): (0,1), (2,0): (0,2), (2,1): (1,2) }
        v, s = build_component_numbering(sh, sm)
        assert v == {(0, 1): 1, (1, 2): 4, (0, 0): 0, (2, 1): 4, (1, 1): 3,
                     (2, 0): 2, (2, 2): 5, (1, 0): 1, (0, 2): 2}
        assert s == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

if __name__ == "__main__":
    main()
