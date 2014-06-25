
from ufl.algorithms import *
forms = load_forms("ExpandIndices2Fail.ufl")


from time import time

for f in forms:
    name = f.form_data().name
    #if name != "a_J": continue

    print("="*80)
    print("Form ", name)
    #print str(f)
    print()
    
    a1 = f
    a2 = f
    
    t = -time()
    a2 = expand_indices2(a2)
    t += time()
    print("expand_indices2 time: %.3f s" % t)
    
    t = -time()
    a1 = expand_indices(a1)
    t += time()
    print("expand_indices time: %.3f s" % t)
    
    print("repr(a1) == repr(a2): ", repr(a1) == repr(a2))

