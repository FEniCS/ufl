
from ufl.algorithms import load_forms, expand_indices

forms = load_forms("../../demo/StiffnessAD.ufl")

for f in forms:
    fd = f.form_data()
    g = fd.form
    print 
    print fd.name
    print 
    print str(f)
    print 
    print str(g)
    print 
    print str(expand_indices(g))
    print 

