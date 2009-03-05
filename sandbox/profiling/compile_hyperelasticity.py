
from ufl.algorithms import load_forms, expand_indices, Graph, partition
forms = load_forms("../../demo/HyperElasticity.ufl")

for form in forms:
    fd = form.form_data()
    f = fd.form
    
    for itg in f.integrals():
        e = itg.integrand()
        e = expand_indices(e)
        G = Graph(e)
        P = partition(G)
        print "|V| =", len(G.V())
        print "|E| =", len(G.E())
        print "|P| =", len(P)
        for i, p in enumerate(P):
            print "|p%d| =" % i, len(p)

# Finished in 80 s on my home laptop:
#   In [1]: %run -p compile_hyperelasticity.py
#   |V| = 474
#   |E| = 895
#   |P| = 2
#   |p0| = 11
#   |p1| = 474
#   |V| = 211
#   |E| = 396
#   |P| = 2
#   |p0| = 8
#   |p1| = 211
#   |V| = 215
#   |E| = 375
#   |P| = 2
#   |p0| = 6
#   |p1| = 215
#   |V| = 75
#   |E| = 126
#   |P| = 2
#   |p0| = 5
#   |p1| = 75

