
#
# TODO:
# - Make variable stacks work for mass matrix, source vector and L2norm
# - Fix variable handling in split_by_dependencies (see get_variable_deps)
# - Fix SpatialDerivative propagation and check with stiffness matrix, H1norm, and something that doesn't apply it directly to a terminal
#

import os, sys, pickle, shutil, glob

from ufl import *
from ufl.classes import *
from ufl.algorithms import *
from ufl.algorithms.dependencies import DependencySet

#---------------------------------------------------------

def dump_integrand_state(name, integrand):
    os.mkdir(name)
    os.chdir(name)
    write_file("repr", repr(integrand))
    write_file("str", str(integrand))
    write_file("latex", ufl2latex(integrand))
    write_file("tree", tree_format(integrand))
    #pickle.dump(integrand, open("integrand.pickle", "w")) # Must define __getstate__ to pickle UFLObject!
    os.chdir("..")

def write_file(name, contents):
    f = open(name, "w")
    f.write(contents)
    f.close()

def compile_integral(integrand, formdata):
    dump_integrand_state("a - initial", integrand)
    
    integrand = mark_duplications(integrand)
    dump_integrand_state("b - mark_duplications", integrand)
    
    integrand = expand_compounds(integrand, formdata.geometric_dimension)
    dump_integrand_state("c - expand_compounds", integrand)
    
    integrand = mark_duplications(integrand)
    dump_integrand_state("d - mark_duplications", integrand)
    
    # TODO: Apply AD stuff for Diff and propagation of SpatialDerivative to Terminal nodes
    
    basisfunction_deps = []
    fs = (False,)*formdata.num_coefficients
    for i in range(formdata.rank):
        bfs = tuple(i == j for j in range(formdata.rank)) # TODO: More dependencies depending on element
        d = DependencySet(bfs, fs)
        basisfunction_deps.append(d)

    function_deps = []
    bfs = (False,)*formdata.rank
    for i in range(formdata.num_coefficients):
        fs = tuple(i == j for j in range(formdata.num_coefficients)) # TODO: More dependencies depending on element
        d = DependencySet(bfs, fs)
        function_deps.append(d)
    
    do_split = True
    if do_split:
        (vinfo, code) = split_by_dependencies(integrand, formdata, basisfunction_deps, function_deps)
    
    print "------ FormData:"
    print formdata
    
    if do_split:
        print "------ Final variable info:"
        print vinfo

        print "------ Stacks:"
        for deps,stack in code.stacks.iteritems():
            print 
            print "Stack with deps =", deps
            print "\n".join(str(v) for v in stack)
 
        print "------ Variable info:"
        for count, v in code.variableinfo.iteritems():
            print
            print "v[%d] =" % count
            print v
        print

    #integrand = mark_dependencies(integrand)
    #integrand = expand_compounds(integrand)#, skip=(Transpose, ...)
    #integrand = compute_diffs(integrand)
    #integrand = propagate_spatial_diffs(integrand)
    ##integrand = expand_compounds(integrand)
    #(vinfo1, code1) = split_by_dependencies(integrand, formadata, basisfunction_deps, function_deps)
    #integrand = mark_dependencies(integrand)
    #(vinfo2, code2) = split_by_dependencies(integrand, formadata, basisfunction_deps, function_deps)




if __name__ == "__main__":

    #---------------------------------------------------------
    
    filename = "../demo/stiffness.ufl"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    forms = load_forms(filename)

    #---------------------------------------------------------

    outputdir = "output_basic_form_compiler"
    shutil.rmtree(outputdir, ignore_errors=True)
    os.mkdir(outputdir)
    os.chdir(outputdir)
    
    #---------------------------------------------------------

    for name, form in forms:
        if len(forms) > 1:
            os.mkdir("form_%s" % name)
            os.chdir("form_%s" % name)
        formdata = FormData(form)
        for i, integral in enumerate(form.cell_integrals()):
            os.mkdir("cell_integral_%d" % i)
            os.chdir("cell_integral_%d" % i)
            integrand = integral.integrand()
            compile_integral(integrand, formdata)
            os.chdir("..")
        for i, integral in enumerate(form.exterior_facet_integrals()):
            os.mkdir("exterior_facet_integral_%d" % i)
            os.chdir("exterior_facet_integral_%d" % i)
            integrand = integral.integrand()
            compile_integral(integrand, formdata)
            os.chdir("..")
        for i, integral in enumerate(form.interior_facet_integrals()):
            os.mkdir("interior_facet_integral_%d" % i)
            os.chdir("interior_facet_integral_%d" % i)
            integrand = integral.integrand()
            compile_integral(integrand, formdata)
            os.chdir("..")
        if len(forms) > 1:
            os.chdir("..")

    #---------------------------------------------------------

