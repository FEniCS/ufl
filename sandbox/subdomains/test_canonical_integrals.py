"""Algorithm sketch to build canonical data structure for integrals over subdomains."""

from ufl import *


# Transitional helper constructor
from ufl.integral import Integral2

def integral_domain_ids(integral):
    did = integral.measure().domain_id()
    if isinstance(did, int):
        return (did,)
    elif isinstance(did, tuple):
        return did
    elif isinstance(did, Region):
        return did.sub_domain_ids()
    elif isinstance(did, Domain):
        return Measure.DOMAIN_ID_EVERYWHERE
    elif isinstance(did, str):
        return did

def restricted_integral(integral, domain_id):
    if integral_domain_ids(integral) == (domain_id,):
        return integral
    else:
        return Integral2(integral.integrand(), integral.domain_type(), domain_id, integral.compiler_data(), integral.domain_data())

def restricted_integral(integral, domain_id):
    if integral_domain_ids(integral) == (domain_id,):
        return integral
    else:
        return Integral2(integral.integrand(), integral.domain_type(), domain_id, integral.compiler_data(), integral.domain_data())

def annotated_integral(integral, compiler_data=None, domain_data=None):
    cd = integral.compiler_data()
    sd = integral.domain_data()
    if cd is compiler_data and sd is domain_data:
        return integral
    else:
        if compiler_data is None:
            compiler_data = cd
        if domain_data is None:
            domain_data = sd
        return Integral2(integral.integrand(), integral.domain_type(), integral.domain_ids(), compiler_data, domain_data)


# Tuple comparison helper
from collections import defaultdict
from ufl.sorting import cmp_expr
from ufl.sorting import sorted_expr

class ExprTupleKey(object):
    __slots__ = ('x',)
    def __init__(self, x):
        self.x = x
    def __lt__(self, other):
        c = cmp_expr(self.x[0], other.x[0])
        if c < 0:
            return True
        elif c > 0:
            return self.x[1] < other.x[1]
        else:
            return False
def expr_tuple_key(expr):
    return ExprTupleKey(expr)

def integral_dict_to_sub_integral_data(integrals):
    # Data structures to return
    sub_integral_data = {}
    domain_data = {}

    # Iterate over domain types in order
    domain_types = ('cell',) # Measure._domain_types_tuple # TODO
    for dt in domain_types:
        # Get integrals list for this domain type if any
        itgs = integrals.get(dt)
        if itgs is not None:
            # Make dict for this domain type with mapping (subdomain id -> integral list)
            sub_integrals = build_sub_integral_list(itgs)

            # Build a canonical representation of integrals for this type of domain,
            # with only one integral object for each compiler_data on each subdomain
            sub_integral_data[dt] = canonicalize_sub_integral_data(sub_integrals)

            # Get domain data object for this domain type and check that we found at most one
            domain_data[dt] = extract_domain_data_from_integrals(itgs)

    # Return result:
    #sub_integral_data[dt][did][:] = [(integrand0, compiler_data0), (integrand1, compiler_data1), ...]
    return sub_integral_data, domain_data

def build_sub_integral_list(itgs):
    sub_integrals = defaultdict(list)

    # Fill sub_integrals with lists of integrals sorted by and restricted to subdomain ids
    for itg in itgs:
        dids = integral_domain_ids(itg)
        assert dids != Measure.DOMAIN_ID_OTHERWISE
        if dids == Measure.DOMAIN_ID_EVERYWHERE:
            # Everywhere integral
            sub_integrals[Measure.DOMAIN_ID_EVERYWHERE].append(itg)
        else:
            # Region or single subdomain id
            for did in dids:
                sub_integrals[did].append(restricted_integral(itg, did)) # Restrict integral to this subdomain!

    # Add everywhere integrals to each single subdomain id integral list
    if Measure.DOMAIN_ID_EVERYWHERE in sub_integrals:
        # We'll consume everywhere integrals...
        ei = sub_integrals[Measure.DOMAIN_ID_EVERYWHERE]
        del sub_integrals[Measure.DOMAIN_ID_EVERYWHERE]
        # ... and produce otherwise integrals instead
        assert Measure.DOMAIN_ID_OTHERWISE not in sub_integrals
        sub_integrals[Measure.DOMAIN_ID_OTHERWISE] = []
        # Restrict everywhere integral to each subdomain and append to each integral list
        for did, itglist in sub_integrals.iteritems():
            for itg in ei:
                itglist.append(restricted_integral(itg, did))
    return sub_integrals

def extract_domain_data_from_integrals(integrals):
    # Keep track of domain data objects, want only one
    ddids = set()
    domain_data = None
    for itg in integrals:
        dd = itg.domain_data()
        if dd is not None:
            domain_data = dd
            ddids.add(id(dd))
    assert len(ddids) <= 1, ("Found multiple domain data objects in form for domain type %s" % dt)
    return domain_data

def typed_integrals_to_sub_integral_data(itgs):
    # Make dict for this domain type with mapping (subdomain id -> integral list)
    sitgs = build_sub_integral_list(itgs)

    # Then finally make a canonical representation of integrals with only
    # one integral object for each compiler_data on each subdomain
    return canonicalize_sub_integral_data(sitgs)

def canonicalize_sub_integral_data(sitgs):
    for did in sitgs:
        # Group integrals by compiler data object id
        by_cdid = {}
        for itg in sitgs[did]:
            cd = itg.compiler_data()
            cdid = id(cd)
            if cdid in by_cdid:
                by_cdid[cdid][0].append(itg)
            else:
                by_cdid[cdid] = ([itg], cd)

        # Accumulate integrands separately for each compiler data object id
        for cdid in by_cdid:
            integrals, cd = by_cdid[cdid]
            # Ensure canonical sorting of more than two integrands
            integrands = sorted_expr((itg.integrand() for itg in integrals))
            integrands_sum = sum(integrands[1:], integrands[0])
            by_cdid[cdid] = (integrands_sum, cd)

        # Sort integrands canonically by integrand first then compiler data
        sitgs[did] = sorted(by_cdid.values(), key=expr_tuple_key)
        # E.g.:
        #sitgs[did][:] = [(integrand0, compiler_data0), (integrand1, compiler_data1), ...]

    return sitgs

# Convert to integral_data format during transitional period:
from ufl.algorithms.analysis import IntegralData
def sub_integral_data_to_integral_data(sub_integral_data):
    integral_data = []
    for domain_type, domain_type_data in sub_integral_data.iteritems():
        for domain_id, sub_domain_integrands in domain_type_data.iteritems():
            integrals = [Integral2(integrand, domain_type, domain_id, compiler_data, None)
                         for integrand, compiler_data in sub_domain_integrands]
            ida = IntegralData(domain_type, domain_id, integrals, {})
            integral_data.append(ida)
    return integral_data


# Print output for inspection:
def print_sub_integral_data(sub_integral_data):
    print
    for domain_type, domain_type_data in sub_integral_data.iteritems():
        print "======", domain_type
        for domain_id, sub_domain_integrands in domain_type_data.iteritems():
            print '---', domain_id,
            for integrand, compiler_data in sub_domain_integrands:
                print
                print "integrand:    ", integrand
                print "compiler data:", compiler_data

# Run for testing and inspection
def test():
    # Mock objects for compiler data and solver data
    comp1 = [1,2,3]
    comp2 = ('a', 'b')
    comp3 = {'1':1}
    sol1 = (0,3,5)
    sol2 = (0,3,7)

    # Basic UFL expressions for integrands
    V = FiniteElement("CG", triangle, 1)
    f = Coefficient(V)
    g = Coefficient(V)
    h = Coefficient(V)

    # FIXME: Replace these with real Integral objects
    # Mock list of integral objects
    integrals = {}
    integrals["cell"] = [# Integrals over 0 with no compiler_data:
                         Integral2(f, "cell", 0, None, None),
                         Integral2(g, "cell", 0, None, sol1),
                         # Integrals over 0 with different compiler_data:
                         Integral2(f**2, "cell", 0, comp1, None),
                         Integral2(g**2, "cell", 0, comp2, None),
                         # Integrals over 1 with same compiler_data object:
                         Integral2(f**3, "cell", 1, comp1, None),
                         Integral2(g**3, "cell", 1, comp1, sol1),
                         # Integral over 0 and 1 with compiler_data object found in 0 but not 1 above:
                         Integral2(f**4, "cell", (0,1), comp2, None),
                         # Integral over 0 and 1 with its own compiler_data object:
                         Integral2(g**4, "cell", (0,1), comp3, None),
                         # Integral over 0 and 1 no compiler_data object:
                         Integral2(h/3, "cell", (0,1), None, None),
                         # Integral over everywhere with no compiler data:
                         Integral2(h/2, "cell", Measure.DOMAIN_ID_EVERYWHERE, None, None),
                         ]


    sub_integral_data, domain_datas = integral_dict_to_sub_integral_data(integrals)

    print
    print "Domain data:"
    print domain_datas
    print

    print_sub_integral_data(sub_integral_data)

    integral_data = sub_integral_data_to_integral_data(sub_integral_data)

test()
