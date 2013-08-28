"""Algorithm sketch to build canonical data structure for integrals over subdomains."""

#from ufl import *

from ufl import Domain, Region, Measure, Form

# Transitional helper constructor
from ufl.integral import Integral

from ufl.common import sorted_items

def integral_domain_ids(integral):
    did = integral.measure().domain_id()
    if isinstance(did, int):
        return (did,)
    elif isinstance(did, tuple):
        return did
    elif isinstance(did, Region):
        return did.subdomain_ids()
    elif isinstance(did, Domain):
        return Measure.DOMAIN_ID_EVERYWHERE
    elif did in Measure.DOMAIN_ID_CONSTANTS:
        return did
    else:
        error("Invalid domain id %s." % did)

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
            return False
        else:
            # NB! Comparing form compiler data here! Assuming this is an ok operation.
            return self.x[1] < other.x[1]
def expr_tuple_key(expr):
    return ExprTupleKey(expr)


def extract_domain_data_from_integral_list(integrals):
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

def extract_domain_data_from_integral_dict(integrals): # TODO: Is this really any better than the existing extract_domain_data?
    domain_data = {}
    # Iterate over domain types in order
    for dt in Measure._domain_types_tuple:
        # Get integrals list for this domain type if any
        if dt in integrals:
            domain_data[dt] = extract_domain_data_from_integral_list(integrals[dt])
    return domain_data


def integral_dict_to_sub_integral_data(integrals):
    # Data structures to return
    sub_integral_data = {}

    # Iterate over domain types in order
    for dt in Measure._domain_types_tuple:
        # Get integrals list for this domain type if any
        itgs = integrals.get(dt)
        if itgs is not None:
            # Make dict for this domain type with mapping (subdomain id -> integral list)
            sub_integrals = build_sub_integral_list(itgs)

            # Build a canonical representation of integrals for this type of domain,
            # with only one integrand for each compiler_data on each subdomain
            sub_integral_data[dt] = canonicalize_sub_integral_data(sub_integrals)

    # Return result:
    #sub_integral_data[dt][did][:] = [(integrand0, compiler_data0), (integrand1, compiler_data1), ...]
    return sub_integral_data

def reconstruct_form_from_sub_integral_data(sub_integral_data, domain_data=None):
    domain_data = domain_data or {}
    integrals = []
    # Iterate over domain types in order
    for dt in Measure._domain_types_tuple:
        dd = domain_data.get(dt)
        # Get integrals list for this domain type if any
        metaintegrands = sub_integral_data.get(dt)
        if metaintegrands is not None:
            for k in sorted(metaintegrands.keys()):
                for integrand, compiler_data in metaintegrands[k]:
                    integrals.append(Integral(integrand, dt, k, compiler_data, dd))
    return Form(integrals)

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
                # Restrict integral to this subdomain!
                sub_integrals[did].append(itg.reconstruct(domain_description=did))

    # Add everywhere integrals to each single subdomain id integral list
    if Measure.DOMAIN_ID_EVERYWHERE in sub_integrals:
        # We'll consume everywhere integrals...
        ei = sub_integrals[Measure.DOMAIN_ID_EVERYWHERE]
        del sub_integrals[Measure.DOMAIN_ID_EVERYWHERE]
        # ... and produce otherwise integrals instead
        assert Measure.DOMAIN_ID_OTHERWISE not in sub_integrals
        sub_integrals[Measure.DOMAIN_ID_OTHERWISE] = []
        # Restrict everywhere integral to each subdomain and append to each integral list
        for did, itglist in sorted_items(sub_integrals):
            for itg in ei:
                # Restrict integral to this subdomain!
                itglist.append(itg.reconstruct(domain_description=did))
    return sub_integrals

def canonicalize_sub_integral_data(sub_integrals):
    for did in sub_integrals:
        # Group integrals by compiler data object id
        by_cdid = {}
        for itg in sub_integrals[did]:
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
        sub_integrals[did] = sorted(by_cdid.values(), key=expr_tuple_key)
        # i.e. the result is on the form:
        #sub_integrals[did][:] = [(integrand0, compiler_data0), (integrand1, compiler_data1), ...]
        # where integrand0 < integrand1 by the canonical ufl expression ordering criteria.

    return sub_integrals

# Convert to integral_data format during transitional period:
from ufl.algorithms.analysis import IntegralData
def convert_sub_integral_data_to_integral_data(sub_integral_data):
    integral_data = []
    for domain_type, domain_type_data in sorted_items(sub_integral_data):
        for domain_id, sub_domain_integrands in sorted_items(domain_type_data):
            integrals = [Integral(integrand, domain_type, domain_id, compiler_data, None)
                         for integrand, compiler_data in sub_domain_integrands]
            ida = IntegralData(domain_type, domain_id, integrals, {})
            integral_data.append(ida)
    return integral_data


# Print output for inspection:
def print_sub_integral_data(sub_integral_data):
    print
    for domain_type, domain_type_data in sorted_items(sub_integral_data):
        print "======", domain_type
        for domain_id, sub_domain_integrands in sorted_items(domain_type_data):
            print '---', domain_id,
            for integrand, compiler_data in sub_domain_integrands:
                print
                print "integrand:    ", integrand
                print "compiler data:", compiler_data

def extract_integral_data_from_integral_dict(integrals):
    sub_integral_data = integral_dict_to_sub_integral_data(integrals)
    if 0: print_sub_integral_data(sub_integral_data) # TODO: Replace integral_data with this through ufl and ffc
    integral_data = convert_sub_integral_data_to_integral_data(sub_integral_data)
    return integral_data
