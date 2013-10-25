"""Algorithms for building canonical data structure for integrals over subdomains."""

from collections import defaultdict

import ufl
from ufl.common import sorted_items
from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.geometry import Domain
from ufl.measure import Measure
from ufl.integral import Integral
from ufl.form import Form

from ufl.sorting import cmp_expr
from ufl.sorting import sorted_expr

class IntegralData(object):
    """Utility class with the members
        (domain, domain_type, domain_id, integrals, metadata)

    where metadata is an empty dictionary that may be used for
    associating metadata with each object.
    """
    __slots__ = ('domain', 'domain_type', 'domain_id', 'integrals', 'metadata')
    def __init__(self, domain, domain_type, domain_id, integrals, metadata):
        ufl_assert(all(domain.label() == itg.domain().label() for itg in integrals),
                   "Domain label mismatch in integral data.")
        ufl_assert(all(domain_type == itg.domain_type() for itg in integrals),
                   "Domain type mismatch in integral data.")
        ufl_assert(all(domain_id == itg.domain_id() for itg in integrals),
                   "Domain id mismatch in integral data.")

        self.domain = domain
        self.domain_type = domain_type
        self.domain_id = domain_id

        self.integrals = integrals

        # TODO: I think we can get rid of this with some refactoring in ffc:
        self.metadata = metadata

    def __lt__(self, other):
        # To preserve behaviour of extract_integral_data:
        return ((self.domain_type, self.domain_id, self.integrals, self.metadata)
                < (other.domain_type, other.domain_id, other.integrals, other.metadata))

    def __eq__(self, other):
        # Currently only used for tests:
        return (self.domain_type == other.domain_type and
                self.domain_id == other.domain_id and
                self.integrals == other.integrals and
                self.metadata == other.metadata)

    def __str__(self):
        return "IntegralData object over domain (%s, %s), with integrals:\n%s\nand metadata:\n%s" % (
            self.domain_type, self.domain_id,
            '\n\n'.join(map(str,self.integrals)), self.metadata)


# Tuple comparison helper
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

def group_integrals_by_domain_and_type(integrals, domains, common_domain):
    """
    Input:
        integrals: list of Integral objects
        domains: list of Domain objects from the parent Form
        common_domain: default Domain object for integrals with no domain

    Output:
        integrals_by_domain_and_type: dict: (domain, domain_type) -> list(Integral)
    """
    integral_data = []
    domains_by_label = dict((domain.label(), domain) for domain in domains)

    integrals_by_domain_and_type = defaultdict(list)
    for itg in integrals:
        # Canonicalize domain
        domain = itg.domain()
        if domain is None:
            domain = common_domain
        domain = domains_by_label.get(domain.label())
        domain_type = itg.domain_type()

        # Append integral to list of integrals with shared key
        integrals_by_domain_and_type[(domain, domain_type)].append(itg)
    return integrals_by_domain_and_type

def integral_domain_ids(integral):
    "Get a tuple of integer subdomains or a valid string subdomain from integral."
    did = integral.domain_id()
    if isinstance(did, int):
        return (did,)
    elif isinstance(did, tuple):
        ufl_assert(all(isinstance(d, int) for d in did),
                   "Expecting only integer subdomains in tuple.")
        return did
    elif did in ("everywhere", "otherwise"):
        # TODO: Define list of valid strings somewhere more central
        return did
    else:
        error("Invalid domain id %s." % did)

def rearrange_integrals_by_single_subdomains(integrals):
    """Rearrange integrals over multiple subdomains to single subdomain integrals.

    Input:
        integrals: list(Integral)

    Output:
        integrals: dict: domain_id -> list(Integral) (reconstructed with single domain_id)
    """
    # Split integrals into lists of everywhere and subdomain integrals
    everywhere_integrals = []
    subdomain_integrals = []
    for itg in integrals:
        dids = integral_domain_ids(itg)
        if dids == "otherwise":
            error("'otherwise' integrals should never occur before preprocessing.")
        elif dids == "everywhere":
            everywhere_integrals.append(itg)
        else:
            subdomain_integrals.append((dids, itg))

    # Fill single_subdomain_integrals with lists of integrals from
    # subdomain_integrals, but split and restricted to single subdomain ids
    single_subdomain_integrals = defaultdict(list)
    for dids, itg in subdomain_integrals:
        # Region or single subdomain id
        for did in dids:
            # Restrict integral to this subdomain!
            single_subdomain_integrals[did].append(itg.reconstruct(domain_id=did))

    # Add everywhere integrals to each single subdomain id integral list
    otherwise_integrals = []
    for ev_itg in everywhere_integrals:
        # Restrict everywhere integral to 'otherwise'
        otherwise_integrals.append(
            ev_itg.reconstruct(domain_id="otherwise"))

        # Restrict everywhere integral to each subdomain
        # and append to each integral list
        for domain_id in sorted(single_subdomain_integrals.keys()):
            single_subdomain_integrals[domain_id].append(
                ev_itg.reconstruct(domain_id=domain_id))

    if otherwise_integrals:
        single_subdomain_integrals["otherwise"] = otherwise_integrals

    return single_subdomain_integrals

def accumulate_integrands_with_same_metadata(integrals):
    """
    Taking input on the form:
        integrals = [integral0, integral1, ...]

    Return result on the form:
        integrands_by_id = [(integrand0, metadata0),
                            (integrand1, metadata1), ...]

    where integrand0 < integrand1 by the canonical ufl expression ordering criteria.
    """
    # Group integrals by compiler data object id
    by_cdid = {}
    for itg in integrals:
        cd = itg.metadata()
        cdid = id(cd) # TODO: Use hash instead of id? Safe to assume this to be a dict of basic python values?
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
    return sorted(by_cdid.values(), key=expr_tuple_key)

def build_integral_data(integrals, domains, common_domain):
    integral_data = []

    # Group integrals by domain and type
    integrals_by_domain_and_type = \
        group_integrals_by_domain_and_type(integrals, domains, common_domain)

    for domain in domains:
        for domain_type in ufl.measure.domain_types():
            # Get integrals with this domain and type
            ddt_integrals = integrals_by_domain_and_type.get((domain, domain_type))
            if ddt_integrals is None:
                continue

            # Group integrals by subdomain id, after splitting e.g.
            #   f*dx((1,2)) + g*dx((2,3)) -> f*dx(1) + (f+g)*dx(2) + g*dx(3)
            # (note: before this call, 'everywhere' is a valid domain_id,
            # and after this call, 'otherwise' is a valid domain_id)
            single_subdomain_integrals = \
                rearrange_integrals_by_single_subdomains(ddt_integrals)

            for domain_id, ss_integrals in sorted_items(single_subdomain_integrals):
                # Accumulate integrands of integrals that share the same compiler data
                integrands_and_cds = \
                    accumulate_integrands_with_same_metadata(ss_integrals)

                # Reconstruct integrals with new integrands and the right domain object
                integrals = [Integral(integrand, domain_type, domain, domain_id, metadata, None)
                             for integrand, metadata in integrands_and_cds]

                # Create new metadata dict for each integral data,
                # this is filled in by ffc to associate compiler
                # specific information with this integral data
                metadata = {}

                # Finally wrap it all in IntegralData object!
                ida = IntegralData(domain, domain_type, domain_id, integrals, {})

                # Store integral data objects in list with canonical ordering
                integral_data.append(ida)

    return integral_data

def reconstruct_form_from_integral_data(integral_data):
    integrals = []
    for ida in integral_data:
        integrals.extend(ida.integrals)
    return Form(integrals)
