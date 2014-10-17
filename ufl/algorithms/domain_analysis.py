"""Algorithms for building canonical data structure for integrals over subdomains."""

# Copyright (C) 2009-2014 Anders Logg and Martin Sandve Alnes
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.

from collections import defaultdict
from six.moves import zip
from six import iteritems

import ufl
from ufl.utils.sorting import sorted_by_key
from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.geometry import Domain
from ufl.measure import Measure
from ufl.integral import Integral
from ufl.form import Form

from ufl.sorting import cmp_expr
from ufl.sorting import sorted_expr
from ufl.utils.sorting import canonicalize_metadata


class IntegralData(object):
    """Utility class with the members
        (domain, integral_type, subdomain_id, integrals, metadata)

    where metadata is an empty dictionary that may be used for
    associating metadata with each object.
    """
    __slots__ = ('domain', 'integral_type', 'subdomain_id', 'integrals', 'metadata',
                 'integral_coefficients', 'enabled_coefficients')
    def __init__(self, domain, integral_type, subdomain_id, integrals, metadata):
        ufl_assert(all(domain.label() == itg.domain().label() for itg in integrals),
                   "Domain label mismatch in integral data.")
        ufl_assert(all(integral_type == itg.integral_type() for itg in integrals),
                   "Domain type mismatch in integral data.")
        ufl_assert(all(subdomain_id == itg.subdomain_id() for itg in integrals),
                   "Domain id mismatch in integral data.")

        self.domain = domain
        self.integral_type = integral_type
        self.subdomain_id = subdomain_id

        self.integrals = integrals

        # This is populated in preprocess using data not available at this stage:
        self.integral_coefficients = None
        self.enabled_coefficients = None

        # TODO: I think we can get rid of this with some refactoring in ffc:
        self.metadata = metadata

    def __lt__(self, other):
        # To preserve behaviour of extract_integral_data:
        return ((self.integral_type, self.subdomain_id, self.integrals, self.metadata)
                < (other.integral_type, other.subdomain_id, other.integrals, other.metadata))

    def __eq__(self, other):
        # Currently only used for tests:
        return (self.integral_type == other.integral_type and
                self.subdomain_id == other.subdomain_id and
                self.integrals == other.integrals and
                self.metadata == other.metadata)

    def __str__(self):
        return "IntegralData object over domain (%s, %s), with integrals:\n%s\nand metadata:\n%s" % (
            self.integral_type, self.subdomain_id,
            '\n\n'.join(map(str, self.integrals)), self.metadata)

def dicts_lt(a, b):
    na = 0 if a is None else len(a)
    nb = 0 if b is None else len(b)
    if na != nb:
        return len(a) < len(b)
    for ia, ib in zip(sorted_by_key(a), sorted_by_key(b)):
        # Assuming keys are sortable (usually str)
        if ia[0] != ib[0]:
            return (ia[0].__class__.__name__, ia[0]) < (ib[0].__class__.__name__, ib[0]) # Hack to preserve type sorting in py3
        # Assuming values are sortable
        if ia[1] != ib[1]:
            return (ia[1].__class__.__name__, ia[1]) < (ib[1].__class__.__name__, ib[1]) # Hack to preserve type sorting in py3

# Tuple comparison helper
class ExprTupleKey(object):
    __slots__ = ('x',)
    def __init__(self, x):
        self.x = x
    def __lt__(self, other):
        # Comparing expression first
        c = cmp_expr(self.x[0], other.x[0])
        if c < 0:
            return True
        elif c > 0:
            return False
        else:
            # Comparing form compiler data
            mds = canonicalize_metadata(self.x[1])
            mdo = canonicalize_metadata(other.x[1])
            return mds < mdo

def group_integrals_by_domain_and_type(integrals, domains):
    """
    Input:
        integrals: list of Integral objects
        domains: list of Domain objects from the parent Form

    Output:
        integrals_by_domain_and_type: dict: (domain, integral_type) -> list(Integral)
    """
    integral_data = []
    domains_by_label = dict((domain.label(), domain) for domain in domains)

    integrals_by_domain_and_type = defaultdict(list)
    for itg in integrals:
        # Canonicalize domain
        domain = itg.domain()
        ufl_assert(domain is not None, "Integrals without a domain is now illegal.")
        domain = domains_by_label.get(domain.label())
        integral_type = itg.integral_type()

        # Append integral to list of integrals with shared key
        integrals_by_domain_and_type[(domain, integral_type)].append(itg)
    return integrals_by_domain_and_type

def integral_subdomain_ids(integral):
    "Get a tuple of integer subdomains or a valid string subdomain from integral."
    did = integral.subdomain_id()
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
        integrals: dict: subdomain_id -> list(Integral) (reconstructed with single subdomain_id)
    """
    # Split integrals into lists of everywhere and subdomain integrals
    everywhere_integrals = []
    subdomain_integrals = []
    for itg in integrals:
        dids = integral_subdomain_ids(itg)
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
            single_subdomain_integrals[did].append(itg.reconstruct(subdomain_id=did))

    # Add everywhere integrals to each single subdomain id integral list
    otherwise_integrals = []
    for ev_itg in everywhere_integrals:
        # Restrict everywhere integral to 'otherwise'
        otherwise_integrals.append(
            ev_itg.reconstruct(subdomain_id="otherwise"))

        # Restrict everywhere integral to each subdomain
        # and append to each integral list
        for subdomain_id in sorted(single_subdomain_integrals.keys()):
            single_subdomain_integrals[subdomain_id].append(
                ev_itg.reconstruct(subdomain_id=subdomain_id))

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
    # Group integrals by compiler data hash
    by_cdid = {}
    for itg in integrals:
        cd = itg.metadata()
        cdid = hash(canonicalize_metadata(cd))
        if cdid not in by_cdid:
            by_cdid[cdid] = ([], cd)
        by_cdid[cdid][0].append(itg)

    # Accumulate integrands separately for each compiler data object id
    for cdid in by_cdid:
        integrals, cd = by_cdid[cdid]
        # Ensure canonical sorting of more than two integrands
        integrands = sorted_expr((itg.integrand() for itg in integrals))
        integrands_sum = sum(integrands[1:], integrands[0])
        by_cdid[cdid] = (integrands_sum, cd)

    # Sort integrands canonically by integrand first then compiler data
    return sorted(by_cdid.values(), key=ExprTupleKey)

def build_integral_data(integrals, domains):
    integral_data = []

    # Group integrals by domain and type
    integrals_by_domain_and_type = \
        group_integrals_by_domain_and_type(integrals, domains)

    for domain in domains:
        for integral_type in ufl.measure.integral_types():
            # Get integrals with this domain and type
            ddt_integrals = integrals_by_domain_and_type.get((domain, integral_type))
            if ddt_integrals is None:
                continue

            # Group integrals by subdomain id, after splitting e.g.
            #   f*dx((1,2)) + g*dx((2,3)) -> f*dx(1) + (f+g)*dx(2) + g*dx(3)
            # (note: before this call, 'everywhere' is a valid subdomain_id,
            # and after this call, 'otherwise' is a valid subdomain_id)
            single_subdomain_integrals = \
                rearrange_integrals_by_single_subdomains(ddt_integrals)

            for subdomain_id, ss_integrals in sorted_by_key(single_subdomain_integrals):
                # Accumulate integrands of integrals that share the same compiler data
                integrands_and_cds = \
                    accumulate_integrands_with_same_metadata(ss_integrals)

                # Reconstruct integrals with new integrands and the right domain object
                integrals = [Integral(integrand, integral_type, domain, subdomain_id, metadata, None)
                             for integrand, metadata in integrands_and_cds]

                # Create new metadata dict for each integral data,
                # this is filled in by ffc to associate compiler
                # specific information with this integral data
                metadata = {}

                # Finally wrap it all in IntegralData object!
                ida = IntegralData(domain, integral_type, subdomain_id, integrals, {})

                # Store integral data objects in list with canonical ordering
                integral_data.append(ida)

    return integral_data

def reconstruct_form_from_integral_data(integral_data):
    integrals = []
    for ida in integral_data:
        integrals.extend(ida.integrals)
    return Form(integrals)
