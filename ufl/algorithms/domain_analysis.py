# -*- coding: utf-8 -*-
"""Algorithms for building canonical data structure for integrals over subdomains."""

# Copyright (C) 2009-2016 Anders Logg and Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from collections import defaultdict

import ufl
from ufl.log import error
from ufl.integral import Integral
from ufl.form import Form
from ufl.protocols import id_or_none
from ufl.sorting import cmp_expr, sorted_expr
from ufl.utils.sorting import canonicalize_metadata, sorted_by_key
from ufl.algorithms.coordinate_derivative_helpers import attach_coordinate_derivatives, strip_coordinate_derivatives
import numbers
import typing


class IntegralData(object):
    """Utility class with the members (domain, integral_type,
        subdomain_id, integrals, metadata)

    where metadata is an empty dictionary that may be used for
    associating metadata with each object.

    """
    __slots__ = ('domain', 'integral_type', 'subdomain_id',
                 'integrals', 'metadata',
                 'integral_coefficients',
                 'enabled_coefficients')

    def __init__(self, domain, integral_type, subdomain_id, integrals,
                 metadata):
        if 1 != len(set(itg.ufl_domain() for itg in integrals)):
            error("Multiple domains mismatch in integral data.")
        if not all(integral_type == itg.integral_type() for itg in integrals):
            error("Integral type mismatch in integral data.")
        if not all(subdomain_id == itg.subdomain_id() for itg in integrals):
            error("Subdomain id mismatch in integral data.")

        self.domain = domain
        self.integral_type = integral_type
        self.subdomain_id = subdomain_id

        self.integrals = integrals

        # This is populated in preprocess using data not available at
        # this stage:
        self.integral_coefficients = None
        self.enabled_coefficients = None

        # TODO: I think we can get rid of this with some refactoring
        # in ffc:
        self.metadata = metadata

    def __lt__(self, other):
        # To preserve behaviour of extract_integral_data:
        return ((self.integral_type, self.subdomain_id,
                 self.integrals, self.metadata) <
                (other.integral_type, other.subdomain_id, other.integrals,
                 other.metadata))

    def __eq__(self, other):
        # Currently only used for tests:
        return (self.integral_type == other.integral_type and
                self.subdomain_id == other.subdomain_id and
                self.integrals == other.integrals and
                self.metadata == other.metadata)

    def __str__(self):
        s = "IntegralData over domain(%s, %s), with integrals:\n%s\nand metadata:\n%s" % (
            self.integral_type, self.subdomain_id,
            '\n\n'.join(map(str, self.integrals)), self.metadata)
        return s


def dicts_lt(a, b):
    na = 0 if a is None else len(a)
    nb = 0 if b is None else len(b)
    if na != nb:
        return len(a) < len(b)
    for ia, ib in zip(sorted_by_key(a), sorted_by_key(b)):
        # Assuming keys are sortable (usually str)
        if ia[0] != ib[0]:
            return (ia[0].__class__.__name__, ia[0]) < (ib[0].__class__.__name__, ib[0])  # Hack to preserve type sorting in py3
        # Assuming values are sortable
        if ia[1] != ib[1]:
            return (ia[1].__class__.__name__, ia[1]) < (ib[1].__class__.__name__, ib[1])  # Hack to preserve type sorting in py3


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
        domains: list of AbstractDomain objects from the parent Form

    Output:
        integrals_by_domain_and_type: dict: (domain, integral_type) -> list(Integral)
    """
    integrals_by_domain_and_type = defaultdict(list)
    for itg in integrals:
        if itg.ufl_domain() is None:
            error("Integral has no domain.")
        key = (itg.ufl_domain(), itg.integral_type())

        # Append integral to list of integrals with shared key
        integrals_by_domain_and_type[key].append(itg)

    return integrals_by_domain_and_type


def integral_subdomain_ids(integral):
    "Get a tuple of integer subdomains or a valid string subdomain from integral."
    did = integral.subdomain_id()
    if isinstance(did, numbers.Integral):
        return (did,)
    elif isinstance(did, tuple):
        if not all(isinstance(d, numbers.Integral) for d in did):
            error("Expecting only integer subdomains in tuple.")
        return did
    elif did in ("everywhere", "otherwise"):
        # TODO: Define list of valid strings somewhere more central
        return did
    else:
        error("Invalid domain id %s." % did)


def rearrange_integrals_by_single_subdomains(integrals: typing.List[Integral],
                                             do_append_everywhere_integrals: bool) -> typing.Dict[int, typing.List[Integral]]:
    """Rearrange integrals over multiple subdomains to single subdomain integrals.

    Input:
        integrals: List of integrals
        do_append_everywhere_integrals: Boolean indicating if integrals defined on the whole domain should
                                        just be restricted to the set of input subdomain ids.

    Output:
        integrals: The integrals reconstructed with single subdomain_id
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
    # subdomain_integrals, but split and restricted to single
    # subdomain ids
    single_subdomain_integrals = defaultdict(list)
    for dids, itg in subdomain_integrals:
        # Region or single subdomain id
        for did in dids:
            # Restrict integral to this subdomain!
            single_subdomain_integrals[did].append(itg.reconstruct(subdomain_id=did))

    # Add everywhere integrals to each single subdomain id integral
    # list
    otherwise_integrals = []
    for ev_itg in everywhere_integrals:
        # Restrict everywhere integral to 'otherwise'
        otherwise_integrals.append(ev_itg.reconstruct(subdomain_id="otherwise"))

        # Restrict everywhere integral to each subdomain
        # and append to each integral list
        if do_append_everywhere_integrals:
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

    # Accumulate integrands separately for each compiler data object
    # id
    for cdid in by_cdid:
        integrals, cd = by_cdid[cdid]
        # Ensure canonical sorting of more than two integrands
        integrands = sorted_expr((itg.integrand() for itg in integrals))
        integrands_sum = sum(integrands[1:], integrands[0])
        by_cdid[cdid] = (integrands_sum, cd)

    # Sort integrands canonically by integrand first then compiler
    # data
    return sorted(by_cdid.values(), key=ExprTupleKey)


def build_integral_data(integrals):
    """Build integral data given a list of integrals.

    :arg integrals: An iterable of :class:`~.Integral` objects.
    :returns: A tuple of :class:`IntegralData` objects.

    The integrals you pass in here must have been rearranged and
    gathered (removing the "everywhere" subdomain_id.  To do this, you
    should call :func:`group_form_integrals`.
    """
    itgs = defaultdict(list)

    # --- Merge integral data that has the same integrals,
    unique_integrals = defaultdict(tuple)
    metadata_table = defaultdict(dict)
    for integral in integrals:
        integrand = integral.integrand()
        integral_type = integral.integral_type()
        ufl_domain = integral.ufl_domain()
        metadata = integral.metadata()
        meta_hash = hash(canonicalize_metadata(metadata))
        subdomain_id = integral.subdomain_id()
        subdomain_data = id_or_none(integral.subdomain_data())
        if subdomain_id == "everywhere":
            raise ValueError("'everywhere' not a valid subdomain id.  Did you forget to call group_form_integrals?")
        unique_integrals[(integral_type, ufl_domain, meta_hash, integrand, subdomain_data)] += (subdomain_id,)
        metadata_table[(integral_type, ufl_domain, meta_hash, integrand, subdomain_data)] = metadata

    for integral_data, subdomain_ids in unique_integrals.items():
        (integral_type, ufl_domain, metadata, integrand, subdomain_data) = integral_data

        integral = Integral(integrand, integral_type, ufl_domain, subdomain_ids, metadata_table[integral_data], subdomain_data)
        # Group for integral data (One integral data object for all
        # integrals with same domain, itype, (but possibly different metadata).
        itgs[(ufl_domain, integral_type, subdomain_ids)].append(integral)

    # Build list with canonical ordering, iteration over dicts
    # is not deterministic across python versions
    def keyfunc(item):
        (d, itype, sid), integrals = item
        return (d._ufl_sort_key_(), itype, (type(sid).__name__, sid))

    integral_datas = []
    for (d, itype, sid), integrals in sorted(itgs.items(), key=keyfunc):
        integral_datas.append(IntegralData(d, itype, sid, integrals, {}))
    return integral_datas


def group_form_integrals(form, domains, do_append_everywhere_integrals=True):
    """Group integrals by domain and type, performing canonical simplification.

    :arg form: the :class:`~.Form` to group the integrals of.
    :arg domains: an iterable of :class:`~.Domain`s.
    :returns: A new :class:`~.Form` with gathered integrands.
    """
    # Group integrals by domain and type
    integrals_by_domain_and_type = group_integrals_by_domain_and_type(form.integrals(), domains)

    integrals = []
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
            single_subdomain_integrals = rearrange_integrals_by_single_subdomains(ddt_integrals, do_append_everywhere_integrals)

            for subdomain_id, ss_integrals in sorted_by_key(single_subdomain_integrals):

                # strip the coordinate derivatives from all integrals
                # this yields a list of the form [(coordinate derivative, integral), ...]
                stripped_integrals_and_coordderivs = strip_coordinate_derivatives(ss_integrals)

                # now group the integrals by the coordinate derivative
                def calc_hash(cd):
                    return sum(sum(tuple_elem._ufl_compute_hash_()
                                   for tuple_elem in tuple_) for tuple_ in cd)
                coordderiv_integrals_dict = {}
                for integral, coordderiv in stripped_integrals_and_coordderivs:
                    coordderivhash = calc_hash(coordderiv)
                    if coordderivhash in coordderiv_integrals_dict:
                        coordderiv_integrals_dict[coordderivhash][1].append(integral)
                    else:
                        coordderiv_integrals_dict[coordderivhash] = (coordderiv, [integral])

                # cd_integrals_dict is now a dict of the form
                # { hash: (CoordinateDerivative, [integral, integral, ...]), ... }
                # we can now put the integrals back together and then afterwards
                # apply the CoordinateDerivative again

                for cdhash, samecd_integrals in sorted_by_key(coordderiv_integrals_dict):

                    # Accumulate integrands of integrals that share the
                    # same compiler data
                    integrands_and_cds = accumulate_integrands_with_same_metadata(samecd_integrals[1])

                    for integrand, metadata in integrands_and_cds:
                        integral = Integral(integrand, integral_type, domain,
                                            subdomain_id, metadata, None)
                        integral = attach_coordinate_derivatives(integral, samecd_integrals[0])
                        integrals.append(integral)
    return Form(integrals)


def reconstruct_form_from_integral_data(integral_data):
    integrals = []
    for ida in integral_data:
        integrals.extend(ida.integrals)
    return Form(integrals)
