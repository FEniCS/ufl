
class Integral:
    def __init__(self, integrand, domain_type, domain_id, compiler_data, solver_data):

        self._domain_type = domain_type

        if isinstance(domain_id, int):
            self._domain_ids = (domain_id,)
        elif isinstance(domain_id, tuple):
            self._domain_ids = domain_id
        elif isinstance(domain_id, str):
            self._domain_ids = domain_id
        else:
            error

        self._integrand = integrand

        self._compiler_data = compiler_data
        self._solver_data = solver_data

    def restricted(self, domain_id):
        if self.domain_ids() == (domain_id,):
            return self
        else:
            return Integral(self.integrand(), self.domain_type(), domain_id, self.compiler_data(), self.solver_data())

    def annotated(self, compiler_data=None, solver_data=None):
        cd = self.compiler_data()
        sd = self.solver_data()
        if cd is compiler_data and sd is solver_data:
            return self
        else:
            if compiler_data is None:
                compiler_data = cd
            if solver_data is None:
                solver_data = sd
            return Integral(self.integrand(), self.domain_type(), self.domain_ids(), compiler_data, solver_data)

    def integrand(self):
        return self._integrand

    def domain_type(self):
        return self._domain_type

    def domain_ids(self):
        return self._domain_ids

    def compiler_data(self):
        return self._compiler_data

    def solver_data(self):
        return self._solver_data

    def __str__(self):
        return "I(%s, %s, %s, %s, %s)" % (self.integrand(), self.domain_type(), self.domain_ids(),
                                          self.compiler_data(), self.solver_data())

    def __repr__(self):
        return "Integral(%r, %r, %r, %r, %r)" % (self.integrand(), self.domain_type(), self.domain_ids(),
                                                 self.compiler_data(), self.solver_data())

# Mock objects for compiler data and solver data
comp1 = [1,2,3]
comp2 = ('a', 'b')
comp3 = {'1':1}
sol1 = (0,3,5)
sol2 = (0,3,7)

# Mock list of integral objects
integrals = {}
integrals["cell"] = [# Integrals over 0 with no compiler_data:
                     Integral("foo", "cell", 0, None, None), Integral("bar", "cell", 0, None, sol1),
                     # Integrals over 0 with different compiler_data:
                     Integral("foy", "cell", 0, comp1, None), Integral("bay", "cell", 0, comp2, None),
                     # Integrals over 1 with same compiler_data object:
                     Integral("foz", "cell", 1, comp1, None), Integral("baz", "cell", 1, comp1, sol1),
                     # Integral over 0 and 1 with compiler_data object found in 0 but not 1 above:
                     Integral("fro", "cell", (0,1), comp2, None),
                     # Integral over 0 and 1 with its own compiler_data object:
                     Integral("bro", "cell", (0,1), comp3, None),
                     # Integral over 0 and 1 no compiler_data object:
                     Integral("reg", "cell", (0,1), None, None),
                     # Integral over everywhere with no compiler data:
                     Integral("evr", "cell", "everywhere", None, None),
                     ]

# Tuple comparison helper
#from ufl.sorting import cmp_expr
cmp_expr = cmp
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


### Algorithm sketch to build canonical data structure for integrals over subdomains
from collections import defaultdict
#from ufl.sorting import sorted_expr
sorted_expr = sorted
sub_integrals = {}
solver_datas = {}
# Iterate over domain types in order
domain_types = ('cell',) # Measure._domain_types_tuple # TODO
for dt in domain_types:
    # Get integrals list for this domain type if any
    itgs = integrals.get(dt)
    if itgs is not None:
        # Keep track of solver data objects, want only one
        sdids = set()
        solver_data = None

        # Make dict for this domain type with mapping (subdomain id -> integral list)
        sitgs = defaultdict(list)
        sub_integrals[dt] = sitgs

        # Now fill sitgs with lists of integrals sorted by and restricted to subdomain ids
        all_dids = set()
        for itg in itgs:
            sd = itg.solver_data()
            if sd is not None:
                solver_data = sd
                sdids.add(id(sd))

            dids = itg.domain_ids()
            if dids == "everywhere":
                # Everywhere integral
                sitgs["everywhere"].append(itg)
            else:
                # Region or single subdomain id
                all_dids.update(dids)
                for did in itg.domain_ids():
                    sitgs[did].append(itg.restricted(did)) # Restrict integral to this subdomain!

        # Add everywhere integrals to each single subdomain id integral list
        if "everywhere" in sitgs:
            ei = sitgs["everywhere"]
            assert "otherwise" not in sitgs
            sitgs["otherwise"] = []
            all_dids.add("otherwise")
            for did in tuple(all_dids) + ("otherwise",):
                for itg in ei:
                    # Restrict everywhere integral to this subdomain!
                    sitgs[did].append(itg.restricted(did))
            del sitgs["everywhere"]
            assert "everywhere" not in all_dids
        # From this point on, treat None as otherwise instead of everywhere

        # Then finally make a canonical representation of integrals with only one integral object for each compiler_data on each subdomain
        for did in all_dids:
            # Group integrals by compiler data object id
            by_cdid = {}
            l = sitgs[did]
            for itg in l:
                cd = itg.compiler_data()
                cdid = id(cd)
                if cdid in by_cdid:
                    by_cdid[cdid][0].append(itg)
                else:
                    by_cdid[cdid] = ([itg], cd)

            # Accumulate integrands separately for each compiler data object id
            for cdid in by_cdid.keys():
                integrals, cd = by_cdid[cdid]
                # Ensure canonical sorting of more than two integrands
                integrands = sorted_expr((itg.integrand() for itg in integrals))
                integrands_sum = ''.join(integrands) # TODO: Use sum with proper ufl integrands
                by_cdid[cdid] = (integrands_sum, cd)

            # Sort integrands canonically by integrand first then compiler data
            sitgs[did] = sorted(by_cdid.values(), key=expr_tuple_key)

            # Result:
            #sub_integrals[dt][did][:] = [(integrand0, compiler_data0), (integrand1, compiler_data1), ...]

        # Store single solver data object for this domain type
        if solver_data is not None:
            solver_datas[dt] = solver_data
        assert len(sdids) <= 1, ("Found multiple solver data objects in form for domain type %s" % dt)

sub_integral_data = sub_integrals
print
for domain_type, domain_type_data in sub_integral_data.iteritems():
    print "======", domain_type
    for domain_id, sub_domain_integrands in domain_type_data.iteritems():
        print '---', domain_id,
        for (integrand, compiler_data) in sub_domain_integrands:
            print
            print "integrand:    ", integrand
            print "compiler data:", compiler_data
print
print solver_datas
print
#return sub_integrals, solver_datas
