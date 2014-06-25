from sys import getsizeof
import ufl
from ufl.algorithms import pre_traversal
from ufl.classes import Terminal, Operator

varname_labels = [\
    ("num_unique_nodes",    "Number of unique nodes:            "),
    ("num_repeated_nodes",  "Number of nodes with repetition:   "),
    ("num_hash_values",     "Number of unique hash values:      "),
    ("num_repr_members",    "Number of nodes with _repr member: "),
    ("footprint_of_repr",   "Memory footprint of repr members:  "),
    ("footprint_of_nodes",  "Memory footprint of node objects:  "),
    ]

class ExpressionAnalysisReport(object):
    def __init__(self):
        self.num_repeated_nodes = 0
        self.num_unique_nodes = 0
        self.num_hash_values = 0
        self.num_repr_members = 0
        self.footprint_of_repr = 0
        self.footprint_of_nodes = 0

    def __iadd__(self, other):
        for k,v in vars(other).iteritems():
            setattr(self, k, getattr(self, k) + v)
        return self

    def __isub__(self, other):
        for k,v in vars(other).iteritems():
            setattr(self, k, getattr(self, k) - v)
        return self

    def __add__(self, other):
        n = ExpressionAnalysisReport()
        n += self
        n += other
        return n

    def __sub__(self, other):
        n = ExpressionAnalysisReport()
        n += self
        n -= other
        return n

    def __str__(self):
        return '\n'.join("%s  %s" % (l,getattr(self,k)) for k, l in varname_labels)

def accumulate_results(reports, e, visited_ids, hashes):
    # Count nodes with repetition
    for report in reports:
        report.num_repeated_nodes += 1

    # Skip already visited, the rest is without repetition
    if id(e) in visited_ids:
        return
    visited_ids.add(id(e))

    # Compute some info
    fp = getsizeof(e)
    h = hash(e)
    if not h in hashes:
        hashes.add(h)
        h = 1
    else:
        h = 0
    r = 1 if hasattr(e, '_repr') else 0
    rfp = getsizeof(e._repr) if r else 0

    # Accumulate into all reports
    for report in reports:
        report.num_unique_nodes += 1
        report.footprint_of_nodes += fp
        report.num_hash_values += h
        report.num_repr_members += r
        report.footprint_of_repr += rfp

def analyse_expression(expr):
    total_report = ExpressionAnalysisReport()
    terminal_report = ExpressionAnalysisReport()
    operator_report = ExpressionAnalysisReport()
    type_reports = dict((cl.__name__, ExpressionAnalysisReport()) for cl in ufl.classes.all_ufl_classes)

    visited_ids = set()
    hashes = set()
    for e in ufl.algorithms.pre_traversal(expr):
        # Get UFL class name
        cln = e.__class__._uflclass.__name__

        # Pick reports to add to
        reports = [total_report, type_reports[cln]]
        if isinstance(e, Operator):
            reports.append(operator_report)
        elif isinstance(e, Terminal):
            reports.append(terminal_report)

        # Add results
        accumulate_results(reports, e, visited_ids, hashes)

    return total_report, terminal_report, operator_report, type_reports

_item_template = """{label}  {{total_report.{key}}}
    - terminals:        {{terminal_report.{key}}}
    - operators:        {{operator_report.{key}}}"""
_class_template = "    - {classname}:{spacing} {key}"

def format_expression_analysis(total_report, terminal_report, operator_report, type_reports):
    classes = sorted(type_reports.keys())
    maxlen = max([len(c) for c in classes] + [0])
    parts = []
    for k, l in varname_labels:
        tmp = _item_template.format(label=l, key=k)
        parts.append(tmp.format(total_report=total_report,
                                terminal_report=terminal_report,
                                operator_report=operator_report))
        for c in classes:
            r = type_reports[c]
            if r.num_unique_nodes > 0:
                sp = "" if len(c) >= maxlen else (maxlen-len(c))*" "
                parts.append(_class_template.format(classname=c, spacing=sp, key=getattr(r,k)))
    return "\n".join(parts)

def formatted_analysis(expr, classes=False):
    res = analyse_expression(expr)
    if not classes:
        res[3].clear()
    # TODO: A table formatting would perhaps be nicer
    return format_expression_analysis(*res)

def rsizeof(obj, visited=None):
    if visited is None:
        visited = set()
    if id(obj) in visited:
        return 0
    visited.add(id(obj))
    s = getsizeof(obj)
    if 1:
        s += sum(rsizeof(o) for o in obj.operands())
    elif hasattr(obj.__class__, '__slots__'):
        s += sum(rsizeof(getattr(obj,n)) for n in obj.__class__.__slots__)
    else:
        s += sum(rsizeof(o) for o in vars(obj).itervalues())
    return s

from pympler.asizeof import asizeof

def test():
    from ufl import FiniteElement, triangle, Coefficient, as_ufl
    V = FiniteElement("CG", triangle, 1)
    u = Coefficient(V)
    v = Coefficient(V)
    o = as_ufl(1)
    g = v+o
    t = as_ufl(2)
    f = g**t
    e = u*f
    print(formatted_analysis(e, classes=True))
    #print rsizeof(V)
    for n in ['o', 't', 'u', 'v', 'g', 'f', 'e']:
        obj = eval(n)
        print(n, getsizeof(obj), rsizeof(obj), asizeof(obj), type(obj).__name__)
    print(id(o._index_dimensions))
    print(id(t._index_dimensions))
    print(id(ufl.common.EmptyDict))
test()

