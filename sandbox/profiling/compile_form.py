#!/usr/bin/env python

"""
Compile a linarized and partitioned graph for 
each integral for each form in a .ufl file.
"""

import sys, time
_msg = None
_t = None
_time_log = []
def tic(msg):
    global _t, _msg
    if _t is not None:
        toc()
    _t = -time.time()
    _msg = msg

def toc():
    global _t, _msg, _time_log
    t = _t+time.time()
    # format message and seconds
    msgtext = (" for %s" % _msg) if _msg else ""
    if t < 1e-2:
        ttext = "0 s"
    else:
        if t >= 60:
            ttext = "%.2f min" % (t / 60)
        else:
            ttext = "%.2f s" % t
    # log and print text
    line = "Time taken%s: %s" % (msgtext, ttext)
    _time_log.append((line, _msg, t))
    print(line)
    _t = None
    _msg = None

from operator import itemgetter
def print_timelog():
    lines = []
    msize = 0
    tsize = 0
    for l, m, t in _time_log:
        msize = max(msize, len(str(m)))
        tsize = max(tsize, len(str(t)))
    time_log = sorted(_time_log, key=itemgetter(2), reverse=True)
    format = "%%%ds: \t%%.2f" % msize
    for l, m, t in time_log:
        lines.append(format % (m, t))
    print("\n".join(lines))

from ufl.algorithms import load_forms, expand_indices, Graph, partition

filename = sys.argv[1] if len(sys.argv) > 1 else "../../demo/PoissonSystem.ufl"
print("Trying to load file ", filename)
tic("load_forms")
forms = load_forms(filename)
toc()

for form in forms:
    fd = form.form_data()
    f = fd.form
    print()
    print("="*80)
    print("== Handling form %s:" % fd.name)
    
    for itg in f.integrals():
        print()
        print("="*80)
        print("== Preparing integral %s:" % str(itg.measure()))
        idstr = "(%s, %s)" % (fd.name, str(itg.measure()))
    
        e = itg.integrand()

        tic("expand_indices %s" % idstr)
        e = expand_indices(e)

        tic("Graph %s" % idstr)
        G = Graph(e)
        V, E = G

        tic("Vout %s" % idstr)
        Vout = G.Vout()

        tic("partition %s" % idstr)
        P, keys = partition(G)
        toc()

        print()
        print("="*80)
        print("== Partition sizes:")
        print("|V| =", len(G.V()))
        print("|E| =", len(G.E()))
        print("|P| =", len(P))
        for i, p in enumerate(P):
            print("|p%d| =" % i, len(p))

        print()
        print("="*80)
        print("== Showing all partitions:")
        for key, part in six.iteritems(P):
            print("-"*60)
            print("-- Partition", key)

            for i in part:
                v = V[i]
                if Vout[i]:
                    ops = " applied to (%s)" % ", ".join("s%d" % j for j in Vout[i])
                else:
                    ops = ""
                print("s%d = %s%s" % (i, v._uflclass.__name__, ops))

print()
print("="*80)
print("== Timing summary:")
print_timelog()

