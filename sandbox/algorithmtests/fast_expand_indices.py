
from ufl import *
from ufl.common import *
from ufl.classes import *
from ufl.algorithms import *
from ufl.permutation import compute_indices


def fast_expand_indices(e):
    assert isinstance(e, Expr)

    G = Graph(e)
    V, E = G
    n = len(V)
    Vout = G.Vout()
    #Vin = G.Vin()
    
    # Cache free indices
    fi   = []
    idim = []
    for v in V:
        if isinstance(v, MultiIndex):
            ii = tuple(j for j in v._indices if isinstance(j, Index))
            idims = {} # Hard problem: Need index dimensions but they're defined by the parent.
        else:
            ii = v.free_indices()
            idims = v.index_dimensions()
        fi.append(ii)
        idim.append(idims)

    # Cache of expanded expressions
    V2 = [{} for _ in V]
    def getv(i, indmap):
        return V2[i][tuple(indmap[j] for j in fi[i])]
    
    # Reversed enumeration list
    RV = list(enumerate(V))
    RV.reverse()
    
    # Map of current index values
    indmap = StackDict()
    
    # Expand all vertices in turn
    for i, v in enumerate(V):
        ii = fi[i]
        if ii:
            idims = idim[i]
            dii = tuple(idims[j] for j in ii)
            perms = compute_indices(dii)
        else:
            perms = [()]
        for p in perms:
            if isinstance(v, MultiIndex):
                # Map to FixedIndex tuple
                comp = []
                k = 0
                for j in v._indices:
                    if isinstance(j, FixedIndex):
                        comp.append(j)
                    elif isinstance(i, Index):
                        comp.append(FixedIndex(p[k]))
                        k += 1
                e = MultiIndex(tuple(comp))

            elif isinstance(v, Terminal):
                # Simply reuse
                e = v

            elif isinstance(v, IndexSum):
                for k in range(v.dimension()):
                    indmap.push(v.index(), k)
                    # Get operands evaluated for this index configuration
                    ops = [getv(j, indmap) for j in Vout[i]]
                    # It is possible to save memory 
                    # by reusing some expressions here
                    e = v.reconstruct(*ops)
                    indmap.pop()

            elif isinstance(v, Indexed):
                # Get operands evaluated for this index configuration
                ops = [getv(j, indmap) for j in Vout[i]]
                e = v.reconstruct(*ops)

            elif isinstance(v, ComponentTensor):
                FIXME # map indices indmap

            else:
                # Get operands evaluated for this index configuration
                ops = [getv(j, indmap) for j in Vout[i]]
                # It is possible to save memory 
                # by reusing some expressions here
                e = v.reconstruct(*ops)
            V2[i][p] = e
        
    return V2[-1][()]


from ufl.testobjects import *
if __name__ == "__main__":
    print fast_expand_indices(v)
    print fast_expand_indices(u*v)
    print fast_expand_indices(vv[0])
    print fast_expand_indices(u*vv[0])
    print fast_expand_indices(vu[0]*vv[0])
    print fast_expand_indices(u*v)
    print fast_expand_indices(vu[i]*vv[i])
