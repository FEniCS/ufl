
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.constantvalue import Zero
from ufl import as_vector
from ufl.algorithms import expand_derivatives

class FormSplitter(MultiFunction):

    def split(self, form, ix, iy=0):
        # remember which block to extract
        self.idx = [ix, iy]
        # FIXME: is this needed?
#        form = expand_derivatives(form)
        return map_integrand_dags(self, form)

    def argument(self, obj):
        Q = obj.ufl_function_space()
        dom = Q.ufl_domain()
        elem = obj.ufl_element()
        args = []
        for i, sub_elem in enumerate(elem.sub_elements()):
            Q_i = FunctionSpace(dom, sub_elem)
            a = Argument(Q_i, obj.number(), part=obj.part())

            indices =[()]
            for m in a.ufl_shape:
                indices = [(k + (j,)) for k in indices for j in range(m)]

            if (i == self.idx[obj.number()]):
                args += [a[j] for j in indices]
            else:
                args += [Zero() for j in indices]
        return as_vector(args)

    def multi_index(self, obj):
        return obj

    expr = MultiFunction.reuse_if_untouched
