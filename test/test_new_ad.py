#!/usr/bin/env py.test

from ufl import *
from ufl.assertions import ufl_assert
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dag
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import MultiIndex, Index, FixedIndex, Terminal, Zero, Grad, Product, Sum, Indexed, IndexSum, ComponentTensor, ExprList, ExprMapping
from ufl.tensors import as_tensor, as_scalar, as_scalars
from ufl.algorithms import tree_format, renumber_indices

class GenericDerivativeRuleset(MultiFunction):
    def __init__(self, var_shape):
        MultiFunction.__init__(self)
        self._var_shape = var_shape


    # --- Error checking for missing handlers and unexpected types

    def expr(self, o, *ops):
        error("Missing differentiation handler for type {0}. Have you added a new type?".format(o._ufl_class_.__name__))

    def unexpected(self, o, *ops):
        error("Unexpected type {0} in AD rules.".format(o._ufl_class_.__name__))

    def override(self, o, *ops):
        error("Type {0} must be overridden in specialized AD rule set.".format(o._ufl_class_.__name__))

    def derivative(self, o, *ops):
        error("Unhandled derivative type {0}, nested differentiation has failed.".format(o._ufl_class_.__name__))

    def fixme(self, o, *ops):
        error("FIXME: Unimplemented differentiation handler for type {0}.".format(o._ufl_class_.__name__))


    # --- Helper functions for creating zeros with the right shapes

    def non_differentiable_terminal(self, o):
        "Labels and indices are not differentiable. It's convenient to return the non-differentiated object."
        return o

    def independent_terminal(self, o):
        "Return a zero with the right shape for terminals independent of differentiation variable."
        return Zero(o.ufl_shape + self._var_shape)

    def independent_operator(self, o):
        "Return a zero with the right shape and indices for operators independent of differentiation variable."
        return Zero(o.ufl_shape + self._var_shape, o.ufl_free_indices, o.ufl_index_dimensions)

    # --- All derivatives need to define grad and averaging

    grad = override
    cell_avg = override
    facet_avg = override

    # --- Default rules for terminals

    # Some types just don't have any derivative, this is just to make algorithm structure generic
    label = non_differentiable_terminal
    multi_index = non_differentiable_terminal

    # Literals are assumed independent of the differentiation variable by default
    constant_value = independent_terminal

    # Rules for form arguments must be specified in specialized rule set
    form_argument = override

    # Rules for geometric quantities must be specified in specialized rule set
    geometric_quantity = override

    # These types are currently assumed independent, but for non-affine domains
    # this no longer holds and we want to implement rules for them.
    facet_normal = independent_terminal
    spatial_coordinate = independent_terminal
    cell_coordinate = independent_terminal

    # Measures of cell entities, assuming independent although
    # this will not be true for all of these for non-affine domains
    cell_volume = independent_terminal
    circumradius = independent_terminal
    facet_area = independent_terminal
    #cell_surface_area = independent_terminal
    min_cell_edge_length = independent_terminal
    max_cell_edge_length = independent_terminal
    min_facet_edge_length = independent_terminal
    max_facet_edge_length = independent_terminal

    # Other stuff
    cell_orientation = independent_terminal
    quadrature_weigth = independent_terminal

    # These types are currently not expected to show up in AD pass.
    # To make some of these available to the end-user, they need to be implemented here.
    facet_coordinate = unexpected
    cell_origin = unexpected
    facet_origin = unexpected
    cell_facet_origin = unexpected
    jacobian = unexpected
    jacobian_determinant = unexpected
    jacobian_inverse = unexpected
    facet_jacobian = unexpected
    facet_jacobian_determinant = unexpected
    facet_jacobian_inverse = unexpected
    cell_facet_jacobian = unexpected
    cell_facet_jacobian_determinant = unexpected
    cell_facet_jacobian_inverse = unexpected
    cell_edge_vectors = unexpected
    facet_edge_vectors = unexpected
    cell_normal = unexpected # TODO: Expecting rename
    #cell_normals = unexpected
    #facet_tangents = unexpected
    #cell_tangents = unexpected
    #cell_midpoint = unexpected
    #facet_midpoint = unexpected


    # --- Default rules for operators

    def variable(self, o, df, l):
        return df

    # --- Indexing and component handling

    def indexed(self, o, Ap, ii):
        # Propagate zeros
        if isinstance(Ap, Zero):
            return self.independent_operator(o)

        # Untangle as_tensor(C[kk], jj)[ii] -> C[ll] to simplify resulting expression
        if isinstance(Ap, ComponentTensor):
            B, jj = Ap.ufl_operands
            if isinstance(B, Indexed):
                C, kk = B.ufl_operands
                Cind = list(kk)
                for i, j in zip(ii, jj):
                    Cind[kk.index(j)] = i
                return Indexed(C, MultiIndex(tuple(Cind)))

        # Otherwise a more generic approach
        r = Ap.rank() - len(ii)
        if r:
            kk = indices(r)
            op = Indexed(Ap, MultiIndex(ii.indices() + kk))
            op = as_tensor(op, kk)
        else:
            op = Indexed(Ap, ii)
        return op

    def list_tensor(self, o, *dops):
        return ListTensor(*dops)

    def component_tensor(self, o, Ap, ii):
        if isinstance(Ap, Zero):
            op = self.independent_operator(o)
        else:
            Ap, jj = as_scalar(Ap)
            op = as_tensor(Ap, ii.indices() + jj)
        return op

    # --- Algebra operators

    def index_sum(self, o, Ap, i):
        return IndexSum(Ap, i)

    def sum(self, o, da, db):
        return da + db

    def product(self, o, da, db):
        # Even though arguments to o are scalar, da and db may be tensor valued
        a, b = o.ufl_operands
        (da, db), ii = as_scalars(da, db)
        pa = Product(da, b)
        pb = Product(a, db)
        s = Sum(pa, pb)
        if ii:
            s = as_tensor(s, ii)
        return s

    def division(self, o, fp, gp):
        f, g = o.ufl_operands

        ufl_assert(is_ufl_scalar(f), "Not expecting nonscalar nominator")
        ufl_assert(is_true_ufl_scalar(g), "Not expecting nonscalar denominator")

        #do_df = 1/g
        #do_dg = -h/g
        #op = do_df*fp + do_df*gp
        #op = (fp - o*gp) / g

        # Get o and gp as scalars, multiply, then wrap as a tensor again
        so, oi = as_scalar(o)
        sgp, gi = as_scalar(gp)
        o_gp = so * sgp
        if oi or gi:
            o_gp = as_tensor(o_gp, oi + gi)
        op = (fp - o_gp) / g

        return op

    def power(self, o, fp, gp):
        f, g = o.ufl_operands

        ufl_assert(is_true_ufl_scalar(f), "Expecting scalar expression f in f**g.")
        ufl_assert(is_true_ufl_scalar(g), "Expecting scalar expression g in f**g.")

        # Derivation of the general case: o = f(x)**g(x)
        #do/df  = g * f**(g-1) = g / f * o
        #do/dg  = ln(f) * f**g = ln(f) * o
        #do/df * df + do/dg * dg = o * (g / f * df + ln(f) * dg)

        if isinstance(dg, Zero):
            # This probably produces better results for the common case of f**constant
            op = fp * g * f**(g-1)
        else:
            # Note: This produces expressions like (1/w)*w**5 instead of w**4
            op = o * (fp * g / f + gp * ln(f))

        return op

    def abs(self, o, df):
        f, = o.ufl_operands
        #return conditional(eq(f, 0), 0, Product(sign(f), df))
        return sign(f) * df

    # --- Mathfunctions

    def math_function(self, o, df):
        # FIXME: Introduce a UserOperator type instead of this hack and define user derivative() function properly
        if hasattr(o, 'derivative'):
            f, = o.ufl_operands
            return df * o.derivative()
        error("Unknown math function.")

    def sqrt(self, o, fp):
        return fp / (2*o)

    def exp(self, o, fp):
        return fp*o

    def ln(self, o, fp):
        f, = o.ufl_operands
        ufl_assert(not isinstance(f, Zero), "Division by zero.")
        return fp / f

    def cos(self, o, fp):
        f, = o.ufl_operands
        return -fp * sin(f)

    def sin(self, o, fp):
        f, = o.ufl_operands
        return fp * cos(f)

    def tan(self, o, fp):
        f, = o.ufl_operands
        return 2.0*fp / (cos(2.0*f) + 1.0)

    def cosh(self, o, fp):
        f, = o.ufl_operands
        return fp * sinh(f)

    def sinh(self, o, fp):
        f, = o.ufl_operands
        return fp * cosh(f)

    def tanh(self, o, fp):
        f, = o.ufl_operands
        def sech(y):
            return (2.0*cosh(y)) / (cosh(2.0*y) + 1.0)
        return fp * sech(f)**2

    def acos(self, o, fp):
        f, = o.ufl_operands
        return -fp / sqrt(1.0 - f**2)

    def asin(self, o, fp):
        f, = o.ufl_operands
        return fp / sqrt(1.0 - f**2)

    def atan(self, o, fp):
        f, = o.ufl_operands
        return fp / (1.0 + f**2)

    def atan_2(self, o, fp, gp):
        f, g = o.ufl_operands
        return (g*fp - f*gp) / (f**2 + g**2)

    def erf(self, o, fp):
        f, = o.ufl_operands
        return fp * (2.0 / sqrt(pi) * exp(-f**2))

    # --- Bessel functions

    def bessel_j(self, o, nup, fp):
        nu, f = o.ufl_operands
        if not (nup is None or isinstance(nup, Zero)):
            error("Differentiation of bessel function w.r.t. nu is not supported.")

        if isinstance(nu, Zero):
            op = -bessel_J(1, f)
        else:
            op = 0.5 * (bessel_J(nu-1, f) - bessel_J(nu+1, f))
        return op * fp

    def bessel_y(self, o, nup, fp):
        nu, f = o.ufl_operands
        if not (nup is None or isinstance(nup, Zero)):
            error("Differentiation of bessel function w.r.t. nu is not supported.")

        if isinstance(nu, Zero):
            op = -bessel_Y(1, f)
        else:
            op = 0.5 * (bessel_Y(nu-1, f) - bessel_Y(nu+1, f))
        return op * fp

    def bessel_i(self, o, nup, fp):
        nu, f = o.ufl_operands
        if not (nup is None or isinstance(nup, Zero)):
            error("Differentiation of bessel function w.r.t. nu is not supported.")

        if isinstance(nu, Zero):
            op = bessel_I(1, f)
        else:
            op = 0.5 * (bessel_I(nu-1, f) + bessel_I(nu+1, f))
        return op * fp

    def bessel_k(self, o, nup, fp):
        nu, f = o.ufl_operands
        if not (nup is None or isinstance(nup, Zero)):
            error("Differentiation of bessel function w.r.t. nu is not supported.")

        if isinstance(nu, Zero):
            op = -bessel_K(1, f)
        else:
            op = -0.5 * (bessel_K(nu-1, f) + bessel_K(nu+1, f))
        return op * fp

    # --- Restrictions

    def restricted(self, o, fp):
        # Restriction and differentiation commutes, at least for the derivatives we support.
        if isinstance(fp, ConstantValue):
            return fp # TODO: Necessary? Can't restriction simplify directly instead?
        else:
            return fp(o._side) # (f+-)' == (f')+-

    # --- Conditionals

    def binary_condition(self, o, dl, dr):
        # Should not be used anywhere...
        return None

    def not_condition(self, o, c):
        # Should not be used anywhere...
        return None

    def conditional(self, o, dc, dt, df):
        if isinstance(dt, Zero) and isinstance(df, Zero):
            # Assuming dt and df have the same indices here, which should be the case
            return dt
        else:
            # Placing t[1],f[1] outside here to avoid getting arguments inside conditionals
            c, t, f = o.ufl_operands
            dc = conditional(c, 1, 0)
            return dc*dt + (1.0 - dc)*df

    def max_value(self, o, df, dg):
        #d/dx max(f, g) =
        # f > g: df/dx
        # f < g: dg/dx
        f, g = o.ufl_operands
        return conditional(f > g, df, dg)

    def min_value(self, o, df, dg):
        #d/dx min(f, g) =
        # f < g: df/dx
        # else: dg/dx
        f, g = o.ufl_operands
        return conditional(f < g, df, dg)


class GradRuleset(GenericDerivativeRuleset):
    def __init__(self, geometric_dimension):
        GenericDerivativeRuleset.__init__(self, var_shape=(geometric_dimension,))
        self._Id = Identity(geometric_dimension)

    # --- Specialized rules for geometric quantities

    def geometric_quantity(self, o):
        "dg/dx = 0 if piecewise constant, otherwise Grad(g)"
        if o.is_cellwise_constant():
            return self.independent_terminal(o)
        else:
            # TODO: Which types does this involve? I don't think the form compilers will handle this.
            return Grad(o)

    def spatial_coordinate(self, o):
        "dx/dx = I"
        return self._Id

    def cell_coordinate(self, o):
        "dX/dx = inv(dx/dX) = inv(J) = K"
        return JacobianInverse(o.domain())

    # TODO: Add more geometry types here, with non-affine domains several should be non-zero.

    # --- Specialized rules for form arguments

    def coefficient(self, o):
        if o.is_cellwise_constant():
            return self.independent_terminal(o)
        return Grad(o)

    def argument(self, o):
        return Grad(o)

    def _argument(self, o): # TODO: Enable this after fixing issue#13, unless we move simplification to a separate stage?
        if o.is_cellwise_constant():
            # Collapse gradient of cellwise constant function to zero
            return AnnotatedZero(o.ufl_shape + self._var_shape, arguments=(o,)) # TODO: Missing this type
        else:
            return Grad(o)

    # --- Nesting of gradients

    def grad(self, o, *dummy_ops):
        "Represent grad(grad(f)) as Grad(Grad(f))."

        # Check that o is a "differential terminal"
        ufl_assert(isinstance(o.ufl_operands[0], (Grad, Terminal)),
                   "Expecting only grads applied to a terminal.")

        return Grad(o)

    def _grad(self, o, *dummy_ops):
        pass
        # TODO: Not sure how to detect that gradient of f is cellwise constant.
        #       Can we trust element degrees?
        #if o.is_cellwise_constant():
        #    return self.terminal(o)
        # TODO: Maybe we can ask "f.has_derivatives_of_order(n)" to check
        #       if we should make a zero here?
        # 1) n = count number of Grads, get f
        # 2) if not f.has_derivatives(n): return zero(...)

    cell_avg = GenericDerivativeRuleset.independent_operator
    facet_avg = GenericDerivativeRuleset.independent_operator


class VariableRuleset(GenericDerivativeRuleset):
    def __init__(self, var):
        GenericDerivativeRuleset.__init__(self, var_shape=var.ufl_shape)
        ufl_assert(not var.ufl_free_indices, "Differentiation variable cannot have free indices.")
        self._variable = var
        self._Id = self._make_identity(self._var_shape)

    def _make_identity(self, sh):
        "Create a higher order identity tensor to represent dv/dv."
        res = None
        if sh == ():
            # Scalar dv/dv is scalar
            return FloatValue(1.0)
        elif len(sh) == 1:
            # Vector v makes dv/dv the identity matrix
            return Identity(sh[0])
        else:
            # TODO: Add a type for this higher order identity?
            # II[i0,i1,i2,j0,j1,j2] = 1 if all((i0==j0, i1==j1, i2==j2)) else 0
            # Tensor v makes dv/dv some kind of higher rank identity tensor
            ind1 = ()
            ind2 = ()
            for d in sh:
                i, j = indices(2)
                dij = Identity(d)[i, j]
                if res is None:
                    res = dij
                else:
                    res *= dij
                ind1 += (i,)
                ind2 += (j,)
            fp = as_tensor(res, ind1 + ind2)
        return fp

    # Explicitly defining dg/dw == 0
    geometric_quantity = GenericDerivativeRuleset.independent_terminal

    # Explicitly defining da/dw == 0
    argument = GenericDerivativeRuleset.independent_terminal
    def _argument(self, o):
        return AnnotatedZero(o.ufl_shape + self._var_shape, arguments=(o,)) # TODO: Missing this type

    def coefficient(self, o):
        """df/dv = Id if v is f else 0.

        Note that if v = variable(f), df/dv is still 0,
        but if v == f, i.e. isinstance(v, Coefficient) == True,
        then df/dv == df/df = Id.
        """
        v = self._variable
        if isinstance(v, Coefficient) and o == v:
            # dv/dv = identity of rank 2*rank(v)
            return self._Id
        else:
            # df/v = 0
            return self.independent_terminal(o)

    def variable(self, o, df, l):
        v = self._variable
        if isinstance(v, Variable) and v.label() == l:
            # dv/dv = identity of rank 2*rank(v)
            return self._Id
        else:
            # df/v = df
            return df

    def grad(self, o, *dummy_ops):
        "Variable derivative of a gradient of a terminal must be 0."
        # Check that o is a "differential terminal"
        ufl_assert(isinstance(o.ufl_operands[0], (Grad, Terminal)),
                   "Expecting only grads applied to a terminal.")
        return self.independent_terminal(o)

    cell_avg = GenericDerivativeRuleset.independent_operator
    facet_avg = GenericDerivativeRuleset.independent_operator

class GateauxDerivativeRuleset(GenericDerivativeRuleset):
    """Apply AFD (Automatic Functional Differentiation) to expression.

    Implements rules for the Gateaux derivative D_w[v](...) defined as

        D_w[v](e) = d/dtau e(w+tau v)|tau=0

    """
    def __init__(self, coefficients, arguments, coefficient_derivatives):
        GenericDerivativeRuleset.__init__(self, var_shape=())

        # Type checking
        ufl_assert(isinstance(coefficients, ExprList), "Expecting a ExprList of coefficients.")
        ufl_assert(isinstance(arguments, ExprList), "Expecting a ExprList of arguments.")
        ufl_assert(isinstance(coefficient_derivatives, ExprMapping), "Expecting a coefficient-coefficient ExprMapping.")

        # The coefficient(s) to differentiate w.r.t. and the argument(s) s.t. D_w[v](e) = d/dtau e(w+tau v)|tau=0
        self._w = coefficients.ufl_operands
        self._v = arguments.ufl_operands
        self._w2v = {w: v for w, v in zip(self._w, self._v)}

        # Build more convenient dict {f: df/dw} for each coefficient f where df/dw is nonzero
        cd = coefficient_derivatives.ufl_operands
        self._cd = {cd[2*i]: cd[2*i+1] for i in range(len(cd)//2)}

    # Explicitly defining dg/dw == 0
    geometric_quantity = GenericDerivativeRuleset.independent_terminal

    # Explicitly defining da/dw == 0
    argument = GenericDerivativeRuleset.independent_terminal

    def coefficient(self, o):
        # Define dw/dw := d/ds [w + s v] = v

        # Return corresponding argument if we can find o among w
        do = self._w2v.get(o)
        if do is not None:
            return do

        # Look for o among coefficient derivatives
        dos = self._cd.get(o)
        if dos is None:
            # If o is not among coefficient derivatives, return do/dw=0
            do = Zero(o.ufl_shape)
            return do
        else:
            # Compute do/dw_j = do/dw_h : v.
            # Since we may actually have a tuple of oprimes and vs in a
            # 'mixed' space, sum over them all to get the complete inner
            # product. Using indices to define a non-compound inner product.

            # Example:
            # (f:g) -> (dfdu:v):g + f:(dgdu:v)
            # shape(dfdu) == shape(f) + shape(v)
            # shape(f) == shape(g) == shape(dfdu : v)

            # Make sure we have a tuple to match the self._v tuple
            if not isinstance(dos, tuple):
                dos = (dos,)
            ufl_assert(len(dos) == len(self._v),
                       "Got a tuple of arguments, expecting a matching tuple of coefficient derivatives.")
            dosum = Zero(o.ufl_shape)
            for do, v in zip(dos, self._v):
                so, oi = as_scalar(do)
                rv = len(v.ufl_shape)
                oi1 = oi[:-rv]
                oi2 = oi[-rv:]
                prod = so*v[oi2]
                if oi1:
                    dosum += as_tensor(prod, oi1)
                else:
                    dosum += prod
            return dosum

    def cell_avg(self, o, fp):
        # Cell average of a single function and differentiation commutes, D_f[v](cell_avg(f)) = cell_avg(v)
        return cell_avg(fp)

    def facet_avg(self, o, fp):
        # Facet average of a single function and differentiation commutes, D_f[v](facet_avg(f)) = facet_avg(v)
        return cell_avg(fp)

    def grad(self, g, *dummy_ops):
        # If we hit this type, it has already been propagated
        # to a coefficient (or grad of a coefficient), # FIXME: Assert this!
        # so we need to take the gradient of the variation or return zero.
        # Complications occur when dealing with derivatives w.r.t. single components...

        # Figure out how many gradients are around the inner terminal
        ngrads = 0
        o = g
        while isinstance(o, Grad):
            o, = o.ufl_operands
            ngrads += 1
        if not isinstance(o, FormArgument):
            error("Expecting gradient of a FormArgument, not %r" % (o,))

        def apply_grads(f):
            for i in range(ngrads):
                f = Grad(f)
            return f

        # Find o among all w without any indexing, which makes this easy
        for (w, v) in zip(self._w, self._v):
            if o == w and isinstance(v, FormArgument):
                # Case: d/dt [w + t v]
                return apply_grads(v)

        # If o is not among coefficient derivatives, return do/dw=0
        gprimesum = Zero(g.ufl_shape)

        def analyse_variation_argument(v):
            # Analyse variation argument
            if isinstance(v, FormArgument):
                # Case: d/dt [w[...] + t v]
                vval, vcomp = v, ()
            elif isinstance(v, Indexed):
                # Case: d/dt [w + t v[...]]
                # Case: d/dt [w[...] + t v[...]]
                vval, vcomp = v.ufl_operands
                vcomp = tuple(vcomp)
            else:
                error("Expecting argument or component of argument.")
            ufl_assert(all(isinstance(k, FixedIndex) for k in vcomp),
                       "Expecting only fixed indices in variation.")
            return vval, vcomp

        def compute_gprimeterm(ngrads, vval, vcomp, wshape, wcomp):
            # Apply gradients directly to argument vval,
            # and get the right indexed scalar component(s)
            kk = indices(ngrads)
            Dvkk = apply_grads(vval)[vcomp+kk]
            # Place scalar component(s) Dvkk into the right tensor positions
            if wshape:
                Ejj, jj = unit_indexed_tensor(wshape, wcomp)
            else:
                Ejj, jj = 1, ()
            gprimeterm = as_tensor(Ejj*Dvkk, jj+kk)
            return gprimeterm

        # Accumulate contributions from variations in different components
        for (w, v) in zip(self._w, self._v):

            # Analyse differentiation variable coefficient
            if isinstance(w, FormArgument):
                if not w == o: continue
                wshape = w.ufl_shape

                if isinstance(v, FormArgument):
                    # Case: d/dt [w + t v]
                    return apply_grads(v)

                elif isinstance(v, ListTensor):
                    # Case: d/dt [w + t <...,v,...>]
                    for wcomp, vsub in unwrap_list_tensor(v):
                        if not isinstance(vsub, Zero):
                            vval, vcomp = analyse_variation_argument(vsub)
                            gprimesum = gprimesum + compute_gprimeterm(ngrads, vval, vcomp, wshape, wcomp)

                else:
                    ufl_assert(wshape == (), "Expecting scalar coefficient in this branch.")
                    # Case: d/dt [w + t v[...]]
                    wval, wcomp = w, ()

                    vval, vcomp = analyse_variation_argument(v)
                    gprimesum = gprimesum + compute_gprimeterm(ngrads, vval, vcomp, wshape, wcomp)

            elif isinstance(w, Indexed): # This path is tested in unit tests, but not actually used?
                # Case: d/dt [w[...] + t v[...]]
                # Case: d/dt [w[...] + t v]
                wval, wcomp = w.ufl_operands
                if not wval == o:
                    continue
                assert isinstance(wval, FormArgument)
                ufl_assert(all(isinstance(k, FixedIndex) for k in wcomp),
                           "Expecting only fixed indices in differentiation variable.")
                wshape = wval.ufl_shape

                vval, vcomp = analyse_variation_argument(v)
                gprimesum = gprimesum + compute_gprimeterm(ngrads, vval, vcomp, wshape, wcomp)

            else:
                error("Expecting coefficient or component of coefficient.")

        # FIXME: Handle other coefficient derivatives: oprimes = self._cd.get(o)

        if 0:
            oprimes = self._cd.get(o)
            if oprimes is None:
                if self._cd:
                    # TODO: Make it possible to silence this message in particular?
                    #       It may be good to have for debugging...
                    warning("Assuming d{%s}/d{%s} = 0." % (o, self._w))
            else:
                # Make sure we have a tuple to match the self._v tuple
                if not isinstance(oprimes, tuple):
                    oprimes = (oprimes,)
                    ufl_assert(len(oprimes) == len(self._v), "Got a tuple of arguments, "+\
                                   "expecting a matching tuple of coefficient derivatives.")

                # Compute dg/dw_j = dg/dw_h : v.
                # Since we may actually have a tuple of oprimes and vs in a
                # 'mixed' space, sum over them all to get the complete inner
                # product. Using indices to define a non-compound inner product.
                for (oprime, v) in zip(oprimes, self._v):
                    error("FIXME: Figure out how to do this with ngrads")
                    so, oi = as_scalar(oprime)
                    rv = len(v.ufl_shape)
                    oi1 = oi[:-rv]
                    oi2 = oi[-rv:]
                    prod = so*v[oi2]
                    if oi1:
                        gprimesum += as_tensor(prod, oi1)
                    else:
                        gprimesum += prod

        return gprimesum


# FIXME: Write UNIT tests for all terminal derivatives!
# FIXME: Add operator derivatives to generic ruleset!
# FIXME: Write UNIT tests for operator derivatives!
# FIXME: Implement nested AD on top!

# TODO: Add more rulesets?
# - DivRuleset
# - CurlRuleset
# - ReferenceGradRuleset
# - ReferenceDivRuleset


class DerivativeRuleDispatcher(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)

    def terminal(self, o):
        return o

    def derivative(self, o, *ops):
        error("Missing derivative handler for {0}.".format(type(o).__name__))

    expr = MultiFunction.reuse_if_untouched

    def grad(self, o, f):
        rules = GradRuleset(o.ufl_shape[-1])
        return map_expr_dag(rules, f)

    def variable_derivative(self, o, f, dummy_v):
        rules = VariableRuleset(o.ufl_operands[1])
        return map_expr_dag(rules, f)

    def coefficient_derivative(self, o, f, dummy_w, dummy_v, dummy_cd):
        dummy, w, v, cd = o.ufl_operands
        rules = GateauxDerivativeRuleset(w, v, cd)
        return map_expr_dag(rules, f)

    def indexed(self, o, Ap, ii):
        # Reuse if untouched
        if Ap is o.ufl_operands[0]:
            return o

        # Untangle as_tensor(C[kk], jj)[ii] -> C[ll] to simplify resulting expression
        if isinstance(Ap, ComponentTensor):
            B, jj = Ap.ufl_operands
            if isinstance(B, Indexed):
                C, kk = B.ufl_operands
                kk = list(kk)
                Cind = list(kk)
                for i, j in zip(ii, jj):
                    Cind[kk.index(j)] = i
                return Indexed(C, MultiIndex(tuple(Cind)))

        # Otherwise a more generic approach
        r = Ap.rank() - len(ii)
        if r:
            kk = indices(r)
            op = Indexed(Ap, MultiIndex(ii.indices() + kk))
            op = as_tensor(op, kk)
        else:
            op = Indexed(Ap, ii)
        return op


#def _nested_apply_derivatives(expr):
#    rules = DerivativeRuleDispatcher()
#    return map_expr_dag(rules, expr)

#def _apply_derivatives(expression):
#    return map_integrands(nested_apply_derivatives, expression)

def apply_derivatives(expression):
    rules = DerivativeRuleDispatcher()
    return map_integrand_dags(rules, expression)


def test_apply_derivatives_doesnt_change_expression_without_derivatives():
    cell = triangle
    d = cell.geometric_dimension()
    V0 = FiniteElement("DG", cell, 0)
    V1 = FiniteElement("Lagrange", cell, 1)

    # Literals
    z = zero((3, 2))
    one = as_ufl(1)
    two = as_ufl(2.0)
    I = Identity(d)
    literals = [z, one, two, I]

    # Geometry
    x = SpatialCoordinate(cell)
    n = FacetNormal(cell)
    volume = CellVolume(cell)
    geometry = [x, n, volume]

    # Arguments
    v0 = TestFunction(V0)
    v1 = TestFunction(V1)
    arguments = [v0, v1]

    # Coefficients
    f0 = Coefficient(V0)
    f1 = Coefficient(V1)
    coefficients = [f0, f1]

    # Expressions
    e0 = f0 + f1
    e1 = v0 * (f1/3 - f0**2)
    e2 = exp(sin(cos(tan(ln(x[0])))))
    expressions = [e0, e1, e2]

    # Check that all are unchanged
    for expr in literals + geometry + arguments + coefficients + expressions:
        # Note the use of "is" here instead of ==, this property
        # is important for efficiency and memory usage
        assert apply_derivatives(expr) is expr


def test_literal_derivatives_are_zero():
    cell = triangle
    d = cell.geometric_dimension()

    # Literals
    one = as_ufl(1)
    two = as_ufl(2.0)
    I = Identity(d)
    E = PermutationSymbol(d)
    literals = [one, two, I]

    # Generic ruleset handles literals directly:
    for l in literals:
        for sh in [(), (d,), (d,d+1)]:
            assert GenericDerivativeRuleset(sh)(l) == zero(l.ufl_shape + sh)

    # Variables
    v0 = variable(one)
    v1 = variable(zero((d,)))
    v2 = variable(I)
    variables = [v0, v1, v2]

    # Test literals via apply_derivatives and variable ruleset:
    for l in literals:
        for v in variables:
            assert apply_derivatives(diff(l, v)) == zero(l.ufl_shape + v.ufl_shape)

    V0 = FiniteElement("DG", cell, 0)
    V1 = FiniteElement("Lagrange", cell, 1)
    u0 = Coefficient(V0)
    u1 = Coefficient(V1)
    v0 = TestFunction(V0)
    v1 = TestFunction(V1)
    args = [(u0, v0), (u1, v1)]

    # Test literals via apply_derivatives and variable ruleset:
    for l in literals:
        for u, v in args:
            assert apply_derivatives(derivative(l, u, v)) == zero(l.ufl_shape + v.ufl_shape)

    # Test grad ruleset directly since grad(literal) is invalid:
    assert GradRuleset(d)(one) == zero((d,))
    assert GradRuleset(d)(one) == zero((d,))


def test_grad_ruleset():
    cell = triangle
    d = cell.geometric_dimension()

    V0 = FiniteElement("DG", cell, 0)
    V1 = FiniteElement("Lagrange", cell, 1)
    V2 = FiniteElement("Lagrange", cell, 2)
    W0 = VectorElement("DG", cell, 0)
    W1 = VectorElement("Lagrange", cell, 1)
    W2 = VectorElement("Lagrange", cell, 2)

    # Literals
    one = as_ufl(1)
    two = as_ufl(2.0)
    I = Identity(d)
    literals = [one, two, I]

    # Geometry
    x = SpatialCoordinate(cell)
    n = FacetNormal(cell)
    volume = CellVolume(cell)
    geometry = [x, n, volume]

    # Arguments
    u0 = TestFunction(V0)
    u1 = TestFunction(V1)
    arguments = [u0, u1]

    # Coefficients
    r = Constant(cell)
    vr = VectorConstant(cell)
    f0 = Coefficient(V0)
    f1 = Coefficient(V1)
    f2 = Coefficient(V2)
    vf0 = Coefficient(W0)
    vf1 = Coefficient(W1)
    vf2 = Coefficient(W2)
    coefficients = [f0, f1, vf0, vf1]

    # Expressions
    e0 = f0 + f1
    e1 = u0 * (f1/3 - f0**2)
    e2 = exp(sin(cos(tan(ln(x[0])))))
    expressions = [e0, e1, e2]

    # Variables
    v0 = variable(one)
    v1 = variable(f1)
    v2 = variable(f0*f1)
    variables = [v0, v1, v2]

    rules = GradRuleset(d)

    # Literals
    assert rules(one) == zero((d,))
    assert rules(two) == zero((d,))
    assert rules(I) == zero((d,d,d))

    # Assumed piecewise constant geometry
    for g in [n, volume]:
        assert rules(g) == zero(g.ufl_shape + (d,))

    # Non-constant geometry
    assert rules(x) == I

    # Arguments
    for u in arguments:
        assert rules(u) == grad(u)

    # Piecewise constant coefficients (Constant)
    assert rules(r) == zero((d,))
    assert rules(vr) == zero((d,d))
    assert rules(grad(r)) == zero((d,d))
    assert rules(grad(vr)) == zero((d,d,d))

    # Piecewise constant coefficients (DG0)
    assert rules(f0) == zero((d,))
    assert rules(vf0) == zero((d,d))
    assert rules(grad(f0)) == zero((d,d))
    assert rules(grad(vf0)) == zero((d,d,d))

    # Piecewise linear coefficients
    assert rules(f1) == grad(f1)
    assert rules(vf1) == grad(vf1)
    #assert rules(grad(f1)) == zero((d,d)) # TODO: Use degree to make this work
    #assert rules(grad(vf1)) == zero((d,d,d))

    # Piecewise quadratic coefficients
    assert rules(grad(f2)) == grad(grad(f2))
    assert rules(grad(vf2)) == grad(grad(vf2))

    # Indexed coefficients
    assert renumber_indices(apply_derivatives(grad(vf2[0]))) == renumber_indices(grad(vf2)[0,:])
    assert renumber_indices(apply_derivatives(grad(vf2[1])[0])) == renumber_indices(grad(vf2)[1,0])

    # Grad of expressions
    assert apply_derivatives(grad(2*f0)) == zero((d,))
    assert renumber_indices(apply_derivatives(grad(2*f1))) == renumber_indices(2*grad(f1))


def test_variable_ruleset():
    pass


def test_gateaux_ruleset():
    pass
