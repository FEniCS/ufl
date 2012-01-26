#!/usr/bin/env python

"""
This file contains snippets from the FEniCS book,
and allows us to test that these can still run
with future versions of UFL. Please don't change
these and please do keep UFL compatible with these
snippets as long as possible.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
from ufltestcase import UflTestCase, main

from ufl import *
from ufl.algorithms import *

class BookTestCase(UflTestCase):

    def test_uflcode_269(self):
        # Finite element spaces
        cell = tetrahedron
        element = VectorElement("Lagrange", cell, 1)

        # Form arguments
        phi0 = TestFunction(element)
        phi1 = TrialFunction(element)
        u = Coefficient(element)
        c1 = Constant(cell)
        c2 = Constant(cell)

        # Deformation gradient Fij = dXi/dxj
        I = Identity(cell.d)
        F = I + grad(u)

        # Right Cauchy-Green strain tensor C with invariants
        C = variable(F.T*F)
        I_C = tr(C)
        II_C = (I_C**2 - tr(C*C))/2

        # Mooney-Rivlin constitutive law
        W = c1*(I_C-3) + c2*(II_C-3)

        # Second Piola-Kirchoff stress tensor
        S = 2*diff(W, C)

        # Weak forms
        L = inner(F*S, grad(phi0))*dx
        a = derivative(L, u, phi1)

    def test_uflcode_316(self):
        shapestring = 'triangle'    
        cell = Cell(shapestring)

    def test_uflcode_323(self):
        cell = tetrahedron

    def test_uflcode_356(self):
        cell = tetrahedron

        P = FiniteElement("Lagrange", cell, 1)
        V = VectorElement("Lagrange", cell, 2)
        T = TensorElement("DG", cell, 0, symmetry=True)

        TH = V*P
        ME = MixedElement(T, V, P)

    def test_uflcode_400(self):
        V = FiniteElement("CG", triangle, 1)
        f = Coefficient(V)
        g = Coefficient(V)
        h = Coefficient(V)
        w = Coefficient(V)
        v = TestFunction(V)
        u = TrialFunction(V)
        # ...
        a = w*dot(grad(u), grad(v))*dx
        L = f*v*dx + g**2*v*ds(0) + h*v*ds(1)

    def test_uflcode_469(self):
        V = FiniteElement("CG", triangle, 1)
        f = Coefficient(V)
        g = Coefficient(V)
        h = Coefficient(V)
        v = TestFunction(V)
        # ...
        dx02 = dx(0, { "integration_order": 2 })
        dx14 = dx(1, { "integration_order": 4 })
        dx12 = dx(1, { "integration_order": 2 })
        L = f*v*dx02 + g*v*dx14 + h*v*dx12

    def test_uflcode_552(self):
        element = FiniteElement("CG", triangle, 1)
        # ...
        phi = Argument(element)
        v = TestFunction(element)
        u = TrialFunction(element)

    def test_uflcode_563(self):
        cell = triangle
        element = FiniteElement("CG", cell, 1)
        # ...
        w = Coefficient(element)
        c = Constant(cell)
        v = VectorConstant(cell)
        M = TensorConstant(cell)

    def test_uflcode_574(self):
        V0 = FiniteElement("CG", triangle, 1)
        V1 = V0
        # ...
        V = V0*V1
        u = Coefficient(V)
        u0, u1 = split(u)

    def test_uflcode_582(self):
        V0 = FiniteElement("CG", triangle, 1)
        V1 = V0
        # ...
        V = V0*V1
        u = Coefficient(V)
        u0, u1 = split(u)
        # ...
        v0, v1 = TestFunctions(V)
        u0, u1 = TrialFunctions(V)
        f0, f1 = Coefficients(V)

    def test_uflcode_644(self):
        V = VectorElement("CG", triangle, 1)
        u = Coefficient(V)
        v = Coefficient(V)
        # ...
        A = outer(u, v)
        Aij = A[i, j]

    def test_uflcode_651(self):
        V = VectorElement("CG", triangle, 1)
        u = Coefficient(V)
        v = Coefficient(V)
        # ...
        Aij = v[j]*u[i]
        A = as_tensor(Aij, (i, j))

    def test_uflcode_671(self):
        i = Index()
        j, k, l = indices(3)

    def test_uflcode_684(self):
        V = VectorElement("CG", triangle, 1)
        v = Coefficient(V)
        # ...
        th = pi/2
        A = as_matrix([[ cos(th), -sin(th)],
                       [ sin(th),  cos(th)]])
        u = A*v

    def test_uflcode_824(self):
        V = VectorElement("CG", triangle, 1)
        f = Coefficient(V)
        # ...
        df = Dx(f, i)
        df = f.dx(i)

    def test_uflcode_886(self):
        cell = triangle
        # ...
        g = sin(cell.x[0])
        v = variable(g)
        f = exp(v**2)
        h = diff(f, v)
        # ...
        #print v
        #print h

    def test_python_894(self):
        # We don't have to keep the string output compatible, so no test here.
        pass
        #>>> print v
        #var0(sin((x)[0]))
        #>>> print h
        #d/d[var0(sin((x)[0]))] (exp((var0(sin((x)[0]))) ** 2))

    def test_uflcode_930(self):
        condition = lt(1, 0)
        true_value = 1
        false_value = 0
        # ...
        f = conditional(condition, true_value, false_value)

    def test_uflcode_1003(self):
        # Not testable, but this is tested below anyway
        "a = derivative(L, w, u)"
        pass

    def test_uflcode_1026(self):
        element = FiniteElement("CG", triangle, 1)
        # ...
        v = TestFunction(element)
        u = TrialFunction(element)
        w = Coefficient(element)
        f = 0.5*w**2*dx
        F = derivative(f, w, v)
        J = derivative(F, w, u)

    def test_uflcode_1050(self):
        Vx = VectorElement("Lagrange", triangle, 1)
        Vy = FiniteElement("Lagrange", triangle, 1)
        u = Coefficient(Vx*Vy)
        x, y = split(u)
        f = inner(grad(x), grad(x))*dx + y*dot(x,x)*dx
        F = derivative(f, u)
        J = derivative(F, u)

    def test_uflcode_1085(self):
        cell = triangle
        # ...
        V = VectorElement("Lagrange", cell, 1)
        T = TensorElement("Lagrange", cell, 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        M = Coefficient(T)
        a = M[i,j]*u[k].dx(j)*v[k].dx(i)*dx
        astar = adjoint(a)

    def test_uflcode_1120(self):
        cell = triangle
        # ...
        V = FiniteElement("Lagrange", cell, 1)
        v = TestFunction(V)
        f = Coefficient(V)
        g = Coefficient(V)
        L = f**2 / (2*g)*v*dx
        L2 = replace(L, { f: g, g: 3})
        L3 = g**2 / 6*v*dx

    def test_uflcode_1157(self):
        cell = triangle
        # ...
        V = FiniteElement("Lagrange", cell, 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Coefficient(V)
        pde = u*v*dx - f*v*dx
        a, L = system(pde)

    def test_uflcode_1190(self):
        cell = triangle
        element = FiniteElement("Lagrange", cell, 1)
        V = element
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Coefficient(V)
        c = variable(Coefficient(V))
        pde = c*u*v*dx - c*f*v*dx
        a, L = system(pde)
        # ...
        u = Coefficient(element)
        sL = diff(L, c) - action(diff(a, c), u)

    def test_uflcode_1195(self):
        cell = triangle
        element = FiniteElement("Lagrange", cell, 1)
        V = element
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Coefficient(V)
        c = variable(Coefficient(V))
        pde = c*u*v*dx - c*f*v*dx
        a, L = system(pde)
        u = Coefficient(element)
        # ...
        sL = sensitivity_rhs(a, u, L, c)

    def test_uflcode_1365(self):
        e = 0
        v = variable(e)
        f = sin(v)
        g = diff(f, v)

    def test_python_1426(self):
        # Covered by the below test
        pass
        #from ufl.algorithms import Graph
        #G = Graph(expression)
        #V, E = G

    def test_python_1446(self):
        cell = triangle
        V = FiniteElement("Lagrange", cell, 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        c = Constant(cell)
        f = Coefficient(V)
        e = c*f**2*u*v

        from ufl.algorithms import Graph, partition
        G = Graph(e)
        V, E, = G

        if 0:
            print "str(e) = %s\n" % str(e)
            print "\n".join("V[%d] = %s" % (i, v) for (i, v) in enumerate(V)), "\n"
            print "\n".join("E[%d] = %s" % (i, e) for (i, e) in enumerate(E)), "\n"

    def test_python_1512(self):
        cell = triangle
        V = FiniteElement("Lagrange", cell, 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        c = Constant(cell)
        f = Coefficient(V)
        e = c*f**2*u*v

        from ufl.algorithms import Graph, partition
        G = Graph(e)
        V, E, = G
        # ...
        Vin = G.Vin()
        Vout = G.Vout()

    def test_python_1557(self):
        cell = triangle
        V = FiniteElement("Lagrange", cell, 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        c = Constant(cell)
        f = Coefficient(V)
        e = c*f**2*u*v

        from ufl.algorithms import Graph, partition
        G = Graph(e)
        V, E, = G
        # ...
        partitions, keys = partition(G)
        for deps in sorted(partitions.keys()):
            P = partitions[deps]
            #print "The following depends on", tuple(deps)
            for i in sorted(P):
                #print "V[%d] = %s" % (i, V[i])
                # ...
                v = V[i]

    def test_python_1843(self):
        def apply_ad(e, ad_routine):
            if isinstance(e, Terminal):
                return e
            ops = [apply_ad(o, ad_routine) for o in e.operands()]
            e = e.reconstruct(*ops)
            if isinstance(e, Derivative):
                e = ad_routine(e)
            return e

    def test_uflcode_1901(self):
        cell = triangle
        element = FiniteElement("Lagrange", cell, 1)
        # ...
        v = Argument(element)
        w = Coefficient(element)

    def test_python_1942(self):
        def walk(expression, pre_action, post_action):
            pre_action(expression)
            for o in expression.operands():
                walk(o)
            post_action(expression)

    def test_python_1955(self):
        def post_traversal(root):
            for o in root.operands():
                yield post_traversal(o)
            yield root

    def test_python_1963(self):
        def post_action(e):
            #print str(e)
            pass
        cell = triangle
        V = FiniteElement("Lagrange", cell, 1)
        u = TrialFunction(V)
        v = TestFunction(V)
        c = Constant(cell)
        f = Coefficient(V)
        expression = c*f**2*u*v
        # ...
        for e in post_traversal(expression):
            post_action(e)

    def test_python_1990(self):
        from ufl.classes import IntValue, Sum
        expression = as_ufl(3)
        def int_operation(x):
            return 7
        # ...
        if isinstance(expression, IntValue):
            result = int_operation(expression)
        elif isinstance(expression, Sum):
            result = sum_operation(expression)
        # etc.
        # ...
        self.assertTrue(result == 7)

    def test_python_2024(self):
        class ExampleFunction(MultiFunction):
            def __init__(self):
                MultiFunction.__init__(self)

            def terminal(self, expression):
                return "Got a Terminal subtype %s." % type(expression)

            def operator(self, expression):
                return "Got an Operator subtype %s." % type(expression)

            def argument(self, expression):
                return "Got an Argument."

            def sum(self, expression):
                return "Got a Sum."

        m = ExampleFunction()

        cell = triangle
        element = FiniteElement("Lagrange", cell, 1)
        x = cell.x
        if 0:
            print m(Argument(element))
            print m(x)
            print m(x[0] + x[1])
            print m(x[0] * x[1])

    def test_python_2066(self):
        def apply(e, multifunction):
            ops = [apply(o, multifunction) for o in e.operands()]
            return multifunction(e, *ops)

    def test_python_2087(self):
        class Replacer(Transformer):
            def __init__(self, mapping):
                Transformer.__init__(self)
                self.mapping = mapping

            def operator(self, e, *ops):
                return e.reconstruct(*ops)

            def terminal(self, e):
                return self.mapping.get(e, e)

        f = Constant(triangle)
        r = Replacer({f: f**2})
        g = r.visit(2*f)

    def test_python_2189(self):
        V = FiniteElement("Lagrange", triangle, 1)
        u = TestFunction(V)
        v = TrialFunction(V)
        f = Coefficient(V)

        # Note no *dx! This is an expression, not a form.
        a = dot(grad(f*u), grad(v))

        ac = expand_compounds(a)
        ad = expand_derivatives(ac)
        ai = expand_indices(ad)

        af = tree_format(a)
        acf = tree_format(ac)
        adf = "\n", tree_format(ad)
        aif = tree_format(ai)

        if 0:
            print "\na: ", str(a),  "\n", tree_format(a)
            print "\nac:", str(ac), "\n", tree_format(ac)
            print "\nad:", str(ad), "\n", tree_format(ad)
            print "\nai:", str(ai), "\n", tree_format(ai)

    def test_python_2328(self):
        cell = triangle
        x = cell.x
        e = x[0] + x[1]
        #print e((0.5, 0.7)) # prints 1.2
        # ...
        self.assertEqual( e((0.5, 0.7)), 1.2 )

    def test_python_2338(self):
        cell = triangle
        x = cell.x
        # ...
        c = Constant(cell)
        e = c*(x[0] + x[1])
        #print e((0.5, 0.7), { c: 10 }) # prints 12.0
        # ...
        self.assertEqual( e((0.5, 0.7), { c: 10 }), 12.0 )

    def test_python_2349(self):
        element = VectorElement("Lagrange", triangle, 1)
        c = Constant(triangle)
        f = Coefficient(element)
        e = c*(f[0] + f[1])
        def fh(x):
            return (x[0], x[1])
        #print e((0.5, 0.7), { c: 10, f: fh }) # prints 12.0
        # ...
        self.assertEqual( e((0.5, 0.7), { c: 10, f: fh }), 12.0 )

    def test_python_2364(self):
        element = FiniteElement("Lagrange", triangle, 1)
        g = Coefficient(element)
        e = g**2 + g.dx(0)**2 + g.dx(1)**2
        def gh(x, der=()):
            if der == ():   return x[0]*x[1]
            if der == (0,): return x[1]
            if der == (1,): return x[0]
        #print e((2, 3), { g: gh }) # prints 49
        # ...
        self.assertEqual( e((2, 3), { g: gh }), 49 )

    def test_python_2462(self):
        cell = triangle
        element = FiniteElement("Lagrange", cell, 1)
        V = element
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Coefficient(V)
        c = variable(Coefficient(V))
        pde = c*u*v*dx - c*f*v*dx
        a, L = system(pde)
        u = Coefficient(element)
        myform = a
        # ...
        #print repr(preprocess(myform).preprocessed_form)
        # ...
        r = repr(preprocess(myform).preprocessed_form)

# Don't touch these lines, they allow you to run this file directly
if __name__ == "__main__":
    main()

