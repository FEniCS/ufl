from dolfin import *

def test():
    mesh1 = UnitIntervalMesh(10)
    mesh2 = UnitSquareMesh(10,10)
    mesh3 = UnitCubeMesh(10,10,10)
    for mesh in (mesh1, mesh2, mesh3):
        cell = mesh.ufl_cell()
        d = cell.d
        x = cell.x
        n = cell.n
        I = Identity(d)
        assert I.shape() == (d,d)
        M = I[i,j]*n[i]*x[j]*ds
        value = assemble(M, mesh=mesh)
        assert abs(value - d) < 1e-10
test()

