
from ufl import *

class MockMesh:
    def __init__(self, ufl_id):
        self._ufl_id = ufl_id
    def ufl_id(self):
        return self._ufl_id
    def ufl_domain(self):
        return Domain(triangle, label="MockMesh_id_%d"%self.ufl_id(), data=self)
    def ufl_measure(self, domain_type="dx", domain_id="everywhere", metadata=None, domain_data=None):
        return Measure(domain_type, domain_id=domain_id, metadata=metadata, domain=self, domain_data=domain_data)

class MockMeshFunction:
    "Mock class for the pydolfin compatibility hack for domain data with [] syntax."
    def __init__(self, ufl_id, mesh):
        self._mesh = mesh
        self._ufl_id = ufl_id
    def ufl_id(self):
        return self._ufl_id
    def mesh(self):
        return self._mesh
    def ufl_measure(self, domain_type=None, domain_id="everywhere", metadata=None):
        return Measure(domain_type, domain_id=domain_id, metadata=metadata,
                       domain=self.mesh(), domain_data=self)
