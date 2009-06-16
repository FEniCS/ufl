
def construct_element(name, domain, degree, value_rank):
    from ufl import FiniteElement
    element = FiniteElement(name, domain, degree)
    assert value_rank == len(element.value_shape())

def construct_vector_element(name, domain, degree, value_rank):
    from ufl import VectorElement
    element = VectorElement(name, domain, degree)
    assert value_rank+1 == len(element.value_shape())

def construct_tensor_element(name, domain, degree, value_rank):
    from ufl import TensorElement
    element = TensorElement(name, domain, degree)
    assert value_rank+2 == len(element.value_shape())

def test_element_construction():
    "Iterate over all registered elements and try to construct instances."
    from ufl.elementlist import ufl_elements
    for k in sorted(ufl_elements.keys()):
        (family, short_name, value_rank, degree_range, domains) = ufl_elements[k]
        if degree_range:
            a, b = degree_range
            if b is None:
                b = a + 2
        for name in (family, short_name):
            for domain in domains:
                for degree in range(a, b):
                    yield construct_element, name, domain, degree, value_rank
                    yield construct_vector_element, name, domain, degree, value_rank
                    yield construct_tensor_element, name, domain, degree, value_rank


