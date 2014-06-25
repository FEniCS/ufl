
from ufl.algorithms import load_forms

m = load_forms("mass.ufl")
s = load_forms("stiffness.ufl")

print("mass:")
print("\n".join(str(f) for f in m))
print("stiffness:")
print("\n".join(str(f) for f in s))
