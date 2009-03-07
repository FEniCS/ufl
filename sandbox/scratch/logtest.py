
import logging
import ufl

# Let the logger emit everything, but print only warnings
l = ufl.get_logger()
l.setLevel(logging.DEBUG)
ufl.get_handler().setLevel(logging.WARNING)

# Let a file handler store all log data
h = logging.FileHandler("testlog") # mode="a", e.g. append, as default, can be changed
h.setLevel(logging.DEBUG)
l.addHandler(h)

# Try getting some log messages
f = ufl.algorithms.load_forms("../../demo/Stiffness.ufl")

print "end of script"

