
import logging
import ufl

# Let the logger emit everything, but print only warnings
l = ufl.get_logger()

# Let a file handler store all log data
hf = logging.FileHandler("testlog") # mode="a", e.g. append, as default, can be changed
l.addHandler(hf)

hl = ufl.log.ufl_logger.add_logfile()
l.addHandler(hl)

ufl.get_handler().setLevel(logging.DEBUG)
hf.setLevel(logging.ERROR)
hl.setLevel(logging.DEBUG)
l.setLevel(logging.DEBUG)

# Try getting some log messages
f = ufl.algorithms.load_forms("../../demo/Stiffness.ufl")
fd = f[0].form_data()

print "end of script"

