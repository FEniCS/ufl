#!/bin/bash
sed -i s/ufl.output/ufl.log/ *.py algorithms/*.py
sed -i s/ufl_info/info/ *.py algorithms/*.py
sed -i s/ufl_warning/warning/ *.py algorithms/*.py
sed -i s/ufl_error/error/ *.py algorithms/*.py
sed -i s/ufl_debug/debug/ *.py algorithms/*.py

