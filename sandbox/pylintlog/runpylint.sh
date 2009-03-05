#!/bin/bash

rm -rf base
rm -rf algorithms
mkdirhier base
mkdirhier algorithms

# Check base system
cd base
pylint --ignore=algorithms -e --files-output=y ufl
# Remove incorrect ufl.log errors
sed -i s/E.\*No.\*ufl.log.\*// *.txt

cd ..

# Check algorithms
cd algorithms
pylint -e --files-output=y ufl.algorithms
# Remove incorrect ufl.log errors
sed -i s/E.\*No.\*ufl.log.\*// *.txt

