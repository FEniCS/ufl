#!/bin/bash

# Change pylint behaviour
# Errors only:
FLAGS=-e
# Defaults:
FLAGS=

# Delete old dirs
rm -rf base
rm -rf algorithms
mkdirhier base
mkdirhier algorithms

# Check base system
cd base
pylint $FLAGS --files-output=y --ignore=algorithms ufl
# Remove incorrect ufl.log errors
sed -i s/E.\*No.\*ufl.log.\*// *.txt
cd ..

# Check algorithms
cd algorithms
pylint $FLAGS --files-output=y ufl.algorithms
# Remove incorrect ufl.log errors
sed -i s/E.\*No.\*ufl.log.\*// *.txt
cd ..

# Print sizes
echo base/ lines:
cat base/*.txt | uniq | wc -l
echo algorithms/ lines:
cat algorithms/*.txt | uniq | wc -l
