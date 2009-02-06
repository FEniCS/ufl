#!/bin/bash
FORMAT=pdf
#APP=circo # not good
#APP=twopi # not good

# select hierarchial structure:
APP=dot

# select circular structure:
#APP=fdp

python printclasstree.py | $APP -T$FORMAT -oclassgraph.$FORMAT
evince classgraph.$FORMAT &
