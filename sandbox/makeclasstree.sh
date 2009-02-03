#!/bin/bash
FORMAT=pdf
APP=twopi
APP=circo
APP=dot
python printclasstree.py | $APP -T$FORMAT -oclassgraph.$FORMAT
evince classgraph.$FORMAT &
