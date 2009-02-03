#!/bin/bash
FORMAT=pdf
python printclasstree.py | dot -T$FORMAT -oclassgraph.$FORMAT
evince classgraph.$FORMAT &
