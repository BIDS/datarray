#!/usr/bin/env python
"""Simple release script for datarray.

Ensure that you've built the docs and pushed those first (after veryfing them
manually).
"""
from __future__ import print_function

import os
from subprocess import call

sh = lambda s: call(s, shell=True)

cwd = os.getcwd()
if not os.path.isfile('setup.py'):
    os.chdir('..')
    if not os.path.isfile('setup.py'):
        print("This script must be run from top-level datarray or tools dir.")
        sys.exit(1)


sh('./setup.py register')
sh('./setup.py sdist --formats=gztar,zip upload')
