#!/usr/bin/env python
# -*- coding: utf-8 -*-

#   PyNNLess -- Yet Another PyNN Abstraction Layer
#   Copyright (C) 2015 Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Sequentially runs all examples, aborts if one example fails. This program exists
for two reasons: On the one hand it acts as kind of a integration test, on the
other hand it allows to directly run the PyNNLess repository on the HBP
Neuromorphic Platform.
"""

import sys
import os
import subprocess

# List all ".py" files in the examples folder and run them
examples = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        "examples")
for example in [f for f in os.listdir(examples) if f[-3:] == ".py"]:
    cmd = ["python", os.path.join(examples, example)] + sys.argv[1:]
    print("run.py: Executing " + cmd[1])
    if subprocess.call(cmd) != 0:
        print("run.py: Previous command exited with error, aborting.")
        sys.exit(1)
print("run.py: Done.")
