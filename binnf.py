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
Reads a binary neural network specification from the specified file and executes
it. Writes the recorded data to the specified binnf file.
"""

import sys
import os
import subprocess
import logging

import pynnless as pynl
import pynnless.pynnless_binnf as binnf

# Check the script parameters
if len(sys.argv) != 4 and len(sys.argv) != 2:
    print("Usage: ./binnf.py <SIMULATOR> [<IN> <OUT>]")
    print("Where <IN> and <OUT> may be \"-\" to read from/write to "
            + " stdin/stdout (default)")
    sys.exit(1)

# Fetch the simulator name from the command line parameters
simulator = sys.argv[1]

# Fetch the input/output file
input_file = (sys.stdin if (len(sys.argv) < 3
        or sys.argv[2] == "-") else open(sys.argv[2], 'rb'))
output_file = (sys.stdout if (len(sys.argv) < 4
        or sys.argv[3] == "-") else open(sys.argv[3], 'wb'))

# Read the input data
network = binnf.read_network(input_file)
print network

# Open the simulator and run the network
#res = pynl.PyNNLess(simulator).run(network)

# Serialise the results
# TODO


