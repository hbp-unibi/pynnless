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

import pynnless.pynnless_binnf as binnf

# Fetch a logger
logger = logging.getLogger("binnf")

# Check the script parameters
if len(sys.argv) != 4:
    print("Usage: ./binnf.py <SIMULATOR> <IN> <OUT>")
    print("Where <IN> and <OUT> may be \"-\" to read from/write to "
            + " stdin/stdout")
    sys.exit(1)
simulator = sys.argv[1]
input_file = sys.stdin if sys.argv[2] == "-" else open(sys.argv[2], 'rb')
output_file = sys.stdout if sys.argv[3] == "-" else open(sys.argv[3], 'wb')

# Read the input file
has_data = False;
logger.info("Reading input data...")

import time
start = time.time()
while True:
    try:
        # Deserialise a single input block
        name, header, matrix = binnf.deseralise(input_file)
        has_data = True
    except:
        # If no data block has been read from the input file, raise the
        # exception. Otherwise most likely the end of the file has been reached.
        # In this case, abort.
        if not has_data:
            raise
        else:
            break
end = time.time()
print(end - start)

