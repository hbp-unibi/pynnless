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
Example of how to run a network specification stored in a JSON file with
PyNNLess.
"""

import sys, os
import json
import common.setup # Common example code (checks command line parameters)
import common.params # Parameters for the models which work with all systems
import pynnless as pynl

# Create a new pl instance with the given backend
backend = sys.argv[1]
sim = pynl.PyNNLess(backend)

# Run the network defined in the "json_network.json" file in the data directory
print("Simulating network...")
res = sim.run(json.load(open(os.path.join(common.setup.datapath,
        'json_network.json'))))
print("Done!")

# Serialize the output as json
print("Writing result " + common.setup.outfile)
json.dump(res, open(common.setup.outfile, 'w'), indent=4)
