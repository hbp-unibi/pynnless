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
Simple usage example of PyNNLess: Creates a network containing a single LIF
neuron and a spike source array. Records the output spikes of the LIF neuron.
"""

import sys
import common.setup # Common example code (checks command line parameters)
import common.params # Parameters for the models which work with all systems
from pynnless import PyNNLess as pl

# Create a new pl instance with the given backend
backend = sys.argv[1]
sim = pl(backend)

# Create and run network with two populations: One population consisting of a
# spike source array and another population consisting of a single neuron. Note
# that the constants used here are simply strings -- the whole network structure
# could thus be stored in a simple JSON file
print("Simulating network...")
synfire_len = 100
res = sim.run({
        "populations": [
            {
                "count": 1,
                "type": pl.TYPE_SOURCE,
                "params": {
                    "spike_times": [10.0],
                }
            },
            {
                # Single LIF neuron with default parameters
                # Note that "record" may also be an array with multiple signals
                "count": synfire_len,
                "type": pl.TYPE_IF_COND_EXP,
                "record": pl.SIG_SPIKES,
                "params": common.params.IF_cond_exp # Use more compatible params
            }
        ],
        "connections": [
            ((0, 0), (1, 0), 0.03, 0.0),
            ((1, synfire_len - 1), (1, 0), 0.03, 0.0)
        ] + [((1, i - 1), (1, i), 0.03, 0.0) for i in xrange(1, synfire_len)]
    }, 1000.0)
print("Done!")

# Print the output spikes (population 1, spikes, neuron 0)
print("Spike Times: " + str(res[1]["spikes"][0]))

# Write the spike times for each neuron to disk (each row contains the spike
# times of a single corresponding neuron).
print("Writing spike times to " + common.setup.outfile)
f = open(common.setup.outfile, "w")
for i in xrange(len(res[1]["spikes"])):
    first = True
    for j in xrange(len(res[1]["spikes"][i])):
        if (not first):
            f.write("\t")
        f.write(str(res[1]["spikes"][i][j]))
        first = False
    f.write("\n")
f.close()

