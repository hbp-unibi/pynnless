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
neuron and a spike source array. Records the output spikes and membrane
potential of the LIF neuron.
"""

import sys
import common.setup # Common example code (checks command line parameters)
import common.params # Parameters for the models which work with all systems
import pynnless as pynl

# Create a new pl instance with the given backend
backend = sys.argv[1]
sim = pynl.PyNNLess(backend)

# Same as in "single_neuron.py", but record the voltage
print("Simulating network...")
res = sim.run(pynl.Network()
        .add_source(spike_times=[0, 1000, 2000])
        .add_population(
            pynl.IfCondExpPopulation(params=common.params.IF_cond_exp)
                .record_spikes()
                .record_v()
        )
        .add_connection((0, 0), (1, 0), weight=0.004))
print("Done!")

# Write the membrane potential for each neuron to disk (first column is time)
print("Writing membrane potential to " + common.setup.outfile)
f = open(common.setup.outfile, "w")

# Iterate over all sample times
for i in xrange(len(res[1]["v_t"])):
    # Write the current sample time
    f.write(str(res[1]["v_t"][i]))

    # Iterate over all neurons in the population and write the value for this
    # sample time
    for j in xrange(len(res[1]["v"])):
        f.write("\t" + str(res[1]["v"][j][i]))
    f.write("\n")
f.close()

