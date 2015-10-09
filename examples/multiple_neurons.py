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
Simple network consisting of 100 disconnected neurons and a spike source array
for each.
"""

import sys
import common.setup # Common example code (checks command line parameters)
import common.params # Parameters for the models which work with all systems
import common.utils # Output functions
import pynnless as pynl

# Create a new pl instance with the given backend
backend = sys.argv[1]
sim = pynl.PyNNLess(backend)

# Create and run network with two populations: One population consisting of a
# spike source arrays and another population consisting of neurons.
print("Simulating network...")
count = 1
res = sim.run(pynl.Network()
        .add_population(
            pynl.SourcePopulation(
                    count=count,
                    spike_times=[100.0 * i for i in xrange(1, 9)])
        )
        .add_population(
            pynl.IfCondExpPopulation(
                    count=count,
                    params=common.params.IF_cond_exp)
                .record_spikes()
        )
        .add_connections([((0, i), (1, i), 0.3, 0.0) for i in xrange(count)]),
        1000.0)
print("Done!")

# Write the spike times for each neuron to disk
print("Writing spike times to " + common.setup.outfile)
common.utils.write_spike_times(common.setup.outfile, res[1]["spikes"])
