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
import pynnless as pynl

# Create a new pl instance with the given backend
backend = sys.argv[1]
sim = pynl.PyNNLess(backend)

# Create and run network with two populations: One population consisting of a
# spike source array and another population consisting of a single neuron. Note
# that the network structure built here can be stored in a JSON file, all
# objects are dictionaries. You are not required to use the Network and
# Population helper classes
print("Simulating network...")
res = sim.run(pynl.Network()
        .add_population(
            pynl.SourcePopulation(spike_times=[20.0, 24.0])
        )
        .add_population(
            pynl.IfCondExpPopulation(params=common.params.IF_cond_exp)
                .record_spikes()
        )
        .add_connection((0, 0), (1, 0), weight=0.015), # weight in µS
        100.0)
print("Done!")

# Print the output spikes (population 1, spikes, neuron 0)
print("Spike Times: " + str(res[1]["spikes"][0]))

