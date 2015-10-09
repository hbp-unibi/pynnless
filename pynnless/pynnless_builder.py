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
Contains the classes which allows to easily build PyNNLess networks without
having to manually fiddle arround with a dictionary of arrays of dictionaries.
"""

import pynnless_exceptions as exceptions
import pynnless_constants as const

class Population(dict):
    """
    Container for the data which is used to represent a neuron population. The
    Population class is a simple dictionary with some convenience methods and
    validations.
    """

    def __init__(self, data={}, count=1,
            _type=const.TYPE_IF_COND_EXP, params={}, record=[]):
        """
        Constructor of a neuron population instance.

        :param data: dictionary the data may be copied from.
        :param count: Number of neurons in the population
        :param _type: Type of the neuron
        :param record: Variables to be recorded
        :param params: Neuron population parameters
        """
        self["count"] = data["count"] if "count" in data else count
        self["type"] = data["type"] if "type" in data else _type
        self["params"] = data["params"] if "params" in data else params
        self["record"] = data["record"] if "record" in data else record

        self._canonicalize()
        self._validate()

    def _validate(self):
        """
        Internally used to ensure the entries have correct values.
        """
        if (self["count"] <= 0):
            raise exceptions.PyNNLessException("Invalid population size: "
                + str(self["count"]))
        if (not self["type"] in const.TYPES):
            raise exceptions.PyNNLessException("Invalid neuron type '"
                + str(self["type"]) + "' supported are " + str(const.TYPES))

    def _canonicalize(self):
        """
        Internal function, makes sure the "record" list is indeed a list, is
        sorted and contains no double entries.
        """
        if isinstance(self["record"], str):
            self["record"] = [self["record"]]
        else:
            self["record"] = list(self["record"])
        self["record"] = list(set(self["record"]))
        return self

    def record(self, signal):
        """
        Adds the given signal to the list of recorded signals.
        """
        self._canonicalize()
        self["record"].append(signal)
        return self._canonicalize()

    def record_spikes(self):
        """
        Adds the SIG_SPIKES signal to the list of recorded signals.
        """
        return self.record(const.SIG_SPIKES)


class SourcePopulation(Population):
    """
    Population of spike sources.
    """
    def __init__(self, data={}, count=1, spike_times=[], record=[]):
        Population.__init__(self, data, count, const.TYPE_SOURCE,
                {"spike_times": spike_times}, record)

    def _canonicalize(self):
        if not "spike_times" in self["params"]:
            self["params"]["spike_times"] = []

    def add_spike(self, t):
        self._canonicalize()
        self["params"]["spike_times"].append(t)

    def add_spikes(self, ts):
        self._canonicalize()
        self["params"]["spike_times"] = self["params"]["spike_times"] + ts

    def sort_spikes(self):
        self._canonicalize()
        self["params"]["spike_times"].sort()


class NeuronPopulation(Population):
    def record_ge(self):
        """
        Adds the SIG_GE signal to the list of recorded signals, triggers
        recording of the excitatory channel conductivity.
        """
        return self.record(const.SIG_GE)

    def record_gi(self):
        """
        Adds the SIG_GI signal to the list of recorded signals, triggers
        recording of the inhibitory channel conductivity.
        """
        return self.record(const.SIG_GI)

    def record_v(self):
        """
        Adds the SIG_V signal to the list of recorded signals, triggers
        recording of the neuron membrane potential.
        """
        return self.record(const.SIG_V)


class IfCondExpPopulation(NeuronPopulation):
    def __init__(self, data={}, count=1, params={}, record=[]):
        NeuronPopulation.__init__(self, data, count, const.TYPE_IF_COND_EXP,
                params, record)


class AdExPopulation(NeuronPopulation):
    def __init__(self, data={}, count=1, params={}, record=[]):
        NeuronPopulation.__init__(self, data, count, const.TYPE_AD_EX, params,
                record)


class Network(dict):
    """
    Represents a spiking neural network. This class merly is a dictionary
    containing a "populations" and an "connections" entry. 
    """

    def __init__(self, data={}, populations=[], connections=[]):
        """
        Constructor of the Network class, either copies the given data object or
        initializes the "populations" and "connections" with the given elements.

        :param data: another Network dictionary from which entries should be
        copied.
        :param populations: array of population descriptors.
        :param connections: array of connection descriptors.
        """
        self["populations"] = data["populations"] if\
                "populations" in data else populations
        self["connections"]  = data["connections"] if\
                "connections" in data else connections

    def add_population(self, data={}, count=1, _type=const.TYPE_IF_COND_EXP,
            params={}, record=[]):
        self["populations"].append(
                Population(data, count, _type, params, record))
        return self

    def add_populations(self, ps):
        self["populations"] = self["populations"] + ps
        return self

    def add_connection(self, src, dst, weight=0.1, delay=0.0):
        self["connections"].append((src, dst, weight, delay))
        return self

    def add_connections(self, cs):
        self["connections"] = self["connections"] + cs
        return self

