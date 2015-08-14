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
Contains the backend code responsible for mapping the current experiment setup
to a PyNN 0.7 or 0.8 simulation. Introduces yet another abstraction layer to
allow the description of a network independent of the actual PyNN version.
"""

import pyNN
import pyNN.common
import pyNN.standardmodels.cells
import numpy as np
import logging
import importlib

# Constants for the supported neuron types
TYPE_SOURCE = "SpikeSourceArray"
TYPE_IF_COND_EXP = "IF_cond_exp"
TYPE_AD_EX = "EIF_cond_exp_isfa_ista"
TYPES = [TYPE_SOURCE, TYPE_IF_COND_EXP, TYPE_AD_EX]

# Constants for the quantities that can be recorded
SIG_SPIKES = "spikes"
SIG_V = "v"
SIG_GE = "gsyn_exc"
SIG_GI = "gsyn_inh"

class PyNNLessException(Exception):
    """
    Exception type used in the PyNNLess module.
    """
    pass

class PyNNLess:
    """
    The backend class is used as an abstraction to the actual PyNN backend,
    which may either be present in version 0.7 or 0.8. Furthermore, it
    constructs the network from a simple graph abstraction.
    """

    # Actual simulator instance, loaded by the "load" method.
    sim = None

    # Name of the simulator
    simulator = ""

    # Copy of the setup parameters
    setup = {}

    # Currently loaded pyNN version (as an integer, either 7 or 8 for 0.7 and
    # 0.8 respectively)
    version = 0

    def _check_version():
        """
        Internally used to check the current PyNN version. Sets the "version"
        variable to the correct API version and raises an exception if the PyNN
        version is not supported.
        """
        if (pyNN.__version__[0:3] == '0.7'):
            self.version = 7
        elif (pyNN.__version__[0:3] == '0.8'):
            self.version = 8
        else:
            raise PyNNLessException("Unsupported PyNN version '"
                + pyNN.__version__ + "', supported are pyNN 0.7 and 0.8")

    def _load_simulator(simulator):
        """
        Internally used to load the simulator with the specified name. Raises
        an PyNNLessException if the specified simulator cannot be loaded.

        :param simulator: simulator name as passed to the constructor.
        """

        # Remap some simulator names passed from the neuromorphic compute
        # platform to the correct simulator name
        if (simulator == "ess"):
            simulator = "hardware.brainscales"

        # Try to load the simulator module, special handling for nmpm1
        self.simulator = simulator
        try:
            if (simulator == "nmpm1"):
                self.sim = importlib.import_module("pyhmf")
            else:
                self.sim = importlib.import_module("pyNN." + simulator)
        except ImportError:
            raise PyNNLessException(
                "Could not find simulator backend " + simulator)

    def _setup_simulator(setup):
        """
        Internally used to setup the simulator with the given setup parameters.

        :param setup: setup dictionary to be passed to the simulator setup.
        """
        # PyNN 0.7 compatibility hack: Update min_delay/timestep if only one
        # of the values is set
        if ((not "min_delay" in setup) and ("timestep" in setup)):
            setup["min_delay"] = setup["timestep"]
        if (("min_delay" in setup) and (not "timestep" in setup)):
            setup["timestep"] = setup["min_delay"]

        # PyNN 0.7 compatibility hack: Force certain parameters to be floating
        # point values
        for key in ["timestep", "min_delay", "max_delay"]:
            if (key in setup):
                setup[key] = float(setup[key])

        # Try to setup the simulator
        self.setup = setup
        self.sim.setup(**setup)

    def __init__(self, simulator, setup = {}):
        """
        Tries to load the PyNN simulator with the given name. Throws an
        exception if the simulator could not be found or no compatible PyNN
        version is loaded. Calls the "setup" method with the given simulator
        setup.

        :param simulator: name of the PyNN backend that should be used. Has to
        be the python module name stripped from the leading "pyNN.".
        Additionally handles some of the (wrong) simulator names passed by the
        HBP Neuromorphic Compute Platform and adds support for NMPM1.
        :param setup: structure containing additional setup parameters to be
        passed to the "setup" method.
        """

        self._check_version()
        self._load_simulator(simulator)
        self._setup_simulator(setup)

    def _build_population(self, population):
        """
        Used internally to creates a PyNN neuron population according to the
        parameters specified in the "population" object.

        :param population: is the population descriptor, containing the count of
        neurons within the population, the neuron type, the neuron parameters
        and whether the parameters should be recorded or not.
        """
        # Read the neuron count, default to one
        count = 1
        if ("count" in population):
            count = int(population["count"])
        if (count <= 0):
            raise PyNNLessException("Invalid population size: " + count)

        # Translate the neuron types to the PyNN neuron type
        type_name = None
        type_ = None
        is_source = False
        if (not "type" in population):
            raise PyNNLessException("Type key not present in network description")
        elif (population["type"] in TYPES):
            type_name = population["type"]
            if (not hasattr(self.sim, type_name)):
                raise PyNNLessException("Neuron type " + type_name
                        + " not supported by backend.")
            type_ = getattr(self.sim, type_name)
            is_source = type_name == TYPE_SOURCE
        else:
            raise PyNNLessException("Invalid neuron type " + type_name +
                " supported are " + str(TYPES))

        # Fetch the default parameters for this neuron type and merge them with
        # parameters given for this population. Due to a bug in sPyNNaker, we
        # need to use the default parameters provided by
        # pyNN.standardmodels.cells
        # See: https://github.com/SpiNNakerManchester/sPyNNaker/issues/120
        #
        # Note: The dict() is important, otherwise we seem to get a reference
        # instead of a copy.
        params = dict(getattr(pyNN.standardmodels.cells,
                type_name).default_parameters)
        if ("params" in population):
            for key, _ in params.items():
                if (key in population["params"]):
                    params[key] = population["params"][key]

        # Fetch the parameter dimensions that should be recorded for this
        # population, make sure the elements in "record" are sorted
        record = []
        if ("record" in population):
            record = population["record"]
            record.sort()

        # Create the output population, in case this is not a source population,
        # also force the neuron membrane potential to be initialized with the
        # neuron membrane potential.
        res = None
        if (self.is_pyNN7()):
            # Work around bug with setting spike_times for SpikeSourceArray in
            # ESS
            if (is_source and self.simulator == "hardware.brainscales"):
                res = self.sim.Population(count, type_)
                res.tset("spike_times", params["spike_times"])
            else:
                res = self.sim.Population(count, type_, params)
            if (not is_source):
                res.initialize("v", params["v_rest"])
            if (SIG_SPIKES in record):
                if (is_source and self.simulator == "spiNNaker"):
                    # Workaround for bug #122 in sPyNNaker
                    # https://github.com/SpiNNakerManchester/sPyNNaker/issues/122
                    logging.warning("spiNNaker backend does not support " +
                             "recording input spikes, returning 'spike_times'.")
                    if ("spike_times" in params):
                        setattr(res, "__fake_spikes", params["spike_times"])
                    else:
                        setattr(res, "__fake_spikes", [[] for _ in xrange(count)])
                else:
                    res.record()
            if (SIG_V in record):
                res.record_v()
            if ((SIG_GE in record) or (SIG_GI in record)):
                res.record_gsyn()
        elif (self.is_pyNN8()):
            res = self.sim.Population(count, type_, params)
            if (not is_source):
                res.initialize(v=params["v_rest"])
            res.record(record)

        return res

    @staticmethod
    def _build_connections(connections, min_delay=0):
        """
        Gets an array of [[pid_src, nid_src], [pid_tar, nid_tar], weight, delay]
        tuples an builds a dictionary of all (pid_src, pid_tar) mappings.
        """
        res = {}
        for connection in connections:
            src, tar, weight, delay = connection
            pids = (src[0], tar[0])
            descrs = (src[1], tar[1], weight, max(min_delay, delay))
            if (pids in res):
                res[pids].append(descrs)
            else:
                res[pids] = [descrs]
        return res

    @staticmethod
    def _convert_pyNN7_spikes(spikes, n):
        """
        Converts a pyNN7 spike train, list of (nid, time)-tuples, into a list
        of lists containing the spike times for each neuron individually.
        """

        # Create one result list for each neuron
        res = [[] for _ in xrange(n)]
        for row in spikes:
            res[int(row[0])].append(row[1])

        # Make sure the resulting lists are sorted by time
        for i in xrange(n):
            res[i].sort()
        return res

    @staticmethod
    def _convert_pyNN8_spikes(spikes):
        """
        Converts a pyNN8 spike train (some custom datastructure), into a list
        of lists containing the spike times for each neuron individually.
        """
        return [
            [np.float32(spikes[i][j]) for j in xrange(len(spikes[i]))]
            for i in xrange(len(spikes))]

    @staticmethod
    def _convert_pyNN7_signal(data, idx, n):
        """
        Converts a pyNN7 data array, list of (nid, time, d1, ..., dN)-tuples
        into a matrix containing the data for each neuron and timestep.
        Unfortunately this mapping step is non-trivial, as we'd like uniformly
        sampled output.

        TODO: Implement interpolation for not sampled values in case this
        actually occurs.
        """
        # Create a list containing all timepoints and a mapping from time to
        # index
        ts = np.asarray(sorted(set([a[1] for a in data])), dtype=np.float32)
        tsidx = dict((ts[i], i) for i in xrange(len(ts)))

        # Create one result list for each neuron, containing exactly ts entries
        ds = np.zeros((n, len(ts)), dtype=np.float32);
        ds.fill(np.nan)
        for row in data:
            ds[int(row[0]), tsidx[np.float32(row[1])]] = row[idx]

        return {"data": ds, "time": ts}

    def _fetch_spikes(self, population):
        """
        Fetches the recorded spikes from a neuron population and performs all
        necessary data structure conversions for the PyNN versions.

        :param population: reference at a PyNN population object from which the
        spikes should be obtained.
        """
        if (hasattr(population, "__fake_spikes")):
            spikes = getattr(population, "__fake_spikes")
            return [spikes for _ in xrange(population.size)]
        if (self.is_pyNN7()):
            return self._convert_pyNN7_spikes(population.getSpikes(),
                population.size)
        elif (self.is_pyNN8()):
            return self._convert_pyNN8_spikes(
                population.get_data().segments[0].spiketrains)
        return []

    def _fetch_signal(self, population, signal):
        """
        Converts an analog signal recorded by PyNN 0.7 or 0.8 to a common
        format with a data matrix containing the values for all neurons in the
        population and an independent timescale.

        :param population: reference at a PyNN population object from which the
        spikes should be obtained.
        :param signal: name of the signal that should be returned.
        """
        if (self.is_pyNN7()):
            if (signal == SIG_V):
                return self._convert_pyNN7_signal(population.get_v(), 2,
                    population.size)
            elif (signal == SIG_GE):
                return self._convert_pyNN7_signal(population.get_gsyn(), 2,
                    population.size)
            elif (signal == SIG_GI):
                # Workaround in bug #124 in sPyNNaker, see
                # https://github.com/SpiNNakerManchester/sPyNNaker/issues/124
                if (self.simulator != "spiNNaker"):
                    return self._convert_pyNN7_signal(population.get_gsyn(), 3,
                        population.size)
        elif (self.is_pyNN8()):
            for array in population.get_data().segments[0].analogsignalarrays:
                if (array.name == signal):
                    return {
                        "data": np.asarray(array, dtype=np.float32).transpose(),
                        "time": np.asarray(array.times, dtype=np.float32)}
        return {"data": np.zeros((population.size, 0), dtype=np.float32),
                "time": np.zeros((0), dtype=np.float32)}

    def run(self, network, time = 1000):
        """
        Builds and runs the network described in the "network" structure.

        :param network: Dictionary with two entries: "populations" and
        "connections", where the first introduces the individual neuron
        populations and their parameters and the latter is an adjacency list
        containing the connection weights and delays between neurons.
        :return: the recorded signals for each population, signal type and
        neuron
        """
        # Generate the neuron populations
        population_count = len(network["populations"]);
        populations = [None for _ in xrange(population_count)]
        for i in xrange(population_count):
            populations[i] = self._build_population(network["populations"][i])

        # Fetch the simulation timestep, work around bug #123 in sPyNNaker
        # See https://github.com/SpiNNakerManchester/sPyNNaker/issues/123
        timestep = pyNN.common.DEFAULT_TIMESTEP
        if (hasattr(self.sim, "get_time_step")):
            timestep = self.sim.get_time_step()
        elif ("timestep" in self.setup):
            timestep = self.setup["timestep"]

        # Build the connection matrices, and perform the actual connections
        connections = self._build_connections(network["connections"], timestep)

        # Work around deprecated nest "ConvergentConnect" method
        for pids, descrs in connections.items():
            self.sim.Projection(populations[pids[0]], populations[pids[1]],
                self.sim.FromListConnector(descrs))

        # Run the simulation
        self.sim.run(time)

        # Gather the recorded data and store it in the result structure
        res = [{} for _ in xrange(population_count)]
        for i in xrange(population_count):
            if "record" in network["populations"][i]:
                for signal in network["populations"][i]["record"]:
                    if (signal == SIG_SPIKES):
                        res[i][signal] = self._fetch_spikes(populations[i])
                    else:
                        data = self._fetch_signal(populations[i], signal);
                        res[i][signal] = data["data"]
                        res[i][signal + "_t"] = data["time"]

        # Cleanup after the simulation ran
        self.sim.end()

        return res;

