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

# PyNN libraries
import pyNN
import pyNN.common
import pyNN.standardmodels.cells
import numpy as np

# Simulator loading and lookup
import importlib
import pkgutil

# Logging
import logging

# For IO-redirection
import os
import sys

# Own classes
import pynnless_builder as builder
import pynnless_constants as const
import pynnless_exceptions as exceptions

# Local logger
logger = logging.getLogger("PyNNLess")

# Temporary file descriptors to the original stdout/stderr
oldstdout = None
oldstderr = None

class PyNNLess:
    """
    The backend class is used as an abstraction to the actual PyNN backend,
    which may either be present in version 0.7 or 0.8. Furthermore, it
    constructs the network from a simple graph abstraction.
    """

    # List containing all supported simulators. Other simulators may also be
    # supported if they follow the PyNN specification, however, these simulators
    # were tested and will be returned by the "backends" method. The simulator
    # names are the normalized simulator names.
    SUPPORTED_SIMULATORS = {
        "nest", "ess", "nmpm1", "nmmc1"
    }

    # Used to map certain simulator names to a more canonical form. This
    # canonical form does not correspond to the name of the actual module
    # includes but the names that "feel more correct".
    NORMALIZED_SIMULATOR_NAMES = {
        "hardware.brainscales": "ess",
        "spiNNaker": "nmmc1",
        "pyhmf": "nmpm1",
    }

    # Maps certain simulator names to the correct PyNN module names. If multiple
    # module names are given, the first found module is used.
    SIMULATOR_IMPORT_MAP = {
        "ess": ["pyNN.hardware.brainscales"],
        "nmmc1": ["pyNN.spiNNaker"],
        "nmpm1": ["pyhmf"],
    }

    # List of simulators that need a call to "end" before the results are
    # retrieved
    PREMATURE_END_SIMULATORS = ["nmpm1"]

    # Map containing certain default setup parameters for the various
    # (normalized) backends. Setup parameters starting with $ are evaluated as
    # Python code.
    DEFAULT_SETUPS = {
        "nest": {
            # No default setup parameters needed
        },
        "ess": {
            "ess_params": {"perfectSynapseTrafo": True},
            "hardware": "$sim.hardwareSetup[\"one-hicann\"]",
            "useSystemSim": True,
        },
        "nmmc1": {
            "timestep": 1.0
        },
        "nmpm1": {
            "neuron_size": 2,
            "hicann": 276
        }
    }

    #
    # Private methods
    #

    # Redirects a given unix file handle to the given file name, returns a new
    # file descriptor pointing at the old fd
    @staticmethod
    def _redirect_fd_to_file(fd, filename):
        """
        Redirects the given Unix file descriptor to a file with the given name.
        """
        newFd = os.dup(fd)
        os.close(fd)
        os.open(filename, os.O_WRONLY | os.O_CREAT)
        return newFd

    # Redirects a given fd to another fd
    @staticmethod
    def _redirect_fd_to_fd(fd1, fd2):
        """
        Redirects the first (old) unix file descriptor to the second file
        descriptor, the file which was previously known as fd1 will now be known
        as fd2
        """
        os.close(fd2)
        os.dup2(fd1, fd2)
        os.close(fd1)

    # Prints the last few lines of a file to stdout
    @staticmethod
    def _tail(filename, title):
        """
        Prints the last 100 lines of the file with the given name.
        """
        with open(filename, 'r') as fd:
            lines = fd.readlines()
            if (len(lines) > 0):
                logger.info("[" + title + "]")
                if (len(lines) > 100):
                    logger.info("[...]")
                for line in lines[-100:]:
                    logger.info(line.strip('\n\r'))
                logger.info("[end]")

    @classmethod
    def _redirect_io(cls):
        """
        Redirects both stderr and stdout to some temporary files.
        """
        global oldstdout, oldstderr
        if (oldstdout == None):
            oldstdout = cls._redirect_fd_to_file(1, "stdout.tmp")
        if (oldstderr == None):
            oldstderr = cls._redirect_fd_to_file(2, "stderr.tmp")

    @classmethod
    def _unredirect_io(cls, tail=True):
        """
        Undos the redirection performed by _redirect_io and prints the last
        few lines of both files (if the "tail" parameter is set ot true).
        """
        global oldstdout, oldstderr

        if (oldstderr != None):
            sys.stderr.flush()
            cls._redirect_fd_to_fd(oldstderr, 2)
            if (tail):
                cls._tail("stderr.tmp", "stderr")
            os.remove("stderr.tmp")
            oldstderr = None

        if (oldstdout != None):
            sys.stdout.flush()
            cls._redirect_fd_to_fd(oldstdout, 1)
            if (tail):
                cls._tail("stdout.tmp", "stdout")
            os.remove("stdout.tmp")
            oldstdout = None

    @staticmethod
    def _check_version(version):
        """
        Internally used to check the current PyNN version. Sets the "version"
        variable to the correct API version and raises an exception if the PyNN
        version is not supported.

        :param version: the PyNN version string to be checked, should be the
        value of pyNN.__version__
        """
        if (version[0:3] == '0.7'):
            return 7
        elif (version[0:3] == '0.8'):
            return 8
        raise exceptions.PyNNLessVersionException("Unsupported PyNN version '"
            + pyNN.__version__ + "', supported are pyNN 0.7 and 0.8")

    @classmethod
    def _lookup_simulator(cls, simulator):
        """
        Internally used to generate the actual imports for the given simulator
        and to normalize the internally used simulator names.

        :param simulator: is the name of the simulator module as passed to the
        constructor.
        :return: a tuple containing the normalized simulator name and an array
        containing possibly corresponing modules (the first existing module
        should be used).
        """
        def unique(seq):
            seen = set()
            return [x for x in seq if not (x in seen or seen.add(x))]

        if (simulator.startswith("pyNN.")):
            simulator = simulator[5:]

        imports = ["pyNN." + simulator]
        normalized = simulator
        if (normalized in cls.NORMALIZED_SIMULATOR_NAMES):
            normalized = cls.NORMALIZED_SIMULATOR_NAMES[normalized]
            imports.append("pyNN." + normalized)
        if (normalized in cls.SIMULATOR_IMPORT_MAP):
            imports.extend(cls.SIMULATOR_IMPORT_MAP[normalized])

        return (normalized, unique(imports))

    @classmethod
    def _load_simulator(cls, simulator):
        """
        Internally used to load the simulator with the specified name. Raises
        an PyNNLessException if the specified simulator cannot be loaded.

        :param simulator: simulator name as passed to the constructor.
        :return: a tuple containing the simulator module and the final simulator
        name.
        """

        # Try to load the simulator module
        sim = None
        normalized, imports = PyNNLess._lookup_simulator(simulator)
        for i in xrange(len(imports)):
            try:
                cls._redirect_io()
                sim = importlib.import_module(imports[i])
                break
            except ImportError:
                if (i + 1 == len(imports)):
                    raise exceptions.PyNNLessSimulatorException(
                        "Could not find simulator, tried to load the " +
                        "following modules: " + str(imports) + ". Simulators " +
                        "which seem to bee supported on this machine are: " +
                        str(PyNNLess.simulators()))
            finally:
                # Set tail=False, do not print ANY clutter from the loader
                cls._unredirect_io(False)
        return (sim, normalized)

    @staticmethod
    def _eval_setup(setup, sim, simulator, version):
        """
        Passes dictionary entries starting with "$" through the Python "eval"
        function. A "\\$" at the beginning of the line is replaced with "$"
        without evaluation, a "\\\\" at the beginning of the line is replaced
        with "\\".

        :param setup: setup dictionary to which the evaluation should be
        applied.
        :param sim: simulation object that should be made available to the
        evaluated code.
        :param simulator: simulator name that should be made available to the
        evaluated code.
        :param version: PyNN version number that should be made available to the
        evaluated code.
        """
        res = {}
        for key, value in setup.items():
            if (isinstance(value, str)):
                if (len(value) >= 1 and value[0] == "$"):
                    value = eval(value[1:])
                elif (len(value) >= 2 and value[0:2] == "\\$"):
                    value = "$" + value[2:]
                elif (len(value) >= 2 and value[0:2] == "\\\\"):
                    value = "\\" + value[2:]
            res[key] = value
        return res

    @classmethod
    def _build_setup(cls, setup, sim, simulator, version):
        """
        Assembles the setup for the given simulator using the default setup and
        the setup given by the user.
        """

        # Evaluate the user setup
        user_setup = cls._eval_setup(setup, sim, simulator, version)

        # Evaluate the default setup for the simulator
        default_setup = {}
        if (simulator in cls.DEFAULT_SETUPS):
            default_setup = cls._eval_setup(cls.DEFAULT_SETUPS[simulator], sim,
                    simulator, version)

        # Merge the user setup into the default setup and return the result
        default_setup.update(user_setup)
        return default_setup

    @staticmethod
    def _setup_nmpm1(sim, setup):
        """
        Performs additional setup necessary for NMPM1. Creates a new Marocco
        (MApping ROuting Calibration and COnfiguration for HICANN Wafers)
        instance and sets it up. Marocco setup parameters were taken from
        https://github.com/electronicvisions/hbp_platform_demo/blob/master/nmpm1/run.py
        """
        from pymarocco import PyMarocco, Placement
        from pyhalbe.Coordinate import HICANNGlobal, Enum

        marocco = PyMarocco()
        marocco.placement.setDefaultNeuronSize(setup["neuron_size"])
        marocco.backend = PyMarocco.Hardware
        marocco.calib_backend = PyMarocco.XML
        marocco.calib_path = "/wang/data/calibration/wafer_0"
        marocco.bkg_gen_isi = 10000

        hicann = HICANNGlobal(Enum(setup["hicann"]))

        # Delete non-standard setup parameters
        del setup["neuron_size"]
        del setup["hicann"]

        # Pass the marocco object and the actual setup to the simulation setup
        # method
        sim.setup(marocco=marocco, **setup)

        # Return the marocco object and a list containing all HICANN
        return {
            "marocco": marocco,
            "hicann": hicann
        }

    def _setup_simulator(self, setup, sim, simulator, version):
        """
        Internally used to setup the simulator with the given setup parameters.

        :param setup: setup dictionary to be passed to the simulator setup.
        """

        # Assemble the setup
        setup = self._build_setup(setup, sim, simulator, version)

        # PyNN 0.7 compatibility hack: Update min_delay/timestep if only one
        # of the values is set
        if ((not "min_delay" in setup) and ("timestep" in setup)):
            setup["min_delay"] = setup["timestep"]
        if (("min_delay" in setup) and (not "timestep" in setup)):
            setup["timestep"] = setup["min_delay"]

        # PyNN 0.7 compatibility hack: Force certain parameters to be floating
        # point values (fixes "1" being passed as "timestep" instead of 1.0)
        for key in ["timestep", "min_delay", "max_delay"]:
            if (key in setup):
                setup[key] = float(setup[key])

        # Try to setup the simulator, do not output the clutter from the
        # simulators
        try:
            self._redirect_io()
            if (simulator == "nmpm1"):
                self.backend_data = self._setup_nmpm1(sim, setup)
            else:
                sim.setup(**setup)
        finally:
            self._unredirect_io()
        return setup

    def _build_population(self, population):
        """
        Used internally to creates a PyNN neuron population according to the
        parameters specified in the "population" object.

        :param population: is the population descriptor, containing the count of
        neurons within the population, the neuron type, the neuron parameters
        and whether the parameters should be recorded or not.
        """

        # Convert the given population dictionary into a managed Population
        # object
        population = builder.Population(population)

        # Fetch the neuron count
        count = population["count"]

        # Translate the neuron types to the PyNN neuron type
        type_name = population["type"]
        if (not hasattr(self.sim, type_name)):
            raise exceptions.PyNNLessException("Neuron type '" + type_name
                    + "' not supported by backend.")
        type_ = getattr(self.sim, type_name)
        is_source = type_name == const.TYPE_SOURCE

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
        for key, _ in params.items():
            if (key in population["params"]):
                # Convert integer parameters to floating point values, fixes bug
                # with PyNN 0.7.5 and NEST 2.2.2
                if isinstance(population["params"][key], int):
                    params[key] = float(population["params"][key])
                else:
                    params[key] = population["params"][key]

        # Issue warnings about ignored parameters
        for key, _ in population["params"].items():
            if (not key in params):
                logger.warning("Given parameter '" + key + "' does not " +
                    "exist for neuron type '" + type_name + "'. Value " +
                    "will be ignored!")

        # Fetch the parameter dimensions that should be recorded for this
        # population, make sure the elements in "record" are sorted
        record = population["record"]
        for signal in record:
            if (not signal in const.SIGNALS):
                logger.warning("Unknown signal \"" + signal
                    + "\". May be ignored by the backend.")

        # Create the output population, in case this is not a source population,
        # also force the neuron membrane potential to be initialized with the
        # neuron membrane potential.
        try:
            self._redirect_io()
            res = self.sim.Population(count, type_, params)

            if (self.version == 7):
                # Initialize membrane potential to v_rest, work around
                # "need more PhD-students"-exception on NMPM1 (where this condition
                # is fulfilled anyways)
                if ((not is_source) and (self.simulator != "nmpm1")):
                    res.initialize("v", params["v_rest"])

                # Setup recording
                if (const.SIG_SPIKES in record):
                    if (is_source and self.simulator == "spiNNaker"):
                        # Workaround for bug #122 in sPyNNaker
                        # https://github.com/SpiNNakerManchester/sPyNNaker/issues/122
                        logger.warning("spiNNaker backend does not support " +
                                 "recording input spikes, returning 'spike_times'.")
                        if ("spike_times" in params):
                            setattr(res, "__fake_spikes", params["spike_times"])
                        else:
                            setattr(res, "__fake_spikes", [[] for _ in xrange(count)])
                    else:
                        res.record()
                if (const.SIG_V in record):
                    res.record_v()
                if ((const.SIG_GE in record) or (const.SIG_GI in record)):
                    res.record_gsyn()
            elif (self.version == 8):
                # Initialize membrane potential to v_rest, work around
                # "need more PhD-students"-exception on NMPM1 (where this condition
                # is fulfilled anyways)
                if ((not is_source) and (self.simulator != "nmpm1")):
                        res.initialize(v=params["v_rest"])

                # Setup recording
                res.record(record)

            # Workaround bug in NMPM1, "size" attribute does not exist
            if (not hasattr(res, "size")):
                setattr(res, "size", count)

            # For NMPM1: register the population in the marocco instance
#            if (self.simulator == "nmpm1"):
#                self.backend_data["marocco"].placement.add(res,
#                      self.backend_data["hicann"])
        finally:
            self._unredirect_io(False)

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
            res[int(row[0])].append(np.float32(row[1]))

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
        ds = np.zeros((n, len(ts)), dtype=np.float32)
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
        if (self.version == 7):
            return self._convert_pyNN7_spikes(population.getSpikes(),
                population.size)
        elif (self.version == 8):
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
        if (self.simulator == "nmpm1"):
            logger.warning("nmpm1 does not support retrieving recorded " +
                    "signals for now")
            return {"data": np.zeros((population.size, 0), dtype=np.float32),
                    "time": np.zeros((0), dtype=np.float32)}
        if (self.version == 7):
            if (signal == const.SIG_V):
                return self._convert_pyNN7_signal(population.get_v(), 2,
                    population.size)
            elif (signal == const.SIG_GE):
                return self._convert_pyNN7_signal(population.get_gsyn(), 2,
                    population.size)
            elif (signal == const.SIG_GI):
                # Workaround in bug #124 in sPyNNaker, see
                # https://github.com/SpiNNakerManchester/sPyNNaker/issues/124
                if (self.simulator != "spiNNaker"):
                    return self._convert_pyNN7_signal(population.get_gsyn(), 3,
                        population.size)
        elif (self.version == 8):
            for array in population.get_data().segments[0].analogsignalarrays:
                if (array.name == signal):
                    return {
                        "data": np.asarray(array, dtype=np.float32).transpose(),
                        "time": np.asarray(array.times, dtype=np.float32)}
        return {"data": np.zeros((population.size, 0), dtype=np.float32),
                "time": np.zeros((0), dtype=np.float32)}

    @staticmethod
    def _get_default_timestep():
        """
        Returns the value of the "DEFAULT_TIMESTEP" attribute, which is stored
        in different places for multiple PyNN versions.
        """
        if hasattr(pyNN.common, "DEFAULT_TIMESTEP"):
            return pyNN.common.DEFAULT_TIMESTEP
        elif (hasattr(pyNN.common, "control")
                and hasattr(pyNN.common.control, "DEFAULT_TIMESTEP")):
            return pyNN.common.control.DEFAULT_TIMESTEP
        raise exceptions.PyNNLessException("DEFAULT_TIMESTEP not defined")

    #
    # Public interface
    #

    # Actual simulator instance, loaded by the "load" method.
    sim = None

    # Name of the simulator
    simulator = ""

    # Copy of the setup parameters
    setup = {}

    # Additional objects required by other backends
    backend_data = {}

    # Currently loaded pyNN version (as an integer, either 7 or 8 for 0.7 and
    # 0.8 respectively)
    version = 0

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

        self.version = self._check_version(pyNN.__version__)
        self.sim, self.simulator = self._load_simulator(simulator)
        self.setup = self._setup_simulator(setup, self.sim, self.simulator,
                self.version)

        logger.info("Loaded and successfully set up simulator \""
            + self.simulator + "\"")

    @classmethod
    def simulators(cls):
        """
        Returns a list of simulators that seem to be supported on this machine.
        """
        res = []
        for simulator in cls.SUPPORTED_SIMULATORS:
            _, imports = PyNNLess._lookup_simulator(simulator)
            for _import in imports:
                try:
                    loader = pkgutil.find_loader(_import)
                    if (isinstance(loader, pkgutil.ImpLoader)):
                        res.append(simulator)
                        break
                except ImportError:
                    pass
        return res

    @classmethod
    def normalized_simulator_name(cls, simulator):
        """
        Returns the normalized name for the given simulator
        """
        return cls._lookup_simulator(simulator)[0]

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

        # Make sure both the "populations" and "connections" arrays have been
        # supplied
        if (not "populations" in network):
            raise exceptions.PyNNLessException("\"populations\" key must be " +
                "present in network description")
        if (not "connections" in network):
            raise exceptions.PyNNLessException("\"connections\" key must be " +
                "present in network description")

        # Generate the neuron populations
        population_count = len(network["populations"])
        populations = [None for _ in xrange(population_count)]
        for i in xrange(population_count):
            populations[i] = self._build_population(network["populations"][i])

        # Fetch the simulation timestep, work around bugs #123 and #147 in
        # sPyNNaker.
        # Do not call get_time_step() on the analogue hardware systems as this
        # will result in an exception.
        timestep = self._get_default_timestep()
        if (hasattr(self.sim, "get_time_step") and not (self.simulator == "ess"
                or self.simulator == "nmpm1" or self.simulator == "nmmc1")):
            timestep = self.sim.get_time_step()
        elif ("timestep" in self.setup):
            timestep = self.setup["timestep"]

        # Build the connection matrices, and perform the actual connections
        connections = self._build_connections(network["connections"], timestep)

        try:
            self._redirect_io()

            # Perform the actual connections
            for pids, descrs in connections.items():
                self.sim.Projection(populations[pids[0]], populations[pids[1]],
                    self.sim.FromListConnector(descrs))

            # Run the simulation
            self.sim.run(time)

            # End the simulation to fetch the results on nmpm1
            if (self.simulator in self.PREMATURE_END_SIMULATORS):
                self.sim.end()

            # Gather the recorded data and store it in the result structure
            res = [{} for _ in xrange(population_count)]
            for i in xrange(population_count):
                if "record" in network["populations"][i]:
                    population = builder.Population(network["populations"][i])
                    for signal in population["record"]:
                        if (signal == const.SIG_SPIKES):
                            res[i][signal] = self._fetch_spikes(populations[i])
                        else:
                            data = self._fetch_signal(populations[i], signal)
                            res[i][signal] = data["data"]
                            res[i][signal + "_t"] = data["time"]

            # End the simulation if this has not been done yet
            if (not (self.simulator in self.PREMATURE_END_SIMULATORS)):
                self.sim.end()
        finally:
            self._unredirect_io()

        return res

