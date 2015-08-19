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
import importlib
import pkgutil
import logging

# Local logger
logger = logging.getLogger("PyNNLess")

class PyNNLessException(Exception):
    """
    Exception type used in the PyNNLess module.
    """
    pass

class PyNNLessVersionException(Exception):
    """
    Indicates an incompatible PyNN version.
    """
    pass

class PyNNLessSimulatorException(Exception):
    """
    Thrown when the given simulator is not found.
    """
    pass

class PyNNLess:
    """
    The backend class is used as an abstraction to the actual PyNN backend,
    which may either be present in version 0.7 or 0.8. Furthermore, it
    constructs the network from a simple graph abstraction.
    """

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
    SIGNALS = [SIG_SPIKES, SIG_V, SIG_GE, SIG_GI]

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
        }
    }

    #
    # Private methods
    #

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
        raise PyNNLessVersionException("Unsupported PyNN version '"
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

    @staticmethod
    def _load_simulator(simulator):
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
                sim = importlib.import_module(imports[i])
                break
            except ImportError:
                if (i + 1 == len(imports)):
                    raise PyNNLessSimulatorException(
                        "Could not find simulator, tried to load the " +
                        "following modules: " + str(imports) + ". Simulators " +
                        "which seem to bee supported on this machine are: " +
                        str(PyNNLess.simulators()))
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
        Performs additional setup necessary for NMPM1.
        """
        import pylogging
        from pymarocco import PyMarocco
        from pyhalbe.Coordinate import SynapseDriverOnHICANN, HICANNGlobal, X,\
                Y, Enum, NeuronOnHICANN
        import Coordinate as C
        import pyhalbe
        import pyredman

        pylogging.set_loglevel(pylogging.get("Default"),
                pylogging.LogLevel.INFO)
        pylogging.set_loglevel(pylogging.get("marocco"),
                pylogging.LogLevel.DEBUG)
        pylogging.set_loglevel(pylogging.get("sthal.HICANNConfigurator.Time"),
                pylogging.LogLevel.DEBUG)

        h = pyredman.Hicann()

        def initBackend(fname):
            lib = pyredman.loadLibrary(fname)
            backend = pyredman.loadBackend(lib)
            if not backend:
                raise Exception('unable to load %s' % fname)
            return backend

        neuron_size = 4

        marocco = PyMarocco()
        marocco.placement.setDefaultNeuronSize(neuron_size)
        marocco.placement.use_output_buffer7_for_dnc_input_and_bg_hack = True
        marocco.placement.minSPL1 = False
        marocco.backend = PyMarocco.Hardware
        marocco.calib_backend = PyMarocco.XML
        marocco.calib_path = "/wang/data/calibration/wafer_0"

        marocco.roqt = "demo.roqt"
        marocco.bio_graph = "demo.dot"

        h276 = pyredman.Hicann()
        h276.drivers().disable(SynapseDriverOnHICANN(C.Enum(6)))
        h276.drivers().disable(SynapseDriverOnHICANN(C.Enum(20)))
        h276.drivers().disable(SynapseDriverOnHICANN(C.Enum(102)))
        h276.drivers().disable(SynapseDriverOnHICANN(C.Enum(104)))
        marocco.defects.inject(HICANNGlobal(Enum(276)), h276)

        h277 = pyredman.Hicann()
        marocco.defects.inject(HICANNGlobal(Enum(277)), h277)

        marocco.pll_freq = 100e6
        marocco.bkg_gen_isi = 10000
        marocco.only_bkg_visible = False

        sim.setup(marocco=marocco, **setup)

        return {
            "marocco": marocco
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

        # Try to setup the simulator
        if (simulator == "nmpm1"):
            self.backend_data = self._setup_nmpm1(sim, setup)
        else:
            sim.setup(**setup)
        return setup

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
            raise PyNNLessException("'type' key not present in description")
        elif (population["type"] in self.TYPES):
            type_name = population["type"]
            if (not hasattr(self.sim, type_name)):
                raise PyNNLessException("Neuron type '" + type_name
                        + "' not supported by backend.")
            type_ = getattr(self.sim, type_name)
            is_source = type_name == self.TYPE_SOURCE
        else:
            raise PyNNLessException("Invalid neuron type '" + type_name +
                "' supported are " + str(self.TYPES))

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

            # Issue warnings about ignored parameters
            for key, _ in population["params"].items():
                if (not key in params):
                    logger.warning("Given parameter '" + key + "' does not " +
                        "exist for neuron type '" + type_name + "'. Value " +
                        "will be ignored!")

        # Fetch the parameter dimensions that should be recorded for this
        # population, make sure the elements in "record" are sorted
        record = []
        if ("record" in population):
            if isinstance(population["record"], str):
                record = [population["record"]]
            else:
                record = list(population["record"])
            record.sort()
            for signal in record:
                if (not signal in self.SIGNALS):
                    logger.warning("Unknown signal \"" + signal
                        + "\". May be ignored by the backend.")

        # Write the sanitized population record back
        population["record"] = record

        # Create the output population, in case this is not a source population,
        # also force the neuron membrane potential to be initialized with the
        # neuron membrane potential.
        res = self.sim.Population(count, type_, params)
        if (self.version == 7):
            # Initialize membrane potential to v_rest, work around
            # "need more PhD-students"-exception on NMPM1 (where this condition
            # is fulfilled anyways)
            if ((not is_source) and (self.simulator != "nmpm1")):
                res.initialize("v", params["v_rest"])

            # Setup recording
            if (self.SIG_SPIKES in record):
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
            if (self.SIG_V in record):
                res.record_v()
            if ((self.SIG_GE in record) or (self.SIG_GI in record)):
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
        if (self.simulator == "nmpm1"):
            from pyhalbe.Coordinate import SynapseDriverOnHICANN, HICANNGlobal,\
                    X, Y, Enum, NeuronOnHICANN
            self.backend_data["marocco"].placement.add(res,
                    HICANNGlobal(Enum(276)))

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
        if (self.version == 7):
            if (signal == self.SIG_V):
                return self._convert_pyNN7_signal(population.get_v(), 2,
                    population.size)
            elif (signal == self.SIG_GE):
                return self._convert_pyNN7_signal(population.get_gsyn(), 2,
                    population.size)
            elif (signal == self.SIG_GI):
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
        raise PyNNLessException("DEFAULT_TIMESTEP not defined")

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
            raise PyNNLessException("\"populations\" key must be present in " +
                "network description")
        if (not "connections" in network):
            raise PyNNLessException("\"connections\" key must be present in " +
                "network description")

        # Generate the neuron populations
        population_count = len(network["populations"])
        populations = [None for _ in xrange(population_count)]
        for i in xrange(population_count):
            populations[i] = self._build_population(network["populations"][i])

        # Fetch the simulation timestep, work around bug #123 in sPyNNaker
        # See https://github.com/SpiNNakerManchester/sPyNNaker/issues/123
        # Do not call get_time_step() on the analogue hardware systems as this
        # will result in an exception.
        timestep = self._get_default_timestep()
        if (hasattr(self.sim, "get_time_step") and (not self.simulator == "ess"
                or self.simulator == "nmpm1")):
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

        # End the simulation to fetch the results on nmpm1
        if (self.simulator in self.PREMATURE_END_SIMULATORS):
            self.sim.end()

        # Gather the recorded data and store it in the result structure
        res = [{} for _ in xrange(population_count)]
        for i in xrange(population_count):
            if "record" in network["populations"][i]:
                for signal in network["populations"][i]["record"]:
                    if (signal == self.SIG_SPIKES):
                        res[i][signal] = self._fetch_spikes(populations[i])
                    else:
                        data = self._fetch_signal(populations[i], signal)
                        res[i][signal] = data["data"]
                        res[i][signal + "_t"] = data["time"]

        # End the simulation if this has not been done yet
        if (not (self.simulator in self.PREMATURE_END_SIMULATORS)):
            self.sim.end()

        return res

