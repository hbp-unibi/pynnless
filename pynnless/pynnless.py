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
import numpy as np

# Simulator loading and lookup
import importlib
import pkgutil

# Logging
import logging

# Standard libraries
import copy
import time
import os
import sys

# Own classes
import pynnless_builder as builder
import pynnless_constants as const
import pynnless_exceptions as exceptions

# Local logger, write to stderr
logger = logging.getLogger("PyNNLess")
if len(logger.handlers) == 0:
    logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)

# Temporary file descriptors to the original stdout/stderr
oldstdout = None
oldstderr = None

# Currently used file names for logging
stdout_fn = None
stderr_fn = None


class PyNNLess:
    """
    The backend class is used as an abstraction to the actual PyNN backend,
    which may either be present in version 0.6, 0.7 or 0.8. Furthermore, it
    constructs the network from a simple graph abstraction.
    """

    # List containing all supported simulators. Other simulators may also be
    # supported if they follow the PyNN specification, however, these simulators
    # were tested and will be returned by the "backends" method. The simulator
    # names are the normalized simulator names.
    SUPPORTED_SIMULATORS = {
        "nest", "ess", "nmpm1", "nmmc1", "spikey"
    }

    # Used to map certain simulator names to a more canonical form. This
    # canonical form does not correspond to the name of the actual module
    # includes but the names that "feel more correct".
    NORMALIZED_SIMULATOR_NAMES = {
        "hardware.brainscales": "ess",
        "spiNNaker": "nmmc1",
        "pyhmf": "nmpm1",
        "hardware.spikey": "spikey",
    }

    # Maps certain simulator names to the correct PyNN module names. If multiple
    # module names are given, the first found module is used.
    SIMULATOR_IMPORT_MAP = {
        "ess": ["pyNN.hardware.brainscales"],
        "nmmc1": ["pyNN.spiNNaker"],
        "nmpm1": ["pyhmf"],
        "spikey": ["pyNN.hardware.spikey"],
    }

    # List of simulators that need a call to "end" before the results are
    # retrieved
    PREMATURE_END_SIMULATORS = ["nmpm1"]

    # List of simulators which are hardware systems without an explicit
    # timestep
    ANALOGUE_SYSTEMS = ["ess", "nmpm1", "spikey"]

    # List of hardware systems
    HARDWARE_SYSTEMS = ["nmpm1", "nmmc1", "spikey"]

    # Map used for remapping neuron types to internal types based on the current
    # simulator
    NEURON_TYPE_REMAP = {
        "spikey": {
            const.TYPE_IF_COND_EXP: "IF_facets_hardware1"
        }
    }

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
            "ignoreHWParameterRanges": True,
            "useSystemSim": True,
        },
        "nmmc1": {
            "timestep": 1.0
        },
        "nmpm1": {
            "neuron_size": 4,
            "hicann": 276
        },
        "spikey": {
            # No default setup parameters needed
        },
    }

    # Time to wait after the last spike has been issued
    AUTO_DURATION_EXTENSION = 1000.0

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
        try:
            with open(filename, 'r') as fd:
                lines = fd.readlines()
                if (len(lines) > 0):
                    logger.info("[" + title + "]")
                    if (len(lines) > 100):
                        logger.info("[...]")
                    for line in lines[-100:]:
                        logger.info(line.strip('\n\r'))
                    logger.info("[end]")
        except:
            pass

    @classmethod
    def _redirect_io(cls, do_redirect=True):
        """
        Redirects both stderr and stdout to some temporary files.
        """
        global oldstdout, oldstderr, stdout_fn, stderr_fn

        # Abort if redirection is disabled
        if not do_redirect:
            return

        r = str(np.random.randint(1 << 16))
        if (oldstdout is None):
            stdout_fn = "stdout." + r + ".tmp"
            oldstdout = cls._redirect_fd_to_file(1, stdout_fn)
        if (oldstderr is None):
            stderr_fn = "stderr." + r + ".tmp"
            oldstderr = cls._redirect_fd_to_file(2, stderr_fn)

    @classmethod
    def _unredirect_io(cls, tail=True):
        """
        Reverts the redirection performed by _redirect_io and prints the last
        few lines of both files (if the "tail" parameter is set ot true).
        """
        global oldstdout, oldstderr, stdout_fn, stderr_fn

        try:
            if oldstderr is not None and stderr_fn is not None:
                sys.stderr.flush()
                try:
                    cls._redirect_fd_to_fd(oldstderr, 2)
                except:
                    pass
                if (tail):
                    cls._tail(stderr_fn, "stderr")
                os.remove(stderr_fn)
        except:
            pass
        finally:
            oldstderr = None
            stderr_fn = None

        try:
            if oldstdout is not None and stdout_fn is not None:
                sys.stdout.flush()
                cls._redirect_fd_to_fd(oldstdout, 1)
                if (tail):
                    cls._tail(stdout_fn, "stdout")
                os.remove(stdout_fn)
        except:
            pass
        finally:
            oldstdout = None
            stdout_fn = None

    @staticmethod
    def _check_version(version=pyNN.__version__):
        """
        Internally used to check the current PyNN version. Sets the "version"
        variable to the correct API version and raises an exception if the PyNN
        version is not supported.

        :param version: the PyNN version string to be checked, should be the
        value of pyNN.__version__
        """
        if (version[0:3] == '0.6'):
            return 6
        elif (version[0:3] == '0.7'):
            return 7
        elif (version[0:3] == '0.8'):
            return 8
        raise exceptions.PyNNLessVersionException("Unsupported PyNN version '"
                                                  + version + "', supported are pyNN 0.6, 0.7 and 0.8")

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
        import pylogging
        from pymarocco import PyMarocco, Placement
        from pyhalbe.Coordinate import HICANNGlobal, Enum

        # Deactivate logging
        for domain in ["Default", "marocco", "sthal.HICANNConfigurator.Time"]:
            pylogging.set_loglevel(
                pylogging.get(domain), pylogging.LogLevel.ERROR)

        # Copy and delete non-standard setup parameters
        neuron_size = setup["neuron_size"]
        hicann_number = setup["hicann"]
        del setup["neuron_size"]
        del setup["hicann"]

        marocco = PyMarocco()
        marocco.placement.setDefaultNeuronSize(neuron_size)
        marocco.placement.use_output_buffer7_for_dnc_input_and_bg_hack = True
        marocco.placement.minSPL1 = False
        marocco.backend = PyMarocco.Hardware
        marocco.calib_backend = PyMarocco.XML
        marocco.calib_path = "/wang/data/calibration/wafer_0"

        hicann = HICANNGlobal(Enum(hicann_number))

        # Pass the marocco object and the actual setup to the simulation setup
        # method
        sim.setup(marocco=marocco, **setup)

        # Return the marocco object and a list containing all HICANN
        return {
            "marocco": marocco,
            "hicann": hicann,
            "neuron_size": neuron_size
        }

    def _setup_simulator(self, setup, sim, simulator, version):
        """
        Internally used to setup the simulator with the given setup parameters.

        :param setup: setup dictionary to be passed to the simulator setup.
        """

        # Assemble the setup
        setup = self._build_setup(setup, sim, simulator, version)

        # Read PyNNLess specific flags
        if "fix_parameters" in setup:
            self.fix_parameters = bool(setup["fix_parameters"])
            del setup["fix_parameters"]
        if "redirect_io" in setup:
            self.do_redirect = bool(setup["redirect_io"])
            del setup["redirect_io"]
        if "summarise_io" in setup:
            self.summarise_io = bool(setup["summarise_io"])
            del setup["summarise_io"]

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
            self._redirect_io(do_redirect=self.do_redirect)
            if (simulator == "nmpm1"):
                self.backend_data = self._setup_nmpm1(sim, setup)
                self.repeat_projections = self.backend_data["neuron_size"]
            else:
                sim.setup(**setup)
        finally:
            self._unredirect_io(self.summarise_io)
        return setup

    def _remap_neuron_type(self, type_name):
        """
        Some neuron types are available under a different name on certain
        simulator backends. This method remaps the given type_name to those
        types.
        """
        if ((self.simulator in self.NEURON_TYPE_REMAP) and
                (type_name in self.NEURON_TYPE_REMAP[self.simulator])):
            return self.NEURON_TYPE_REMAP[self.simulator][type_name]
        return type_name

    def _fix_parameters(self, params, params_orig, type_name):
        """
        Performs a few backend specific parameter adaptations.
        """

        # Abort if parameter adaptation has been deactivated
        if not self.fix_parameters:
            return params

        # ESS specific adaptations
        if ((self.simulator == "ess") and
                (not self.setup["ignoreHWParameterRanges"])):
            if type_name == const.TYPE_IF_COND_EXP:
                if params["cm"] != 0.2:
                    params["cm"] = 0.2
                    self.parameter_warnings.add("cm set to 0.2")
                if params["e_rev_E"] != 0.0:
                    params["e_rev_E"] = 0.0
                    self.parameter_warnings.add("e_rev_E set to 0.0 mV")
                if params["e_rev_I"] != -100.0:
                    params["e_rev_I"] = -100.0
                    self.parameter_warnings.add("e_rev_I set to -100.0 mV")
                if params["v_rest"] != -50.0:
                    vOffs = (-50.0) - params["v_rest"]
                    params["v_rest"] = params["v_rest"] + vOffs
                    params["v_reset"] = params["v_reset"] + vOffs
                    params["v_thresh"] = params["v_thresh"] + vOffs
                    self.parameter_warnings.add("set v_rest to -50.0 mV, " +
                                                "offset v_thresh, v_reset")

        # Spikey specific adaptations
        if self.simulator == "spikey":
            if type_name == "IF_facets_hardware1":
                # Shift the voltages below -55.0 mV
                vs = [params["v_rest"], params["v_reset"], params["v_thresh"],
                      params["e_rev_I"]]
                vMax = max(vs)
                if vMax > -55.0:
                    vOffs = (-55.0) - vMax
                    params["e_rev_I"] = params["e_rev_I"] + vOffs
                    params["v_rest"] = params["v_rest"] + vOffs
                    params["v_reset"] = params["v_reset"] + vOffs
                    params["v_thresh"] = params["v_thresh"] + vOffs
                    self.parameter_warnings.add("Neuron potentials were "
                                                + "shifted to stay below -55mV")

        # Convert g_leak to tau_m and vice versa
        cm = None
        if "cm" in params:
            cm = params["cm"] * 1e-9
        elif self.simulator == "spikey":
            cm = 0.2e-9
        if ("tau_m" in params_orig) and ("g_leak" in params_orig):
            if ("tau_m" in params):
                self.parameter_warnings.add(
                    "Specified both tau_m and g_leak, using tau_m")
            else:
                self.parameter_warnings.add(
                    "Specified both tau_m and g_leak, using g_leak")
        elif (not cm is None):
            if ("g_leak" in params) and ("tau_m" in params_orig):
                # g_leak [nS] = cm [nF] / tau_m [ms]
                params["g_leak"] = (cm / (params_orig["tau_m"] * 1e-3)) * 1e9
                self.parameter_warnings.add("Converted tau_m to g_leak")
            if ("tau_m" in params) and ("g_leak" in params_orig):
                # tau_m [ms] [nS] = cm [nF] / g_leak [ms]
                params["tau_m"] = (cm / (params_orig["g_leak"] * 1e-6)) * 1e3
                self.parameter_warnings.add("Converted g_leak to tau_m")

        return params

    def _init_cells(self, is_source, cells, params):
        # Initialize membrane potential to v_rest on systems
        # where the initialize method is available (not NMPM1
        # and SPIKEY)
        if ((not is_source) and hasattr(cells, "initialize")):
            try:
                cells.initialize("v", params["v_rest"])
            except:
                # This does not seem to be implemented on most
                # platforms
                self.warnings.add("Neuron membrane potential " +
                                  "initialization failed")
                pass

    def _build_population(self, population, min_delay=0):
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
        type_name = self._remap_neuron_type(population["type"])
        if (not hasattr(self.sim, type_name)):
            raise exceptions.PyNNLessException("Neuron type '" + type_name
                                               + "' not supported by backend.")
        type_ = getattr(self.sim, type_name)
        is_source = type_name == const.TYPE_SOURCE

        params = [[]] * len(population["params"])
        for i in xrange(len(population["params"])):
            # Fetch the default parameters for this neuron type and merge them
            # with parameters given for this population.
            params[i] = self.merge_default_parameters(population["params"][i],
                                                      type_name, type_)

            # For some hardware platforms we need to adapt the parameters a
            # little for the system to run -- if any change is done, PyNNLess
            # issues a warning notifying the user about these adaptations
            params[i] = self._fix_parameters(params[i], population["params"][i],
                                             type_name)

            # Issue warnings about ignored parameters
            for key, _ in population["params"][i].items():
                if (not key in params[i]):
                    self.warnings.add("Given parameter '" + key + "' does not " +
                                      "exist for neuron type '" + type_name + "'. Value " +
                                      "will be ignored!")

        # Fetch the parameter dimensions that should be recorded for this
        # population, make sure the elements in "record" are sorted
        record = population["record"]
        for signal in record:
            if (not signal in const.SIGNALS):
                self.warnings.add("Unknown signal \"" + signal
                                  + "\". May be ignored by the backend.")

        # Make sure the spike times are larger or equal to one -- this
        # otherwise causes a problem with the spikes simply discarded when
        # using NEST
        for i in xrange(len(params)):
            if "spike_times" in params[i]:
                min_t = max(min_delay, 1.0)
                for j, t in enumerate(params[i]["spike_times"]):
                    if t < min_t:
                        params[i]["spike_times"][j] = min_t
                params[i]["spike_times"].sort()

        # Create the output population, in case this is not a source population,
        # also force the neuron membrane potential to be initialized with the
        # neuron membrane potential.
        try:
            self._redirect_io(do_redirect=self.do_redirect)
            # Use global parameters if the length of the parameter list is
            # exactly one -- otherwise we'll override the parameters given here
            # in a later step.
            # Note: deepcopy is needed because of spyNNaker bug #161
            res = self.sim.Population(count, type_, copy.deepcopy(params[0]))
            if len(params) == 1:
                self._init_cells(is_source, res, params[0])
            else:
                for i in xrange(count):
                    if hasattr(self.sim, "PopulationView"):
                        # The PopulationView class is the best way to set
                        # individual neuron parameters in a population, however
                        # it is not available on NM-MC1 and Spikey
                        view = self.sim.PopulationView(res, [i])
                        self._init_cells(is_source, view, params[i])
                        if self.version <= 7:
                            view.set(params[i])
                        else:
                            view.set(**params[i])
#                    # Only for reference: This works nowhere (except Spikey)
#                    elif hasattr(res, "__getitem__"):
#                        if not is_source:
#                            try:
#                                self.sim.initialize(res[i], params[i]["v_rest"])
#                            except:
#                                pass # Does not work with Spikey and NMPM1
#                        if self.simulator == "spikey":
#                            for key in params[i].keys():
#                                self.sim.set(res[i], res[i].cellclass, key, params[i][key])
#                        elif self.version <= 7:
#                            self.sim.set(res[i], params[i])
#                        else:
#                            self.sim.set(res[i], **params[i])
                    else:
                        # Use the tset method which has a less convenient
                        # interface and requires an array for each parameter,
                        # containing the values for each neuron.

                        # First assemble a list of parameter values for each
                        # parameter key. Note that the available parameter keys
                        # were unified in the merge_default_parameters method
                        keys = params[0].keys()
                        tvals = dict([(key, [[]] * count) for key in keys])
                        uni = dict([(key, True) for key in keys])
                        for i in xrange(count):
                            for k in keys:
                                tvals[k][i] = params[i][k]
                                uni[k] = uni[k] and tvals[k][0] == tvals[k][i]

                        # Actually call tset for all keys for which there is a
                        # difference
                        for k in tvals.keys():
                            if not uni[k]:
                                res.tset(k, tvals[k])
        finally:
            self._unredirect_io(False)

        # Increment the neuron counter needed to work around a bug in spikey,
        # store the neuron index in the created population
        if not is_source:
            if self.simulator == "spikey":
                setattr(res, "__offs", self.neuron_count)
            self.neuron_count += count

        # Setup recording
        if (self.version <= 7):
            # Setup recording
            if (const.SIG_SPIKES in record):
                res.record()
            if (const.SIG_V in record):
                # Special handling for voltage recording with Spikey
                if (self.simulator == "spikey"):
                    if (self.record_v_count > 0 or count > 1):
                        self.warnings.add("Spikey can only record from a " +
                                          "single neuron. Only recording membrane " +
                                          "potential for the first neuron in the first " +
                                          "for which the membrane potential should be " +
                                          "recorded")
                    if self.record_v_count == 0:
                        setattr(res, "__spikey_record_v", True)
                        self.sim.record_v(res[0], '')
                else:
                    res.record_v()

                # Increment the record_v_count variable
                self.record_v_count += count
            if ((const.SIG_GE in record) or (const.SIG_GI in record)):
                res.record_gsyn()
        elif (self.version == 8):
            # Setup recording
            res.record(record)

        # Workaround bug in NMPM1, "size" attribute does not exist
        if (not hasattr(res, "size")):
            setattr(res, "size", count)

        # For NMPM1: register the population in the marocco instance
        if self.simulator == "nmpm1" and not is_source:
            try:
                self._redirect_io(do_redirect=self.do_redirect)
                self.backend_data["marocco"].placement.add(res,
                                                           self.backend_data["hicann"])
            finally:
                self._unredirect_io(False)

        return res

    @staticmethod
    def _build_connections(connections, min_delay=0, separate=False):
        """
        Gets an array of [[pid_src, nid_src], [pid_tar, nid_tar], weight, delay]
        tuples an builds a dictionary of all (pid_src, pid_tar) mappings.
        """
        res_exc = {}
        res_inh = {}
        for connection in connections:
            src, tar, weight, delay = connection
            pids = (src[0], tar[0])
            descrs = (
                src[1], tar[1], abs(weight) if separate else weight, max(
                    min_delay, delay))
            res_tar = res_exc if (not separate) or weight > 0 else res_inh
            if (pids in res_tar):
                res_tar[pids].append(descrs)
            else:
                res_tar[pids] = [descrs]
        return res_exc, res_inh

    @staticmethod
    def _convert_pyNN7_spikes(spikes, n, idx_offs=0, t_scale=1.0):
        """
        Converts a pyNN7 spike train, list of (nid, time)-tuples, into a list
        of lists containing the spike times for each neuron individually.
        """

        # Create one result list for each neuron
        res = [[] for _ in xrange(n)]
        for row in spikes:
            nIdx = int(row[0]) - idx_offs
            if nIdx >= 0 and nIdx < n:
                res[nIdx].append(float(row[1]) * t_scale)
            elif row[0] >= 0 and row[0] < n:
                # In case the Spikey indexing bug gets fixed, this code should
                # execute instead of the above.
                res[int(row[0])].append(float(row[1]) * t_scale)

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
            [float(spikes[i][j]) for j in xrange(len(spikes[i]))]
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

    def _fetch_spikey_voltage(self, population):
        """
        Hack which returns the voltages recorded by spikey.
        """
        vs = self.sim.membraneOutput
        ts = self.sim.timeMembraneOutput

        res = np.zeros((population.size, len(ts)))
        res[0] = vs
        return {"data": res, "time": ts}

    def _fetch_spikes(self, population):
        """
        Fetches the recorded spikes from a neuron population and performs all
        necessary data structure conversions for the PyNN versions.

        :param population: reference at a PyNN population object from which the
        spikes should be obtained.
        """
        if (self.version <= 7):
            # Workaround for spikey, which seems to index the neurons globally
            # instead of per-neuron
            idx_offs = (getattr(population, "__offs")
                        if hasattr(population, "__offs") else 0)
            if self.simulator == "nmpm1":
                return self._convert_pyNN7_spikes(population.getSpikes(),
                                                  population.size, idx_offs=idx_offs + 1, t_scale=1000.0)
            else:
                return self._convert_pyNN7_spikes(population.getSpikes(),
                                                  population.size, idx_offs=idx_offs)
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
            self.warnings.add("nmpm1 does not support retrieving recorded " +
                              "signals for now")
            return {"data": np.zeros((population.size, 0), dtype=np.float32),
                    "time": np.zeros((0), dtype=np.float32)}
        if (self.version <= 7):
            if (signal == const.SIG_V):
                # Special handling for the spikey simulator
                if (self.simulator == "spikey"):
                    if (hasattr(population, "__spikey_record_v")):
                        return self._fetch_spikey_voltage(population)
                else:
                    return self._convert_pyNN7_signal(population.get_v(), 2,
                                                      population.size)
            elif (signal == const.SIG_GE):
                return self._convert_pyNN7_signal(population.get_gsyn(), 2,
                                                  population.size)
            elif (signal == const.SIG_GI):
                # Workaround in bug #124 in sPyNNaker, see
                # https://github.com/SpiNNakerManchester/sPyNNaker/issues/124
                if (self.simulator != "nmmc1"):
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
        return 0.1  # Above values are not defined in PyNN 0.6

    @classmethod
    def _auto_duration(cls, network):
        """
        Automatically calculates a network duration according to the last input
        spike.
        """

        def _max_recursive(v, vs):
            for v2 in vs:
                if isinstance(v2, list):
                    v = _max_recursive(v, v2)
                else:
                    v = max(v, v2)
            return v

        def _max_spike_time(duration, params):
            if "spike_times" in params:
                return _max_recursive(duration, params["spike_times"])
            return duration

        duration = 0
        for p in network["populations"]:
            if ("type" in p) and (p["type"] == const.TYPE_SOURCE):
                if "params" in p:
                    if isinstance(p["params"], list):
                        for i in xrange(len(p["params"])):
                            duration = _max_spike_time(
                                duration, p["params"][i])
                    else:
                        duration = _max_spike_time(duration, p["params"])
        return duration + cls.AUTO_DURATION_EXTENSION

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

    # Number of times the connections must be repeated -- required for NMPM1
    repeat_projections = 1

    # Set of changes performed on the neuron parameters -- changes are collected
    # in a set and printed before a simulation is started
    parameter_warnings = set()

    # Set of generic warnings that should be issued before the simulation is
    # started
    warnings = set()

    # Flag which indicates whether the parameters should be adapted or not. Can
    # be deactivated by setting the corresponding "fix_parameters" setup flag.
    fix_parameters = True

    # Number of times the output potential has been recorded -- some platforms
    # only support a limited number of voltage recordings. Only valid for
    # PyNN 0.7 or lower.
    record_v_count = 0

    # Global count of non-source neurons within all populations
    neuron_count = 0

    # Total time for the "run" method in ms
    time_total = 0.0

    # Simulation time (the time sim.run ran)
    time_sim = 0.0

    # Finalization time (time after sim.run)
    time_finalize = 0.0

    # Initialization time (time before sim.run)
    time_initialize = 0.0

    # Flag indicating whether the I/O should be redirected
    do_redirect = True

    # Flag indicating whether the I/O should be summarised
    summarise_io = True

    def __init__(self, simulator, setup={}):
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
        passed to the "setup" method. Special PyNNLess specific setup parameters
        include the "fix_parameters" flag which indicates whether backend
        specific parameter adaptations should be performed.
        """

        self.version = self._check_version()
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

    @classmethod
    def default_parameters(cls, type_name):
        """
        Returns the default parameters for a certain neuron type.

        :param type_name: is the neuron type name
        """

        # In case we're dealing with PyNN 0.6, use the "cells" module,
        # otherwise the "standardmodels.cells"
        if cls._check_version() == 6:
            import pyNN.cells
            module = pyNN.cells
        else:
            import pyNN.standardmodels.cells
            module = pyNN.standardmodels.cells

        # The "dict" makes sure a copy is returned
        return dict(getattr(module, type_name).default_parameters)

    @classmethod
    def merge_default_parameters(cls, params, type_name, type_=None):
        """
        Merges the given parameter object with the default parameters for the
        given neuron type. Removes any keys from params that are not listed in
        the default parameters.

        :params params: parameters to be merged with the default parameters.
        :params type_name: name of the neuron type for which the default
        parameters should be retrieved.
        :param type_: neuron type class -- if it exposes a "default_parameters"
        attribute this value is used for default parameters
        """
        # Try to fetch the default parameters
        if (type_ is not None and hasattr(type_, "default_parameters")):
            res = dict(type_.default_parameters)
        else:
            res = cls.default_parameters(type_name)
        # Only copy existing parameter keys from the user-supplied parameters
        for key, _ in res.items():
            if (key in params):
                # Convert integer parameters to floating point values, fixes bug
                # with PyNN 0.7.5 and NEST 2.2.2
                if isinstance(params[key], int):
                    res[key] = float(params[key])
                else:
                    res[key] = params[key]
        # The default empty PyNN "spike_times" parameter is faulty
        if (not "spike_times" in params) and ("spike_times" in res):
            del res["spike_times"]
        return res

    @staticmethod
    def clamp_parameters(params):
        """
        This method can be used to make sure that the given parameters fall
        within their boundaries -- this is e.g. important if noise has been
        added to the parameters.
        """
        res = dict(params)
        for key in res:
            if key in const.PARAMETER_LIMITS:
                if "min" in const.PARAMETER_LIMITS[key]:
                    res[key] = max(
                        res[key], const.PARAMETER_LIMITS[key]["min"])
                if "max" in const.PARAMETER_LIMITS[key]:
                    res[key] = min(
                        res[key], const.PARAMETER_LIMITS[key]["max"])
        return res

    def get_time_step(self):
        # Fetch the simulation timestep, work around bugs #123 and #147 in
        # sPyNNaker.
        # Do not call get_time_step() on the analogue hardware systems as this
        # will result in an exception.
        timestep = self._get_default_timestep()
        if (hasattr(self.sim, "get_time_step") and not (
                (self.simulator in self.ANALOGUE_SYSTEMS) or
                (self.simulator == "nmmc1"))):
            timestep = self.sim.get_time_step()
        elif ("timestep" in self.setup):
            timestep = self.setup["timestep"]
        return timestep

    @classmethod
    def get_simulator_info_static(cls, simulator, inst=None):
        """
        Returns information about the specified simulator without actually
        setting it up.
        """
        # Make sure the given simulator is in its canonical form
        simulator, _ = PyNNLess._lookup_simulator(simulator)

        # Lookup the concurrency
        res = {}
        if not simulator in cls.HARDWARE_SYSTEMS:
            import multiprocessing
            res["max_neuron_count"] = 512
            res["concurrency"] = multiprocessing.cpu_count()
            res["is_hardware"] = False
            res["is_software"] = True
            res["is_emulation"] = False
        else:
            res["concurrency"] = 1
            res["is_hardware"] = True
            res["is_software"] = False
            res["is_emulation"] = False

        # Whether the system alows only one set of neuron parameters
        res["shared_parameters"] = []

        # Whether spike sources actually count as neurons in the
        # "max_neuron_count" measure
        res["sources_are_neurons"] = False

        # Set hardware-specific limitations
        if simulator == "ess":
            res["max_neuron_count"] = 224
            res["is_emulation"] = True
        elif simulator == "nmpm1":
            size = 4 if inst is None else inst.backend_data["neuron_size"]
            res["max_neuron_count"] = 224 // size
        elif simulator == "nmmc1":
            # res["max_neuron_count"] = 3 * 48 * 16 * 128 # TODO: Actual board
            # size
            res["sources_are_neurons"] = True
            res["max_neuron_count"] = 1500
        elif simulator == "spikey":
            res["max_neuron_count"] = 192
            res["shared_parameters"] = ["v_rest", "v_reset", "v_thresh",
                                        "e_rev_I"]

        return res

    def get_simulator_info(self):
        """
        Returns information about the currently selected simulator -- the
        maximum number of neurons and how many simulations can run in parallel
        on a single machine.
        """
        return self.get_simulator_info_static(self.simulator, self)

    def get_time_info(self):
        """
        Returns timing information about the last run. All times are in seconds.
        """
        return {
            "total": self.time_total,
            "sim": self.time_sim,
            "initialize": self.time_initialize,
            "finalize": self.time_finalize
        }

    def run(self, network, duration=0):
        """
        Builds and runs the network described in the "network" structure.

        :param network: Dictionary with two entries: "populations" and
        "connections", where the first introduces the individual neuron
        populations and their parameters and the latter is an adjacency list
        containing the connection weights and delays between neurons.
        :param duration: Simulation duration. If smaller than or equal to zero,
        the simulation duration is automatically determined depending on the
        last input spike time.
        :return: the recorded signals for each population, signal type and
        neuron
        """

        # First time measurement point
        t1 = time.clock()

        # Make sure both the "populations" and "connections" arrays have been
        # supplied
        if (not "populations" in network):
            raise exceptions.PyNNLessException("\"populations\" key must be " +
                                               "present in network description")
        if (not "connections" in network):
            raise exceptions.PyNNLessException("\"connections\" key must be " +
                                               "present in network description")

        # Reset some state variables
        self.parameter_warnings = set()
        self.warnings = set()
        self.record_v_count = 0
        self.neuron_count = 0

        # Fetch the timestep
        timestep = self.get_time_step()

        # Automatically fetch the runtime of the network if none is given
        if duration <= 0:
            duration = self._auto_duration(network)

        # Round up the duration to the timestep -- fixes a problem with
        # SpiNNaker
        duration = int((duration + timestep) / timestep) * timestep

        # Generate the neuron populations
        population_count = len(network["populations"])
        populations = [None for _ in xrange(population_count)]
        for i in xrange(population_count):
            populations[i] = self._build_population(
                network["populations"][i], timestep)

        # Build the connection matrices, and perform the actual connections
        separate_connections = self.simulator == "nmmc1"
        connections_exc, connections_inh = self._build_connections(
            network["connections"], timestep, separate=separate_connections)

        # Inform the user about the parameter adaptations and other warnings
        for warning in self.warnings:
            logger.warning(warning)
        for warning in self.parameter_warnings:
            logger.warning("Adapted neuron parameters: " + warning)
        if len(self.parameter_warnings) != 0:
            logger.warning("Parameter adaptations have been performed. Set " +
                           "the setup flag \"fix_parameters\" to False to suppress this " +
                           "behaviour.")
        self.warnings = set()

        try:
            self._redirect_io(do_redirect=self.do_redirect)

            # Perform the actual connections
            for pids, descrs in connections_exc.items():
                for _ in xrange(self.repeat_projections):
                    self.sim.Projection(
                        populations[pids[0]], populations[pids[1]],
                        self.sim.FromListConnector(descrs))
            for pids, descrs in connections_inh.items():
                for _ in xrange(self.repeat_projections):
                    self.sim.Projection(
                        populations[pids[0]], populations[pids[1]],
                        self.sim.FromListConnector(descrs), target="inhibitory")

            # Run the simulation, measure time
            t2 = time.clock()
            self.sim.run(duration)
            t3 = time.clock()

            # End the simulation to fetch the results on nmpm1
            if (self.simulator in self.PREMATURE_END_SIMULATORS):
                self.sim.end()

            # Gather the recorded data and store it in the result structure
            res = [{} for _ in xrange(population_count)]
            for i in xrange(population_count):
                if "record" in network["populations"][i]:
                    signals = builder.Population.canonicalize_record(
                        network["populations"][i]["record"])
                    for signal in signals:
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
            self._unredirect_io(self.summarise_io)

        # Print post-execution warnings
        for warning in self.warnings:
            logger.warning(warning)

        # Store the time measurements, can be retrieved using the
        # "get_time_info" method
        t4 = time.clock()
        self.time_total = t4 - t1
        self.time_sim = t3 - t2
        self.time_initialize = t2 - t1
        self.time_finalize = t4 - t3

        return res

