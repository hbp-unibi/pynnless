# -*- coding: utf-8 -*-

#   PyNAM -- Python Neural Associative Memory Simulator and Evaluator
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
Provides a wrapper around PyNNLess which isolates running the network into its
own process, thus allowing to use PyNNLess multiple times and ensuring a "fresh"
state each time. Furthermore it serializes access to the hardware systems,
allowing multiple processes using the specified backend to be started
concurrently.

Note however, that the provided PyNNLessIsolated class comes with a performance
overhead due to inter process communication and that not all public methods of
the PyNNLess class are implemented to the full extent.
"""

import traceback
import multiprocessing

from pynnless import PyNNLess
from pynnless_utils import FileLock

def _PyNNLessIsolatedMain(q, lockfile, simulator, setup, network, duration):
    """
    Function to be executed in its own isolated process.
    """
    with FileLock(lockfile, release=False):
        inst = None
        res = None
        times = None
        exception = None
        try:
            inst = PyNNLess(simulator, setup)
            res = inst.run(network, duration)
            times = inst.get_time_info()
        except:
            exception = traceback.format_exc()

        q.put((res, times, exception))

class PyNNLessIsolated:
    #
    # Public interface
    #

    # Actual simulator instance, loaded by the "load" method.
    sim = None

    # Name of the simulator
    simulator = ""

    # Copy of the setup parameters
    setup = {}

    # Time information received from the subprocess
    times = {}

    def __init__(self, simulator, setup = {}, concurrent=False):
        self.simulator = simulator
        self.setup = setup
        self.concurrent = concurrent

    @staticmethod
    def simulators():
        return PyNNLess.simulators()

    @staticmethod
    def normalized_simulator_name(simulator):
        return PyNNLess.normalized_simulator_name(simulator)

    @staticmethod
    def default_parameters(type_name):
        return PyNNLess.default_parameters(type_name)

    @staticmethod
    def merge_default_parameters(params, type_name, type_=None):
        return PyNNLess.merge_default_parameters(params, type_name, type_=None)

    @staticmethod
    def clamp_parameters(params):
        return PyNNLess.clamp_parameters(params)

    @staticmethod
    def get_simulator_info_static(simulator, inst=None):
        return PyNNLess.get_simulator_info_static(simulator, inst=None)

    def get_simulator_info(self):
        return self.get_simulator_info_static(self.simulator)

    def get_time_info(self):
        return self.times

    def run(self, network, duration = 0):
        # Fetch some simulator information -- serialize access to the simulator
        # backend if we're dealing with a hardware system
        info = self.get_simulator_info()
        lockfile = None
        if info["is_hardware"]:
            lockfile = '.~' + self.simulator

        # Call the _PyNNLess_Async_Main method in another process and wait for
        # the response
        q = multiprocessing.Queue()
        p = multiprocessing.Process(
                target=_PyNNLessIsolatedMain,
                args=(q, lockfile, self.simulator, self.setup, network,
                    duration)
            )
        p.start()
        res, self.times, exception = q.get()
        p.join()

        # Rethrow an exception if one happened in the child process
        if exception != None:
            raise Exception(exception)

        # Return the computation result
        return res

