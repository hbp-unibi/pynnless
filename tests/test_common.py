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
Tests common functionality independent of the backend.
"""

import unittest
import numpy as np

from pynnless import *

class TestCommon(unittest.TestCase):

    def test_check_version(self):
        """
        Tests whether the "_check_version" method works as expected.
        """
        self.assertEqual(7, PyNNLess._check_version("0.7.3"))
        self.assertEqual(7, PyNNLess._check_version("0.7.5"))
        self.assertEqual(8, PyNNLess._check_version("0.8beta2"))
        self.assertRaises(PyNNLessVersionException,
                lambda: PyNNLess._check_version("0.5"))

    def test_lookup_simulator(self):
        """
        Tests the static lookup_simulator method and checks whether all special
        cases are covered.
        """
        self.assertEqual(("bla", ["pyNN.bla"]),
                PyNNLess._lookup_simulator("bla"))
        self.assertEqual(("bla", ["pyNN.bla"]),
                PyNNLess._lookup_simulator("pyNN.bla"))
        self.assertEqual(("ess", ["pyNN.ess",
                "pyNN.hardware.brainscales"]),
                PyNNLess._lookup_simulator("ess"))
        self.assertEqual(("ess", ["pyNN.hardware.brainscales", "pyNN.ess"]),
                PyNNLess._lookup_simulator("hardware.brainscales"))
        self.assertEqual(("nmmc1", ["pyNN.nmmc1", "pyNN.spiNNaker"]),
                PyNNLess._lookup_simulator("nmmc1"))
        self.assertEqual(("nmmc1", ["pyNN.spiNNaker",
                "pyNN.nmmc1"]), PyNNLess._lookup_simulator("spiNNaker"))
        self.assertEqual(("nmpm1", ["pyNN.nmpm1", "pyhmf"]),
                PyNNLess._lookup_simulator("nmpm1"))
        self.assertEqual(("nmpm1", ["pyNN.pyhmf", "pyNN.nmpm1", "pyhmf"]),
                PyNNLess._lookup_simulator("pyhmf"))
        self.assertEqual(("nmpm1", ["pyNN.pyhmf", "pyNN.nmpm1", "pyhmf"]),
                PyNNLess._lookup_simulator("pyNN.pyhmf"))

    def test_build_connections(self):
        """
        Tests whether the internal "_build_connections" method behaves as
        expected.
        """
        connections = PyNNLess._build_connections([
            [(1, 0), (3, 5), 0.1, 0.0],
            [(2, 2), (3, 5), 0.2, 0.1],
            [(1, 0), (4, 1), 0.3, 0.2],
            [(1, 0), (4, 2), 0.4, 0.3],
        ])

        self.assertEqual({
            (1, 3): [(0, 5, 0.1, 0.0)],
            (2, 3): [(2, 5, 0.2, 0.1)],
            (1, 4): [(0, 1, 0.3, 0.2),
                     (0, 2, 0.4, 0.3)],
        }, connections)

    def test_convert_pyNN7_spikes(self):
        """
        Tests whether the internal "_convert_pyNN7_spikes" method behaves as
        expected.
        """
        spikes = PyNNLess._convert_pyNN7_spikes([
            [1, 0.2],
            [1, 0.1],
            [1, 0.3],
            [0, 1.2],
            [2, 2.3],
            [2, 2.2],
            [3, 0.2],
            [3, 0.1],
        ], 4)
        expected = [
            map(np.float32, [1.2]),
            map(np.float32, [0.1, 0.2, 0.3]),
            map(np.float32, [2.2, 2.3]),
            map(np.float32, [0.1, 0.2])
        ]
        self.assertEqual(spikes, expected)

    def test_convert_pyNN8_spikes(self):
        spikes = PyNNLess._convert_pyNN8_spikes([
            np.asarray([1.2]),
            np.asarray([0.1, 0.2, 0.3]),
            np.asarray([2.2, 2.3]),
            np.asarray([0.1, 0.2]),
        ])
        expected = [
            map(np.float32, [1.2]),
            map(np.float32, [0.1, 0.2, 0.3]),
            map(np.float32, [2.2, 2.3]),
            map(np.float32, [0.1, 0.2])
        ]
        self.assertEqual(spikes, expected)

    def test_convert_pyNN7_signal(self):
        """
        Tests whether the internal "_convert_pyNN7_signal" method behaves as
        expected.
        """
        signal = PyNNLess._convert_pyNN7_signal([
            [0, 0.0, 0.1, 0.2],
            [0, 0.1, 0.1, 0.3],
            [0, 0.2, 0.1, 0.4],
            [1, 0.0, 0.1, 0.5],
            [1, 0.1, 0.1, 0.6],
            [1, 0.2, 0.1, 0.7],
            [2, 0.0, 0.1, 0.8],
            [2, 0.1, 0.1, 0.9],
            [2, 0.2, 0.1, 1.0],
            [3, 0.0, 0.1, 1.1],
            [3, 0.1, 0.1, 1.2],
            [3, 0.3, 0.1, 1.3],
        ], 3, 4)

        expected = {
            "data": np.asarray([
                [0.2, 0.3, 0.4, np.nan],
                [0.5, 0.6, 0.7, np.nan],
                [0.8, 0.9, 1.0, np.nan],
                [1.1, 1.2, np.nan, 1.3]], dtype=np.float32),
            "time": np.asarray(
                [0.0, 0.1, 0.2, 0.3], dtype=np.float32)
        }

        np.testing.assert_equal(expected["data"], signal["data"])
        np.testing.assert_equal(expected["time"], signal["time"])

