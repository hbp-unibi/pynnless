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
Tests the classes in the pynnless_builder submodule.
"""

import unittest

from pynnless import *

class TestBuilder(unittest.TestCase):

    def test_population(self):
        self.assertRaises(PyNNLessException, lambda: Population(count=0))
        self.assertRaises(PyNNLessException, lambda: Population(count=-1))
        self.assertRaises(PyNNLessException, lambda: Population(_type="foo"))

        pop = Population()
        self.assertEqual(1, pop["count"])
        self.assertEqual(TYPE_IF_COND_EXP, pop["type"])
        self.assertEqual([], pop["record"])
        self.assertEqual({}, pop["params"])

        pop.record_spikes()
        self.assertEqual([SIG_SPIKES], pop["record"])

        pop = Population(count=5, _type=TYPE_IF_COND_EXP)
        pop2 = Population(pop)
        self.assertEqual(pop, pop2)
