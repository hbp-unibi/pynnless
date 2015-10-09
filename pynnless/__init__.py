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

# Import the "PyNNLess" class to the top-level package namespace
from pynnless import PyNNLess

# Import the "Builder" classes ito the top-level package namespace
from pynnless_builder import Population
from pynnless_builder import SourcePopulation
from pynnless_builder import IfCondExpPopulation
from pynnless_builder import AdExPopulation
from pynnless_builder import Network

# Import all constants from "Constants"
from pynnless_constants import *

# Import the exception classes
from pynnless_exceptions import PyNNLessException
from pynnless_exceptions import PyNNLessVersionException

# Current version of the "PyNNLess" wrapper
__version__ = "1.0.0"

# Export all classes
__all__ = [
    'PyNNLess', 'PyNNLessException', 'PyNNLessVersionException', 'Population',
    'SourcePopulation', 'IfCondExpPopulation', 'AdExPopulation', 'Network',
    'SIGNALS', 'SIG_SPIKES', 'SIG_GE', 'SIG_GI', 'SIG_V', 'TYPES', 'TYPE_AD_EX',
    'TYPE_SOURCE', 'TYPE_IF_COND_EXP', 'PARAMETER_LIMITS'
    ]
