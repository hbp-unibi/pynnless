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
from pynnless import PyNNLessException
from pynnless import PyNNLessVersionException

# Current version of the "PyNNLess" wrapper
__version__ = "1.0.0"

# Export all classes
__all__ = ['PyNNLess', 'PyNNLessException', 'PyNNLessVersionException']
