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
Common preamble to be used by all examples
"""

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)

# Add the top-level directory to the path. This allows you to test the examples
# without having to install PyNNLess. You do not need this code one you have
# installed PyNNLess
import sys
import os
import __main__
sys.path.append(os.path.join(os.path.dirname(__main__.__file__), ".."))

# Import PyNNLess
from pynnless import PyNNLess as pl

# Assemble the script output file name
outpath = os.path.join(os.path.dirname(__main__.__file__), "out")
if not os.path.exists(outpath):
    os.makedirs(outpath)
outfile = os.path.join(outpath,
        os.path.basename(os.path.splitext(__main__.__file__)[0] +
            "_" + pl.normalized_simulator_name(sys.argv[1]) + ".txt"))

# Check the command line arguments
print("")
print("PyNNLess Example Script")
print("=======================")
print("")
if len(sys.argv) != 2:
    print("Usage: " + __main__.__file__ + " <SIMULATOR>")
    print
    simulators = pl.simulators()
    if (len(simulators) ==  0):
        print("You do not seem to have any PyNN backends installed. To get "
            + "started you should install NEST.")
    else:
        print("Where <SIMULATOR> may for example be one of the following "
            + "(these simulators have been auto-detected on your system):")
        print("\t" + str(pl.simulators()))
    print
    sys.exit(1)

