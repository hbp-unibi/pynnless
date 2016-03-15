#!/bin/sh
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

find -depth \(\
		\(\
			   -name "*.backup"\
			-o -name "*~"\
			-o -name "*.pyc"\
			-o -name "core.[0-9]*[0-9]"\
			-o -name "stderr.*.tmp"\
			-o -name "stdout.*.tmp"\
			-o -name "spiketrain.in"\
			-o -name "spikeyconfig.out"\
			-o -name "logfile.txt"\
			-o -wholename "*examples/out*"\
			-o -wholename "*examples/fpga_conf*"\
			-o -wholename "*reports*"\
			-o -wholename "*application_generated_data*"\
			-o -wholename "*README.html"\
		\) -a \! \(\
			   -wholename "*.git/*"\
			-o -wholename "*.svn" \)\
		\) -delete -print
