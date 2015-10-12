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

import copy

def is_function(f):
    """
    Returns True if f is callable, False otherwise.
    """
    return hasattr(f, '__call__')

def init_key(tar, src, key, default):
    """
    Inits the key "key" in the target dictionary "tar" with the corresponding
    value in "src". If there is no such value in "src", uses the given default
    value instead.
    """
    if key in src:
        tar[key] = copy.deepcopy(src[key])
    else:
        if is_function(default):
            tar[key] = copy.deepcopy(default())
        else:
            tar[key] = copy.deepcopy(default)

