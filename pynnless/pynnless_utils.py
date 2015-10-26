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


# Simple FileLock implementation -- adapted from
# https://raw.githubusercontent.com/derpston/python-simpleflock/master/src/simpleflock.py
#

import time
import os
import fcntl
import errno

class FileLock:
    """
    Provides the simplest possible interface to flock-based file locking.
    Intended for use with the `with` syntax. It will create/truncate/delete the
    lock file as necessary.
    """

    def __init__(self, path, timeout = None):
        self._path = path
        self._timeout = timeout
        self._fd = None

    def __enter__(self):
        # Simply do nothing if no path was given
        if self._path == None:
            return

        self._fd = os.open(self._path, os.O_CREAT)
        start_lock_search = time.time()
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except IOError, ex:
                if ex.errno != errno.EAGAIN: # Resource temporarily unavailable
                    raise
                elif self._timeout is not None and time.time() > (start_lock_search + self._timeout):
                    # Exceeded the user-specified timeout.
                    raise

            # TODO It would be nice to avoid an arbitrary sleep here, but spinning
            # without a delay is also undesirable.
            time.sleep(0.1)

    def __exit__(self, *args):
        # Simply do nothing if no path was given
        if self._path == None:
            return

        # Unlock the file and close the handle
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None

        # Try to remove the lock file, but don't try too hard because it is
        # unnecessary. This is mostly to help the user see whether a lock
        # exists by examining the filesystem.
        try:
            os.unlink(self._path)
        except:
            pass

