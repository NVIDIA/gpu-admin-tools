#
# SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

class DeviceField:
    """Wrapper for a device register/setting defined by a bitfield class and
    accessible with dev.read()/write() at the specified offset"""
    def __init__(self, bitfield_class, dev, offset, name=None):
        self.dev = dev
        self.offset = offset
        self.bitfield_class = bitfield_class
        self.size = bitfield_class.size
        if name is None:
            name = bitfield_class.__name__
        self.name = name
        self._read()

    def _read(self):
        raw = self.dev.read(self.offset, self.size)
        self.value = self.bitfield_class(raw, name=self.name)
        return self.value

    def _write(self):
        self.dev.write(self.offset, self.value.raw, self.size)

    def __getitem__(self, field):
        self._read()
        return self.value[field]

    def __setitem__(self, field, val):
        self._read()
        self.value[field] = val
        self._write()

    def write_only(self, field, val):
        """Write to the device with only the field set as specified. Useful for W0C bits"""

        bf = self.bitfield_class(-1)
        bf[field] = val
        self.dev.write(self.offset, bf.raw, self.size)
        self._read()

    def write_raw(self, value):
        self.value.raw = value
        self._write()
        self._read()

    def __str__(self):
        self._read()
        return str(self.value)
