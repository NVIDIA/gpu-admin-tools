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

import platform
is_linux = platform.system() == "Linux"

# Use libc's ffs() on Linux and fall back to a native implementation otherwise.
if is_linux:
    import ctypes
    libc = ctypes.cdll.LoadLibrary('libc.so.6')

    def ffs(n):
        return libc.ffs(n)
else:
    def ffs(n):
        return (n & (-n)).bit_length()

class Bitfield(object):
    """Wrapper around bitfields, see PciUncorrectableErrors for an example"""
    fields = {}

    def __init__(self, raw, name=None):
        self.raw = raw
        if name is None:
            name = self.__class__.__name__
        self.name = name

    def __field_get_mask(self, field):
        bits = self.__class__.fields[field]
        if isinstance(bits, int):
            return bits

        assert isinstance(bits, tuple)
        high_bit = bits[0]
        low_bit = bits[1]

        mask = (1 << (high_bit - low_bit + 1)) - 1
        mask <<= low_bit
        return mask

    def __field_get_shift(self, field):
        mask = self.__field_get_mask(field)
        assert mask != 0
        return ffs(mask) - 1

    def __getitem__(self, field):
        mask = self.__field_get_mask(field)
        shift = self.__field_get_shift(field)
        return (self.raw & mask) >> shift

    def __setitem__(self, field, val):
        mask = self.__field_get_mask(field)
        shift = self.__field_get_shift(field)

        val = val << shift
        assert (val & ~mask) == 0, "value 0x%x mask 0x%x" % (val, mask)

        self.raw = (self.raw & ~mask) | val

    def __str__(self):
        return self.name + " " + str(self.values()) + " raw " + hex(self.raw)

    def values(self):
        vals = {}
        for f in self.__class__.fields:
            vals[f] = self[f]

        return vals

    def non_zero(self):
        ret = {}
        for k, v in self.values().items():
            if v != 0:
                ret[k] = v
        return ret

    def non_zero_fields(self):
        ret = []
        for k, v in self.values().items():
            if v != 0:
                ret.append(k)
        return ret
