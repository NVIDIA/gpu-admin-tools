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

import struct

def _struct_fmt(size):
   if size == 1:
       return "B"
   elif size == 2:
       return "=H"
   elif size == 4:
       return "=I"
   elif size == 8:
       return "=Q"
   else:
       assert 0, "Unhandled size %d" % size

def ints_from_data(data, size):
    fmt = _struct_fmt(size)
    # Wrap data in bytes() for python 2.6 compatibility
    data = bytes(data)
    ints = []
    for offset in range(0, len(data), size):
        ints.append(struct.unpack(fmt, data[offset : offset + size])[0])

    return ints

def int_from_data(data, size):
    fmt = _struct_fmt(size)
    # Wrap data in bytes() for python 2.6 compatibility
    return struct.unpack(fmt, bytes(data))[0]

def data_from_int(integer, size=4):
    fmt = _struct_fmt(size)
    return struct.pack(fmt, integer)

def bytearray_from_ints(array_of_ints, size=4):
    ba = bytearray()
    for i in array_of_ints:
        ba.extend(data_from_int(i, size))
    return ba

def ints_from_bytearray(ba, int_size):
    ints = []
    for i in range(0, len(ba), int_size):
        data = ba[i:int_size]
        ints.append(int_from_data(ba[i : i + int_size], int_size))
    return ints

def read_ints_from_path(path, offset, int_size, int_num=-1):
    with open(path, 'rb') as f:
        f.seek(offset, 0)
        if int_num == -1:
            size = -1
        else:
            size = int_size * int_num

        return ints_from_data(f.read(size), int_size)
