#
# SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import struct

from .ints_to_bytes import int_from_data, data_from_int

class FileRaw:
    def __init__(self, path, offset, size):
        self.fd = os.open(path, os.O_RDWR | os.O_SYNC)
        self.base_offset = offset
        self.size = size

    def __del__(self):
        if hasattr(self, "fd"):
            os.close(self.fd)

    def write(self, offset, data, size):
        os.lseek(self.fd, offset, os.SEEK_SET)
        os.write(self.fd, data_from_int(data, size))

    def write8(self, offset, data):
        self.write(offset, data, 1)

    def write16(self, offset, data):
        self.write(offset, data, 2)

    def write32(self, offset, data):
        self.write(offset, data, 4)

    def read(self, offset, size):
        os.lseek(self.fd, offset, os.SEEK_SET)
        data = os.read(self.fd, size)
        assert data, "offset %s size %d %s" % (hex(offset), size, data)
        return int_from_data(data, size)

    def read8(self, offset):
        return self.read(offset, 1)

    def read16(self, offset):
        return self.read(offset, 2)

    def read32(self, offset):
        return self.read(offset, 4)

    def read_format(self, fmt, offset):
        size = struct.calcsize(fmt)
        os.lseek(self.fd, offset, os.SEEK_SET)
        data = os.read(self.fd, size)
        return struct.unpack(fmt, data)
