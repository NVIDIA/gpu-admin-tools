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

import mmap

import platform
is_linux = platform.system() == "Linux"

if is_linux:
    import ctypes
    libc = ctypes.cdll.LoadLibrary('libc.so.6')

    # Set the mmap and munmap arg and return types.
    # last mmap arg is off_t which ctypes doesn't have. Assume it's long as that what gcc defines it to.
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mmap.restype = ctypes.c_void_p
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munmap.restype = ctypes.c_int

class FileMap:

    def __init__(self, path, offset, size):
        self.size = size
        with open(path, "r+b") as f:
            prot = mmap.PROT_READ | mmap.PROT_WRITE
            # Try mmap.mmap() first for error checking even if we end up using numpy
            mapped = mmap.mmap(f.fileno(), size, mmap.MAP_SHARED, prot, offset=offset)
            self.mapped = memoryview(mapped)
            self.map_8 = self.mapped.cast("B")
            self.map_16 = self.mapped.cast("H")
            self.map_32 = self.mapped.cast("I")


    def write8(self, offset, data):
        self.map_8[offset // 1] = data

    def write16(self, offset, data):
        self.map_16[offset // 2] = data

    def write32(self, offset, data):
        self.map_32[offset // 4] = data

    def read8(self, offset):
        return self.map_8[offset // 1]

    def read16(self, offset):
        return self.map_16[offset // 2]

    def read32(self, offset):
        return self.map_32[offset // 4]

    def read(self, offset, size):
        if size == 1:
            return self.read8(offset)
        elif size == 2:
            return self.read16(offset)
        elif size == 4:
            return self.read32(offset)
        else:
            raise ValueError(f"Unhandled read size {size}")
