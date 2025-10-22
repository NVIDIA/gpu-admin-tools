#
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Iterable

from .ints_to_bytes import *


class NiceStructMeta(type):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls.fmt_string = cls._construct_format_string()
        cls.size = struct.calcsize(cls.fmt_string)

class NiceStruct(metaclass=NiceStructMeta):
    _fields_ = []

    def __init__(self, bytedata=None):
        if bytedata == None:
            bytedata = b'\0' * self.size
        self.from_bytes(bytedata)

    @classmethod
    def _construct_format_string(cls):
        fmt = '<'
        bitfield_accumulated_size = 0
        bitfield_format = ''

        for field in cls._fields_:
            # Bitfield
            if len(field) == 3:
                name, current_format, bits = field
                bitfield_accumulated_size += bits

                if bitfield_format == '':
                    bitfield_format = current_format

                boundary_size = struct.calcsize(bitfield_format) * 8
                if bitfield_accumulated_size == boundary_size:
                    fmt += bitfield_format
                    bitfield_accumulated_size = 0
                    bitfield_format = ''
                elif bitfield_accumulated_size > boundary_size:
                    raise ValueError(f"Too many bits {bitfield_accumulated_size} for {field}")
            else:
                if bitfield_accumulated_size > 0:
                    raise ValueError(f"{field} starting with bitfields unfinished")
                fmt += field[1]

        if bitfield_accumulated_size > 0:
            raise ValueError(f"Bitfield unfinished with {bitfield_accumulated_size} bits")

        return fmt

    def get_pretty_value_dict(self):
        pretty = {}
        for field in self._fields_:
            name = field[0]
            value = getattr(self, name, "Undefined")
            value_str = f"{value}"
            if isinstance(value, int):
                value_str += f" ({value:#x})"
            pretty[name] = value_str
        return pretty

    def pretty_print(self):
        for name, value_str in self.get_pretty_value_dict().items():
            print(f"  {name}: {value_str}")

    def __str__(self):
        return f"{self.get_pretty_value_dict()}"

    def from_bytes(self, bytedata):
        unpacked_values = struct.unpack_from(self.fmt_string, bytedata)

        unpacked_pos = 0
        bitfield_counter = 0
        bitfield_accumulated_size = 0
        bitfield_format = ''
        current_value = None

        for field in self._fields_:
            if len(field) == 3:
                name, current_format, bits = field
                bitfield_accumulated_size += bits

                if bitfield_format == '':
                    bitfield_format = current_format
                    boundary_size = struct.calcsize(bitfield_format) * 8


                if bitfield_counter == 0:
                    current_value = unpacked_values[unpacked_pos]
                    unpacked_pos += 1

                mask = (1 << bits) - 1
                value = (current_value >> bitfield_counter) & mask
                setattr(self, name, value)

                bitfield_counter += bits

                if bitfield_accumulated_size == boundary_size:
                    bitfield_accumulated_size = 0
                    bitfield_counter = 0
                    bitfield_format = ''
                else:
                    # _construct_format_string() enforces that bitfields always
                    # specify all bits of the type they are using.
                    assert bitfield_accumulated_size < boundary_size
            else:
                name, format_string = field
                # _construct_format_string() enforces that bitfields always
                # specify all bits of the type they are using.
                assert bitfield_counter == 0

                if len(format_string) > 1 and format_string[-1] != "s" and format_string[:-1].isdigit():
                    # Handle array-like fields. Skip "s" as it's handled in a special way as bytes()
                    count = int(format_string[:-1])
                    values = unpacked_values[unpacked_pos:unpacked_pos+count]
                    setattr(self, name, values)
                    unpacked_pos += count
                else:
                    setattr(self, name, unpacked_values[unpacked_pos])
                    unpacked_pos += 1

    def to_bytes(self):
        packed_values = []
        bitfield_value = 0
        bitfield_counter = 0
        bitfield_accumulated_size = 0
        bitfield_format = ''

        for field in self._fields_:
            if len(field) == 3:
                name, current_format, bits = field
                value = getattr(self, name)
                bitfield_value |= (value & ((1 << bits) - 1)) << bitfield_counter
                bitfield_counter += bits

                if bitfield_format == '':
                    bitfield_format = current_format

                bitfield_accumulated_size += bits
                boundary_size = struct.calcsize(bitfield_format) * 8

                if bitfield_accumulated_size == boundary_size:
                    packed_values.append(bitfield_value)
                    bitfield_value = 0
                    bitfield_counter = 0
                    bitfield_accumulated_size = 0
                    bitfield_format = ''
                else:
                    # _construct_format_string() enforces that bitfields always
                    # specify all bits of the type they are using.
                    assert bitfield_accumulated_size < boundary_size

            else:
                # _construct_format_string() enforces that bitfields always
                # specify all bits of the type they are using.
                assert bitfield_counter == 0

                name, format_string = field
                value = getattr(self, name)
                if len(format_string) > 1 and format_string[:-1].isdigit():
                    # Handle array-like fields
                    packed_values.extend(value)
                else:
                    packed_values.append(value)

        return struct.pack(self.fmt_string, *packed_values)

    def from_ints(self, ints: Iterable[int]) -> None:
        self.from_bytes(bytearray_view_from_ints(ints))

    def to_int_list(self, int_size: int = 4) -> list[int]:
        return ints_from_data(self.to_bytes(), int_size)

    def to_int(self, int_size: int = 4) -> int:
        int_list = self.to_int_list(int_size)
        if len(int_list) != 1:
            raise ValueError(f"{self.name} doesn't fit in a single int with {int_size} bytes")
        return int_list[0]

    def from_int(self, integer: int) -> None:
        self.from_ints([integer])

class NiceStructArray:
    def __init__(self, struct_class, count):
        self.struct_class = struct_class
        self.size = struct_class.size * count
        self.array = [struct_class() for _ in range(count)]

    def __getitem__(self, index):
        return self.array[index]

    def __setitem__(self, index, value):
        if not isinstance(value, self.struct_class):
            raise ValueError(f"Value must be an instance of {self.struct_class.__name__}")
        self.array[index] = value

    def __len__(self):
        return len(self.array)

    def __iter__(self):
        return iter(self.array)

    def from_bytes(bytedata):
        if len(bytedata) != self.size:
            raise ValueError("Byte data does not match expected size")

        struct_size = self.struct_class.size
        for i in range(count):
            chunk = bytedata[i*struct_size : (i+1)*struct_size]
            instance.array[i].from_bytes(chunk)

        return instance

    def to_bytes(self):
        return b''.join(item.to_bytes() for item in self.array)
