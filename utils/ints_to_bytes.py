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

import array
import struct
import sys
from collections.abc import Iterable
from typing import Any

_INT_SIZE_TO_FORMAT = {1: "B", 2: "H", 4: "I", 8: "Q"}
_STRUCT_BY_SIZE = {size: struct.Struct(f"={fmt}") for size, fmt in _INT_SIZE_TO_FORMAT.items()}


def _require_int_size(size: int) -> None:
    if size not in _INT_SIZE_TO_FORMAT:
        raise AssertionError(f"Unhandled size {size}")


def _byte_view(data: Any) -> memoryview:
    if isinstance(data, memoryview):
        view = data
    else:
        try:
            view = memoryview(data)
        except TypeError:
            view = memoryview(bytes(data))

    if not view.contiguous:
        view = memoryview(bytes(view))

    if view.format != "B":
        try:
            view = view.cast("B")
        except TypeError:
            view = memoryview(bytes(view))

    return view


def ints_from_data(data: Any, size: int) -> list[int]:
    _require_int_size(size)
    view = _byte_view(data)

    if view.nbytes % size != 0:
        raise struct.error(f"unpack requires a buffer of {size} bytes")

    if size == 1:
        return list(view)

    return view.cast(_INT_SIZE_TO_FORMAT[size]).tolist()


def int_from_data(data: Any, size: int) -> int:
    _require_int_size(size)

    if isinstance(data, (bytes, bytearray)):
        if len(data) != size:
            raise struct.error(f"unpack requires a buffer of {size} bytes")
        if size == 1:
            return data[0]
        return _STRUCT_BY_SIZE[size].unpack(data)[0]

    view = _byte_view(data)

    if view.nbytes != size:
        raise struct.error(f"unpack requires a buffer of {size} bytes")

    if size == 1:
        return view[0]

    return _STRUCT_BY_SIZE[size].unpack_from(view)[0]


def data_from_int(integer: int, size: int = 4) -> bytes:
    _require_int_size(size)
    return integer.to_bytes(size, byteorder=sys.byteorder)


def bytearray_view_from_ints(int_array: Iterable[int], type_code: str = "I") -> memoryview:
    arr = array.array(type_code, int_array)
    return memoryview(arr).cast("B").toreadonly()


def array_view_from_bytearray(ba: Any, type_code: str = "I") -> memoryview:
    arr = array.array(type_code)
    arr.frombytes(_byte_view(ba))
    return memoryview(arr).toreadonly()


def read_ints_from_path(path: str, offset: int, int_size: int, int_num: int = -1) -> list[int]:
    with open(path, "rb") as f:
        f.seek(offset, 0)
        if int_num == -1:
            size = -1
        else:
            size = int_size * int_num

        return ints_from_data(f.read(size), int_size)
