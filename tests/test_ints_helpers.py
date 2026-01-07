import array
import struct
import time
from collections import namedtuple

import pytest

from utils.formatted_tuple import FormattedTuple
from utils.ints_to_bytes import (
    array_view_from_bytearray,
    bytearray_view_from_ints,
    data_from_int,
    int_from_data,
    ints_from_data,
)
from utils.nice_struct import NiceStruct


class DemoStruct(NiceStruct):
    name = "DemoStruct"
    _fields_ = [
        ("a", "I"),
        ("b", "I"),
    ]


class DemoFormattedTuple(FormattedTuple):
    namedtuple = namedtuple("DemoTuple", ["value"])
    struct = struct.Struct("<I")


def test_ints_from_data_roundtrip():
    raw = bytes(range(32))
    ints = ints_from_data(raw, 4)
    assert ints == [int.from_bytes(raw[i : i + 4], "little") for i in range(0, len(raw), 4)]
    rebuilt = bytearray()
    for value in ints:
        rebuilt.extend(data_from_int(value, 4))
    assert rebuilt == raw


def test_ints_from_data_rejects_partial_chunks():
    with pytest.raises(struct.error):
        ints_from_data(b"\x01\x02\x03", 2)


def test_int_from_data_validates_length():
    assert int_from_data(b"\x01\x02\x03\x04", 4) == 0x04030201
    with pytest.raises(struct.error):
        int_from_data(b"\x01\x02", 4)


def test_int_helpers_reject_unknown_sizes():
    with pytest.raises(AssertionError):
        ints_from_data(b"\x01\x02\x03", 3)
    with pytest.raises(AssertionError):
        data_from_int(0x01, 3)


def test_bytearray_view_from_int_array_accepts_iterables():
    values = (0x01020304, 0x05060708)
    view_from_tuple = bytearray_view_from_ints(values)
    assert isinstance(view_from_tuple, memoryview)
    assert view_from_tuple.readonly
    assert view_from_tuple.tolist() == [4, 3, 2, 1, 8, 7, 6, 5]

    gen_values = (i for i in values)
    view_from_gen = bytearray_view_from_ints(gen_values)
    assert view_from_gen.tolist() == [4, 3, 2, 1, 8, 7, 6, 5]


def test_array_view_from_bytearray_returns_int_view():
    as_ints = array.array("I", [0x01020304, 0x05060708])
    mv = array_view_from_bytearray(as_ints.tobytes())
    assert isinstance(mv, memoryview)
    assert mv.readonly
    assert list(mv) == list(as_ints)


def test_nice_struct_from_int_array_with_non_list():
    values = (0x01020304, 0x05060708)
    instance = DemoStruct()
    instance.from_ints(values)
    assert instance.a == values[0]
    assert instance.b == values[1]

    instance.from_ints(i for i in values)
    assert instance.a == values[0]
    assert instance.b == values[1]

    assert instance.to_int_list() == list(values)


def test_formatted_tuple_make_accepts_sequence():
    result = DemoFormattedTuple._make([1, 2, 3, 4])
    assert result.value == 0x04030201

    result = DemoFormattedTuple._make(bytearray(b"\x01\x02\x03\x04"))
    assert result.value == 0x04030201

    with pytest.raises(struct.error):
        DemoFormattedTuple._make(b"\x01\x02")
