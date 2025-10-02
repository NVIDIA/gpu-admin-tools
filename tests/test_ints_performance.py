import struct
import time

import pytest

# New, optimized implementations from the codebase
from utils.ints_to_bytes import bytearray_view_from_ints as bytearray_view_from_ints_new
from utils.ints_to_bytes import data_from_int as data_from_int_new
from utils.ints_to_bytes import int_from_data as int_from_data_new
from utils.ints_to_bytes import ints_from_data as ints_from_data_new

# --- Original implementations (pre-optimization) for comparison ---


def _struct_fmt_original(size):
    if size == 1:
        return "B"
    elif size == 2:
        return "=H"
    elif size == 4:
        return "=I"
    elif size == 8:
        return "=Q"
    else:
        assert 0, f"Unhandled size {size}"


def ints_from_data_original(data, size):
    fmt = _struct_fmt_original(size)
    data = bytes(data)
    ints = []
    for offset in range(0, len(data), size):
        ints.append(struct.unpack(fmt, data[offset : offset + size])[0])
    return ints


def int_from_data_original(data, size):
    fmt = _struct_fmt_original(size)
    return struct.unpack(fmt, bytes(data))[0]


def data_from_int_original(integer, size=4):
    fmt = _struct_fmt_original(size)
    return struct.pack(fmt, integer)


def bytearray_from_ints_original(array_of_ints, size=4):
    ba = bytearray()
    for i in array_of_ints:
        ba.extend(data_from_int_original(i, size))
    return ba


# ----------------------------------------------------------------


@pytest.mark.performance
def test_ints_from_data_performance_comparison():
    chunk = bytes(range(256)) * 1024  # 256 KiB
    iterations = 100

    start_original = time.perf_counter()
    for _ in range(iterations):
        ints_from_data_original(chunk, 4)
    duration_original = time.perf_counter() - start_original

    start_new = time.perf_counter()
    for _ in range(iterations):
        ints_from_data_new(chunk, 4)
    duration_new = time.perf_counter() - start_new

    print(f"\nPerformance for ints_from_data ({iterations} iterations on 256KiB chunk):")
    print(f"  - Original (struct): {duration_original:.4f}s")
    print(f"  - New (int.from_bytes): {duration_new:.4f}s")
    assert duration_new < duration_original


@pytest.mark.performance
def test_int_from_data_performance_comparison():
    data = b"\xde\xad\xbe\xef"
    iterations = 500000

    start_original = time.perf_counter()
    for _ in range(iterations):
        int_from_data_original(data, 4)
    duration_original = time.perf_counter() - start_original

    start_new = time.perf_counter()
    for _ in range(iterations):
        int_from_data_new(data, 4)
    duration_new = time.perf_counter() - start_new

    print(f"\nPerformance for int_from_data ({iterations} iterations):")
    print(f"  - Original (struct): {duration_original:.4f}s")
    print(f"  - New (int.from_bytes): {duration_new:.4f}s")
    assert duration_new < duration_original


@pytest.mark.performance
def test_data_from_int_performance_comparison():
    integer = 0xDEADBEEF
    iterations = 500000

    start_original = time.perf_counter()
    for _ in range(iterations):
        data_from_int_original(integer, 4)
    duration_original = time.perf_counter() - start_original

    start_new = time.perf_counter()
    for _ in range(iterations):
        data_from_int_new(integer, 4)
    duration_new = time.perf_counter() - start_new

    print(f"\nPerformance for data_from_int ({iterations} iterations):")
    print(f"  - Original (struct): {duration_original:.4f}s")
    print(f"  - New (int.to_bytes): {duration_new:.4f}s")
    assert duration_new < duration_original


@pytest.mark.performance
def test_array_creation_performance_comparison():
    int_list = list(range(1024 * 10))  # 10k integers
    iterations = 100

    start_original = time.perf_counter()
    for _ in range(iterations):
        bytearray_from_ints_original(int_list, 4)
    duration_original = time.perf_counter() - start_original

    start_new = time.perf_counter()
    for _ in range(iterations):
        bytearray_view_from_ints_new(int_list)
    duration_new = time.perf_counter() - start_new

    print(f"\nPerformance for array creation ({iterations} iterations on 10k ints):")
    print(f"  - Original (bytearray extend): {duration_original:.4f}s")
    print(f"  - New (array.extend + memoryview): {duration_new:.4f}s")
    assert duration_new < duration_original
