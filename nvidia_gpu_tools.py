#!/usr/bin/env python3

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

from __future__ import print_function
from enum import Enum
import os
import mmap
import struct
from struct import Struct
import json
import time
import sys
import random
import optparse
import traceback
from logging import debug, info, warning, error
import logging
from pathlib import Path
from utils import NiceStruct
from utils import int_from_data, data_from_int, bytearray_from_ints, ints_from_bytearray, read_ints_from_path
from utils import formatted_tuple_from_data
from gpu.defines import *
from pci.defines import *

if hasattr(time, "perf_counter"):
    perf_counter = time.perf_counter
else:
    perf_counter = time.time

import platform
is_windows = platform.system() == "Windows"
is_linux = platform.system() == "Linux"

is_sysfs_available = is_linux


if is_linux:
    import ctypes

# By default use /dev/mem for MMIO, can be changed with --mmio-access-type sysfs
mmio_access_type = "devmem"

VERSION = "v2024.02.14o"

SYS_DEVICES = "/sys/bus/pci/devices/"

def sysfs_find_parent(device):
    device = os.path.basename(device)
    for device_dir in os.listdir(SYS_DEVICES):
        dev_path = os.path.join(SYS_DEVICES, device_dir)
        for f in os.listdir(dev_path):
            if f == device:
                return dev_path
    return None

def find_gpus_sysfs(bdf_pattern=None):
    gpus = []
    other = []
    dev_paths = []
    for device_dir in os.listdir("/sys/bus/pci/devices/"):
        dev_path = os.path.join("/sys/bus/pci/devices/", device_dir)
        bdf = device_dir
        if bdf_pattern:
            if bdf_pattern not in bdf:
                continue
        vendor = open(os.path.join(dev_path, "vendor")).readlines()
        vendor = vendor[0].strip()
        if vendor != "0x10de":
            continue
        cls = open(os.path.join(dev_path, "class")).readlines()
        cls = cls[0].strip()
        if cls != "0x030000" and cls != "0x030200" and cls != "0x068000":
            continue
        dev_paths.append(dev_path)

    def devpath_to_id(dev_path):
        bdf = os.path.basename(dev_path)
        return int(bdf.replace(":","").replace(".",""), base=16)

    dev_paths = sorted(dev_paths, key=devpath_to_id)
    for dev_path in dev_paths:
        gpu = None
        cls = open(os.path.join(dev_path, "class")).readlines()
        cls = cls[0].strip()
        try:
            if cls == "0x068000":
                dev = NvSwitch(dev_path=dev_path)
            else:
                dev = Gpu(dev_path=dev_path)
        except UnknownGpuError as err:
            error("Unknown Nvidia device %s: %s", dev_path, str(err))
            dev = NvidiaDevice(dev_path=dev_path)
            other.append(dev)
            continue
        except Exception as err:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            error("GPU %s broken: %s", dev_path, str(err))
            dev = BrokenGpu(dev_path=dev_path)
        gpus.append(dev)

    return (gpus, other)


def find_gpus(bdf=None):
    if is_sysfs_available:
        return find_gpus_sysfs(bdf)

class PageInfo(object):
    def __init__(self, vaddr, num_pages=1, pid="self"):
        pagemap_path = "/proc/{0}/pagemap".format(pid)
        self.page_size = os.sysconf("SC_PAGE_SIZE")
        offset  = (vaddr // self.page_size) * 8
        self._pagemap_entries = read_ints_from_path(pagemap_path, offset, int_size=8, int_num=num_pages)

    def physical_address(self, page):
        pfn = self._pagemap_entries[page] & 0x7FFFFFFFFFFFFF
        return pfn * self.page_size


class FileRaw(object):
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

class FileMap(object):

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

# Check that modules needed to access devices on the system are available
def check_device_module_deps():
    pass


GPU_ARCHES = ["kepler", "maxwell", "pascal", "volta", "turing", "ampere", "ada", "hopper"]
NVSWITCH_MAP = {
    0x6000a1: {
        "name": "LR10",
        "arch": "limerock",
        "other_falcons": ["soe"],
        "falcons_cfg": {
            "soe": {
                "imem_size": 65536,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
        },
    },
    0x7000a1: {
        "name": "NVSwitch_gen3",
        "arch": "laguna",
        "nvlink": {
            "number": 64,
            "links_per_group": 4,
            "base_offset": 0x1000000,
            "per_group_offset": 0x100000,
        },
        "other_falcons": ["fsp"],
        "needs_falcons_cfg": False,
    }
}
NVSWITCH_ARCHES = ["limerock", "laguna"]

# For architectures with multiple products, match by device id as well. The
# values from this map are what's used in the GPU_MAP.
GPU_MAP_MULTIPLE = {
    0x170000a1: {
        "devids": {
            0x20b7: "A30",
        },
        "default": "A100",
    },

    0xb72000a1: {
        "devids": {
            0x2236: "A10",
            0x2237: "A10G",
        },
        "default": "A10",
    },
    0x180000a1: {
        "devids": {
            0x2330: "H100-SXM",
            0x2336: "H100-SXM",
            0x2322: "H800-PCIE",
            0x2324: "H800-SXM",
        },
        "default": "H100-PCIE",
    },

}

GPU_MAP = {
    0x0e40a0a2: {
        "name": "K520",
        "arch": "kepler",
        "pmu_reset_in_pmc": True,
        "memory_clear_supported": False,
        "forcing_ecc_on_after_reset_supported": False,
        "nvdec": [],
        "nvenc": [],
        "other_falcons": ["msvld", "mspdec", "msppp", "msenc", "hda", "disp"],
        "falcons_cfg": {
            "pmu": {
                "imem_size": 24576,
                "dmem_size": 24576,
                "imem_port_count": 1,
                "dmem_port_count": 4,
            },
            "msvld": {
                "imem_size": 8192,
                "dmem_size": 4096,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "msppp": {
                "imem_size": 2560,
                "dmem_size": 2048,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "msenc": {
                "imem_size": 16384,
                "dmem_size": 6144,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "mspdec": {
                "imem_size": 5120,
                "dmem_size": 4096,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "hda": {
                "imem_size": 4096,
                "dmem_size": 4096,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "disp": {
                "imem_size": 16384,
                "dmem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 4,
            },
        }
    },
    0x0f22d0a1: {
        "name": "K80",
        "arch": "kepler",
        "pmu_reset_in_pmc": True,
        "memory_clear_supported": False,
        "forcing_ecc_on_after_reset_supported": False,
        "nvdec": [],
        "nvenc": [],
        "other_falcons": ["msvld", "mspdec", "msppp", "msenc", "hda", "disp"],
        "falcons_cfg": {
            "pmu": {
                "imem_size": 24576,
                "dmem_size": 24576,
                "imem_port_count": 1,
                "dmem_port_count": 4,
            },
            "msvld": {
                "imem_size": 8192,
                "dmem_size": 4096,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "msppp": {
                "imem_size": 2560,
                "dmem_size": 2048,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "msenc": {
                "imem_size": 16384,
                "dmem_size": 6144,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "mspdec": {
                "imem_size": 5120,
                "dmem_size": 4096,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "hda": {
                "imem_size": 4096,
                "dmem_size": 4096,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "disp": {
                "imem_size": 16384,
                "dmem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 4,
            },
        },
    },
    0x124320a1: {
        "name": "M60",
        "arch": "maxwell",
        "pmu_reset_in_pmc": True,
        "memory_clear_supported": False,
        "forcing_ecc_on_after_reset_supported": False,
        "nvdec": [0],
        "nvenc": [0, 1],
        "other_falcons": ["sec"],
        "falcons_cfg": {
            "pmu": {
                "imem_size": 49152,
                "dmem_size": 40960,
                "imem_port_count": 1,
                "dmem_port_count": 4,
            },
            "nvdec0": {
                "imem_size": 8192,
                "dmem_size": 6144,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvenc0": {
                "imem_size": 16384,
                "dmem_size": 12288,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvenc1": {
                "imem_size": 16384,
                "dmem_size": 12288,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "sec": {
                "imem_size": 32768,
                "dmem_size": 16384,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
        },
    },
    0x130000a1: {
        "name": "P100",
        "arch": "pascal",
        "pmu_reset_in_pmc": True,
        "memory_clear_supported": False,
        "forcing_ecc_on_after_reset_supported": False,
        "nvdec": [0],
        "nvenc": [0, 1, 2],
        "other_falcons": ["sec"],
        "falcons_cfg": {
            "pmu": {
                "imem_size": 49152,
                "dmem_size": 40960,
                "imem_port_count": 1,
                "dmem_port_count": 4,
            },
            "nvdec0": {
                "imem_size": 10240,
                "dmem_size": 10240,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvenc0": {
                "imem_size": 16384,
                "dmem_size": 16384,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvenc1": {
                "imem_size": 16384,
                "dmem_size": 16384,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvenc2": {
                "imem_size": 16384,
                "dmem_size": 16384,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "sec": {
                "imem_size": 32768,
                "dmem_size": 16384,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
        },
    },
    0x134000a1: {
        "name": "P4",
        "arch": "pascal",
        "pmu_reset_in_pmc": False,
        "memory_clear_supported": False,
        "forcing_ecc_on_after_reset_supported": False,
        "nvdec": [0],
        "nvenc": [0, 1],
        "other_falcons": ["sec"],
        "falcons_cfg": {
            "pmu": {
                "imem_size": 65536,
                "dmem_size": 65536,
                "imem_port_count": 1,
                "dmem_port_count": 4,
            },
            "nvdec0": {
                "imem_size": 10240,
                "dmem_size": 24576,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvenc0": {
                "imem_size": 16384,
                "dmem_size": 16384,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvenc1": {
                "imem_size": 16384,
                "dmem_size": 16384,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "sec": {
                "imem_size": 65536,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
        },
    },
    0x132000a1: {
        "name": "P40",
        "arch": "pascal",
        "pmu_reset_in_pmc": False,
        "memory_clear_supported": False,
        "forcing_ecc_on_after_reset_supported": False,
        "nvdec": [0],
        "nvenc": [0, 1],
        "other_falcons": ["sec"],
        "falcons_cfg": {
            "pmu": {
                "imem_size": 65536,
                "dmem_size": 65536,
                "imem_port_count": 1,
                "dmem_port_count": 4,
            },
            "nvdec0": {
                "imem_size": 10240,
                "dmem_size": 24576,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvenc0": {
                "imem_size": 16384,
                "dmem_size": 16384,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvenc1": {
                "imem_size": 16384,
                "dmem_size": 16384,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "sec": {
                "imem_size": 65536,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
        },
    },
    0x140000a1: {
        "name": "V100",
        "arch": "volta",
        "pmu_reset_in_pmc": False,
        "memory_clear_supported": False,
        "forcing_ecc_on_after_reset_supported": False,
        "nvdec": [0],
        "nvenc": [0, 1, 2],
        "other_falcons": ["sec", "gsp", "fb", "minion"],
        "falcons_cfg": {
            "pmu": {
                "imem_size": 65536,
                "dmem_size": 65536,
                "imem_port_count": 1,
                "dmem_port_count": 4,
            },
            "nvdec0": {
                "imem_size": 10240,
                "dmem_size": 24576,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvenc0": {
                "imem_size": 16384,
                "dmem_size": 16384,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvenc1": {
                "imem_size": 16384,
                "dmem_size": 16384,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvenc2": {
                "imem_size": 16384,
                "dmem_size": 16384,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "gsp": {
                "imem_size": 65536,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "sec": {
                "imem_size": 65536,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "fb": {
                "imem_size": 16384,
                "dmem_size": 16384,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "minion": {
                "imem_size": 8192,
                "dmem_size": 4096,
                "imem_port_count": 1,
                "dmem_port_count": 4,
            },
        }
    },


    0x164000a1: {
        "name": "T4",
        "arch": "turing",
        "pmu_reset_in_pmc": False,
        "memory_clear_supported": True,
        "forcing_ecc_on_after_reset_supported": True,
        "nvdec": [0, 1],
        "nvenc": [0],
        "other_falcons": ["sec", "gsp", "fb", "minion"],
        "falcons_cfg": {
            "pmu": {
                "imem_size": 65536,
                "dmem_size": 65536,
                "imem_port_count": 1,
                "dmem_port_count": 4,
            },
            "nvdec0": {
                "imem_size": 32768,
                "dmem_size": 32768,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvdec1": {
                "imem_size": 32768,
                "dmem_size": 32768,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvenc0": {
                "imem_size": 32768,
                "dmem_size": 32768,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "gsp": {
                "imem_size": 65536,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "sec": {
                "imem_size": 65536,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "fb": {
                "imem_size": 65536,
                "dmem_size": 65536,
                "imem_port_count": 1,
                "dmem_port_count": 4,
            },
            "minion": {
                "imem_size": 16384,
                "dmem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 4,
            },
        },
    },
    "A100": {
        "name": "A100",
        "arch": "ampere",
        "pmu_reset_in_pmc": False,
        "memory_clear_supported": True,
        "forcing_ecc_on_after_reset_supported": True,
        "nvdec": [0, 1, 2, 3, 4],
        "nvenc": [],
        "other_falcons": ["sec", "gsp"],
        "nvlink": {
            "number": 12,
            "links_per_group": 4,
            "base_offset": 0xa00000,
            "per_group_offset": 0x40000,
        },
        "falcons_cfg": {
            "pmu": {
                "imem_size": 131072,
                "dmem_size": 131072,
                "imem_port_count": 1,
                "dmem_port_count": 4,
            },
            "nvdec0": {
                "imem_size": 32768,
                "dmem_size": 32768,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvdec1": {
                "imem_size": 32768,
                "dmem_size": 32768,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvdec2": {
                "imem_size": 32768,
                "dmem_size": 32768,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvdec3": {
                "imem_size": 32768,
                "dmem_size": 32768,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvdec4": {
                "imem_size": 32768,
                "dmem_size": 32768,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "gsp": {
                "imem_size": 65536,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "sec": {
                "imem_size": 65536,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
        },
    },
    "A30": {
        "name": "A30",
        "arch": "ampere",
        "pmu_reset_in_pmc": False,
        "memory_clear_supported": True,
        "forcing_ecc_on_after_reset_supported": True,
        "nvdec": [0, 1, 2, 3],
        "nvenc": [],
        "other_falcons": ["sec", "gsp"],
        "falcons_cfg": {
            "pmu": {
                "imem_size": 131072,
                "dmem_size": 131072,
                "imem_port_count": 1,
                "dmem_port_count": 4,
            },
            "nvdec0": {
                "imem_size": 32768,
                "dmem_size": 32768,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvdec1": {
                "imem_size": 32768,
                "dmem_size": 32768,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvdec2": {
                "imem_size": 32768,
                "dmem_size": 32768,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "nvdec3": {
                "imem_size": 32768,
                "dmem_size": 32768,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "gsp": {
                "imem_size": 65536,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
            "sec": {
                "imem_size": 65536,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
            },
        },
    },
    "A10": {
        "name": "A10",
        "arch": "ampere",
        "pmu_reset_in_pmc": False,
        "memory_clear_supported": True,

        # Forcing ECC on after reset is not supported in the 94.02.5C.00.04
        # VBIOS. Adding support is pending.
        "forcing_ecc_on_after_reset_supported": False,

        "nvdec": [0, 1],
        "nvenc": [0],
        "other_falcons": ["sec", "gsp"],
        "falcons_cfg": {
            "pmu": {
                "imem_size": 147456,
                "dmem_size": 180224,
                "imem_port_count": 1,
                "dmem_port_count": 4,
                "default_core_falcon": False,
                "can_run_ns": False,
            },
            "nvdec0": {
                "imem_size": 49152,
                "dmem_size": 40960,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": True,
                "can_run_ns": False,
            },
            "nvdec1": {
                "imem_size": 49152,
                "dmem_size": 40960,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": True,
                "can_run_ns": False,
            },
            "nvenc0": {
                "imem_size": 32768,
                "dmem_size": 32768,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": True,
                "can_run_ns": False,
            },
            "gsp": {
                "imem_size": 90112,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": False,
                "can_run_ns": False,
            },
            "sec": {
                "imem_size": 90112,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": False,
                "can_run_ns": False,
            },
        },
    },
    "A10G": {
        "name": "A10G",
        "arch": "ampere",
        "pmu_reset_in_pmc": False,
        "memory_clear_supported": True,
        "forcing_ecc_on_after_reset_supported": True,
        "nvdec": [0, 1],
        "nvenc": [0],
        "other_falcons": ["sec", "gsp"],
        "falcons_cfg": {
            "pmu": {
                "imem_size": 147456,
                "dmem_size": 180224,
                "imem_port_count": 1,
                "dmem_port_count": 4,
                "default_core_falcon": False,
                "can_run_ns": False,
            },
            "nvdec0": {
                "imem_size": 49152,
                "dmem_size": 40960,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": True,
                "can_run_ns": False,
            },
            "nvdec1": {
                "imem_size": 49152,
                "dmem_size": 40960,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": True,
                "can_run_ns": False,
            },
            "nvenc0": {
                "imem_size": 32768,
                "dmem_size": 32768,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": True,
                "can_run_ns": False,
            },
            "gsp": {
                "imem_size": 90112,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": False,
                "can_run_ns": False,
            },
            "sec": {
                "imem_size": 90112,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": False,
                "can_run_ns": False,
            },
        },
    },

    0xb77000a1: {
        "name": "A16",
        "arch": "ampere",
        "pmu_reset_in_pmc": False,
        "memory_clear_supported": True,
        "forcing_ecc_on_after_reset_supported": True,
        "nvdec": [],
        "nvenc": [],
        "other_falcons": ["sec", "gsp"],
        "falcons_cfg": {
            "pmu": {
                "imem_size": 147456,
                "dmem_size": 180224,
                "imem_port_count": 1,
                "dmem_port_count": 4,
                "default_core_falcon": False,
                "can_run_ns": False,
            },
            "gsp": {
                "imem_size": 90112,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": False,
                "can_run_ns": False,
            },
            "sec": {
                "imem_size": 90112,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": False,
                "can_run_ns": False,
            },
        }
    },

    0x194000a1: {
        "name": "L4",
        "arch": "ada",
        "pmu_reset_in_pmc": False,
        "memory_clear_supported": True,
        "forcing_ecc_on_after_reset_supported": True,
        "nvdec": [0, 1],
        "nvenc": [0],
        "other_falcons": ["sec"],
        "falcons_cfg": {
            "pmu": {
                "imem_size": 147456,
                "dmem_size": 180224,
                "imem_port_count": 1,
                "dmem_port_count": 4,
                "default_core_falcon": False,
                "can_run_ns": False,
            },
            "nvdec0": {
                "imem_size": 49152,
                "dmem_size": 40960,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": True,
                "can_run_ns": False,
            },
            "nvdec1": {
                "imem_size": 49152,
                "dmem_size": 40960,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": True,
                "can_run_ns": False,
            },
            "nvenc0": {
                "imem_size": 32768,
                "dmem_size": 32768,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": True,
                "can_run_ns": False,
            },
            "sec": {
                "imem_size": 90112,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": False,
                "can_run_ns": False,
            },
        },
    },

    0x192000a1: {
        "name": "L40S",
        "arch": "ada",
        "pmu_reset_in_pmc": False,
        "memory_clear_supported": True,
        "forcing_ecc_on_after_reset_supported": True,
        "nvdec": [0, 1],
        "nvenc": [0],
        "other_falcons": ["sec"],
        "falcons_cfg": {
            "pmu": {
                "imem_size": 147456,
                "dmem_size": 180224,
                "imem_port_count": 1,
                "dmem_port_count": 4,
                "default_core_falcon": False,
                "can_run_ns": False,
            },
            "nvdec0": {
                "imem_size": 49152,
                "dmem_size": 40960,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": True,
                "can_run_ns": False,
            },
            "nvdec1": {
                "imem_size": 49152,
                "dmem_size": 40960,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": True,
                "can_run_ns": False,
            },
            "nvenc0": {
                "imem_size": 32768,
                "dmem_size": 32768,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": True,
                "can_run_ns": False,
            },
            "sec": {
                "imem_size": 90112,
                "dmem_size": 65536,
                "emem_size": 8192,
                "imem_port_count": 1,
                "dmem_port_count": 1,
                "default_core_falcon": False,
                "can_run_ns": False,
            },
        },
    },

    "H100-PCIE": {
        "name": "H100-PCIE",
        "arch": "hopper",
        "pmu_reset_in_pmc": False,
        "memory_clear_supported": True,
        "forcing_ecc_on_after_reset_supported": True,
        "nvdec": [],
        "nvenc": [],
        "other_falcons": ["fsp"],
        "nvlink": {
            "number": 18,
            "links_per_group": 6,
            "base_offset": 0xa00000,
            "per_group_offset": 0x40000,
        },
        "needs_falcons_cfg": False,
    },
    "H100-SXM": {
        "name": "H100-SXM",
        "arch": "hopper",
        "pmu_reset_in_pmc": False,
        "memory_clear_supported": True,
        "forcing_ecc_on_after_reset_supported": True,
        "nvdec": [],
        "nvenc": [],
        "other_falcons": ["fsp"],
        "nvlink": {
            "number": 18,
            "links_per_group": 6,
            "base_offset": 0xa00000,
            "per_group_offset": 0x40000,
        },
        "needs_falcons_cfg": False,
    },
    "H800-PCIE": {
        "name": "H800-PCIE",
        "arch": "hopper",
        "pmu_reset_in_pmc": False,
        "memory_clear_supported": True,
        "forcing_ecc_on_after_reset_supported": True,
        "nvdec": [],
        "nvenc": [],
        "other_falcons": ["fsp"],
        "nvlink": {
            "number": 18,
            "links_per_group": 6,
            "base_offset": 0xa00000,
            "per_group_offset": 0x40000,
        },
        "needs_falcons_cfg": False,
    },
    "H800-SXM": {
        "name": "H800-SXM",
        "arch": "hopper",
        "pmu_reset_in_pmc": False,
        "memory_clear_supported": True,
        "forcing_ecc_on_after_reset_supported": True,
        "nvdec": [],
        "nvenc": [],
        "other_falcons": ["fsp"],
        "nvlink": {
            "number": 18,
            "links_per_group": 6,
            "base_offset": 0xa00000,
            "per_group_offset": 0x40000,
        },
        "needs_falcons_cfg": False,
    },
}

if is_linux:
    import ctypes
    libc = ctypes.cdll.LoadLibrary('libc.so.6')

    # Set the mmap and munmap arg and return types.
    # last mmap arg is off_t which ctypes doesn't have. Assume it's long as that what gcc defines it to.
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mmap.restype = ctypes.c_void_p
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munmap.restype = ctypes.c_int

class RawBitfield(object):
    def __init__(self, value=0):
        self.value = value

    def __get_mask(self, key):
        if not isinstance(key, slice):
            raise TypeError("Wrong type for key {0}".format(type(key)))

        start, stop, stride = key.indices(32)
        if stride != 1:
            raise IndexError("Stride has to be 1, got {0}".format(stride))

        # slices have stop exclusive
        mask = (1 << (stop - start)) - 1
        mask <<= start
        return (mask, start)

    def __getitem__(self, key):
        mask, start = self.__get_mask(key)
        return (self.value & mask) >> start

    def __setitem__(self, key, bits):
        mask, start = self.__get_mask(key)
        if bits & ~(mask >> start) != 0:
            raise ValueError("Too many bits set for mask 0x{0:x} bits 0x{1:x}".format(mask >> start, bits))
        self.value = (self.value & ~mask) | (bits << start)

class GpuBitfield(RawBitfield):
    def __init__(self, gpu, offset, init_value=None, deferred=False):
        self.gpu = gpu
        self.offset = offset
        self.deferred = deferred
        if init_value != None:
            value = init_value
        else:
            value = self.gpu.read(self.offset)
        super(GpuBitfield, self).__init__(value)

    def __getitem__(self, key):
        self.value = self.gpu.read(self.offset)
        return super(GpuBitfield, self).__getitem__(key)

    def __setitem__(self, key, bits):
        super(GpuBitfield, self).__setitem__(key, bits)
        if not self.deferred:
            self.gpu.write(self.offset, self.value)

    def commit(self):
        self.gpu.write(self.offset, self.value)

class DeviceField(object):
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
        """Write to the device with only the field set as specified. Useful for W1C bits"""

        bf = self.bitfield_class(0)
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

DEVICES = { }

class Device(object):
    def __init__(self):
        self.parent = None
        self.children = []

    def is_hidden(self):
        return True

    def has_aer(self):
        return False

    def is_bridge(self):
        return False

    def is_root(self):
        return self.parent == None

    def is_gpu(self):
        return False

    def is_nvswitch(self):
        return False

    def is_plx(self):
        return False

    def is_intel(self):
        return False

    def has_dpc(self):
        return False

    def has_acs(self):
        return False

    def has_exp(self):
        return False

class PciDevice(Device):
    @staticmethod
    def _open_config(dev_path):
        dev_path_config = os.path.join(dev_path, "config")
        return FileRaw(dev_path_config, 0, os.path.getsize(dev_path_config))

    @staticmethod
    def find_class_for_device(dev_path):
        pci_dev = PciDevice(dev_path)
        if pci_dev.has_exp():
            # Root port
            if pci_dev.pciflags["TYPE"] == 0x4:
                if pci_dev.vendor == 0x8086:
                    return IntelRootPort
                return PciBridge

            # Upstream port
            if pci_dev.pciflags["TYPE"] == 0x5:
                # PlxBridge assumes full access to config space. If not full
                # config space is available, fall back to a regular PciBridge.
                if pci_dev.config.size >= 4096 and pci_dev.vendor == 0x10b5:
                    return PlxBridge
                return PciBridge

            # Downstream port
            if pci_dev.pciflags["TYPE"] == 0x6:
                if pci_dev.config.size >= 4096 and pci_dev.vendor == 0x10b5:
                    return PlxBridge
                return PciBridge

            # Endpoint
            if pci_dev.pciflags["TYPE"] == 0x0:
                if pci_dev.vendor == 0x10de:
                    return Gpu

        if pci_dev.header_type == 0x1:
            return PciBridge
        else:
            if pci_dev.vendor == 0x10de:
                return Gpu
            return PciDevice

    @staticmethod
    def init_dispatch(dev_path):
        cls = PciDevice.find_class_for_device(dev_path)
        if cls:
            return cls(dev_path)
        return None

    @staticmethod
    def find_or_init(dev_path):
        if dev_path == None:
            if -1 not in DEVICES:
                DEVICES[-1] = Device()
            return DEVICES[-1]
        bdf = os.path.basename(dev_path)
        if bdf in DEVICES:
            return DEVICES[bdf]
        dev = PciDevice.init_dispatch(dev_path)
        DEVICES[bdf] = dev
        return dev

    def _map_cfg_space(self):
        return self._open_config(self.dev_path)

    def __init__(self, dev_path):
        self.parent = None
        self.children = []
        self.dev_path = dev_path
        self.bdf = os.path.basename(dev_path)
        self.config = self._map_cfg_space()

        self.vendor = self.config.read16(0)
        self.device = self.config.read16(2)
        self.header_type = self.config.read8(0xe)
        self.cfg_space_broken = False
        self._init_caps()
        self._init_bars()
        if not self.cfg_space_broken:
            self.command = DeviceField(PciCommand, self.config, PCI_COMMAND)
            if self.has_exp():
                self.pciflags = DeviceField(PciExpFlags, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_FLAGS)
                self.devcap = DeviceField(PciDevCap, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_DEVCAP)
                self.devctl = DeviceField(PciDevCtl, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_DEVCTL)
                self.devctl2 = DeviceField(PciDevCtl2, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_DEVCTL2)
                self.link_cap = DeviceField(PciLinkCap, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_LNKCAP)
                self.link_ctl = DeviceField(PciLinkControl, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_LNKCTL)
                self.link_status = DeviceField(PciLinkStatus, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_LNKSTA)
                # Root port or downstream port
                if self.pciflags["TYPE"] == 0x4 or self.pciflags["TYPE"] == 0x6:
                    self.link_ctl_2 = DeviceField(PciLinkControl2, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_LNKCTL2)
                if self.pciflags["TYPE"] == 4:
                    self.rtctl = DeviceField(PciRootControl, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_RTCTL)
                if self.pciflags["SLOT"] == 1:
                    self.slot_ctl = DeviceField(PciSlotControl, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_SLTCTL)
                    self.slot_status = DeviceField(PciSlotStatus, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_SLTSTA)
            if self.has_aer():
                self.uncorr_status = DeviceField(PciUncorrectableErrors, self.config, self.ext_caps[PCI_EXT_CAP_ID_ERR] + PCI_ERR_UNCOR_STATUS, name="UNCOR_STATUS")
                self.uncorr_mask   = DeviceField(PciUncorrectableErrors, self.config, self.ext_caps[PCI_EXT_CAP_ID_ERR] + PCI_ERR_UNCOR_MASK, name="UNCOR_MASK")
                self.uncorr_sever  = DeviceField(PciUncorrectableErrors, self.config, self.ext_caps[PCI_EXT_CAP_ID_ERR] + PCI_ERR_UNCOR_SEVER, name="UNCOR_SEVER")
            if self.has_pm():
                self.pmctrl = DeviceField(PciPmControl, self.config, self.caps[PCI_CAP_ID_PM] + PCI_PM_CTRL)
            if self.has_acs():
                self.acs_ctl = DeviceField(AcsCtl, self.config, self.ext_caps[PCI_EXT_CAP_ID_ACS] + PCI_EXT_ACS_CTL)
            if self.has_dpc():
                self.dpc_ctrl   = DeviceField(DpcCtl, self.config, self.ext_caps[PCI_EXT_CAP_ID_DPC] + PCI_EXP_DPC_CTL)
                self.dpc_status = DeviceField(DpcStatus, self.config, self.ext_caps[PCI_EXT_CAP_ID_DPC] + PCI_EXP_DPC_STATUS)

        if is_sysfs_available:
            self.parent = PciDevice.find_or_init(sysfs_find_parent(dev_path))
        else:
            # Create a dummy device as the parent if sysfs is not available
            self.parent = Device()

    def _save_cfg_space(self):
        self.saved_cfg_space = {}
        for offset in GPU_CFG_SPACE_OFFSETS:
            if offset >= self.config.size:
                continue
            self.saved_cfg_space[offset] = self.config.read32(offset)
            #debug("%s saving cfg space %s = %s", self, hex(offset), hex(self.saved_cfg_space[offset]))

    def _restore_cfg_space(self):
        assert self.saved_cfg_space
        for offset in sorted(self.saved_cfg_space):
            old = self.config.read32(offset)
            new = self.saved_cfg_space[offset]
            #debug("%s restoring cfg space %s = %s to %s", self, hex(offset), hex(old), hex(new))
            self.config.write32(offset, new)

    def is_hidden(self):
        return False

    def has_aer(self):
        return PCI_EXT_CAP_ID_ERR in self.ext_caps

    def has_sriov(self):
        return PCI_EXT_CAP_ID_SRIOV in self.ext_caps

    def has_dpc(self):
        return PCI_EXT_CAP_ID_DPC in self.ext_caps

    def has_acs(self):
        return PCI_EXT_CAP_ID_ACS in self.ext_caps

    def has_exp(self):
        return PCI_CAP_ID_EXP in self.caps

    def has_pm(self):
        return PCI_CAP_ID_PM in self.caps

    def reinit(self):
        self.__init__(self.dev_path)

    def get_root_port(self):
        dev = self.parent
        while dev.parent != None and not dev.parent.is_hidden():
            dev = dev.parent
        return dev

    def get_first_plx_parent(self):
        dev = self.parent
        while dev != None:
            if dev.is_plx():
                return dev
            dev = dev.parent
        return None

    def _bar_num_to_sysfs_resource(self, barnum):
        sysfs_num = barnum
        # sysfs has gaps in case of 64-bit BARs
        for b in range(barnum):
            if self.bars[b][2]:
                sysfs_num += 1
        return sysfs_num

    def _init_bars_sysfs(self):
        self.bars = []
        resources = open(os.path.join(self.dev_path, "resource")).readlines()

        # Consider only first 6 resources
        for bar_line in resources[:6]:
            bar_line = bar_line.split(" ")
            addr = int(bar_line[0], base=16)
            end = int(bar_line[1], base=16)
            flags = int(bar_line[2], base=16)
            # Skip non-MMIO regions
            if flags & 0x1 != 0:
                continue
            if addr != 0:
                size = end - addr + 1
                is_64bit = False
                if (flags >> 1) & 0x3 == 0x2:
                    is_64bit = True
                self.bars.append((addr, size, is_64bit))

    def _bar_reg_mask(self, offset, high):
        all_1 = 0xffffffff
        org = self.config.read32(offset)
        self.config.write32(offset, all_1)
        value = self.config.read32(offset)
        self.config.write32(offset, org)
        if not high:
            value &= ~0xf
        return value

    def _bar_size_32(self, offset):
        return (~self._bar_reg_mask(offset, high=False) & (2**32 - 1)) + 1

    def _bar_size_64(self, offset):
        mask = self._bar_reg_mask(offset, high=False) | (self._bar_reg_mask(offset + 4, high=True) << 32)
        return (~mask & (2**64 - 1)) + 1

    def _init_bars_config_space(self):
        self.bars = []
        if self.header_type == 0x0:
            max_bars = 6
        else:
            max_bars = 2

        bar_num = 0
        while bar_num < max_bars:
            bar_reg = self.config.read32(0x10 + bar_num * 4)
            is_mmio = bar_reg & 0x1 == 0
            if not is_mmio:
                continue
            is_64bit = (bar_reg >> 1) & 0x3 == 0x2
            bar_addr = bar_reg & ~0xf
            if is_64bit:
                bar_addr |= self.config.read32(0x10 + (bar_num + 1) * 4) << 32
                bar_size = self._bar_size_64(0x10 + bar_num * 4)
                bar_num += 2
            else:
                bar_size = self._bar_size_32(0x10 + bar_num * 4)
                bar_num += 1
            if bar_addr != 0:
                self.bars.append((bar_addr, bar_size, is_64bit))

    def _init_bars(self):
        if is_sysfs_available:
            self._init_bars_sysfs()
        else:
            self._init_bars_config_space()

    def _map_bar(self, bar_num, bar_size=None):
        bar_addr = self.bars[bar_num][0]
        if not bar_size:
            bar_size = self.bars[bar_num][1]

        if mmio_access_type == "sysfs":
            return FileMap(os.path.join(self.dev_path, f"resource{self._bar_num_to_sysfs_resource(bar_num)}"), 0, bar_size)
        else:
            return FileMap("/dev/mem", bar_addr, bar_size)

    def _init_caps(self):
        self.caps = {}
        self.ext_caps = {}
        cap_offset = self.config.read8(PCI_CAPABILITY_LIST)
        data = 0
        if cap_offset == 0xff:
            self.cfg_space_broken = True
            error("Broken device %s", self.dev_path)
            return
        while cap_offset != 0:
            data = self.config.read32(cap_offset)
            cap_id = data & CAP_ID_MASK
            self.caps[cap_id] = cap_offset
            cap_offset = (data >> 8) & 0xff

        self._init_ext_caps()


    def _init_ext_caps(self):
        if self.config.size <= PCI_CFG_SPACE_SIZE:
            return

        offset = PCI_CFG_SPACE_SIZE
        header = self.config.read32(PCI_CFG_SPACE_SIZE)
        while offset != 0:
            cap = header & 0xffff
            self.ext_caps[cap] = offset

            offset = (header >> 20) & 0xffc
            header = self.config.read32(offset)

    def __str__(self):
        return "PCI %s %s:%s" % (self.bdf, hex(self.vendor), hex(self.device))

    def __hash__(self):
        return hash((self.bdf, self.vendor, self.device))

    def set_command_memory(self, enable):
        self.command["MEMORY"] = 1 if enable else 0

    def set_bus_master(self, enable):
        self.command["MASTER"] = 1 if enable else 0

    def cfg_read8(self, offset):
        return self.config.read8(offset)

    def cfg_read32(self, offset):
        return self.config.read32(offset)

    def cfg_write32(self, offset, data):
        self.config.write32(offset, data)

    def sanity_check_cfg_space(self):
        # Use an offset unlikely to be intercepted in case of virtualization
        vendor = self.config.read16(0xf0)
        return vendor != 0xffff

    def sanity_check_cfg_space_bars(self):
        """Check whether BAR0 is configured"""
        bar0 = self.config.read32(NV_XVE_BAR0)
        if bar0 == 0:
            return False
        if bar0 == 0xffffffff:
            return False
        return True

    def sysfs_power_control_get(self):
        path = os.path.join(self.dev_path, "power", "control")
        if not os.path.exists(path):
            debug(f"{self} path not present: '{path}'")
            return "not_present"
        return open(path, "r").readlines()[0].strip()

    def sysfs_power_control_set(self, mode):
        path = os.path.join(self.dev_path, "power", "control")
        if not os.path.exists(path):
            debug("%s path not present: '%s'", self, path)
            return
        with open(path, "w") as f:
            f.write(mode)

    def sysfs_remove(self):
        remove_path = os.path.join(self.dev_path, "remove")
        if not os.path.exists(remove_path):
            debug("%s remove not present: '%s'", self, remove_path)
        with open(remove_path, "w") as f:
            f.write("1")

    def sysfs_rescan(self):
        path = os.path.join(self.dev_path, "rescan")
        if not os.path.exists(path):
            debug("%s path not present: '%s'", self, path)
        with open(path, "w") as f:
            f.write("1")

    def sysfs_unbind(self):
        unbind_path = os.path.join(self.dev_path, "driver", "unbind")
        if not os.path.exists(unbind_path):
            debug("%s unbind not present: '%s', already unbound?", self, unbind_path)
            return
        with open(unbind_path, "w") as f:
            f.write(self.bdf)
        debug("%s unbind done", self)

    def sysfs_bind(self, driver):
        bind_path = os.path.join("/sys/bus/pci/drivers/", driver, "bind")
        if not os.path.exists(bind_path):
            debug("%s bind not present: '%s'", self, bind_path)
            return
        with open(bind_path, "w") as f:
            f.write(self.bdf)
        debug("%s bind to %s done", self, driver)

    def sysfs_reset(self):
        reset_path = os.path.join(self.dev_path, "reset")
        if not os.path.exists(reset_path):
            error("%s reset not present: '%s'", self, reset_path)
        with open(reset_path, "w") as rf:
            rf.write("1")

    def reset_with_os(self):
        if is_linux:
            return self.sysfs_reset()

        # For now fallback to a custom implementation on Windows
        if self.is_flr_supported():
            return self.reset_with_flr()
        return self.reset_with_sbr()

    def is_flr_supported(self):
        if not self.has_exp():
            return False

        return self.devcap["FLR"] == 1

PCI_BRIDGE_CONTROL = 0x3e
class PciBridgeControl(Bitfield):
    size = 1
    fields = {
            # Enable parity detection on secondary interface
            "PARITY": 0x01,

            # The same for SERR forwarding
            "SERR": 0x02,

            # Enable ISA mode
            "ISA": 0x04,

            # Forward VGA addresses
            "VGA": 0x08,

            # Report master aborts
            "MASTER_ABORT": 0x20,

            # Secondary bus reset (SBR)
            "BUS_RESET": 0x40,

            # Fast Back2Back enabled on secondary interface
            "FAST_BACK": 0x80,
    }

    def __str__(self):
        return "{ Bridge control " + str(self.values()) + " raw " + hex(self.raw) + " }"


class PciBridge(PciDevice):
    def __init__(self, dev_path):
        super(PciBridge, self).__init__(dev_path)
        self.bridge_ctl = DeviceField(PciBridgeControl, self.config, PCI_BRIDGE_CONTROL)
        if self.parent:
            self.parent.children.append(self)

    def is_bridge(self):
        return True

    def _set_link_disable(self, disable):
        self.link_ctl["LD"] = 1 if disable else 0
        debug("%s %s link disable, %s", self, "setting" if disable else "unsetting", self.link_ctl)

    def _set_sbr(self, reset):
        self.bridge_ctl["BUS_RESET"] = 1 if reset else 0
        debug("%s %s bus reset, %s",
              self, "setting" if reset else "unsetting", self.bridge_ctl)

    def toggle_link(self):
        self._set_link_disable(True)
        time.sleep(0.1)
        self._set_link_disable(False)
        time.sleep(0.1)

    def toggle_sbr(self, sleep_after=True):
        modified_slot_ctl = False
        if self.has_exp() and self.pciflags["SLOT"] == 1:
            saved_dll = self.slot_ctl["DLLSCE"]
            saved_hpie = self.slot_ctl["HPIE"]
            # Disable link state change notification
            self.slot_ctl["DLLSCE"] = 0
            self.slot_ctl["HPIE"] = 0
            modified_slot_ctl = True

        self._set_sbr(True)
        time.sleep(0.1)
        self._set_sbr(False)
        if sleep_after:
            time.sleep(0.3)

        if modified_slot_ctl:
            # Clear any pending interrupts for presence and link state
            self.slot_status["PDC"] = 1
            self.slot_status["DLLSC"] = 1

            self.slot_ctl["DLLSCE"] = saved_dll
            self.slot_ctl["HPIE"] = saved_hpie



class IntelRootPort(PciBridge):
    def __init__(self, dev_path):
        super(IntelRootPort, self).__init__(dev_path)

    def is_intel(self):
        return True

    def __str__(self):
        return "Intel root port %s" % self.bdf



class PlxBridge(PciBridge):
    def __init__(self, dev_path):
        super(PlxBridge, self).__init__(dev_path)


    def __str__(self):
        return "PLX %s" % self.bdf

    def is_plx(self):
        return True


NV_XVE_DEV_CTRL = 0x4
NV_XVE_BAR0 = 0x10
NV_XVE_BAR1_LO = 0x14
NV_XVE_BAR1_HI = 0x18
NV_XVE_BAR2_LO = 0x1c
NV_XVE_BAR2_HI = 0x20
NV_XVE_BAR3 = 0x24
NV_XVE_VCCAP_CTRL0 = 0x114

GPU_CFG_SPACE_OFFSETS = [
    NV_XVE_DEV_CTRL,
    NV_XVE_BAR0,
    NV_XVE_BAR1_LO,
    NV_XVE_BAR1_HI,
    NV_XVE_BAR2_LO,
    NV_XVE_BAR2_HI,
    NV_XVE_BAR3,
    NV_XVE_VCCAP_CTRL0,
]

class BrokenGpu(PciDevice):
    def __init__(self, dev_path):
        super(BrokenGpu, self).__init__(dev_path)
        self.name = "BrokenGpu"
        self.cfg_space_working = False
        self.bars_configured = False
        self.cfg_space_working = self.sanity_check_cfg_space()
        error("Config space working %s", str(self.cfg_space_working))
        if self.cfg_space_working:
            self.bars_configured = self.sanity_check_cfg_space_bars()

        if self.parent:
            self.parent.children.append(self)

    def is_gpu(self):
        return True

    def is_broken_gpu(self):
        return True

    def reset_with_sbr(self):
        assert self.parent.is_bridge()
        self.parent.toggle_sbr()
        return self.sanity_check_cfg_space()

    def is_driver_loaded(self):
        return False

    def __str__(self):
        return "GPU %s [broken, cfg space working %d bars configured %d]" % (self.bdf, self.cfg_space_working, self.bars_configured)

class NvidiaDeviceInternal:
    pass


class NvidiaDevice(PciDevice, NvidiaDeviceInternal):
    def __init__(self, dev_path):
        super(NvidiaDevice, self).__init__(dev_path)

        if self.has_pm():
            if is_linux:
                if self.pmctrl["STATE"] != 0:
                    prev_power_state = self.pmctrl["STATE"]
                    prev_power_control = self.sysfs_power_control_get()
                    if prev_power_control != "on":
                        import atexit

                        self.sysfs_power_control_set("on")
                        power_state = self.pmctrl["STATE"]
                        warning(f"{self} was in D{prev_power_state}, forced power control to on (prev {prev_power_control}). New state D{power_state}")
                        def restore_power():
                            warning(f"{self} restoring power control to {prev_power_control}")
                            self.sysfs_power_control_set(prev_power_control)

                        atexit.register(restore_power)

            if self.pmctrl["STATE"] != 0:
                warning("%s not in D0 (current state %d), forcing it to D0", self, self.pmctrl["STATE"])
                self.pmctrl["STATE"] = 0

        self.bar0_addr = self.bars[0][0]
        self.fsp_rpc = None
        self._mod_name = None

        if self.parent:
            self.parent.children.append(self)

    def common_init(self):
        self.nvlink = None
        if "nvlink" in self.props:
            self.nvlink = self.props["nvlink"]

    @property
    def is_nvlink_supported(self):
        return self.nvlink is not None

    @property
    def has_pdi(self):
        return False

    def is_gpu(self):
        return False

    def is_broken_gpu(self):
        return False

    def is_unknown(self):
        return True

    def reset_with_sbr(self):
        assert False

    def is_in_recovery(self):
        return False

    def write(self, reg, data):
        self.bar0.write32(reg, data)

    def write_verbose(self, reg, data):
        old = self.read(reg)
        self.bar0.write32(reg, data)
        new = self.read(reg)
        debug("%s writing %s = %s (old %s diff %s) new %s", self, hex(reg), hex(data), hex(old), hex(data ^ old), hex(new))

    def sanity_check(self):
        if not self.sanity_check_cfg_space():
            debug("%s sanity check of config space failed", self)
            return False

        boot = self.read_bad_ok(NV_PMC_BOOT_0)
        if boot == 0xffffffff:
            debug(f"{self} sanity check of mmio failed, 0x0 = 0x{boot:x}")
            return False
        if boot >> 16 == 0xbadf:
            debug(f"{self} sanity check of mmio failed, 0x0 = 0x{boot:x}")
            return False

        return True

    def reset_pre(self, reset_with_flr=None):
        if reset_with_flr == None:
            reset_with_flr = self.is_flr_supported()

        debug("%s reset_pre FLR supported %s, FLR being used %s", self, self.is_flr_supported(), reset_with_flr)

        self.expected_sbr_only_scratch = (1 if reset_with_flr else 0)

        flr_scratch = self.flr_resettable_scratch()
        sbr_scratch = self.sbr_resettable_scratch()

        self.write_verbose(flr_scratch, 0x1)
        self.write_verbose(sbr_scratch, 0x1)

        if self.read(sbr_scratch) == 0:
            debug(f"{self} SBR scratch writes not sticking")
            self.expected_sbr_only_scratch = 0

    def reset_post(self):
        flr_scratch = self.flr_resettable_scratch()
        sbr_scratch = self.sbr_resettable_scratch()

        debug(f"{self} reset_post flr-scratch after 0x{self.read_bad_ok(flr_scratch):x}, sbr-only scratch 0x{self.read_bad_ok(sbr_scratch):x}, flr cap {self.is_flr_supported()}")

    def reset_with_flr(self):
        assert self.is_flr_supported()

        self.reset_pre(reset_with_flr=True)

        debug("%s asserting FLR", self)
        self.devctl['BCR_FLR'] = 1
        time.sleep(.1)
        while True:
            crs = self.config.read32(0)
            debug("CRS 0x%x", crs)
            if crs != 0xffff0001:
                break
            time.sleep(.1)

        self._restore_cfg_space()
        self.set_command_memory(True)
        if not self.sanity_check():
            return False

        self.reset_post()

        return True

    def reset_with_sbr(self):
        self.reset_pre(reset_with_flr=False)

        assert self.parent.is_bridge()
        self.parent.toggle_sbr()

        self._restore_cfg_space()
        self.set_command_memory(True)
        if not self.sanity_check():
            return False

        self.reset_post()

        return True

    def sysfs_reset(self):
        self.reset_pre()

        super(NvidiaDevice, self).sysfs_reset()

        self.reset_post()

    def _init_fsp_rpc(self):
        if self.fsp_rpc != None:
            return

        # Wait for boot to be done such that FSP is available
        self.wait_for_boot()

        self.init_falcons()

        self.fsp_rpc = FspRpc(self.fsp, channel_num=2)

    def poll_register(self, name, offset, value, timeout, sleep_interval=0.01, mask=0xffffffff, debug_print=False):
        timestamp = perf_counter()
        while True:
            loop_stamp = perf_counter()
            try:
                if value >> 16 == 0xbadf:
                    reg = self.read_bad_ok(offset)
                else:
                    reg = self.read(offset)
            except:
                error("Failed to read falcon register %s (%s)", name, hex(offset))
                raise

            if reg & mask == value:
                if debug_print:
                    debug("Register %s (%s) = %s after %f secs", name, hex(offset), hex(value), perf_counter() - timestamp)
                return
            if loop_stamp - timestamp > timeout:
                raise GpuError("Timed out polling register %s (%s), value %s is not the expected %s. Timeout %f secs" % (name, hex(offset), hex(reg), hex(value), timeout))
            if sleep_interval > 0.0:
                time.sleep(sleep_interval)

    def poll_register_any_bit(self, name, offset, mask, timeout, sleep_interval=0.01, debug_print=False):
        timestamp = perf_counter()
        while True:
            loop_stamp = perf_counter()
            try:
                reg = self.read(offset)
            except:
                error("Failed to read register %s (%s)", name, hex(offset))
                raise

            if reg & mask != 0:
                if debug_print:
                    debug("Register %s (%s) = %s after %f secs", name, hex(offset), hex(reg), perf_counter() - timestamp)
                return
            if loop_stamp - timestamp > timeout:
                raise GpuError("Timed out polling register %s (%s), value %s & %s is still 0. Timeout %f secs" % (name, hex(offset), hex(reg), hex(mask), timeout))
            if sleep_interval > 0.0:
                time.sleep(sleep_interval)

    def get_pdi(self):
        assert self.has_pdi
        pdi = int(self.read(0x820344))
        pdi = int(self.read(0x820348)) << 32 | pdi
        return pdi


    def block_nvlinks(self, nvlinks):
        assert self.is_nvlink_supported

        if self.name == "A100":
            for nvlink in nvlinks:
                self.block_nvlink_a100(nvlink)
            return

        if self.has_fsp:
            self._init_fsp_rpc()
            self.fsp_rpc.prc_block_nvlinks(nvlinks, persistent=False)

    def bitfield(self, offset, init_value=None, deferred=False):
        return GpuBitfield(self, offset, init_value, deferred)


    def _nvlink_group_offset(self, group, reg=0):
        return self.nvlink["base_offset"] + group * self.nvlink["per_group_offset"] + reg

    def _nvlink_nvlipt_offset(self, group, reg=0):
        return self._nvlink_group_offset(group) + 0x2000 + reg

    def _nvlink_minion_offset(self, group, reg=0):
        return self._nvlink_group_offset(group) + 0x4000 + reg

    def _nvlink_link_offset(self, link, reg=0):
        group = link // self.nvlink["links_per_group"]
        local_link = link % self.nvlink["links_per_group"]
        return self._nvlink_group_offset(group) + 0x10000 + local_link * 0x8000 + reg

    def _nvlink_nvldl_offset(self, link, reg=0):
        return self._nvlink_link_offset(link) + 0x0 + reg

    def _nvlink_nvltlc_offset(self, link, reg=0):
        return self._nvlink_link_offset(link) + 0x5000 + reg

    def _nvlink_nvlipt_lnk_offset(self, link, reg=0):
        return self._nvlink_link_offset(link) + 0x7000 + reg

    def _nvlink_nport_top_offset(self, link, reg=0):
        return self._nvlink_link_offset(link) + 0x40000 + reg

    def _nvlink_offset_func(self, unit):
        if unit == "io_ctrl":
            return (lambda group, reg: self._nvlink_group_offset(group, reg))
        if unit == "minion":
            return (lambda group, reg: self._nvlink_minion_offset(group, reg))

    def _nvlink_query_enabled_links(self):
        self.nvlink_enabled_links = []
        groups = set()
        links = self.nvlink["number"]
        for link in range(links):
            offset = self._nvlink_nvlipt_lnk_offset(link, 0x600)
            data = self.read_bad_ok(offset)
            if data >> 16 == 0xbadf:
                continue
            if self.nvlink_get_link_state(link) == "disable":
                continue
            self.nvlink_enabled_links.append(link)
            groups.add(link // self.nvlink["links_per_group"])
        self.nvlink_enabled_groups = sorted(groups)
        return self.nvlink_enabled_links

    def nvlink_debug_nport(self):
        self._nvlink_query_enabled_links()
        nport_regs = [
            ("NV_INTERNAL", 0x00000054),
        ]

        for name, unit_offset in nport_regs:
            for link in self.nvlink_enabled_links:
                offset = self._nvlink_nport_top_offset(link, unit_offset)
                data = self.read_bad_ok(offset)
                data_2 = data >> 15
                debug(f"{self} link {link:2d} {name} 0x{offset:x} = 0x{data:x} 0x{data_2:x}")

    def nvlink_debug_nvlipt_lnk_basic_state(self):
        regs = [
            ("NV_INTERNAL", 0x00000280),
            ("NV_INTERNAL", 0x000000314),
            ("NV_INTERNAL", 0x00000484),
            ("NV_INTERNAL", 0x00000508),
            ("NV_INTERNAL", 0x00000090),
            ("NV_INTERNAL", 0x000000480),
            ("NV_INTERNAL", 0x0000004a0),
            ("NV_INTERNAL", 0x0000004a4),
            ("NV_INTERNAL", 0x0000004a8),
            ("NV_INTERNAL", 0x0000004ac),
            ("NV_INTERNAL", 0x00000600),
            ("NV_INTERNAL", 0x00000604),
            ("NV_INTERNAL", 0x00000608),
            ("NV_INTERNAL", 0x0000000c),
            ("NV_INTERNAL", 0x00000018),
            ("NV_INTERNAL", 0x00000000),
            ("NV_INTERNAL", 0x0000064c),
            ("NV_INTERNAL", 0x00000650),
            ("NV_INTERNAL", 0x00000654),
            ("NV_INTERNAL", 0x0000000c),
            ("NV_INTERNAL", 0x00000018),
            ("NV_INTERNAL", 0x00000000),
            ("NV_INTERNAL", 0x0000060c),
            ("NV_INTERNAL", 0x00000380),
        ]
        for reg in regs:
            for link in self.nvlink_enabled_links:
                name, unit_offset = reg
                offset = self._nvlink_nvlipt_lnk_offset(link, unit_offset)
                data = self.read_bad_ok(offset)
                debug(f"{self} link {link:2d} {name} 0x{unit_offset:x} 0x{offset:x} = 0x{data:x}")

    def nvlink_debug_nvltlc_basic_state(self):
        regs = [
            ("NV_INTERNAL", 0x00000280),
            ("NV_INTERNAL", 0x00000a80),
            ("NV_INTERNAL", 0x00001280),
            ("NV_INTERNAL", 0x000012a0),
            ("NV_INTERNAL", 0x00001a80),
            ("NV_INTERNAL", 0x00001aa0),
            ("NV_INTERNAL", 0x00001124),
            ("NV_INTERNAL", 0x00001904),
        ]
        for reg in regs:
            for link in self.nvlink_enabled_links:
                name, unit_offset = reg
                offset = self._nvlink_nvltlc_offset(link, unit_offset)
                data = self.read_bad_ok(offset)
                debug(f"{self} link {link:2d} {name} 0x{unit_offset:x} 0x{offset:x} = 0x{data:x}")

    def nvlink_debug_nvldl_basic_state(self):
        regs = [
            ("NV_INTERNAL", 0x00000000),
            ("NV_INTERNAL", 0x00000050),
            ("NV_INTERNAL", 0x00000054),
            ("NV_INTERNAL", 0x00000060),
            ("NV_INTERNAL", 0x00000070),
            ("NV_INTERNAL", 0x00000058),
            ("NV_INTERNAL", 0x0000008c),
            ("NV_INTERNAL", 0x00000290),
            ("NV_INTERNAL", 0x00003288),
            ("NV_INTERNAL", 0x0000329c),
            ("NV_INTERNAL", 0x00003294),
            ("NV_INTERNAL", 0x00003398),
            ("NV_INTERNAL", 0x0000339c),
            ("NV_INTERNAL", 0x00003290),
            ("NV_INTERNAL", 0x000033a8),
            ("NV_INTERNAL", 0x00003028),
            ("NV_INTERNAL", 0x00003050),
            ("NV_INTERNAL", 0x00002284),
            ("NV_INTERNAL", 0x00002288),
            ("NV_INTERNAL", 0x00002028),
            ("NV_INTERNAL", 0x00002398),
            ("NV_INTERNAL", 0x00002840),
            ("NV_INTERNAL", 0x000028a0),
            ("NV_INTERNAL", 0x00002900),
            ("NV_INTERNAL", 0x000000a0),
            ("NV_INTERNAL", 0x000000ac),
            ("NV_INTERNAL", 0x000000b0),
            ("NV_INTERNAL", 0x00002140),
            ("NV_INTERNAL", 0x00002148),
            ("NV_INTERNAL", 0x00002150),
            ("NV_INTERNAL", 0x00002154),
            ("NV_INTERNAL", 0x00002158),
            ("NV_INTERNAL", 0x0000215c),
            ("NV_INTERNAL", 0x00003140),
            ("NV_INTERNAL", 0x00003150),
            ("NV_INTERNAL", 0x00003154),
            ("NV_INTERNAL", 0x00003158),
            ("NV_INTERNAL", 0x0000315c),
        ]
        for reg in regs:
            for link in self.nvlink_enabled_links:
                name, nvldl_offset = reg
                offset = self._nvlink_nvldl_offset(link, nvldl_offset)
                data = self.read_bad_ok(offset)
                debug(f"{self} link {link:2d} {name} 0x{nvldl_offset:x} 0x{offset:x} = 0x{data:x}")

    def nvlink_debug_minion_basic_state(self):
        group_regs = [
            ("NV_INTERNAL", 0x00002830),
            ("NV_INTERNAL", 0x00002810),
            ("NV_INTERNAL", 0x2818)
        ]
        link_regs = [
            ("NV_INTERNAL", 0x00002a00),
        ]
        for g in self.nvlink_enabled_groups:
            for reg in group_regs:
                name, minion_offset = reg
                offset = self._nvlink_minion_offset(g, minion_offset)
                data = self.read_bad_ok(offset)
                debug(f"{self} minion {g:2d} {name} 0x{minion_offset:x} 0x{offset:x} = 0x{data:x}")

            for local_link in range(self.nvlink["links_per_group"]):
                for reg in link_regs:
                    name, minion_offset = reg
                    offset = self._nvlink_minion_offset(g, minion_offset) + 4 * local_link
                    data = self.read_bad_ok(offset)
                    debug(f"{self} minion {g:2d} link {local_link:2d} {name} 0x{minion_offset:x} 0x{offset:x} = 0x{data:x}")

    def nvlink_get_link_state(self, link):

        offset = self._nvlink_nvlipt_lnk_offset(link, 0x484)
        data = self.read_bad_ok(offset)
        if data >> 16 == 0xbadf:
            return "badf"

        state = data & 0xf
        states = {
            0x1: "active",
            0x2: "l2",
            0x5: "active_pending",
            0x8: "empty",
            0x9: "reset",
            0xd: "shutdown",
            0xe: "contain",
            0xf: "disable",
        }
        if state in states:
            return states[state]
        else:
            return str(state)

    def nvlink_get_link_states(self):
        self._nvlink_query_enabled_links()
        states = []
        for link in self.nvlink_enabled_links:
            states.append(self.nvlink_get_link_state(link))
        return states

    def nvlink_dl_get_link_state(self, link):
        offset = self._nvlink_nvldl_offset(link, 0)
        data = self.read_bad_ok(offset)
        if data >> 16 == 0xbadf:
            return "0xbadf"

        state = data & 0xff
        states = {
            0x0: "init",
            0xc: "hwpcfg",
            0x1: "hwcfg",
            0x2: "swcfg",
            0x3: "active",
            0x4: "fault",
            0x5: "sleep",
            0x8: "rcvy ac",
            0xa: "rcvy rx",
            0xb: "train",
            0xd: "test",
        }
        if state in states:
            return states[state]
        else:
            return str(state)

    def nvlink_dl_get_link_states(self):
        self._nvlink_query_enabled_links()
        states = []
        for link in self.nvlink_enabled_links:
            states.append(self.nvlink_dl_get_link_state(link))
        return states

    def nvlink_is_link_in_hs(self, link):
        link_state = self.nvlink_dl_get_link_state(link)
        return link_state == "active" or link_state == "sleep"

    def nvlink_get_links_in_hs(self):
        self._nvlink_query_enabled_links()
        links_in_hs = []
        for link in self.nvlink_enabled_links:
            if self.nvlink_is_link_in_hs(link):
                links_in_hs.append(link)
        return links_in_hs

    def nvlink_debug(self):
        from collections import Counter
        self.wait_for_boot()
        self._nvlink_query_enabled_links()
        links = self.nvlink_get_links_in_hs()
        link_states = Counter(self.nvlink_dl_get_link_states())
        info(f"{self} {self.module_name} trained {len(links)} links {links} dl link states {link_states}")
        topo = NVLINK_TOPOLOGY_HGX_8_H100
        for link in self.nvlink_enabled_links:
            peer_link, peer_name, _ = topo[self.module_name][link]
            info(f"{self} {self.module_name} link {link} -> {peer_name}:{peer_link} {self.nvlink_dl_get_link_state(link)} {self.nvlink_get_link_state(link)}")
        self.nvlink_debug_minion_basic_state()
        self.nvlink_debug_nvlipt_lnk_basic_state()
        self.nvlink_debug_nvltlc_basic_state()
        self.nvlink_debug_nvldl_basic_state()
        if self.is_nvswitch():
            self.nvlink_debug_nport()
            
    def detect_nvlink(self):
        from collections import Counter
        self.wait_for_boot()
        self._nvlink_query_enabled_links()
        links = self.nvlink_get_links_in_hs()
        #link_dl_states = Counter(self.nvlink_dl_get_link_states())
        link_states = Counter(self.nvlink_get_link_states())
        #tab = "\t"
        output_mapping = {
            "name": self.name,
            "dev_path": self.dev_path,
            "nvlink": {
                "count": len(links),
                "active": link_states['active']
            }
        }
        #print(f"NVLinks:{tab}{len(links)}\nActive:{tab*2}{link_states['active']}")
        print(json.dumps(output_mapping))
        # self.nvlink_debug_minion_basic_state()
        # self.nvlink_debug_nvlipt_lnk_basic_state()
        # self.nvlink_debug_nvltlc_basic_state()
        # self.nvlink_debug_nvldl_basic_state()
        # if self.is_nvswitch():
        #     self.nvlink_debug_nport()



    def __str__(self):
        return "Nvidia %s BAR0 0x%x devid %s" % (self.bdf, self.bar0_addr, hex(self.device))





class GpuMemPort(object):
    def __init__(self, name, mem_control_reg, max_size, falcon):
        self.name = name
        self.control_reg = mem_control_reg
        self.data_reg = self.control_reg + NV_PPWR_FALCON_IMEMD(0) - NV_PPWR_FALCON_IMEMC(0)
        self.offset = 0
        self.max_size = max_size
        self.auto_inc_read = False
        self.auto_inc_write = False
        self.secure_imem = False
        self.falcon = falcon
        self.need_to_write_config_to_hw = True

    def __str__(self):
        return "%s offset %d (0x%x) incr %d incw %d max size %d (0x%x) control reg 0x%x = 0x%x" % (self.name,
                self.offset, self.offset, self.auto_inc_read, self.auto_inc_write,
                self.max_size, self.max_size,
                self.control_reg, self.falcon.gpu.read(self.control_reg))

    def configure(self, offset, inc_read=True, inc_write=True, secure_imem=False):
        need_to_write = self.need_to_write_config_to_hw

        if offset != self.offset:
            self.offset = offset
            need_to_write = True

        if self.auto_inc_read != inc_read:
            self.auto_inc_read = inc_read
            need_to_write = True

        if self.auto_inc_write != inc_write:
            self.auto_inc_write = inc_write
            need_to_write = True

        if self.secure_imem != secure_imem:
            self.secure_imem = secure_imem
            need_to_write = True

        if not need_to_write:
            return

        memc_value = offset
        if inc_read:
            memc_value |= NV_PPWR_FALCON_IMEMC_AINCR_TRUE
        if inc_write:
            memc_value |= NV_PPWR_FALCON_IMEMC_AINCW_TRUE
        if secure_imem:
            memc_value |= NV_PPWR_FALCON_IMEMC_SECURE_ENABLED

        self.falcon.gpu.write(self.control_reg, memc_value)
        self.need_to_write_config_to_hw = False

    def handle_offset_wraparound(self):
        if self.offset == self.max_size:
            self.configure(0, self.auto_inc_read, self.auto_inc_write, self.secure_imem)

    def read(self, size):
        data = []
        for offset in range(0, size, 4):
            # MEM could match 0xbadf... so use read_bad_ok()
            data.append(self.falcon.gpu.read_bad_ok(self.data_reg))

        if self.auto_inc_read:
            self.offset += size

        self.handle_offset_wraparound()

        return data

    def write(self, data, debug_write=False):
        for d in data:
            if debug_write:
                control = self.falcon.gpu.read(self.control_reg)
                debug("Writing data %s = %s offset %s, control %s", hex(self.data_reg), hex(d), hex(self.offset), hex(control))
            self.falcon.gpu.write(self.data_reg, d)
            if self.auto_inc_write:
                self.offset += 4

        self.handle_offset_wraparound()

class GpuImemPort(GpuMemPort):
    def __init__(self, name, mem_control_reg, max_size, falcon):
        super(GpuImemPort, self).__init__(name, mem_control_reg, max_size, falcon)
        self.imemt_reg = self.control_reg + NV_PPWR_FALCON_IMEMT(0) - NV_PPWR_FALCON_IMEMC(0)

    def write_with_tags(self, data, virt_base, debug_write=False):
        for data_32 in data:
            if virt_base & 0xff == 0:
                if debug_write:
                    debug("Writing tag %s = %s offset %s", hex(self.imemt_reg), hex(virt_base), hex(self.offset))
                self.falcon.gpu.write(self.imemt_reg, virt_base >> 8)

            if debug_write:
                control = self.falcon.gpu.read(self.control_reg)
                debug("Writing data %s = %s offset %s, control %s", hex(self.data_reg), hex(data_32), hex(self.offset), hex(control))
            self.falcon.gpu.write(self.data_reg, data_32)

            virt_base += 4
            self.offset += 4

        while virt_base & 0xff != 0:
            self.falcon.gpu.write(self.data_reg, 0)
            virt_base += 4
            self.offset += 4

        self.handle_offset_wraparound()

class GpuFalcon(object):
    def __init__(self, name, cpuctl, device, pmc_enable_mask=None, pmc_device_enable_mask=None):
        self.name = name
        self.device = device
        self.gpu = device
        self.base_page = cpuctl & ~0xfff
        self.base_page_emem = getattr(self, 'base_page_emem', self.base_page)
        self.cpuctl = cpuctl
        self.pmc_enable_mask = pmc_enable_mask
        self.pmc_device_enable_mask = pmc_device_enable_mask
        self.no_outside_reset = getattr(self, 'no_outside_reset', False)
        self.has_emem = getattr(self, 'has_emem', False)
        self.num_emem_ports = getattr(self, 'num_emem_ports', 1)
        self._max_imem_size = None
        self._max_dmem_size = None
        self._max_emem_size = None
        self._imem_port_count = None
        self._dmem_port_count = None
        self._default_core_falcon = None
        self._can_run_ns = None

        self.csb_offset_mailbox0 = getattr(self, 'csb_offset_mailbox0', 0x40)

        self.mem_ports = []
        self.enable()
        self.mem_spaces = ["imem", "dmem"]

        self.imem_ports = []
        for p in range(0, self.imem_port_count):
            name = self.name + "_imem_%d" % p
            mem_control_reg = self.imemc + p * 16
            max_size = self.max_imem_size
            self.imem_ports.append(GpuImemPort(name, mem_control_reg, max_size, self))

        self.dmem_ports = []
        for p in range(0, self.dmem_port_count):
            name = self.name + "_dmem_%d" % p
            mem_control_reg = self.dmemc + p * 8
            max_size = self.max_dmem_size
            self.dmem_ports.append(GpuMemPort(name, mem_control_reg, max_size, self))

        self.emem_ports = []
        if self.has_emem:
            self.mem_spaces.append("emem")
            self._init_emem_ports()

        self.mem_ports = self.imem_ports + self.dmem_ports + self.emem_ports

    def _init_emem_ports(self):
        assert self.has_emem
        for p in range(self.num_emem_ports):
            name = self.name + f"_emem_{p}"
            self.emem_ports.append(GpuMemPort(name, self.base_page_emem + 0xac0 + p * 8, self.max_emem_size, self))

    @property
    def imemc(self):
        return self.cpuctl + NV_PPWR_FALCON_IMEMC(0) - NV_PPWR_FALCON_CPUCTL

    @property
    def dmemc(self):
        return self.cpuctl + NV_PPWR_FALCON_DMEMC(0) - NV_PPWR_FALCON_CPUCTL

    @property
    def bootvec(self):
        return self.cpuctl + NV_PPWR_FALCON_BOOTVEC - NV_PPWR_FALCON_CPUCTL

    @property
    def dmactl(self):
        return self.cpuctl + NV_PPWR_FALCON_DMACTL - NV_PPWR_FALCON_CPUCTL

    @property
    def engine_reset(self):
        return self.cpuctl + NV_PPWR_FALCON_ENGINE_RESET - NV_PPWR_FALCON_CPUCTL

    @property
    def hwcfg(self):
        return self.cpuctl + NV_PPWR_FALCON_HWCFG - NV_PPWR_FALCON_CPUCTL

    @property
    def hwcfg1(self):
        return self.cpuctl + NV_PPWR_FALCON_HWCFG1 - NV_PPWR_FALCON_CPUCTL

    @property
    def hwcfg_emem(self):
        return self.cpuctl + 0x9bc

    @property
    def dmemd(self):
        return self.dmemc + NV_PPWR_FALCON_IMEMD(0) - NV_PPWR_FALCON_IMEMC(0)

    @property
    def imemd(self):
        return self.imemc + NV_PPWR_FALCON_IMEMD(0) - NV_PPWR_FALCON_IMEMC(0)

    @property
    def imemt(self):
        return self.imemc + NV_PPWR_FALCON_IMEMT(0) - NV_PPWR_FALCON_IMEMC(0)

    @property
    def mailbox0(self):
        return self.base_page + 0x40

    @property
    def mailbox1(self):
        return self.base_page + 0x44

    @property
    def sctl(self):
        return self.base_page + 0x240

    @property
    def max_imem_size(self):
        if self._max_imem_size:
            return self._max_imem_size

        if self.name not in self.gpu.falcons_cfg:
            if self.gpu.needs_falcons_cfg:
                error("Missing imem/dmem config for falcon %s, falling back to hwcfg", self.name)
            self._max_imem_size = self.max_imem_size_from_hwcfg()
        else:
            # Use the imem size provided in the GPU config
            self._max_imem_size = self.gpu.falcons_cfg[self.name]["imem_size"]

        # And make sure it matches HW
        if self._max_imem_size != self.max_imem_size_from_hwcfg():
            raise GpuError("HWCFG imem doesn't match %d != %d" % (self._max_imem_size, self.max_imem_size_from_hwcfg()))

        return self._max_imem_size

    @property
    def max_dmem_size(self):
        if self._max_dmem_size:
            return self._max_dmem_size

        if self.name not in self.gpu.falcons_cfg:
            if self.gpu.needs_falcons_cfg:
                error("Missing imem/dmem config for falcon %s, falling back to hwcfg", self.name)
            self._max_dmem_size = self.max_dmem_size_from_hwcfg()
        else:
            # Use the dmem size provided in the GPU config
            self._max_dmem_size = self.gpu.falcons_cfg[self.name]["dmem_size"]

        # And make sure it matches HW
        if self._max_dmem_size != self.max_dmem_size_from_hwcfg():
            raise GpuError("HWCFG dmem doesn't match %d != %d" % (self._max_dmem_size, self.max_dmem_size_from_hwcfg()))

        return self._max_dmem_size

    @property
    def max_emem_size(self):
        if self._max_emem_size:
            return self._max_emem_size

        if self.name not in self.gpu.falcons_cfg or "emem_size" not in self.gpu.falcons_cfg[self.name]:
            if self.gpu.needs_falcons_cfg:
                error("Missing emem config for falcon %s, falling back to hwcfg", self.name)
            self._max_emem_size = self.max_emem_size_from_hwcfg()
        else:
            # Use the emem size provided in the GPU config
            self._max_emem_size = self.gpu.falcons_cfg[self.name]["emem_size"]

        # And make sure it matches HW
        if self._max_emem_size != self.max_emem_size_from_hwcfg():
            raise GpuError("HWCFG emem doesn't match %d != %d" % (self._max_emem_size, self.max_emem_size_from_hwcfg()))

        return self._max_emem_size

    @property
    def dmem_port_count(self):
        if self._dmem_port_count:
            return self._dmem_port_count

        if self.name not in self.gpu.falcons_cfg or "dmem_port_count" not in self.gpu.falcons_cfg[self.name]:
            if self.gpu.needs_falcons_cfg:
                error("%s missing dmem port count for falcon %s, falling back to hwcfg", self.gpu, self.name)
            self._dmem_port_count = self.dmem_port_count_from_hwcfg()
        else:
            # Use the dmem port count provided in the GPU config
            self._dmem_port_count = self.gpu.falcons_cfg[self.name]["dmem_port_count"]

        # And make sure it matches HW
        if self._dmem_port_count != self.dmem_port_count_from_hwcfg():
            raise GpuError("HWCFG dmem port count doesn't match %d != %d" % (self._dmem_port_count, self.dmem_port_count_from_hwcfg()))

        return self._dmem_port_count

    @property
    def imem_port_count(self):
        if self._imem_port_count:
            return self._imem_port_count

        if self.name not in self.gpu.falcons_cfg or "imem_port_count" not in self.gpu.falcons_cfg[self.name]:
            if self.gpu.needs_falcons_cfg:
                error("%s missing imem port count for falcon %s, falling back to hwcfg", self.gpu, self.name)
            self._imem_port_count = self.imem_port_count_from_hwcfg()
        else:
            # Use the imem port count provided in the GPU config
            self._imem_port_count = self.gpu.falcons_cfg[self.name]["imem_port_count"]

        # And make sure it matches HW
        if self._imem_port_count != self.imem_port_count_from_hwcfg():
            raise GpuError("HWCFG imem port count doesn't match %d != %d" % (self._imem_port_count, self.imem_port_count_from_hwcfg()))

        return self._imem_port_count

    @property
    def default_core_falcon(self):
        if self._default_core_falcon is not None:
            return self._default_core_falcon

        if self.name not in self.gpu.falcons_cfg or "default_core_falcon" not in self.gpu.falcons_cfg[self.name]:
            self._default_core_falcon = not self.gpu.has_fsp
        else:
            self._default_core_falcon = self.gpu.falcons_cfg[self.name]["default_core_falcon"]

        if not self._default_core_falcon and not self.supports_two_cores_from_hwcfg():
            raise GpuError("%s HWCFG two core suppport mismatch with defaulting to non falcon" % (self.name))

        return self._default_core_falcon

    @property
    def can_run_ns(self):
        if self._can_run_ns is not None:
            return self._can_run_ns

        if self.name not in self.gpu.falcons_cfg or "can_run_ns" not in self.gpu.falcons_cfg[self.name]:
            self._can_run_ns = True
        else:
            self._can_run_ns = self.gpu.falcons_cfg[self.name]["can_run_ns"]

        if self._can_run_ns and self.has_hs_boot():
            raise GpuError("%s incompatible properties, can run NS and HS boot" % (self.name))

        return self._can_run_ns

    def supports_two_cores_from_hwcfg(self):
        if not self.gpu.is_turing_plus:
            return False

        return self.gpu.read(self.base_page + 0xf4) & (0x1 << 10) != 0

    def has_hs_boot(self):
        if not self.gpu.is_turing_plus:
            return False

        return self.gpu.read(self.base_page + 0xf4) & (0x1 << 14) != 0

    def max_imem_size_from_hwcfg(self):
        if self.device.is_nvswitch() or self.gpu.is_ampere_plus:
            hwcfg = self.gpu.read(self.base_page + 0x278)
            return (hwcfg & 0xfff) * 256
        else:
            hwcfg = self.gpu.read(self.hwcfg)
            return (hwcfg & 0x1ff) * 256

    def max_dmem_size_from_hwcfg(self):
        if self.device.is_nvswitch() or self.gpu.is_ampere_plus:
            hwcfg = self.gpu.read(self.base_page + 0x278)
            return ((hwcfg >> 16) & 0xfff) * 256
        else:
            hwcfg = self.gpu.read(self.hwcfg)
            return ((hwcfg >> 9) & 0x1ff) * 256

    def max_emem_size_from_hwcfg(self):
        assert self.has_emem
        hwcfg = self.gpu.read(self.hwcfg_emem)
        return (hwcfg & 0x1ff) * 256

    def imem_port_count_from_hwcfg(self):
        hwcfg = self.gpu.read(self.hwcfg1)
        return ((hwcfg >> 8) & 0xf)

    def dmem_port_count_from_hwcfg(self):
        hwcfg = self.gpu.read(self.hwcfg1)
        return ((hwcfg >> 12) & 0xf)

    def get_mem_ports(self, mem):
        if mem == "imem":
            return self.imem_ports
        elif mem == "dmem":
            return self.dmem_ports
        elif mem == "emem":
            assert self.has_emem
            return self.emem_ports
        else:
            assert 0, "Unknown mem %s" % mem

    def get_mem_port(self, mem, port=0):
        return self.get_mem_ports(mem)[port]

    def load_imem(self, data, phys_base, virt_base, secure=False, virtual_tag=True, debug_load=False):
        self.imem_ports[0].configure(offset=phys_base, secure_imem=secure)
        if virtual_tag:
            self.imem_ports[0].write_with_tags(data, virt_base=virt_base, debug_write=debug_load)
        else:
            self.imem_ports[0].write(data, debug_write=debug_load)

    def read_port(self, port, phys_base, size):
        port.configure(offset=phys_base)
        return port.read(size)

    def write_port(self, port, data, phys_base, debug_write=False):
        port.configure(offset=phys_base)
        port.write(data, debug_write)

    def read_imem(self, phys_base, size):
        return self.read_port(self.imem_ports[0], phys_base, size)

    def load_dmem(self, data, phys_base, debug_load=False):
        self.write_port(self.dmem_ports[0], data, phys_base, debug_write=debug_load)

    def read_dmem(self, phys_base, size):
        return self.read_port(self.dmem_ports[0], phys_base, size)

    def write_emem(self, data, phys_base, port=0, debug_write=False):
        self.write_port(self.emem_ports[port], data, phys_base, debug_write=debug_write)

    def read_emem(self, phys_base, size, port=0):
        return self.read_port(self.emem_ports[port], phys_base, size)

    def execute(self, bootvec=0, wait=True):
        self.gpu.write(self.bootvec, bootvec)
        self.gpu.write(self.dmactl, 0)
        self.gpu.write(self.cpuctl, 2)
        if wait:
            self.wait_for_halt()

    def wait_for_halt(self, timeout=5, sleep_interval=0):
        self.gpu.poll_register(self.name + " cpuctl", self.cpuctl, 0x10, timeout=timeout, sleep_interval=sleep_interval)

    def wait_for_stop(self, timeout=0.001):
        self.gpu.poll_register(self.name + " cpuctl", self.cpuctl, 0x1 << 5, timeout, sleep_interval=0)

    def wait_for_start(self, timeout=0.001):
        self.gpu.poll_register(self.name + " cpuctl", self.cpuctl, 0, timeout, sleep_interval=0)

    def sreset(self, timeout=0.001):
        self.gpu.write(self.cpuctl, 0x1 << 2)
        # Falcon doesn't respond to PRI for a short time after reset, sleep for
        # a moment.
        time.sleep(0.000016)
        if not self.default_core_falcon:
            self.select_core_falcon()
        self.wait_for_halt(timeout, 0.01)
        self.reset_mem_ports()

    def disable(self):
        if self.no_outside_reset:
            # No outside reset means best we can do is halt.
            self.halt()
        elif self.pmc_enable_mask:
            pmc_enable = self.gpu.read(NV_PMC_ENABLE)
            self.gpu.write(NV_PMC_ENABLE, pmc_enable & ~self.pmc_enable_mask)
        elif self.pmc_device_enable_mask:
            enable = self.gpu.read(NV_PMC_DEVICE_ENABLE)
            self.gpu.write(NV_PMC_DEVICE_ENABLE, enable & ~self.pmc_device_enable_mask)
        else:
            self.gpu.write(self.engine_reset, 1)

    def halt(self, wait_for_halt=True):
        self.gpu.write(self.cpuctl, 0x1 << 3)
        # Falcon doesn't respond to PRI for a short time after halt, sleep for
        # a moment.
        time.sleep(0.000016)
        if wait_for_halt:
            self.wait_for_halt()

    def start(self, wait_for_start=True, timeout=0.001):
        self.gpu.write(self.cpuctl, 0x1 << 1)
        if wait_for_start:
            self.wait_for_start(timeout)

    def enable(self):
        if self.device.is_gpu() and self.device.is_ada:
            if self.device.read_bad_ok(self.base_page + 0x40c) == 0xbadf5620:
                debug(f"{self.name} resetting on enable()")
                self.gpu.write(self.engine_reset, 1)

        if self.no_outside_reset:
            pass
        elif self.pmc_enable_mask:
            pmc_enable = self.gpu.read(NV_PMC_ENABLE)
            self.gpu.write(NV_PMC_ENABLE, pmc_enable | self.pmc_enable_mask)
        elif self.pmc_device_enable_mask:
            enable = self.gpu.read(NV_PMC_DEVICE_ENABLE)
            self.gpu.write(NV_PMC_DEVICE_ENABLE, enable | self.pmc_device_enable_mask)
        else:
            self.gpu.write(self.engine_reset, 0)

        if not self.device.has_fsp:
            if not self.default_core_falcon:
                self.select_core_falcon()

            self.gpu.poll_register(self.name + " dmactl", self.dmactl, value=0, timeout=1, mask=0x6)
        self.reset_mem_ports()

    def reset_mem_ports(self):
        for m in self.mem_ports:
            m.need_to_write_config_to_hw = True

    def reset_raw(self):
        self.disable()
        if not self.is_disabled():
            raise GpuError("%s falcon %s not disabled during reset. Is reset protected?" % (self.gpu, self.name))
        self.enable()

    def reset(self):
        self.reset_raw()

    def is_halted(self):
        cpuctl = self.gpu.read(self.cpuctl)
        return cpuctl & 0x10 != 0

    def is_stopped(self):
        cpuctl = self.gpu.read(self.cpuctl)
        return cpuctl & (0x1 << 5) != 0

    def is_running(self):
        cpuctl = self.gpu.read(self.cpuctl)
        return cpuctl == 0

    def is_disabled(self):
        assert not self.no_outside_reset
        if self.pmc_enable_mask:
            pmc_enable = self.gpu.read(NV_PMC_ENABLE)
            return (pmc_enable & self.pmc_enable_mask) == 0
        elif self.pmc_device_enable_mask:
            enable = self.gpu.read(NV_PMC_DEVICE_ENABLE)
            return (enable & self.pmc_device_enable_mask) == 0
        else:
            return self.gpu.read(self.engine_reset) == 1

    def is_hsmode(self):
        return (self.gpu.read(self.sctl) & 0x2) != 0

    def _select_core(self, select_falcon):
        core_select = self.gpu.bitfield(self.base_page + 0x1668)
        core_select[4:5] = 0 if select_falcon else 1

    def select_core_falcon(self):
        self._select_core(True)

    def set_pkc(self, sig, engine, ucode):
        self.gpu.write(self.base2 + 0x210, sig)

        self.gpu.write(self.base2 + 0x19c, engine)

        self.gpu.write(self.base2 + 0x198, ucode)

        self.gpu.write(self.base2 + 0x180, 0x1)

class PmuFalcon(GpuFalcon):
    def __init__(self, gpu):
        if gpu.is_pmu_reset_in_pmc:
            pmc_enable_mask = NV_PMC_ENABLE_PWR
        else:
            pmc_enable_mask = None
        super(PmuFalcon, self).__init__("pmu", NV_PPWR_FALCON_CPUCTL, gpu, pmc_enable_mask=pmc_enable_mask)

    def reset(self):
        self.gpu.stop_preos()
        self.reset_raw()

    @property
    def fbif_ctl(self):
        return self.base_page + 0xe24

    @property
    def fbif_transcfg(self):
        return self.base_page + 0xe00

class MsvldFalcon(GpuFalcon):
    def __init__(self, gpu):
        pmc_enable_mask = NV_PMC_ENABLE_MSVLD

        self.csb_offset_mailbox0 = 0x1000

        super(MsvldFalcon, self).__init__("msvld", NV_PMSVLD_FALCON_CPUCTL, gpu, pmc_enable_mask=pmc_enable_mask)

class MspppFalcon(GpuFalcon):
    def __init__(self, gpu):
        pmc_enable_mask = NV_PMC_ENABLE_MSPPP

        self.csb_offset_mailbox0 = 0x1000

        super(MspppFalcon, self).__init__("msppp", NV_PMSPPP_FALCON_CPUCTL, gpu, pmc_enable_mask=pmc_enable_mask)

class MspdecFalcon(GpuFalcon):
    def __init__(self, gpu):
        pmc_enable_mask = NV_PMC_ENABLE_MSPDEC

        self.csb_offset_mailbox0 = 0x1000

        super(MspdecFalcon, self).__init__("mspdec", NV_PMSPDEC_FALCON_CPUCTL, gpu, pmc_enable_mask=pmc_enable_mask)

class MsencFalcon(GpuFalcon):
    def __init__(self, gpu):
        pmc_enable_mask = NV_PMC_ENABLE_MSENC
        super(MsencFalcon, self).__init__("msenc", NV_PMSENC_FALCON_CPUCTL, gpu, pmc_enable_mask=pmc_enable_mask)

class HdaFalcon(GpuFalcon):
    def __init__(self, gpu):
        self.no_outside_reset = True
        super(HdaFalcon, self).__init__("hda", NV_PHDAFALCON_FALCON_CPUCTL, gpu)

class DispFalcon(GpuFalcon):
    def __init__(self, gpu):
        pmc_enable_mask = NV_PMC_ENABLE_PDISP

        super(DispFalcon, self).__init__("disp", NV_PDISP_FALCON_CPUCTL, gpu, pmc_enable_mask=pmc_enable_mask)

        self.csb_offset_mailbox0 = self.mailbox0

class GspFalcon(GpuFalcon):
    def __init__(self, gpu):
        if gpu.is_volta_plus:
            self.has_emem = True

        self.csb_offset_mailbox0 = 0x1000

        super(GspFalcon, self).__init__("gsp", NV_PGSP_FALCON_CPUCTL, gpu)

    @property
    def fbif_ctl(self):
        return self.base_page + 0x624

    @property
    def fbif_transcfg(self):
        return self.base_page + 0x600

class SecFalcon(GpuFalcon):
    def __init__(self, gpu):
        if gpu.is_turing_plus:
            psec_cpuctl = NV_PSEC_FALCON_CPUCTL_TURING
        else:
            psec_cpuctl = NV_PSEC_FALCON_CPUCTL_MAXWELL

        if gpu.arch == "maxwell" or gpu.name == "P100":
            pmc_enable_mask = NV_PMC_ENABLE_SEC
        else:
            pmc_enable_mask = None

        if gpu.is_pascal_10x_plus:
            self.has_emem = True

        if gpu.is_ampere_10x_plus:
            self.base2 = 0x841000

        self.csb_offset_mailbox0 = 0x1000

        super(SecFalcon, self).__init__("sec", psec_cpuctl, gpu, pmc_enable_mask=pmc_enable_mask)

    @property
    def fbif_ctl(self):
        return self.base_page + 0x624

    @property
    def fbif_transcfg(self):
        return self.base_page + 0x600

    def read_ucode_version(self, ucode):
        return self.gpu.read(self.base_page + 0x11a8 + 4 * ucode)

class FbFalcon(GpuFalcon):
    def __init__(self, gpu):

        self.no_outside_reset = True
        self.csb_offset_mailbox0 = 0x9a4040

        super(FbFalcon, self).__init__("fb", NV_PFBFALCON_FALCON_CPUCTL, gpu)

class NvDecFalcon(GpuFalcon):
    def __init__(self, gpu, nvdec=0):
        if gpu.is_ampere_plus:
            cpuctl = NV_PNVDEC_FALCON_CPUCTL_AMPERE(nvdec)
        elif gpu.is_turing_plus:
            cpuctl = NV_PNVDEC_FALCON_CPUCTL_TURING(nvdec)
        else:
            cpuctl = NV_PNVDEC_FALCON_CPUCTL_MAXWELL(nvdec)

        if gpu.is_ampere_plus:
            pmc_mask = None
            pmc_device_mask = {
                    0: 0x1 << 15,
                    1: 0x1 << 16,
                    2: 0x1 << 20,
                    3: 0x1 << 4,
                    4: 0x1 << 5
                }[nvdec]
        else:
            pmc_mask = NV_PMC_ENABLE_NVDEC(nvdec)
            pmc_device_mask = None

        if gpu.is_ampere_10x_plus:
            self.base2 = [0x849c00, 0x84dc00][nvdec]

        self.csb_offset_mailbox0 = 0x1000

        super(NvDecFalcon, self).__init__("nvdec%s" % nvdec, cpuctl, gpu, pmc_enable_mask=pmc_mask, pmc_device_enable_mask=pmc_device_mask)

class NvEncFalcon(GpuFalcon):
    def __init__(self, gpu, nvenc=0):

        if gpu.is_ampere_plus:
            pmc_mask = None
            pmc_device_mask = {
                    0: 0x1 << 18,
                }[nvenc]
        else:
            pmc_mask = NV_PMC_ENABLE_NVENC(nvenc)
            pmc_device_mask = None

        super(NvEncFalcon, self).__init__("nvenc%s" % nvenc, NV_PNVENC_FALCON_CPUCTL(nvenc), gpu, pmc_enable_mask=pmc_mask, pmc_device_enable_mask=pmc_device_mask)

class MinionFalcon(GpuFalcon):
    def __init__(self, gpu):
        pmc_mask = NV_PMC_ENABLE_NVLINK
        super(MinionFalcon, self).__init__("minion", NV_PMINION_FALCON_CPUCTL, gpu, pmc_enable_mask=pmc_mask)

class SoeFalcon(GpuFalcon):
    def __init__(self, nvswitch):
        self.has_emem = True
        self.csb_offset_mailbox0 = 0x1000
        super(SoeFalcon, self).__init__("soe", 0x840100, nvswitch, pmc_enable_mask=None)

class FspFalcon(GpuFalcon):
    def __init__(self, device):
        self.no_outside_reset = True
        self.has_emem = True
        self.base_page_emem = 0x8f2000
        self.num_emem_ports = 8
        super(FspFalcon, self).__init__("fsp", 0x8f0100, device, pmc_enable_mask=None)

    def queue_head_off(self, i):
        return self.base_page + 0x2c00 + i * 8

    def queue_tail_off(self, i):
        return self.base_page + 0x2c04 + i * 8

    def msg_queue_head_off(self, i):
        return self.base_page + 0x2c80 + i * 8

    def msg_queue_tail_off(self, i):
        return self.base_page + 0x2c84 + i * 8

class FspRpc(object):
    def __init__(self, fsp_falcon, channel_num):
        self.falcon = fsp_falcon
        self.device = self.falcon.device
        self.channel_num = channel_num

        self.nvdm_emem_base = self.channel_num * 1024

        self.reset_rpc_state()

    def __str__(self):
        return f"{self.device} FSP-RPC"

    def reset_rpc_state(self):
        if self.is_queue_empty() and self.is_msg_queue_empty():
            debug(f"{self} both queues empty; queue {self.read_queue_state()} msg queue {self.read_msg_queue_state()}")
            return

        debug(f"{self} one of the queues not empty, waiting for things to settle; queue {self.read_queue_state()} msg queue {self.read_msg_queue_state()}")
        self.poll_for_msg_queue(timeout_fatal=False)
        debug(f"{self} after wait; queue {self.read_queue_state()} msg queue {self.read_msg_queue_state()}")

        # Reset both queues
        self.write_queue_head_tail(self.nvdm_emem_base, self.nvdm_emem_base)
        self.device.write_verbose(self.falcon.msg_queue_tail_off(self.channel_num), self.nvdm_emem_base)
        self.device.write_verbose(self.falcon.msg_queue_head_off(self.channel_num), self.nvdm_emem_base)

    def read_queue_state(self):
        return (self.device.read(self.falcon.queue_head_off(self.channel_num)),
                self.device.read(self.falcon.queue_tail_off(self.channel_num)))

    def is_queue_empty(self):
        mhead, mtail = self.read_queue_state()
        return mhead == mtail

    def write_queue_head_tail(self, head, tail):
        self.device.write_verbose(self.falcon.queue_tail_off(self.channel_num), tail)
        self.device.write_verbose(self.falcon.queue_head_off(self.channel_num), head)

    def read_msg_queue_state(self):
        return (self.device.read(self.falcon.msg_queue_head_off(self.channel_num)),
                self.device.read(self.falcon.msg_queue_tail_off(self.channel_num)))

    def is_msg_queue_empty(self):
        mhead, mtail = self.read_msg_queue_state()
        return mhead == mtail

    def write_msg_queue_tail(self, tail):
        self.device.write_verbose(self.falcon.msg_queue_tail_off(self.channel_num), tail)


    def poll_for_msg_queue(self, timeout=5, sleep_interval=0.01, timeout_fatal=True):
        timestamp = perf_counter()
        while True:
            loop_stamp = perf_counter()
            mhead, mtail = self.read_msg_queue_state()
            if mhead != mtail:
                return
            if loop_stamp - timestamp > timeout:
                if timeout_fatal:
                    raise GpuError(f"Timed out polling for {self.falcon.name} message queue on channel {self.channel_num}. head {mhead} == tail {mtail}")
                else:
                    return
            if sleep_interval > 0.0:
                time.sleep(sleep_interval)

    def poll_for_queue_empty(self, timeout=1, sleep_interval=0.01):
        timestamp = perf_counter()
        while True:
            loop_stamp = perf_counter()
            if self.is_queue_empty():
                return
            if loop_stamp - timestamp > timeout:
                mhead, mtail = self.read_queue_state()
                raise GpuError(f"Timed out polling for {self.falcon.name} cmd queue to be empty on channel {self.channel_num}. head {mhead} != tail {mtail}")
            if sleep_interval > 0.0:
                time.sleep(sleep_interval)

    def prc_cmd(self, data):
        mctp_header = MctpHeader()
        mctp_msg_header = MctpMessageHeader()

        mctp_msg_header.nvdm_type = 0x13


        self.device.wait_for_boot()

        self.poll_for_queue_empty()
        head, tail = self.read_queue_state()
        if head != tail:
            raise GpuError(f"RPC cmd queue not empty head {head} tail {tail}")
        mhead, mtail = self.read_msg_queue_state()
        if mhead != mtail:
            raise GpuError(f"RPC msg queue not empty head {mhead} tail {mtail}")

        cdata = mctp_header.to_int_array() + mctp_msg_header.to_int_array() + data
        debug(f"{self} command {[hex(d) for d in cdata]}")
        self.falcon.write_emem(cdata, phys_base=self.nvdm_emem_base, port=self.channel_num)
        self.write_queue_head_tail(self.nvdm_emem_base, self.nvdm_emem_base + (len(cdata) - 1) * 4)
        rpc_time = perf_counter()
        self.poll_for_msg_queue()
        rpc_time = perf_counter() - rpc_time
        debug(f"{self} response took {rpc_time*1000:.1f} ms")

        mhead, mtail = self.read_msg_queue_state()
        debug(f"{self} msg queue after poll {mhead} {mtail}")
        msize = mtail - mhead + 4
        mdata = self.falcon.read_emem(self.nvdm_emem_base, msize, port=self.channel_num)
        debug(f"{self} response {[hex(d) for d in mdata]}")

        # Reset the tail before checking for errors
        self.write_msg_queue_tail(mhead)

        if msize < 5 * 4:
            raise GpuError(f"{self} response size {msize} is smaller than expected. Data {[hex(d) for d in mdata]}")
        mctp_msg_header.from_int(mdata[1])
        if mctp_msg_header.nvdm_type != 0x15:
            raise GpuError(f"{self} message wrong nvdm_type. Data {[hex(d) for d in mdata]}")
        if mdata[3] != 0x13:
            raise GpuError(f"{self} message request type 0x{mdata[3]:x} not matching the command. Data {[hex(d) for d in mdata]}")
        if mdata[4] != 0x0:
            raise GpuError(f"{self} failed with error 0x{mdata[4]:x}. Data {[hex(d) for d in mdata]}")

        return mdata[5:]

    def prc_ecc(self, enable_ecc, persistent):
        # ECC is sub msg 0x1
        prc = 0x1
        if persistent:
            prc |= 0x3 << 8
        else:
            prc |= 0x1 << 8

        if enable_ecc:
            prc |= 0x1 << 16
        else:
            prc |= 0x0 << 16

        data = self.prc_cmd([prc])
        if len(data) != 0:
            raise GpuError(f"RPC wrong response size {len(data)}. Data {[hex(d) for d in data]}")

    def prc_block_nvlinks(self, nvlinks, persistent):
        # NVLINK config is sub msg 0xa
        prc = 0xa
        if persistent:
            prc |= 0x3 << 8
        else:
            prc |= 0x1 << 8

        # The mask is 64-bit
        nvlink_mask = 0
        for nvlink in nvlinks:
            nvlink_mask |= 1 << nvlink

        # First 2 bytes
        prc |= (nvlink_mask & 0xffff) << 16

        # Next 4 bytes
        prc_1 = (nvlink_mask >> 16) & 0xffffffff

        # Last 2 bytes
        prc_2 = (nvlink_mask >> 48)

        data = self.prc_cmd([prc, prc_1, prc_2])
        if len(data) != 0:
            raise GpuError(f"RPC wrong response size {len(data)}. Data {[hex(d) for d in data]}")

    def prc_knob_read(self, knob_id):
        # Knob read is sub msg 0xc
        prc = 0xc
        prc |= 0x2 << 8
        prc |= knob_id << 16

        knob_name = PrcKnob.str_from_knob_id(knob_id)

        debug(f"{self} reading knob {knob_name}")

        data = self.prc_cmd([prc])
        if len(data) != 1:
            raise GpuError(f"RPC wrong response size {len(data)}. Data {[hex(d) for d in data]}")

        # The knob value is 16-bits and the other extra 16-bits may not be 0-initialized.
        knob_value = data[0] & 0xffff

        debug(f"{self} read knob {knob_name} = 0x{knob_value:x}")

        return knob_value

    def prc_knob_write(self, knob_id, value):
        # Knob write is sub msg 0xd
        prc = 0xd
        prc |= 0x2 << 8
        prc |= knob_id << 16

        prc_1 = value

        knob_name = PrcKnob.str_from_knob_id(knob_id)

        debug(f"{self} writing knob {knob_name} = {value:#x}")

        data = self.prc_cmd([prc, prc_1])
        if len(data) != 0:
            raise GpuError(f"RPC wrong response size {len(data)}. Data {[hex(d) for d in data]}")

        debug(f"{self} wrote knob {knob_name} = {value:#x}")

    def prc_knob_check_and_write(self, knob_id, value):
        old_value = self.prc_knob_read(knob_id)
        if old_value != value:
            self.prc_knob_write(knob_id, value)


class UnknownDevice(Exception):
    pass

class UnknownGpuError(Exception):
    pass

class BrokenGpuError(Exception):
    pass

class GpuError(Exception):
    pass


class MctpHeader(NiceStruct):
    _fields_ = [
            ("version", "I", 4),
            ("rsvd0", "I", 4),
            ("deid", "I", 8),
            ("seid", "I", 8),
            ("tag", "I", 3),
            ("to", "I", 1),
            ("seq", "I", 2),
            ("eom", "I", 1),
            ("som", "I", 1),
        ]

    def __init__(self):
        super().__init__()

        self.som = 1
        self.eom = 1

class MctpMessageHeader(NiceStruct):
    _fields_ = [
            ("type", "I", 7),
            ("ic", "I", 1),
            ("vendor_id", "I", 16),
            ("nvdm_type", "I", 8),
    ]

    def __init__(self):
        super().__init__()

        self.type = 0x7e
        self.vendor_id = 0x10de

class NvSwitch(NvidiaDevice):
    def __init__(self, dev_path):
        self.name = "?"
        self.bar0_addr = 0

        super(NvSwitch, self).__init__(dev_path)

        if not self.sanity_check_cfg_space():
            debug("%s sanity check of config space failed", self)
            raise BrokenGpuError()

        # Enable MMIO
        self.set_command_memory(True)
        self.bar0_addr = self.bars[0][0]
        self.bar0_size = NVSWITCH_BAR0_SIZE
        self.bar0 = self._map_bar(0)

        self.pmcBoot0 = self.read(NV_PMC_BOOT_0)

        if self.pmcBoot0 == 0xffffffff:
            debug("%s sanity check of bar0 failed", self)
            raise BrokenGpuError()

        if self.pmcBoot0 not in NVSWITCH_MAP:
            for off in [0x0, 0x88000, 0x88004]:
                debug("%s offset 0x%x = 0x%x", self.bdf, off, self.read(off))
            raise UnknownGpuError("GPU %s %s bar0 %s" % (self.bdf, hex(self.pmcBoot0), hex(self.bar0_addr)))

        props = NVSWITCH_MAP[self.pmcBoot0]
        self.props = props
        self.name = props["name"]
        self.arch = props["arch"]
        #self.sanity_check()
        self._save_cfg_space()
        self.is_memory_clear_supported = False

        self.bios = None
        self.falcons = None
        self.falcon_dma_initialized = False
        self.falcons_cfg = props.get("falcons_cfg", {})
        self.needs_falcons_cfg = props.get("needs_falcons_cfg", {})

        self.common_init()

    def is_nvswitch(self):
        return True

    @property
    def is_laguna_plus(self):
        return NVSWITCH_ARCHES.index(self.arch) >= NVSWITCH_ARCHES.index("laguna")

    @property
    def has_fsp(self):
        return self.is_laguna_plus

    @property
    def has_pdi(self):
        return self.is_laguna_plus

    def _is_read_good(self, reg, data):
        return data >> 16 != 0xbadf

    def read_bad_ok(self, reg):
        data = self.bar0.read32(reg)
        return data

    def check_read(self, reg):
        data = self.bar0.read32(reg)
        return self._is_read_good(reg, data)

    def read(self, reg):
        data = self.bar0.read32(reg)
        if not self._is_read_good(reg, data):
            raise GpuError("gpu %s reg %s = %s, bad?" % (self, hex(reg), hex(data)))
        return data

    def is_broken_gpu(self):
        return False

    def is_driver_loaded(self):
        return False

    def dump_bar0(self):
        bar0_data = bytearray()
        for offset in range(0, self.bar0_size, 4):
            if offset % (128 * 1024) == 0:
                debug("Dumped %d bytes so far", offset)
            data = self.bar0.read32(offset)
            bar0_data.extend(data_from_int(data, 4))

        return bar0_data

    def flr_resettable_scratch(self):
        return 0xdfe0

    def sbr_resettable_scratch(self):
        if self.is_laguna_plus:
            return 0x91288
        return 0x88e10


    def init_falcons(self):
        if self.falcons is not None:
            return

        self.falcons = []

        if "soe" in self.props['other_falcons']:
            self.soe = SoeFalcon(self)
            self.falcons.append(self.soe)
        if "fsp" in self.props['other_falcons']:
            self.fsp = FspFalcon(self)
            self.falcons.append(self.fsp)

    def soe_stop_ucode(self):
        if self.soe.is_disabled():
            return

        if self.soe.is_halted():
            return

        soe_progress = self.read(0x28514)
        debug("{0} soe progress 0x{1:x}".format(self, soe_progress))
        if soe_progress & 0xf != 0x5:
            self.soe_stop_gfw_ucode()
        else:
            self.soe_stop_driver_ucode()

    def soe_stop_gfw_ucode(self):

        self.poll_register("soe_boot", 0x284ec, value=0x3ff, timeout=0.5)

        self.write_verbose(0x2851c, 0x1)
        self.soe.wait_for_halt(timeout=0.15)
        debug("%s stopped fw ucode", self)

    def soe_stop_driver_ucode(self):
        self.write_verbose(0x2851c, 0x1)

        self.poll_register("soe_ready_for_reset", 0x8403c4, value=0x1 << 4, mask=0x1 << 4, timeout=0.15)
        debug("%s stopped driver ucode", self)

    def is_in_recovery(self):
        if not self.is_laguna_plus:
            return False
        flags = self.read(0x66120)
        if flags != 0:
            debug(f"{self} boot flags 0x{flags:x}")
        return (flags >> 30) & 0x1 == 0x1

    def is_boot_done(self):
        assert self.is_laguna_plus
        if self.is_laguna_plus:
            data = self.read(0x660bc)
            if data == 0xff:
                return True
        return False

    def wait_for_boot(self):
        if self.is_laguna_plus:
            try:
                self.poll_register("boot_complete", 0x660bc, 0xff, 5)
            except GpuError as err:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)
                self.debug_dump()
                raise

    def read_module_id_ls10(self):
        gpios = [0x0, 0x1]

        mod_id = 0
        org_value = self.read(0xd740)
        for i, gpio in enumerate(gpios):
            self.write(0xd740, i)
            bit = (self.read(0xd740) >> 9) & 0x1
            mod_id |= bit << i
        self.write(0xd740, org_value)

        return mod_id

    def read_module_id(self):
        if self.is_laguna_plus:
            return self.read_module_id_ls10()
        else:
            raise GpuError(f"{self} unknown module id")

    @property
    def is_module_name_supported(self):
        return self.is_laguna_plus

    @property
    def module_name(self):
        if self._mod_name != None:
            return self._mod_name
        self._mod_name = f"NVSwitch_{self.read_module_id()}"
        return self._mod_name

    def debug_dump(self):
        offsets = []
        if self.is_laguna_plus:
            offsets.append(("boot_status", 0x660bc))
            offsets.append(("boot_flags", 0x66120))
            for i in range(16):
                offsets.append((f"sw_scratch_{i:02d}", 0x284e0 + i * 4))
            for i in range(4):
                offsets.append((f"fsp_scratch_{i}", 0x8f0320 + i * 4))
        for name, offset in offsets:
            data = self.read_bad_ok(offset)
            info(f"{self} BAR0 {name} 0x{offset:x} = 0x{data:x}")


    def __str__(self):
        return "NvSwitch %s %s %s BAR0 0x%x" % (self.bdf, self.name, hex(self.device), self.bar0_addr)

class PrcKnob(Enum):
    PRC_KNOB_ID_1                                   = 1

    PRC_KNOB_ID_2                                   = 2

    PRC_KNOB_ID_3                                   = 3

    PRC_KNOB_ID_4                                   = 4

    PRC_KNOB_ID_CCD_ALLOW_INB                       = 5
    PRC_KNOB_ID_CCD                                 = 6
    PRC_KNOB_ID_CCM_ALLOW_INB                       = 7
    PRC_KNOB_ID_CCM                                 = 8
    PRC_KNOB_ID_BAR0_DECOUPLER_ALLOW_INB            = 9
    PRC_KNOB_ID_BAR0_DECOUPLER                      = 10

    PRC_KNOB_ID_33                                  = 33

    PRC_KNOB_ID_34                                  = 34

    @classmethod
    def str_from_knob_id(cls, knob_id):
        try:
            prc_knob = PrcKnob(knob_id)
            knob_name = f"{prc_knob.name} "
        except ValueError:
            knob_name = ""

        knob_name += f"{knob_id} ({knob_id:#x})"
        return knob_name

class Gpu(NvidiaDevice):
    def __init__(self, dev_path):
        self.name = "?"
        self.bar0_addr = 0

        super(Gpu, self).__init__(dev_path)

        if not self.sanity_check_cfg_space():
            debug("%s sanity check of config space failed", self)
            raise BrokenGpuError()

        # Enable MMIO
        self.set_command_memory(True)

        self.bar0_addr = self.bars[0][0]
        self.bar0_size = GPU_BAR0_SIZE
        self.bar1_addr = self.bars[1][0]

        self.bar0 = self._map_bar(0)
        # Map just a small part of BAR1 as we don't need it all
        self.bar1 = self._map_bar(1, 1024 * 1024)

        self.pmcBoot0 = self.read(NV_PMC_BOOT_0)

        if self.pmcBoot0 == 0xffffffff:
            debug("%s sanity check of bar0 failed", self)
            raise BrokenGpuError()

        gpu_map_key = self.pmcBoot0

        if gpu_map_key in GPU_MAP_MULTIPLE:
            match = GPU_MAP_MULTIPLE[self.pmcBoot0]
            # Check for a device id match. Fall back to the default, if not found.
            gpu_map_key = GPU_MAP_MULTIPLE[self.pmcBoot0]["devids"].get(self.device, match["default"])

        if gpu_map_key not in GPU_MAP:
            for off in [0x0, 0x88000, 0x88004, 0x92000]:
                debug("%s offset 0x%x = 0x%x", self.bdf, off, self.read_bad_ok(off))
            raise UnknownGpuError("GPU %s %s bar0 %s" % (self.bdf, hex(self.pmcBoot0), hex(self.bar0_addr)))

        self.gpu_props = GPU_MAP[gpu_map_key]
        gpu_props = self.gpu_props
        self.props = gpu_props
        self.name = gpu_props["name"]
        self.arch = gpu_props["arch"]
        self.is_pmu_reset_in_pmc = gpu_props["pmu_reset_in_pmc"]
        self.is_memory_clear_supported = gpu_props["memory_clear_supported"]
        # Querying ECC state relies on being able to initialize/clear memory
        self.is_ecc_query_supported = self.is_memory_clear_supported
        self.is_cc_query_supported = self.is_hopper_plus
        self.is_forcing_ecc_on_after_reset_supported = gpu_props["forcing_ecc_on_after_reset_supported"]
        self.is_setting_ecc_after_reset_supported = self.is_ampere_plus
        self.is_mig_mode_supported = self.is_ampere_100
        if not self.sanity_check():
            debug("%s sanity check failed", self)
            raise BrokenGpuError()

        self._save_cfg_space()
        self.init_priv_ring()

        self.bar0_window_base = 0
        self.bar0_window_initialized = False
        self.bios = None
        self.falcons = None
        self.falcon_dma_initialized = False
        self.falcons_cfg = gpu_props.get("falcons_cfg", {})
        self.needs_falcons_cfg = gpu_props.get("needs_falcons_cfg", {})

        if self.is_ampere_plus:
            graphics_mask = 0
            graphics_bits = [12]
            if self.is_ampere_100:
                graphics_bits += [1, 9, 10, 11, 13, 14, 18]
            for gb in graphics_bits:
                graphics_mask |= (0x1 << gb)

            self.pmc_device_graphics_mask = graphics_mask
        self.hulk_ucode_data = None

        self.common_init()

    def init_falcons(self):
        if self.falcons is not None:
            return

        self.falcons = []
        gpu_props = self.gpu_props

        if not self.is_hopper_plus:
            self.pmu = PmuFalcon(self)
            self.falcons.append(self.pmu)

        if 0 in gpu_props["nvdec"]:
            self.nvdec = NvDecFalcon(self, 0)
            self.falcons.append(self.nvdec)

        for nvdec in gpu_props["nvdec"]:
            # nvdec 0 added above
            if nvdec == 0:
                continue
            self.falcons.append(NvDecFalcon(self, nvdec))
        for nvenc in gpu_props["nvenc"]:
            self.falcons.append(NvEncFalcon(self, nvenc))
        if "msvld" in gpu_props["other_falcons"]:
            self.falcons.append(MsvldFalcon(self))
        if "msppp" in gpu_props["other_falcons"]:
            self.falcons.append(MspppFalcon(self))
        if "msenc" in gpu_props["other_falcons"]:
            self.falcons.append(MsencFalcon(self))
        if "mspdec" in gpu_props["other_falcons"]:
            self.falcons.append(MspdecFalcon(self))
        if "hda" in gpu_props["other_falcons"]:
            self.falcons.append(HdaFalcon(self))
        if "disp" in gpu_props["other_falcons"]:
            self.falcons.append(DispFalcon(self))
        if "gsp" in gpu_props["other_falcons"]:
            self.gsp = GspFalcon(self)
            self.falcons.append(self.gsp)
        if "sec" in gpu_props["other_falcons"]:
            self.sec = SecFalcon(self)
            self.falcons.append(self.sec)
        if "fb" in gpu_props["other_falcons"]:
            self.falcons.append(FbFalcon(self))
        if "minion" in gpu_props["other_falcons"]:
            self.falcons.append(MinionFalcon(self))
        if "fsp" in gpu_props["other_falcons"]:
            self.fsp = FspFalcon(self)
            self.falcons.append(self.fsp)

    @property
    def is_maxwell_plus(self):
        return GPU_ARCHES.index(self.arch) >= GPU_ARCHES.index("maxwell")

    @property
    def is_pascal(self):
        return GPU_ARCHES.index(self.arch) == GPU_ARCHES.index("pascal")

    @property
    def is_pascal_plus(self):
        return GPU_ARCHES.index(self.arch) >= GPU_ARCHES.index("pascal")

    @property
    def is_pascal_10x_plus(self):
        return self.is_pascal_plus and self.name != "P100"

    @property
    def is_pascal_10x(self):
        return self.is_pascal and self.name != "P100"

    @property
    def is_volta(self):
        return GPU_ARCHES.index(self.arch) == GPU_ARCHES.index("volta")

    @property
    def is_volta_plus(self):
        return GPU_ARCHES.index(self.arch) >= GPU_ARCHES.index("volta")

    @property
    def is_turing(self):
        return GPU_ARCHES.index(self.arch) == GPU_ARCHES.index("turing")

    @property
    def is_turing_plus(self):
        return GPU_ARCHES.index(self.arch) >= GPU_ARCHES.index("turing")

    @property
    def is_ampere(self):
        return GPU_ARCHES.index(self.arch) == GPU_ARCHES.index("ampere")

    @property
    def is_ampere_plus(self):
        return GPU_ARCHES.index(self.arch) >= GPU_ARCHES.index("ampere")

    @property
    def is_ampere_100(self):
        return self.name in ["A100", "A30"]

    @property
    def is_ampere_10x(self):
        return self.is_ampere and not self.is_ampere_100

    @property
    def is_ampere_10x_plus(self):
        return self.is_ampere_plus and not self.is_ampere_100

    @property
    def is_ada(self):
        return GPU_ARCHES.index(self.arch) == GPU_ARCHES.index("ada")

    @property
    def is_ada_plus(self):
        return GPU_ARCHES.index(self.arch) >= GPU_ARCHES.index("ada")

    @property
    def is_hopper(self):
        return GPU_ARCHES.index(self.arch) == GPU_ARCHES.index("hopper")

    @property
    def is_hopper_plus(self):
        return GPU_ARCHES.index(self.arch) >= GPU_ARCHES.index("hopper")

    @property
    def is_hopper_100(self):
        return self.name in ["H100-PCIE", "H100-SXM"]

    @property
    def has_fsp(self):
        return self.is_hopper_plus

    @property
    def has_pdi(self):
        return self.is_ampere_plus

    def is_gpu(self):
        return True

    def is_in_recovery(self):
        if not self.is_hopper_plus:
            return False
        flags = self.read(0x20120)
        if flags != 0:
            debug(f"{self} boot flags 0x{flags:x}")
        return (flags >> 30) & 0x1 == 0x1

    @property
    def is_module_name_supported(self):
        return self.name == "H100-SXM"

    @property
    def module_name(self):
        if self._mod_name != None:
            return self._mod_name
        self._mod_name = f"SXM_{self.read_module_id() + 1}"
        return self._mod_name

    def vbios_scratch_register(self, index):
        if self.is_turing_plus:
            return 0x1400 + index * 4
        else:
            return 0x1580 + index * 4


    def _scrubber_status(self):
        assert self.is_turing_plus

        scratch = self.read(0x1180fc)
        return (scratch >> 29) & 0x7



    def is_broken_gpu(self):
        return False

    def reset_with_link_disable(self):
        self.parent.toggle_link()
        if not self.sanity_check_cfg_space():
            return False

        self._restore_cfg_space()

        # Enable MMIO
        self.set_command_memory(True)

        return self.sanity_check()

    def reset_with_sbr(self):
        status = super(Gpu, self).reset_with_sbr()
        if not status:
            return False

        # Reinit priv ring
        self.init_priv_ring()

        # Reinitialize falcons if they were already initialized
        if self.falcons:
            self.falcons = None
            self.init_falcons()

        return True

    def reset_with_flr(self):
        status = super(Gpu, self).reset_with_flr()
        if not status:
            return False

        # Reinit priv ring
        self.init_priv_ring()

        # Reinitialize falcons if they were already initialized
        if self.falcons:
            self.falcons = None
            self.init_falcons()

        return True

    def get_memory_size(self):
        config = self.read(0x100ce0)
        scale = config & 0xf
        mag = (config >> 4) & 0x3f
        size = mag << (scale + 20)
        ecc_tax = (config >> 30) & 1
        if ecc_tax:
            size = size * 15 // 16

        if self.is_ampere_100:
            row_remapped_size = int(self.read(0x1fa830)) - int(self.read(0x1fa82c))
            if row_remapped_size <= 0:
                raise GpuError("{0} is in an unexpected state. Is the driver loaded or VBIOS too old?".format(self))
            row_remapped_size += 16
            row_remapped_size *= 256
            debug("Size reserved for row remap %d", row_remapped_size)
            size -= row_remapped_size

        return size

    def get_ecc_state(self):
        if self.is_pascal_plus:
            config = self.read(0x9A0470)
        else:
            config = self.read(0x10F470)
        ecc_on = (config & 1) == 1
        return ecc_on

    def query_final_ecc_state(self):
        # To get the final ECC state, we need to wait for the memory to be
        # fully initialized. clear_memory() guarantees that.
        self.clear_memory()

        return self.get_ecc_state()

    def query_cc_mode(self):
        assert self.is_cc_query_supported
        self.wait_for_boot()

        cc_reg = self.read(0x1182cc)
        cc_state = cc_reg & 0x3
        if cc_state == 0x3:
            return "devtools"
        elif cc_state == 0x1:
            return "on"
        elif cc_state == 0x0:
            return "off"

        raise GpuError(f"Unexpected CC state 0x{cc_reg}")

    def set_cc_mode(self, mode):
        assert self.is_cc_query_supported

        cc_mode = 0x0
        cc_dev_mode = 0x0
        bar0_decoupler_val = 0x0
        if mode == "on":
            cc_mode = 0x1
            bar0_decoupler_val = 0x2
        elif mode == "devtools":
            cc_mode = 0x1
            cc_dev_mode = 0x1
        elif mode == "off":
            pass
        else:
            raise ValueError(f"Invalid mode {mode}")

        self._init_fsp_rpc()

        if cc_mode == 0x1:
            self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_2.value, 0x0)
            self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_4.value, 0x0)
            self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_34.value, 0x0)

        self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_BAR0_DECOUPLER.value, bar0_decoupler_val)
        self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_CCD.value, cc_dev_mode)
        self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_CCM.value, cc_mode)

    def query_cc_settings(self):
        assert self.is_cc_query_supported

        self._init_fsp_rpc()

        knobs = [
            ("enable", PrcKnob.PRC_KNOB_ID_CCM.value),
            ("enable-devtools", PrcKnob.PRC_KNOB_ID_CCD.value),
            ("enable-bar0-filter", PrcKnob.PRC_KNOB_ID_BAR0_DECOUPLER.value),

            ("enable-allow-inband-control", PrcKnob.PRC_KNOB_ID_CCM_ALLOW_INB.value),
            ("enable-devtools-allow-inband-control", PrcKnob.PRC_KNOB_ID_CCD_ALLOW_INB.value),
            ("enable-bar0-filter-allow-inband-control", PrcKnob.PRC_KNOB_ID_BAR0_DECOUPLER_ALLOW_INB.value),
        ]

        knob_state = []

        for name, knob_id in knobs:
            knob_value = self.fsp_rpc.prc_knob_read(knob_id)
            knob_state.append((name, knob_value))

        return knob_state

    def query_prc_knobs(self):
        assert self.has_fsp

        self._init_fsp_rpc()

        knob_state = []

        for knob in PrcKnob:
            knob_value = self.fsp_rpc.prc_knob_read(knob.value)
            knob_state.append((knob.name, knob_value))

        return knob_state


    def is_boot_done(self):
        assert self.is_turing_plus
        if self.is_hopper_plus:
            data = self.read(0x200bc)
            if data == 0xff:
                return True
        else:
            data = self.read(0x118234)
            if data == 0x3ff:
                return True
        return False

    def wait_for_boot(self):
        assert self.is_turing_plus
        if self.is_hopper_plus:
            try:
                self.poll_register("boot_complete", 0x200bc, 0xff, 5)
            except GpuError as err:
                _, _, tb = sys.exc_info()
                debug("{} boot not done 0x{:x} = 0x{:x}".format(self, 0x200bc, self.read(0x200bc)))
                for offset in range(0, 4*4, 4):
                    debug_offset = 0x8f0320 + offset
                    debug(" 0x{:x} = 0x{:x}".format(debug_offset, self.read(debug_offset)))
                traceback.print_tb(tb)
                raise
        else:
            self.poll_register("boot_complete", 0x118234, 0x3ff, 5)

    def clear_memory(self):
        assert self.is_memory_clear_supported

        self.wait_for_memory_clear()

    def wait_for_memory_clear(self):
        assert self.is_memory_clear_supported

        if self.is_turing_plus:
            # Turing does multiple clears asynchronously and we need to wait
            # for the last one to start first. Waiting for boot to be done does
            # that.
            self.wait_for_boot()

        self.poll_register("memory_clear_finished", 0x100b20, 0x1, 5)

    def force_ecc_on_after_reset_turing(self):
        assert self.is_turing

        scratch = self.read(0x118f78)
        self.write_verbose(0x118f78, scratch | 0x01000000)
        info("%s forced ECC to be enabled after next reset", self)

    def force_ecc_on_after_reset(self):
        assert self.is_forcing_ecc_on_after_reset_supported

        if self.is_ampere_plus:
            return self.set_ecc_mode_after_reset(enabled=True)
        else:
            return self.force_ecc_on_after_reset_turing()

    def _set_ecc_mode_after_reset_hopper(self, enabled):
        assert self.is_hopper_plus

        self._init_fsp_rpc()

        self.fsp_rpc.prc_ecc(enable_ecc=enabled, persistent=True)

        info("%s set ECC to be %s after next reset", self, "enabled" if enabled else "disabled")

    def set_ecc_mode_after_reset(self, enabled):
        assert self.is_ampere_plus

        if self.is_hopper_plus:
            return self._set_ecc_mode_after_reset_hopper(enabled)

        if self.is_ampere_10x_plus:
            scratch_offset = 0x118f08
        else:
            scratch_offset = 0x118f78

        scratch = self.bitfield(scratch_offset)
        scratch[12:14] = 3 if enabled else 2

        info("%s set ECC to be %s after next reset", self, "enabled" if enabled else "disabled")

    def set_mig_mode_after_reset(self, enabled):
        assert self.is_mig_mode_supported

        scratch = self.bitfield(0x118f78)
        scratch[14:16] = 3 if enabled else 2

        info("%s set MIG to be %s after next reset", self, "enabled" if enabled else "disabled")

    def query_mig_mode(self):
        assert self.is_mig_mode_supported

        # To get accurate MIG state, we need to wait for the boot to finish.
        self.wait_for_boot()

        mig_status = (self.read(self.vbios_scratch_register(1)) >> 13) & 0x7
        debug("%s MIG status 0x%x", self, mig_status)

        return mig_status & 0x4 == 0x4

    def test_mig_toggle(self):
        assert self.is_mig_mode_supported

        org_state = self.query_mig_mode()

        self.set_mig_mode_after_reset(not org_state)
        self.sysfs_reset()
        new_state = self.query_mig_mode()
        if org_state == new_state:
            raise GpuError("{0} MIG mode failed to switch from {1} to {2}".format(self, org_state, new_state))
        self.set_mig_mode_after_reset(org_state)
        self.sysfs_reset()
        new_state = self.query_mig_mode()
        if org_state != new_state:
            raise GpuError("{0} MIG mode failed to switch back to original state {1}".format(self, org_state))

    def test_ecc_toggle(self):
        org_state = self.query_final_ecc_state()

        if self.is_ampere_plus:
            self.set_ecc_mode_after_reset(not org_state)
            self.sysfs_reset()
            new_state = self.query_final_ecc_state()
            if org_state == new_state:
                raise GpuError("{0} ECC mode failed to switch from {1} to {2}".format(self, org_state, new_state))
            self.set_ecc_mode_after_reset(org_state)
            self.sysfs_reset()
            new_state = self.query_final_ecc_state()
            if org_state != new_state:
                raise GpuError("{0} ECC mode failed to switch back to original state {1}".format(self, org_state))
        else:
            self.force_ecc_on_after_reset()
            self.sysfs_reset()
            new_state = self.query_final_ecc_state()
            if not new_state:
                raise GpuError("{0} ECC mode failed to enable".format(self))

    def test_cc_mode_switch(self):
        org_mode = self.query_cc_mode()

        self._init_fsp_rpc()
        toggle_2 = self.fsp_rpc.prc_knob_read(PrcKnob.PRC_KNOB_ID_1.value) == 0x1
        toggle_4 = self.fsp_rpc.prc_knob_read(PrcKnob.PRC_KNOB_ID_3.value) == 0x1
        toggle_34 = self.fsp_rpc.prc_knob_read(PrcKnob.PRC_KNOB_ID_33.value) == 0x1
        info(f"{self} test CC switching org_mode {org_mode} toggle_2 {toggle_2} toggle_4 {toggle_4} toggle_34 {toggle_34}")

        prev_mode = org_mode

        for iter in range(5):
            for mode in ["devtools", "on", "off"]:
                debug(f"{self} switching CC to {mode} in iter {iter}")
                if toggle_2 and prev_mode != "on" and iter > 1:
                    self.fsp_rpc.prc_knob_write(PrcKnob.PRC_KNOB_ID_2.value, 0x1)
                if toggle_4 and prev_mode != "on" and iter > 2:
                    self.fsp_rpc.prc_knob_write(PrcKnob.PRC_KNOB_ID_4.value, 0x1)
                if toggle_34 and prev_mode != "on" and iter > 3:
                    self.fsp_rpc.prc_knob_write(PrcKnob.PRC_KNOB_ID_34.value, 0x1)

                self.set_cc_mode(mode)
                self.reset_with_os()
                new_mode = self.query_cc_mode()
                if new_mode != mode:
                    raise GpuError(f"{self} CC mode failed to switch to {mode} in iter {iter}. Current mode is {new_mode}")
                debug(f"{self} CC switched to {mode} in iter {iter}")
                prev_mode = new_mode

        self.set_cc_mode(org_mode)
        self.reset_with_os()

    def _init_bar0_window(self):
        self.set_bus_master(True)
        if self.is_turing_plus:
            cfg = self.read(0x100ed0)
            self.write(0x100ed0, cfg & ~0x4)

        if self.is_memory_clear_supported:
            # Leverage GPU init performed by clearing memory to speed up some of
            # the operations, if it's implemented.
            self.clear_memory()

        self.bar0_window_initialized = True

    def config_bar0_window(self, addr, sysmem=False):
        if not self.bar0_window_initialized:
            self._init_bar0_window()

        new_window = addr >> 16
        new_window &= ~0xf
        if sysmem:
            new_window |= NV_BAR0_WINDOW_CFG_TARGET_SYSMEM_COHERENT
        if self.bar0_window_base != new_window:
            self.bar0_window_base = new_window
            self.write(NV_BAR0_WINDOW_CFG, new_window)
        bar0_window_addr = NV_BAR0_WINDOW + (addr & 0xfffff)
        return bar0_window_addr

    def bar0_window_read(self, address, size):
        offset = self.config_bar0_window(address)
        return self.bar0.read(offset, size)

    def bar0_window_write32(self, address, data):
        offset = self.config_bar0_window(address)
        return self.bar0.write32(offset, data)

    def bar0_window_sys_read(self, address, size):
        offset = self.config_bar0_window(address, sysmem=True)
        return self.bar0.read(offset, size)

    def bar0_window_sys_read32(self, address):
        return self.bar0_window_sys_read(address, 4)

    def bar0_window_sys_write32(self, address, data):
        offset = self.config_bar0_window(address, sysmem=True)
        return self.bar0.write32(offset, data)

    def _falcon_dma(self, falcon, address, size, write, sysmem):
        cfg = self.bitfield(falcon.fbif_transcfg)

        cfg[2:3] = 0x1

        cfg[0:2] = 0x1 if sysmem else 0x0

        self.write(falcon.base_page + 0x684, 1)

        ctl = self.bitfield(falcon.fbif_ctl)
        ctl[4:5] = 1
        ctl[7:8] = 1

        dmactl = self.bitfield(falcon.dmactl)
        dmactl[0:1] = 0

        self.write(falcon.base_page + 0x110, (address >> 8) & 0xffffffff)
        self.write(falcon.base_page + 0x128, address >> 40)

        # GPU DMA supports 47-bits, but top bits can be globally forced. This
        # only works if there are no other DMAs happening at the same time.
        self.write(0x100f04, address >> 47)

        offset = address % 256
        if offset % size != 0:
            raise ValueError("DMA address needs to be aligned to size, address 0x{0:x} size 0x{1:x}".format(address, size))

        self.write(falcon.base_page + 0x11c, offset)
        self.write(falcon.base_page + 0x114, 0)

        dma_cmd = self.bitfield(falcon.base_page + 0x118, init_value=0, deferred=True)
        if write:
            dma_cmd[5:6] = 1

        sizes = {4:0, 8:1, 16:2, 32:3, 64:4, 128:5, 256:6}
        if size not in sizes:
            raise ValueError("Invalid size {0}".format(size))
        dma_cmd[8:11] = sizes[size]
        dma_cmd.commit()

        self.poll_register("dma done", falcon.base_page + 0x118, value=0x1 << 1, mask=0x1 << 1, timeout=0.1)

        self.write(0x100f04, 0)

    def _falcon_dma_init(self):
        self.init_falcons()
        if hasattr(self, "gsp"):
            falcon = self.gsp
        else:
            falcon = self.pmu

        if self.falcon_dma_initialized:
            return falcon

        self.set_bus_master(True)
        if self.is_memory_clear_supported:
            self.clear_memory()

        falcon.reset()

        self.falcon_dma_initialized = True
        return falcon

    def dma_sys_write(self, address, data):
        falcon = self._falcon_dma_init()
        dmem = falcon.load_dmem(data, phys_base=0)
        size = len(data) * 4
        self._falcon_dma(falcon, address, size, write=True, sysmem=True)

    def dma_sys_read(self, address, size):
        falcon = self._falcon_dma_init()

        self._falcon_dma(falcon, address, size, write=False, sysmem=True)
        dmem = falcon.read_dmem(phys_base=0, size=size)
        return dmem

    def dma_sys_write32(self, address, data):
        return self.dma_sys_write(address, [data])

    def dma_sys_read32(self, address):
        return self.dma_sys_read(address, 4)[0]

    def _is_read_good(self, reg, data):
        return data >> 16 != 0xbadf

    def read_bad_ok(self, reg):
        data = self.bar0.read32(reg)
        return data

    def check_read(self, reg):
        data = self.bar0.read32(reg)
        return self._is_read_good(reg, data)

    def read(self, reg):
        data = self.bar0.read32(reg)
        if not self._is_read_good(reg, data):
            raise GpuError("gpu %s reg %s = %s, bad?" % (self, hex(reg), hex(data)))
        return data

    def read_bar1(self, offset):
        return self.bar1.read32(offset)

    def write_bar1(self, offset, data):
        return self.bar1.write32(offset, data)

    def preos_erot_handshake_ada(self):
        assert self.is_ada
        erot_grant_reg = self.vbios_scratch_register(0)
        for retry in range(3):
            preos_scratch = self.read(self.vbios_scratch_register(0x6))
            data = self.read(erot_grant_reg)
            debug(f"{self} EROT handshake before {data:#x} preos scratch {preos_scratch:#x}")
            if (data >> 3) & 0x1 == 1:
                break


            if (preos_scratch >> 16 & 0xff) in [0x7, 0x8]:
                break

            time.sleep(.01)

        if (data >> 3) & 0x1 == 0:
            # No EROT
            return

        if (data >> 5) & 0x1 == 1:
            # Handshake already complete
            return

        self.write_verbose(erot_grant_reg, data | 0x1<<4)
        self.poll_register("ada_erot_handshake", erot_grant_reg, 0x1<<5, timeout=5, mask=0x1<<5)
        data = self.read(erot_grant_reg)
        debug(f"{self} EROT handshake after {data:#x}")

    def stop_preos(self):
        if self.is_hopper_plus:
            return

        if self.is_turing_plus:
            self.wait_for_boot()

        self.init_falcons()

        if self.pmu.is_disabled():
            return

        if self.pmu.is_halted():
            return

        if self.is_ada:
            self.preos_erot_handshake_ada()

        if self.is_turing_plus:
            # On Turing+, pre-OS ucode disables the offset applied to PROM_DATA
            # when requested through a scratch register. Do it always for simplicity.
            reg = self.vbios_scratch_register(1)
            reg_value = self.read(reg)
            reg_value |= (0x1 << 11)
            self.write(reg, reg_value)

        if self.is_maxwell_plus:
            preos_started_reg = self.vbios_scratch_register(6)
            start = perf_counter()
            preos_status = (self.read(preos_started_reg) >> 12) & 0xf
            while preos_status == 0:
                if perf_counter() - start > 5:
                    raise GpuError("Timed out waiting for preos to start %s" % hex(preos_status))
                preos_status = (self.read(preos_started_reg) >> 12) & 0xf

        if self.is_volta_plus:
            preos_stop_reg = self.vbios_scratch_register(1)
            data = self.read(preos_stop_reg)
            self.write(preos_stop_reg, data | 0x200)
            self.pmu.wait_for_halt()
        elif self.arch == "kepler":
            self.pmu.sreset()
        else:
            self.write(0x10a7bc, 0)

            self.poll_register("preos_idle", 0x10a984, 0x1<<4, timeout=0.001, mask=0x1<<4)

            self.pmu.reset_raw()
            self.write(0x10a7bc, 1)

    # Init priv ring (internal bus)
    def init_priv_ring(self):
        self.write(0x12004c, 0x4)
        self.write(0x122204, 0x2)

    # Reset priv ring (internal bus)
    def reset_priv_ring(self):
        pmc_enable = self.read(NV_PMC_ENABLE)
        self.write(NV_PMC_ENABLE, pmc_enable & ~NV_PMC_ENABLE_PRIV_RING)
        self.write(NV_PMC_ENABLE, pmc_enable | NV_PMC_ENABLE_PRIV_RING)
        self.init_priv_ring()

    def disable_pgraph(self):
        if self.is_ampere_plus:
            enable = self.read(NV_PMC_DEVICE_ENABLE)
            self.write(NV_PMC_DEVICE_ENABLE, enable & ~self.pmc_device_graphics_mask)
        else:
            pmc_enable = self.read(NV_PMC_ENABLE)
            self.write(NV_PMC_ENABLE, pmc_enable & ~NV_PMC_ENABLE_PGRAPH)

    def disable_perfmon(self):
        pmc_enable = self.read(NV_PMC_ENABLE)
        self.write(NV_PMC_ENABLE, pmc_enable & ~NV_PMC_ENABLE_PERFMON)

    def is_pgraph_disabled(self):
        if self.is_ampere_plus:
            enable = self.read(NV_PMC_DEVICE_ENABLE)
            return (enable & self.pmc_device_graphics_mask) == 0
        else:
            pmc_enable = self.read(NV_PMC_ENABLE)
            return (pmc_enable & NV_PMC_ENABLE_PGRAPH) == 0

    def is_driver_loaded(self):
        if self.is_hopper_plus:
            # TODO, for now assume no driver.
            return False

        if self.is_pascal_10x_plus and self.read(0x10a3c0) == 1:
            return False

        reg = self.read_bad_ok(0x10a080)

        # PMU might be disabled, assume driver is not being used in this case.
        if reg >> 16 == 0xbadf:
            return False

        # Otherwise any non-zero means the driver is loaded.
        return reg != 0

    def print_falcons(self):
        self.init_priv_ring()
        self.init_falcons()
        for falcon in self.falcons:
            falcon.enable()
        print("\"" + self.name + "\": {")
        for falcon in self.falcons:
            name = falcon.name
            print("    \"" + name + "\": {")
            print("        \"imem_size\": " + str(falcon.max_imem_size) + ",")
            print("        \"dmem_size\": " + str(falcon.max_dmem_size) + ",")
            if falcon.has_emem:
                print("        \"emem_size\": " + str(falcon.max_emem_size) + ",")
            print("        \"imem_port_count\": " + str(falcon.imem_port_count) + ",")
            print("        \"dmem_port_count\": " + str(falcon.dmem_port_count) + ",")
            print("    },")
        print("},")

    def flr_resettable_scratch(self):
        if self.is_volta_plus:
            return self.vbios_scratch_register(22)
        else:
            return self.vbios_scratch_register(15)

    def sbr_resettable_scratch(self):
        if self.is_hopper_plus:
            return 0x91288
        if self.is_ampere_plus:
            return 0x88e10
        return self.flr_resettable_scratch()


    def _nvlink_offset(self, link, reg=0):
        io_ctrl_base = 0xA00000
        per_io_ctrl_offset = 0x40000
        io_bases = 3
        link_0_offset = 0x17000
        per_link_offset = 0x8000
        links_per_io = 4

        iob = link // links_per_io
        iolink = link % links_per_io
        link_offset = io_ctrl_base + per_io_ctrl_offset * iob + link_0_offset + per_link_offset * iolink
        return link_offset + reg

    def nvlink_write(self, link, reg, data):
        reg_offset = self._nvlink_offset(link, reg)
        self.write(reg_offset, data)

    def nvlink_write_verbose(self, link, reg, data):
        reg_offset = self._nvlink_offset(link, reg)
        self.write_verbose(reg_offset, data)

    def nvlink_read(self, link, reg):
        reg_offset = self._nvlink_offset(link, reg)
        return self.read(reg_offset)

    def block_nvlink(self, nvlink):
        assert self.name == "A100"

        if self.name == "A100":
            return self.block_nvlink_a100(nvlink)

    def block_nvlink_a100(self, link, lock=True):
        assert link >= 0
        assert link < 12

        self.nvlink_write_verbose(link, 0x64c, 0x1)


        if lock:
            self.nvlink_write_verbose(link, 0x650, 0x1)


    def block_nvlink(self, nvlink):
        assert self.name == "A100"

        if self.name == "A100":
            return self.block_nvlink_a100(nvlink)

    def read_module_id_h100(self):

        if self.device in [0x2330, 0x2331, 0x2336, 0x2324]:
            gpios = [0x9, 0x11, 0x12]
        else:
            raise GpuError(f"{self} has unknown mapping for module id")

        mod_id = 0
        for i, gpio in enumerate(gpios):
            bit = (self.read(0x21200 + 4 * gpio) >> 14) & 0x1
            mod_id |= bit << i

        if self.device in [0x2330, 0x2331, 0x2336]:
            mod_id ^= 0x4

        return mod_id

    def read_module_id(self):
        if self.is_hopper_plus:
            return self.read_module_id_h100()

    def debug_dump(self):
        offsets = []
        if self.is_hopper_plus:
            offsets.append(("boot_status", 0x200bc))
            offsets.append(("boot_flags", 0x20120))
            for i in range(4):
                offsets.append((f"fsp_status_{i}", 0x8f0320 + i * 4))
            offsets.append((f"prc_0", 0x92de0))
            offsets.append((f"prc_1", 0x92de4))
            offsets.append((f"prc_2", 0x92de8))
            offsets.append((f"prc_cold", 0x92dc0))
        elif self.is_turing_plus:
            for i in range(4):
                offsets.append((f"boot_status_{i}", 0x118234 + i * 4))

        num_vbios_scratches = 32
        if self.is_turing_plus:
            num_vbios_scratches = 64
        for i in range(num_vbios_scratches):
            offsets.append((f"vbios_scratch_{i:02d}", self.vbios_scratch_register(i)))

        if not self.is_hopper_plus:
            for i in range(3):
                offsets.append((f"vbios_ifr_{i:02d}", 0x1720 + i * 4))

        for name, offset in offsets:
            data = self.read_bad_ok(offset)
            info(f"{self} {name} 0x{offset:x} = 0x{data:x}")


    def __str__(self):
        return "GPU %s %s %s BAR0 0x%x" % (self.bdf, self.name, hex(self.device), self.bar0_addr)

    def __eq__(self, other):
        return self.bar0_addr == other.bar0_addr


def print_topo_indent(root, indent):
    if root.is_hidden():
        indent = indent - 1
    else:
        print(" " * indent, root)
    for c in root.children:
        print_topo_indent(c, indent + 1)


def gpu_dma_test(gpu, verify_reads=True, verify_writes=True):
    if is_windows:
        error("%s DMA test on Windows is not supported currently", gpu)
        return

    #verify_reads=False
    #verify_writes=False

    use_falcon_dma = False
    if gpu.is_ampere_plus:
        # Ampere+ cannot use the bar0 window for accessing sysmem any more. Use
        # the falcon DMA path that's functional but much slower.
        use_falcon_dma = True

    if use_falcon_dma:
        sysmem_write = lambda pa, data: gpu.dma_sys_write32(pa, data)
        sysmem_read = lambda pa: gpu.dma_sys_read32(pa)
    else:
        sysmem_write = lambda pa, data: gpu.bar0_window_sys_write32(pa, data)
        sysmem_read = lambda pa: gpu.bar0_window_sys_read32(pa)

    page_size = os.sysconf("SC_PAGE_SIZE")
    total_size = 0

    # Approximate total
    total_mem_size = page_size * os.sysconf('SC_PHYS_PAGES')
    info("Total memory ~%d GBs, verifying reads %s verifying writes %s", total_mem_size // 2**30, verify_reads, verify_writes)

    start = perf_counter()
    last_debug = start
    buffers = []
    chunk_size = 8 * 1024 * 1024

    min_pa = 2**128
    max_pa = 0

    # Can't really pin pages without a kernel driver, but mlockall() should be
    # good enough for our purpose.
    # 1 is MCL_CURRENT
    # 2 is MCL_FUTURE
    flags = 2
    libc.mlockall(ctypes.c_int(flags))

    while True:
        try:
            buf = mmap.mmap(-1, chunk_size)
        except Exception as err:
            if chunk_size == 4096:
                info("Cannot allocate any more memory")
                return
            info("Failed to allocate %d bytes, retrying with half the size", chunk_size)
            chunk_size /= 2
            continue

        # Hold onto the memory so that it cannot be allocated again
        buffers.append(buf)

        base_addr = ctypes.addressof(ctypes.c_int.from_buffer(buf))
        assert base_addr % page_size == 0

        ctypes.memset(base_addr, 0xab, chunk_size)

        num_pages = chunk_size // page_size
        page_info = PageInfo(base_addr, num_pages)

        stride = 1

        for va in range(base_addr, base_addr + chunk_size, page_size * stride):
            offset = va - base_addr
            page_index = offset // page_size
            pa = page_info.physical_address(page_index)
            min_pa = min(min_pa, pa)
            max_pa = max(max_pa, pa)

            if verify_reads:
                gpu_data = sysmem_read(pa)
                if gpu_data != 0xabababab:
                    error("VA 0x{:x} PA 0x{:x} GPU didn't read expected data after CPU write, saw 0x{:x}".format(va, pa, gpu_data))

            if verify_writes:
                sysmem_write(pa, 0xbcbcbcbc)
                gpu_data_2 = sysmem_read(pa)
                cpu_data = int_from_data(buf[offset : offset + 4], 4)
                if cpu_data != 0xbcbcbcbc:
                    error("PA 0x{:x} CPU didn't read expected data after GPU write, saw 0x{:x}".format(pa, cpu_data))
                if gpu_data_2 != 0xbcbcbcbc:
                    error("PA 0x{:x} GPU didn't read expected data after GPU write, saw 0x{:x}".format(pa, gpu_data_2))

            if perf_counter() - last_debug > 1:
                last_debug = perf_counter()
                mbs = max(total_size, 1) / (1024 * 1024.)
                t = last_debug - start
                time_left = (total_mem_size - total_size) / (mbs / t) / (1024 * 1024.)

                pa_diff_gb = (max_pa - min_pa) // 2**30
                info("So far verified %.1f MB, %.1f MB/s, time %.1f s, time left ~%.1f s, min PA 0x%x max PA 0x%x max-min %d GB", mbs, mbs/t, t, time_left, min_pa, max_pa, pa_diff_gb)

            total_size += page_size * stride

def pcie_p2p_test(gpus):
    for g1 in gpus:
        for g2 in gpus:
            if g1 == g2:
                continue
            fail = False

            g2_bar0 = g2.bar0_addr
            g2_boot_p2p = g1.dma_sys_read32(g2_bar0)
            g2_boot = g2.read(0)
            if g2_boot != g2_boot_p2p:
                error(" {0} cannot read BAR0 of {1}, read 0x{2:x} != 0x{3:x}".format(g1, g2, g2_boot_p2p, g2_boot))
                fail = True

            scratch_offset = g2.flr_resettable_scratch()
            g2.write(scratch_offset, 0)
            g1.dma_sys_write32(g2_bar0 + scratch_offset, 0xcafe)
            g2_scratch = g2.read(scratch_offset)
            g2.write(scratch_offset, 0)
            if g2_scratch != 0xcafe:
                error(" {0} cannot write BAR0 of {1}, read 0x{2:x} != 0x{3:x}".format(g1, g2, g2_scratch, 0xcafe))
                fail = True

            g2_bar1 = g2.bar1_addr
            bar1_offset = 0
            g2_bar1_misc_p2p = g1.dma_sys_read32(g2_bar1 + bar1_offset)
            g2_bar1_misc = g2.bar1.read32(bar1_offset)
            if g2_bar1_misc_p2p != g2_bar1_misc:
                error(" {0} cannot read BAR1 of {1}, read 0x{2:x} != 0x{3:x}".format(g1, g2, g2_bar1_misc_p2p, g2_bar1_misc))
                fail = True

            g2.bar1.write32(bar1_offset, 0x0)
            g1.dma_sys_write32(g2_bar1 + bar1_offset, 0xcafe)
            g2_bar1_scratch = g2.bar1.read32(bar1_offset)
            g2.bar1.write32(bar1_offset, 0x0)

            if g2_bar1_scratch != 0xcafe:
                error(" {0} cannot write BAR1 of {1}, read 0x{2:x} != 0x{3:x}".format(g1, g2, g2_bar1_scratch, 0xcafe))
                fail = True

            if not fail:
                info(" {0} can access {1} p2p boot 0x{2:x} bar1 0x{3:x}".format(g1, g2, g2_boot_p2p, g2_bar1_misc))



def print_topo():
    print("Topo:")
    for c in DEVICES:
        dev = DEVICES[c]
        if dev.is_root():
            print_topo_indent(dev, 1)
    sys.stdout.flush()

def sysfs_pci_rescan():
    with open("/sys/bus/pci/rescan", "w") as rf:
        rf.write("1")

def create_args():
    argp = optparse.OptionParser(usage="usage: %prog [options]")
    argp.add_option("--gpu", type="int", default=-1)
    argp.add_option("--gpu-bdf", help="Select a single GPU by providing a substring of the BDF, e.g. '01:00'.")
    argp.add_option("--gpu-name", help="Select a single GPU by providing a substring of the GPU name, e.g. 'T4'. If multiple GPUs match, the first one will be used.")
    argp.add_option("--no-gpu", action='store_true', help="Do not use any of the GPUs; commands requiring one will not work.")
    argp.add_option("--log", type="choice", choices=['debug', 'info', 'warning', 'error', 'critical'], default='error')
    argp.add_option("--mmio-access-type", type="choice", choices=['devmem', 'sysfs'], default='devmem',
                      help="On Linux, specify whether to do MMIO through /dev/mem or /sys/bus/pci/devices/.../resourceN")

    argp.add_option("--recover-broken-gpu", action='store_true', default=False,
                      help="""Attempt recovering a broken GPU (unresponsive config space or MMIO) by performing an SBR. If the GPU is
broken from the beginning and hence correct config space wasn't saved then
reenumarate it in the OS by sysfs remove/rescan to restore BARs etc.""")
    argp.add_option("--reset-with-sbr", action='store_true', default=False,
                      help="Reset the GPU with SBR and restore its config space settings, before any other actions")
    argp.add_option("--reset-with-flr", action='store_true', default=False,
                      help="Reset the GPU with FLR and restore its config space settings, before any other actions")
    argp.add_option("--reset-with-os", action='store_true', default=False,
                      help="Reset with OS through /sys/.../reset")
    argp.add_option("--remove-from-os", action='store_true', default=False,
                      help="Remove from OS through /sys/.../remove")
    argp.add_option("--unbind-gpu", action='store_true', default=False, help="Unbind GPU")
    argp.add_option("--unbind-gpus", action='store_true', default=False, help="Unbind GPUs")
    argp.add_option("--bind-gpu", help="Bind GPUs to the specified driver")
    argp.add_option("--bind-gpus", help="Bind GPUs to the specified driver")
    argp.add_option("--query-ecc-state", action='store_true', default=False,
                      help="Query the ECC state of the GPU")
    argp.add_option("--query-cc-mode", action='store_true', default=False,
                      help="Query the current Confidential Computing (CC) mode of the GPU.")
    argp.add_option("--query-cc-settings", action='store_true', default=False,
                      help="Query the Confidential Computing (CC) settings of the GPU."
                      "This prints the lower level setting knobs that will take effect upon GPU reset.")
    argp.add_option("--query-prc-knobs", action='store_true', default=False,
                      help="Query all the Product Reconfiguration (PRC) knobs.")
    argp.add_option("--set-cc-mode", type='choice', choices=["off", "on", "devtools"],
                      help="Configure Confidentail Computing (CC) mode. The choices are off (disabled), on (enabled) or devtools (enabled in DevTools mode)."
                      "The GPU needs to be reset to make the selected mode active. See --reset-after-cc-mode-switch for one way of doing it.")
    argp.add_option("--reset-after-cc-mode-switch", action='store_true', default=False,
                    help="Reset the GPU after switching CC mode such that it is activated immediately.")
    argp.add_option("--test-cc-mode-switch", action='store_true', default=False,
                    help="Test switching CC modes.")
    argp.add_option("--query-l4-serial-number", action='store_true', default=False,
                    help="Query the L4 certificate serial number without the MSB. The MSB could be either 0x41 or 0x40 based on the RoT returning the certificate chain.")
    argp.add_option("--query-module-name", action='store_true', help="Query the module name (aka physical ID and module ID). Supported only on H100 SXM and NVSwitch_gen3")
    argp.add_option("--clear-memory", action='store_true', default=False,
                      help="Clear the contents of the GPU memory. Supported on Pascal+ GPUs. Assumes the GPU has been reset with SBR prior to this operation and can be comined with --reset-with-sbr if not.")

    argp.add_option("--debug-dump", action='store_true', default=False, help="Dump various state from the device for debug")
    argp.add_option("--nvlink-debug-dump", action="store_true", help="Dump NVLINK debug state.")
    argp.add_option("--detect-nvlink", action="store_true", help="Determines whether NVLINK is present.")
    argp.add_option("--force-ecc-on-after-reset", action='store_true', default=False,
                    help="Force ECC to be enabled after a subsequent GPU reset")
    argp.add_option("--test-ecc-toggle", action='store_true', default=False,
                    help="Test toggling ECC mode.")
    argp.add_option("--query-mig-mode", action='store_true', default=False,
                    help="Query whether MIG mode is enabled.")
    argp.add_option("--force-mig-off-after-reset", action='store_true', default=False,
                    help="Force MIG mode to be disabled after a subsequent GPU reset")
    argp.add_option("--test-mig-toggle", action='store_true', default=False,
                    help="Test toggling MIG mode.")
    argp.add_option("--block-nvlink", type='int', action='append',
                    help="Block the specified NVLink. Can be specified multiple times to block more NVLinks. NVLinks will be blocked until an SBR. Supported on A100 only.")
    argp.add_option("--block-all-nvlinks", action='store_true', default=False,
                    help="Block all NVLinks. NVLinks will be blocked until a subsequent SBR. Supported on A100 only.")
    argp.add_option("--dma-test", action='store_true', default=False,
                    help="Check that GPUs are able to perform DMA to all/most of available system memory.")
    argp.add_option("--test-pcie-p2p", action='store_true', default=False,
                    help="Check that all GPUs are able to perform DMA to each other.")
    argp.add_option("--read-sysmem-pa", type='int', help="""Use GPU's DMA to read 32-bits from the specified sysmem physical address""")
    argp.add_option("--write-sysmem-pa", type='int', nargs=2, help="""Use GPU's DMA to write specified 32-bits to the specified sysmem physical address""")
    argp.add_option("--read-config-space", type='int', nargs=1, help="""Read 32-bits from device's config space at specified offset""")
    argp.add_option("--write-config-space", type='int', nargs=2, help="""Write 32-bit to device's config space at specified offset""")
    argp.add_option("--read-bar0", type='int', nargs=1, help="""Read 32-bits from GPU BAR0 at specified offset""")
    argp.add_option("--write-bar0", type='int', nargs=2, help="""Write 32-bit to GPU BAR0 at specified offset""")
    argp.add_option("--read-bar1", type='int', nargs=1, help="""Read 32-bits from GPU BAR1 at specified offset""")
    argp.add_option("--write-bar1", type='int', nargs=2, help="""Write 32-bit to GPU BAR1 at specified offset""")
    argp.add_option("--ignore-nvidia-driver", action='store_true', default=False, help="Do not treat nvidia driver apearing to be loaded as an error")

    return argp

# Called instead of main() when imported as a library rather than run as a
# command.
def init():
    global opts

    argp = create_args()
    (opts, _) = argp.parse_args([])

def main():
    #print(f"NVIDIA GPU Tools version {VERSION}")
    #print(f"Command line arguments: {sys.argv}")
    sys.stdout.flush()

    global opts

    argp = create_args()
    (opts, args) = argp.parse_args()

    if len(args) != 0:
        print("ERROR: Exactly zero positional argument expected.")
        argp.print_usage()
        sys.exit(1)

    logging.basicConfig(level=getattr(logging, opts.log.upper()),
                        format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')

    global mmio_access_type
    if is_linux:
        mmio_access_type = opts.mmio_access_type


    if not opts.no_gpu:
        check_device_module_deps()

    if opts.gpu_bdf is not None:
        gpus, other = find_gpus(opts.gpu_bdf)
        if len(gpus) == 0:
            error("Matching for {0} found nothing".format(opts.gpu_bdf))
            sys.exit(1)
        elif len(gpus) > 1:
            error("Matching for {0} found more than one GPU {1}".format(opts.gpu_bdf, ", ".join([str(g) for g in gpus])))
            sys.exit(1)
        else:
            gpu = gpus[0]
    elif opts.gpu_name is not None:
        gpus, other = find_gpus()
        gpus = [g for g in gpus if opts.gpu_name in g.name]
        if len(gpus) == 0:
            error("Matching for {0} found nothing".format(opts.gpu_name))
            sys.exit(1)
        gpu = gpus[0]
    elif opts.no_gpu:
        gpu = None
        info("Using no GPU")
    else:
        gpus, other = find_gpus()
        print("GPUs:")
        for i, g in enumerate(gpus):
            print(" ", i, g)
        print("Other:")
        for i, o in enumerate(other):
            print(" ", i, o)
        sys.stdout.flush()

        if opts.gpu == -1:
            info("No GPU specified, select GPU with --gpu, --gpu-bdf, or --gpu-name")
            return 0

        if opts.gpu >= len(gpus):
            raise ValueError("GPU index out of bounds")
        gpu = gpus[opts.gpu]


    if gpu:
        if not opts.detect_nvlink:
            print_topo()
        info("Selected %s", gpu)
        if gpu.is_driver_loaded():
            if not opts.ignore_nvidia_driver:
                error("The nvidia driver appears to be using %s, aborting. Specify --ignore-nvidia-driver to ignore this check.", gpu)
                sys.exit(1)
            else:
                warning("The nvidia driver appears to be using %s, but --ignore-nvidia-driver was specified, continuing.", gpu)

        if not gpu.is_broken_gpu():

            if gpu.is_in_recovery():
                warning(f"{gpu} is in recovery")
            else:
                if gpu.is_gpu() and gpu.is_hopper_plus:
                    if gpu.is_boot_done():
                        cc_mode = gpu.query_cc_mode()
                        if cc_mode != "off":
                            warning(f"{gpu} has CC mode {cc_mode}, some functionality may not work")

    if gpu:

        if opts.unbind_gpu:
            gpu.sysfs_unbind()

        if opts.unbind_gpus:
            for dev in gpus:
                if dev.is_gpu():
                    dev.sysfs_unbind()

        if opts.bind_gpu:
            gpu.sysfs_bind(opts.bind_gpu)

        if opts.bind_gpus:
            for dev in gpus:
                if dev.is_gpu():
                    dev.sysfs_bind(opts.bind_gpus)


        if gpu.is_broken_gpu():
            # If the GPU is broken, try to recover it if requested,
            # otherwise just exit immediately
            if opts.recover_broken_gpu:
                if gpu.parent.is_hidden():
                    error("Cannot recover the GPU as the upstream port is hidden")
                    sys.exit(1)
                    return

                # Reset the GPU with SBR and if successful,
                # remove and rescan it to recover BARs
                if gpu.reset_with_sbr():
                    gpu.sysfs_remove()
                    sysfs_pci_rescan()
                    gpu.reinit()
                    if gpu.is_broken_gpu():
                        error("Failed to recover %s", gpu)
                        sys.exit(1)
                    else:
                        info("Recovered %s", gpu)
                else:
                    error("Failed to recover %s %s", gpu, gpu.parent.link_status)
                    sys.exit(1)
            else:
                error("%s is broken and --recover-broken-gpu was not specified, returning failure.", gpu)
                sys.exit(1)
            return


    # Reset the GPU with SBR, if requested
    if opts.reset_with_sbr:
        if not gpu.parent.is_bridge():
            error("Cannot reset the GPU with SBR as the upstream bridge is not accessible")
        else:
            gpu.reset_with_sbr()

    # Reset the GPU with FLR, if requested
    if opts.reset_with_flr:
        if gpu.is_flr_supported():
            gpu.reset_with_flr()
        else:
            error("Cannot reset the GPU with FLR as it is not supported")

    if opts.reset_with_os:
        gpu.sysfs_reset()

    if opts.remove_from_os:
        gpu.sysfs_remove()

    if opts.query_ecc_state:
        if not gpu.is_gpu() or not gpu.is_ecc_query_supported:
            error("Querying ECC state not supported on %s", gpu)
            sys.exit(1)

        ecc_state = gpu.query_final_ecc_state()
        info("%s ECC is %s", gpu, "enabled" if ecc_state else "disabled")

    if opts.query_cc_settings:
        if not gpu.is_gpu() or not gpu.is_cc_query_supported:
            error(f"Querying CC settings is not supported on {gpu}")
            sys.exit(1)

        cc_settings = gpu.query_cc_settings()
        info(f"{gpu} CC settings:")
        for name, value in cc_settings:
            info(f"  {name} = {value}")

    if opts.query_prc_knobs:
        if not gpu.has_fsp:
            error(f"Querying PRC knobs is not supported on {gpu}")
            sys.exit(1)

        prc_knobs = gpu.query_prc_knobs()
        info(f"{gpu} PRC knobs:")
        for name, value in prc_knobs:
            info(f"  {name} = {value}")

    if opts.set_cc_mode:
        if not gpu.is_gpu() or not gpu.is_cc_query_supported:
            error(f"Configuring CC not supported on {gpu}")
            sys.exit(1)

        try:
            gpu.set_cc_mode(opts.set_cc_mode)
        except GpuError as err:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            gpu.debug_dump()
            prc_knobs = gpu.query_prc_knobs()
            debug(f"{gpu} PRC knobs:")
            for name, value in prc_knobs:
                debug(f"  {name} = {value}")
            raise

        info(f"{gpu} CC mode set to {opts.set_cc_mode}. It will be active after GPU reset.")
        if opts.reset_after_cc_mode_switch:
            gpu.reset_with_os()
            new_mode = gpu.query_cc_mode()
            if new_mode != opts.set_cc_mode:
                raise GpuError(f"{gpu} failed to switch to CC mode {opts.set_cc_mode}, current mode is {new_mode}.")
            info(f"{gpu} was reset to apply the new CC mode.")

    if opts.query_cc_mode:
        if not gpu.is_gpu() or not gpu.is_cc_query_supported:
            error(f"Querying CC mode is not supported on {gpu}")
            sys.exit(1)

        cc_mode = gpu.query_cc_mode()
        info(f"{gpu} CC mode is {cc_mode}")

    if opts.test_cc_mode_switch:
        if not gpu.is_gpu() or not gpu.is_cc_query_supported:
            error(f"Configuring CC not supported on {gpu}")
            sys.exit(1)
        try:
            gpu.test_cc_mode_switch()
        except GpuError as err:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            gpu.debug_dump()
            prc_knobs = gpu.query_prc_knobs()
            debug(f"{gpu} PRC knobs:")
            for name, value in prc_knobs:
                debug(f"  {name} = {value}")
            raise

    if opts.query_l4_serial_number:
        if not gpu.has_pdi:
            error(f"Querying L4 serial number not supported on {gpu}")
            sys.exit(1)
        print(f"L4 serial number: {gpu.get_pdi():#x}")

    if opts.clear_memory:
        if gpu.is_memory_clear_supported:
            gpu.clear_memory()

        else:
            error("Clearing memory not supported on %s", gpu)


    if opts.debug_dump:
        info(f"{gpu} debug dump:")
        gpu.debug_dump()



    if opts.force_ecc_on_after_reset:
        if gpu.is_forcing_ecc_on_after_reset_supported:
            gpu.force_ecc_on_after_reset()
        else:
            error("Forcing ECC on after reset not supported on %s", gpu)

    if opts.test_ecc_toggle:
        if gpu.is_forcing_ecc_on_after_reset_supported:
            try:
                gpu.test_ecc_toggle()
            except Exception as err:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)
                error("%s", str(err))
                error("%s testing ECC toggle failed", gpu)
                sys.exit(1)
        else:
            error("Toggling ECC not supported on %s", gpu)
            sys.exit(1)

    if opts.query_mig_mode:
        mig_state = gpu.query_mig_mode()
        info("%s MIG mode is %s", gpu, "enabled" if mig_state else "disabled")

    if opts.force_mig_off_after_reset:
        if gpu.is_mig_mode_supported:
            gpu.set_mig_mode_after_reset(enabled=False)
        else:
            error("Forcing MIG off after reset not supported on %s", gpu)

    if opts.test_mig_toggle:
        if gpu.is_mig_mode_supported:
            try:
                gpu.test_mig_toggle()
            except Exception as err:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)
                error("%s", str(err))
                error("%s testing MIG toggle failed", gpu)
                sys.exit(1)
        else:
            error("Toggling MIG not supported on %s", gpu)
            sys.exit(1)



    if opts.block_all_nvlinks or opts.block_nvlink:
        if not gpu.is_nvlink_supported:
            error(f"{gpu} does not support NVLink blocking")
            sys.exit(1)

        num_nvlinks = gpu.nvlink["number"]

        if opts.block_all_nvlinks:
            links_to_block = range(num_nvlinks)
        else:
            links_to_block = opts.block_nvlink

        for link in links_to_block:
            if link < 0 or link >= num_nvlinks:
                error(f"Invalid link {link}, num nvlinks {num_nvlinks}")
                sys.exit(1)

        gpu.block_nvlinks(links_to_block)
        info(f"{gpu} blocked NVLinks {links_to_block}")

    if opts.dma_test:
        gpu_dma_test(gpu)

    if opts.test_pcie_p2p:
        pcie_p2p_test([gpu for gpu in gpus if gpu.is_gpu()])

    if opts.read_sysmem_pa:
        addr = opts.read_sysmem_pa

        data = gpu.dma_sys_read32(addr)
        info("%s read PA 0x%x = 0x%x", gpu, addr, data)

    if opts.write_sysmem_pa:
        addr = opts.write_sysmem_pa[0]
        data = opts.write_sysmem_pa[1]

        gpu.dma_sys_write32(addr, data)
        info("%s wrote PA 0x%x = 0x%x", gpu, addr, data)

    if opts.read_config_space is not None:
        addr = opts.read_config_space
        data = gpu.config.read32(addr)
        info("Config space read 0x%x = 0x%x", addr, data)

    if opts.write_config_space is not None:
        addr = opts.write_config_space[0]
        data = opts.write_config_space[1]
        gpu.config.write32(addr, data)

    if opts.read_bar0 is not None:
        addr = opts.read_bar0
        data = gpu.read(addr)
        info("BAR0 read 0x%x = 0x%x", addr, data)

    if opts.write_bar0 is not None:
        addr = opts.write_bar0[0]
        data = opts.write_bar0[1]
        gpu.write_verbose(addr, data)

    if opts.read_bar1 is not None:
        addr = opts.read_bar1
        data = gpu.read_bar1(addr)
        info("BAR1 read 0x%x = 0x%x", addr, data)

    if opts.write_bar1:
        addr = opts.write_bar1[0]
        data = opts.write_bar1[1]

        gpu.write_bar1(addr, data)



    if opts.detect_nvlink:
        gpu.detect_nvlink()

    if opts.nvlink_debug_dump:
        gpu.nvlink_debug()


NVLINK_TOPOLOGY_HGX_8_H100 = {
    "NVSwitch_0": { 32: ( 9, "SXM_4", 0), 33: ( 8, "SXM_4", 1), 34: ( 6, "SXM_6", 0), 35: ( 7, "SXM_6", 1), 36: (13, "SXM_4", 2), 37: (12, "SXM_4", 3), 38: (15, "SXM_6", 2), 39: (14, "SXM_6", 3), 40: ( 2, "SXM_1", 0), 41: ( 3, "SXM_1", 1), 42: (15, "SXM_2", 0), 43: (14, "SXM_2", 1), 44: (12, "SXM_1", 2), 45: (13, "SXM_1", 3), 46: ( 8, "SXM_2", 2), 47: ( 9, "SXM_2", 3), 48: (13, "SXM_3", 3), 49: (12, "SXM_3", 2), 50: (17, "SXM_7", 3), 51: (16, "SXM_7", 2), 52: ( 7, "SXM_3", 1), 53: ( 6, "SXM_3", 0), 54: (13, "SXM_7", 1), 55: (12, "SXM_7", 0), 56: (12, "SXM_8", 3), 57: (13, "SXM_8", 2), 58: ( 7, "SXM_5", 3), 59: ( 6, "SXM_5", 2), 60: (17, "SXM_8", 1), 61: (16, "SXM_8", 0), 62: (12, "SXM_5", 1), 63: (13, "SXM_5", 0), },
    "NVSwitch_1": {  0: (17, "SXM_3", 3),  1: (16, "SXM_3", 4),  2: ( 2, "SXM_2", 1),  3: ( 3, "SXM_2", 2),  4: ( 7, "SXM_2", 3),  5: ( 6, "SXM_2", 4),  6: ( 8, "SXM_6", 3),  7: ( 9, "SXM_6", 4), 32: (11, "SXM_2", 0), 33: (10, "SXM_3", 0), 34: (17, "SXM_6", 0), 35: (16, "SXM_6", 1), 36: ( 0, "SXM_1", 0), 37: ( 1, "SXM_1", 1), 38: ( 3, "SXM_3", 1), 39: ( 2, "SXM_3", 2), 40: (11, "SXM_1", 2), 41: (10, "SXM_8", 0), 42: (11, "SXM_6", 2), 43: (10, "SXM_7", 0), 44: ( 5, "SXM_8", 1), 45: ( 4, "SXM_8", 2), 46: (16, "SXM_1", 3), 47: (17, "SXM_1", 4), 48: (17, "SXM_5", 4), 49: (16, "SXM_5", 3), 50: ( 2, "SXM_4", 4), 51: ( 3, "SXM_4", 3), 52: (11, "SXM_5", 2), 53: (10, "SXM_4", 0), 54: ( 0, "SXM_7", 4), 55: ( 1, "SXM_7", 3), 56: ( 1, "SXM_5", 1), 57: ( 0, "SXM_5", 0), 58: ( 4, "SXM_7", 2), 59: ( 5, "SXM_7", 1), 60: ( 0, "SXM_8", 4), 61: ( 1, "SXM_8", 3), 62: (14, "SXM_4", 2), 63: (15, "SXM_4", 1), },
    "NVSwitch_2": {  0: ( 4, "SXM_6", 2),  1: ( 5, "SXM_6", 3),  2: ( 7, "SXM_4", 2),  3: ( 6, "SXM_4", 3), 16: (14, "SXM_3", 4), 17: (15, "SXM_3", 3), 18: (11, "SXM_8", 4), 19: (10, "SXM_6", 4), 32: ( 1, "SXM_6", 0), 33: ( 0, "SXM_6", 1), 34: (10, "SXM_2", 4), 35: (11, "SXM_4", 4), 36: (15, "SXM_5", 0), 37: (14, "SXM_5", 1), 38: (16, "SXM_4", 0), 39: (17, "SXM_4", 1), 40: ( 5, "SXM_2", 0), 41: ( 4, "SXM_2", 1), 42: (15, "SXM_1", 1), 43: (14, "SXM_1", 2), 44: (10, "SXM_5", 4), 45: (10, "SXM_1", 0), 46: ( 0, "SXM_2", 2), 47: ( 1, "SXM_2", 3), 48: (15, "SXM_7", 3), 49: (14, "SXM_7", 2), 50: ( 8, "SXM_3", 2), 51: ( 9, "SXM_3", 1), 52: (11, "SXM_3", 0), 53: (11, "SXM_7", 4), 54: (14, "SXM_8", 3), 55: (15, "SXM_8", 2), 56: ( 8, "SXM_7", 1), 57: ( 9, "SXM_7", 0), 58: ( 7, "SXM_8", 1), 59: ( 6, "SXM_8", 0), 60: ( 2, "SXM_5", 3), 61: ( 3, "SXM_5", 2), 62: ( 6, "SXM_1", 4), 63: ( 7, "SXM_1", 3), },
    "NVSwitch_3": { 32: (13, "SXM_6", 0), 33: (12, "SXM_6", 1), 34: (12, "SXM_2", 0), 35: (13, "SXM_2", 1), 36: ( 3, "SXM_6", 2), 37: ( 2, "SXM_6", 3), 38: (16, "SXM_2", 2), 39: (17, "SXM_2", 3), 40: ( 7, "SXM_7", 0), 41: ( 6, "SXM_7", 1), 42: ( 5, "SXM_4", 0), 43: ( 4, "SXM_4", 1), 44: ( 3, "SXM_7", 2), 45: ( 2, "SXM_7", 3), 46: ( 1, "SXM_4", 2), 47: ( 0, "SXM_4", 3), 48: ( 4, "SXM_5", 3), 49: ( 5, "SXM_5", 2), 50: ( 2, "SXM_8", 3), 51: ( 3, "SXM_8", 2), 52: ( 9, "SXM_5", 1), 53: ( 8, "SXM_5", 0), 54: ( 8, "SXM_8", 1), 55: ( 9, "SXM_8", 0), 56: ( 5, "SXM_3", 3), 57: ( 4, "SXM_3", 2), 58: ( 4, "SXM_1", 3), 59: ( 5, "SXM_1", 2), 60: ( 1, "SXM_3", 1), 61: ( 0, "SXM_3", 0), 62: ( 9, "SXM_1", 1), 63: ( 8, "SXM_1", 0), },
    "SXM_1": {  0: (36, "NVSwitch_1", 0),  1: (37, "NVSwitch_1", 1),  2: (40, "NVSwitch_0", 0),  3: (41, "NVSwitch_0", 1),  4: (58, "NVSwitch_3", 3),  5: (59, "NVSwitch_3", 2),  6: (62, "NVSwitch_2", 4),  7: (63, "NVSwitch_2", 3),  8: (63, "NVSwitch_3", 0),  9: (62, "NVSwitch_3", 1), 10: (45, "NVSwitch_2", 0), 11: (40, "NVSwitch_1", 2), 12: (44, "NVSwitch_0", 2), 13: (45, "NVSwitch_0", 3), 14: (43, "NVSwitch_2", 2), 15: (42, "NVSwitch_2", 1), 16: (46, "NVSwitch_1", 3), 17: (47, "NVSwitch_1", 4), },
    "SXM_2": {  0: (46, "NVSwitch_2", 2),  1: (47, "NVSwitch_2", 3),  2: ( 2, "NVSwitch_1", 1),  3: ( 3, "NVSwitch_1", 2),  4: (41, "NVSwitch_2", 1),  5: (40, "NVSwitch_2", 0),  6: ( 5, "NVSwitch_1", 4),  7: ( 4, "NVSwitch_1", 3),  8: (46, "NVSwitch_0", 2),  9: (47, "NVSwitch_0", 3), 10: (34, "NVSwitch_2", 4), 11: (32, "NVSwitch_1", 0), 12: (34, "NVSwitch_3", 0), 13: (35, "NVSwitch_3", 1), 14: (43, "NVSwitch_0", 1), 15: (42, "NVSwitch_0", 0), 16: (38, "NVSwitch_3", 2), 17: (39, "NVSwitch_3", 3), },
    "SXM_3": {  0: (61, "NVSwitch_3", 0),  1: (60, "NVSwitch_3", 1),  2: (39, "NVSwitch_1", 2),  3: (38, "NVSwitch_1", 1),  4: (57, "NVSwitch_3", 2),  5: (56, "NVSwitch_3", 3),  6: (53, "NVSwitch_0", 0),  7: (52, "NVSwitch_0", 1),  8: (50, "NVSwitch_2", 2),  9: (51, "NVSwitch_2", 1), 10: (33, "NVSwitch_1", 0), 11: (52, "NVSwitch_2", 0), 12: (49, "NVSwitch_0", 2), 13: (48, "NVSwitch_0", 3), 14: (16, "NVSwitch_2", 4), 15: (17, "NVSwitch_2", 3), 16: ( 1, "NVSwitch_1", 4), 17: ( 0, "NVSwitch_1", 3), },
    "SXM_4": {  0: (47, "NVSwitch_3", 3),  1: (46, "NVSwitch_3", 2),  2: (50, "NVSwitch_1", 4),  3: (51, "NVSwitch_1", 3),  4: (43, "NVSwitch_3", 1),  5: (42, "NVSwitch_3", 0),  6: ( 3, "NVSwitch_2", 3),  7: ( 2, "NVSwitch_2", 2),  8: (33, "NVSwitch_0", 1),  9: (32, "NVSwitch_0", 0), 10: (53, "NVSwitch_1", 0), 11: (35, "NVSwitch_2", 4), 12: (37, "NVSwitch_0", 3), 13: (36, "NVSwitch_0", 2), 14: (62, "NVSwitch_1", 2), 15: (63, "NVSwitch_1", 1), 16: (38, "NVSwitch_2", 0), 17: (39, "NVSwitch_2", 1), },
    "SXM_5": {  0: (57, "NVSwitch_1", 0),  1: (56, "NVSwitch_1", 1),  2: (60, "NVSwitch_2", 3),  3: (61, "NVSwitch_2", 2),  4: (48, "NVSwitch_3", 3),  5: (49, "NVSwitch_3", 2),  6: (59, "NVSwitch_0", 2),  7: (58, "NVSwitch_0", 3),  8: (53, "NVSwitch_3", 0),  9: (52, "NVSwitch_3", 1), 10: (44, "NVSwitch_2", 4), 11: (52, "NVSwitch_1", 2), 12: (62, "NVSwitch_0", 1), 13: (63, "NVSwitch_0", 0), 14: (37, "NVSwitch_2", 1), 15: (36, "NVSwitch_2", 0), 16: (49, "NVSwitch_1", 3), 17: (48, "NVSwitch_1", 4), },
    "SXM_6": {  0: (33, "NVSwitch_2", 1),  1: (32, "NVSwitch_2", 0),  2: (37, "NVSwitch_3", 3),  3: (36, "NVSwitch_3", 2),  4: ( 0, "NVSwitch_2", 2),  5: ( 1, "NVSwitch_2", 3),  6: (34, "NVSwitch_0", 0),  7: (35, "NVSwitch_0", 1),  8: ( 6, "NVSwitch_1", 3),  9: ( 7, "NVSwitch_1", 4), 10: (19, "NVSwitch_2", 4), 11: (42, "NVSwitch_1", 2), 12: (33, "NVSwitch_3", 1), 13: (32, "NVSwitch_3", 0), 14: (39, "NVSwitch_0", 3), 15: (38, "NVSwitch_0", 2), 16: (35, "NVSwitch_1", 1), 17: (34, "NVSwitch_1", 0), },
    "SXM_7": {  0: (54, "NVSwitch_1", 4),  1: (55, "NVSwitch_1", 3),  2: (45, "NVSwitch_3", 3),  3: (44, "NVSwitch_3", 2),  4: (58, "NVSwitch_1", 2),  5: (59, "NVSwitch_1", 1),  6: (41, "NVSwitch_3", 1),  7: (40, "NVSwitch_3", 0),  8: (56, "NVSwitch_2", 1),  9: (57, "NVSwitch_2", 0), 10: (43, "NVSwitch_1", 0), 11: (53, "NVSwitch_2", 4), 12: (55, "NVSwitch_0", 0), 13: (54, "NVSwitch_0", 1), 14: (49, "NVSwitch_2", 2), 15: (48, "NVSwitch_2", 3), 16: (51, "NVSwitch_0", 2), 17: (50, "NVSwitch_0", 3), },
    "SXM_8": {  0: (60, "NVSwitch_1", 4),  1: (61, "NVSwitch_1", 3),  2: (50, "NVSwitch_3", 3),  3: (51, "NVSwitch_3", 2),  4: (45, "NVSwitch_1", 2),  5: (44, "NVSwitch_1", 1),  6: (59, "NVSwitch_2", 0),  7: (58, "NVSwitch_2", 1),  8: (54, "NVSwitch_3", 1),  9: (55, "NVSwitch_3", 0), 10: (41, "NVSwitch_1", 0), 11: (18, "NVSwitch_2", 4), 12: (56, "NVSwitch_0", 3), 13: (57, "NVSwitch_0", 2), 14: (54, "NVSwitch_2", 3), 15: (55, "NVSwitch_2", 2), 16: (61, "NVSwitch_0", 0), 17: (60, "NVSwitch_0", 1), },
}

if __name__ == "__main__":
    main()
else:
    init()
