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
import array
import collections
import time
import sys
import traceback
from logging import debug, info, warning, error
from pathlib import Path

from utils import platform_config
from utils import data_from_int, array_view_from_bytearray, read_ints_from_path
from utils import formatted_tuple_from_data
from gpu.defines import *
from pci.defines import *
from pci import PciDevice
from gpu import UnknownGpuError, BrokenGpuError, BrokenGpuErrorWithInfo, BrokenGpuErrorSecFault
from gpu.prc import PrcKnob
from gpu import GpuError, GpuPollTimeout, GpuRpcTimeout, FspRpcError
from gpu import GpuProperties

if hasattr(time, "perf_counter"):
    perf_counter = time.perf_counter
else:
    perf_counter = time.time

GPU_ARCHES = ["unknown", "kepler", "maxwell", "pascal", "volta", "turing", "ampere", "ada", "hopper", "blackwell"]
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
}

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

class BrokenGpu(PciDevice):
    def __init__(self, dev_path, sec_fault=None, err_info=None):
        self.cfg_space_working = False
        self.bars_configured = False
        self.sec_fault = sec_fault
        self.err_info = err_info
        super(BrokenGpu, self).__init__(dev_path)
        self.name = "BrokenGpu"
        self.cfg_space_working = self.sanity_check_cfg_space()
        if self.cfg_space_working:
            self.bars_configured = self.sanity_check_cfg_space_bars()

        if self.sec_fault and self.bars_configured:
            self.bar0 = self._map_bar(0)

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
        if self.sec_fault:
            return f"Device {self.bdf} {self.device:#x} in sec fault {self.sec_fault:#x}"
        if self.err_info:
            return f"Device {self.bdf} {self.device:#x} broken {self.err_info}"
        return "GPU %s [broken, cfg space working %d bars configured %d]" % (self.bdf, self.cfg_space_working, self.bars_configured)

class NvidiaDeviceInternal:
    pass


class NvidiaDevice(PciDevice, NvidiaDeviceInternal):
    _cached_device_units = None

    @property
    def device_units(self):
        from gpu.units import gpu_units_cached
        return gpu_units_cached()

    def __init__(self, dev_path):
        super(NvidiaDevice, self).__init__(dev_path)

        if self.has_pm():
            if platform_config.is_linux:
                prev_power_control = self.sysfs_power_control_get()
                if prev_power_control != "on":
                    prev_power_state = self.pmctrl["STATE"]
                    import atexit

                    self.sysfs_power_control_set("on")
                    power_state = self.pmctrl["STATE"]
                    warning(f"{self} was in D{prev_power_state}/control:{prev_power_control}, forced power control to on. New state D{power_state}")
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

        self.is_reset_coupling_supported = False

        self.knob_defaults = {}
        self.is_pcie = False
        self.is_sxm = False
        self.has_c2c = False
        self.is_nvlink_supported = False

        self.units = {}

        if self.parent:
            self.parent.children.append(self)

    def common_init(self):
        self.nvlink = None
        if "nvlink" in self.props:
            self.nvlink = self.props["nvlink"]

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
        if self.is_gpu() and self.is_blackwell_plus:
            return

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
        if self.is_gpu() and self.is_blackwell_plus:
            return

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

    def reset_with_sbr(self, retry_count=1, fail_callback=None):
        self.reset_pre(reset_with_flr=False)

        assert self.parent.is_bridge()
        success = self.parent.toggle_sbr(retry_count=retry_count, fail_callback=fail_callback)
        if not success:
            return False
        if not self.sanity_check_cfg_space():
            return False

        self._restore_cfg_space()
        self.set_command_memory(True)
        if not self.sanity_check():
            return False

        self.reset_post()

        return True

    def _init_fsp_rpc(self):
        if self.fsp_rpc != None:
            return

        # Wait for boot to be done such that FSP is available
        try:
            self.wait_for_boot(silent_on_failure=True)
        except GpuPollTimeout as err:
            warning(f"{self} has not booted successfully within a timeout, but FSP RPC might still be available. Continuing")
            pass

        self.init_falcons()

        if self.is_gpu() and self.is_blackwell_plus:
            self.fsp_rpc = FspRpc(self.fsp, "mnoc", channel_num=0)
        else:
            self.fsp_rpc = FspRpc(self.fsp, "emem", channel_num=2)

        if self.is_gpu() and self.is_hopper:
            self.fsp_rpc_mods = FspRpc(self.fsp, "emem", channel_num=1)

    def poll_register(self, name, offset, value, timeout, sleep_interval=0.01, mask=0xffffffff, debug_print=False, badf_ok=False, not_value=None, trace=False, read_function=None):
        if read_function is None:
            if (value and value >> 16 == 0xbadf) or badf_ok:
                read_function = self.read_bad_ok
            else:
                read_function = self.read

        prev_value = None

        timestamp = perf_counter()
        while True:
            loop_stamp = perf_counter()
            try:
                reg = read_function(offset)
                if trace and reg != prev_value:
                    debug(f"{self} observed new value for {offset:#x} = {reg:#x} after {(perf_counter() - timestamp)*1000:.1f} ms")
                    prev_value = reg
            except:
                error("Failed to read register %s (%s)", name, hex(offset))
                raise

            if value != None:
                if reg & mask == value:
                    if debug_print:
                        debug(f"Register {name} 0x{offset:x} = 0x{reg:x} after {perf_counter() - timestamp:.001f} secs")
                    return
            else:
                if reg & mask != not_value:
                    if debug_print:
                        debug(f"Register {name} 0x{offset:x} = 0x{reg:x} after {perf_counter() - timestamp:.001f} secs")
                    return

            if loop_stamp - timestamp > timeout:
                raise GpuPollTimeout(f"Timed out polling register {name} ({offset:#x}), value {reg:#x} is not the expected {value:#x}. Timeout {timeout:.1f} secs")
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

    @property
    def device_info_instances(self):
        assert self.is_hopper_plus

        if self._device_info_instances is not None:
            return self._device_info_instances

        num_rows = self.regs.read(self.regs.top.NV_PTOP_DEVICE_INFO_CFG_NUM_ROWS)

        in_chain = False
        devices = []
        device = []
        for i in range(0, num_rows):
            data = self.regs.read(self.regs.top.NV_PTOP_DEVICE_INFO2(i))
            if in_chain or data != 0:
                device.append(data.value)
            in_chain = data.ROW_CHAIN == 1
            if not in_chain and len(device) != 0:
                devices.append(device)
                device = []

        self._device_info_instances = collections.defaultdict(list)

        for d in devices:
            device_type = (d[0] >> 24) & 0x7f
            device_inst = (d[0] >> 16) & 0xff
            self._device_info_instances[device_type].append(device_inst)

        return self._device_info_instances

    def _nvlink_query_enabled_links_b100(self):
        self.nvlink_enabled_links = self.device_info_instances[0x1c]
        return self.nvlink_enabled_links

    def _nvlink_query_enabled_links(self):
        if hasattr(self, "nvlink_enabled_links"):
            return self.nvlink_enabled_links

        if self.is_gpu() and self.is_blackwell_plus:
            return self._nvlink_query_enabled_links_b100()
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
            ("NV_INTERNAL", 0x00003040),
            ("NV_INTERNAL", 0x00004040),
            ("NV_INTERNAL", 0x00004048),
        ]

        for name, unit_offset in nport_regs:
            for link in self.nvlink_enabled_links:
                offset = self._nvlink_nport_top_offset(link, unit_offset)
                data = self.read_bad_ok(offset)
                debug(f"{self} link {link:2d} {name} 0x{offset:x} = 0x{data:x}")

    def nvlink_debug_nvlipt_basic_state(self):
        group_regs = [
            ("NV_INTERNAL", 0x00000100),
            ("NV_INTERNAL", 0x00000104),
            ("NV_INTERNAL", 0x00000108),
            ("NV_INTERNAL", 0x0000010c),
            ("NV_INTERNAL", 0x00000110),
            ("NV_INTERNAL", 0x00000114),
            ("NV_INTERNAL", 0x00000200),
        ]
        for g in self.nvlink_enabled_groups:
            for name, unit_offset in group_regs:
                offset = self._nvlink_nvlipt_offset(g, unit_offset)
                data = self.read_bad_ok(offset)
                debug(f"{self} group {g:2d} {name} 0x{unit_offset:x} 0x{offset:x} = 0x{data:x}")

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
            ("NV_INTERNAL", 0x0000060c),
            ("NV_INTERNAL", 0x00000610),
            ("NV_INTERNAL", 0x00000618),
            ("NV_INTERNAL", 0x0000061c),
            ("NV_INTERNAL", 0x00000624),
            ("NV_INTERNAL", 0x00000628),
            ("NV_INTERNAL", 0x00000638),
            ("NV_INTERNAL", 0x0000063c),
            ("NV_INTERNAL", 0x0000064c),
            ("NV_INTERNAL", 0x00000650),
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
            ("NV_INTERNAL", 0x00000788),
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

    def nvlink_get_link_states_debug_b100(self):
        mse_link_states = self.nvlink_get_link_states()

        llu_offsets = [
            ("port_state", 0x4004),
        ]
        plu_offsets = [
            ("linkup_state", 0x50dc),
        ]
        offsets = [(f"llu {name}", 0x10000 + offset) for name, offset in llu_offsets]
        offsets += [(f"plu {name}", 0x18000 + offset) for name, offset in plu_offsets]

        for link in self.nvlink_enabled_links:
            debug(f"{self} link {link:02d} MSE state {mse_link_states[link]}")
            for name, offset in offsets:
                full_offset = 0x3200000 + link * 0x40000 + offset
                state = self.read(full_offset)
                debug(f"{self} link {link:02d} {name} {full_offset:#x} {offset:#x} = {state:#x}")

    def nvlink_get_link_states_b100(self):
        if len(self.nvlink_enabled_links) == 0:
            return []
        self.init_mse()
        return self.mse.portlist_status()

    def nvlink_get_link_states(self):
        self._nvlink_query_enabled_links()
        if self.is_blackwell_plus:
            return self.nvlink_get_link_states_b100()

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
        if self.is_gpu() and self.is_blackwell_plus:
            return self.nvlink_get_link_states()[link] == "up"
        link_state = self.nvlink_dl_get_link_state(link)
        return link_state == "active" or link_state == "sleep"

    def nvlink_get_links_in_hs(self):
        links_in_hs = []
        self._nvlink_query_enabled_links()

        if self.is_gpu() and self.is_blackwell_plus:
            states = self.nvlink_get_link_states()
            for link in self._nvlink_query_enabled_links():
                if states[link] == "up":
                    links_in_hs.append(link)
            return links_in_hs

        for link in self.nvlink_enabled_links:
            if self.nvlink_is_link_in_hs(link):
                links_in_hs.append(link)
        return links_in_hs

    def nvlink_debug_h100(self):
        from collections import Counter
        self.wait_for_boot()
        self._nvlink_query_enabled_links()
        links = self.nvlink_get_links_in_hs()
        link_states = Counter(self.nvlink_dl_get_link_states())
        info(f"{self} trained {len(links)} links {links} dl link states {link_states}")
        if self.is_nvswitch() or (self.is_sxm and not self.has_c2c):
            topo = NVLINK_TOPOLOGY_HGX_8_H100
            for link in self.nvlink_enabled_links:
                # Some links may be enabled unexpectedly
                if link in topo[self.module_name]:
                    peer_link, peer_name, _ = topo[self.module_name][link]
                else:
                    peer_link = "?"
                    peer_name = "?"
                info(f"{self} {self.module_name} link {link} -> {peer_name}:{peer_link} {self.nvlink_dl_get_link_state(link)} {self.nvlink_get_link_state(link)}")
        self.nvlink_debug_nvlipt_basic_state()
        self.nvlink_debug_minion_basic_state()
        self.nvlink_debug_nvlipt_lnk_basic_state()
        self.nvlink_debug_nvltlc_basic_state()
        self.nvlink_debug_nvldl_basic_state()
        if self.is_nvswitch():
            self.nvlink_debug_nport()

    def nvlink_debug_b100(self):
        self._nvlink_query_enabled_links()
        if len(self.nvlink_enabled_links) == 0:
            info(f"{self} has no NVLINKs enabled")
            return

        from collections import Counter
        link_states = self.nvlink_get_link_states()
        state_counter = Counter(link_states)
        info(f"{self} NVLink states {state_counter}")
        self.nvlink_get_link_states_debug_b100()

    def nvlink_debug(self):
        if self.is_gpu() and self.is_blackwell_plus:
            return self.nvlink_debug_b100()
        return self.nvlink_debug_h100()



    def __str__(self):
        return "Nvidia %s BAR0 0x%x devid %s" % (self.bdf, self.bar0_addr, hex(self.device))

    def knobs_query(self, knobs):
        current_state = {}
        for knob in knobs:
            if knob == "cc":
                current_state["cc"] = self.query_cc_mode()
            elif knob == "ppcie":
                current_state["ppcie"] = self.query_ppcie_mode()
            elif knob == "ecc":
                current_state["ecc"] = self.query_final_ecc_state()
            elif knob == "mig":
                current_state["mig"] = self.query_mig_mode()
            else:
                raise ValueError(f"{self} Unhandled {knob}")

        return current_state

    def knobs_set(self, knobs: list, assume_no_pending_settings: bool):
        modified_knobs = []
        current_state = {}

        if self.is_cc_query_supported and self.query_cc_mode() == "on":
            info(f"{self} has CC mode enabled, assuming all knobs need updates")
            assume_no_pending_settings = False

        if self.is_ppcie_query_supported and self.query_ppcie_mode() == "on":
            info(f"{self} has PPCIE mode enabled, assuming all knobs need updates")
            assume_no_pending_settings = False

        if assume_no_pending_settings:
            current_state = self.knobs_query(knobs.keys())
        else:
            for knob in knobs.keys():
                current_state[knob] = "unknown"

        for knob, knob_value in knobs.items():
            if current_state[knob] == knob_value:
                debug(f"{self} knob {knob} already in {knob_value} state, skipping")
                continue

            if knob == "cc":
                self.set_cc_mode(knob_value)
            elif knob == "ppcie":
                try:
                    self.set_ppcie_mode(knob_value)
                except GpuError as err:
                    if isinstance(err, FspRpcError) and err.is_invalid_knob_error:
                        debug(f"{self} does not support PPCIe on current FW, skipping")
                        continue
                    raise
            elif knob == "ecc":
                if self.is_ampere_plus:
                    self.set_ecc_mode_after_reset(knob_value)
                else:
                    if not knob_value:
                        raise ValueError(f"{self} knob {knob} only supports enabled state")
                    self.force_ecc_on_after_reset()
            elif knob == "mig":
                self.set_mig_mode_after_reset(knob_value)
            else:
                raise ValueError(f"{self} Unhandled {knob}")
            modified_knobs.append(knob)

        return modified_knobs

    def knobs_reset_to_defaults(self, knobs, assume_no_pending_settings):
        knobs_to_reset = {}
        for k in knobs:
            if k == "all":
                knobs_to_reset = self.knob_defaults
                break
            if k not in self.knob_defaults:
                raise ValueError(f"{self} doesn't support knob {k}")
            knobs_to_reset[k] = self.knob_defaults[k]

        return self.knobs_set(knobs_to_reset, assume_no_pending_settings)

    def knobs_reset_to_defaults_test(self):
        self.reset_with_os()
        modified = self.knobs_set(self.knob_defaults, True)
        if len(modified) != 0:
            self.reset_with_os()

        combinations = []
        for knob, value in self.knob_defaults.items():
            if isinstance(value, bool):
                if knob == "ecc" and not self.is_ampere_plus:
                    combinations.append([("ecc", True)])
                else:
                    combinations.append([(knob, True), (knob, False)])
            elif knob == "cc":
                combinations.append([(knob, "on"), (knob, "off"), (knob, "devtools")])
            elif knob == "ppcie":
                combinations.append([(knob, "on"), (knob, "off")])
            else:
                raise ValueError("Unhandled knob {knob}")

        # Iterate over all possible combinations
        import itertools
        for test_knobs_tuple in itertools.product(*combinations):
            test_knobs = dict(test_knobs_tuple)

            debug(f"{self} test knobs {test_knobs}")
            self.knobs_set(test_knobs, True)
            self.reset_with_os()
            debug(f"{self} test knobs set {test_knobs}")

            cc_or_ppcie = test_knobs.get("ppcie", "off") == "on" or test_knobs.get("cc", "off") == "on"

            if not cc_or_ppcie:
                test_knobs_check = self.knobs_query(test_knobs.keys())
                if test_knobs_check != test_knobs:
                    raise GpuError(f"{self} knobs not matching after reset {test_knobs_check} != {test_knobs}")

            modified = self.knobs_reset_to_defaults(["all"], True)

            if not cc_or_ppcie:
                for knob, value in test_knobs.items():
                    if self.knob_defaults[knob] != value:
                        if knob not in modified:
                            raise GpuError(f"{self} knob {knob} not modified as expected, test {test_knobs} defaults {self.knob_defaults}")
                for modified_knob in modified:
                    if self.knob_defaults[modified_knob] == test_knobs[modified_knob]:
                        raise GpuError(f"{self} knob {knob} modified unnecessarily, test {test_knobs} defaults {self.knob_defaults}")
            else:
                if set(modified) != set(self.knob_defaults.keys()):
                    raise GpuError(f"{self} CC/PPCIE on but not all knobs were modified, test {modified} defaults {self.knob_defaults}")

            self.reset_with_os()
            debug(f"{self} test knobs modified {modified}")

            current = self.knobs_query(test_knobs.keys())
            if current != self.knob_defaults:
                raise GpuError(f"{self} knobs not matching after reset {current} != {self.knob_defaults}")

        self.reset_with_os()
        modified = self.knobs_set(self.knob_defaults, True)
        if len(modified) != 0:
            self.reset_with_os()

    def query_prc_knobs(self):
        assert self.has_fsp

        self._init_fsp_rpc()

        knob_state = []

        for knob in PrcKnob:
            knob_name = PrcKnob.str_from_knob_id(knob.value)
            try:
                knob_value = self.fsp_rpc.prc_knob_read(knob.value)
            except FspRpcError as err:
                if err.is_invalid_knob_error:
                    knob_state.append((knob_name, "invalid"))
                    continue
                raise
            knob_state.append((knob_name, knob_value))

        return knob_state

    def set_ppcie_mode(self, mode):
        assert self.is_ppcie_query_supported

        ppcie_mode = 0x0
        bar0_decoupler_val = 0x0
        if mode == "on":
            ppcie_mode = 0x1
            # No BAR0 decoupler on switches
            if self.is_gpu():
                bar0_decoupler_val = 0x2
        elif mode == "off":
            pass
        else:
            raise ValueError(f"Invalid mode {mode}")

        self._init_fsp_rpc()

        cc_knob_value = self.fsp_rpc.prc_knob_read(PrcKnob.PRC_KNOB_ID_CCM.value)
        if cc_knob_value == 1:
            info(f"CC is currently active. It will be turned off before switching to PPCIe.")

        if ppcie_mode == 0x1:
            self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_2.value, 0x0)
            self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_4.value, 0x0)
            self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_34.value, 0x0)
            self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_CCD.value, 0x0)
            self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_CCM.value, 0x0)

        self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_BAR0_DECOUPLER.value, bar0_decoupler_val)
        self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_PPCIE.value, ppcie_mode)

    def query_ppcie_settings(self):
        assert self.is_ppcie_query_supported

        self._init_fsp_rpc()

        knobs = [
            ("enable", PrcKnob.PRC_KNOB_ID_PPCIE.value),
            ("enable-allow-inband-control", PrcKnob.PRC_KNOB_ID_PPCIE_ALLOW_INB.value),
        ]

        knob_state = []

        for name, knob_id in knobs:
            knob_value = self.fsp_rpc.prc_knob_read(knob_id)
            knob_state.append((name, knob_value))

        return knob_state

    def test_ppcie_mode_switch(self):
        org_mode = self.query_ppcie_mode()

        self._init_fsp_rpc()
        toggle_2 = self.fsp_rpc.prc_knob_read(PrcKnob.PRC_KNOB_ID_1.value) == 0x1
        toggle_4 = self.fsp_rpc.prc_knob_read(PrcKnob.PRC_KNOB_ID_3.value) == 0x1
        toggle_34 = self.fsp_rpc.prc_knob_read(PrcKnob.PRC_KNOB_ID_33.value) == 0x1
        toggle_cc = self.is_cc_query_supported
        info(f"{self} test PPCIE switching org_mode {org_mode} toggle_2 {toggle_2} toggle_4 {toggle_4} toggle_34 {toggle_34}")

        prev_mode = org_mode

        for iter in range(5):
            for mode in ["on", "off"]:
                debug(f"{self} switching CC to {mode} in iter {iter}")
                if toggle_2 and prev_mode != "on" and iter > 1:
                    self.fsp_rpc.prc_knob_write(PrcKnob.PRC_KNOB_ID_2.value, 0x1)
                if toggle_4 and prev_mode != "on" and iter > 2:
                    self.fsp_rpc.prc_knob_write(PrcKnob.PRC_KNOB_ID_4.value, 0x1)
                if toggle_34 and prev_mode != "on" and iter > 3:
                    self.fsp_rpc.prc_knob_write(PrcKnob.PRC_KNOB_ID_34.value, 0x1)
                if toggle_cc and prev_mode != "on" and iter > 4:
                    self.fsp_rpc.prc_knob_write(PrcKnob.PRC_KNOB_ID_CCM.value, 0x1)
                    self.fsp_rpc.prc_knob_write(PrcKnob.PRC_KNOB_ID_CCD.value, 0x1)

                self.set_ppcie_mode(mode)
                self.reset_with_os()
                new_mode = self.query_ppcie_mode()
                if new_mode != mode:
                    raise GpuError(f"{self} PPCIE mode failed to switch to {mode} in iter {iter}. Current mode is {new_mode}")
                debug(f"{self} PPCIE switched to {mode} in iter {iter}")
                prev_mode = new_mode

        self.set_ppcie_mode(org_mode)
        self.reset_with_os()

    def set_next_sbr_to_fundamental_reset(self):
        assert self.is_reset_coupling_supported

        self._init_fsp_rpc()
        if self.fsp_rpc.prc_knob_read(PrcKnob.PRC_KNOB_ID_FORCE_RESET_COUPLING.value) == 0:
            if self.fsp_rpc.prc_knob_read(PrcKnob.PRC_KNOB_ID_FORCE_RESET_COUPLING_ALLOW_INB.value) == 0:
                error(f"{self} reset coupling is disabled and not allowed to be enabled through in-band. Please enable the 'Force test coupling' permissions through out-of-band APIs.")
                return False

            self.fsp_rpc.prc_knob_write(PrcKnob.PRC_KNOB_ID_FORCE_RESET_COUPLING.value, 1)

        if self.is_hopper:
            reset_regs = [0x91238]
        elif self.is_blackwell_plus:
            reset_regs = [0x80840, 0x80844]
        reset_settings = ", ".join([f"{o:#x} = {self.read_bad_ok(o):#x}" for o in reset_regs])
        debug(f"{self} reset settings before coupling {reset_settings}")
        self.fsp_rpc_mods.prc_couple_reset()
        reset_settings = ", ".join([f"{o:#x} = {self.read_bad_ok(o):#x}" for o in reset_regs])
        debug(f"{self} reset settings after coupling {reset_settings}")

        return True


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

    def __str__(self):
        return self.name

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
    def fbif_ctl2(self):
        return self.fbif_ctl + 0x60

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

class OfaFalcon(GpuFalcon):
    def __init__(self, gpu):
        self.no_outside_reset = True
        super().__init__("ofa", 0x844100, gpu)

    @property
    def fbif_ctl(self):
        return self.base_page + 0x424

    @property
    def fbif_transcfg(self):
        return self.base_page + 0x400

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

from gpu.fsp_mctp import MctpHeader, MctpMessageHeader

class FspRpc(object):
    def __init__(self, fsp_falcon, channel_type, channel_num):
        self.falcon = fsp_falcon
        self.device = self.falcon.device

        if channel_type == "mnoc":
            from gpu import FspMnocRpc
            self.transport = FspMnocRpc(self.device, channel_num)
        elif channel_type == "emem":
            from gpu import FspEmemRpc
            self.transport = FspEmemRpc(fsp_falcon, channel_num)
        else:
            raise ValueError(f"Invalid channel type {channel_type}")

    def __str__(self):
        return f"{self.device} FSP-RPC"


    def send_cmd(self, nvdm_type, data, timeout=5, sync=True, seid=0):
        mctp_header = MctpHeader()
        mctp_header.seid = 0
        mctp_msg_header = MctpMessageHeader()
        max_packet_size = self.transport.max_packet_size_bytes // 4

        mctp_msg_header.nvdm_type = nvdm_type

        total_size = len(data) + mctp_header.size // 4 + mctp_msg_header.size // 4
        if total_size > max_packet_size:
            mctp_header.eom = 0

        pdata = [mctp_header.to_int(), mctp_msg_header.to_int()] + data
        remaining_data = pdata[max_packet_size: ]
        pdata = pdata[:max_packet_size]

        debug(f"{self} sending first packet. Total size {total_size * 4} bytes. First packet {len(pdata) * 4} bytes")
        self.transport.send_data(pdata)

        while len(remaining_data) != 0:
            mctp_header.som = 0
            mctp_header.seq = (mctp_header.seq + 1) % 4
            if len(remaining_data) + mctp_header.size // 4 <= max_packet_size:
                mctp_header.eom = 1
            pdata = [mctp_header.to_int()] + remaining_data
            remaining_data = pdata[max_packet_size:]
            pdata = pdata[:max_packet_size]

            debug(f"Sending extra packet {len(pdata) * 4} bytes remaining data {len(remaining_data) * 4} bytes")
            self.transport.send_data(pdata)

        if not sync:
            return

        mdata = self.transport.receive_data(timeout)
        msize = len(mdata) * 4
        debug(f"{self} response {[hex(d) for d in mdata]}")

        if msize < 5 * 4:
            raise GpuError(f"{self} response size {msize} is smaller than expected. Data {[hex(d) for d in mdata]}")
        mctp_msg_header.from_int(mdata[1])
        if mctp_msg_header.nvdm_type != 0x15:
            raise GpuError(f"{self} message wrong nvdm_type. Data {[hex(d) for d in mdata]}")
        if mdata[3] != nvdm_type:
            raise GpuError(f"{self} message request type 0x{mdata[3]:x} not matching the command 0x{nvdm_type:x}. Data {[hex(d) for d in mdata]}")
        if mdata[4] != 0x0:
            raise FspRpcError(self, mdata[4], mdata)

        return mdata[5:]

    def prc_cmd(self, data, sync=True):
        return self.send_cmd(0x13, data, sync=sync)

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

        prc_data = [prc, prc_1]

        if self.device.is_nvswitch() or self.device.is_gpu() and self.device.is_hopper:
            prc_data.append(prc_2)
        else:
            if prc_2 != 0:
                raise GpuError(f"{self} prc_2 is not 0 for non-hopper device")

        data = self.prc_cmd(prc_data)
        if len(data) != 0:
            raise GpuError(f"RPC wrong response size {len(data)}. Data {[hex(d) for d in data]}")

    def prc_couple_reset(self):
        prc = 0x4
        # One shot
        prc |= 0x1 << 8

        prc |= 0x1 << 16

        data = self.prc_cmd([prc])
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

    def fbdma_enable(self):
        try:
            self.send_cmd(0x22, [0x1], timeout=1)
        except GpuRpcTimeout as err:
            if self.device.is_hopper:
                error(f"{self.device} Enabling DMA timed out. Most likely the GPU needs to be upgraded to FW >=96.00.B3.00.00")
            raise

    def fbdma_disable(self):
        self.send_cmd(0x22, [0x0], timeout=1)

    def inforom_read(self, object_name, object_size, object_offset):
        import struct

        inforom_msg_struct = struct.Struct("<BB3sHH")
        object_name = object_name.encode('utf-8')[:3].ljust(3, b'\x00')
        packed_message = inforom_msg_struct.pack(
            0x03,
            0xFF,
            object_name,
            object_size,
            object_offset
        )

        send_message = []
        for i in range(0, len(packed_message), 4):
            send_message.append(int.from_bytes(packed_message[i:i+4], byteorder='little'))

        return self.send_cmd(0x17, send_message, timeout=5)

    def inforom_write(self, object_name, object_size, object_offset, data):
        import struct

        inforom_msg_struct = struct.Struct("<BB3sHH")
        object_name = object_name.encode('utf-8')[:3].ljust(3, b'\x00')
        packed_message = inforom_msg_struct.pack(
            0x04,
            0xFF,
            object_name,
            object_size,
            object_offset
        )

        send_message = [
            int.from_bytes(packed_message[0:4], byteorder='little'),
            int.from_bytes(packed_message[4:8], byteorder='little')
        ]

        # Get the last byte (`00` from `object_offset`)
        last_byte = packed_message[8]

        # Convert data to bytes
        data_bytes = b''.join(struct.pack("<I", d) for d in data)

        send_message.append(int.from_bytes(bytes([last_byte]) + data_bytes[:3], byteorder='little'))

        # Process remaining `data` bytes in 32-bit chunks
        for i in range(3, len(data_bytes), 4):
            send_message.append(int.from_bytes(data_bytes[i:i+4], byteorder='little'))

        self.send_cmd(0x17, send_message, timeout=5)

    def recreate_inforom_fs(self):
        self.send_cmd(0x17, [0x5], timeout=10)

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
            for off in [0x0, 0x88000, 0x88004, 0x92000, 0x8f0320]:
                debug("%s offset 0x%x = 0x%x", self.bdf, off, self.read_bad_ok(off))
            raise UnknownGpuError("GPU %s %s bar0 %s" % (self.bdf, hex(self.pmcBoot0), hex(self.bar0_addr)))

        props = NVSWITCH_MAP[self.pmcBoot0]
        self.props = props
        self.name = props["name"]
        self.arch = props["arch"]
        self.chip = props.get("chip", None)
        #self.sanity_check()
        self._save_cfg_space()
        self.is_memory_clear_supported = False

        self.bios = None
        self.falcons = None
        self.falcon_dma_initialized = False
        self.falcons_cfg = props.get("falcons_cfg", {})
        self.needs_falcons_cfg = props.get("needs_falcons_cfg", {})

        self.is_ppcie_query_supported = self.is_laguna_plus
        self.is_cc_query_supported = False

        if self.is_ppcie_query_supported:
            self.knob_defaults = {"ppcie": "off"}

        for unit in self.device_units.values():
            unit.create_instance(self)

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
        bar0_data_array = array.array('I')
        for offset in range(0, self.bar0_size, 4):
            if offset % (128 * 1024) == 0:
                debug("Dumped %d bytes so far", offset)
            data = self.bar0.read32(offset)
            bar0_data_array.append(data)
        return memoryview(bar0_data_array.tobytes()).toreadonly()

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

    def wait_for_boot(self, silent_on_failure=False):
        if self.is_laguna_plus:
            try:
                self.poll_register("boot_complete", 0x660bc, 0xff, 5)
            except GpuError as err:
                if silent_on_failure:
                    raise
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

    def query_ppcie_mode(self):
        assert self.is_ppcie_query_supported
        self.wait_for_boot()

        ppcie_reg = self.read(0x28c50)
        ppcie_state = ppcie_reg & 0x1
        if ppcie_state == 0x1:
            return "on"
        else:
            return "off"


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

        self.arch = "unknown"
        self.chip = "unknown"
        self._device_info_instances = None

        arch, chip = GpuProperties.get_chip_family(self.device)
        if arch is not None:
            self.arch = arch
            self.chip = chip
            debug(f"{self} detected as {arch} {chip}")

        if self.chip == "unknown":
            if self.device >= 0x22f0 and self.device < 0x2380:
                self.arch = "hopper"
                self.chip = "gh100"

        if self.chip != "unknown":
            # Default name for unknown GPUs. Known GPUs will override this based
            # on extra properties below
            self.name = f"Generic-{self.chip.upper()}"

            from gpu.regs.core import RegisterInterface
            self.regs = RegisterInterface(self)

        if self.is_hopper_plus:
            self.is_pmu_reset_in_pmc = self.is_pascal_10x_plus
            self.is_memory_clear_supported = self.is_turing_plus

            self.is_forcing_ecc_on_after_reset_supported = self.is_turing_plus
            self.needs_falcons_cfg = not self.is_hopper_plus
            if self.is_hopper_plus:
                # Fill out the prop values as we are bypassing GPU_MAP later for
                # newer GPUs. Over time GPU_MAP will not be the main source of
                # GPU properties and instead device id lookup will be used.
                self.gpu_props = {"other_falcons": ["fsp"], "nvdec": [], "nvenc": []}
                if self.is_hopper:
                    self.gpu_props["nvlink"] = {
                        "number": 18,
                        "links_per_group": 6,
                        "base_offset": 0xa00000,
                        "per_group_offset": 0x40000,
                    }
                self.props = self.gpu_props

        if self.is_blackwell_plus:
            self.wait_for_bar_firewall()

        self.pmcBoot0 = self.read_bad_ok(NV_PMC_BOOT_0)

        if self.pmcBoot0 == 0xffffffff:
            debug("%s sanity check of bar0 failed", self)
            raise BrokenGpuError()

        if self.pmcBoot0 in [0xbadf0200, 0xbad00200]:
            if self.is_blackwell_plus:
                if (0x10de, 0) in self.dvsec_caps:
                    sec_fault = self.config_read_dvsec_cap(0x10de, 0x0, 0x14)
                else:
                    sec_fault = 0xcafebad0
            elif self.is_hopper:
                sec_fault = self.config.read32(0x2b4)

            debug(f"{self} boot {self.pmcBoot0:#x} sec fault {sec_fault:#x}")
            raise BrokenGpuErrorSecFault(self.pmcBoot0, sec_fault)

        if self.chip == "unknown":
            gpu_map_key = self.pmcBoot0
            if gpu_map_key in GPU_MAP_MULTIPLE:
                match = GPU_MAP_MULTIPLE[self.pmcBoot0]
                # Check for a device id match. Fall back to the default, if not found.
                gpu_map_key = GPU_MAP_MULTIPLE[self.pmcBoot0]["devids"].get(self.device, match["default"])

            if gpu_map_key not in GPU_MAP:
                for off in [0x0, 0x88000, 0x88004, 0x92000, 0x8f0320]:
                    debug("%s offset 0x%x = 0x%x", self.bdf, off, self.read_bad_ok(off))
                raise UnknownGpuError("GPU %s %s bar0 %s" % (self.bdf, hex(self.pmcBoot0), hex(self.bar0_addr)))

            self.gpu_props = GPU_MAP[gpu_map_key]
            gpu_props = self.gpu_props
            self.props = gpu_props
            self.name = gpu_props["name"]
            self.arch = gpu_props["arch"]
            self.chip = gpu_props.get("chip", None)
            self.is_pmu_reset_in_pmc = gpu_props["pmu_reset_in_pmc"]
            self.is_memory_clear_supported = gpu_props["memory_clear_supported"]
            self.is_forcing_ecc_on_after_reset_supported = gpu_props["forcing_ecc_on_after_reset_supported"]
        else:
            gpu_props = self.gpu_props

        # Querying ECC state relies on being able to initialize/clear memory
        self.is_ecc_query_supported = self.is_memory_clear_supported
        self.is_cc_query_supported = self.is_hopper_plus
        self.is_ppcie_query_supported = self.is_hopper
        self.is_reset_coupling_supported = self.is_hopper
        self.is_bar0_firewall_supported = self.is_blackwell_plus
        self.is_setting_ecc_after_reset_supported = self.is_ampere_plus
        self.is_mig_mode_supported = self.is_ampere_100
        if not self.sanity_check():
            debug("%s sanity check failed", self)
            raise BrokenGpuError()

        gpu_extra_props = GpuProperties(self.pmcBoot0, self.device, self.ssid).get_properties()
        if gpu_extra_props['name'] != None:
            self.name = gpu_extra_props['name']

        if self.is_hopper:
            self.has_module_id_bit_flip = "has_module_id_bit_flip" in gpu_extra_props["flags"]
        if self.is_hopper_plus:
            self.is_sxm = "is_sxm" in gpu_extra_props["flags"]
            self.is_pcie = "is_pcie" in gpu_extra_props["flags"]
            self.has_c2c = "has_c2c" in gpu_extra_props["flags"]

        self._save_cfg_space()
        self.init_priv_ring()

        for unit in self.device_units.values():
            unit.create_instance(self)

        self.bar0_window_base = 0
        self.bar0_window_initialized = False
        self.bios = None
        self.falcons = None
        self.falcon_for_dma = None
        self.falcons_cfg = gpu_props.get("falcons_cfg", {})
        self.needs_falcons_cfg = gpu_props.get("needs_falcons_cfg", {})
        self.mse = None


        if self.is_ampere_plus:
            graphics_mask = 0
            graphics_bits = [12]
            if self.is_ampere_100:
                graphics_bits += [1, 9, 10, 11, 13, 14, 18]
            for gb in graphics_bits:
                graphics_mask |= (0x1 << gb)

            self.pmc_device_graphics_mask = graphics_mask
        self.hulk_ucode_data = None

        if self.is_turing_plus:
            self.knob_defaults['ecc'] = True
        if self.is_mig_mode_supported:
            self.knob_defaults['mig'] = False
        if self.is_cc_query_supported:
            self.knob_defaults['cc'] = "off"
        if self.is_ppcie_query_supported:
            self.knob_defaults['ppcie'] = "off"

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
    def is_blackwell(self):
        return GPU_ARCHES.index(self.arch) == GPU_ARCHES.index("blackwell")

    @property
    def is_blackwell_plus(self):
        return GPU_ARCHES.index(self.arch) >= GPU_ARCHES.index("blackwell")

    @property
    def is_blackwell_1xx(self):
        return self.is_blackwell and self.chip.startswith("gb1")

    @property
    def is_blackwell_2xx(self):
        return self.is_blackwell and self.chip.startswith("gb2")

    @property
    def is_blackwell_2xx_plus(self):
        if GPU_ARCHES.index(self.arch) > GPU_ARCHES.index("blackwell"):
            return True
        return self.is_blackwell_2xx

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
        if self.is_hopper:
            flags = self.regs.read(self.regs.therm_int.NV_R_NRELBYZA)
            if flags != 0:
                debug(f"{self} boot flags {flags}")
            boot = self.read_bad_ok(0x0)
            if boot == 0:
                debug(f"{self} BAR0 offset 0x0 is 0, which implies recovery mode")
                return True
            return flags.F_UIZAVLDE == 1

        status = self.read(0x8aa128)
        if status & 0xff not in [0x0, 0x1]:
            debug(f"{self} recovery status {status:#x}")
            return True
        return False

    @property
    def is_module_name_supported(self):
        return self.is_sxm

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

        self.reset_post()

        return self.sanity_check()

    def reset_post(self):
        super(Gpu, self).reset_post()

        # Reinit priv ring
        self.init_priv_ring()

        # Reinitialize falcons if they were already initialized
        if self.falcons:
            self.falcons = None
            self.init_falcons()

        if self.mse:
            # After reset, MSE needs to be reinitialized, if used again.
            self.mse.remove_atexit_cleanup()
            self.mse = None

    def get_memory_size(self):
        if self.is_blackwell_plus:
            config_offset = 0x1fa3e0
        else:
            config_offset = 0x100ce0
        if self.is_hopper_plus:
            mag_mask = (1<<(27-4+1)) - 1
        else:
            mag_mask = (1<<(9-4+1)) - 1


        config = self.read(config_offset)
        scale = config & 0xf
        mag = (config >> 4) & mag_mask
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
        if self.is_hopper_plus:
            self.wait_for_boot()
            if self.is_ppcie_query_supported and self.query_ppcie_mode() == "on":
                raise GpuError(f"{self} has PPCIE mode on and querying ECC is blocked")
            if self.is_cc_query_supported and self.query_cc_mode() == "on":
                raise GpuError(f"{self} has CC mode on and querying ECC is blocked")

        # To get the final ECC state, we need to wait for the memory to be
        # fully initialized. clear_memory() guarantees that.
        self.clear_memory()

        return self.get_ecc_state()

    def query_cc_mode_hopper(self):
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
        else:
            return "invalid-devtools-only-fix-by-setting-cc-mode"

    def query_cc_mode_blackwell(self):
        assert self.is_cc_query_supported
        self.wait_for_boot()

        cc_reg = self.read(0x590)
        cc_state = cc_reg & 0x3
        if cc_state == 0x3:
            return "devtools"
        elif cc_state == 0x1:
            return "on"
        elif cc_state == 0x0:
            return "off"
        else:
            return "invalid-devtools-only-fix-by-setting-cc-mode"

    def query_cc_mode(self):
        assert self.is_cc_query_supported

        if self.is_hopper:
            return self.query_cc_mode_hopper()

        if self.is_blackwell_plus:
            return self.query_cc_mode_blackwell()

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

        ppcie_supported = self.is_ppcie_query_supported
        if ppcie_supported:
            try:
                ppcie_knob_value = self.fsp_rpc.prc_knob_read(PrcKnob.PRC_KNOB_ID_PPCIE.value)
                if ppcie_knob_value == 1:
                    info(f"{self} has PPCIe enabled. It will be turned off before switching to CC.")
            except FspRpcError as err:
                if err.is_invalid_knob_error:
                    debug(f"{self} has older FW that doesn't support PPCIE.")
                    ppcie_supported = False
                else:
                    raise

        if cc_mode == 0x1:
            self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_2.value, 0x0)
            self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_4.value, 0x0)
            self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_34.value, 0x0)
            if ppcie_supported:
                self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_PPCIE.value, 0x0)

        if self.is_hopper:
            self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_BAR0_DECOUPLER.value, bar0_decoupler_val)

        # Always enable CC mode knob first and disable it last. This prevents us
        # from entering an invalid CC mode state where only the CCD knob is
        # enabled.
        if cc_mode == 0x1:
            self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_CCM.value, cc_mode)
            self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_CCD.value, cc_dev_mode)
        else:
            self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_CCD.value, cc_dev_mode)
            self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_CCM.value, cc_mode)

    def query_bar0_firewall_mode(self):
        assert self.is_bar0_firewall_supported

        config = self.read(0x590)
        if config & 0x4 == 0x4:
            return "on"
        else:
            return "off"

    def set_bar0_firewall_mode(self, mode):
        assert self.is_bar0_firewall_supported

        self._init_fsp_rpc()
        if mode == "on":
            bar0_decoupler_val = 0x2
        else:
            bar0_decoupler_val = 0x0

        self.fsp_rpc.prc_knob_check_and_write(PrcKnob.PRC_KNOB_ID_BAR0_DECOUPLER.value, bar0_decoupler_val)


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

    def query_ppcie_mode(self):
        assert self.is_ppcie_query_supported
        self.wait_for_boot()

        ppcie_reg = self.read(0x1182cc)
        ppcie_state = ppcie_reg & 0x20
        if ppcie_state == 0x20:
            return "on"
        else:
            return "off"


    def is_boot_done(self):
        assert self.is_turing_plus
        if self.is_hopper_plus:
            return self.regs.is_set(self.regs.therm.NV_THERM_I2CS_SCRATCH_FSP_BOOT_COMPLETE_STATUS_SUCCESS)
        else:
            data = self.read(0x118234)
            if data == 0x3ff:
                return True
        return False

    def wait_for_bar_firewall(self):
        assert self.is_blackwell_plus

        if (0x10de, 0) in self.dvsec_caps:
            mask = (0x1<<20)
            self.poll_register("bar_firewall", self.dvsec_caps[0x10de, 0] + 0x8, value=0x0, timeout=5, mask=mask, read_function=self.config.read32)
            return

        if self.read_bad_ok(0) == 0xffffffff:
            warning(f"{self} missing the DVSEC 10de:0 cap and BAR firewall might be active, falling back to 3s sleep")
            time.sleep(3)
            if self.read_bad_ok(0) == 0xffffffff:
                error(f"{self} BAR0 still broken after 3s sleep")
                raise BrokenGpuErrorWithInfo("Blackwell+ BAR0 not accessible and no DVSEC 10de:0 cap exposed")

    def wait_for_boot(self, silent_on_failure=False):
        assert self.is_turing_plus
        if self.is_hopper_plus:
            offset = self.regs.therm.NV_THERM_I2CS_SCRATCH_FSP_BOOT_COMPLETE.address

            badf_ok = self.is_blackwell
            timeout_value = 5
            try:
                if self.is_blackwell_plus:
                    timeout_value = 10
                    self.wait_for_bar_firewall()

                self.poll_register("boot_complete", offset, 0xff, timeout_value, badf_ok=badf_ok)
            except GpuError as err:
                if silent_on_failure:
                    raise
                _, _, tb = sys.exc_info()
                debug("{} boot not done 0x{:x} = 0x{:x}".format(self, offset, self.read(offset)))
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

        if self.is_blackwell_plus:
            self.poll_register("memory_clear_finished", 0x8a004c, 0x1, 5)
        else:
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
            self.reset_with_os()
            new_state = self.query_final_ecc_state()
            if org_state == new_state:
                raise GpuError("{0} ECC mode failed to switch from {1} to {2}".format(self, org_state, new_state))
            self.set_ecc_mode_after_reset(org_state)
            self.reset_with_os()
            new_state = self.query_final_ecc_state()
            if org_state != new_state:
                raise GpuError("{0} ECC mode failed to switch back to original state {1}".format(self, org_state))
        else:
            self.force_ecc_on_after_reset()
            self.reset_with_os()
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
        if self.is_turing_plus and not self.is_hopper_plus:
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
            assert not self.is_hopper_plus
            new_window |= NV_BAR0_WINDOW_CFG_TARGET_SYSMEM_COHERENT
        if self.bar0_window_base != new_window:
            self.bar0_window_base = new_window
            if self.is_hopper_plus:
                self.write(0x10fd40, new_window)
            else:
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

        self.write(falcon.fbif_ctl2, 1)

        ctl = self.bitfield(falcon.fbif_ctl)
        ctl[4:5] = 1
        ctl[7:8] = 1

        dmactl = self.bitfield(falcon.dmactl)
        dmactl[0:1] = 0

        self.write(falcon.base_page + 0x110, (address >> 8) & 0xffffffff)
        self.write(falcon.base_page + 0x128, (address >> 40) & 0xffffffff)

        # GPU DMA supports 47-bits, but top bits can be globally forced. This
        # only works if there are no other DMAs happening at the same time.
        self.write(0x100f04, (address >> 47) & 0xffffffff)

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
        if self.falcon_for_dma:
            return self.falcon_for_dma

        self.init_falcons()

        if self.is_hopper_plus:
            self._init_fsp_rpc()
            self.fsp_rpc.fbdma_enable()

            if self.read_bad_ok(0x8443c0) >> 16 == 0xbadf:
                raise GpuError(f"{self} does not support DMA with current FW.")

            falcon = OfaFalcon(self)
        elif hasattr(self, "gsp"):
            falcon = self.gsp
        else:
            falcon = self.pmu

        self.set_bus_master(True)
        if self.is_memory_clear_supported:
            self.clear_memory()

        if not falcon.no_outside_reset:
            falcon.reset()

        debug(f"{self} Using falcon {falcon} for DMA")

        self.falcon_for_dma = falcon
        return falcon

    def dma_sys_write(self, address, data):
        falcon = self._falcon_dma_init()
        falcon.load_dmem(data, phys_base=0)
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
        if not self.is_blackwell_plus:
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


    def read_module_id_h100(self):

        assert self.is_sxm

        gpios = [0x9, 0x11, 0x12]

        mod_id = 0
        for i, gpio in enumerate(gpios):
            bit = (self.read(0x21200 + 4 * gpio) >> 14) & 0x1
            mod_id |= bit << i

        if self.has_module_id_bit_flip:
            mod_id ^= 0x4

        return mod_id

    def read_module_id_b100(self):
        assert self.is_blackwell
        self.init_mse()
        info = self.mse.get_platform_info()
        return info.moduleId

    def read_module_id(self):
        if self.is_hopper and self.is_sxm:
            return self.read_module_id_h100()
        elif self.is_blackwell and self.is_sxm:
            return self.read_module_id_b100()
        else:
            raise GpuError(f"{self} unknown module id")

    def init_mse(self):
        assert self.is_blackwell_plus

        from gpu import MseRpc
        if not self.mse:
            self.wait_for_boot()
            self.mse = MseRpc(self)

        return self.mse

    def debug_dump(self):
        offsets = []
        if self.is_hopper_plus:
            offsets.append(("boot_status", 0x200bc))
            offsets.append(("boot_flags", 0x20120))
            for i in range(4):
                offsets.append((f"fsp_status_{i}", 0x8f0320 + i * 4))
            if self.is_hopper:
                offsets.append((f"last_reset", 0x9128c))
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

        for unit in self.units.values():
            info(f"{unit} debug print")
            unit.debug_print()


    def __str__(self):
        return "GPU %s %s %s BAR0 0x%x" % (self.bdf, self.name, hex(self.device), self.bar0_addr)

    def __eq__(self, other):
        return self.bar0_addr == other.bar0_addr

# Inject the Gpu class into devices module.
# TODO: clean this up once Gpu is refactored out.
from pci import devices
devices.NvidiaDevice = NvidiaDevice
devices.Gpu = Gpu
devices.NvSwitch = NvSwitch
devices.BrokenGpu = BrokenGpu


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

# TEMP: Provide backwards compatibility to old code expecting find_gpus in the
# top level namespace.
find_gpus = devices.find_gpus

import cli.main

if __name__ == "__main__":
    cli.main.main()
else:
    cli.main.init()
