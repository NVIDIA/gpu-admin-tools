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

import os
from logging import error, warning
import sys
import traceback

from utils import FileRaw
from .device import Device, PciDevice
from .bridge import PciBridge, PlxBridge, IntelRootPort
from .cx7 import Cx7
from gpu import UnknownGpuError, BrokenGpuError, BrokenGpuErrorWithInfo, BrokenGpuErrorSecFault
from utils.sysfs import find_dev_path_by_bdf, sorted_dev_paths, find_dev_paths, read_number_from_file
from utils import platform_config
def parse_array_index(index_str, array):
    """Parse array index/slice notation and return selected items from array.

    Args:
        index_str: String containing the index/slice notation (e.g. "1", "1:5", "::2")
        array: List to index/slice from

    Returns:
        List containing selected items from the array

    Supports Python slice notation including start:stop:step
    """
    if ':' in index_str:  # Handle slicing
        start, stop, step = None, None, None
        parts = index_str.split(':')
        if len(parts) >= 1 and parts[0]: start = int(parts[0])
        if len(parts) >= 2 and parts[1]: stop = int(parts[1])
        if len(parts) >= 3 and parts[2]: step = int(parts[2])
        return array[start:stop:step]
    else:  # Handle single index
        idx = int(index_str)
        if idx < len(array):
            return [array[idx]]
        else:
            error(f"Index {idx} out of range for array of length {len(array)}")
            return []

def find_devices_from_string(devices_str):
    """Parse a device string and return matching PCI devices.

    Args:
        devices_str: String specifying devices to find. Supports multiple comma-separated specifiers:
            - "gpus" - Find all NVIDIA GPUs
            - "gpus[n]" - Find nth NVIDIA GPU
            - "gpus[n:m]" - Find NVIDIA GPUs from index n to m
            - "nvswitches" - Find all NVIDIA NVSwitches
            - "nvswitches[n]" - Find nth NVIDIA NVSwitch
            - "vendor:device" - Find devices matching 4-digit hex vendor:device ID
            - "domain:bus:device.function" - Find device at specific BDF address

    Returns:
        List of PciDevice objects matching the specifications

    Examples:
        find_devices_from_string("gpus[0]") # First GPU
        find_devices_from_string("gpus[0:2]") # First two GPUs
        find_devices_from_string("10de:2204,10de:1111")
        find_devices_from_string("0000:65:00.0") # Device at BDF 0000:65:00.0
    """
    devices = []

    for dev_str in devices_str.split(','):
        dev_str = dev_str.strip()

        # Split into base string and optional array index
        base_str = dev_str.split('[')[0]
        idx_str = None
        if '[' in dev_str:
            idx_str = dev_str[dev_str.index('[')+1:dev_str.index(']')]

        # Handle special keywords
        if base_str == "gpus":
            matching_paths = find_dev_paths(vendor_ids=[0x10de], device_ids=None, class_ids=[0x030000, 0x030200])
            if not matching_paths:
                warning("No GPU devices found")
                continue

        elif base_str == "nvswitches":
            matching_paths = find_dev_paths(vendor_ids=[0x10de], device_ids=None, class_ids=[0x068000])
            if not matching_paths:
                warning("No NVSwitch devices found")
                continue

        # Handle vendor:device format
        elif ':' in base_str and len(base_str.split(':')) == 2 and all(len(x) == 4 for x in base_str.split(':')):
            vendor_id = int(base_str.split(':')[0], 16)
            device_id = int(base_str.split(':')[1], 16)

            matching_paths = find_dev_paths(vendor_ids=[vendor_id], device_ids=[device_id], class_ids=None)
            if not matching_paths:
                warning(f"No devices found matching vendor:device {vendor_id:x}:{device_id:x}")
                continue

        # Handle BDF format
        elif ':' in base_str:
            dev_path = find_dev_path_by_bdf(base_str)
            if dev_path:
                matching_paths = [dev_path]
            else:
                warning(f"No device found at BDF {base_str}")
                continue
        else:
            raise ValueError(f"Unknown device specifier: {dev_str}")

        # Sort the dev paths before applying any indexing
        matching_paths = sorted_dev_paths(matching_paths)

        if idx_str is not None:
            matching_paths = parse_array_index(idx_str, matching_paths)

        # Intialize the devices
        devices.extend([PciDevices.find_or_init(p) for p in matching_paths])

    return devices

class PciDevices:
    DEVICES = {}


    @staticmethod
    def _open_config(dev_path):
        dev_path_config = os.path.join(dev_path, "config")
        return FileRaw(dev_path_config, 0, os.path.getsize(dev_path_config))

    @classmethod
    def find_class_for_device(cls, dev_path):
        vendor = read_number_from_file(os.path.join(dev_path, "vendor"))

        # Detect NVIDIA GPUs and NvSwitches by looking at the vendor and class
        if vendor == 0x10de:
            class_id = read_number_from_file(os.path.join(dev_path, "class"))
            if class_id in [0x030000, 0x030200]:
                return Gpu
            if class_id == 0x068000:
                return NvSwitch

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
                if pci_dev.vendor == 0x15b3 and pci_dev.device == 0x1979:
                    return Cx7
                return PciBridge

            # Downstream port
            if pci_dev.pciflags["TYPE"] == 0x6:
                if pci_dev.config.size >= 4096 and pci_dev.vendor == 0x10b5:
                    return PlxBridge
                if pci_dev.vendor == 0x15b3 and pci_dev.device == 0x1979:
                    return Cx7
                return PciBridge

            # Endpoint
            if pci_dev.pciflags["TYPE"] == 0x0:
                return PciDevice

        if pci_dev.header_type == 0x1:
            return PciBridge
        else:
            return PciDevice

    @classmethod
    def init_dispatch(cls, dev_path):
        dev_cls = cls.find_class_for_device(dev_path)
        if dev_cls:
            try:
                dev = dev_cls(dev_path=dev_path)
            except UnknownGpuError as err:
                error("Unknown Nvidia device %s: %s", dev_path, str(err))
                dev = NvidiaDevice(dev_path=dev_path)
            except BrokenGpuErrorWithInfo as err:
                error(f"Device {dev_path} broken: {err.err_info}")
                dev = BrokenGpu(dev_path=dev_path, err_info=err.err_info)
            except BrokenGpuErrorSecFault as err:
                error(f"Device {dev_path} in sec fault boot={err.boot:#x} sec_fault={err.sec_fault:#x}")
                dev = BrokenGpu(dev_path=dev_path, sec_fault=err.sec_fault)
            except BrokenGpuError as err:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)
                error(f"Device {dev_path} broken {err}")
                dev = BrokenGpu(dev_path=dev_path)
            return dev
        return None

    @classmethod
    def find_or_init(cls, dev_path):
        if dev_path == None:
            if -1 not in cls.DEVICES:
                cls.DEVICES[-1] = Device()
            return cls.DEVICES[-1]
        bdf = os.path.basename(dev_path)
        if bdf in cls.DEVICES:
            return cls.DEVICES[bdf]
        dev = cls.init_dispatch(dev_path)
        cls.DEVICES[bdf] = dev
        return dev

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
        except BrokenGpuErrorWithInfo as err:
            error(f"Device {dev_path} broken: {err.err_info}")
            dev = BrokenGpu(dev_path=dev_path, err_info=err.err_info)
        except BrokenGpuErrorSecFault as err:
            error(f"Device {dev_path} in sec fault boot={err.boot:#x} sec_fault={err.sec_fault:#x}")
            dev = BrokenGpu(dev_path=dev_path, sec_fault=err.sec_fault)
        except Exception as err:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            error(f"Device {dev_path} broken {err}")
            dev = BrokenGpu(dev_path=dev_path)
        gpus.append(dev)

    return (gpus, other)


def find_gpus(bdf=None):
    if platform_config.is_sysfs_available:
        return find_gpus_sysfs(bdf)
    assert False
    return []
