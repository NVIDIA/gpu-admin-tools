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
from logging import error
import sys
import traceback

from utils import FileRaw
from .device import Device, PciDevice
from .bridge import PciBridge, PlxBridge, IntelRootPort
from .cx7 import Cx7
from gpu import UnknownGpuError, BrokenGpuError, BrokenGpuErrorWithInfo, BrokenGpuErrorSecFault
from utils.sysfs import read_number_from_file
from utils import platform_config


def discover_devices(devices=None, gpu_bdf=None):
    """Read PCI identities, using BDF selectors to narrow sysfs discovery."""
    if platform_config.is_sysfs_available:
        from .discovery import discover_sysfs_devices
        return discover_sysfs_devices(devices=devices, gpu_bdf=gpu_bdf)
    raise ValueError("No PCI discovery backend is available")


def initialize_devices(infos):
    """Initialize only selected devices, checking backend limits first."""
    selected = {}
    for info in infos:
        selected.setdefault(info.bdf, info)
    infos = list(selected.values())
    if not infos:
        return []
    if platform_config.is_sysfs_available:
        for info in infos:
            if info.dev_path is None:
                raise ValueError(f"Device {info.bdf} has no sysfs path")
        return [PciDevices.find_or_init(info.dev_path) for info in infos]
    raise ValueError("No PCI initialization backend is available")


def find_devices_from_string(devices_str):
    """Compatibility wrapper: discover, select, then initialize PCI devices."""
    from .discovery import select_devices
    infos = select_devices(discover_devices(devices=devices_str), devices=devices_str)
    return initialize_devices(infos)


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
    """Compatibility inventory of initialized NVIDIA devices."""
    from .discovery import discover_sysfs_devices
    infos = [info for info in discover_sysfs_devices(gpu_bdf=bdf_pattern)
             if (info.is_gpu() or info.is_nvswitch()) and
             (not bdf_pattern or bdf_pattern.strip().lower() in info.bdf)]
    gpus, other = [], []
    for info in infos:
        try:
            device = initialize_devices([info])[0]
        except Exception as err:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            error("Device %s broken: %s", info.dev_path, err)
            device = BrokenGpu(dev_path=info.dev_path)
        if device.is_gpu() or device.is_nvswitch():
            gpus.append(device)
        else:
            other.append(device)
    return gpus, other


def find_gpus(bdf=None):
    if platform_config.is_sysfs_available:
        return find_gpus_sysfs(bdf)
    assert False
    return []
