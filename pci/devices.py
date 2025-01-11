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
import platform

from utils import FileRaw
from .device import Device, PciDevice
from .bridge import PciBridge, PlxBridge, IntelRootPort
from .cx7 import Cx7

class PciDevices:
    DEVICES = {}


    @staticmethod
    def _open_config(dev_path):
        dev_path_config = os.path.join(dev_path, "config")
        return FileRaw(dev_path_config, 0, os.path.getsize(dev_path_config))

    @classmethod
    def find_class_for_device(cls, dev_path):
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
                if pci_dev.vendor == 0x10de:
                    # WAR: Gpu is injected by nvidia_gpu_tools.py, pending being
                    # refactored out to its own module.
                    return Gpu

        if pci_dev.header_type == 0x1:
            return PciBridge
        else:
            if pci_dev.vendor == 0x10de:
                return Gpu
            return PciDevice

    @classmethod
    def init_dispatch(cls, dev_path):
        dev_cls = cls.find_class_for_device(dev_path)
        if dev_cls:
            return dev_cls(dev_path)
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
