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

import os

def sysfs_find_devices(filter_vendor = None, filter_devid = None):
    dev_paths = []
    for device_dir in os.listdir("/sys/bus/pci/devices/"):
        dev_path = os.path.join("/sys/bus/pci/devices/", device_dir)
        if filter_vendor is not None:
            vendor = open(os.path.join(dev_path, "vendor")).readlines()
            vendor = int(vendor[0].strip(), base=16)
            if vendor != filter_vendor:
                continue
            if filter_devid is not None:
                devid = open(os.path.join(dev_path, "device")).readlines()
                devid = int(devid[0].strip(), base=16)
                if devid != filter_devid:
                    continue
        dev_paths.append(dev_path)
    return dev_paths

def sysfs_find_parent(sysfs_device_path):
    # Get a sysfs path with PCIe topology like:
    # /sys/devices/pci0000:00/0000:00:0d.0/0000:05:00.0/0000:06:00.0/0000:07:00.0
    topo_path = os.path.realpath(sysfs_device_path)

    parent = os.path.dirname(topo_path)

    # No more parents once we reach /sys/devices/pci*
    if os.path.basename(parent).startswith("pci"):
        return None

    return parent

def sysfs_find_topo_bdfs(sysfs_device_path):
    # Get a sysfs path with PCIe topology like:
    # /sys/devices/pci0000:00/0000:00:0d.0/0000:05:00.0/0000:06:00.0/0000:07:00.0
    topo_path = os.path.realpath(sysfs_device_path)

    # Split the path and skip /sys/devices/pci... prefix
    topo_bdfs = topo_path.split("/")[4:]
    topo_bdfs.reverse()
    return topo_bdfs
