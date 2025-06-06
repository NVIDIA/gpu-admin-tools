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

def pci_rescan():
    with open("/sys/bus/pci/rescan", "w") as rf:
        rf.write("1")

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

def find_dev_path_by_bdf(bdf_pattern):
    for device_dir in os.listdir("/sys/bus/pci/devices/"):
        dev_path = os.path.join("/sys/bus/pci/devices/", device_dir)
        bdf = device_dir
        if bdf_pattern in bdf:
            return dev_path
    return None

def read_number_from_file(path):
    with open(path) as f:
        value = f.readline().strip()
        return int(value, 0)

def sorted_dev_paths(paths):
    def devpath_to_id(dev_path):
        bdf = os.path.basename(dev_path)
        return int(bdf.replace(":","").replace(".",""), base=16)

    return sorted(paths, key=devpath_to_id)

def find_dev_paths(vendor_ids, device_ids, class_ids):
    """Find all PCI device paths in sysfs matching the specified criteria.

    Args:
        vendor_ids: List of vendor IDs to match, or None to ignore
        device_ids: List of device IDs to match, or None to ignore
        class_ids: List of class IDs to match, or None to ignore

    Returns:
        List of matching device paths

    At least one of vendor_ids, device_ids or class_ids must be non-None.
    """
    if vendor_ids is None and device_ids is None and class_ids is None:
        raise ValueError("At least one of vendor_ids, device_ids or class_ids must be specified")

    matching_paths = []

    for device_dir in os.listdir("/sys/bus/pci/devices/"):
        dev_path = os.path.join("/sys/bus/pci/devices/", device_dir)

        if vendor_ids is not None:
            vendor = read_number_from_file(os.path.join(dev_path, "vendor"))
            if vendor not in vendor_ids:
                continue

        if device_ids is not None:
            device = read_number_from_file(os.path.join(dev_path, "device"))
            if device not in device_ids:
                continue

        if class_ids is not None:
            cls = read_number_from_file(os.path.join(dev_path, "class"))
            if cls not in class_ids:
                continue
        matching_paths.append(dev_path)

    return matching_paths
