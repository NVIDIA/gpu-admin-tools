#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Linux VFIO UAPI definitions shared by MMIO access and driver management.

Kept separate from platform-specific operations so CLI discovery also works
on hosts without fcntl. See include/uapi/linux/vfio.h.
"""

import ctypes


VFIO_API_VERSION = 0
VFIO_TYPE1_IOMMU = 1
VFIO_TYPE1v2_IOMMU = 3

# VFIO uses _IO(';', 100 + n), including for requests carrying structures.
# These ioctl encodings are valid on the architectures supported below.
VFIO_GET_API_VERSION = 0x3b64
VFIO_CHECK_EXTENSION = 0x3b65
VFIO_SET_IOMMU = 0x3b66
VFIO_GROUP_GET_STATUS = 0x3b67
VFIO_GROUP_SET_CONTAINER = 0x3b68
VFIO_GROUP_UNSET_CONTAINER = 0x3b69
VFIO_GROUP_GET_DEVICE_FD = 0x3b6a
VFIO_DEVICE_GET_INFO = 0x3b6b
VFIO_DEVICE_GET_REGION_INFO = 0x3b6c

VFIO_GROUP_FLAGS_VIABLE = 1 << 0
VFIO_GROUP_FLAGS_CONTAINER_SET = 1 << 1
VFIO_DEVICE_FLAGS_PCI = 1 << 1
VFIO_REGION_INFO_FLAG_READ = 1 << 0
VFIO_REGION_INFO_FLAG_WRITE = 1 << 1
VFIO_REGION_INFO_FLAG_MMAP = 1 << 2
VFIO_REGION_INFO_FLAG_CAPS = 1 << 3
VFIO_REGION_INFO_CAP_SPARSE_MMAP = 1
VFIO_PCI_CONFIG_REGION_INDEX = 7


class VfioGroupStatus(ctypes.Structure):
    _fields_ = [("argsz", ctypes.c_uint32), ("flags", ctypes.c_uint32)]


class VfioDeviceInfo(ctypes.Structure):
    # The original fixed prefix is sufficient; device capabilities are unused.
    _fields_ = [("argsz", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("num_regions", ctypes.c_uint32), ("num_irqs", ctypes.c_uint32)]


class VfioRegionInfo(ctypes.Structure):
    _fields_ = [("argsz", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("index", ctypes.c_uint32), ("cap_offset", ctypes.c_uint32),
                ("size", ctypes.c_uint64), ("offset", ctypes.c_uint64)]
