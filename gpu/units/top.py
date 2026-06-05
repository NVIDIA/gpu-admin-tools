#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import collections
from ..unit import GpuUnit
from ..device_info import GpuDeviceInfo


class DeviceTypes:
    """Short-name access to device type enum values.

    Strips the register prefix so consumers can use e.g.
    gpu.top.device_types.NVLPW instead of knowing the register naming.
    """
    def __init__(self, enum_field):
        prefix = enum_field.name + '_'
        for name, val in enum_field.values.items():
            short = name[len(prefix):] if name.startswith(prefix) else name
            setattr(self, short, val.value)


class GpuTop(GpuUnit):
    name = "top"
    device_types = None  # Set by subclass to DeviceTypes instance

    def __init__(self, gpu):
        super().__init__(gpu)
        gpu.top = self
        self.regs = gpu.regs
        self._device_info_instances = None
        self.num_fbpas = int(self.regs.read(self.regs.top_int.NV_R_WTQAYGOT, check_bad=True).VALUE)
        self.num_ltcs = int(self.regs.read(self.regs.top_int.NV_R_RLGXXLDD, check_bad=True).VALUE)
        self.num_slices_per_ltc = int(
            self.regs.read(self.regs.top_int.NV_R_KMZTDFMK, check_bad=True).VALUE
        )


class GpuTopHopper(GpuTop):
    """Hopper/Blackwell TOP using DEVICE_INFO2 array."""

    def __init__(self, gpu):
        super().__init__(gpu)
        self.device_types = DeviceTypes(gpu.regs.top.NV_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM)

    @property
    def device_info_instances(self):
        if self._device_info_instances is not None:
            return self._device_info_instances

        self._device_info_instances = collections.defaultdict(list)
        num_rows = self.regs.read(self.regs.top.NV_PTOP_DEVICE_INFO_CFG_NUM_ROWS)

        in_chain = False
        device = 0
        device_offset = 0
        for i in range(0, num_rows):
            data = self.regs.read(self.regs.top.NV_PTOP_DEVICE_INFO2(i))
            if in_chain or data != 0:
                device |= (data.value << device_offset)
                device_offset += 32
            in_chain = data.ROW_CHAIN == 1
            if not in_chain and device != 0:
                regs = self.regs.top
                type = regs.NV_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM.raw_value_from_int(device)
                instance = regs.NV_PTOP_DEVICE_INFO2_DEV_INSTANCE_ID.raw_value_from_int(device)
                pri_base = regs.NV_PTOP_DEVICE_INFO2_DEV_DEVICE_PRI_BASE.raw_value_from_int(device)
                pri_base <<= (regs.NV_PTOP_DEVICE_INFO2_DEV_DEVICE_PRI_BASE.lsb & 31)

                info = GpuDeviceInfo(type, instance, pri_base)
                self._device_info_instances[info.type].append(info)
                device = 0
                device_offset = 0

        return self._device_info_instances
