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
import collections
from ..device_info import GpuDeviceInfo
from ..error import GpuError
from .top import GpuTop, DeviceTypes


class GpuTopRubin(GpuTop):
    """Rubin+ TOP using ZB device info with linked-list traversal."""

    def __init__(self, gpu):
        super().__init__(gpu)
        self.device_types = DeviceTypes(gpu.regs.top_zb.NV_PTOP_ZB_DEVICE_INFO_DEV_TYPE_ENUM)
        self.primary_ptop_pri_base = int(gpu.regs.top_zb.NV_PTOP_ZB_NEXT_PTOP_PRI_BASE_VALUE_UGD0)
        self.null_ptop_pri_base = int(gpu.regs.top_zb.NV_PTOP_ZB_NEXT_PTOP_PRI_BASE_VALUE_NULL)

    @property
    def device_info_instances(self):
        if self._device_info_instances is not None:
            return self._device_info_instances

        self._device_info_instances = collections.defaultdict(list)

        next_ptop_pri_base = self.primary_ptop_pri_base

        while next_ptop_pri_base != self.null_ptop_pri_base:
            if next_ptop_pri_base == 0:
                cc_mode = self.device.query_cc_mode()
                if cc_mode == "on":
                    return self._device_info_instances
                self._device_info_instances = None
                raise GpuError(f"{self.device} PTOP returned zero PRI base with CC mode {cc_mode}")

            cfg = self.regs.read(self.regs.top_zb.NV_PTOP_ZB_DEVICE_INFO_CFG, base=next_ptop_pri_base)
            num_rows = cfg.NUM_ROWS.value

            row_idx = 0
            while row_idx < num_rows:
                row = self.regs.read(self.regs.top_zb.NV_PTOP_ZB_DEVICE_INFO(row_idx), base=next_ptop_pri_base)
                row_idx += 1

                # Skip invalid rows (value 0)
                if row.value == 0:
                    continue

                # Found a valid device entry - collect all rows for this device
                device = row.value

                # Read second row (always present for valid devices)
                row2 = self.regs.read(self.regs.top_zb.NV_PTOP_ZB_DEVICE_INFO(row_idx), base=next_ptop_pri_base)
                row_idx += 1
                device |= (row2.value << 32)

                # Get number of extra rows from second row (DEVICE_ENTRY_EXTRA is at bits 33:32)
                num_extra_rows = self.regs.top_zb.NV_PTOP_ZB_DEVICE_INFO_DEV_DEVICE_ENTRY_EXTRA.raw_value_from_int(device)

                # Read extra rows
                for i in range(num_extra_rows):
                    extra_row = self.regs.read(self.regs.top_zb.NV_PTOP_ZB_DEVICE_INFO(row_idx), base=next_ptop_pri_base)
                    row_idx += 1
                    device |= (extra_row.value << (64 + i * 32))

                regs = self.regs.top_zb
                type = regs.NV_PTOP_ZB_DEVICE_INFO_DEV_TYPE_ENUM.raw_value_from_int(device)
                instance = regs.NV_PTOP_ZB_DEVICE_INFO_DEV_INSTANCE_ID.raw_value_from_int(device)
                pri_base = regs.NV_PTOP_ZB_DEVICE_INFO_DEV_DEVICE_PRI_BASE.raw_value_from_int(device)
                pri_base <<= (regs.NV_PTOP_ZB_DEVICE_INFO_DEV_DEVICE_PRI_BASE.lsb & 31)

                info = GpuDeviceInfo(type, instance, pri_base)
                self._device_info_instances[info.type].append(info)

            # Get next PTOP base for linked-list traversal
            next_ptop_pri_base = self.regs.read(self.regs.top_zb.NV_PTOP_ZB_NEXT_PTOP_PRI_BASE, base=next_ptop_pri_base).value

        return self._device_info_instances
