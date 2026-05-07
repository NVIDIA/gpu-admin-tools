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

from gpu.unit import GpuUnitAutoBase


class GpuTopAuto(GpuUnitAutoBase):
    name = "top"
    order = 0  # Initialize first - other units may depend on device info

    @classmethod
    def create_instance(cls, device):
        if not device.is_gpu() or not device.is_hopper_plus:
            return None


        from gpu.units.top import GpuTopHopper
        return GpuTopHopper(device)


class NvlinkAuto(GpuUnitAutoBase):
    name = "nvlink"

    @classmethod
    def create_instance(cls, device):
        if device.is_nvswitch():
            if device.is_laguna_plus:
                from gpu.units.nvlink_hopper import LagunaNvlink
                return LagunaNvlink(device)
            return None

        if device.is_blackwell_plus:
            from gpu.units.nvlink_blackwell import BlackwellNvlink
            return BlackwellNvlink(device)

        if device.is_hopper_plus:
            from gpu.units.nvlink_hopper import HopperNvlink
            return HopperNvlink(device)

        if device.is_ampere_100:
            from gpu.units.nvlink_ampere import AmpereNvlink
            return AmpereNvlink(device)

        return None


class GpuC2CAuto(GpuUnitAutoBase):
    name = "c2c"

    @classmethod
    def create_instance(cls, device):
        if not device.is_gpu():
            return None

        if not device.has_c2c:
            return None

        if device.is_blackwell_plus:
            from gpu.units.c2c import GpuC2CBlackwell
            return GpuC2CBlackwell(device)

        from gpu.units.c2c import GpuC2C
        return GpuC2C(device)


class MiscDebugAuto(GpuUnitAutoBase):
    name = "misc"

    @classmethod
    def create_instance(cls, device):
        if not device.is_gpu():
            return None
        from gpu.units.misc_debug import MiscDebug
        return MiscDebug(device)
