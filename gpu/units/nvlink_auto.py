#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from ..unit import GpuUnitAutoBase

class NvlinkAuto(GpuUnitAutoBase):
    name = "nvlink"

    @classmethod
    def create_instance(cls, device):
        if device.is_nvswitch():
            if device.is_laguna_plus:
                from .nvlink_hopper import LagunaNvlink
                return LagunaNvlink(device)
            return None

        if device.is_blackwell_plus:
            from .nvlink_blackwell import BlackwellNvlink
            return BlackwellNvlink(device)

        if device.is_hopper_plus:
            from .nvlink_hopper import HopperNvlink
            return HopperNvlink(device)

        if device.is_ampere_100:
            from .nvlink_ampere import AmpereNvlink
            return AmpereNvlink(device)

        return None
