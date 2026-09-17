#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .devid_names import GPU_NAME_BY_DEVID, GPU_NAME_BY_DEVID_SSID
from .devid_properties import GPU_PROPS_BY_DEVID
from .devid_chips import GPU_DEVID_CHIPS

# Matches NVSWITCH_MAP in nvidia_gpu_tools.py.
_NVSWITCH_METADATA = {
    0x1AF1: ("LR10", "limerock", "lr10"),
    0x22A3: ("NVSwitch_gen3", "laguna", "ls10"),
}


class GpuProperties:
    def __init__(self, boot0, devid, ssid):
        self.boot0 = boot0
        self.devid = devid
        self.ssid = ssid

    def _get_properties(self, devid):
        name = GPU_NAME_BY_DEVID_SSID.get(
            (devid, self.ssid), GPU_NAME_BY_DEVID.get(devid))
        props = GPU_PROPS_BY_DEVID.get((devid, self.ssid), [])
        return {
            "name": name,
            "flags": props,
        }

    def get_properties(self):
        return self._get_properties(self.devid)

    def get_properties_by_chip_and_ssid(self):
        """Get properties for this SSID from the reported device's chip range."""
        if (self.devid, self.ssid) in GPU_PROPS_BY_DEVID:
            return self.get_properties()

        for devid_low, devid_high, _arch, _chip in GPU_DEVID_CHIPS:
            if devid_low <= self.devid <= devid_high:
                for property_devid, property_ssid in GPU_PROPS_BY_DEVID:
                    if (devid_low <= property_devid <= devid_high and
                        property_ssid == self.ssid):
                        return self._get_properties(property_devid)
                break

        return self.get_properties()

    def get_metadata(self):
        """Return PCI-ID-derived identity without accessing the device."""
        arch, chip = self.get_chip_family(self.devid)
        properties = self.get_properties()
        name = properties["name"]
        if name is None and chip is not None:
            name = "Generic-" + chip.upper()
        return {"name": name, "arch": arch, "chip": chip}

    @staticmethod
    def get_nvswitch_metadata(devid):
        name, arch, chip = _NVSWITCH_METADATA.get(devid, (None, None, None))
        return {"name": name, "arch": arch, "chip": chip}

    @staticmethod
    def get_chip_family(devid):
        for devid_low, devid_high, arch, chip in GPU_DEVID_CHIPS:
            if devid >= devid_low and devid <= devid_high:
                return arch, chip
        return None, None
