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

from .devid_names import GPU_NAME_BY_DEVID
from .devid_properties import GPU_PROPS_BY_DEVID
from .devid_chips import GPU_DEVID_CHIPS

class GpuProperties:
    def __init__(self, boot0, devid, ssid):
        self.boot0 = boot0
        self.devid = devid
        self.ssid = ssid

    def get_properties(self):
        name = GPU_NAME_BY_DEVID.get(self.devid, None)
        props = GPU_PROPS_BY_DEVID.get((self.devid, self.ssid), [])
        return {
            "name": name,
            "flags": props,
        }

    @staticmethod
    def get_chip_family(devid):
        for devid_low, devid_high, arch, chip in GPU_DEVID_CHIPS:
            if devid >= devid_low and devid <= devid_high:
                return arch, chip
        return None, None
