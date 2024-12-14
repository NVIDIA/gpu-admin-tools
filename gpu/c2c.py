#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from .unit import GpuUnit

from logging import info

class GpuC2C(GpuUnit):
    num_links = 10

    def __init__(self, gpu):
        super().__init__(gpu, "c2c")

        self.instances = self.device.device_info_instances[0x19]

    def firmware_status(self):
        status = self.read(self.device.vbios_scratch_register(38))
        if status == 0:
            return "not started"
        elif status == 0xff:
            return "up"
        else:
            return f"fail {status:#x}"

    def debug_print(self):
        info(f"{self.device} C2C firmware status {self.firmware_status()} num links {self.num_links} instances {self.instances}")

class GpuC2CBlackwell(GpuC2C):
    num_links = 14
