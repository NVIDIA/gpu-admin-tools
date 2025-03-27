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

from logging import debug

from .nvlink import NvlinkBase, NvlinkFspInterface

class BlackwellNvlink(NvlinkBase, NvlinkFspInterface):
    does_flr_reenable_links = True

    def __init__(self, device):
        super().__init__(device)

        self.present_links = self.device.device_info_instances[self.device.regs.top_int.NV_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_V_EFTKKQKC.value]
        self.num_nvlinks = len(self.present_links)

    def get_enabled_nvlinks(self):
        return self.present_links

    def get_blocked_nvlinks(self):
        self.device.init_mse()
        link_states = self.device.mse.portlist_status()
        blocked_links = []
        for link in range(self.num_nvlinks):
            if link_states[link] == "disabled":
                blocked_links.append(link)
        return blocked_links
