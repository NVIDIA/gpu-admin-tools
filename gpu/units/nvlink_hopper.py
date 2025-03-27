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

from logging import debug, info

from .nvlink import NvlinkBase, NvlinkFspInterface

class LagunaNvlink(NvlinkBase, NvlinkFspInterface):
    does_flr_reenable_links = True

    num_nvlinks = 64

    _links_per_group = 4
    _base_offset = 0x1000000
    _per_group_offset = 0x100000

    def get_enabled_nvlinks(self):
        enabled_nvlinks = []
        for link in range(self.num_nvlinks):
            link_state = self.device.nvlink_get_link_state(link)
            if self.device.nvlink_get_link_state(link) not in ["badf", "disable"]:
                enabled_nvlinks.append(link)
        return enabled_nvlinks

    def get_blocked_nvlinks(self):
        blocked_links = []
        for link in range(self.num_nvlinks):
            if self.is_nvlink_blocked(link):
                blocked_links.append(link)
        return blocked_links

    def is_nvlink_blocked(self, link):
        link_state = self.device.nvlink_get_link_state(link)
        return link_state == "disable"
class HopperNvlink(LagunaNvlink):
    num_nvlinks = 18

    _links_per_group = 6
    _base_offset = 0xa00000
    _per_group_offset = 0x40000

    def __init__(self, device):
        super().__init__(device)
        ioctrl_instances = device.device_info_instances[18]
        self.num_nvlinks = len(ioctrl_instances) * self._links_per_group
