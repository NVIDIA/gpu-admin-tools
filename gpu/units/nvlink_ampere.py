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

from .nvlink import NvlinkBase

class AmpereNvlink(NvlinkBase):
    num_nvlinks = 12

    _links_per_group = 4
    _base_offset = 0xa00000
    _per_group_offset = 0x40000

    does_flr_reenable_links = False
    is_reset_needed_after_blocking = False

    def _nvlink_offset(self, link, reg=0):
        link_0_offset = 0x17000
        per_link_offset = 0x8000

        iob = link // self._links_per_group
        iolink = link % self._links_per_group
        link_offset = self._base_offset + self._per_group_offset * iob + link_0_offset + per_link_offset * iolink
        return link_offset + reg

    def nvlink_write_verbose(self, link, reg, data):
        reg_offset = self._nvlink_offset(link, reg)
        self.device.write_verbose(reg_offset, data)

    def block_nvlink(self, link, lock=True):
        assert link >= 0
        assert link < 12

        self.nvlink_write_verbose(link, 0x64c, 0x1)


        if lock:
            self.nvlink_write_verbose(link, 0x650, 0x1)


    def block_nvlinks(self, nvlinks):
        for nvlink in nvlinks:
            self.block_nvlink(nvlink)
        return True

    def get_blocked_nvlinks(self):
        blocked_links = []
        for link in range(self.num_nvlinks):
            if self.is_nvlink_blocked(link):
                blocked_links.append(link)
        return blocked_links

    def is_nvlink_blocked(self, link):
        return self.read(self._nvlink_offset(link, 0x64c)) == 0x1 and self.read(self._nvlink_offset(link, 0x650)) == 0x1

    def get_enabled_nvlinks(self):
        return list(range(self.num_nvlinks))
