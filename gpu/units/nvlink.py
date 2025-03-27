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

from ..unit import GpuUnit

from logging import info, error

class NvlinkBase(GpuUnit):
    name = "nvlink"
    is_reset_needed_after_blocking = True

    def __init__(self, device):
        super().__init__(device)
        device.nvlink_unit = self
        device.is_nvlink_supported = True

    def test_nvlink_blocking(self):
        if self.num_nvlinks == 0:
            info(f"{self} has no NVLinks to block, exiting early")
            return True

        links_to_block = self.get_enabled_nvlinks()
        info(f"{self} blocking {len(links_to_block)} links {links_to_block}")
        self.block_nvlinks(links_to_block)
        if self.is_reset_needed_after_blocking or not self.does_flr_reenable_links:
            self.device.reset_with_os()
            self.device.wait_for_boot()

        blocked_links = set(self.get_blocked_nvlinks())

        if not blocked_links.issuperset(links_to_block):
            error(f"{self} Not all links blocked: {blocked_links} != {links_to_block}")
            return False

        info(f"{self} unblocking {len(links_to_block)} links")
        if self.does_flr_reenable_links:
            self.device.reset_with_os()
        else:
            self.device.reset_with_sbr()

        self.device.wait_for_boot()

        blocked_links = set(self.get_blocked_nvlinks())
        if blocked_links & set(links_to_block):
            error(f"{self} not all links unblocked: {blocked_links}  {links_to_block}")
            return False
        info(f"{self} links unblocked as expected")

        return True

class NvlinkFspInterface:
    def block_nvlinks(self, nvlinks):
        self.device._init_fsp_rpc()
        self.device.fsp_rpc.prc_block_nvlinks(nvlinks, persistent=False)
