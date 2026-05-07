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

from .nvlink import NvlinkBase, NvlinkFspInterface
from .nvlink_blackwell_link import BlackwellNvlinkLink

class BlackwellNvlink(NvlinkBase, NvlinkFspInterface):
    does_flr_reenable_links = True

    def __init__(self, device):
        super().__init__(device)

        self.nvlpw_devices = self.device.top.device_info_instances[self.device.top.device_types.NVLPW]
        self.present_links = [d.instance for d in self.nvlpw_devices]
        self.num_nvlinks = len(self.present_links)

        self.links = {}
        for dev_info in self.nvlpw_devices:
            i = dev_info.instance
            link = BlackwellNvlinkLink(self.device, i, dev_info.pri_base)
            self.links[i] = link

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

    def _capture_mse_registers(self, capture):
        base = self.device.regs.nvlc_discovery_int.NV_R_SBMVNEVV.address
        capture.subunit("mse").module(self.device.regs.mse_int, base=base)

    def _capture_ir_registers(self, capture):
        base = self.device.regs.nbu_net_discovery_int.NV_R_HHFZGPBS.address
        capture.subunit("netir").module(self.device.regs.ir_int, base=base)

    def _capture_microcontroller_registers(self, capture):
        self._capture_mse_registers(capture)
        self._capture_ir_registers(capture)

    def debug_dump_capture(self, capture, _options):
        mse_capture = capture.subunit("mse")

        # MSE link states
        try:
            self.device.init_mse()
            states = list(self.device.mse.portlist_status())
        except Exception as err:  # pylint: disable=broad-except
            mse_capture.data(
                "link_states",
                {"status": "error", "error": str(err)},
                interesting=str(err),
            )
            return
        interesting = None if all(state == "up" for state in states) else "NVLinks down"
        mse_capture.data("link_states", states, interesting=interesting)

        self._capture_microcontroller_registers(capture)

        # Per-link register captures
        for link in sorted(self.links.values(), key=lambda lnk: lnk.link_index):
            link.debug_dump_registers(capture)
