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

import atexit
from logging import debug

from utils import NiceStruct

from .mnoc import GpuMnoc


class MseHeader(NiceStruct):
    _fields_ = [
            ("version", "I", 8),
            ("status", "I", 6),
            ("rsvd0", "I", 2),
            ("ssid", "I", 8),
            ("dsid", "I", 8),

            ("ctxid", "I", 16),
            ("cmd_opcode", "I", 12),
            ("cmd_class", "I", 4),

            ("credit", "I", 11),
            ("credit_priority", "I", 2),
            ("credit_reset", "I", 1),
            ("rsvd1", "I", 2),

            ("flags_type", "I", 1),

            ("flags_final", "I", 1),
            ("flags_reset", "I", 1),
            ("flags_rsvd", "I", 3),
            ("priority", "I", 2),
            ("endpoint_specific", "I", 8),

            ("rsvd3", "I", 32),
        ]

    def __init__(self):
        super().__init__()

    @property
    def is_response(self):
        return self.flags_type == 1

class GetPlatformInfoRsp(NiceStruct):
    _fields_ = [
        ("ibGuid", "16s"),
        ("rackGuid", "16s"),
        ("chassisPhysicalSlotNumber", "B"),
        ("computeSlotIndex", "B"),
        ("nodeIndex", "B"),
        ("peerType", "B"),
        ("moduleId", "B"),
        ("rsvd", "35s"),
    ]

class MseRpc:
    def __init__(self, device):
        self.device = device

        self.context_id = 0x1

        self.mnoc = GpuMnoc(device, "MSE MNOC", 0x2c00000 + 0x1e00, 1)

        self._negotiate()

        atexit.register(self.exit)

    def __str__(self):
        return f"{self.device} MSE"

    def remove_atexit_cleanup(self):
        atexit.unregister(self.exit)

    def exit(self):
        self.goodbye()

    def send_cmd(self, cmd_class, cmd_opcode, cmd_data, reset=False):
        mse_header = MseHeader()
        mse_header.ctxid = self.context_id
        self.context_id += 1
        mse_header.cmd_class = cmd_class
        mse_header.cmd_opcode = cmd_opcode
        mse_header.credit = 8
        mse_header.flags_reset = 1 if reset else 0

        mse_header.ssid = 9
        mse_header.dsid = 4

        data_to_send = mse_header.to_int_list() + cmd_data

        self.mnoc.send_data(data_to_send)

        return self.process_incoming()

    def process_incoming(self):
        mse_header = MseHeader()

        while True:
            resp_data = self.mnoc.receive_data()
            mse_header.from_ints(resp_data[:4])
            if mse_header.is_response:
                return resp_data[4:]
            else:
                debug(f"Got a request, dropping for now {[hex(d) for d in resp_data]}")

    def _negotiate(self):
        self.send_cmd(1, 0, [0, 0], reset=True)

    def portlist_status(self):
        resp = self.send_cmd(2, 2, [(0x1<<18)-1])


        links = resp[0]
        link_states = [link_state for dword in resp[1:] for link_state in (dword & 0xffff, (dword >> 16) & 0xffff)]

        link_state_map = {
            1: "down",
            2: "up",
            4: "sleep",
            5: "down_lock",
            6: "polling",
            7: "training",
            8: "training_failure",
            9: "training_failure_locked",
           10: "physical_up",
        }

        nice_link_states = []
        for s in link_states:
            status = s >> 8
            if status == 1:
                nice_link_states.append("disabled")
                continue
            if status != 2:
                nice_link_states.append(f"unknown {status:#x}")
                continue
            state = s & 0xff
            nice_link_states.append(link_state_map.get(state, f"unknown {state:#x}"))

        return nice_link_states

    def goodbye(self):
        self.send_cmd(2, 11, [])

    def get_platform_info(self):
        platform_info_raw = self.send_cmd(2, 0x30, [])
        platform_info = GetPlatformInfoRsp()
        platform_info.from_ints(platform_info_raw)
        return platform_info
