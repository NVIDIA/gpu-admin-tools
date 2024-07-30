#
# SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

class GpuMnoc:
    def __init__(self, device, name, base, port):
        self.device = device
        self.name = name
        self.base = base
        self.port = port

    def __str__(self):
        return f"{self.device} {self.name} port {self.port}"

    @property
    def offset_rx_fifo_data(self):
        return self.base + self.port * 8

    @property
    def offset_info_send_mbox(self):
        return self.base + 0x104 + self.port * 12

    @property
    def offset_rdata_send_mbox(self):
        return self.offset_info_send_mbox + 4

    @property
    def offset_info_receive_mbox(self):
        return self.base + 0x184 + self.port * 12

    @property
    def offset_wdata_receive_mbox(self):
        return self.offset_info_receive_mbox + 4

    def poll_for_message_ready(self):
        self.device.poll_register(f"{self} message ready", self.offset_info_send_mbox, value=0x1<<24, timeout=5, mask=0x1<<24)

    def poll_for_receive_ready(self):
        self.device.poll_register(f"{self} receive ready", self.offset_info_receive_mbox, value=0x1<<24, timeout=5, mask=0x1<<24)

    def poll_for_receive_credits(self):
        self.device.poll_register(f"{self} credits", self.offset_info_receive_mbox, value=0x1<<26, timeout=1, mask=0x1<<26)

    def send_data(self, data):
        size = len(data) * 4

        self.poll_for_receive_ready()
        sent_bytes = 0

        msg_metadata = size
        msg_metadata |= 0x1 << 20
        self.device.write(self.offset_info_receive_mbox, msg_metadata)

        for d in data:
            if sent_bytes % 64 == 0:
                self.poll_for_receive_credits()
            self.device.write(self.offset_wdata_receive_mbox, d)

    def receive_data(self):
        self.poll_for_message_ready()

        msg_size = self.device.read(self.offset_info_send_mbox) & 0xfffff
        debug(f"{self} receiving message size {msg_size}")

        data = []
        received = 0
        while received < msg_size:
            data.append(self.device.read(self.offset_rdata_send_mbox))
            received += 4

        return data
