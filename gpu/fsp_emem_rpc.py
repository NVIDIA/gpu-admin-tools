#
# SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from .error import GpuError, FspRpcError, GpuRpcTimeout

import time
if hasattr(time, "perf_counter"):
    perf_counter = time.perf_counter
else:
    perf_counter = time.time

class FspEmemRpc:
    def __init__(self, fsp_falcon, channel_num):
        self.falcon = fsp_falcon
        self.device = self.falcon.device
        self.channel_num = channel_num

        self.nvdm_emem_base = self.channel_num * 1024
        self.max_packet_size_bytes = 1024

        self.reset_rpc_state()

    def __str__(self):
        return f"{self.device} FSP-EMEM-RPC"

    def reset_rpc_state(self):
        if self.is_queue_empty() and self.is_msg_queue_empty():
            debug(f"{self} both queues empty; queue {self.read_queue_state()} msg queue {self.read_msg_queue_state()}")
            return

        debug(f"{self} one of the queues not empty, waiting for things to settle; queue {self.read_queue_state()} msg queue {self.read_msg_queue_state()}")
        self.poll_for_msg_queue(timeout_fatal=False)
        debug(f"{self} after wait; queue {self.read_queue_state()} msg queue {self.read_msg_queue_state()}")

        # Reset both queues
        self.write_queue_head_tail(self.nvdm_emem_base, self.nvdm_emem_base)
        self.device.write_verbose(self.falcon.msg_queue_tail_off(self.channel_num), self.nvdm_emem_base)
        self.device.write_verbose(self.falcon.msg_queue_head_off(self.channel_num), self.nvdm_emem_base)

    def read_queue_state(self):
        return (self.device.read(self.falcon.queue_head_off(self.channel_num)),
                self.device.read(self.falcon.queue_tail_off(self.channel_num)))

    def is_queue_empty(self):
        mhead, mtail = self.read_queue_state()
        return mhead == mtail

    def write_queue_head_tail(self, head, tail):
        self.device.write_verbose(self.falcon.queue_tail_off(self.channel_num), tail)
        self.device.write_verbose(self.falcon.queue_head_off(self.channel_num), head)

    def read_msg_queue_state(self):
        return (self.device.read(self.falcon.msg_queue_head_off(self.channel_num)),
                self.device.read(self.falcon.msg_queue_tail_off(self.channel_num)))

    def is_msg_queue_empty(self):
        mhead, mtail = self.read_msg_queue_state()
        return mhead == mtail

    def write_msg_queue_tail(self, tail):
        self.device.write_verbose(self.falcon.msg_queue_tail_off(self.channel_num), tail)

    def poll_for_msg_queue(self, timeout=5, sleep_interval=0.01, timeout_fatal=True):
        timestamp = perf_counter()
        while True:
            loop_stamp = perf_counter()
            mhead, mtail = self.read_msg_queue_state()
            if mhead != mtail:
                return
            if loop_stamp - timestamp > timeout:
                if timeout_fatal:
                    raise GpuRpcTimeout(f"Timed out polling for {self.falcon.name} message queue on channel {self.channel_num}. head {mhead} == tail {mtail}")
                else:
                    return
            if sleep_interval > 0.0:
                time.sleep(sleep_interval)

    def poll_for_queue_empty(self, timeout=1, sleep_interval=0.001):
        timestamp = perf_counter()
        while True:
            loop_stamp = perf_counter()
            if self.is_queue_empty():
                return

            # Check for an unexpected early response
            mhead, mtail = self.read_msg_queue_state()
            if mhead != mtail:
                data = self.receive_data()
                debug(f"{self} unexpected msg while waiting for queue to be empty {[hex(d) for d in data]}")

            if loop_stamp - timestamp > timeout:
                mhead, mtail = self.read_queue_state()
                raise GpuRpcTimeout(f"Timed out polling for {self.falcon.name} cmd queue to be empty on channel {self.channel_num}. head {mhead} != tail {mtail}")
            if sleep_interval > 0.0:
                time.sleep(sleep_interval)

    def send_data(self, data):
        self.poll_for_queue_empty()

        debug(f"{self} packet {[hex(d) for d in data[:20]]}...")
        self.falcon.write_emem(data, phys_base=self.nvdm_emem_base, port=self.channel_num)
        self.write_queue_head_tail(self.nvdm_emem_base, self.nvdm_emem_base + (len(data) - 1) * 4)

    def receive_data(self, timeout=5):
        rpc_time = perf_counter()
        self.poll_for_msg_queue(timeout=timeout)
        rpc_time = perf_counter() - rpc_time
        debug(f"{self} response took {rpc_time*1000:.1f} ms")

        mhead, mtail = self.read_msg_queue_state()
        debug(f"{self} msg queue after poll {mhead} {mtail}")
        msize = mtail - mhead + 4
        mdata = self.falcon.read_emem(self.nvdm_emem_base, msize, port=self.channel_num)
        debug(f"{self} response {[hex(d) for d in mdata]}")

        # Reset the tail before checking for errors
        self.write_msg_queue_tail(mhead)

        return mdata
