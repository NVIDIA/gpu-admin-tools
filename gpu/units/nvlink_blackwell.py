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

import time

from ..error import GpuError, GpuPollTimeout
from .nvlink import NvlinkBase, NvlinkFspInterface
from .nvlink_blackwell_link import BlackwellNvlinkLink


class NvlinkWaitTimeout(GpuPollTimeout):
    def __init__(self, nvlink, timeout, elapsed, state_timings, final_states):
        self.nvlink = nvlink
        self.timeout = timeout
        self.elapsed = elapsed
        self.state_timings = {
            link_index: dict(timings)
            for link_index, timings in state_timings.items()
        }
        self.final_states = dict(final_states)
        self.pending_links = [
            link_index
            for link_index, timings in self.state_timings.items()
            if "up" not in timings or self.final_states[link_index] != "up"
        ]
        final_state_text = ", ".join(
            f"link {link_index}={state}"
            for link_index, state in self.final_states.items()
        )
        super().__init__(
            f"{nvlink} timed out after {elapsed:.1f}s waiting for all NVLinks "
            f"to become fully up; pending links {self.pending_links}; "
            f"final states: {final_state_text}"
        )


class _NvlinkWait:
    def __init__(self, nvlink, timeout, started_at):
        self.nvlink = nvlink
        self.timeout = nvlink.wait_timeout if timeout is None else timeout
        if self.timeout < 0:
            raise ValueError("timeout must be non-negative")

        self.started_at = time.monotonic() if started_at is None else started_at
        self.link_indices = list(range(nvlink.num_nvlinks))
        if not self.link_indices:
            raise GpuError(f"{nvlink} has no NVLinks to wait for")

        self.state_timings = {
            link_index: {}
            for link_index in self.link_indices
        }
        self.elapsed = 0.0
        self.mse = nvlink.device.init_mse()

    def poll(self):
        states = list(self.mse.portlist_status())
        self.elapsed = time.monotonic() - self.started_at
        if len(states) != len(self.link_indices):
            raise GpuError(
                f"{self.nvlink} expected {len(self.link_indices)} NVLink states, "
                f"received {len(states)}: {states}"
            )
        last_states = dict(zip(self.link_indices, states))

        deadline_expired = self.elapsed > self.timeout
        for link_index, state in last_states.items():
            if deadline_expired:
                continue
            if state not in self.state_timings[link_index]:
                self.state_timings[link_index][state] = self.elapsed

        if not deadline_expired and all(state == "up" for state in states):
            return self.state_timings

        if self.elapsed >= self.timeout:
            raise NvlinkWaitTimeout(
                self.nvlink,
                self.timeout,
                self.elapsed,
                self.state_timings,
                last_states,
            )

        return None

    def next_poll_delay(self, poll_interval=None):
        if poll_interval is None:
            poll_interval = self.nvlink.wait_poll_interval
        if poll_interval < 0:
            raise ValueError("poll_interval must be non-negative")
        remaining = self.started_at + self.timeout - time.monotonic()
        return min(poll_interval, max(0, remaining))


class BlackwellNvlink(NvlinkBase, NvlinkFspInterface):
    does_flr_reenable_links = True
    wait_timeout = 20
    wait_poll_interval = 0.1

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

    def start_wait_for_nvlink(self, timeout=None, started_at=None):
        """Create a non-blocking NVLink waiter for coordinated polling."""
        return _NvlinkWait(self, timeout, started_at)

    def wait_for_nvlink(self, timeout=None, poll_interval=None, started_at=None):
        """Wait for every NVLink to become physically and fully up.

        The returned dictionary is keyed by link index. Each value maps every
        observed MSE state to the elapsed seconds at which it was first seen.
        A link must be ``up`` in the final poll even if it reached that state
        during an earlier poll.

        ``started_at`` may be a shared ``time.monotonic()`` value when several
        NVLink units need directly comparable timings and timeout deadlines.

        Raises NvlinkWaitTimeout if all links are not fully up by the timeout.
        The exception records partial ``state_timings``, ``elapsed``, and
        ``final_states`` metadata.
        """
        if poll_interval is None:
            poll_interval = self.wait_poll_interval
        if poll_interval < 0:
            raise ValueError("poll_interval must be non-negative")

        waiter = self.start_wait_for_nvlink(timeout=timeout, started_at=started_at)
        while True:
            state_timings = waiter.poll()
            if state_timings is not None:
                return state_timings

            delay = waiter.next_poll_delay(poll_interval)
            if delay > 0:
                time.sleep(delay)

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
