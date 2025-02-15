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

import time
from time import perf_counter
from logging import debug, info, error

from .device import PciDevice
from .defines import *
from utils import DeviceField

class PciBridge(PciDevice):
    def __init__(self, dev_path):
        super(PciBridge, self).__init__(dev_path)
        self.bridge_ctl = DeviceField(PciBridgeControl, self.config, PCI_BRIDGE_CONTROL)
        if self.parent:
            self.parent.children.append(self)

    def __str__(self):
        return f"PciBridge {self.bdf} {self.device:x}:{self.vendor:x}"

    def is_bridge(self):
        return True

    def _set_link_disable(self, disable):
        self.link_ctl["LD"] = 1 if disable else 0
        debug("%s %s link disable, %s", self, "setting" if disable else "unsetting", self.link_ctl)

    def _set_sbr(self, reset):
        self.bridge_ctl["BUS_RESET"] = 1 if reset else 0
        debug("%s %s bus reset, %s",
              self, "setting" if reset else "unsetting", self.bridge_ctl)

    def toggle_link(self):
        self._set_link_disable(True)
        time.sleep(0.1)
        self._set_link_disable(False)
        time.sleep(0.1)

    def wait_for_link(self, timeout=5):
        link_time = perf_counter()
        prev_link_status = None
        while True:
            if perf_counter() - link_time > timeout:
                error(f"{self} Link failed to train")
                return False, timeout
            # Read once
            link_status = self.link_status._read()
            if link_status != prev_link_status:
                debug(f"{self} new link status after {(perf_counter() - link_time) * 1000:.1f}ms {link_status}")
                prev_link_status = link_status

            if link_status['LT'] == 1:
                continue
            if link_status['DLLLA'] == 0:
                continue
            break

        link_time = perf_counter() - link_time
        speed = link_status["CLS"]
        debug(f"{self} link training to gen{speed} done after {link_time*1000:.1f} ms")
        return True, link_time

    def toggle_sbr(self, retry_count=1, fail_callback=None):
        success = False

        modified_slot_ctl = False
        if self.has_exp() and self.pciflags["SLOT"] == 1:
            saved_dll = self.slot_ctl["DLLSCE"]
            saved_hpie = self.slot_ctl["HPIE"]
            # Disable link state change notification
            self.slot_ctl["DLLSCE"] = 0
            self.slot_ctl["HPIE"] = 0
            modified_slot_ctl = True

        saved_acs_sv = None
        if self.is_cx7 and self.has_acs:
            # DRS during reset can cause ACS source validation violations
            saved_acs_sv = self.acs_ctl["SOURCE_VALIDATION"]
            self.acs_ctl["SOURCE_VALIDATION"] = 0
            if saved_acs_sv:
                debug(f"{self} disabling ACS source validation across SBR, saved value {saved_acs_sv}")

        for i in range(retry_count):
            self._set_sbr(True)
            time.sleep(0.1)
            self._set_sbr(False)

            success, time_to_train = self.wait_for_link()
            if success:
                break

            error(f"{self} link training failed after SBR try {i}")
            if fail_callback:
                fail_callback()

        if modified_slot_ctl:
            if time_to_train < 0.3:
                # We used to have a 300ms sleep after SBR, now we poll link
                # status and link training can be done quite quickly in some
                # cases. In practical testing, it looks like some of the slot
                # status detection is delayed. For now just add an extra sleep
                # if link training was faster than the 300ms we used to sleep
                # for.
                debug(f"{self} extra sleep to handle slot status racing")
                time.sleep(0.3 - time_to_train)

            # Clear any pending interrupts for presence and link state
            self.slot_status["PDC"] = 1
            self.slot_status["DLLSC"] = 1

            self.slot_ctl["DLLSCE"] = saved_dll
            self.slot_ctl["HPIE"] = saved_hpie

        if saved_acs_sv is not None:
            self.acs_ctl["SOURCE_VALIDATION"] = saved_acs_sv

        return success


class IntelRootPort(PciBridge):
    def __init__(self, dev_path):
        super(IntelRootPort, self).__init__(dev_path)

    def is_intel(self):
        return True

    def __str__(self):
        return "Intel root port %s" % self.bdf


class PlxBridge(PciBridge):
    def __init__(self, dev_path):
        super(PlxBridge, self).__init__(dev_path)


    def __str__(self):
        return "PLX %s" % self.bdf

    def is_plx(self):
        return True

