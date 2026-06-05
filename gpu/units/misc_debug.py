#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from ..error import FspRpcError
from ..prc import PrcKnob
from ..unit import GpuUnit


class MiscDebug(GpuUnit):
    name = "misc"

    def __init__(self, gpu):
        super().__init__(gpu)
        gpu.misc_debug = self

    def debug_dump_capture(self, capture, _options):
        if not self.device.is_hopper_plus:
            return

        self._capture_boot_status(capture)
        self._capture_fsp_status(capture)
        self._capture_prc_knobs(capture)
        self._capture_register_modules(capture)

    def _capture_boot_status(self, capture):
        record = capture.register(self.device.regs.therm.NV_THERM_I2CS_SCRATCH_FSP_BOOT_COMPLETE)
        if record.get("status") == "unreadable":
            capture.set_interesting(record, "GPU boot status unreadable")
        elif record.reg is None or record.reg.STATUS != self.device.regs.therm.NV_THERM_I2CS_SCRATCH_FSP_BOOT_COMPLETE_STATUS_SUCCESS:
            capture.set_interesting(record, "GPU boot not finished")

    def _capture_fsp_status(self, capture):
        for index in range(4):
            capture.offset(f"fsp_status_{index}", 0x8F0320 + index * 4)

    def _capture_prc_knobs(self, capture):
        if not self.device.has_fsp:
            return

        record = capture.data_pairs("prc_knobs", self._query_prc_knob_entries)
        if record.get("status") == "error":
            capture.set_interesting(record, record.get("error", "PRC knobs unavailable"))

    def _query_prc_knob_entries(self):
        self.device._init_fsp_rpc()

        entries = []
        for knob in PrcKnob:
            knob_id = knob.value
            try:
                knob_value = self.device.fsp_rpc.prc_knob_read(knob_id)
            except FspRpcError as err:
                if err.is_invalid_knob_error:
                    knob_value = "invalid"
                else:
                    raise
            entries.append({
                "key": str(knob_id),
                "name": PrcKnob.str_from_knob_id(knob_id),
                "value": knob_value,
            })
        return entries

    def _capture_register_modules(self, capture):
        capture.module(self.device.regs.therm_int)
        capture.module(self.device.regs.pbus)
        for module_name in (
            "gsp_int",
            "fsp_int",
            "pwr_int",
            "sec_int",
            "ce_int",
            "pfb_mmu_int",
        ):
            register_module = getattr(self.device.regs, module_name, None)
            if register_module is None:
                continue
            capture.module(register_module)
