#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from utils.auto_extend import AutoExtend


class GpuPriUnit:
    """Base class for GPU PRI (Private Register Interface) addressable units."""

    def __init__(self, gpu, pri_base):
        self.gpu = gpu
        self.pri_base = pri_base

    def read_reg(self, register):
        """Read a register at this unit's pri_base offset."""
        return self.gpu.regs.read(register, base=self.pri_base)


class NvlpwTlw(GpuPriUnit, AutoExtend):
    """NVLPW TLW (Transaction Layer Wrapper) for one NVLink."""

    @staticmethod
    def _compute_pri_base(gpu, nvlpw_pri_base):
        discovery = gpu.regs.nvlpw_discovery_int
        nvlpw_base_0 = discovery.NV_R_AIWXTDSO.address
        tlw_base_0 = discovery.NV_R_ZZEGTPWK.address
        return nvlpw_pri_base + (tlw_base_0 - nvlpw_base_0)

    def __init__(self, gpu, link_index, nvlpw_pri_base):
        super().__init__(gpu, self._compute_pri_base(gpu, nvlpw_pri_base))
        self.link_index = link_index

    def debug_dump_registers(self, capture):
        capture.module(
            self.gpu.regs.nvlpw_tlw_int,
            base=self.pri_base,
            instance=f"link{self.link_index}",
        )

    def __str__(self):
        return f"tlw link{self.link_index} @ {self.pri_base:#x}"

    def __repr__(self):
        return str(self)


class NvlpwRlw(GpuPriUnit, AutoExtend):
    """NVLPW RLW (Response Layer Wrapper) for one NVLink."""

    @staticmethod
    def _compute_pri_base(gpu, nvlpw_pri_base):
        discovery = gpu.regs.nvlpw_discovery_int
        nvlpw_base_0 = discovery.NV_R_AIWXTDSO.address
        rlw_base_0 = discovery.NV_R_OICXKDYJ.address
        return nvlpw_pri_base + (rlw_base_0 - nvlpw_base_0)

    def __init__(self, gpu, link_index, nvlpw_pri_base):
        super().__init__(gpu, self._compute_pri_base(gpu, nvlpw_pri_base))
        self.link_index = link_index

    def debug_dump_registers(self, capture):
        capture.module(
            self.gpu.regs.nvlpw_rlw_int,
            base=self.pri_base,
            instance=f"link{self.link_index}",
        )

    def __str__(self):
        return f"rlw link{self.link_index} @ {self.pri_base:#x}"

    def __repr__(self):
        return str(self)


class NvlinkLlu(GpuPriUnit):
    """NVLink LLU (Link Layer Unit) port state for a single link."""

    PORT_STATE_UP = 0x4

    @staticmethod
    def _compute_pri_base(gpu, link_index):
        discovery = gpu.regs.nbu_net_discovery_int
        base_0 = discovery.NV_R_JNQPXUWH.address
        base_1 = discovery.NV_R_DCYOUALB.address
        return base_0 + link_index * (base_1 - base_0)

    def __init__(self, gpu, link_index):
        super().__init__(gpu, self._compute_pri_base(gpu, link_index))
        self.link_index = link_index

    def read_port_state(self):
        reg = self.gpu.regs.llu_int.NV_R_NPQLXAHE
        val = self.read_reg(reg)
        raw = int(val)
        state = int(val.F_CCAWKYFP)
        return {
            "raw": raw,
            "state": state,
            "up": (raw >> 16) != 0xBADF and state == self.PORT_STATE_UP,
        }

    def debug_dump_registers(self, capture):
        capture.module(
            self.gpu.regs.llu_int,
            base=self.pri_base,
            instance=f"link{self.link_index}",
        )

    def __str__(self):
        return f"llu link{self.link_index} @ {self.pri_base:#x}"

    def __repr__(self):
        return str(self)


class NvlinkPlu(GpuPriUnit):
    """NVLink PLU (Physical Layer Unit) port state for a single link."""

    PORT_STATE_UP = 0x5

    @staticmethod
    def _compute_pri_base(gpu, link_index):
        discovery = gpu.regs.nbu_net_discovery_int
        base_0 = discovery.NV_R_TIYQJVBI.address
        base_1 = discovery.NV_R_QREIAGKN.address
        return base_0 + link_index * (base_1 - base_0)

    def __init__(self, gpu, link_index):
        super().__init__(gpu, self._compute_pri_base(gpu, link_index))
        self.link_index = link_index

    def _port_state_register(self):
        regs = self.gpu.regs.plu_int
        return regs.NV_R_ZXCXJFYF

    def read_port_state(self):
        reg = self._port_state_register()
        val = self.read_reg(reg)
        raw = int(val)
        state = int(val.F_JKKXLPAF)
        return {
            "raw": raw,
            "state": state,
            "up": (raw >> 16) != 0xBADF and state == self.PORT_STATE_UP,
        }

    def debug_dump_registers(self, capture):
        capture.module(
            self.gpu.regs.plu_int,
            base=self.pri_base,
            instance=f"link{self.link_index}",
        )

    def __str__(self):
        return f"plu link{self.link_index} @ {self.pri_base:#x}"

    def __repr__(self):
        return str(self)


class NvlinkUglNvl(GpuPriUnit, AutoExtend):
    """NVLink UGL NVL per-lane FSM state for a single link."""

    @staticmethod
    def _compute_pri_base(gpu, link_index):
        discovery = gpu.regs.nbu_net_discovery_int
        base_0 = discovery.NV_R_YMHQORTY.address
        base_1 = discovery.NV_R_XLMTINBC.address
        return base_0 + link_index * (base_1 - base_0)

    def __init__(self, gpu, link_index):
        super().__init__(gpu, self._compute_pri_base(gpu, link_index))
        self.link_index = link_index

    def debug_dump_registers(self, capture):
        capture.module(
            self.gpu.regs.ugl_nvl_int,
            base=self.pri_base,
            instance=f"link{self.link_index}",
        )

    def __str__(self):
        return f"ugl_nvl link{self.link_index} @ {self.pri_base:#x}"

    def __repr__(self):
        return str(self)


class BlackwellNvlinkLink:
    """Represents a single NVLink with all its sub-units."""

    def __init__(self, gpu, link_index, nvlpw_pri_base):
        self.gpu = gpu
        self.link_index = link_index
        self.tlw = NvlpwTlw(gpu, link_index, nvlpw_pri_base)
        self.rlw = NvlpwRlw(gpu, link_index, nvlpw_pri_base)
        self.llu = NvlinkLlu(gpu, link_index)
        self.plu = NvlinkPlu(gpu, link_index)
        self.ugl = NvlinkUglNvl(gpu, link_index)

    def read_port_state(self):
        """Read combined LLU + PLU port state."""
        llu = self.llu.read_port_state()
        plu = self.plu.read_port_state()
        return {
            "llu": llu,
            "plu": plu,
            "healthy": llu["up"] and plu["up"],
        }

    def debug_dump_registers(self, capture):
        self.tlw.debug_dump_registers(capture)
        self.rlw.debug_dump_registers(capture)
        self.llu.debug_dump_registers(capture)
        self.plu.debug_dump_registers(capture)
        self.ugl.debug_dump_registers(capture)

    def __str__(self):
        return f"nvlink link{self.link_index}"

    def __repr__(self):
        return str(self)
