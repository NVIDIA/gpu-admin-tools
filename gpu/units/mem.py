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

from gpu.unit import GpuReplicatedRegisterModuleUnit, GpuTopReplicatedRegisterModuleUnit


class Fbpa(GpuReplicatedRegisterModuleUnit):
    name = "fbpa"

    def __init__(self, gpu):
        super().__init__(gpu, gpu.regs.fbpa_int)
        self.fbpa0_pri_base = gpu.regs.fbpa_int.NV_D_BXFCDUCX.start_address
        self.fbpa_pri_stride = gpu.regs.hwproject_int.NV_R_AHDWYAAJ.address

    def _stride_instances(self, fbpa_count):
        return [
            (f"fbpa{index}", self.fbpa0_pri_base + index * self.fbpa_pri_stride)
            for index in range(fbpa_count)
        ]

    def _instances(self):
        return self._stride_instances(self.device.top.num_fbpas)


class Ltc(GpuReplicatedRegisterModuleUnit):
    name = "ltc"

    def __init__(self, gpu):
        super().__init__(gpu, gpu.regs.ltc_int)
        self.ltc_pri_base = gpu.regs.hwproject_int.NV_R_VCPEDJKB.address
        self.ltc_pri_stride = gpu.regs.hwproject_int.NV_R_UAUSKSSZ.address
        self.lts_pri_stride = gpu.regs.hwproject_int.NV_R_HHBKRCSQ.address

    def _stride_instances(self, ltc_count, slices_per_ltc):
        instances = []
        for ltc_index in range(ltc_count):
            for slice_index in range(slices_per_ltc):
                instances.append(
                    (
                        f"ltc{ltc_index}.lts{slice_index}",
                        self.ltc_pri_base + ltc_index * self.ltc_pri_stride + slice_index * self.lts_pri_stride,
                    )
                )
        return instances

    def _instances(self):
        return self._stride_instances(self.device.top.num_ltcs, self.device.top.num_slices_per_ltc)


class Lrcc(GpuReplicatedRegisterModuleUnit):
    name = "lrcc"

    def __init__(self, gpu):
        super().__init__(gpu, gpu.regs.lrcc_int)
        self.lrcc0_pri_base = gpu.regs.lrcc_base_int.NV_R_CGZRVUFL.address
        self.lrcc_pri_stride = gpu.regs.hwproject_int.NV_R_LBEODTMC.address
        self.lrc_pri_stride = gpu.regs.hwproject_int.NV_R_GHRXMGNP.address

    def _stride_instances(self, lrcc_count, lrc_count):
        instances = []
        for lrcc_index in range(lrcc_count):
            for lrc_index in range(lrc_count):
                instances.append(
                    (
                        f"lrcc{lrcc_index}.lrc{lrc_index}",
                        self.lrcc0_pri_base + lrcc_index * self.lrcc_pri_stride + lrc_index * self.lrc_pri_stride,
                    )
                )
        return instances

    def _instances(self):
        return self._stride_instances(self.device.top.num_ltcs, self.device.top.num_slices_per_ltc)


class Sysltc(GpuTopReplicatedRegisterModuleUnit):
    name = "sysltc"

    def __init__(self, gpu):
        super().__init__(gpu, gpu.regs.sysltc_int, gpu.top.device_types.SYSLTC)
