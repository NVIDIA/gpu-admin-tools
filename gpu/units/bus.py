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

from gpu.unit import GpuFixedBaseRegisterModuleUnit, GpuRegisterModuleUnit


class Xal(GpuRegisterModuleUnit):
    name = "xal"

    def __init__(self, gpu):
        super().__init__(gpu, gpu.regs.xal_int)


class Xpl(GpuFixedBaseRegisterModuleUnit):
    name = "xpl"

    def __init__(self, gpu):
        super().__init__(gpu, gpu.regs.xpl_int, gpu.regs.hwproject_int.NV_R_UARLJOVK.address)


class Xtl(GpuFixedBaseRegisterModuleUnit):
    name = "xtl"

    def __init__(self, gpu):
        super().__init__(gpu, gpu.regs.xtl_int, gpu.regs.hwproject_int.NV_R_PRPUINXR.address)


class Pxuc(GpuFixedBaseRegisterModuleUnit):
    name = "pxuc"

    def __init__(self, gpu):
        super().__init__(gpu, gpu.regs.pxuc_int, gpu.regs.hwproject_int.NV_R_PSBVMZJO.address)
