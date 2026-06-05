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

from gpu.unit import GpuFixedBaseRegisterModuleUnit, GpuTopReplicatedRegisterModuleUnit


class Hubmmu(GpuTopReplicatedRegisterModuleUnit):
    name = "hubmmu"

    def __init__(self, gpu):
        super().__init__(gpu, gpu.regs.hubmmu_int, gpu.top.device_types.HUBMMU)


class Hshubmmu(GpuFixedBaseRegisterModuleUnit):
    name = "hshubmmu"

    def __init__(self, gpu):
        # GH100 exposes HSHUBMMU registers but does not enumerate HSHUBMMU in TOP.
        super().__init__(gpu, gpu.regs.hshubmmu_int, gpu.regs.hshubmmu_base_int.NV_D_HTIARHWW.start_address)


class HshubmmuBlackwell(GpuTopReplicatedRegisterModuleUnit):
    name = "hshubmmu"

    def __init__(self, gpu):
        super().__init__(gpu, gpu.regs.hshubmmu_int, gpu.top.device_types.HSHUBMMU)


class Fbhub(GpuTopReplicatedRegisterModuleUnit):
    name = "fbhub"

    def __init__(self, gpu):
        super().__init__(gpu, gpu.regs.fbhub_int, gpu.top.device_types.FBHUB)


class Hshub(GpuTopReplicatedRegisterModuleUnit):
    name = "hshub"

    def __init__(self, gpu):
        super().__init__(gpu, gpu.regs.hshub_int, gpu.top.device_types.HSHUB)
