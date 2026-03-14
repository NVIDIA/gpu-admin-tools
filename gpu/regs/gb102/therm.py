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

from gpu.regs.core import RegisterMetadata, FieldMetadata, ValueMetadata, ArrayMetadata, DeviceMetadata

# Registers identical to gh100
from gpu.regs.gh100.therm import (
    NV_THERM_I2CS_SCRATCH,
    NV_THERM_I2CS_SCRATCH_DATA,
    NV_THERM_I2CS_SCRATCH_DATA_INIT,
    NV_THERM_I2CS_SCRATCH_FSP_BOOT_COMPLETE,
    NV_THERM_I2CS_SCRATCH_FSP_BOOT_COMPLETE_STATUS,
    NV_THERM_I2CS_SCRATCH_FSP_BOOT_COMPLETE_STATUS_FAILED,
    NV_THERM_I2CS_SCRATCH_FSP_BOOT_COMPLETE_STATUS_SUCCESS,
)

