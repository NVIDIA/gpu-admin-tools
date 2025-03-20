#
# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Register definitions
NV_PMC_SCRATCH_RESET_2_CC = RegisterMetadata(
    name='NV_PMC_SCRATCH_RESET_2_CC',
    address=0x590
)

NV_PMC_SCRATCH_RESET_2_CC_DEV_ENABLED = FieldMetadata(
    name='NV_PMC_SCRATCH_RESET_2_CC_DEV_ENABLED',
    msb=1,
    lsb=1,
    register=NV_PMC_SCRATCH_RESET_2_CC
)

NV_PMC_SCRATCH_RESET_2_CC_DEV_ENABLED_FALSE = ValueMetadata(
    name='NV_PMC_SCRATCH_RESET_2_CC_DEV_ENABLED_FALSE',
    value=0,
    field=NV_PMC_SCRATCH_RESET_2_CC_DEV_ENABLED
)
NV_PMC_SCRATCH_RESET_2_CC_DEV_ENABLED_TRUE = ValueMetadata(
    name='NV_PMC_SCRATCH_RESET_2_CC_DEV_ENABLED_TRUE',
    value=1,
    field=NV_PMC_SCRATCH_RESET_2_CC_DEV_ENABLED
)

NV_PMC_SCRATCH_RESET_2_CC_MODE_ENABLED = FieldMetadata(
    name='NV_PMC_SCRATCH_RESET_2_CC_MODE_ENABLED',
    msb=0,
    lsb=0,
    register=NV_PMC_SCRATCH_RESET_2_CC
)

NV_PMC_SCRATCH_RESET_2_CC_MODE_ENABLED_FALSE = ValueMetadata(
    name='NV_PMC_SCRATCH_RESET_2_CC_MODE_ENABLED_FALSE',
    value=0,
    field=NV_PMC_SCRATCH_RESET_2_CC_MODE_ENABLED
)
NV_PMC_SCRATCH_RESET_2_CC_MODE_ENABLED_TRUE = ValueMetadata(
    name='NV_PMC_SCRATCH_RESET_2_CC_MODE_ENABLED_TRUE',
    value=1,
    field=NV_PMC_SCRATCH_RESET_2_CC_MODE_ENABLED
)

NV_PMC_SCRATCH_RESET_2_CC_NVLE_MODE_ENABLED = FieldMetadata(
    name='NV_PMC_SCRATCH_RESET_2_CC_NVLE_MODE_ENABLED',
    msb=6,
    lsb=6,
    register=NV_PMC_SCRATCH_RESET_2_CC
)

NV_PMC_SCRATCH_RESET_2_CC_NVLE_MODE_ENABLED_FALSE = ValueMetadata(
    name='NV_PMC_SCRATCH_RESET_2_CC_NVLE_MODE_ENABLED_FALSE',
    value=0,
    field=NV_PMC_SCRATCH_RESET_2_CC_NVLE_MODE_ENABLED
)
NV_PMC_SCRATCH_RESET_2_CC_NVLE_MODE_ENABLED_TRUE = ValueMetadata(
    name='NV_PMC_SCRATCH_RESET_2_CC_NVLE_MODE_ENABLED_TRUE',
    value=1,
    field=NV_PMC_SCRATCH_RESET_2_CC_NVLE_MODE_ENABLED
)

# Array definitions
NV_PMC_SCRATCH_RESET_2 = ArrayMetadata(
    name='NV_PMC_SCRATCH_RESET_2',
    base_address=0x580,
    stride=4,
    size=16
)

NV_PMC_SCRATCH_RESET_2_VALUE = FieldMetadata(
    name='NV_PMC_SCRATCH_RESET_2_VALUE',
    msb=31,
    lsb=0,
    register=NV_PMC_SCRATCH_RESET_2
)

NV_PMC_SCRATCH_RESET_2_VALUE_INIT = ValueMetadata(
    name='NV_PMC_SCRATCH_RESET_2_VALUE_INIT',
    value=0,
    field=NV_PMC_SCRATCH_RESET_2_VALUE
)

