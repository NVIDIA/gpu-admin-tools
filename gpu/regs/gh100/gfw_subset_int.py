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


# Register definitions
NV_R_URHLCNAU = RegisterMetadata(
    name='NV_R_URHLCNAU',
    address=0x1498
)

NV_R_URHLCNAU_STATUS = FieldMetadata(
    name='NV_R_URHLCNAU_STATUS',
    msb=7,
    lsb=0,
    register=NV_R_URHLCNAU
)

NV_R_URHLCNAU_STATUS_V_UUXHHBHV = ValueMetadata(
    name='NV_R_URHLCNAU_STATUS_V_UUXHHBHV',
    value=5,
    field=NV_R_URHLCNAU_STATUS
)
NV_R_URHLCNAU_STATUS_V_KISGFKTQ = ValueMetadata(
    name='NV_R_URHLCNAU_STATUS_V_KISGFKTQ',
    value=3,
    field=NV_R_URHLCNAU_STATUS
)
NV_R_URHLCNAU_STATUS_V_XRYTKQIA = ValueMetadata(
    name='NV_R_URHLCNAU_STATUS_V_XRYTKQIA',
    value=1,
    field=NV_R_URHLCNAU_STATUS
)
NV_R_URHLCNAU_STATUS_V_SWQGTAHD = ValueMetadata(
    name='NV_R_URHLCNAU_STATUS_V_SWQGTAHD',
    value=7,
    field=NV_R_URHLCNAU_STATUS
)
NV_R_URHLCNAU_STATUS_V_ELFPBSDW = ValueMetadata(
    name='NV_R_URHLCNAU_STATUS_V_ELFPBSDW',
    value=6,
    field=NV_R_URHLCNAU_STATUS
)
NV_R_URHLCNAU_STATUS_V_DEGLMMQM = ValueMetadata(
    name='NV_R_URHLCNAU_STATUS_V_DEGLMMQM',
    value=4,
    field=NV_R_URHLCNAU_STATUS
)
NV_R_URHLCNAU_STATUS_V_LLJYZPOG = ValueMetadata(
    name='NV_R_URHLCNAU_STATUS_V_LLJYZPOG',
    value=2,
    field=NV_R_URHLCNAU_STATUS
)
NV_R_URHLCNAU_STATUS_V_ZIXPEGCF = ValueMetadata(
    name='NV_R_URHLCNAU_STATUS_V_ZIXPEGCF',
    value=8,
    field=NV_R_URHLCNAU_STATUS
)
NV_R_URHLCNAU_STATUS_V_ZAVHFCMQ = ValueMetadata(
    name='NV_R_URHLCNAU_STATUS_V_ZAVHFCMQ',
    value=10,
    field=NV_R_URHLCNAU_STATUS
)
NV_R_URHLCNAU_STATUS_V_MHLMLFZA = ValueMetadata(
    name='NV_R_URHLCNAU_STATUS_V_MHLMLFZA',
    value=11,
    field=NV_R_URHLCNAU_STATUS
)
NV_R_URHLCNAU_STATUS_V_NLLQALRD = ValueMetadata(
    name='NV_R_URHLCNAU_STATUS_V_NLLQALRD',
    value=9,
    field=NV_R_URHLCNAU_STATUS
)
NV_R_URHLCNAU_STATUS_V_CLOEANJZ = ValueMetadata(
    name='NV_R_URHLCNAU_STATUS_V_CLOEANJZ',
    value=255,
    field=NV_R_URHLCNAU_STATUS
)
NV_R_URHLCNAU_STATUS_V_LVDKOIRS = ValueMetadata(
    name='NV_R_URHLCNAU_STATUS_V_LVDKOIRS',
    value=0,
    field=NV_R_URHLCNAU_STATUS
)

