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
NV_R_NSSUPEIW_PRIV_LEVEL_MASK = RegisterMetadata(
    name='NV_R_NSSUPEIW_PRIV_LEVEL_MASK',
    address=0x708
)

NV_R_NSSUPEIW_PRIV_LEVEL_MASK_READ_PROTECTION = FieldMetadata(
    name='NV_R_NSSUPEIW_PRIV_LEVEL_MASK_READ_PROTECTION',
    msb=3,
    lsb=0,
    register=NV_R_NSSUPEIW_PRIV_LEVEL_MASK
)

NV_R_NSSUPEIW_PRIV_LEVEL_MASK_READ_PROTECTION_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_NSSUPEIW_PRIV_LEVEL_MASK_READ_PROTECTION_V_ZRRJKDVX',
    value=15,
    field=NV_R_NSSUPEIW_PRIV_LEVEL_MASK_READ_PROTECTION
)

NV_R_NSSUPEIW_PRIV_LEVEL_MASK_READ_VIOLATION = FieldMetadata(
    name='NV_R_NSSUPEIW_PRIV_LEVEL_MASK_READ_VIOLATION',
    msb=8,
    lsb=8,
    register=NV_R_NSSUPEIW_PRIV_LEVEL_MASK
)

NV_R_NSSUPEIW_PRIV_LEVEL_MASK_READ_VIOLATION_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_NSSUPEIW_PRIV_LEVEL_MASK_READ_VIOLATION_V_ZRRJKDVX',
    value=1,
    field=NV_R_NSSUPEIW_PRIV_LEVEL_MASK_READ_VIOLATION
)

NV_R_NSSUPEIW_PRIV_LEVEL_MASK_SOURCE_ENABLE = FieldMetadata(
    name='NV_R_NSSUPEIW_PRIV_LEVEL_MASK_SOURCE_ENABLE',
    msb=31,
    lsb=12,
    register=NV_R_NSSUPEIW_PRIV_LEVEL_MASK
)

NV_R_NSSUPEIW_PRIV_LEVEL_MASK_SOURCE_ENABLE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_NSSUPEIW_PRIV_LEVEL_MASK_SOURCE_ENABLE_V_ZRRJKDVX',
    value=1048575,
    field=NV_R_NSSUPEIW_PRIV_LEVEL_MASK_SOURCE_ENABLE
)

NV_R_NSSUPEIW_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL = FieldMetadata(
    name='NV_R_NSSUPEIW_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL',
    msb=10,
    lsb=10,
    register=NV_R_NSSUPEIW_PRIV_LEVEL_MASK
)

NV_R_NSSUPEIW_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_NSSUPEIW_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL_V_ZRRJKDVX',
    value=1,
    field=NV_R_NSSUPEIW_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL
)

NV_R_NSSUPEIW_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL = FieldMetadata(
    name='NV_R_NSSUPEIW_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL',
    msb=11,
    lsb=11,
    register=NV_R_NSSUPEIW_PRIV_LEVEL_MASK
)

NV_R_NSSUPEIW_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_NSSUPEIW_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL_V_ZRRJKDVX',
    value=1,
    field=NV_R_NSSUPEIW_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL
)

NV_R_NSSUPEIW_PRIV_LEVEL_MASK_WRITE_PROTECTION = FieldMetadata(
    name='NV_R_NSSUPEIW_PRIV_LEVEL_MASK_WRITE_PROTECTION',
    msb=7,
    lsb=4,
    register=NV_R_NSSUPEIW_PRIV_LEVEL_MASK
)

NV_R_NSSUPEIW_PRIV_LEVEL_MASK_WRITE_PROTECTION_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_NSSUPEIW_PRIV_LEVEL_MASK_WRITE_PROTECTION_V_ZRRJKDVX',
    value=15,
    field=NV_R_NSSUPEIW_PRIV_LEVEL_MASK_WRITE_PROTECTION
)

NV_R_NSSUPEIW_PRIV_LEVEL_MASK_WRITE_VIOLATION = FieldMetadata(
    name='NV_R_NSSUPEIW_PRIV_LEVEL_MASK_WRITE_VIOLATION',
    msb=9,
    lsb=9,
    register=NV_R_NSSUPEIW_PRIV_LEVEL_MASK
)

NV_R_NSSUPEIW_PRIV_LEVEL_MASK_WRITE_VIOLATION_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_NSSUPEIW_PRIV_LEVEL_MASK_WRITE_VIOLATION_V_ZRRJKDVX',
    value=1,
    field=NV_R_NSSUPEIW_PRIV_LEVEL_MASK_WRITE_VIOLATION
)

NV_R_XQNYUKHU_PRIV_LEVEL_MASK = RegisterMetadata(
    name='NV_R_XQNYUKHU_PRIV_LEVEL_MASK',
    address=0x2708
)

NV_R_XQNYUKHU_PRIV_LEVEL_MASK_READ_PROTECTION = FieldMetadata(
    name='NV_R_XQNYUKHU_PRIV_LEVEL_MASK_READ_PROTECTION',
    msb=3,
    lsb=0,
    register=NV_R_XQNYUKHU_PRIV_LEVEL_MASK
)

NV_R_XQNYUKHU_PRIV_LEVEL_MASK_READ_PROTECTION_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XQNYUKHU_PRIV_LEVEL_MASK_READ_PROTECTION_V_ZRRJKDVX',
    value=15,
    field=NV_R_XQNYUKHU_PRIV_LEVEL_MASK_READ_PROTECTION
)

NV_R_XQNYUKHU_PRIV_LEVEL_MASK_READ_VIOLATION = FieldMetadata(
    name='NV_R_XQNYUKHU_PRIV_LEVEL_MASK_READ_VIOLATION',
    msb=8,
    lsb=8,
    register=NV_R_XQNYUKHU_PRIV_LEVEL_MASK
)

NV_R_XQNYUKHU_PRIV_LEVEL_MASK_READ_VIOLATION_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XQNYUKHU_PRIV_LEVEL_MASK_READ_VIOLATION_V_ZRRJKDVX',
    value=1,
    field=NV_R_XQNYUKHU_PRIV_LEVEL_MASK_READ_VIOLATION
)

NV_R_XQNYUKHU_PRIV_LEVEL_MASK_SOURCE_ENABLE = FieldMetadata(
    name='NV_R_XQNYUKHU_PRIV_LEVEL_MASK_SOURCE_ENABLE',
    msb=31,
    lsb=12,
    register=NV_R_XQNYUKHU_PRIV_LEVEL_MASK
)

NV_R_XQNYUKHU_PRIV_LEVEL_MASK_SOURCE_ENABLE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XQNYUKHU_PRIV_LEVEL_MASK_SOURCE_ENABLE_V_ZRRJKDVX',
    value=1048575,
    field=NV_R_XQNYUKHU_PRIV_LEVEL_MASK_SOURCE_ENABLE
)

NV_R_XQNYUKHU_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL = FieldMetadata(
    name='NV_R_XQNYUKHU_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL',
    msb=10,
    lsb=10,
    register=NV_R_XQNYUKHU_PRIV_LEVEL_MASK
)

NV_R_XQNYUKHU_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XQNYUKHU_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL_V_ZRRJKDVX',
    value=1,
    field=NV_R_XQNYUKHU_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL
)

NV_R_XQNYUKHU_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL = FieldMetadata(
    name='NV_R_XQNYUKHU_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL',
    msb=11,
    lsb=11,
    register=NV_R_XQNYUKHU_PRIV_LEVEL_MASK
)

NV_R_XQNYUKHU_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XQNYUKHU_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL_V_ZRRJKDVX',
    value=1,
    field=NV_R_XQNYUKHU_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL
)

NV_R_XQNYUKHU_PRIV_LEVEL_MASK_WRITE_PROTECTION = FieldMetadata(
    name='NV_R_XQNYUKHU_PRIV_LEVEL_MASK_WRITE_PROTECTION',
    msb=7,
    lsb=4,
    register=NV_R_XQNYUKHU_PRIV_LEVEL_MASK
)

NV_R_XQNYUKHU_PRIV_LEVEL_MASK_WRITE_PROTECTION_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XQNYUKHU_PRIV_LEVEL_MASK_WRITE_PROTECTION_V_ZRRJKDVX',
    value=15,
    field=NV_R_XQNYUKHU_PRIV_LEVEL_MASK_WRITE_PROTECTION
)

NV_R_XQNYUKHU_PRIV_LEVEL_MASK_WRITE_VIOLATION = FieldMetadata(
    name='NV_R_XQNYUKHU_PRIV_LEVEL_MASK_WRITE_VIOLATION',
    msb=9,
    lsb=9,
    register=NV_R_XQNYUKHU_PRIV_LEVEL_MASK
)

NV_R_XQNYUKHU_PRIV_LEVEL_MASK_WRITE_VIOLATION_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XQNYUKHU_PRIV_LEVEL_MASK_WRITE_VIOLATION_V_ZRRJKDVX',
    value=1,
    field=NV_R_XQNYUKHU_PRIV_LEVEL_MASK_WRITE_VIOLATION
)

NV_R_UWDZGSFU_PRIV_LEVEL_MASK = RegisterMetadata(
    name='NV_R_UWDZGSFU_PRIV_LEVEL_MASK',
    address=0x4708
)

NV_R_UWDZGSFU_PRIV_LEVEL_MASK_READ_PROTECTION = FieldMetadata(
    name='NV_R_UWDZGSFU_PRIV_LEVEL_MASK_READ_PROTECTION',
    msb=3,
    lsb=0,
    register=NV_R_UWDZGSFU_PRIV_LEVEL_MASK
)

NV_R_UWDZGSFU_PRIV_LEVEL_MASK_READ_PROTECTION_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_UWDZGSFU_PRIV_LEVEL_MASK_READ_PROTECTION_V_ZRRJKDVX',
    value=15,
    field=NV_R_UWDZGSFU_PRIV_LEVEL_MASK_READ_PROTECTION
)

NV_R_UWDZGSFU_PRIV_LEVEL_MASK_READ_VIOLATION = FieldMetadata(
    name='NV_R_UWDZGSFU_PRIV_LEVEL_MASK_READ_VIOLATION',
    msb=8,
    lsb=8,
    register=NV_R_UWDZGSFU_PRIV_LEVEL_MASK
)

NV_R_UWDZGSFU_PRIV_LEVEL_MASK_READ_VIOLATION_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_UWDZGSFU_PRIV_LEVEL_MASK_READ_VIOLATION_V_ZRRJKDVX',
    value=1,
    field=NV_R_UWDZGSFU_PRIV_LEVEL_MASK_READ_VIOLATION
)

NV_R_UWDZGSFU_PRIV_LEVEL_MASK_SOURCE_ENABLE = FieldMetadata(
    name='NV_R_UWDZGSFU_PRIV_LEVEL_MASK_SOURCE_ENABLE',
    msb=31,
    lsb=12,
    register=NV_R_UWDZGSFU_PRIV_LEVEL_MASK
)

NV_R_UWDZGSFU_PRIV_LEVEL_MASK_SOURCE_ENABLE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_UWDZGSFU_PRIV_LEVEL_MASK_SOURCE_ENABLE_V_ZRRJKDVX',
    value=1048575,
    field=NV_R_UWDZGSFU_PRIV_LEVEL_MASK_SOURCE_ENABLE
)

NV_R_UWDZGSFU_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL = FieldMetadata(
    name='NV_R_UWDZGSFU_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL',
    msb=10,
    lsb=10,
    register=NV_R_UWDZGSFU_PRIV_LEVEL_MASK
)

NV_R_UWDZGSFU_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_UWDZGSFU_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL_V_ZRRJKDVX',
    value=1,
    field=NV_R_UWDZGSFU_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL
)

NV_R_UWDZGSFU_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL = FieldMetadata(
    name='NV_R_UWDZGSFU_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL',
    msb=11,
    lsb=11,
    register=NV_R_UWDZGSFU_PRIV_LEVEL_MASK
)

NV_R_UWDZGSFU_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_UWDZGSFU_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL_V_ZRRJKDVX',
    value=1,
    field=NV_R_UWDZGSFU_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL
)

NV_R_UWDZGSFU_PRIV_LEVEL_MASK_WRITE_PROTECTION = FieldMetadata(
    name='NV_R_UWDZGSFU_PRIV_LEVEL_MASK_WRITE_PROTECTION',
    msb=7,
    lsb=4,
    register=NV_R_UWDZGSFU_PRIV_LEVEL_MASK
)

NV_R_UWDZGSFU_PRIV_LEVEL_MASK_WRITE_PROTECTION_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_UWDZGSFU_PRIV_LEVEL_MASK_WRITE_PROTECTION_V_ZRRJKDVX',
    value=15,
    field=NV_R_UWDZGSFU_PRIV_LEVEL_MASK_WRITE_PROTECTION
)

NV_R_UWDZGSFU_PRIV_LEVEL_MASK_WRITE_VIOLATION = FieldMetadata(
    name='NV_R_UWDZGSFU_PRIV_LEVEL_MASK_WRITE_VIOLATION',
    msb=9,
    lsb=9,
    register=NV_R_UWDZGSFU_PRIV_LEVEL_MASK
)

NV_R_UWDZGSFU_PRIV_LEVEL_MASK_WRITE_VIOLATION_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_UWDZGSFU_PRIV_LEVEL_MASK_WRITE_VIOLATION_V_ZRRJKDVX',
    value=1,
    field=NV_R_UWDZGSFU_PRIV_LEVEL_MASK_WRITE_VIOLATION
)

NV_R_TCVDBRXJ_PRIV_LEVEL_MASK = RegisterMetadata(
    name='NV_R_TCVDBRXJ_PRIV_LEVEL_MASK',
    address=0x6708
)

NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_READ_PROTECTION = FieldMetadata(
    name='NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_READ_PROTECTION',
    msb=3,
    lsb=0,
    register=NV_R_TCVDBRXJ_PRIV_LEVEL_MASK
)

NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_READ_PROTECTION_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_READ_PROTECTION_V_ZRRJKDVX',
    value=15,
    field=NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_READ_PROTECTION
)

NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_READ_VIOLATION = FieldMetadata(
    name='NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_READ_VIOLATION',
    msb=8,
    lsb=8,
    register=NV_R_TCVDBRXJ_PRIV_LEVEL_MASK
)

NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_READ_VIOLATION_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_READ_VIOLATION_V_ZRRJKDVX',
    value=1,
    field=NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_READ_VIOLATION
)

NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_SOURCE_ENABLE = FieldMetadata(
    name='NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_SOURCE_ENABLE',
    msb=31,
    lsb=12,
    register=NV_R_TCVDBRXJ_PRIV_LEVEL_MASK
)

NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_SOURCE_ENABLE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_SOURCE_ENABLE_V_ZRRJKDVX',
    value=1048575,
    field=NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_SOURCE_ENABLE
)

NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL = FieldMetadata(
    name='NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL',
    msb=10,
    lsb=10,
    register=NV_R_TCVDBRXJ_PRIV_LEVEL_MASK
)

NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL_V_ZRRJKDVX',
    value=1,
    field=NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_SOURCE_READ_CONTROL
)

NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL = FieldMetadata(
    name='NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL',
    msb=11,
    lsb=11,
    register=NV_R_TCVDBRXJ_PRIV_LEVEL_MASK
)

NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL_V_ZRRJKDVX',
    value=1,
    field=NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_SOURCE_WRITE_CONTROL
)

NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_WRITE_PROTECTION = FieldMetadata(
    name='NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_WRITE_PROTECTION',
    msb=7,
    lsb=4,
    register=NV_R_TCVDBRXJ_PRIV_LEVEL_MASK
)

NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_WRITE_PROTECTION_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_WRITE_PROTECTION_V_ZRRJKDVX',
    value=15,
    field=NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_WRITE_PROTECTION
)

NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_WRITE_VIOLATION = FieldMetadata(
    name='NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_WRITE_VIOLATION',
    msb=9,
    lsb=9,
    register=NV_R_TCVDBRXJ_PRIV_LEVEL_MASK
)

NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_WRITE_VIOLATION_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_WRITE_VIOLATION_V_ZRRJKDVX',
    value=1,
    field=NV_R_TCVDBRXJ_PRIV_LEVEL_MASK_WRITE_VIOLATION
)

NV_R_VNICEGYY = RegisterMetadata(
    name='NV_R_VNICEGYY',
    address=0x788
)

NV_R_VNICEGYY_F_AUUPAXHZ = FieldMetadata(
    name='NV_R_VNICEGYY_F_AUUPAXHZ',
    msb=7,
    lsb=7,
    register=NV_R_VNICEGYY
)

NV_R_VNICEGYY_F_AUUPAXHZ_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VNICEGYY_F_AUUPAXHZ_V_ZRRJKDVX',
    value=0,
    field=NV_R_VNICEGYY_F_AUUPAXHZ
)

NV_R_VNICEGYY_F_REDTGIDZ = FieldMetadata(
    name='NV_R_VNICEGYY_F_REDTGIDZ',
    msb=4,
    lsb=4,
    register=NV_R_VNICEGYY
)

NV_R_VNICEGYY_F_REDTGIDZ_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VNICEGYY_F_REDTGIDZ_V_ZRRJKDVX',
    value=1,
    field=NV_R_VNICEGYY_F_REDTGIDZ
)

NV_R_VNICEGYY_F_QZQAGPDP = FieldMetadata(
    name='NV_R_VNICEGYY_F_QZQAGPDP',
    msb=0,
    lsb=0,
    register=NV_R_VNICEGYY
)

NV_R_VNICEGYY_F_QZQAGPDP_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VNICEGYY_F_QZQAGPDP_V_ZRRJKDVX',
    value=0,
    field=NV_R_VNICEGYY_F_QZQAGPDP
)

NV_R_VNICEGYY_F_FWGUVILY = FieldMetadata(
    name='NV_R_VNICEGYY_F_FWGUVILY',
    msb=10,
    lsb=8,
    register=NV_R_VNICEGYY
)

NV_R_VNICEGYY_F_FWGUVILY_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VNICEGYY_F_FWGUVILY_V_ZRRJKDVX',
    value=0,
    field=NV_R_VNICEGYY_F_FWGUVILY
)

NV_R_VNICEGYY_F_WHRWJGMO = FieldMetadata(
    name='NV_R_VNICEGYY_F_WHRWJGMO',
    msb=5,
    lsb=5,
    register=NV_R_VNICEGYY
)

NV_R_VNICEGYY_F_WHRWJGMO_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VNICEGYY_F_WHRWJGMO_V_ZRRJKDVX',
    value=0,
    field=NV_R_VNICEGYY_F_WHRWJGMO
)

NV_R_TTGMCXQP = RegisterMetadata(
    name='NV_R_TTGMCXQP',
    address=0x830
)

NV_R_TTGMCXQP_F_PXIMDYUO = FieldMetadata(
    name='NV_R_TTGMCXQP_F_PXIMDYUO',
    msb=30,
    lsb=0,
    register=NV_R_TTGMCXQP
)

NV_R_TTGMCXQP_F_PXIMDYUO_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TTGMCXQP_F_PXIMDYUO_V_ZRRJKDVX',
    value=0,
    field=NV_R_TTGMCXQP_F_PXIMDYUO
)

NV_R_TTGMCXQP_VALID = FieldMetadata(
    name='NV_R_TTGMCXQP_VALID',
    msb=31,
    lsb=31,
    register=NV_R_TTGMCXQP
)

NV_R_TTGMCXQP_VALID_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TTGMCXQP_VALID_V_ZRRJKDVX',
    value=0,
    field=NV_R_TTGMCXQP_VALID
)

NV_R_STSPSEPZ = RegisterMetadata(
    name='NV_R_STSPSEPZ',
    address=0x794,
    debug_dump={'tags': ['intr_status'], 'path': 'nvlink.netir', 'samples': 1000}
)

NV_R_STSPSEPZ_F_AQFWRERE = FieldMetadata(
    name='NV_R_STSPSEPZ_F_AQFWRERE',
    msb=1,
    lsb=1,
    register=NV_R_STSPSEPZ
)

NV_R_STSPSEPZ_F_AQFWRERE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_STSPSEPZ_F_AQFWRERE_V_ZRRJKDVX',
    value=0,
    field=NV_R_STSPSEPZ_F_AQFWRERE
)

NV_R_STSPSEPZ_F_WXNDRGMV = FieldMetadata(
    name='NV_R_STSPSEPZ_F_WXNDRGMV',
    msb=2,
    lsb=2,
    register=NV_R_STSPSEPZ
)

NV_R_STSPSEPZ_F_WXNDRGMV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_STSPSEPZ_F_WXNDRGMV_V_ZRRJKDVX',
    value=0,
    field=NV_R_STSPSEPZ_F_WXNDRGMV
)

NV_R_STSPSEPZ_F_HGPEBRTD = FieldMetadata(
    name='NV_R_STSPSEPZ_F_HGPEBRTD',
    msb=4,
    lsb=4,
    register=NV_R_STSPSEPZ
)

NV_R_STSPSEPZ_F_HGPEBRTD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_STSPSEPZ_F_HGPEBRTD_V_ZRRJKDVX',
    value=0,
    field=NV_R_STSPSEPZ_F_HGPEBRTD
)

NV_R_STSPSEPZ_F_TTDEBOJC = FieldMetadata(
    name='NV_R_STSPSEPZ_F_TTDEBOJC',
    msb=5,
    lsb=5,
    register=NV_R_STSPSEPZ
)

NV_R_STSPSEPZ_F_TTDEBOJC_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_STSPSEPZ_F_TTDEBOJC_V_ZRRJKDVX',
    value=0,
    field=NV_R_STSPSEPZ_F_TTDEBOJC
)

NV_R_STSPSEPZ_F_WXLVGPLS = FieldMetadata(
    name='NV_R_STSPSEPZ_F_WXLVGPLS',
    msb=3,
    lsb=3,
    register=NV_R_STSPSEPZ
)

NV_R_STSPSEPZ_F_WXLVGPLS_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_STSPSEPZ_F_WXLVGPLS_V_ZRRJKDVX',
    value=0,
    field=NV_R_STSPSEPZ_F_WXLVGPLS
)

NV_R_STSPSEPZ_F_GZUNECFA = FieldMetadata(
    name='NV_R_STSPSEPZ_F_GZUNECFA',
    msb=0,
    lsb=0,
    register=NV_R_STSPSEPZ
)

NV_R_STSPSEPZ_F_GZUNECFA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_STSPSEPZ_F_GZUNECFA_V_ZRRJKDVX',
    value=0,
    field=NV_R_STSPSEPZ_F_GZUNECFA
)

NV_R_LJHTJOLZ = RegisterMetadata(
    name='NV_R_LJHTJOLZ',
    address=0x828
)

NV_R_LJHTJOLZ_F_FMDCUIAD = FieldMetadata(
    name='NV_R_LJHTJOLZ_F_FMDCUIAD',
    msb=31,
    lsb=0,
    register=NV_R_LJHTJOLZ
)

NV_R_LJHTJOLZ_F_FMDCUIAD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LJHTJOLZ_F_FMDCUIAD_V_ZRRJKDVX',
    value=0,
    field=NV_R_LJHTJOLZ_F_FMDCUIAD
)

NV_R_WOHJOTQL = RegisterMetadata(
    name='NV_R_WOHJOTQL',
    address=0x82c
)

NV_R_WOHJOTQL_F_FMDCUIAD = FieldMetadata(
    name='NV_R_WOHJOTQL_F_FMDCUIAD',
    msb=31,
    lsb=0,
    register=NV_R_WOHJOTQL
)

NV_R_WOHJOTQL_F_FMDCUIAD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_WOHJOTQL_F_FMDCUIAD_V_ZRRJKDVX',
    value=0,
    field=NV_R_WOHJOTQL_F_FMDCUIAD
)

NV_R_PCBTKNXJ = RegisterMetadata(
    name='NV_R_PCBTKNXJ',
    address=0x824
)

NV_R_PCBTKNXJ_F_IYSPMDKO = FieldMetadata(
    name='NV_R_PCBTKNXJ_F_IYSPMDKO',
    msb=31,
    lsb=0,
    register=NV_R_PCBTKNXJ
)

NV_R_PCBTKNXJ_F_IYSPMDKO_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PCBTKNXJ_F_IYSPMDKO_V_ZRRJKDVX',
    value=0,
    field=NV_R_PCBTKNXJ_F_IYSPMDKO
)

NV_R_KDYBKKQJ = RegisterMetadata(
    name='NV_R_KDYBKKQJ',
    address=0x820
)

NV_R_KDYBKKQJ_F_PXIMDYUO = FieldMetadata(
    name='NV_R_KDYBKKQJ_F_PXIMDYUO',
    msb=29,
    lsb=0,
    register=NV_R_KDYBKKQJ
)

NV_R_KDYBKKQJ_F_PXIMDYUO_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_KDYBKKQJ_F_PXIMDYUO_V_ZRRJKDVX',
    value=0,
    field=NV_R_KDYBKKQJ_F_PXIMDYUO
)

NV_R_KDYBKKQJ_VALID = FieldMetadata(
    name='NV_R_KDYBKKQJ_VALID',
    msb=31,
    lsb=31,
    register=NV_R_KDYBKKQJ
)

NV_R_KDYBKKQJ_VALID_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_KDYBKKQJ_VALID_V_ZRRJKDVX',
    value=0,
    field=NV_R_KDYBKKQJ_VALID
)

NV_R_KDYBKKQJ_WRITE = FieldMetadata(
    name='NV_R_KDYBKKQJ_WRITE',
    msb=30,
    lsb=30,
    register=NV_R_KDYBKKQJ
)

NV_R_KDYBKKQJ_WRITE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_KDYBKKQJ_WRITE_V_ZRRJKDVX',
    value=0,
    field=NV_R_KDYBKKQJ_WRITE
)

NV_R_ELPQHXVO = RegisterMetadata(
    name='NV_R_ELPQHXVO',
    address=0x7ec,
    debug_dump={'tags': ['pc'], 'path': 'nvlink.netir', 'samples': 1000, 'aggregation': 'histogram'}
)

NV_R_ELPQHXVO_F_MYDNDUNC = FieldMetadata(
    name='NV_R_ELPQHXVO_F_MYDNDUNC',
    msb=31,
    lsb=0,
    register=NV_R_ELPQHXVO
)

NV_R_YPVCNZDM = RegisterMetadata(
    name='NV_R_YPVCNZDM',
    address=0x2788
)

NV_R_YPVCNZDM_F_AUUPAXHZ = FieldMetadata(
    name='NV_R_YPVCNZDM_F_AUUPAXHZ',
    msb=7,
    lsb=7,
    register=NV_R_YPVCNZDM
)

NV_R_YPVCNZDM_F_AUUPAXHZ_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YPVCNZDM_F_AUUPAXHZ_V_ZRRJKDVX',
    value=0,
    field=NV_R_YPVCNZDM_F_AUUPAXHZ
)

NV_R_YPVCNZDM_F_REDTGIDZ = FieldMetadata(
    name='NV_R_YPVCNZDM_F_REDTGIDZ',
    msb=4,
    lsb=4,
    register=NV_R_YPVCNZDM
)

NV_R_YPVCNZDM_F_REDTGIDZ_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YPVCNZDM_F_REDTGIDZ_V_ZRRJKDVX',
    value=1,
    field=NV_R_YPVCNZDM_F_REDTGIDZ
)

NV_R_YPVCNZDM_F_QZQAGPDP = FieldMetadata(
    name='NV_R_YPVCNZDM_F_QZQAGPDP',
    msb=0,
    lsb=0,
    register=NV_R_YPVCNZDM
)

NV_R_YPVCNZDM_F_QZQAGPDP_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YPVCNZDM_F_QZQAGPDP_V_ZRRJKDVX',
    value=0,
    field=NV_R_YPVCNZDM_F_QZQAGPDP
)

NV_R_YPVCNZDM_F_FWGUVILY = FieldMetadata(
    name='NV_R_YPVCNZDM_F_FWGUVILY',
    msb=10,
    lsb=8,
    register=NV_R_YPVCNZDM
)

NV_R_YPVCNZDM_F_FWGUVILY_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YPVCNZDM_F_FWGUVILY_V_ZRRJKDVX',
    value=0,
    field=NV_R_YPVCNZDM_F_FWGUVILY
)

NV_R_YPVCNZDM_F_WHRWJGMO = FieldMetadata(
    name='NV_R_YPVCNZDM_F_WHRWJGMO',
    msb=5,
    lsb=5,
    register=NV_R_YPVCNZDM
)

NV_R_YPVCNZDM_F_WHRWJGMO_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YPVCNZDM_F_WHRWJGMO_V_ZRRJKDVX',
    value=0,
    field=NV_R_YPVCNZDM_F_WHRWJGMO
)

NV_R_QTXXHYWI = RegisterMetadata(
    name='NV_R_QTXXHYWI',
    address=0x2830
)

NV_R_QTXXHYWI_F_PXIMDYUO = FieldMetadata(
    name='NV_R_QTXXHYWI_F_PXIMDYUO',
    msb=30,
    lsb=0,
    register=NV_R_QTXXHYWI
)

NV_R_QTXXHYWI_F_PXIMDYUO_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QTXXHYWI_F_PXIMDYUO_V_ZRRJKDVX',
    value=0,
    field=NV_R_QTXXHYWI_F_PXIMDYUO
)

NV_R_QTXXHYWI_VALID = FieldMetadata(
    name='NV_R_QTXXHYWI_VALID',
    msb=31,
    lsb=31,
    register=NV_R_QTXXHYWI
)

NV_R_QTXXHYWI_VALID_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QTXXHYWI_VALID_V_ZRRJKDVX',
    value=0,
    field=NV_R_QTXXHYWI_VALID
)

NV_R_FSWWCPCY = RegisterMetadata(
    name='NV_R_FSWWCPCY',
    address=0x2794,
    debug_dump={'tags': ['intr_status'], 'path': 'nvlink.netir', 'samples': 1000}
)

NV_R_FSWWCPCY_F_AQFWRERE = FieldMetadata(
    name='NV_R_FSWWCPCY_F_AQFWRERE',
    msb=1,
    lsb=1,
    register=NV_R_FSWWCPCY
)

NV_R_FSWWCPCY_F_AQFWRERE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FSWWCPCY_F_AQFWRERE_V_ZRRJKDVX',
    value=0,
    field=NV_R_FSWWCPCY_F_AQFWRERE
)

NV_R_FSWWCPCY_F_WXNDRGMV = FieldMetadata(
    name='NV_R_FSWWCPCY_F_WXNDRGMV',
    msb=2,
    lsb=2,
    register=NV_R_FSWWCPCY
)

NV_R_FSWWCPCY_F_WXNDRGMV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FSWWCPCY_F_WXNDRGMV_V_ZRRJKDVX',
    value=0,
    field=NV_R_FSWWCPCY_F_WXNDRGMV
)

NV_R_FSWWCPCY_F_HGPEBRTD = FieldMetadata(
    name='NV_R_FSWWCPCY_F_HGPEBRTD',
    msb=4,
    lsb=4,
    register=NV_R_FSWWCPCY
)

NV_R_FSWWCPCY_F_HGPEBRTD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FSWWCPCY_F_HGPEBRTD_V_ZRRJKDVX',
    value=0,
    field=NV_R_FSWWCPCY_F_HGPEBRTD
)

NV_R_FSWWCPCY_F_TTDEBOJC = FieldMetadata(
    name='NV_R_FSWWCPCY_F_TTDEBOJC',
    msb=5,
    lsb=5,
    register=NV_R_FSWWCPCY
)

NV_R_FSWWCPCY_F_TTDEBOJC_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FSWWCPCY_F_TTDEBOJC_V_ZRRJKDVX',
    value=0,
    field=NV_R_FSWWCPCY_F_TTDEBOJC
)

NV_R_FSWWCPCY_F_WXLVGPLS = FieldMetadata(
    name='NV_R_FSWWCPCY_F_WXLVGPLS',
    msb=3,
    lsb=3,
    register=NV_R_FSWWCPCY
)

NV_R_FSWWCPCY_F_WXLVGPLS_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FSWWCPCY_F_WXLVGPLS_V_ZRRJKDVX',
    value=0,
    field=NV_R_FSWWCPCY_F_WXLVGPLS
)

NV_R_FSWWCPCY_F_GZUNECFA = FieldMetadata(
    name='NV_R_FSWWCPCY_F_GZUNECFA',
    msb=0,
    lsb=0,
    register=NV_R_FSWWCPCY
)

NV_R_FSWWCPCY_F_GZUNECFA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FSWWCPCY_F_GZUNECFA_V_ZRRJKDVX',
    value=0,
    field=NV_R_FSWWCPCY_F_GZUNECFA
)

NV_R_VTBTRZNA = RegisterMetadata(
    name='NV_R_VTBTRZNA',
    address=0x2828
)

NV_R_VTBTRZNA_F_FMDCUIAD = FieldMetadata(
    name='NV_R_VTBTRZNA_F_FMDCUIAD',
    msb=31,
    lsb=0,
    register=NV_R_VTBTRZNA
)

NV_R_VTBTRZNA_F_FMDCUIAD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VTBTRZNA_F_FMDCUIAD_V_ZRRJKDVX',
    value=0,
    field=NV_R_VTBTRZNA_F_FMDCUIAD
)

NV_R_IZYKEUUL = RegisterMetadata(
    name='NV_R_IZYKEUUL',
    address=0x282c
)

NV_R_IZYKEUUL_F_FMDCUIAD = FieldMetadata(
    name='NV_R_IZYKEUUL_F_FMDCUIAD',
    msb=31,
    lsb=0,
    register=NV_R_IZYKEUUL
)

NV_R_IZYKEUUL_F_FMDCUIAD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_IZYKEUUL_F_FMDCUIAD_V_ZRRJKDVX',
    value=0,
    field=NV_R_IZYKEUUL_F_FMDCUIAD
)

NV_R_YRXCKMBR = RegisterMetadata(
    name='NV_R_YRXCKMBR',
    address=0x2824
)

NV_R_YRXCKMBR_F_IYSPMDKO = FieldMetadata(
    name='NV_R_YRXCKMBR_F_IYSPMDKO',
    msb=31,
    lsb=0,
    register=NV_R_YRXCKMBR
)

NV_R_YRXCKMBR_F_IYSPMDKO_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YRXCKMBR_F_IYSPMDKO_V_ZRRJKDVX',
    value=0,
    field=NV_R_YRXCKMBR_F_IYSPMDKO
)

NV_R_BILYNUIP = RegisterMetadata(
    name='NV_R_BILYNUIP',
    address=0x2820
)

NV_R_BILYNUIP_F_PXIMDYUO = FieldMetadata(
    name='NV_R_BILYNUIP_F_PXIMDYUO',
    msb=29,
    lsb=0,
    register=NV_R_BILYNUIP
)

NV_R_BILYNUIP_F_PXIMDYUO_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_BILYNUIP_F_PXIMDYUO_V_ZRRJKDVX',
    value=0,
    field=NV_R_BILYNUIP_F_PXIMDYUO
)

NV_R_BILYNUIP_VALID = FieldMetadata(
    name='NV_R_BILYNUIP_VALID',
    msb=31,
    lsb=31,
    register=NV_R_BILYNUIP
)

NV_R_BILYNUIP_VALID_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_BILYNUIP_VALID_V_ZRRJKDVX',
    value=0,
    field=NV_R_BILYNUIP_VALID
)

NV_R_BILYNUIP_WRITE = FieldMetadata(
    name='NV_R_BILYNUIP_WRITE',
    msb=30,
    lsb=30,
    register=NV_R_BILYNUIP
)

NV_R_BILYNUIP_WRITE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_BILYNUIP_WRITE_V_ZRRJKDVX',
    value=0,
    field=NV_R_BILYNUIP_WRITE
)

NV_R_UJFQYDOQ = RegisterMetadata(
    name='NV_R_UJFQYDOQ',
    address=0x27ec,
    debug_dump={'tags': ['pc'], 'path': 'nvlink.netir', 'samples': 1000, 'aggregation': 'histogram'}
)

NV_R_UJFQYDOQ_F_MYDNDUNC = FieldMetadata(
    name='NV_R_UJFQYDOQ_F_MYDNDUNC',
    msb=31,
    lsb=0,
    register=NV_R_UJFQYDOQ
)

NV_R_RWEJDRRM = RegisterMetadata(
    name='NV_R_RWEJDRRM',
    address=0x4788
)

NV_R_RWEJDRRM_F_AUUPAXHZ = FieldMetadata(
    name='NV_R_RWEJDRRM_F_AUUPAXHZ',
    msb=7,
    lsb=7,
    register=NV_R_RWEJDRRM
)

NV_R_RWEJDRRM_F_AUUPAXHZ_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_RWEJDRRM_F_AUUPAXHZ_V_ZRRJKDVX',
    value=0,
    field=NV_R_RWEJDRRM_F_AUUPAXHZ
)

NV_R_RWEJDRRM_F_REDTGIDZ = FieldMetadata(
    name='NV_R_RWEJDRRM_F_REDTGIDZ',
    msb=4,
    lsb=4,
    register=NV_R_RWEJDRRM
)

NV_R_RWEJDRRM_F_REDTGIDZ_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_RWEJDRRM_F_REDTGIDZ_V_ZRRJKDVX',
    value=1,
    field=NV_R_RWEJDRRM_F_REDTGIDZ
)

NV_R_RWEJDRRM_F_QZQAGPDP = FieldMetadata(
    name='NV_R_RWEJDRRM_F_QZQAGPDP',
    msb=0,
    lsb=0,
    register=NV_R_RWEJDRRM
)

NV_R_RWEJDRRM_F_QZQAGPDP_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_RWEJDRRM_F_QZQAGPDP_V_ZRRJKDVX',
    value=0,
    field=NV_R_RWEJDRRM_F_QZQAGPDP
)

NV_R_RWEJDRRM_F_FWGUVILY = FieldMetadata(
    name='NV_R_RWEJDRRM_F_FWGUVILY',
    msb=10,
    lsb=8,
    register=NV_R_RWEJDRRM
)

NV_R_RWEJDRRM_F_FWGUVILY_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_RWEJDRRM_F_FWGUVILY_V_ZRRJKDVX',
    value=0,
    field=NV_R_RWEJDRRM_F_FWGUVILY
)

NV_R_RWEJDRRM_F_WHRWJGMO = FieldMetadata(
    name='NV_R_RWEJDRRM_F_WHRWJGMO',
    msb=5,
    lsb=5,
    register=NV_R_RWEJDRRM
)

NV_R_RWEJDRRM_F_WHRWJGMO_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_RWEJDRRM_F_WHRWJGMO_V_ZRRJKDVX',
    value=0,
    field=NV_R_RWEJDRRM_F_WHRWJGMO
)

NV_R_DJCICOEA = RegisterMetadata(
    name='NV_R_DJCICOEA',
    address=0x4830
)

NV_R_DJCICOEA_F_PXIMDYUO = FieldMetadata(
    name='NV_R_DJCICOEA_F_PXIMDYUO',
    msb=30,
    lsb=0,
    register=NV_R_DJCICOEA
)

NV_R_DJCICOEA_F_PXIMDYUO_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DJCICOEA_F_PXIMDYUO_V_ZRRJKDVX',
    value=0,
    field=NV_R_DJCICOEA_F_PXIMDYUO
)

NV_R_DJCICOEA_VALID = FieldMetadata(
    name='NV_R_DJCICOEA_VALID',
    msb=31,
    lsb=31,
    register=NV_R_DJCICOEA
)

NV_R_DJCICOEA_VALID_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DJCICOEA_VALID_V_ZRRJKDVX',
    value=0,
    field=NV_R_DJCICOEA_VALID
)

NV_R_ZOWCMTPS = RegisterMetadata(
    name='NV_R_ZOWCMTPS',
    address=0x4794,
    debug_dump={'tags': ['intr_status'], 'path': 'nvlink.netir', 'samples': 1000}
)

NV_R_ZOWCMTPS_F_AQFWRERE = FieldMetadata(
    name='NV_R_ZOWCMTPS_F_AQFWRERE',
    msb=1,
    lsb=1,
    register=NV_R_ZOWCMTPS
)

NV_R_ZOWCMTPS_F_AQFWRERE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZOWCMTPS_F_AQFWRERE_V_ZRRJKDVX',
    value=0,
    field=NV_R_ZOWCMTPS_F_AQFWRERE
)

NV_R_ZOWCMTPS_F_WXNDRGMV = FieldMetadata(
    name='NV_R_ZOWCMTPS_F_WXNDRGMV',
    msb=2,
    lsb=2,
    register=NV_R_ZOWCMTPS
)

NV_R_ZOWCMTPS_F_WXNDRGMV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZOWCMTPS_F_WXNDRGMV_V_ZRRJKDVX',
    value=0,
    field=NV_R_ZOWCMTPS_F_WXNDRGMV
)

NV_R_ZOWCMTPS_F_HGPEBRTD = FieldMetadata(
    name='NV_R_ZOWCMTPS_F_HGPEBRTD',
    msb=4,
    lsb=4,
    register=NV_R_ZOWCMTPS
)

NV_R_ZOWCMTPS_F_HGPEBRTD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZOWCMTPS_F_HGPEBRTD_V_ZRRJKDVX',
    value=0,
    field=NV_R_ZOWCMTPS_F_HGPEBRTD
)

NV_R_ZOWCMTPS_F_TTDEBOJC = FieldMetadata(
    name='NV_R_ZOWCMTPS_F_TTDEBOJC',
    msb=5,
    lsb=5,
    register=NV_R_ZOWCMTPS
)

NV_R_ZOWCMTPS_F_TTDEBOJC_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZOWCMTPS_F_TTDEBOJC_V_ZRRJKDVX',
    value=0,
    field=NV_R_ZOWCMTPS_F_TTDEBOJC
)

NV_R_ZOWCMTPS_F_WXLVGPLS = FieldMetadata(
    name='NV_R_ZOWCMTPS_F_WXLVGPLS',
    msb=3,
    lsb=3,
    register=NV_R_ZOWCMTPS
)

NV_R_ZOWCMTPS_F_WXLVGPLS_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZOWCMTPS_F_WXLVGPLS_V_ZRRJKDVX',
    value=0,
    field=NV_R_ZOWCMTPS_F_WXLVGPLS
)

NV_R_ZOWCMTPS_F_GZUNECFA = FieldMetadata(
    name='NV_R_ZOWCMTPS_F_GZUNECFA',
    msb=0,
    lsb=0,
    register=NV_R_ZOWCMTPS
)

NV_R_ZOWCMTPS_F_GZUNECFA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZOWCMTPS_F_GZUNECFA_V_ZRRJKDVX',
    value=0,
    field=NV_R_ZOWCMTPS_F_GZUNECFA
)

NV_R_FDUWOCYR = RegisterMetadata(
    name='NV_R_FDUWOCYR',
    address=0x4828
)

NV_R_FDUWOCYR_F_FMDCUIAD = FieldMetadata(
    name='NV_R_FDUWOCYR_F_FMDCUIAD',
    msb=31,
    lsb=0,
    register=NV_R_FDUWOCYR
)

NV_R_FDUWOCYR_F_FMDCUIAD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FDUWOCYR_F_FMDCUIAD_V_ZRRJKDVX',
    value=0,
    field=NV_R_FDUWOCYR_F_FMDCUIAD
)

NV_R_NYNKFBPK = RegisterMetadata(
    name='NV_R_NYNKFBPK',
    address=0x482c
)

NV_R_NYNKFBPK_F_FMDCUIAD = FieldMetadata(
    name='NV_R_NYNKFBPK_F_FMDCUIAD',
    msb=31,
    lsb=0,
    register=NV_R_NYNKFBPK
)

NV_R_NYNKFBPK_F_FMDCUIAD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_NYNKFBPK_F_FMDCUIAD_V_ZRRJKDVX',
    value=0,
    field=NV_R_NYNKFBPK_F_FMDCUIAD
)

NV_R_KIYCHXFK = RegisterMetadata(
    name='NV_R_KIYCHXFK',
    address=0x4824
)

NV_R_KIYCHXFK_F_IYSPMDKO = FieldMetadata(
    name='NV_R_KIYCHXFK_F_IYSPMDKO',
    msb=31,
    lsb=0,
    register=NV_R_KIYCHXFK
)

NV_R_KIYCHXFK_F_IYSPMDKO_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_KIYCHXFK_F_IYSPMDKO_V_ZRRJKDVX',
    value=0,
    field=NV_R_KIYCHXFK_F_IYSPMDKO
)

NV_R_UPLZCDYK = RegisterMetadata(
    name='NV_R_UPLZCDYK',
    address=0x4820
)

NV_R_UPLZCDYK_F_PXIMDYUO = FieldMetadata(
    name='NV_R_UPLZCDYK_F_PXIMDYUO',
    msb=29,
    lsb=0,
    register=NV_R_UPLZCDYK
)

NV_R_UPLZCDYK_F_PXIMDYUO_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_UPLZCDYK_F_PXIMDYUO_V_ZRRJKDVX',
    value=0,
    field=NV_R_UPLZCDYK_F_PXIMDYUO
)

NV_R_UPLZCDYK_VALID = FieldMetadata(
    name='NV_R_UPLZCDYK_VALID',
    msb=31,
    lsb=31,
    register=NV_R_UPLZCDYK
)

NV_R_UPLZCDYK_VALID_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_UPLZCDYK_VALID_V_ZRRJKDVX',
    value=0,
    field=NV_R_UPLZCDYK_VALID
)

NV_R_UPLZCDYK_WRITE = FieldMetadata(
    name='NV_R_UPLZCDYK_WRITE',
    msb=30,
    lsb=30,
    register=NV_R_UPLZCDYK
)

NV_R_UPLZCDYK_WRITE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_UPLZCDYK_WRITE_V_ZRRJKDVX',
    value=0,
    field=NV_R_UPLZCDYK_WRITE
)

NV_R_MCDKKRBG = RegisterMetadata(
    name='NV_R_MCDKKRBG',
    address=0x47ec,
    debug_dump={'tags': ['pc'], 'path': 'nvlink.netir', 'samples': 1000, 'aggregation': 'histogram'}
)

NV_R_MCDKKRBG_F_MYDNDUNC = FieldMetadata(
    name='NV_R_MCDKKRBG_F_MYDNDUNC',
    msb=31,
    lsb=0,
    register=NV_R_MCDKKRBG
)

NV_R_LQUNLRMR = RegisterMetadata(
    name='NV_R_LQUNLRMR',
    address=0x6788
)

NV_R_LQUNLRMR_F_AUUPAXHZ = FieldMetadata(
    name='NV_R_LQUNLRMR_F_AUUPAXHZ',
    msb=7,
    lsb=7,
    register=NV_R_LQUNLRMR
)

NV_R_LQUNLRMR_F_AUUPAXHZ_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LQUNLRMR_F_AUUPAXHZ_V_ZRRJKDVX',
    value=0,
    field=NV_R_LQUNLRMR_F_AUUPAXHZ
)

NV_R_LQUNLRMR_F_REDTGIDZ = FieldMetadata(
    name='NV_R_LQUNLRMR_F_REDTGIDZ',
    msb=4,
    lsb=4,
    register=NV_R_LQUNLRMR
)

NV_R_LQUNLRMR_F_REDTGIDZ_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LQUNLRMR_F_REDTGIDZ_V_ZRRJKDVX',
    value=1,
    field=NV_R_LQUNLRMR_F_REDTGIDZ
)

NV_R_LQUNLRMR_F_QZQAGPDP = FieldMetadata(
    name='NV_R_LQUNLRMR_F_QZQAGPDP',
    msb=0,
    lsb=0,
    register=NV_R_LQUNLRMR
)

NV_R_LQUNLRMR_F_QZQAGPDP_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LQUNLRMR_F_QZQAGPDP_V_ZRRJKDVX',
    value=0,
    field=NV_R_LQUNLRMR_F_QZQAGPDP
)

NV_R_LQUNLRMR_F_FWGUVILY = FieldMetadata(
    name='NV_R_LQUNLRMR_F_FWGUVILY',
    msb=10,
    lsb=8,
    register=NV_R_LQUNLRMR
)

NV_R_LQUNLRMR_F_FWGUVILY_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LQUNLRMR_F_FWGUVILY_V_ZRRJKDVX',
    value=0,
    field=NV_R_LQUNLRMR_F_FWGUVILY
)

NV_R_LQUNLRMR_F_WHRWJGMO = FieldMetadata(
    name='NV_R_LQUNLRMR_F_WHRWJGMO',
    msb=5,
    lsb=5,
    register=NV_R_LQUNLRMR
)

NV_R_LQUNLRMR_F_WHRWJGMO_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LQUNLRMR_F_WHRWJGMO_V_ZRRJKDVX',
    value=0,
    field=NV_R_LQUNLRMR_F_WHRWJGMO
)

NV_R_VXGWAHOX = RegisterMetadata(
    name='NV_R_VXGWAHOX',
    address=0x6830
)

NV_R_VXGWAHOX_F_PXIMDYUO = FieldMetadata(
    name='NV_R_VXGWAHOX_F_PXIMDYUO',
    msb=30,
    lsb=0,
    register=NV_R_VXGWAHOX
)

NV_R_VXGWAHOX_F_PXIMDYUO_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VXGWAHOX_F_PXIMDYUO_V_ZRRJKDVX',
    value=0,
    field=NV_R_VXGWAHOX_F_PXIMDYUO
)

NV_R_VXGWAHOX_VALID = FieldMetadata(
    name='NV_R_VXGWAHOX_VALID',
    msb=31,
    lsb=31,
    register=NV_R_VXGWAHOX
)

NV_R_VXGWAHOX_VALID_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VXGWAHOX_VALID_V_ZRRJKDVX',
    value=0,
    field=NV_R_VXGWAHOX_VALID
)

NV_R_IJJCEDMT = RegisterMetadata(
    name='NV_R_IJJCEDMT',
    address=0x6794,
    debug_dump={'tags': ['intr_status'], 'path': 'nvlink.netir', 'samples': 1000}
)

NV_R_IJJCEDMT_F_AQFWRERE = FieldMetadata(
    name='NV_R_IJJCEDMT_F_AQFWRERE',
    msb=1,
    lsb=1,
    register=NV_R_IJJCEDMT
)

NV_R_IJJCEDMT_F_AQFWRERE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_IJJCEDMT_F_AQFWRERE_V_ZRRJKDVX',
    value=0,
    field=NV_R_IJJCEDMT_F_AQFWRERE
)

NV_R_IJJCEDMT_F_WXNDRGMV = FieldMetadata(
    name='NV_R_IJJCEDMT_F_WXNDRGMV',
    msb=2,
    lsb=2,
    register=NV_R_IJJCEDMT
)

NV_R_IJJCEDMT_F_WXNDRGMV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_IJJCEDMT_F_WXNDRGMV_V_ZRRJKDVX',
    value=0,
    field=NV_R_IJJCEDMT_F_WXNDRGMV
)

NV_R_IJJCEDMT_F_HGPEBRTD = FieldMetadata(
    name='NV_R_IJJCEDMT_F_HGPEBRTD',
    msb=4,
    lsb=4,
    register=NV_R_IJJCEDMT
)

NV_R_IJJCEDMT_F_HGPEBRTD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_IJJCEDMT_F_HGPEBRTD_V_ZRRJKDVX',
    value=0,
    field=NV_R_IJJCEDMT_F_HGPEBRTD
)

NV_R_IJJCEDMT_F_TTDEBOJC = FieldMetadata(
    name='NV_R_IJJCEDMT_F_TTDEBOJC',
    msb=5,
    lsb=5,
    register=NV_R_IJJCEDMT
)

NV_R_IJJCEDMT_F_TTDEBOJC_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_IJJCEDMT_F_TTDEBOJC_V_ZRRJKDVX',
    value=0,
    field=NV_R_IJJCEDMT_F_TTDEBOJC
)

NV_R_IJJCEDMT_F_WXLVGPLS = FieldMetadata(
    name='NV_R_IJJCEDMT_F_WXLVGPLS',
    msb=3,
    lsb=3,
    register=NV_R_IJJCEDMT
)

NV_R_IJJCEDMT_F_WXLVGPLS_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_IJJCEDMT_F_WXLVGPLS_V_ZRRJKDVX',
    value=0,
    field=NV_R_IJJCEDMT_F_WXLVGPLS
)

NV_R_IJJCEDMT_F_GZUNECFA = FieldMetadata(
    name='NV_R_IJJCEDMT_F_GZUNECFA',
    msb=0,
    lsb=0,
    register=NV_R_IJJCEDMT
)

NV_R_IJJCEDMT_F_GZUNECFA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_IJJCEDMT_F_GZUNECFA_V_ZRRJKDVX',
    value=0,
    field=NV_R_IJJCEDMT_F_GZUNECFA
)

NV_R_DDWJOZIG = RegisterMetadata(
    name='NV_R_DDWJOZIG',
    address=0x6828
)

NV_R_DDWJOZIG_F_FMDCUIAD = FieldMetadata(
    name='NV_R_DDWJOZIG_F_FMDCUIAD',
    msb=31,
    lsb=0,
    register=NV_R_DDWJOZIG
)

NV_R_DDWJOZIG_F_FMDCUIAD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DDWJOZIG_F_FMDCUIAD_V_ZRRJKDVX',
    value=0,
    field=NV_R_DDWJOZIG_F_FMDCUIAD
)

NV_R_GUZPRMLE = RegisterMetadata(
    name='NV_R_GUZPRMLE',
    address=0x682c
)

NV_R_GUZPRMLE_F_FMDCUIAD = FieldMetadata(
    name='NV_R_GUZPRMLE_F_FMDCUIAD',
    msb=31,
    lsb=0,
    register=NV_R_GUZPRMLE
)

NV_R_GUZPRMLE_F_FMDCUIAD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_GUZPRMLE_F_FMDCUIAD_V_ZRRJKDVX',
    value=0,
    field=NV_R_GUZPRMLE_F_FMDCUIAD
)

NV_R_LEDITTSI = RegisterMetadata(
    name='NV_R_LEDITTSI',
    address=0x6824
)

NV_R_LEDITTSI_F_IYSPMDKO = FieldMetadata(
    name='NV_R_LEDITTSI_F_IYSPMDKO',
    msb=31,
    lsb=0,
    register=NV_R_LEDITTSI
)

NV_R_LEDITTSI_F_IYSPMDKO_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LEDITTSI_F_IYSPMDKO_V_ZRRJKDVX',
    value=0,
    field=NV_R_LEDITTSI_F_IYSPMDKO
)

NV_R_BVEDRQNP = RegisterMetadata(
    name='NV_R_BVEDRQNP',
    address=0x6820
)

NV_R_BVEDRQNP_F_PXIMDYUO = FieldMetadata(
    name='NV_R_BVEDRQNP_F_PXIMDYUO',
    msb=29,
    lsb=0,
    register=NV_R_BVEDRQNP
)

NV_R_BVEDRQNP_F_PXIMDYUO_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_BVEDRQNP_F_PXIMDYUO_V_ZRRJKDVX',
    value=0,
    field=NV_R_BVEDRQNP_F_PXIMDYUO
)

NV_R_BVEDRQNP_VALID = FieldMetadata(
    name='NV_R_BVEDRQNP_VALID',
    msb=31,
    lsb=31,
    register=NV_R_BVEDRQNP
)

NV_R_BVEDRQNP_VALID_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_BVEDRQNP_VALID_V_ZRRJKDVX',
    value=0,
    field=NV_R_BVEDRQNP_VALID
)

NV_R_BVEDRQNP_WRITE = FieldMetadata(
    name='NV_R_BVEDRQNP_WRITE',
    msb=30,
    lsb=30,
    register=NV_R_BVEDRQNP
)

NV_R_BVEDRQNP_WRITE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_BVEDRQNP_WRITE_V_ZRRJKDVX',
    value=0,
    field=NV_R_BVEDRQNP_WRITE
)

NV_R_XOATKXOJ = RegisterMetadata(
    name='NV_R_XOATKXOJ',
    address=0x67ec,
    debug_dump={'tags': ['pc'], 'path': 'nvlink.netir', 'samples': 1000, 'aggregation': 'histogram'}
)

NV_R_XOATKXOJ_F_MYDNDUNC = FieldMetadata(
    name='NV_R_XOATKXOJ_F_MYDNDUNC',
    msb=31,
    lsb=0,
    register=NV_R_XOATKXOJ
)

