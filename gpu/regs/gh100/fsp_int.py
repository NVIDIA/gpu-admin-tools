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
NV_R_YTOYHZGE = RegisterMetadata(
    name='NV_R_YTOYHZGE',
    address=0x8f0530,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_YTOYHZGE_F_PXCBEXUQ = FieldMetadata(
    name='NV_R_YTOYHZGE_F_PXCBEXUQ',
    msb=8,
    lsb=8,
    register=NV_R_YTOYHZGE
)

NV_R_YTOYHZGE_F_PXCBEXUQ_V_ACRDDIMM = ValueMetadata(
    name='NV_R_YTOYHZGE_F_PXCBEXUQ_V_ACRDDIMM',
    value=1,
    field=NV_R_YTOYHZGE_F_PXCBEXUQ
)
NV_R_YTOYHZGE_F_PXCBEXUQ_FALSE = ValueMetadata(
    name='NV_R_YTOYHZGE_F_PXCBEXUQ_FALSE',
    value=0,
    field=NV_R_YTOYHZGE_F_PXCBEXUQ
)
NV_R_YTOYHZGE_F_PXCBEXUQ_TRUE = ValueMetadata(
    name='NV_R_YTOYHZGE_F_PXCBEXUQ_TRUE',
    value=1,
    field=NV_R_YTOYHZGE_F_PXCBEXUQ
)

NV_R_YTOYHZGE_F_IXCWUGWK = FieldMetadata(
    name='NV_R_YTOYHZGE_F_IXCWUGWK',
    msb=12,
    lsb=12,
    register=NV_R_YTOYHZGE
)

NV_R_YTOYHZGE_F_IXCWUGWK_V_ACRDDIMM = ValueMetadata(
    name='NV_R_YTOYHZGE_F_IXCWUGWK_V_ACRDDIMM',
    value=1,
    field=NV_R_YTOYHZGE_F_IXCWUGWK
)
NV_R_YTOYHZGE_F_IXCWUGWK_FALSE = ValueMetadata(
    name='NV_R_YTOYHZGE_F_IXCWUGWK_FALSE',
    value=0,
    field=NV_R_YTOYHZGE_F_IXCWUGWK
)
NV_R_YTOYHZGE_F_IXCWUGWK_TRUE = ValueMetadata(
    name='NV_R_YTOYHZGE_F_IXCWUGWK_TRUE',
    value=1,
    field=NV_R_YTOYHZGE_F_IXCWUGWK
)

NV_R_YTOYHZGE_F_VTCPIJDF = FieldMetadata(
    name='NV_R_YTOYHZGE_F_VTCPIJDF',
    msb=10,
    lsb=10,
    register=NV_R_YTOYHZGE
)

NV_R_YTOYHZGE_F_VTCPIJDF_V_ACRDDIMM = ValueMetadata(
    name='NV_R_YTOYHZGE_F_VTCPIJDF_V_ACRDDIMM',
    value=1,
    field=NV_R_YTOYHZGE_F_VTCPIJDF
)
NV_R_YTOYHZGE_F_VTCPIJDF_FALSE = ValueMetadata(
    name='NV_R_YTOYHZGE_F_VTCPIJDF_FALSE',
    value=0,
    field=NV_R_YTOYHZGE_F_VTCPIJDF
)
NV_R_YTOYHZGE_F_VTCPIJDF_TRUE = ValueMetadata(
    name='NV_R_YTOYHZGE_F_VTCPIJDF_TRUE',
    value=1,
    field=NV_R_YTOYHZGE_F_VTCPIJDF
)

NV_R_YTOYHZGE_F_JECCSHZH = FieldMetadata(
    name='NV_R_YTOYHZGE_F_JECCSHZH',
    msb=11,
    lsb=11,
    register=NV_R_YTOYHZGE
)

NV_R_YTOYHZGE_F_JECCSHZH_V_ACRDDIMM = ValueMetadata(
    name='NV_R_YTOYHZGE_F_JECCSHZH_V_ACRDDIMM',
    value=1,
    field=NV_R_YTOYHZGE_F_JECCSHZH
)
NV_R_YTOYHZGE_F_JECCSHZH_FALSE = ValueMetadata(
    name='NV_R_YTOYHZGE_F_JECCSHZH_FALSE',
    value=0,
    field=NV_R_YTOYHZGE_F_JECCSHZH
)
NV_R_YTOYHZGE_F_JECCSHZH_TRUE = ValueMetadata(
    name='NV_R_YTOYHZGE_F_JECCSHZH_TRUE',
    value=1,
    field=NV_R_YTOYHZGE_F_JECCSHZH
)

NV_R_YTOYHZGE_F_SLJISBLK = FieldMetadata(
    name='NV_R_YTOYHZGE_F_SLJISBLK',
    msb=0,
    lsb=0,
    register=NV_R_YTOYHZGE
)

NV_R_YTOYHZGE_F_SLJISBLK_V_ACRDDIMM = ValueMetadata(
    name='NV_R_YTOYHZGE_F_SLJISBLK_V_ACRDDIMM',
    value=1,
    field=NV_R_YTOYHZGE_F_SLJISBLK
)
NV_R_YTOYHZGE_F_SLJISBLK_FALSE = ValueMetadata(
    name='NV_R_YTOYHZGE_F_SLJISBLK_FALSE',
    value=0,
    field=NV_R_YTOYHZGE_F_SLJISBLK
)
NV_R_YTOYHZGE_F_SLJISBLK_TRUE = ValueMetadata(
    name='NV_R_YTOYHZGE_F_SLJISBLK_TRUE',
    value=1,
    field=NV_R_YTOYHZGE_F_SLJISBLK
)

NV_R_YTOYHZGE_F_BZCZRXGZ = FieldMetadata(
    name='NV_R_YTOYHZGE_F_BZCZRXGZ',
    msb=9,
    lsb=9,
    register=NV_R_YTOYHZGE
)

NV_R_YTOYHZGE_F_BZCZRXGZ_V_ACRDDIMM = ValueMetadata(
    name='NV_R_YTOYHZGE_F_BZCZRXGZ_V_ACRDDIMM',
    value=1,
    field=NV_R_YTOYHZGE_F_BZCZRXGZ
)
NV_R_YTOYHZGE_F_BZCZRXGZ_FALSE = ValueMetadata(
    name='NV_R_YTOYHZGE_F_BZCZRXGZ_FALSE',
    value=0,
    field=NV_R_YTOYHZGE_F_BZCZRXGZ
)
NV_R_YTOYHZGE_F_BZCZRXGZ_TRUE = ValueMetadata(
    name='NV_R_YTOYHZGE_F_BZCZRXGZ_TRUE',
    value=1,
    field=NV_R_YTOYHZGE_F_BZCZRXGZ
)

