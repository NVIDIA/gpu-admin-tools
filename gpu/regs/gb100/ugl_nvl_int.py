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
NV_R_FXPBSRRH = RegisterMetadata(
    name='NV_R_FXPBSRRH',
    address=0x833c
)

NV_R_FXPBSRRH_F_GOOBMVPA = FieldMetadata(
    name='NV_R_FXPBSRRH_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_FXPBSRRH
)

NV_R_FXPBSRRH_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FXPBSRRH_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_FXPBSRRH_F_GOOBMVPA
)

NV_R_FXPBSRRH_F_JEHCVUZD = FieldMetadata(
    name='NV_R_FXPBSRRH_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_FXPBSRRH
)

NV_R_FXPBSRRH_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FXPBSRRH_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_FXPBSRRH_F_JEHCVUZD
)

NV_R_FXPBSRRH_F_FJYPATAE = FieldMetadata(
    name='NV_R_FXPBSRRH_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_FXPBSRRH
)

NV_R_FXPBSRRH_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FXPBSRRH_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_FXPBSRRH_F_FJYPATAE
)

NV_R_FXPBSRRH_F_ILMXLABV = FieldMetadata(
    name='NV_R_FXPBSRRH_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_FXPBSRRH
)

NV_R_FXPBSRRH_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FXPBSRRH_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_FXPBSRRH_F_ILMXLABV
)

NV_R_FXPBSRRH_F_NTHKCISA = FieldMetadata(
    name='NV_R_FXPBSRRH_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_FXPBSRRH
)

NV_R_FXPBSRRH_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FXPBSRRH_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_FXPBSRRH_F_NTHKCISA
)

NV_R_XDILWGSA = RegisterMetadata(
    name='NV_R_XDILWGSA',
    address=0x8344
)

NV_R_XDILWGSA_F_GOOBMVPA = FieldMetadata(
    name='NV_R_XDILWGSA_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_XDILWGSA
)

NV_R_XDILWGSA_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XDILWGSA_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_XDILWGSA_F_GOOBMVPA
)

NV_R_XDILWGSA_F_JEHCVUZD = FieldMetadata(
    name='NV_R_XDILWGSA_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_XDILWGSA
)

NV_R_XDILWGSA_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XDILWGSA_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_XDILWGSA_F_JEHCVUZD
)

NV_R_XDILWGSA_F_FJYPATAE = FieldMetadata(
    name='NV_R_XDILWGSA_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_XDILWGSA
)

NV_R_XDILWGSA_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XDILWGSA_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_XDILWGSA_F_FJYPATAE
)

NV_R_XDILWGSA_F_ILMXLABV = FieldMetadata(
    name='NV_R_XDILWGSA_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_XDILWGSA
)

NV_R_XDILWGSA_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XDILWGSA_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_XDILWGSA_F_ILMXLABV
)

NV_R_XDILWGSA_F_NTHKCISA = FieldMetadata(
    name='NV_R_XDILWGSA_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_XDILWGSA
)

NV_R_XDILWGSA_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XDILWGSA_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_XDILWGSA_F_NTHKCISA
)

NV_R_VQYDUCRS = RegisterMetadata(
    name='NV_R_VQYDUCRS',
    address=0x8340
)

NV_R_VQYDUCRS_F_GOOBMVPA = FieldMetadata(
    name='NV_R_VQYDUCRS_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_VQYDUCRS
)

NV_R_VQYDUCRS_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VQYDUCRS_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_VQYDUCRS_F_GOOBMVPA
)

NV_R_VQYDUCRS_F_JEHCVUZD = FieldMetadata(
    name='NV_R_VQYDUCRS_F_JEHCVUZD',
    msb=3,
    lsb=0,
    register=NV_R_VQYDUCRS
)

NV_R_VQYDUCRS_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VQYDUCRS_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_VQYDUCRS_F_JEHCVUZD
)

NV_R_VQYDUCRS_F_FJYPATAE = FieldMetadata(
    name='NV_R_VQYDUCRS_F_FJYPATAE',
    msb=4,
    lsb=4,
    register=NV_R_VQYDUCRS
)

NV_R_VQYDUCRS_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VQYDUCRS_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_VQYDUCRS_F_FJYPATAE
)

NV_R_VQYDUCRS_F_ILMXLABV = FieldMetadata(
    name='NV_R_VQYDUCRS_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_VQYDUCRS
)

NV_R_VQYDUCRS_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VQYDUCRS_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_VQYDUCRS_F_ILMXLABV
)

NV_R_VQYDUCRS_F_NTHKCISA = FieldMetadata(
    name='NV_R_VQYDUCRS_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_VQYDUCRS
)

NV_R_VQYDUCRS_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VQYDUCRS_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_VQYDUCRS_F_NTHKCISA
)

NV_R_ZYDEIRUT = RegisterMetadata(
    name='NV_R_ZYDEIRUT',
    address=0x8334
)

NV_R_ZYDEIRUT_F_GOOBMVPA = FieldMetadata(
    name='NV_R_ZYDEIRUT_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_ZYDEIRUT
)

NV_R_ZYDEIRUT_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZYDEIRUT_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_ZYDEIRUT_F_GOOBMVPA
)

NV_R_ZYDEIRUT_F_JEHCVUZD = FieldMetadata(
    name='NV_R_ZYDEIRUT_F_JEHCVUZD',
    msb=3,
    lsb=0,
    register=NV_R_ZYDEIRUT
)

NV_R_ZYDEIRUT_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZYDEIRUT_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_ZYDEIRUT_F_JEHCVUZD
)

NV_R_ZYDEIRUT_F_FJYPATAE = FieldMetadata(
    name='NV_R_ZYDEIRUT_F_FJYPATAE',
    msb=4,
    lsb=4,
    register=NV_R_ZYDEIRUT
)

NV_R_ZYDEIRUT_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZYDEIRUT_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_ZYDEIRUT_F_FJYPATAE
)

NV_R_ZYDEIRUT_F_ILMXLABV = FieldMetadata(
    name='NV_R_ZYDEIRUT_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_ZYDEIRUT
)

NV_R_ZYDEIRUT_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZYDEIRUT_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_ZYDEIRUT_F_ILMXLABV
)

NV_R_ZYDEIRUT_F_NTHKCISA = FieldMetadata(
    name='NV_R_ZYDEIRUT_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_ZYDEIRUT
)

NV_R_ZYDEIRUT_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZYDEIRUT_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_ZYDEIRUT_F_NTHKCISA
)

NV_R_SREEGKPL = RegisterMetadata(
    name='NV_R_SREEGKPL',
    address=0x83b8
)

NV_R_SREEGKPL_F_GOOBMVPA = FieldMetadata(
    name='NV_R_SREEGKPL_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_SREEGKPL
)

NV_R_SREEGKPL_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_SREEGKPL_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_SREEGKPL_F_GOOBMVPA
)

NV_R_SREEGKPL_F_JEHCVUZD = FieldMetadata(
    name='NV_R_SREEGKPL_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_SREEGKPL
)

NV_R_SREEGKPL_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_SREEGKPL_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_SREEGKPL_F_JEHCVUZD
)

NV_R_SREEGKPL_F_FJYPATAE = FieldMetadata(
    name='NV_R_SREEGKPL_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_SREEGKPL
)

NV_R_SREEGKPL_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_SREEGKPL_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_SREEGKPL_F_FJYPATAE
)

NV_R_SREEGKPL_F_ILMXLABV = FieldMetadata(
    name='NV_R_SREEGKPL_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_SREEGKPL
)

NV_R_SREEGKPL_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_SREEGKPL_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_SREEGKPL_F_ILMXLABV
)

NV_R_SREEGKPL_F_NTHKCISA = FieldMetadata(
    name='NV_R_SREEGKPL_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_SREEGKPL
)

NV_R_SREEGKPL_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_SREEGKPL_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_SREEGKPL_F_NTHKCISA
)

NV_R_QFAJHGIB = RegisterMetadata(
    name='NV_R_QFAJHGIB',
    address=0x83c4
)

NV_R_QFAJHGIB_F_GOOBMVPA = FieldMetadata(
    name='NV_R_QFAJHGIB_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_QFAJHGIB
)

NV_R_QFAJHGIB_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QFAJHGIB_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_QFAJHGIB_F_GOOBMVPA
)

NV_R_QFAJHGIB_F_JEHCVUZD = FieldMetadata(
    name='NV_R_QFAJHGIB_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_QFAJHGIB
)

NV_R_QFAJHGIB_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QFAJHGIB_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_QFAJHGIB_F_JEHCVUZD
)

NV_R_QFAJHGIB_F_FJYPATAE = FieldMetadata(
    name='NV_R_QFAJHGIB_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_QFAJHGIB
)

NV_R_QFAJHGIB_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QFAJHGIB_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_QFAJHGIB_F_FJYPATAE
)

NV_R_QFAJHGIB_F_ILMXLABV = FieldMetadata(
    name='NV_R_QFAJHGIB_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_QFAJHGIB
)

NV_R_QFAJHGIB_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QFAJHGIB_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_QFAJHGIB_F_ILMXLABV
)

NV_R_QFAJHGIB_F_NTHKCISA = FieldMetadata(
    name='NV_R_QFAJHGIB_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_QFAJHGIB
)

NV_R_QFAJHGIB_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QFAJHGIB_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_QFAJHGIB_F_NTHKCISA
)

NV_R_DTSCTTFU = RegisterMetadata(
    name='NV_R_DTSCTTFU',
    address=0x83c0
)

NV_R_DTSCTTFU_F_GOOBMVPA = FieldMetadata(
    name='NV_R_DTSCTTFU_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_DTSCTTFU
)

NV_R_DTSCTTFU_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DTSCTTFU_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_DTSCTTFU_F_GOOBMVPA
)

NV_R_DTSCTTFU_F_JEHCVUZD = FieldMetadata(
    name='NV_R_DTSCTTFU_F_JEHCVUZD',
    msb=3,
    lsb=0,
    register=NV_R_DTSCTTFU
)

NV_R_DTSCTTFU_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DTSCTTFU_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_DTSCTTFU_F_JEHCVUZD
)

NV_R_DTSCTTFU_F_FJYPATAE = FieldMetadata(
    name='NV_R_DTSCTTFU_F_FJYPATAE',
    msb=4,
    lsb=4,
    register=NV_R_DTSCTTFU
)

NV_R_DTSCTTFU_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DTSCTTFU_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_DTSCTTFU_F_FJYPATAE
)

NV_R_DTSCTTFU_F_ILMXLABV = FieldMetadata(
    name='NV_R_DTSCTTFU_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_DTSCTTFU
)

NV_R_DTSCTTFU_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DTSCTTFU_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_DTSCTTFU_F_ILMXLABV
)

NV_R_DTSCTTFU_F_NTHKCISA = FieldMetadata(
    name='NV_R_DTSCTTFU_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_DTSCTTFU
)

NV_R_DTSCTTFU_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DTSCTTFU_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_DTSCTTFU_F_NTHKCISA
)

NV_R_NATURWUI = RegisterMetadata(
    name='NV_R_NATURWUI',
    address=0x8b3c
)

NV_R_NATURWUI_F_GOOBMVPA = FieldMetadata(
    name='NV_R_NATURWUI_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_NATURWUI
)

NV_R_NATURWUI_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_NATURWUI_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_NATURWUI_F_GOOBMVPA
)

NV_R_NATURWUI_F_JEHCVUZD = FieldMetadata(
    name='NV_R_NATURWUI_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_NATURWUI
)

NV_R_NATURWUI_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_NATURWUI_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_NATURWUI_F_JEHCVUZD
)

NV_R_NATURWUI_F_FJYPATAE = FieldMetadata(
    name='NV_R_NATURWUI_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_NATURWUI
)

NV_R_NATURWUI_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_NATURWUI_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_NATURWUI_F_FJYPATAE
)

NV_R_NATURWUI_F_ILMXLABV = FieldMetadata(
    name='NV_R_NATURWUI_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_NATURWUI
)

NV_R_NATURWUI_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_NATURWUI_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_NATURWUI_F_ILMXLABV
)

NV_R_NATURWUI_F_NTHKCISA = FieldMetadata(
    name='NV_R_NATURWUI_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_NATURWUI
)

NV_R_NATURWUI_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_NATURWUI_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_NATURWUI_F_NTHKCISA
)

NV_R_YBRYTIAR = RegisterMetadata(
    name='NV_R_YBRYTIAR',
    address=0x8b44
)

NV_R_YBRYTIAR_F_GOOBMVPA = FieldMetadata(
    name='NV_R_YBRYTIAR_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_YBRYTIAR
)

NV_R_YBRYTIAR_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YBRYTIAR_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_YBRYTIAR_F_GOOBMVPA
)

NV_R_YBRYTIAR_F_JEHCVUZD = FieldMetadata(
    name='NV_R_YBRYTIAR_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_YBRYTIAR
)

NV_R_YBRYTIAR_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YBRYTIAR_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_YBRYTIAR_F_JEHCVUZD
)

NV_R_YBRYTIAR_F_FJYPATAE = FieldMetadata(
    name='NV_R_YBRYTIAR_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_YBRYTIAR
)

NV_R_YBRYTIAR_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YBRYTIAR_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_YBRYTIAR_F_FJYPATAE
)

NV_R_YBRYTIAR_F_ILMXLABV = FieldMetadata(
    name='NV_R_YBRYTIAR_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_YBRYTIAR
)

NV_R_YBRYTIAR_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YBRYTIAR_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_YBRYTIAR_F_ILMXLABV
)

NV_R_YBRYTIAR_F_NTHKCISA = FieldMetadata(
    name='NV_R_YBRYTIAR_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_YBRYTIAR
)

NV_R_YBRYTIAR_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YBRYTIAR_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_YBRYTIAR_F_NTHKCISA
)

NV_R_QVALAOZO = RegisterMetadata(
    name='NV_R_QVALAOZO',
    address=0x8b40
)

NV_R_QVALAOZO_F_GOOBMVPA = FieldMetadata(
    name='NV_R_QVALAOZO_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_QVALAOZO
)

NV_R_QVALAOZO_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QVALAOZO_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_QVALAOZO_F_GOOBMVPA
)

NV_R_QVALAOZO_F_JEHCVUZD = FieldMetadata(
    name='NV_R_QVALAOZO_F_JEHCVUZD',
    msb=3,
    lsb=0,
    register=NV_R_QVALAOZO
)

NV_R_QVALAOZO_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QVALAOZO_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_QVALAOZO_F_JEHCVUZD
)

NV_R_QVALAOZO_F_FJYPATAE = FieldMetadata(
    name='NV_R_QVALAOZO_F_FJYPATAE',
    msb=4,
    lsb=4,
    register=NV_R_QVALAOZO
)

NV_R_QVALAOZO_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QVALAOZO_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_QVALAOZO_F_FJYPATAE
)

NV_R_QVALAOZO_F_ILMXLABV = FieldMetadata(
    name='NV_R_QVALAOZO_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_QVALAOZO
)

NV_R_QVALAOZO_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QVALAOZO_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_QVALAOZO_F_ILMXLABV
)

NV_R_QVALAOZO_F_NTHKCISA = FieldMetadata(
    name='NV_R_QVALAOZO_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_QVALAOZO
)

NV_R_QVALAOZO_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QVALAOZO_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_QVALAOZO_F_NTHKCISA
)

NV_R_VVTMRHUS = RegisterMetadata(
    name='NV_R_VVTMRHUS',
    address=0x8b34
)

NV_R_VVTMRHUS_F_GOOBMVPA = FieldMetadata(
    name='NV_R_VVTMRHUS_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_VVTMRHUS
)

NV_R_VVTMRHUS_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VVTMRHUS_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_VVTMRHUS_F_GOOBMVPA
)

NV_R_VVTMRHUS_F_JEHCVUZD = FieldMetadata(
    name='NV_R_VVTMRHUS_F_JEHCVUZD',
    msb=3,
    lsb=0,
    register=NV_R_VVTMRHUS
)

NV_R_VVTMRHUS_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VVTMRHUS_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_VVTMRHUS_F_JEHCVUZD
)

NV_R_VVTMRHUS_F_FJYPATAE = FieldMetadata(
    name='NV_R_VVTMRHUS_F_FJYPATAE',
    msb=4,
    lsb=4,
    register=NV_R_VVTMRHUS
)

NV_R_VVTMRHUS_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VVTMRHUS_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_VVTMRHUS_F_FJYPATAE
)

NV_R_VVTMRHUS_F_ILMXLABV = FieldMetadata(
    name='NV_R_VVTMRHUS_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_VVTMRHUS
)

NV_R_VVTMRHUS_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VVTMRHUS_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_VVTMRHUS_F_ILMXLABV
)

NV_R_VVTMRHUS_F_NTHKCISA = FieldMetadata(
    name='NV_R_VVTMRHUS_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_VVTMRHUS
)

NV_R_VVTMRHUS_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_VVTMRHUS_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_VVTMRHUS_F_NTHKCISA
)

NV_R_AGGDUMHS = RegisterMetadata(
    name='NV_R_AGGDUMHS',
    address=0x8bb8
)

NV_R_AGGDUMHS_F_GOOBMVPA = FieldMetadata(
    name='NV_R_AGGDUMHS_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_AGGDUMHS
)

NV_R_AGGDUMHS_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_AGGDUMHS_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_AGGDUMHS_F_GOOBMVPA
)

NV_R_AGGDUMHS_F_JEHCVUZD = FieldMetadata(
    name='NV_R_AGGDUMHS_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_AGGDUMHS
)

NV_R_AGGDUMHS_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_AGGDUMHS_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_AGGDUMHS_F_JEHCVUZD
)

NV_R_AGGDUMHS_F_FJYPATAE = FieldMetadata(
    name='NV_R_AGGDUMHS_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_AGGDUMHS
)

NV_R_AGGDUMHS_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_AGGDUMHS_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_AGGDUMHS_F_FJYPATAE
)

NV_R_AGGDUMHS_F_ILMXLABV = FieldMetadata(
    name='NV_R_AGGDUMHS_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_AGGDUMHS
)

NV_R_AGGDUMHS_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_AGGDUMHS_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_AGGDUMHS_F_ILMXLABV
)

NV_R_AGGDUMHS_F_NTHKCISA = FieldMetadata(
    name='NV_R_AGGDUMHS_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_AGGDUMHS
)

NV_R_AGGDUMHS_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_AGGDUMHS_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_AGGDUMHS_F_NTHKCISA
)

NV_R_BAQAOGZZ = RegisterMetadata(
    name='NV_R_BAQAOGZZ',
    address=0x8bc4
)

NV_R_BAQAOGZZ_F_GOOBMVPA = FieldMetadata(
    name='NV_R_BAQAOGZZ_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_BAQAOGZZ
)

NV_R_BAQAOGZZ_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_BAQAOGZZ_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_BAQAOGZZ_F_GOOBMVPA
)

NV_R_BAQAOGZZ_F_JEHCVUZD = FieldMetadata(
    name='NV_R_BAQAOGZZ_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_BAQAOGZZ
)

NV_R_BAQAOGZZ_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_BAQAOGZZ_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_BAQAOGZZ_F_JEHCVUZD
)

NV_R_BAQAOGZZ_F_FJYPATAE = FieldMetadata(
    name='NV_R_BAQAOGZZ_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_BAQAOGZZ
)

NV_R_BAQAOGZZ_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_BAQAOGZZ_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_BAQAOGZZ_F_FJYPATAE
)

NV_R_BAQAOGZZ_F_ILMXLABV = FieldMetadata(
    name='NV_R_BAQAOGZZ_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_BAQAOGZZ
)

NV_R_BAQAOGZZ_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_BAQAOGZZ_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_BAQAOGZZ_F_ILMXLABV
)

NV_R_BAQAOGZZ_F_NTHKCISA = FieldMetadata(
    name='NV_R_BAQAOGZZ_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_BAQAOGZZ
)

NV_R_BAQAOGZZ_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_BAQAOGZZ_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_BAQAOGZZ_F_NTHKCISA
)

NV_R_PHTWIASQ = RegisterMetadata(
    name='NV_R_PHTWIASQ',
    address=0x8bc0
)

NV_R_PHTWIASQ_F_GOOBMVPA = FieldMetadata(
    name='NV_R_PHTWIASQ_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_PHTWIASQ
)

NV_R_PHTWIASQ_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PHTWIASQ_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_PHTWIASQ_F_GOOBMVPA
)

NV_R_PHTWIASQ_F_JEHCVUZD = FieldMetadata(
    name='NV_R_PHTWIASQ_F_JEHCVUZD',
    msb=3,
    lsb=0,
    register=NV_R_PHTWIASQ
)

NV_R_PHTWIASQ_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PHTWIASQ_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_PHTWIASQ_F_JEHCVUZD
)

NV_R_PHTWIASQ_F_FJYPATAE = FieldMetadata(
    name='NV_R_PHTWIASQ_F_FJYPATAE',
    msb=4,
    lsb=4,
    register=NV_R_PHTWIASQ
)

NV_R_PHTWIASQ_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PHTWIASQ_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_PHTWIASQ_F_FJYPATAE
)

NV_R_PHTWIASQ_F_ILMXLABV = FieldMetadata(
    name='NV_R_PHTWIASQ_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_PHTWIASQ
)

NV_R_PHTWIASQ_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PHTWIASQ_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_PHTWIASQ_F_ILMXLABV
)

NV_R_PHTWIASQ_F_NTHKCISA = FieldMetadata(
    name='NV_R_PHTWIASQ_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_PHTWIASQ
)

NV_R_PHTWIASQ_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PHTWIASQ_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_PHTWIASQ_F_NTHKCISA
)

