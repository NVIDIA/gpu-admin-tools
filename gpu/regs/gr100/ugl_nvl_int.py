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
NV_R_PFWGRNHE = RegisterMetadata(
    name='NV_R_PFWGRNHE',
    address=0x424
)

NV_R_PFWGRNHE_F_GOOBMVPA = FieldMetadata(
    name='NV_R_PFWGRNHE_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_PFWGRNHE
)

NV_R_PFWGRNHE_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PFWGRNHE_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_PFWGRNHE_F_GOOBMVPA
)

NV_R_PFWGRNHE_F_JEHCVUZD = FieldMetadata(
    name='NV_R_PFWGRNHE_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_PFWGRNHE
)

NV_R_PFWGRNHE_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PFWGRNHE_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_PFWGRNHE_F_JEHCVUZD
)

NV_R_PFWGRNHE_F_FJYPATAE = FieldMetadata(
    name='NV_R_PFWGRNHE_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_PFWGRNHE
)

NV_R_PFWGRNHE_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PFWGRNHE_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_PFWGRNHE_F_FJYPATAE
)

NV_R_PFWGRNHE_F_ILMXLABV = FieldMetadata(
    name='NV_R_PFWGRNHE_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_PFWGRNHE
)

NV_R_PFWGRNHE_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PFWGRNHE_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_PFWGRNHE_F_ILMXLABV
)

NV_R_PFWGRNHE_F_NTHKCISA = FieldMetadata(
    name='NV_R_PFWGRNHE_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_PFWGRNHE
)

NV_R_PFWGRNHE_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PFWGRNHE_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_PFWGRNHE_F_NTHKCISA
)

NV_R_OYCEKENK = RegisterMetadata(
    name='NV_R_OYCEKENK',
    address=0x408
)

NV_R_OYCEKENK_F_GOOBMVPA = FieldMetadata(
    name='NV_R_OYCEKENK_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_OYCEKENK
)

NV_R_OYCEKENK_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_OYCEKENK_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_OYCEKENK_F_GOOBMVPA
)

NV_R_OYCEKENK_F_JEHCVUZD = FieldMetadata(
    name='NV_R_OYCEKENK_F_JEHCVUZD',
    msb=3,
    lsb=0,
    register=NV_R_OYCEKENK
)

NV_R_OYCEKENK_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_OYCEKENK_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_OYCEKENK_F_JEHCVUZD
)

NV_R_OYCEKENK_F_FJYPATAE = FieldMetadata(
    name='NV_R_OYCEKENK_F_FJYPATAE',
    msb=4,
    lsb=4,
    register=NV_R_OYCEKENK
)

NV_R_OYCEKENK_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_OYCEKENK_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_OYCEKENK_F_FJYPATAE
)

NV_R_OYCEKENK_F_ILMXLABV = FieldMetadata(
    name='NV_R_OYCEKENK_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_OYCEKENK
)

NV_R_OYCEKENK_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_OYCEKENK_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_OYCEKENK_F_ILMXLABV
)

NV_R_OYCEKENK_F_NTHKCISA = FieldMetadata(
    name='NV_R_OYCEKENK_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_OYCEKENK
)

NV_R_OYCEKENK_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_OYCEKENK_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_OYCEKENK_F_NTHKCISA
)

NV_R_MRLWCHQZ = RegisterMetadata(
    name='NV_R_MRLWCHQZ',
    address=0x404
)

NV_R_MRLWCHQZ_F_GOOBMVPA = FieldMetadata(
    name='NV_R_MRLWCHQZ_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_MRLWCHQZ
)

NV_R_MRLWCHQZ_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_MRLWCHQZ_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_MRLWCHQZ_F_GOOBMVPA
)

NV_R_MRLWCHQZ_F_JEHCVUZD = FieldMetadata(
    name='NV_R_MRLWCHQZ_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_MRLWCHQZ
)

NV_R_MRLWCHQZ_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_MRLWCHQZ_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_MRLWCHQZ_F_JEHCVUZD
)

NV_R_MRLWCHQZ_F_FJYPATAE = FieldMetadata(
    name='NV_R_MRLWCHQZ_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_MRLWCHQZ
)

NV_R_MRLWCHQZ_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_MRLWCHQZ_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_MRLWCHQZ_F_FJYPATAE
)

NV_R_MRLWCHQZ_F_ILMXLABV = FieldMetadata(
    name='NV_R_MRLWCHQZ_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_MRLWCHQZ
)

NV_R_MRLWCHQZ_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_MRLWCHQZ_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_MRLWCHQZ_F_ILMXLABV
)

NV_R_MRLWCHQZ_F_NTHKCISA = FieldMetadata(
    name='NV_R_MRLWCHQZ_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_MRLWCHQZ
)

NV_R_MRLWCHQZ_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_MRLWCHQZ_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_MRLWCHQZ_F_NTHKCISA
)

NV_R_WHIDZMOX = RegisterMetadata(
    name='NV_R_WHIDZMOX',
    address=0x41c
)

NV_R_WHIDZMOX_F_GOOBMVPA = FieldMetadata(
    name='NV_R_WHIDZMOX_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_WHIDZMOX
)

NV_R_WHIDZMOX_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_WHIDZMOX_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_WHIDZMOX_F_GOOBMVPA
)

NV_R_WHIDZMOX_F_JEHCVUZD = FieldMetadata(
    name='NV_R_WHIDZMOX_F_JEHCVUZD',
    msb=1,
    lsb=0,
    register=NV_R_WHIDZMOX
)

NV_R_WHIDZMOX_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_WHIDZMOX_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_WHIDZMOX_F_JEHCVUZD
)

NV_R_WHIDZMOX_F_FJYPATAE = FieldMetadata(
    name='NV_R_WHIDZMOX_F_FJYPATAE',
    msb=2,
    lsb=2,
    register=NV_R_WHIDZMOX
)

NV_R_WHIDZMOX_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_WHIDZMOX_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_WHIDZMOX_F_FJYPATAE
)

NV_R_WHIDZMOX_F_ILMXLABV = FieldMetadata(
    name='NV_R_WHIDZMOX_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_WHIDZMOX
)

NV_R_WHIDZMOX_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_WHIDZMOX_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_WHIDZMOX_F_ILMXLABV
)

NV_R_WHIDZMOX_F_NTHKCISA = FieldMetadata(
    name='NV_R_WHIDZMOX_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_WHIDZMOX
)

NV_R_WHIDZMOX_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_WHIDZMOX_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_WHIDZMOX_F_NTHKCISA
)

NV_R_BLOYMNMF = RegisterMetadata(
    name='NV_R_BLOYMNMF',
    address=0x414
)

NV_R_BLOYMNMF_F_GOOBMVPA = FieldMetadata(
    name='NV_R_BLOYMNMF_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_BLOYMNMF
)

NV_R_BLOYMNMF_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_BLOYMNMF_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_BLOYMNMF_F_GOOBMVPA
)

NV_R_BLOYMNMF_F_JEHCVUZD = FieldMetadata(
    name='NV_R_BLOYMNMF_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_BLOYMNMF
)

NV_R_BLOYMNMF_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_BLOYMNMF_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_BLOYMNMF_F_JEHCVUZD
)

NV_R_BLOYMNMF_F_FJYPATAE = FieldMetadata(
    name='NV_R_BLOYMNMF_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_BLOYMNMF
)

NV_R_BLOYMNMF_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_BLOYMNMF_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_BLOYMNMF_F_FJYPATAE
)

NV_R_BLOYMNMF_F_ILMXLABV = FieldMetadata(
    name='NV_R_BLOYMNMF_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_BLOYMNMF
)

NV_R_BLOYMNMF_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_BLOYMNMF_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_BLOYMNMF_F_ILMXLABV
)

NV_R_BLOYMNMF_F_NTHKCISA = FieldMetadata(
    name='NV_R_BLOYMNMF_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_BLOYMNMF
)

NV_R_BLOYMNMF_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_BLOYMNMF_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_BLOYMNMF_F_NTHKCISA
)

NV_R_LVEBIBMM = RegisterMetadata(
    name='NV_R_LVEBIBMM',
    address=0x418
)

NV_R_LVEBIBMM_F_GOOBMVPA = FieldMetadata(
    name='NV_R_LVEBIBMM_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_LVEBIBMM
)

NV_R_LVEBIBMM_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LVEBIBMM_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_LVEBIBMM_F_GOOBMVPA
)

NV_R_LVEBIBMM_F_JEHCVUZD = FieldMetadata(
    name='NV_R_LVEBIBMM_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_LVEBIBMM
)

NV_R_LVEBIBMM_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LVEBIBMM_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_LVEBIBMM_F_JEHCVUZD
)

NV_R_LVEBIBMM_F_FJYPATAE = FieldMetadata(
    name='NV_R_LVEBIBMM_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_LVEBIBMM
)

NV_R_LVEBIBMM_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LVEBIBMM_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_LVEBIBMM_F_FJYPATAE
)

NV_R_LVEBIBMM_F_ILMXLABV = FieldMetadata(
    name='NV_R_LVEBIBMM_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_LVEBIBMM
)

NV_R_LVEBIBMM_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LVEBIBMM_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_LVEBIBMM_F_ILMXLABV
)

NV_R_LVEBIBMM_F_NTHKCISA = FieldMetadata(
    name='NV_R_LVEBIBMM_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_LVEBIBMM
)

NV_R_LVEBIBMM_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LVEBIBMM_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_LVEBIBMM_F_NTHKCISA
)

NV_R_GTJNCCIC = RegisterMetadata(
    name='NV_R_GTJNCCIC',
    address=0x420
)

NV_R_GTJNCCIC_F_GOOBMVPA = FieldMetadata(
    name='NV_R_GTJNCCIC_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_GTJNCCIC
)

NV_R_GTJNCCIC_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_GTJNCCIC_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_GTJNCCIC_F_GOOBMVPA
)

NV_R_GTJNCCIC_F_JEHCVUZD = FieldMetadata(
    name='NV_R_GTJNCCIC_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_GTJNCCIC
)

NV_R_GTJNCCIC_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_GTJNCCIC_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_GTJNCCIC_F_JEHCVUZD
)

NV_R_GTJNCCIC_F_FJYPATAE = FieldMetadata(
    name='NV_R_GTJNCCIC_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_GTJNCCIC
)

NV_R_GTJNCCIC_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_GTJNCCIC_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_GTJNCCIC_F_FJYPATAE
)

NV_R_GTJNCCIC_F_ILMXLABV = FieldMetadata(
    name='NV_R_GTJNCCIC_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_GTJNCCIC
)

NV_R_GTJNCCIC_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_GTJNCCIC_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_GTJNCCIC_F_ILMXLABV
)

NV_R_GTJNCCIC_F_NTHKCISA = FieldMetadata(
    name='NV_R_GTJNCCIC_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_GTJNCCIC
)

NV_R_GTJNCCIC_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_GTJNCCIC_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_GTJNCCIC_F_NTHKCISA
)

NV_R_DEYRXJKC = RegisterMetadata(
    name='NV_R_DEYRXJKC',
    address=0x410
)

NV_R_DEYRXJKC_F_GOOBMVPA = FieldMetadata(
    name='NV_R_DEYRXJKC_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_DEYRXJKC
)

NV_R_DEYRXJKC_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DEYRXJKC_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_DEYRXJKC_F_GOOBMVPA
)

NV_R_DEYRXJKC_F_JEHCVUZD = FieldMetadata(
    name='NV_R_DEYRXJKC_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_DEYRXJKC
)

NV_R_DEYRXJKC_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DEYRXJKC_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_DEYRXJKC_F_JEHCVUZD
)

NV_R_DEYRXJKC_F_FJYPATAE = FieldMetadata(
    name='NV_R_DEYRXJKC_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_DEYRXJKC
)

NV_R_DEYRXJKC_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DEYRXJKC_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_DEYRXJKC_F_FJYPATAE
)

NV_R_DEYRXJKC_F_ILMXLABV = FieldMetadata(
    name='NV_R_DEYRXJKC_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_DEYRXJKC
)

NV_R_DEYRXJKC_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DEYRXJKC_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_DEYRXJKC_F_ILMXLABV
)

NV_R_DEYRXJKC_F_NTHKCISA = FieldMetadata(
    name='NV_R_DEYRXJKC_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_DEYRXJKC
)

NV_R_DEYRXJKC_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DEYRXJKC_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_DEYRXJKC_F_NTHKCISA
)

NV_R_LOZICWLQ = RegisterMetadata(
    name='NV_R_LOZICWLQ',
    address=0x428
)

NV_R_LOZICWLQ_F_GOOBMVPA = FieldMetadata(
    name='NV_R_LOZICWLQ_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_LOZICWLQ
)

NV_R_LOZICWLQ_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LOZICWLQ_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_LOZICWLQ_F_GOOBMVPA
)

NV_R_LOZICWLQ_F_JEHCVUZD = FieldMetadata(
    name='NV_R_LOZICWLQ_F_JEHCVUZD',
    msb=1,
    lsb=0,
    register=NV_R_LOZICWLQ
)

NV_R_LOZICWLQ_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LOZICWLQ_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_LOZICWLQ_F_JEHCVUZD
)

NV_R_LOZICWLQ_F_FJYPATAE = FieldMetadata(
    name='NV_R_LOZICWLQ_F_FJYPATAE',
    msb=2,
    lsb=2,
    register=NV_R_LOZICWLQ
)

NV_R_LOZICWLQ_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LOZICWLQ_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_LOZICWLQ_F_FJYPATAE
)

NV_R_LOZICWLQ_F_ILMXLABV = FieldMetadata(
    name='NV_R_LOZICWLQ_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_LOZICWLQ
)

NV_R_LOZICWLQ_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LOZICWLQ_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_LOZICWLQ_F_ILMXLABV
)

NV_R_LOZICWLQ_F_NTHKCISA = FieldMetadata(
    name='NV_R_LOZICWLQ_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_LOZICWLQ
)

NV_R_LOZICWLQ_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_LOZICWLQ_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_LOZICWLQ_F_NTHKCISA
)

NV_R_ATZNZTAI = RegisterMetadata(
    name='NV_R_ATZNZTAI',
    address=0x40c
)

NV_R_ATZNZTAI_F_GOOBMVPA = FieldMetadata(
    name='NV_R_ATZNZTAI_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_ATZNZTAI
)

NV_R_ATZNZTAI_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ATZNZTAI_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_ATZNZTAI_F_GOOBMVPA
)

NV_R_ATZNZTAI_F_JEHCVUZD = FieldMetadata(
    name='NV_R_ATZNZTAI_F_JEHCVUZD',
    msb=3,
    lsb=0,
    register=NV_R_ATZNZTAI
)

NV_R_ATZNZTAI_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ATZNZTAI_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_ATZNZTAI_F_JEHCVUZD
)

NV_R_ATZNZTAI_F_FJYPATAE = FieldMetadata(
    name='NV_R_ATZNZTAI_F_FJYPATAE',
    msb=4,
    lsb=4,
    register=NV_R_ATZNZTAI
)

NV_R_ATZNZTAI_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ATZNZTAI_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_ATZNZTAI_F_FJYPATAE
)

NV_R_ATZNZTAI_F_ILMXLABV = FieldMetadata(
    name='NV_R_ATZNZTAI_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_ATZNZTAI
)

NV_R_ATZNZTAI_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ATZNZTAI_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_ATZNZTAI_F_ILMXLABV
)

NV_R_ATZNZTAI_F_NTHKCISA = FieldMetadata(
    name='NV_R_ATZNZTAI_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_ATZNZTAI
)

NV_R_ATZNZTAI_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ATZNZTAI_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_ATZNZTAI_F_NTHKCISA
)

NV_R_HLVPTZGP = RegisterMetadata(
    name='NV_R_HLVPTZGP',
    address=0x400
)

NV_R_HLVPTZGP_F_GOOBMVPA = FieldMetadata(
    name='NV_R_HLVPTZGP_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_HLVPTZGP
)

NV_R_HLVPTZGP_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_HLVPTZGP_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_HLVPTZGP_F_GOOBMVPA
)

NV_R_HLVPTZGP_F_JEHCVUZD = FieldMetadata(
    name='NV_R_HLVPTZGP_F_JEHCVUZD',
    msb=4,
    lsb=0,
    register=NV_R_HLVPTZGP
)

NV_R_HLVPTZGP_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_HLVPTZGP_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_HLVPTZGP_F_JEHCVUZD
)

NV_R_HLVPTZGP_F_FJYPATAE = FieldMetadata(
    name='NV_R_HLVPTZGP_F_FJYPATAE',
    msb=5,
    lsb=5,
    register=NV_R_HLVPTZGP
)

NV_R_HLVPTZGP_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_HLVPTZGP_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_HLVPTZGP_F_FJYPATAE
)

NV_R_HLVPTZGP_F_ILMXLABV = FieldMetadata(
    name='NV_R_HLVPTZGP_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_HLVPTZGP
)

NV_R_HLVPTZGP_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_HLVPTZGP_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_HLVPTZGP_F_ILMXLABV
)

NV_R_HLVPTZGP_F_NTHKCISA = FieldMetadata(
    name='NV_R_HLVPTZGP_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_HLVPTZGP
)

NV_R_HLVPTZGP_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_HLVPTZGP_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_HLVPTZGP_F_NTHKCISA
)

NV_R_DLWCRZOR = RegisterMetadata(
    name='NV_R_DLWCRZOR',
    address=0x4a0
)

NV_R_DLWCRZOR_F_GOOBMVPA = FieldMetadata(
    name='NV_R_DLWCRZOR_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_DLWCRZOR
)

NV_R_DLWCRZOR_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DLWCRZOR_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_DLWCRZOR_F_GOOBMVPA
)

NV_R_DLWCRZOR_F_JEHCVUZD = FieldMetadata(
    name='NV_R_DLWCRZOR_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_DLWCRZOR
)

NV_R_DLWCRZOR_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DLWCRZOR_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_DLWCRZOR_F_JEHCVUZD
)

NV_R_DLWCRZOR_F_FJYPATAE = FieldMetadata(
    name='NV_R_DLWCRZOR_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_DLWCRZOR
)

NV_R_DLWCRZOR_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DLWCRZOR_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_DLWCRZOR_F_FJYPATAE
)

NV_R_DLWCRZOR_F_ILMXLABV = FieldMetadata(
    name='NV_R_DLWCRZOR_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_DLWCRZOR
)

NV_R_DLWCRZOR_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DLWCRZOR_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_DLWCRZOR_F_ILMXLABV
)

NV_R_DLWCRZOR_F_NTHKCISA = FieldMetadata(
    name='NV_R_DLWCRZOR_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_DLWCRZOR
)

NV_R_DLWCRZOR_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DLWCRZOR_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_DLWCRZOR_F_NTHKCISA
)

NV_R_SREEGKPL = RegisterMetadata(
    name='NV_R_SREEGKPL',
    address=0x588
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

NV_R_IXTSERAT = RegisterMetadata(
    name='NV_R_IXTSERAT',
    address=0x514
)

NV_R_IXTSERAT_F_GOOBMVPA = FieldMetadata(
    name='NV_R_IXTSERAT_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_IXTSERAT
)

NV_R_IXTSERAT_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_IXTSERAT_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_IXTSERAT_F_GOOBMVPA
)

NV_R_IXTSERAT_F_JEHCVUZD = FieldMetadata(
    name='NV_R_IXTSERAT_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_IXTSERAT
)

NV_R_IXTSERAT_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_IXTSERAT_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_IXTSERAT_F_JEHCVUZD
)

NV_R_IXTSERAT_F_FJYPATAE = FieldMetadata(
    name='NV_R_IXTSERAT_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_IXTSERAT
)

NV_R_IXTSERAT_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_IXTSERAT_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_IXTSERAT_F_FJYPATAE
)

NV_R_IXTSERAT_F_ILMXLABV = FieldMetadata(
    name='NV_R_IXTSERAT_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_IXTSERAT
)

NV_R_IXTSERAT_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_IXTSERAT_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_IXTSERAT_F_ILMXLABV
)

NV_R_IXTSERAT_F_NTHKCISA = FieldMetadata(
    name='NV_R_IXTSERAT_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_IXTSERAT
)

NV_R_IXTSERAT_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_IXTSERAT_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_IXTSERAT_F_NTHKCISA
)

NV_R_PNKUYVOT = RegisterMetadata(
    name='NV_R_PNKUYVOT',
    address=0x510
)

NV_R_PNKUYVOT_F_GOOBMVPA = FieldMetadata(
    name='NV_R_PNKUYVOT_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_PNKUYVOT
)

NV_R_PNKUYVOT_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PNKUYVOT_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_PNKUYVOT_F_GOOBMVPA
)

NV_R_PNKUYVOT_F_JEHCVUZD = FieldMetadata(
    name='NV_R_PNKUYVOT_F_JEHCVUZD',
    msb=4,
    lsb=0,
    register=NV_R_PNKUYVOT
)

NV_R_PNKUYVOT_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PNKUYVOT_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_PNKUYVOT_F_JEHCVUZD
)

NV_R_PNKUYVOT_F_FJYPATAE = FieldMetadata(
    name='NV_R_PNKUYVOT_F_FJYPATAE',
    msb=5,
    lsb=5,
    register=NV_R_PNKUYVOT
)

NV_R_PNKUYVOT_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PNKUYVOT_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_PNKUYVOT_F_FJYPATAE
)

NV_R_PNKUYVOT_F_ILMXLABV = FieldMetadata(
    name='NV_R_PNKUYVOT_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_PNKUYVOT
)

NV_R_PNKUYVOT_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PNKUYVOT_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_PNKUYVOT_F_ILMXLABV
)

NV_R_PNKUYVOT_F_NTHKCISA = FieldMetadata(
    name='NV_R_PNKUYVOT_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_PNKUYVOT
)

NV_R_PNKUYVOT_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PNKUYVOT_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_PNKUYVOT_F_NTHKCISA
)

NV_R_MOBILSEL = RegisterMetadata(
    name='NV_R_MOBILSEL',
    address=0x500
)

NV_R_MOBILSEL_F_GOOBMVPA = FieldMetadata(
    name='NV_R_MOBILSEL_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_MOBILSEL
)

NV_R_MOBILSEL_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_MOBILSEL_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_MOBILSEL_F_GOOBMVPA
)

NV_R_MOBILSEL_F_JEHCVUZD = FieldMetadata(
    name='NV_R_MOBILSEL_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_MOBILSEL
)

NV_R_MOBILSEL_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_MOBILSEL_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_MOBILSEL_F_JEHCVUZD
)

NV_R_MOBILSEL_F_FJYPATAE = FieldMetadata(
    name='NV_R_MOBILSEL_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_MOBILSEL
)

NV_R_MOBILSEL_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_MOBILSEL_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_MOBILSEL_F_FJYPATAE
)

NV_R_MOBILSEL_F_ILMXLABV = FieldMetadata(
    name='NV_R_MOBILSEL_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_MOBILSEL
)

NV_R_MOBILSEL_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_MOBILSEL_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_MOBILSEL_F_ILMXLABV
)

NV_R_MOBILSEL_F_NTHKCISA = FieldMetadata(
    name='NV_R_MOBILSEL_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_MOBILSEL
)

NV_R_MOBILSEL_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_MOBILSEL_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_MOBILSEL_F_NTHKCISA
)

NV_R_ZOTAPDVL = RegisterMetadata(
    name='NV_R_ZOTAPDVL',
    address=0xc24
)

NV_R_ZOTAPDVL_F_GOOBMVPA = FieldMetadata(
    name='NV_R_ZOTAPDVL_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_ZOTAPDVL
)

NV_R_ZOTAPDVL_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZOTAPDVL_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_ZOTAPDVL_F_GOOBMVPA
)

NV_R_ZOTAPDVL_F_JEHCVUZD = FieldMetadata(
    name='NV_R_ZOTAPDVL_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_ZOTAPDVL
)

NV_R_ZOTAPDVL_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZOTAPDVL_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_ZOTAPDVL_F_JEHCVUZD
)

NV_R_ZOTAPDVL_F_FJYPATAE = FieldMetadata(
    name='NV_R_ZOTAPDVL_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_ZOTAPDVL
)

NV_R_ZOTAPDVL_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZOTAPDVL_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_ZOTAPDVL_F_FJYPATAE
)

NV_R_ZOTAPDVL_F_ILMXLABV = FieldMetadata(
    name='NV_R_ZOTAPDVL_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_ZOTAPDVL
)

NV_R_ZOTAPDVL_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZOTAPDVL_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_ZOTAPDVL_F_ILMXLABV
)

NV_R_ZOTAPDVL_F_NTHKCISA = FieldMetadata(
    name='NV_R_ZOTAPDVL_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_ZOTAPDVL
)

NV_R_ZOTAPDVL_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZOTAPDVL_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_ZOTAPDVL_F_NTHKCISA
)

NV_R_DHOKIDNU = RegisterMetadata(
    name='NV_R_DHOKIDNU',
    address=0xc08
)

NV_R_DHOKIDNU_F_GOOBMVPA = FieldMetadata(
    name='NV_R_DHOKIDNU_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_DHOKIDNU
)

NV_R_DHOKIDNU_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DHOKIDNU_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_DHOKIDNU_F_GOOBMVPA
)

NV_R_DHOKIDNU_F_JEHCVUZD = FieldMetadata(
    name='NV_R_DHOKIDNU_F_JEHCVUZD',
    msb=3,
    lsb=0,
    register=NV_R_DHOKIDNU
)

NV_R_DHOKIDNU_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DHOKIDNU_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_DHOKIDNU_F_JEHCVUZD
)

NV_R_DHOKIDNU_F_FJYPATAE = FieldMetadata(
    name='NV_R_DHOKIDNU_F_FJYPATAE',
    msb=4,
    lsb=4,
    register=NV_R_DHOKIDNU
)

NV_R_DHOKIDNU_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DHOKIDNU_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_DHOKIDNU_F_FJYPATAE
)

NV_R_DHOKIDNU_F_ILMXLABV = FieldMetadata(
    name='NV_R_DHOKIDNU_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_DHOKIDNU
)

NV_R_DHOKIDNU_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DHOKIDNU_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_DHOKIDNU_F_ILMXLABV
)

NV_R_DHOKIDNU_F_NTHKCISA = FieldMetadata(
    name='NV_R_DHOKIDNU_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_DHOKIDNU
)

NV_R_DHOKIDNU_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DHOKIDNU_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_DHOKIDNU_F_NTHKCISA
)

NV_R_YFAVFKTV = RegisterMetadata(
    name='NV_R_YFAVFKTV',
    address=0xc04
)

NV_R_YFAVFKTV_F_GOOBMVPA = FieldMetadata(
    name='NV_R_YFAVFKTV_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_YFAVFKTV
)

NV_R_YFAVFKTV_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YFAVFKTV_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_YFAVFKTV_F_GOOBMVPA
)

NV_R_YFAVFKTV_F_JEHCVUZD = FieldMetadata(
    name='NV_R_YFAVFKTV_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_YFAVFKTV
)

NV_R_YFAVFKTV_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YFAVFKTV_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_YFAVFKTV_F_JEHCVUZD
)

NV_R_YFAVFKTV_F_FJYPATAE = FieldMetadata(
    name='NV_R_YFAVFKTV_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_YFAVFKTV
)

NV_R_YFAVFKTV_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YFAVFKTV_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_YFAVFKTV_F_FJYPATAE
)

NV_R_YFAVFKTV_F_ILMXLABV = FieldMetadata(
    name='NV_R_YFAVFKTV_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_YFAVFKTV
)

NV_R_YFAVFKTV_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YFAVFKTV_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_YFAVFKTV_F_ILMXLABV
)

NV_R_YFAVFKTV_F_NTHKCISA = FieldMetadata(
    name='NV_R_YFAVFKTV_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_YFAVFKTV
)

NV_R_YFAVFKTV_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YFAVFKTV_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_YFAVFKTV_F_NTHKCISA
)

NV_R_FPNMDHMA = RegisterMetadata(
    name='NV_R_FPNMDHMA',
    address=0xc1c
)

NV_R_FPNMDHMA_F_GOOBMVPA = FieldMetadata(
    name='NV_R_FPNMDHMA_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_FPNMDHMA
)

NV_R_FPNMDHMA_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FPNMDHMA_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_FPNMDHMA_F_GOOBMVPA
)

NV_R_FPNMDHMA_F_JEHCVUZD = FieldMetadata(
    name='NV_R_FPNMDHMA_F_JEHCVUZD',
    msb=1,
    lsb=0,
    register=NV_R_FPNMDHMA
)

NV_R_FPNMDHMA_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FPNMDHMA_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_FPNMDHMA_F_JEHCVUZD
)

NV_R_FPNMDHMA_F_FJYPATAE = FieldMetadata(
    name='NV_R_FPNMDHMA_F_FJYPATAE',
    msb=2,
    lsb=2,
    register=NV_R_FPNMDHMA
)

NV_R_FPNMDHMA_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FPNMDHMA_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_FPNMDHMA_F_FJYPATAE
)

NV_R_FPNMDHMA_F_ILMXLABV = FieldMetadata(
    name='NV_R_FPNMDHMA_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_FPNMDHMA
)

NV_R_FPNMDHMA_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FPNMDHMA_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_FPNMDHMA_F_ILMXLABV
)

NV_R_FPNMDHMA_F_NTHKCISA = FieldMetadata(
    name='NV_R_FPNMDHMA_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_FPNMDHMA
)

NV_R_FPNMDHMA_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_FPNMDHMA_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_FPNMDHMA_F_NTHKCISA
)

NV_R_EZNGUDGH = RegisterMetadata(
    name='NV_R_EZNGUDGH',
    address=0xc14
)

NV_R_EZNGUDGH_F_GOOBMVPA = FieldMetadata(
    name='NV_R_EZNGUDGH_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_EZNGUDGH
)

NV_R_EZNGUDGH_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_EZNGUDGH_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_EZNGUDGH_F_GOOBMVPA
)

NV_R_EZNGUDGH_F_JEHCVUZD = FieldMetadata(
    name='NV_R_EZNGUDGH_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_EZNGUDGH
)

NV_R_EZNGUDGH_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_EZNGUDGH_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_EZNGUDGH_F_JEHCVUZD
)

NV_R_EZNGUDGH_F_FJYPATAE = FieldMetadata(
    name='NV_R_EZNGUDGH_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_EZNGUDGH
)

NV_R_EZNGUDGH_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_EZNGUDGH_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_EZNGUDGH_F_FJYPATAE
)

NV_R_EZNGUDGH_F_ILMXLABV = FieldMetadata(
    name='NV_R_EZNGUDGH_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_EZNGUDGH
)

NV_R_EZNGUDGH_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_EZNGUDGH_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_EZNGUDGH_F_ILMXLABV
)

NV_R_EZNGUDGH_F_NTHKCISA = FieldMetadata(
    name='NV_R_EZNGUDGH_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_EZNGUDGH
)

NV_R_EZNGUDGH_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_EZNGUDGH_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_EZNGUDGH_F_NTHKCISA
)

NV_R_QWOJUVUO = RegisterMetadata(
    name='NV_R_QWOJUVUO',
    address=0xc18
)

NV_R_QWOJUVUO_F_GOOBMVPA = FieldMetadata(
    name='NV_R_QWOJUVUO_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_QWOJUVUO
)

NV_R_QWOJUVUO_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QWOJUVUO_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_QWOJUVUO_F_GOOBMVPA
)

NV_R_QWOJUVUO_F_JEHCVUZD = FieldMetadata(
    name='NV_R_QWOJUVUO_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_QWOJUVUO
)

NV_R_QWOJUVUO_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QWOJUVUO_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_QWOJUVUO_F_JEHCVUZD
)

NV_R_QWOJUVUO_F_FJYPATAE = FieldMetadata(
    name='NV_R_QWOJUVUO_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_QWOJUVUO
)

NV_R_QWOJUVUO_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QWOJUVUO_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_QWOJUVUO_F_FJYPATAE
)

NV_R_QWOJUVUO_F_ILMXLABV = FieldMetadata(
    name='NV_R_QWOJUVUO_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_QWOJUVUO
)

NV_R_QWOJUVUO_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QWOJUVUO_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_QWOJUVUO_F_ILMXLABV
)

NV_R_QWOJUVUO_F_NTHKCISA = FieldMetadata(
    name='NV_R_QWOJUVUO_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_QWOJUVUO
)

NV_R_QWOJUVUO_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_QWOJUVUO_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_QWOJUVUO_F_NTHKCISA
)

NV_R_PBJCZRLS = RegisterMetadata(
    name='NV_R_PBJCZRLS',
    address=0xc20
)

NV_R_PBJCZRLS_F_GOOBMVPA = FieldMetadata(
    name='NV_R_PBJCZRLS_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_PBJCZRLS
)

NV_R_PBJCZRLS_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PBJCZRLS_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_PBJCZRLS_F_GOOBMVPA
)

NV_R_PBJCZRLS_F_JEHCVUZD = FieldMetadata(
    name='NV_R_PBJCZRLS_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_PBJCZRLS
)

NV_R_PBJCZRLS_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PBJCZRLS_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_PBJCZRLS_F_JEHCVUZD
)

NV_R_PBJCZRLS_F_FJYPATAE = FieldMetadata(
    name='NV_R_PBJCZRLS_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_PBJCZRLS
)

NV_R_PBJCZRLS_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PBJCZRLS_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_PBJCZRLS_F_FJYPATAE
)

NV_R_PBJCZRLS_F_ILMXLABV = FieldMetadata(
    name='NV_R_PBJCZRLS_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_PBJCZRLS
)

NV_R_PBJCZRLS_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PBJCZRLS_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_PBJCZRLS_F_ILMXLABV
)

NV_R_PBJCZRLS_F_NTHKCISA = FieldMetadata(
    name='NV_R_PBJCZRLS_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_PBJCZRLS
)

NV_R_PBJCZRLS_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PBJCZRLS_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_PBJCZRLS_F_NTHKCISA
)

NV_R_WHVLTPKB = RegisterMetadata(
    name='NV_R_WHVLTPKB',
    address=0xc10
)

NV_R_WHVLTPKB_F_GOOBMVPA = FieldMetadata(
    name='NV_R_WHVLTPKB_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_WHVLTPKB
)

NV_R_WHVLTPKB_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_WHVLTPKB_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_WHVLTPKB_F_GOOBMVPA
)

NV_R_WHVLTPKB_F_JEHCVUZD = FieldMetadata(
    name='NV_R_WHVLTPKB_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_WHVLTPKB
)

NV_R_WHVLTPKB_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_WHVLTPKB_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_WHVLTPKB_F_JEHCVUZD
)

NV_R_WHVLTPKB_F_FJYPATAE = FieldMetadata(
    name='NV_R_WHVLTPKB_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_WHVLTPKB
)

NV_R_WHVLTPKB_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_WHVLTPKB_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_WHVLTPKB_F_FJYPATAE
)

NV_R_WHVLTPKB_F_ILMXLABV = FieldMetadata(
    name='NV_R_WHVLTPKB_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_WHVLTPKB
)

NV_R_WHVLTPKB_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_WHVLTPKB_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_WHVLTPKB_F_ILMXLABV
)

NV_R_WHVLTPKB_F_NTHKCISA = FieldMetadata(
    name='NV_R_WHVLTPKB_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_WHVLTPKB
)

NV_R_WHVLTPKB_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_WHVLTPKB_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_WHVLTPKB_F_NTHKCISA
)

NV_R_EAGBLJRI = RegisterMetadata(
    name='NV_R_EAGBLJRI',
    address=0xc28
)

NV_R_EAGBLJRI_F_GOOBMVPA = FieldMetadata(
    name='NV_R_EAGBLJRI_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_EAGBLJRI
)

NV_R_EAGBLJRI_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_EAGBLJRI_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_EAGBLJRI_F_GOOBMVPA
)

NV_R_EAGBLJRI_F_JEHCVUZD = FieldMetadata(
    name='NV_R_EAGBLJRI_F_JEHCVUZD',
    msb=1,
    lsb=0,
    register=NV_R_EAGBLJRI
)

NV_R_EAGBLJRI_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_EAGBLJRI_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_EAGBLJRI_F_JEHCVUZD
)

NV_R_EAGBLJRI_F_FJYPATAE = FieldMetadata(
    name='NV_R_EAGBLJRI_F_FJYPATAE',
    msb=2,
    lsb=2,
    register=NV_R_EAGBLJRI
)

NV_R_EAGBLJRI_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_EAGBLJRI_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_EAGBLJRI_F_FJYPATAE
)

NV_R_EAGBLJRI_F_ILMXLABV = FieldMetadata(
    name='NV_R_EAGBLJRI_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_EAGBLJRI
)

NV_R_EAGBLJRI_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_EAGBLJRI_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_EAGBLJRI_F_ILMXLABV
)

NV_R_EAGBLJRI_F_NTHKCISA = FieldMetadata(
    name='NV_R_EAGBLJRI_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_EAGBLJRI
)

NV_R_EAGBLJRI_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_EAGBLJRI_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_EAGBLJRI_F_NTHKCISA
)

NV_R_TDZLZJFP = RegisterMetadata(
    name='NV_R_TDZLZJFP',
    address=0xc0c
)

NV_R_TDZLZJFP_F_GOOBMVPA = FieldMetadata(
    name='NV_R_TDZLZJFP_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_TDZLZJFP
)

NV_R_TDZLZJFP_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TDZLZJFP_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_TDZLZJFP_F_GOOBMVPA
)

NV_R_TDZLZJFP_F_JEHCVUZD = FieldMetadata(
    name='NV_R_TDZLZJFP_F_JEHCVUZD',
    msb=3,
    lsb=0,
    register=NV_R_TDZLZJFP
)

NV_R_TDZLZJFP_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TDZLZJFP_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_TDZLZJFP_F_JEHCVUZD
)

NV_R_TDZLZJFP_F_FJYPATAE = FieldMetadata(
    name='NV_R_TDZLZJFP_F_FJYPATAE',
    msb=4,
    lsb=4,
    register=NV_R_TDZLZJFP
)

NV_R_TDZLZJFP_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TDZLZJFP_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_TDZLZJFP_F_FJYPATAE
)

NV_R_TDZLZJFP_F_ILMXLABV = FieldMetadata(
    name='NV_R_TDZLZJFP_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_TDZLZJFP
)

NV_R_TDZLZJFP_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TDZLZJFP_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_TDZLZJFP_F_ILMXLABV
)

NV_R_TDZLZJFP_F_NTHKCISA = FieldMetadata(
    name='NV_R_TDZLZJFP_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_TDZLZJFP
)

NV_R_TDZLZJFP_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TDZLZJFP_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_TDZLZJFP_F_NTHKCISA
)

NV_R_YBQCGFTQ = RegisterMetadata(
    name='NV_R_YBQCGFTQ',
    address=0xc00
)

NV_R_YBQCGFTQ_F_GOOBMVPA = FieldMetadata(
    name='NV_R_YBQCGFTQ_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_YBQCGFTQ
)

NV_R_YBQCGFTQ_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YBQCGFTQ_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_YBQCGFTQ_F_GOOBMVPA
)

NV_R_YBQCGFTQ_F_JEHCVUZD = FieldMetadata(
    name='NV_R_YBQCGFTQ_F_JEHCVUZD',
    msb=4,
    lsb=0,
    register=NV_R_YBQCGFTQ
)

NV_R_YBQCGFTQ_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YBQCGFTQ_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_YBQCGFTQ_F_JEHCVUZD
)

NV_R_YBQCGFTQ_F_FJYPATAE = FieldMetadata(
    name='NV_R_YBQCGFTQ_F_FJYPATAE',
    msb=5,
    lsb=5,
    register=NV_R_YBQCGFTQ
)

NV_R_YBQCGFTQ_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YBQCGFTQ_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_YBQCGFTQ_F_FJYPATAE
)

NV_R_YBQCGFTQ_F_ILMXLABV = FieldMetadata(
    name='NV_R_YBQCGFTQ_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_YBQCGFTQ
)

NV_R_YBQCGFTQ_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YBQCGFTQ_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_YBQCGFTQ_F_ILMXLABV
)

NV_R_YBQCGFTQ_F_NTHKCISA = FieldMetadata(
    name='NV_R_YBQCGFTQ_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_YBQCGFTQ
)

NV_R_YBQCGFTQ_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_YBQCGFTQ_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_YBQCGFTQ_F_NTHKCISA
)

NV_R_DFGLEXLL = RegisterMetadata(
    name='NV_R_DFGLEXLL',
    address=0xca0
)

NV_R_DFGLEXLL_F_GOOBMVPA = FieldMetadata(
    name='NV_R_DFGLEXLL_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_DFGLEXLL
)

NV_R_DFGLEXLL_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DFGLEXLL_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_DFGLEXLL_F_GOOBMVPA
)

NV_R_DFGLEXLL_F_JEHCVUZD = FieldMetadata(
    name='NV_R_DFGLEXLL_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_DFGLEXLL
)

NV_R_DFGLEXLL_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DFGLEXLL_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_DFGLEXLL_F_JEHCVUZD
)

NV_R_DFGLEXLL_F_FJYPATAE = FieldMetadata(
    name='NV_R_DFGLEXLL_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_DFGLEXLL
)

NV_R_DFGLEXLL_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DFGLEXLL_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_DFGLEXLL_F_FJYPATAE
)

NV_R_DFGLEXLL_F_ILMXLABV = FieldMetadata(
    name='NV_R_DFGLEXLL_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_DFGLEXLL
)

NV_R_DFGLEXLL_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DFGLEXLL_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_DFGLEXLL_F_ILMXLABV
)

NV_R_DFGLEXLL_F_NTHKCISA = FieldMetadata(
    name='NV_R_DFGLEXLL_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_DFGLEXLL
)

NV_R_DFGLEXLL_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DFGLEXLL_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_DFGLEXLL_F_NTHKCISA
)

NV_R_AGGDUMHS = RegisterMetadata(
    name='NV_R_AGGDUMHS',
    address=0xd88
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

NV_R_XPDFOXCD = RegisterMetadata(
    name='NV_R_XPDFOXCD',
    address=0xd14
)

NV_R_XPDFOXCD_F_GOOBMVPA = FieldMetadata(
    name='NV_R_XPDFOXCD_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_XPDFOXCD
)

NV_R_XPDFOXCD_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XPDFOXCD_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_XPDFOXCD_F_GOOBMVPA
)

NV_R_XPDFOXCD_F_JEHCVUZD = FieldMetadata(
    name='NV_R_XPDFOXCD_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_XPDFOXCD
)

NV_R_XPDFOXCD_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XPDFOXCD_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_XPDFOXCD_F_JEHCVUZD
)

NV_R_XPDFOXCD_F_FJYPATAE = FieldMetadata(
    name='NV_R_XPDFOXCD_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_XPDFOXCD
)

NV_R_XPDFOXCD_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XPDFOXCD_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_XPDFOXCD_F_FJYPATAE
)

NV_R_XPDFOXCD_F_ILMXLABV = FieldMetadata(
    name='NV_R_XPDFOXCD_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_XPDFOXCD
)

NV_R_XPDFOXCD_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XPDFOXCD_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_XPDFOXCD_F_ILMXLABV
)

NV_R_XPDFOXCD_F_NTHKCISA = FieldMetadata(
    name='NV_R_XPDFOXCD_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_XPDFOXCD
)

NV_R_XPDFOXCD_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_XPDFOXCD_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_XPDFOXCD_F_NTHKCISA
)

NV_R_PVKEGDZN = RegisterMetadata(
    name='NV_R_PVKEGDZN',
    address=0xd10
)

NV_R_PVKEGDZN_F_GOOBMVPA = FieldMetadata(
    name='NV_R_PVKEGDZN_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_PVKEGDZN
)

NV_R_PVKEGDZN_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PVKEGDZN_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_PVKEGDZN_F_GOOBMVPA
)

NV_R_PVKEGDZN_F_JEHCVUZD = FieldMetadata(
    name='NV_R_PVKEGDZN_F_JEHCVUZD',
    msb=4,
    lsb=0,
    register=NV_R_PVKEGDZN
)

NV_R_PVKEGDZN_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PVKEGDZN_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_PVKEGDZN_F_JEHCVUZD
)

NV_R_PVKEGDZN_F_FJYPATAE = FieldMetadata(
    name='NV_R_PVKEGDZN_F_FJYPATAE',
    msb=5,
    lsb=5,
    register=NV_R_PVKEGDZN
)

NV_R_PVKEGDZN_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PVKEGDZN_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_PVKEGDZN_F_FJYPATAE
)

NV_R_PVKEGDZN_F_ILMXLABV = FieldMetadata(
    name='NV_R_PVKEGDZN_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_PVKEGDZN
)

NV_R_PVKEGDZN_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PVKEGDZN_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_PVKEGDZN_F_ILMXLABV
)

NV_R_PVKEGDZN_F_NTHKCISA = FieldMetadata(
    name='NV_R_PVKEGDZN_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_PVKEGDZN
)

NV_R_PVKEGDZN_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_PVKEGDZN_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_PVKEGDZN_F_NTHKCISA
)

NV_R_TXESHYHE = RegisterMetadata(
    name='NV_R_TXESHYHE',
    address=0xd00
)

NV_R_TXESHYHE_F_GOOBMVPA = FieldMetadata(
    name='NV_R_TXESHYHE_F_GOOBMVPA',
    msb=30,
    lsb=30,
    register=NV_R_TXESHYHE
)

NV_R_TXESHYHE_F_GOOBMVPA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TXESHYHE_F_GOOBMVPA_V_ZRRJKDVX',
    value=0,
    field=NV_R_TXESHYHE_F_GOOBMVPA
)

NV_R_TXESHYHE_F_JEHCVUZD = FieldMetadata(
    name='NV_R_TXESHYHE_F_JEHCVUZD',
    msb=2,
    lsb=0,
    register=NV_R_TXESHYHE
)

NV_R_TXESHYHE_F_JEHCVUZD_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TXESHYHE_F_JEHCVUZD_V_ZRRJKDVX',
    value=0,
    field=NV_R_TXESHYHE_F_JEHCVUZD
)

NV_R_TXESHYHE_F_FJYPATAE = FieldMetadata(
    name='NV_R_TXESHYHE_F_FJYPATAE',
    msb=3,
    lsb=3,
    register=NV_R_TXESHYHE
)

NV_R_TXESHYHE_F_FJYPATAE_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TXESHYHE_F_FJYPATAE_V_ZRRJKDVX',
    value=0,
    field=NV_R_TXESHYHE_F_FJYPATAE
)

NV_R_TXESHYHE_F_ILMXLABV = FieldMetadata(
    name='NV_R_TXESHYHE_F_ILMXLABV',
    msb=29,
    lsb=29,
    register=NV_R_TXESHYHE
)

NV_R_TXESHYHE_F_ILMXLABV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TXESHYHE_F_ILMXLABV_V_ZRRJKDVX',
    value=0,
    field=NV_R_TXESHYHE_F_ILMXLABV
)

NV_R_TXESHYHE_F_NTHKCISA = FieldMetadata(
    name='NV_R_TXESHYHE_F_NTHKCISA',
    msb=31,
    lsb=31,
    register=NV_R_TXESHYHE
)

NV_R_TXESHYHE_F_NTHKCISA_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_TXESHYHE_F_NTHKCISA_V_ZRRJKDVX',
    value=0,
    field=NV_R_TXESHYHE_F_NTHKCISA
)

