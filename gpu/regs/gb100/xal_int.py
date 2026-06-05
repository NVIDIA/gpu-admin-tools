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
from gpu.regs.gh100.xal_int import (
    NV_R_RTXEVWIZ,
    NV_R_RTXEVWIZ_F_FMDCUIAD,
    NV_R_RTXEVWIZ_F_FMDCUIAD_V_GQHYEKHS,
    NV_R_LZSOUPVS,
    NV_R_LZSOUPVS_F_OJYQBJXU,
    NV_R_WLBKDJZT,
    NV_R_WLBKDJZT_F_CTBCFRUB,
    NV_R_WLBKDJZT_F_CTBCFRUB_V_GQHYEKHS,
    NV_R_WLBKDJZT_F_QVYAARWP,
    NV_R_WLBKDJZT_F_QVYAARWP_V_GQHYEKHS,
    NV_R_ICINMFYU,
    NV_R_ICINMFYU_F_DBYSWVHB,
    NV_R_ICINMFYU_F_DBYSWVHB_V_DOGBFDTH,
    NV_R_ICINMFYU_F_DBYSWVHB_V_PIUJQBKV,
    NV_R_ICINMFYU_F_OJXVVOEC,
    NV_R_ICINMFYU_F_OJXVVOEC_V_DOGBFDTH,
    NV_R_ICINMFYU_F_OJXVVOEC_V_PIUJQBKV,
    NV_R_ICINMFYU_F_VEOHPHHB,
    NV_R_ICINMFYU_F_VEOHPHHB_V_DOGBFDTH,
    NV_R_ICINMFYU_F_VEOHPHHB_V_PIUJQBKV,
    NV_R_ICINMFYU_F_UURIHQGO,
    NV_R_ICINMFYU_F_UURIHQGO_V_DOGBFDTH,
    NV_R_ICINMFYU_F_UURIHQGO_V_PIUJQBKV,
    NV_R_ICINMFYU_F_WQYWMRVP,
    NV_R_ICINMFYU_F_WQYWMRVP_V_DOGBFDTH,
    NV_R_ICINMFYU_F_WQYWMRVP_V_PIUJQBKV,
    NV_R_ICINMFYU_F_MLEOLJIS,
    NV_R_ICINMFYU_F_MLEOLJIS_V_DOGBFDTH,
    NV_R_ICINMFYU_F_MLEOLJIS_V_PIUJQBKV,
    NV_R_VLTEGMKD,
    NV_R_VLTEGMKD_F_CTBCFRUB,
    NV_R_VLTEGMKD_F_CTBCFRUB_V_GQHYEKHS,
    NV_R_VLTEGMKD_F_QVYAARWP,
    NV_R_VLTEGMKD_F_QVYAARWP_V_GQHYEKHS,
    NV_R_NYYRIQPJ,
    NV_R_NYYRIQPJ_F_JEHSTEUR,
    NV_R_NYYRIQPJ_F_JEHSTEUR_V_GQHYEKHS,
    NV_R_TOZOZAJX,
    NV_R_TOZOZAJX_F_JEHSTEUR,
    NV_R_TOZOZAJX_F_JEHSTEUR_V_GQHYEKHS,
)

# Register definitions
NV_R_SXUEHZYB = RegisterMetadata(
    name='NV_R_SXUEHZYB',
    address=0x10f330,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_SXUEHZYB_F_FJYCNPHP = FieldMetadata(
    name='NV_R_SXUEHZYB_F_FJYCNPHP',
    msb=31,
    lsb=2,
    register=NV_R_SXUEHZYB
)

NV_R_SXUEHZYB_F_FJYCNPHP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_SXUEHZYB_F_FJYCNPHP_V_GQHYEKHS',
    value=0,
    field=NV_R_SXUEHZYB_F_FJYCNPHP
)

NV_R_AGXHZVDW = RegisterMetadata(
    name='NV_R_AGXHZVDW',
    address=0x10f338,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_AGXHZVDW_F_CSXZWYIM = FieldMetadata(
    name='NV_R_AGXHZVDW_F_CSXZWYIM',
    msb=13,
    lsb=8,
    register=NV_R_AGXHZVDW
)

NV_R_AGXHZVDW_F_CSXZWYIM_V_GQHYEKHS = ValueMetadata(
    name='NV_R_AGXHZVDW_F_CSXZWYIM_V_GQHYEKHS',
    value=0,
    field=NV_R_AGXHZVDW_F_CSXZWYIM
)

NV_R_AGXHZVDW_F_OIGWNDTI = FieldMetadata(
    name='NV_R_AGXHZVDW_F_OIGWNDTI',
    msb=31,
    lsb=31,
    register=NV_R_AGXHZVDW
)

NV_R_AGXHZVDW_F_OIGWNDTI_V_GQHYEKHS = ValueMetadata(
    name='NV_R_AGXHZVDW_F_OIGWNDTI_V_GQHYEKHS',
    value=0,
    field=NV_R_AGXHZVDW_F_OIGWNDTI
)

NV_R_AGXHZVDW_F_JGDYKRTC = FieldMetadata(
    name='NV_R_AGXHZVDW_F_JGDYKRTC',
    msb=1,
    lsb=0,
    register=NV_R_AGXHZVDW
)

NV_R_AGXHZVDW_F_JGDYKRTC_V_YIYDETAJ = ValueMetadata(
    name='NV_R_AGXHZVDW_F_JGDYKRTC_V_YIYDETAJ',
    value=2,
    field=NV_R_AGXHZVDW_F_JGDYKRTC
)
NV_R_AGXHZVDW_F_JGDYKRTC_V_PFJFQFSM = ValueMetadata(
    name='NV_R_AGXHZVDW_F_JGDYKRTC_V_PFJFQFSM',
    value=1,
    field=NV_R_AGXHZVDW_F_JGDYKRTC
)
NV_R_AGXHZVDW_F_JGDYKRTC_V_CRFJBTNJ = ValueMetadata(
    name='NV_R_AGXHZVDW_F_JGDYKRTC_V_CRFJBTNJ',
    value=0,
    field=NV_R_AGXHZVDW_F_JGDYKRTC
)

NV_R_LPJIOAEO = RegisterMetadata(
    name='NV_R_LPJIOAEO',
    address=0x10f320,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_LPJIOAEO_F_FJYCNPHP = FieldMetadata(
    name='NV_R_LPJIOAEO_F_FJYCNPHP',
    msb=31,
    lsb=2,
    register=NV_R_LPJIOAEO
)

NV_R_LPJIOAEO_F_FJYCNPHP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LPJIOAEO_F_FJYCNPHP_V_GQHYEKHS',
    value=0,
    field=NV_R_LPJIOAEO_F_FJYCNPHP
)

NV_R_CTSRXRJA = RegisterMetadata(
    name='NV_R_CTSRXRJA',
    address=0x10f328,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_CTSRXRJA_F_XZCIOFTX = FieldMetadata(
    name='NV_R_CTSRXRJA_F_XZCIOFTX',
    msb=2,
    lsb=1,
    register=NV_R_CTSRXRJA
)

NV_R_CTSRXRJA_F_XZCIOFTX_V_WDAOZBUG = ValueMetadata(
    name='NV_R_CTSRXRJA_F_XZCIOFTX_V_WDAOZBUG',
    value=0,
    field=NV_R_CTSRXRJA_F_XZCIOFTX
)
NV_R_CTSRXRJA_F_XZCIOFTX_V_OLQLFEZG = ValueMetadata(
    name='NV_R_CTSRXRJA_F_XZCIOFTX_V_OLQLFEZG',
    value=1,
    field=NV_R_CTSRXRJA_F_XZCIOFTX
)
NV_R_CTSRXRJA_F_XZCIOFTX_V_JTQCCLDC = ValueMetadata(
    name='NV_R_CTSRXRJA_F_XZCIOFTX_V_JTQCCLDC',
    value=2,
    field=NV_R_CTSRXRJA_F_XZCIOFTX
)
NV_R_CTSRXRJA_F_XZCIOFTX_V_SMHGBZYZ = ValueMetadata(
    name='NV_R_CTSRXRJA_F_XZCIOFTX_V_SMHGBZYZ',
    value=3,
    field=NV_R_CTSRXRJA_F_XZCIOFTX
)

NV_R_CTSRXRJA_F_CSXZWYIM = FieldMetadata(
    name='NV_R_CTSRXRJA_F_CSXZWYIM',
    msb=13,
    lsb=8,
    register=NV_R_CTSRXRJA
)

NV_R_CTSRXRJA_F_CSXZWYIM_V_GQHYEKHS = ValueMetadata(
    name='NV_R_CTSRXRJA_F_CSXZWYIM_V_GQHYEKHS',
    value=0,
    field=NV_R_CTSRXRJA_F_CSXZWYIM
)

NV_R_CTSRXRJA_F_OIGWNDTI = FieldMetadata(
    name='NV_R_CTSRXRJA_F_OIGWNDTI',
    msb=31,
    lsb=31,
    register=NV_R_CTSRXRJA
)

NV_R_CTSRXRJA_F_OIGWNDTI_V_GQHYEKHS = ValueMetadata(
    name='NV_R_CTSRXRJA_F_OIGWNDTI_V_GQHYEKHS',
    value=0,
    field=NV_R_CTSRXRJA_F_OIGWNDTI
)

NV_R_CTSRXRJA_F_JGDYKRTC = FieldMetadata(
    name='NV_R_CTSRXRJA_F_JGDYKRTC',
    msb=0,
    lsb=0,
    register=NV_R_CTSRXRJA
)

NV_R_CTSRXRJA_F_JGDYKRTC_V_PFJFQFSM = ValueMetadata(
    name='NV_R_CTSRXRJA_F_JGDYKRTC_V_PFJFQFSM',
    value=1,
    field=NV_R_CTSRXRJA_F_JGDYKRTC
)
NV_R_CTSRXRJA_F_JGDYKRTC_V_CRFJBTNJ = ValueMetadata(
    name='NV_R_CTSRXRJA_F_JGDYKRTC_V_CRFJBTNJ',
    value=0,
    field=NV_R_CTSRXRJA_F_JGDYKRTC
)

