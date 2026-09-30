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


# Registers identical to gb100
from gpu.regs.gb100.xal_int import (
    NV_R_SXUEHZYB,
    NV_R_SXUEHZYB_F_FJYCNPHP,
    NV_R_SXUEHZYB_F_FJYCNPHP_V_GQHYEKHS,
    NV_R_AGXHZVDW,
    NV_R_AGXHZVDW_F_CSXZWYIM,
    NV_R_AGXHZVDW_F_CSXZWYIM_V_GQHYEKHS,
    NV_R_AGXHZVDW_F_OIGWNDTI,
    NV_R_AGXHZVDW_F_OIGWNDTI_V_GQHYEKHS,
    NV_R_AGXHZVDW_F_JGDYKRTC,
    NV_R_AGXHZVDW_F_JGDYKRTC_V_YIYDETAJ,
    NV_R_AGXHZVDW_F_JGDYKRTC_V_PFJFQFSM,
    NV_R_AGXHZVDW_F_JGDYKRTC_V_CRFJBTNJ,
    NV_R_LPJIOAEO,
    NV_R_LPJIOAEO_F_FJYCNPHP,
    NV_R_LPJIOAEO_F_FJYCNPHP_V_GQHYEKHS,
    NV_R_CTSRXRJA,
    NV_R_CTSRXRJA_F_XZCIOFTX,
    NV_R_CTSRXRJA_F_XZCIOFTX_V_WDAOZBUG,
    NV_R_CTSRXRJA_F_XZCIOFTX_V_OLQLFEZG,
    NV_R_CTSRXRJA_F_XZCIOFTX_V_JTQCCLDC,
    NV_R_CTSRXRJA_F_XZCIOFTX_V_SMHGBZYZ,
    NV_R_CTSRXRJA_F_CSXZWYIM,
    NV_R_CTSRXRJA_F_CSXZWYIM_V_GQHYEKHS,
    NV_R_CTSRXRJA_F_OIGWNDTI,
    NV_R_CTSRXRJA_F_OIGWNDTI_V_GQHYEKHS,
    NV_R_CTSRXRJA_F_JGDYKRTC,
    NV_R_CTSRXRJA_F_JGDYKRTC_V_PFJFQFSM,
    NV_R_CTSRXRJA_F_JGDYKRTC_V_CRFJBTNJ,
)

# Registers identical to gh100
from gpu.regs.gh100.xal_int import (
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
NV_R_XLHFQAGN = RegisterMetadata(
    name='NV_R_XLHFQAGN',
    address=0x10f824,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_XLHFQAGN_F_FMDCUIAD = FieldMetadata(
    name='NV_R_XLHFQAGN_F_FMDCUIAD',
    msb=27,
    lsb=0,
    register=NV_R_XLHFQAGN
)

NV_R_XLHFQAGN_F_FMDCUIAD_V_GQHYEKHS = ValueMetadata(
    name='NV_R_XLHFQAGN_F_FMDCUIAD_V_GQHYEKHS',
    value=0,
    field=NV_R_XLHFQAGN_F_FMDCUIAD
)

NV_R_XLHFQAGN_WRITE = FieldMetadata(
    name='NV_R_XLHFQAGN_WRITE',
    msb=31,
    lsb=31,
    register=NV_R_XLHFQAGN
)

NV_R_XLHFQAGN_WRITE_FALSE = ValueMetadata(
    name='NV_R_XLHFQAGN_WRITE_FALSE',
    value=0,
    field=NV_R_XLHFQAGN_WRITE
)
NV_R_XLHFQAGN_WRITE_TRUE = ValueMetadata(
    name='NV_R_XLHFQAGN_WRITE_TRUE',
    value=1,
    field=NV_R_XLHFQAGN_WRITE
)

NV_R_RTXEVWIZ = RegisterMetadata(
    name='NV_R_RTXEVWIZ',
    address=0x10f390,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_RTXEVWIZ_F_FMDCUIAD = FieldMetadata(
    name='NV_R_RTXEVWIZ_F_FMDCUIAD',
    msb=27,
    lsb=0,
    register=NV_R_RTXEVWIZ
)

NV_R_RTXEVWIZ_F_FMDCUIAD_V_GQHYEKHS = ValueMetadata(
    name='NV_R_RTXEVWIZ_F_FMDCUIAD_V_GQHYEKHS',
    value=0,
    field=NV_R_RTXEVWIZ_F_FMDCUIAD
)

NV_R_UGXZRGTH = RegisterMetadata(
    name='NV_R_UGXZRGTH',
    address=0x10fc18
)

NV_R_UGXZRGTH_F_OJYQBJXU = FieldMetadata(
    name='NV_R_UGXZRGTH_F_OJYQBJXU',
    msb=5,
    lsb=0,
    register=NV_R_UGXZRGTH
)

NV_R_HNLRXBGY = RegisterMetadata(
    name='NV_R_HNLRXBGY',
    address=0x10fc10,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_HNLRXBGY_F_CTBCFRUB = FieldMetadata(
    name='NV_R_HNLRXBGY_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_HNLRXBGY
)

NV_R_HNLRXBGY_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_HNLRXBGY_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_HNLRXBGY_F_CTBCFRUB
)

NV_R_HNLRXBGY_F_QVYAARWP = FieldMetadata(
    name='NV_R_HNLRXBGY_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_HNLRXBGY
)

NV_R_HNLRXBGY_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_HNLRXBGY_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_HNLRXBGY_F_QVYAARWP
)

NV_R_ONXHWMFI = RegisterMetadata(
    name='NV_R_ONXHWMFI',
    address=0x10fc0c,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_ONXHWMFI_F_DBYSWVHB = FieldMetadata(
    name='NV_R_ONXHWMFI_F_DBYSWVHB',
    msb=0,
    lsb=0,
    register=NV_R_ONXHWMFI
)

NV_R_ONXHWMFI_F_DBYSWVHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ONXHWMFI_F_DBYSWVHB_V_DOGBFDTH',
    value=0,
    field=NV_R_ONXHWMFI_F_DBYSWVHB
)
NV_R_ONXHWMFI_F_DBYSWVHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ONXHWMFI_F_DBYSWVHB_V_PIUJQBKV',
    value=1,
    field=NV_R_ONXHWMFI_F_DBYSWVHB
)

NV_R_ONXHWMFI_F_OJXVVOEC = FieldMetadata(
    name='NV_R_ONXHWMFI_F_OJXVVOEC',
    msb=8,
    lsb=8,
    register=NV_R_ONXHWMFI
)

NV_R_ONXHWMFI_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ONXHWMFI_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_ONXHWMFI_F_OJXVVOEC
)
NV_R_ONXHWMFI_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ONXHWMFI_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_ONXHWMFI_F_OJXVVOEC
)

NV_R_ONXHWMFI_F_VEOHPHHB = FieldMetadata(
    name='NV_R_ONXHWMFI_F_VEOHPHHB',
    msb=9,
    lsb=9,
    register=NV_R_ONXHWMFI
)

NV_R_ONXHWMFI_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ONXHWMFI_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_ONXHWMFI_F_VEOHPHHB
)
NV_R_ONXHWMFI_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ONXHWMFI_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_ONXHWMFI_F_VEOHPHHB
)

NV_R_ONXHWMFI_F_UURIHQGO = FieldMetadata(
    name='NV_R_ONXHWMFI_F_UURIHQGO',
    msb=1,
    lsb=1,
    register=NV_R_ONXHWMFI
)

NV_R_ONXHWMFI_F_UURIHQGO_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ONXHWMFI_F_UURIHQGO_V_DOGBFDTH',
    value=0,
    field=NV_R_ONXHWMFI_F_UURIHQGO
)
NV_R_ONXHWMFI_F_UURIHQGO_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ONXHWMFI_F_UURIHQGO_V_PIUJQBKV',
    value=1,
    field=NV_R_ONXHWMFI_F_UURIHQGO
)

NV_R_ONXHWMFI_F_WQYWMRVP = FieldMetadata(
    name='NV_R_ONXHWMFI_F_WQYWMRVP',
    msb=16,
    lsb=16,
    register=NV_R_ONXHWMFI
)

NV_R_ONXHWMFI_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ONXHWMFI_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_ONXHWMFI_F_WQYWMRVP
)
NV_R_ONXHWMFI_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ONXHWMFI_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_ONXHWMFI_F_WQYWMRVP
)

NV_R_ONXHWMFI_F_MLEOLJIS = FieldMetadata(
    name='NV_R_ONXHWMFI_F_MLEOLJIS',
    msb=17,
    lsb=17,
    register=NV_R_ONXHWMFI
)

NV_R_ONXHWMFI_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ONXHWMFI_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_ONXHWMFI_F_MLEOLJIS
)
NV_R_ONXHWMFI_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ONXHWMFI_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_ONXHWMFI_F_MLEOLJIS
)

NV_R_ANLSITRL = RegisterMetadata(
    name='NV_R_ANLSITRL',
    address=0x10fc14,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_ANLSITRL_F_CTBCFRUB = FieldMetadata(
    name='NV_R_ANLSITRL_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_ANLSITRL
)

NV_R_ANLSITRL_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_ANLSITRL_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_ANLSITRL_F_CTBCFRUB
)

NV_R_ANLSITRL_F_QVYAARWP = FieldMetadata(
    name='NV_R_ANLSITRL_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_ANLSITRL
)

NV_R_ANLSITRL_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_ANLSITRL_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_ANLSITRL_F_QVYAARWP
)

