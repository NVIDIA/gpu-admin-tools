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
NV_R_IFONBPTD = RegisterMetadata(
    name='NV_R_IFONBPTD',
    address=0x43c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_IFONBPTD_F_MJNVPZBK = FieldMetadata(
    name='NV_R_IFONBPTD_F_MJNVPZBK',
    msb=5,
    lsb=5,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_MJNVPZBK_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_MJNVPZBK_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_MJNVPZBK
)
NV_R_IFONBPTD_F_MJNVPZBK_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_MJNVPZBK_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_MJNVPZBK
)

NV_R_IFONBPTD_F_QHVMMPVH = FieldMetadata(
    name='NV_R_IFONBPTD_F_QHVMMPVH',
    msb=16,
    lsb=16,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_QHVMMPVH_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_QHVMMPVH_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_QHVMMPVH
)
NV_R_IFONBPTD_F_QHVMMPVH_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_QHVMMPVH_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_QHVMMPVH
)

NV_R_IFONBPTD_F_BMQGEYHF = FieldMetadata(
    name='NV_R_IFONBPTD_F_BMQGEYHF',
    msb=11,
    lsb=11,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_BMQGEYHF_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_BMQGEYHF_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_BMQGEYHF
)
NV_R_IFONBPTD_F_BMQGEYHF_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_BMQGEYHF_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_BMQGEYHF
)

NV_R_IFONBPTD_F_NFDMGIMX = FieldMetadata(
    name='NV_R_IFONBPTD_F_NFDMGIMX',
    msb=10,
    lsb=10,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_NFDMGIMX_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_NFDMGIMX_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_NFDMGIMX
)
NV_R_IFONBPTD_F_NFDMGIMX_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_NFDMGIMX_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_NFDMGIMX
)

NV_R_IFONBPTD_F_GTMXWCYR = FieldMetadata(
    name='NV_R_IFONBPTD_F_GTMXWCYR',
    msb=13,
    lsb=13,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_GTMXWCYR_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_GTMXWCYR_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_GTMXWCYR
)
NV_R_IFONBPTD_F_GTMXWCYR_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_GTMXWCYR_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_GTMXWCYR
)

NV_R_IFONBPTD_F_VZVFUCFG = FieldMetadata(
    name='NV_R_IFONBPTD_F_VZVFUCFG',
    msb=0,
    lsb=0,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_VZVFUCFG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_VZVFUCFG_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_VZVFUCFG
)
NV_R_IFONBPTD_F_VZVFUCFG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_VZVFUCFG_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_VZVFUCFG
)

NV_R_IFONBPTD_F_LMNVFLXF = FieldMetadata(
    name='NV_R_IFONBPTD_F_LMNVFLXF',
    msb=1,
    lsb=1,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_LMNVFLXF_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_LMNVFLXF_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_LMNVFLXF
)
NV_R_IFONBPTD_F_LMNVFLXF_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_LMNVFLXF_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_LMNVFLXF
)

NV_R_IFONBPTD_F_KLHQDLZS = FieldMetadata(
    name='NV_R_IFONBPTD_F_KLHQDLZS',
    msb=2,
    lsb=2,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_KLHQDLZS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_KLHQDLZS_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_KLHQDLZS
)
NV_R_IFONBPTD_F_KLHQDLZS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_KLHQDLZS_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_KLHQDLZS
)

NV_R_IFONBPTD_F_YGRKOIJR = FieldMetadata(
    name='NV_R_IFONBPTD_F_YGRKOIJR',
    msb=3,
    lsb=3,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_YGRKOIJR_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_YGRKOIJR_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_YGRKOIJR
)
NV_R_IFONBPTD_F_YGRKOIJR_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_YGRKOIJR_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_YGRKOIJR
)

NV_R_IFONBPTD_F_AFXUVSUW = FieldMetadata(
    name='NV_R_IFONBPTD_F_AFXUVSUW',
    msb=4,
    lsb=4,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_AFXUVSUW_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_AFXUVSUW_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_AFXUVSUW
)
NV_R_IFONBPTD_F_AFXUVSUW_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_AFXUVSUW_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_AFXUVSUW
)

NV_R_IFONBPTD_F_PBZTSBBN = FieldMetadata(
    name='NV_R_IFONBPTD_F_PBZTSBBN',
    msb=14,
    lsb=14,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_PBZTSBBN_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_PBZTSBBN_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_PBZTSBBN
)
NV_R_IFONBPTD_F_PBZTSBBN_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_PBZTSBBN_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_PBZTSBBN
)

NV_R_IFONBPTD_F_BXPLDVKC = FieldMetadata(
    name='NV_R_IFONBPTD_F_BXPLDVKC',
    msb=12,
    lsb=12,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_BXPLDVKC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_BXPLDVKC_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_BXPLDVKC
)
NV_R_IFONBPTD_F_BXPLDVKC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_BXPLDVKC_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_BXPLDVKC
)

NV_R_IFONBPTD_F_EJDHFWJF = FieldMetadata(
    name='NV_R_IFONBPTD_F_EJDHFWJF',
    msb=17,
    lsb=17,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_EJDHFWJF_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_EJDHFWJF_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_EJDHFWJF
)
NV_R_IFONBPTD_F_EJDHFWJF_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_EJDHFWJF_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_EJDHFWJF
)

NV_R_IFONBPTD_F_QIJLFLWE = FieldMetadata(
    name='NV_R_IFONBPTD_F_QIJLFLWE',
    msb=9,
    lsb=9,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_QIJLFLWE_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_QIJLFLWE_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_QIJLFLWE
)
NV_R_IFONBPTD_F_QIJLFLWE_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_QIJLFLWE_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_QIJLFLWE
)

NV_R_IFONBPTD_F_QEHPAHCB = FieldMetadata(
    name='NV_R_IFONBPTD_F_QEHPAHCB',
    msb=8,
    lsb=8,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_QEHPAHCB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_QEHPAHCB_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_QEHPAHCB
)
NV_R_IFONBPTD_F_QEHPAHCB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_QEHPAHCB_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_QEHPAHCB
)

NV_R_IFONBPTD_F_PAESGSDA = FieldMetadata(
    name='NV_R_IFONBPTD_F_PAESGSDA',
    msb=15,
    lsb=15,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_PAESGSDA_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_PAESGSDA_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_PAESGSDA
)
NV_R_IFONBPTD_F_PAESGSDA_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_PAESGSDA_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_PAESGSDA
)

NV_R_IFONBPTD_F_RWBUHQGQ = FieldMetadata(
    name='NV_R_IFONBPTD_F_RWBUHQGQ',
    msb=7,
    lsb=7,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_RWBUHQGQ_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_RWBUHQGQ_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_RWBUHQGQ
)
NV_R_IFONBPTD_F_RWBUHQGQ_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_RWBUHQGQ_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_RWBUHQGQ
)

NV_R_IFONBPTD_F_DUBPNWKW = FieldMetadata(
    name='NV_R_IFONBPTD_F_DUBPNWKW',
    msb=6,
    lsb=6,
    register=NV_R_IFONBPTD
)

NV_R_IFONBPTD_F_DUBPNWKW_V_DOGBFDTH = ValueMetadata(
    name='NV_R_IFONBPTD_F_DUBPNWKW_V_DOGBFDTH',
    value=0,
    field=NV_R_IFONBPTD_F_DUBPNWKW
)
NV_R_IFONBPTD_F_DUBPNWKW_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IFONBPTD_F_DUBPNWKW_V_PIUJQBKV',
    value=1,
    field=NV_R_IFONBPTD_F_DUBPNWKW
)

NV_R_XVMSABEP = RegisterMetadata(
    name='NV_R_XVMSABEP',
    address=0xe00,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_XVMSABEP_F_CIIFQXDM = FieldMetadata(
    name='NV_R_XVMSABEP_F_CIIFQXDM',
    msb=23,
    lsb=16,
    register=NV_R_XVMSABEP
)

NV_R_XVMSABEP_F_CIIFQXDM_V_GQHYEKHS = ValueMetadata(
    name='NV_R_XVMSABEP_F_CIIFQXDM_V_GQHYEKHS',
    value=0,
    field=NV_R_XVMSABEP_F_CIIFQXDM
)

NV_R_XVMSABEP_F_RISFHBOF = FieldMetadata(
    name='NV_R_XVMSABEP_F_RISFHBOF',
    msb=15,
    lsb=8,
    register=NV_R_XVMSABEP
)

NV_R_XVMSABEP_F_RISFHBOF_V_GQHYEKHS = ValueMetadata(
    name='NV_R_XVMSABEP_F_RISFHBOF_V_GQHYEKHS',
    value=0,
    field=NV_R_XVMSABEP_F_RISFHBOF
)

NV_R_XVMSABEP_F_DJNTXIUO = FieldMetadata(
    name='NV_R_XVMSABEP_F_DJNTXIUO',
    msb=7,
    lsb=0,
    register=NV_R_XVMSABEP
)

NV_R_XVMSABEP_F_DJNTXIUO_V_GQHYEKHS = ValueMetadata(
    name='NV_R_XVMSABEP_F_DJNTXIUO_V_GQHYEKHS',
    value=0,
    field=NV_R_XVMSABEP_F_DJNTXIUO
)

NV_R_XVMSABEP_F_YKIDRUDL = FieldMetadata(
    name='NV_R_XVMSABEP_F_YKIDRUDL',
    msb=31,
    lsb=24,
    register=NV_R_XVMSABEP
)

NV_R_XVMSABEP_F_YKIDRUDL_V_GQHYEKHS = ValueMetadata(
    name='NV_R_XVMSABEP_F_YKIDRUDL_V_GQHYEKHS',
    value=0,
    field=NV_R_XVMSABEP_F_YKIDRUDL
)

NV_R_IYWEFOWF = RegisterMetadata(
    name='NV_R_IYWEFOWF',
    address=0xe0c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_IYWEFOWF_F_OOQWCNRV = FieldMetadata(
    name='NV_R_IYWEFOWF_F_OOQWCNRV',
    msb=15,
    lsb=0,
    register=NV_R_IYWEFOWF
)

NV_R_IYWEFOWF_F_OOQWCNRV_V_GQHYEKHS = ValueMetadata(
    name='NV_R_IYWEFOWF_F_OOQWCNRV_V_GQHYEKHS',
    value=0,
    field=NV_R_IYWEFOWF_F_OOQWCNRV
)

NV_R_DXTGTWEB = RegisterMetadata(
    name='NV_R_DXTGTWEB',
    address=0xd18,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_DXTGTWEB_F_KIMMTSOV = FieldMetadata(
    name='NV_R_DXTGTWEB_F_KIMMTSOV',
    msb=0,
    lsb=0,
    register=NV_R_DXTGTWEB
)

NV_R_DXTGTWEB_F_KIMMTSOV_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_DXTGTWEB_F_KIMMTSOV_V_ZRRJKDVX',
    value=0,
    field=NV_R_DXTGTWEB_F_KIMMTSOV
)

NV_R_MEAZTBRA = RegisterMetadata(
    name='NV_R_MEAZTBRA',
    address=0x3c8,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_MEAZTBRA_F_PDYOAZAL = FieldMetadata(
    name='NV_R_MEAZTBRA_F_PDYOAZAL',
    msb=3,
    lsb=3,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_PDYOAZAL_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_PDYOAZAL_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_PDYOAZAL
)
NV_R_MEAZTBRA_F_PDYOAZAL_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_PDYOAZAL_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_PDYOAZAL
)

NV_R_MEAZTBRA_F_CDJDWAEA = FieldMetadata(
    name='NV_R_MEAZTBRA_F_CDJDWAEA',
    msb=4,
    lsb=4,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_CDJDWAEA_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_CDJDWAEA_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_CDJDWAEA
)
NV_R_MEAZTBRA_F_CDJDWAEA_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_CDJDWAEA_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_CDJDWAEA
)

NV_R_MEAZTBRA_F_BOWBRLZL = FieldMetadata(
    name='NV_R_MEAZTBRA_F_BOWBRLZL',
    msb=5,
    lsb=5,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_BOWBRLZL_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_BOWBRLZL_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_BOWBRLZL
)
NV_R_MEAZTBRA_F_BOWBRLZL_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_BOWBRLZL_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_BOWBRLZL
)

NV_R_MEAZTBRA_F_NVFIZSAH = FieldMetadata(
    name='NV_R_MEAZTBRA_F_NVFIZSAH',
    msb=6,
    lsb=6,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_NVFIZSAH_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_NVFIZSAH_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_NVFIZSAH
)
NV_R_MEAZTBRA_F_NVFIZSAH_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_NVFIZSAH_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_NVFIZSAH
)

NV_R_MEAZTBRA_F_LPUMLTUV = FieldMetadata(
    name='NV_R_MEAZTBRA_F_LPUMLTUV',
    msb=21,
    lsb=21,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_LPUMLTUV_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_LPUMLTUV_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_LPUMLTUV
)
NV_R_MEAZTBRA_F_LPUMLTUV_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_LPUMLTUV_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_LPUMLTUV
)

NV_R_MEAZTBRA_F_SDEKVMLM = FieldMetadata(
    name='NV_R_MEAZTBRA_F_SDEKVMLM',
    msb=18,
    lsb=18,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_SDEKVMLM_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_SDEKVMLM_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_SDEKVMLM
)
NV_R_MEAZTBRA_F_SDEKVMLM_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_SDEKVMLM_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_SDEKVMLM
)

NV_R_MEAZTBRA_F_ZOKAJUCQ = FieldMetadata(
    name='NV_R_MEAZTBRA_F_ZOKAJUCQ',
    msb=20,
    lsb=20,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_ZOKAJUCQ_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_ZOKAJUCQ_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_ZOKAJUCQ
)
NV_R_MEAZTBRA_F_ZOKAJUCQ_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_ZOKAJUCQ_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_ZOKAJUCQ
)

NV_R_MEAZTBRA_F_TXVSWBOA = FieldMetadata(
    name='NV_R_MEAZTBRA_F_TXVSWBOA',
    msb=19,
    lsb=19,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_TXVSWBOA_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_TXVSWBOA_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_TXVSWBOA
)
NV_R_MEAZTBRA_F_TXVSWBOA_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_TXVSWBOA_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_TXVSWBOA
)

NV_R_MEAZTBRA_F_RFMIYFYW = FieldMetadata(
    name='NV_R_MEAZTBRA_F_RFMIYFYW',
    msb=27,
    lsb=27,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_RFMIYFYW_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_RFMIYFYW_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_RFMIYFYW
)
NV_R_MEAZTBRA_F_RFMIYFYW_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_RFMIYFYW_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_RFMIYFYW
)

NV_R_MEAZTBRA_F_CLKJKYDK = FieldMetadata(
    name='NV_R_MEAZTBRA_F_CLKJKYDK',
    msb=28,
    lsb=28,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_CLKJKYDK_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_CLKJKYDK_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_CLKJKYDK
)
NV_R_MEAZTBRA_F_CLKJKYDK_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_CLKJKYDK_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_CLKJKYDK
)

NV_R_MEAZTBRA_F_ETCIHKHK = FieldMetadata(
    name='NV_R_MEAZTBRA_F_ETCIHKHK',
    msb=26,
    lsb=26,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_ETCIHKHK_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_ETCIHKHK_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_ETCIHKHK
)
NV_R_MEAZTBRA_F_ETCIHKHK_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_ETCIHKHK_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_ETCIHKHK
)

NV_R_MEAZTBRA_F_KUHAFDXT = FieldMetadata(
    name='NV_R_MEAZTBRA_F_KUHAFDXT',
    msb=25,
    lsb=25,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_KUHAFDXT_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_KUHAFDXT_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_KUHAFDXT
)
NV_R_MEAZTBRA_F_KUHAFDXT_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_KUHAFDXT_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_KUHAFDXT
)

NV_R_MEAZTBRA_F_ZINQFUIG = FieldMetadata(
    name='NV_R_MEAZTBRA_F_ZINQFUIG',
    msb=22,
    lsb=22,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_ZINQFUIG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_ZINQFUIG_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_ZINQFUIG
)
NV_R_MEAZTBRA_F_ZINQFUIG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_ZINQFUIG_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_ZINQFUIG
)

NV_R_MEAZTBRA_F_DCOPCWLH = FieldMetadata(
    name='NV_R_MEAZTBRA_F_DCOPCWLH',
    msb=24,
    lsb=24,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_DCOPCWLH_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_DCOPCWLH_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_DCOPCWLH
)
NV_R_MEAZTBRA_F_DCOPCWLH_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_DCOPCWLH_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_DCOPCWLH
)

NV_R_MEAZTBRA_F_VVPXTCYP = FieldMetadata(
    name='NV_R_MEAZTBRA_F_VVPXTCYP',
    msb=29,
    lsb=29,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_VVPXTCYP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_VVPXTCYP_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_VVPXTCYP
)
NV_R_MEAZTBRA_F_VVPXTCYP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_VVPXTCYP_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_VVPXTCYP
)

NV_R_MEAZTBRA_F_HECUUMTS = FieldMetadata(
    name='NV_R_MEAZTBRA_F_HECUUMTS',
    msb=30,
    lsb=30,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_HECUUMTS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_HECUUMTS_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_HECUUMTS
)
NV_R_MEAZTBRA_F_HECUUMTS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_HECUUMTS_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_HECUUMTS
)

NV_R_MEAZTBRA_F_KKDSPMHE = FieldMetadata(
    name='NV_R_MEAZTBRA_F_KKDSPMHE',
    msb=23,
    lsb=23,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_KKDSPMHE_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_KKDSPMHE_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_KKDSPMHE
)
NV_R_MEAZTBRA_F_KKDSPMHE_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_KKDSPMHE_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_KKDSPMHE
)

NV_R_MEAZTBRA_F_MJJYVJAG = FieldMetadata(
    name='NV_R_MEAZTBRA_F_MJJYVJAG',
    msb=7,
    lsb=7,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_MJJYVJAG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_MJJYVJAG_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_MJJYVJAG
)
NV_R_MEAZTBRA_F_MJJYVJAG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_MJJYVJAG_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_MJJYVJAG
)

NV_R_MEAZTBRA_F_LWVRGFOO = FieldMetadata(
    name='NV_R_MEAZTBRA_F_LWVRGFOO',
    msb=8,
    lsb=8,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_LWVRGFOO_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_LWVRGFOO_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_LWVRGFOO
)
NV_R_MEAZTBRA_F_LWVRGFOO_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_LWVRGFOO_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_LWVRGFOO
)

NV_R_MEAZTBRA_F_DITNJHOC = FieldMetadata(
    name='NV_R_MEAZTBRA_F_DITNJHOC',
    msb=10,
    lsb=10,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_DITNJHOC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_DITNJHOC_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_DITNJHOC
)
NV_R_MEAZTBRA_F_DITNJHOC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_DITNJHOC_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_DITNJHOC
)

NV_R_MEAZTBRA_F_XGEZKGAE = FieldMetadata(
    name='NV_R_MEAZTBRA_F_XGEZKGAE',
    msb=11,
    lsb=11,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_XGEZKGAE_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_XGEZKGAE_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_XGEZKGAE
)
NV_R_MEAZTBRA_F_XGEZKGAE_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_XGEZKGAE_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_XGEZKGAE
)

NV_R_MEAZTBRA_F_QPUESYQV = FieldMetadata(
    name='NV_R_MEAZTBRA_F_QPUESYQV',
    msb=12,
    lsb=12,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_QPUESYQV_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_QPUESYQV_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_QPUESYQV
)
NV_R_MEAZTBRA_F_QPUESYQV_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_QPUESYQV_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_QPUESYQV
)

NV_R_MEAZTBRA_F_ZVMEODOQ = FieldMetadata(
    name='NV_R_MEAZTBRA_F_ZVMEODOQ',
    msb=9,
    lsb=9,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_ZVMEODOQ_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_ZVMEODOQ_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_ZVMEODOQ
)
NV_R_MEAZTBRA_F_ZVMEODOQ_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_ZVMEODOQ_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_ZVMEODOQ
)

NV_R_MEAZTBRA_F_VZQPWPEN = FieldMetadata(
    name='NV_R_MEAZTBRA_F_VZQPWPEN',
    msb=14,
    lsb=14,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_VZQPWPEN_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_VZQPWPEN_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_VZQPWPEN
)
NV_R_MEAZTBRA_F_VZQPWPEN_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_VZQPWPEN_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_VZQPWPEN
)

NV_R_MEAZTBRA_F_BXORYTGA = FieldMetadata(
    name='NV_R_MEAZTBRA_F_BXORYTGA',
    msb=13,
    lsb=13,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_BXORYTGA_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_BXORYTGA_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_BXORYTGA
)
NV_R_MEAZTBRA_F_BXORYTGA_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_BXORYTGA_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_BXORYTGA
)

NV_R_MEAZTBRA_F_PBDGFTGM = FieldMetadata(
    name='NV_R_MEAZTBRA_F_PBDGFTGM',
    msb=2,
    lsb=2,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_PBDGFTGM_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_PBDGFTGM_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_PBDGFTGM
)
NV_R_MEAZTBRA_F_PBDGFTGM_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_PBDGFTGM_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_PBDGFTGM
)

NV_R_MEAZTBRA_F_WNFBZVYT = FieldMetadata(
    name='NV_R_MEAZTBRA_F_WNFBZVYT',
    msb=1,
    lsb=1,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_WNFBZVYT_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_WNFBZVYT_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_WNFBZVYT
)
NV_R_MEAZTBRA_F_WNFBZVYT_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_WNFBZVYT_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_WNFBZVYT
)

NV_R_MEAZTBRA_F_USRGJYVI = FieldMetadata(
    name='NV_R_MEAZTBRA_F_USRGJYVI',
    msb=0,
    lsb=0,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_USRGJYVI_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_USRGJYVI_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_USRGJYVI
)
NV_R_MEAZTBRA_F_USRGJYVI_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_USRGJYVI_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_USRGJYVI
)

NV_R_MEAZTBRA_F_LIRMOYQN = FieldMetadata(
    name='NV_R_MEAZTBRA_F_LIRMOYQN',
    msb=15,
    lsb=15,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_LIRMOYQN_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_LIRMOYQN_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_LIRMOYQN
)
NV_R_MEAZTBRA_F_LIRMOYQN_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_LIRMOYQN_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_LIRMOYQN
)

NV_R_MEAZTBRA_F_AZPPAOOR = FieldMetadata(
    name='NV_R_MEAZTBRA_F_AZPPAOOR',
    msb=16,
    lsb=16,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_AZPPAOOR_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_AZPPAOOR_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_AZPPAOOR
)
NV_R_MEAZTBRA_F_AZPPAOOR_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_AZPPAOOR_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_AZPPAOOR
)

NV_R_MEAZTBRA_F_HCYPSNFE = FieldMetadata(
    name='NV_R_MEAZTBRA_F_HCYPSNFE',
    msb=17,
    lsb=17,
    register=NV_R_MEAZTBRA
)

NV_R_MEAZTBRA_F_HCYPSNFE_V_DOGBFDTH = ValueMetadata(
    name='NV_R_MEAZTBRA_F_HCYPSNFE_V_DOGBFDTH',
    value=0,
    field=NV_R_MEAZTBRA_F_HCYPSNFE
)
NV_R_MEAZTBRA_F_HCYPSNFE_V_PIUJQBKV = ValueMetadata(
    name='NV_R_MEAZTBRA_F_HCYPSNFE_V_PIUJQBKV',
    value=1,
    field=NV_R_MEAZTBRA_F_HCYPSNFE
)

