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
NV_R_GCYYDXUO = RegisterMetadata(
    name='NV_R_GCYYDXUO',
    address=0x3d4,
    zero_based=True
)

NV_R_GCYYDXUO_F_IKQZFICP = FieldMetadata(
    name='NV_R_GCYYDXUO_F_IKQZFICP',
    msb=11,
    lsb=0,
    register=NV_R_GCYYDXUO
)

NV_R_GCYYDXUO_F_CLQFSQFY = FieldMetadata(
    name='NV_R_GCYYDXUO_F_CLQFSQFY',
    msb=27,
    lsb=12,
    register=NV_R_GCYYDXUO
)

NV_R_GCYYDXUO_F_OJYQBJXU = FieldMetadata(
    name='NV_R_GCYYDXUO_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_GCYYDXUO
)

NV_R_GCYYDXUO_F_SYSBRQCF = FieldMetadata(
    name='NV_R_GCYYDXUO_F_SYSBRQCF',
    msb=31,
    lsb=28,
    register=NV_R_GCYYDXUO
)

NV_R_GCYYDXUO_F_SYSBRQCF_V_WSMYDMLN = ValueMetadata(
    name='NV_R_GCYYDXUO_F_SYSBRQCF_V_WSMYDMLN',
    value=1,
    field=NV_R_GCYYDXUO_F_SYSBRQCF
)
NV_R_GCYYDXUO_F_SYSBRQCF_V_QSQHCHVJ = ValueMetadata(
    name='NV_R_GCYYDXUO_F_SYSBRQCF_V_QSQHCHVJ',
    value=0,
    field=NV_R_GCYYDXUO_F_SYSBRQCF
)

NV_R_YWFSIOZF = RegisterMetadata(
    name='NV_R_YWFSIOZF',
    address=0x3dc,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_YWFSIOZF_F_CTBCFRUB = FieldMetadata(
    name='NV_R_YWFSIOZF_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_YWFSIOZF
)

NV_R_YWFSIOZF_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_YWFSIOZF_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_YWFSIOZF_F_CTBCFRUB
)

NV_R_YWFSIOZF_F_QVYAARWP = FieldMetadata(
    name='NV_R_YWFSIOZF_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_YWFSIOZF
)

NV_R_YWFSIOZF_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_YWFSIOZF_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_YWFSIOZF_F_QVYAARWP
)

NV_R_BMMUWFFP = RegisterMetadata(
    name='NV_R_BMMUWFFP',
    address=0x3e0,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_BMMUWFFP_F_QEDACPOE = FieldMetadata(
    name='NV_R_BMMUWFFP_F_QEDACPOE',
    msb=2,
    lsb=2,
    register=NV_R_BMMUWFFP
)

NV_R_BMMUWFFP_F_QEDACPOE_V_DOGBFDTH = ValueMetadata(
    name='NV_R_BMMUWFFP_F_QEDACPOE_V_DOGBFDTH',
    value=0,
    field=NV_R_BMMUWFFP_F_QEDACPOE
)
NV_R_BMMUWFFP_F_QEDACPOE_V_PIUJQBKV = ValueMetadata(
    name='NV_R_BMMUWFFP_F_QEDACPOE_V_PIUJQBKV',
    value=1,
    field=NV_R_BMMUWFFP_F_QEDACPOE
)

NV_R_BMMUWFFP_F_RILJVQVY = FieldMetadata(
    name='NV_R_BMMUWFFP_F_RILJVQVY',
    msb=0,
    lsb=0,
    register=NV_R_BMMUWFFP
)

NV_R_BMMUWFFP_F_RILJVQVY_V_DOGBFDTH = ValueMetadata(
    name='NV_R_BMMUWFFP_F_RILJVQVY_V_DOGBFDTH',
    value=0,
    field=NV_R_BMMUWFFP_F_RILJVQVY
)
NV_R_BMMUWFFP_F_RILJVQVY_V_PIUJQBKV = ValueMetadata(
    name='NV_R_BMMUWFFP_F_RILJVQVY_V_PIUJQBKV',
    value=1,
    field=NV_R_BMMUWFFP_F_RILJVQVY
)

NV_R_BMMUWFFP_F_OJXVVOEC = FieldMetadata(
    name='NV_R_BMMUWFFP_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_BMMUWFFP
)

NV_R_BMMUWFFP_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_BMMUWFFP_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_BMMUWFFP_F_OJXVVOEC
)
NV_R_BMMUWFFP_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_BMMUWFFP_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_BMMUWFFP_F_OJXVVOEC
)

NV_R_BMMUWFFP_F_VEOHPHHB = FieldMetadata(
    name='NV_R_BMMUWFFP_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_BMMUWFFP
)

NV_R_BMMUWFFP_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_BMMUWFFP_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_BMMUWFFP_F_VEOHPHHB
)
NV_R_BMMUWFFP_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_BMMUWFFP_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_BMMUWFFP_F_VEOHPHHB
)

NV_R_BMMUWFFP_F_KVYOKDXN = FieldMetadata(
    name='NV_R_BMMUWFFP_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_BMMUWFFP
)

NV_R_BMMUWFFP_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_BMMUWFFP_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_BMMUWFFP_F_KVYOKDXN
)
NV_R_BMMUWFFP_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_BMMUWFFP_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_BMMUWFFP_F_KVYOKDXN
)
NV_R_BMMUWFFP_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_BMMUWFFP_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_BMMUWFFP_F_KVYOKDXN
)

NV_R_BMMUWFFP_F_YATBWBQB = FieldMetadata(
    name='NV_R_BMMUWFFP_F_YATBWBQB',
    msb=3,
    lsb=3,
    register=NV_R_BMMUWFFP
)

NV_R_BMMUWFFP_F_YATBWBQB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_BMMUWFFP_F_YATBWBQB_V_DOGBFDTH',
    value=0,
    field=NV_R_BMMUWFFP_F_YATBWBQB
)
NV_R_BMMUWFFP_F_YATBWBQB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_BMMUWFFP_F_YATBWBQB_V_PIUJQBKV',
    value=1,
    field=NV_R_BMMUWFFP_F_YATBWBQB
)

NV_R_BMMUWFFP_F_MALHTZIR = FieldMetadata(
    name='NV_R_BMMUWFFP_F_MALHTZIR',
    msb=1,
    lsb=1,
    register=NV_R_BMMUWFFP
)

NV_R_BMMUWFFP_F_MALHTZIR_V_DOGBFDTH = ValueMetadata(
    name='NV_R_BMMUWFFP_F_MALHTZIR_V_DOGBFDTH',
    value=0,
    field=NV_R_BMMUWFFP_F_MALHTZIR
)
NV_R_BMMUWFFP_F_MALHTZIR_V_PIUJQBKV = ValueMetadata(
    name='NV_R_BMMUWFFP_F_MALHTZIR_V_PIUJQBKV',
    value=1,
    field=NV_R_BMMUWFFP_F_MALHTZIR
)

NV_R_BMMUWFFP_F_WQYWMRVP = FieldMetadata(
    name='NV_R_BMMUWFFP_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_BMMUWFFP
)

NV_R_BMMUWFFP_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_BMMUWFFP_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_BMMUWFFP_F_WQYWMRVP
)
NV_R_BMMUWFFP_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_BMMUWFFP_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_BMMUWFFP_F_WQYWMRVP
)

NV_R_BMMUWFFP_F_MLEOLJIS = FieldMetadata(
    name='NV_R_BMMUWFFP_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_BMMUWFFP
)

NV_R_BMMUWFFP_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_BMMUWFFP_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_BMMUWFFP_F_MLEOLJIS
)
NV_R_BMMUWFFP_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_BMMUWFFP_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_BMMUWFFP_F_MLEOLJIS
)

NV_R_HIZPZKKM = RegisterMetadata(
    name='NV_R_HIZPZKKM',
    address=0x3d8,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_HIZPZKKM_F_CTBCFRUB = FieldMetadata(
    name='NV_R_HIZPZKKM_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_HIZPZKKM
)

NV_R_HIZPZKKM_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_HIZPZKKM_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_HIZPZKKM_F_CTBCFRUB
)

NV_R_HIZPZKKM_F_QVYAARWP = FieldMetadata(
    name='NV_R_HIZPZKKM_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_HIZPZKKM
)

NV_R_HIZPZKKM_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_HIZPZKKM_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_HIZPZKKM_F_QVYAARWP
)

NV_R_PCFSZNHU = RegisterMetadata(
    name='NV_R_PCFSZNHU',
    address=0x3e8,
    zero_based=True
)

NV_R_PCFSZNHU_F_IKQZFICP = FieldMetadata(
    name='NV_R_PCFSZNHU_F_IKQZFICP',
    msb=11,
    lsb=0,
    register=NV_R_PCFSZNHU
)

NV_R_PCFSZNHU_F_CLQFSQFY = FieldMetadata(
    name='NV_R_PCFSZNHU_F_CLQFSQFY',
    msb=27,
    lsb=12,
    register=NV_R_PCFSZNHU
)

NV_R_PCFSZNHU_F_OJYQBJXU = FieldMetadata(
    name='NV_R_PCFSZNHU_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_PCFSZNHU
)

NV_R_PCFSZNHU_F_SYSBRQCF = FieldMetadata(
    name='NV_R_PCFSZNHU_F_SYSBRQCF',
    msb=31,
    lsb=28,
    register=NV_R_PCFSZNHU
)

NV_R_PCFSZNHU_F_SYSBRQCF_V_QLKRUFSJ = ValueMetadata(
    name='NV_R_PCFSZNHU_F_SYSBRQCF_V_QLKRUFSJ',
    value=1,
    field=NV_R_PCFSZNHU_F_SYSBRQCF
)
NV_R_PCFSZNHU_F_SYSBRQCF_V_DNLOVNKK = ValueMetadata(
    name='NV_R_PCFSZNHU_F_SYSBRQCF_V_DNLOVNKK',
    value=0,
    field=NV_R_PCFSZNHU_F_SYSBRQCF
)
NV_R_PCFSZNHU_F_SYSBRQCF_V_IZZYIJZW = ValueMetadata(
    name='NV_R_PCFSZNHU_F_SYSBRQCF_V_IZZYIJZW',
    value=3,
    field=NV_R_PCFSZNHU_F_SYSBRQCF
)
NV_R_PCFSZNHU_F_SYSBRQCF_V_YSXOTDUE = ValueMetadata(
    name='NV_R_PCFSZNHU_F_SYSBRQCF_V_YSXOTDUE',
    value=2,
    field=NV_R_PCFSZNHU_F_SYSBRQCF
)
NV_R_PCFSZNHU_F_SYSBRQCF_V_AFJPTNWE = ValueMetadata(
    name='NV_R_PCFSZNHU_F_SYSBRQCF_V_AFJPTNWE',
    value=5,
    field=NV_R_PCFSZNHU_F_SYSBRQCF
)
NV_R_PCFSZNHU_F_SYSBRQCF_V_EKRZBNQQ = ValueMetadata(
    name='NV_R_PCFSZNHU_F_SYSBRQCF_V_EKRZBNQQ',
    value=4,
    field=NV_R_PCFSZNHU_F_SYSBRQCF
)
NV_R_PCFSZNHU_F_SYSBRQCF_V_MBEPYSRH = ValueMetadata(
    name='NV_R_PCFSZNHU_F_SYSBRQCF_V_MBEPYSRH',
    value=7,
    field=NV_R_PCFSZNHU_F_SYSBRQCF
)
NV_R_PCFSZNHU_F_SYSBRQCF_V_QNKPLDPJ = ValueMetadata(
    name='NV_R_PCFSZNHU_F_SYSBRQCF_V_QNKPLDPJ',
    value=6,
    field=NV_R_PCFSZNHU_F_SYSBRQCF
)
NV_R_PCFSZNHU_F_SYSBRQCF_V_WSMYDMLN = ValueMetadata(
    name='NV_R_PCFSZNHU_F_SYSBRQCF_V_WSMYDMLN',
    value=1,
    field=NV_R_PCFSZNHU_F_SYSBRQCF
)
NV_R_PCFSZNHU_F_SYSBRQCF_V_QSQHCHVJ = ValueMetadata(
    name='NV_R_PCFSZNHU_F_SYSBRQCF_V_QSQHCHVJ',
    value=0,
    field=NV_R_PCFSZNHU_F_SYSBRQCF
)

NV_R_UQBXNPUG = RegisterMetadata(
    name='NV_R_UQBXNPUG',
    address=0x3f0,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_UQBXNPUG_F_CTBCFRUB = FieldMetadata(
    name='NV_R_UQBXNPUG_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_UQBXNPUG
)

NV_R_UQBXNPUG_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_UQBXNPUG_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_UQBXNPUG_F_CTBCFRUB
)

NV_R_UQBXNPUG_F_QVYAARWP = FieldMetadata(
    name='NV_R_UQBXNPUG_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_UQBXNPUG
)

NV_R_UQBXNPUG_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_UQBXNPUG_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_UQBXNPUG_F_QVYAARWP
)

NV_R_RBSCTCLD = RegisterMetadata(
    name='NV_R_RBSCTCLD',
    address=0x3f4,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_RBSCTCLD_F_MBNDGEBT = FieldMetadata(
    name='NV_R_RBSCTCLD_F_MBNDGEBT',
    msb=2,
    lsb=2,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_MBNDGEBT_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_MBNDGEBT_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_MBNDGEBT
)
NV_R_RBSCTCLD_F_MBNDGEBT_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_MBNDGEBT_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_MBNDGEBT
)

NV_R_RBSCTCLD_F_QMCVUZCR = FieldMetadata(
    name='NV_R_RBSCTCLD_F_QMCVUZCR',
    msb=0,
    lsb=0,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_QMCVUZCR_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_QMCVUZCR_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_QMCVUZCR
)
NV_R_RBSCTCLD_F_QMCVUZCR_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_QMCVUZCR_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_QMCVUZCR
)

NV_R_RBSCTCLD_F_LISTPCJU = FieldMetadata(
    name='NV_R_RBSCTCLD_F_LISTPCJU',
    msb=6,
    lsb=6,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_LISTPCJU_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_LISTPCJU_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_LISTPCJU
)
NV_R_RBSCTCLD_F_LISTPCJU_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_LISTPCJU_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_LISTPCJU
)

NV_R_RBSCTCLD_F_NFAXKKVT = FieldMetadata(
    name='NV_R_RBSCTCLD_F_NFAXKKVT',
    msb=4,
    lsb=4,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_NFAXKKVT_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_NFAXKKVT_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_NFAXKKVT
)
NV_R_RBSCTCLD_F_NFAXKKVT_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_NFAXKKVT_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_NFAXKKVT
)

NV_R_RBSCTCLD_F_PQQXIYJD = FieldMetadata(
    name='NV_R_RBSCTCLD_F_PQQXIYJD',
    msb=10,
    lsb=10,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_PQQXIYJD_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_PQQXIYJD_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_PQQXIYJD
)
NV_R_RBSCTCLD_F_PQQXIYJD_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_PQQXIYJD_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_PQQXIYJD
)

NV_R_RBSCTCLD_F_CDQZOFQZ = FieldMetadata(
    name='NV_R_RBSCTCLD_F_CDQZOFQZ',
    msb=8,
    lsb=8,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_CDQZOFQZ_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_CDQZOFQZ_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_CDQZOFQZ
)
NV_R_RBSCTCLD_F_CDQZOFQZ_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_CDQZOFQZ_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_CDQZOFQZ
)

NV_R_RBSCTCLD_F_NNTXRWGK = FieldMetadata(
    name='NV_R_RBSCTCLD_F_NNTXRWGK',
    msb=14,
    lsb=14,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_NNTXRWGK_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_NNTXRWGK_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_NNTXRWGK
)
NV_R_RBSCTCLD_F_NNTXRWGK_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_NNTXRWGK_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_NNTXRWGK
)

NV_R_RBSCTCLD_F_KEFMPESZ = FieldMetadata(
    name='NV_R_RBSCTCLD_F_KEFMPESZ',
    msb=12,
    lsb=12,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_KEFMPESZ_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_KEFMPESZ_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_KEFMPESZ
)
NV_R_RBSCTCLD_F_KEFMPESZ_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_KEFMPESZ_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_KEFMPESZ
)

NV_R_RBSCTCLD_F_QEDACPOE = FieldMetadata(
    name='NV_R_RBSCTCLD_F_QEDACPOE',
    msb=2,
    lsb=2,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_QEDACPOE_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_QEDACPOE_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_QEDACPOE
)
NV_R_RBSCTCLD_F_QEDACPOE_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_QEDACPOE_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_QEDACPOE
)

NV_R_RBSCTCLD_F_RILJVQVY = FieldMetadata(
    name='NV_R_RBSCTCLD_F_RILJVQVY',
    msb=0,
    lsb=0,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_RILJVQVY_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_RILJVQVY_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_RILJVQVY
)
NV_R_RBSCTCLD_F_RILJVQVY_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_RILJVQVY_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_RILJVQVY
)

NV_R_RBSCTCLD_F_OJXVVOEC = FieldMetadata(
    name='NV_R_RBSCTCLD_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_OJXVVOEC
)
NV_R_RBSCTCLD_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_OJXVVOEC
)

NV_R_RBSCTCLD_F_VEOHPHHB = FieldMetadata(
    name='NV_R_RBSCTCLD_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_VEOHPHHB
)
NV_R_RBSCTCLD_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_VEOHPHHB
)

NV_R_RBSCTCLD_F_KVYOKDXN = FieldMetadata(
    name='NV_R_RBSCTCLD_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_RBSCTCLD_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_RBSCTCLD_F_KVYOKDXN
)
NV_R_RBSCTCLD_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_RBSCTCLD_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_RBSCTCLD_F_KVYOKDXN
)
NV_R_RBSCTCLD_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_RBSCTCLD_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_RBSCTCLD_F_KVYOKDXN
)

NV_R_RBSCTCLD_F_UKNAZCPR = FieldMetadata(
    name='NV_R_RBSCTCLD_F_UKNAZCPR',
    msb=3,
    lsb=3,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_UKNAZCPR_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_UKNAZCPR_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_UKNAZCPR
)
NV_R_RBSCTCLD_F_UKNAZCPR_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_UKNAZCPR_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_UKNAZCPR
)

NV_R_RBSCTCLD_F_BWNKJKYV = FieldMetadata(
    name='NV_R_RBSCTCLD_F_BWNKJKYV',
    msb=1,
    lsb=1,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_BWNKJKYV_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_BWNKJKYV_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_BWNKJKYV
)
NV_R_RBSCTCLD_F_BWNKJKYV_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_BWNKJKYV_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_BWNKJKYV
)

NV_R_RBSCTCLD_F_UIWEYREK = FieldMetadata(
    name='NV_R_RBSCTCLD_F_UIWEYREK',
    msb=7,
    lsb=7,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_UIWEYREK_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_UIWEYREK_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_UIWEYREK
)
NV_R_RBSCTCLD_F_UIWEYREK_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_UIWEYREK_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_UIWEYREK
)

NV_R_RBSCTCLD_F_FODFTHTI = FieldMetadata(
    name='NV_R_RBSCTCLD_F_FODFTHTI',
    msb=5,
    lsb=5,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_FODFTHTI_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_FODFTHTI_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_FODFTHTI
)
NV_R_RBSCTCLD_F_FODFTHTI_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_FODFTHTI_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_FODFTHTI
)

NV_R_RBSCTCLD_F_XWPBESES = FieldMetadata(
    name='NV_R_RBSCTCLD_F_XWPBESES',
    msb=11,
    lsb=11,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_XWPBESES_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_XWPBESES_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_XWPBESES
)
NV_R_RBSCTCLD_F_XWPBESES_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_XWPBESES_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_XWPBESES
)

NV_R_RBSCTCLD_F_GSGZSHYE = FieldMetadata(
    name='NV_R_RBSCTCLD_F_GSGZSHYE',
    msb=9,
    lsb=9,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_GSGZSHYE_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_GSGZSHYE_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_GSGZSHYE
)
NV_R_RBSCTCLD_F_GSGZSHYE_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_GSGZSHYE_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_GSGZSHYE
)

NV_R_RBSCTCLD_F_OYOTZJRI = FieldMetadata(
    name='NV_R_RBSCTCLD_F_OYOTZJRI',
    msb=15,
    lsb=15,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_OYOTZJRI_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_OYOTZJRI_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_OYOTZJRI
)
NV_R_RBSCTCLD_F_OYOTZJRI_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_OYOTZJRI_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_OYOTZJRI
)

NV_R_RBSCTCLD_F_WSMIOSPE = FieldMetadata(
    name='NV_R_RBSCTCLD_F_WSMIOSPE',
    msb=13,
    lsb=13,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_WSMIOSPE_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_WSMIOSPE_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_WSMIOSPE
)
NV_R_RBSCTCLD_F_WSMIOSPE_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_WSMIOSPE_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_WSMIOSPE
)

NV_R_RBSCTCLD_F_YATBWBQB = FieldMetadata(
    name='NV_R_RBSCTCLD_F_YATBWBQB',
    msb=3,
    lsb=3,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_YATBWBQB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_YATBWBQB_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_YATBWBQB
)
NV_R_RBSCTCLD_F_YATBWBQB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_YATBWBQB_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_YATBWBQB
)

NV_R_RBSCTCLD_F_MALHTZIR = FieldMetadata(
    name='NV_R_RBSCTCLD_F_MALHTZIR',
    msb=1,
    lsb=1,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_MALHTZIR_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_MALHTZIR_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_MALHTZIR
)
NV_R_RBSCTCLD_F_MALHTZIR_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_MALHTZIR_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_MALHTZIR
)

NV_R_RBSCTCLD_F_WQYWMRVP = FieldMetadata(
    name='NV_R_RBSCTCLD_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_WQYWMRVP
)
NV_R_RBSCTCLD_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_WQYWMRVP
)

NV_R_RBSCTCLD_F_MLEOLJIS = FieldMetadata(
    name='NV_R_RBSCTCLD_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_RBSCTCLD
)

NV_R_RBSCTCLD_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_RBSCTCLD_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_RBSCTCLD_F_MLEOLJIS
)
NV_R_RBSCTCLD_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_RBSCTCLD_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_RBSCTCLD_F_MLEOLJIS
)

NV_R_NFRCATXE = RegisterMetadata(
    name='NV_R_NFRCATXE',
    address=0x3ec,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_NFRCATXE_F_CTBCFRUB = FieldMetadata(
    name='NV_R_NFRCATXE_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_NFRCATXE
)

NV_R_NFRCATXE_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_NFRCATXE_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_NFRCATXE_F_CTBCFRUB
)

NV_R_NFRCATXE_F_QVYAARWP = FieldMetadata(
    name='NV_R_NFRCATXE_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_NFRCATXE
)

NV_R_NFRCATXE_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_NFRCATXE_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_NFRCATXE_F_QVYAARWP
)

