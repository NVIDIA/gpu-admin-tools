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
NV_R_CAPNLTRH = RegisterMetadata(
    name='NV_R_CAPNLTRH',
    address=0x594,
    zero_based=True
)

NV_R_CAPNLTRH_F_OJYQBJXU = FieldMetadata(
    name='NV_R_CAPNLTRH_F_OJYQBJXU',
    msb=10,
    lsb=0,
    register=NV_R_CAPNLTRH
)

NV_R_LNTDDVZE = RegisterMetadata(
    name='NV_R_LNTDDVZE',
    address=0x58c,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_LNTDDVZE_F_CTBCFRUB = FieldMetadata(
    name='NV_R_LNTDDVZE_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_LNTDDVZE
)

NV_R_LNTDDVZE_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LNTDDVZE_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_LNTDDVZE_F_CTBCFRUB
)

NV_R_LNTDDVZE_F_QVYAARWP = FieldMetadata(
    name='NV_R_LNTDDVZE_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_LNTDDVZE
)

NV_R_LNTDDVZE_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LNTDDVZE_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_LNTDDVZE_F_QVYAARWP
)

NV_R_YFQGGFSX = RegisterMetadata(
    name='NV_R_YFQGGFSX',
    address=0x588,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_YFQGGFSX_F_DBYSWVHB = FieldMetadata(
    name='NV_R_YFQGGFSX_F_DBYSWVHB',
    msb=0,
    lsb=0,
    register=NV_R_YFQGGFSX
)

NV_R_YFQGGFSX_F_DBYSWVHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_YFQGGFSX_F_DBYSWVHB_V_DOGBFDTH',
    value=0,
    field=NV_R_YFQGGFSX_F_DBYSWVHB
)
NV_R_YFQGGFSX_F_DBYSWVHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_YFQGGFSX_F_DBYSWVHB_V_PIUJQBKV',
    value=1,
    field=NV_R_YFQGGFSX_F_DBYSWVHB
)

NV_R_YFQGGFSX_F_UURIHQGO = FieldMetadata(
    name='NV_R_YFQGGFSX_F_UURIHQGO',
    msb=8,
    lsb=8,
    register=NV_R_YFQGGFSX
)

NV_R_YFQGGFSX_F_UURIHQGO_V_DOGBFDTH = ValueMetadata(
    name='NV_R_YFQGGFSX_F_UURIHQGO_V_DOGBFDTH',
    value=0,
    field=NV_R_YFQGGFSX_F_UURIHQGO
)
NV_R_YFQGGFSX_F_UURIHQGO_V_PIUJQBKV = ValueMetadata(
    name='NV_R_YFQGGFSX_F_UURIHQGO_V_PIUJQBKV',
    value=1,
    field=NV_R_YFQGGFSX_F_UURIHQGO
)

NV_R_QLKCSVDV = RegisterMetadata(
    name='NV_R_QLKCSVDV',
    address=0x584,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_QLKCSVDV_F_SFBPPWLV = FieldMetadata(
    name='NV_R_QLKCSVDV_F_SFBPPWLV',
    msb=0,
    lsb=0,
    register=NV_R_QLKCSVDV
)

NV_R_QLKCSVDV_F_SFBPPWLV_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QLKCSVDV_F_SFBPPWLV_V_DOGBFDTH',
    value=0,
    field=NV_R_QLKCSVDV_F_SFBPPWLV
)
NV_R_QLKCSVDV_F_SFBPPWLV_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QLKCSVDV_F_SFBPPWLV_V_PIUJQBKV',
    value=1,
    field=NV_R_QLKCSVDV_F_SFBPPWLV
)

NV_R_QLKCSVDV_F_NFYIVCEN = FieldMetadata(
    name='NV_R_QLKCSVDV_F_NFYIVCEN',
    msb=2,
    lsb=2,
    register=NV_R_QLKCSVDV
)

NV_R_QLKCSVDV_F_NFYIVCEN_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QLKCSVDV_F_NFYIVCEN_V_DOGBFDTH',
    value=0,
    field=NV_R_QLKCSVDV_F_NFYIVCEN
)
NV_R_QLKCSVDV_F_NFYIVCEN_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QLKCSVDV_F_NFYIVCEN_V_PIUJQBKV',
    value=1,
    field=NV_R_QLKCSVDV_F_NFYIVCEN
)

NV_R_QLKCSVDV_F_QMOEAKCW = FieldMetadata(
    name='NV_R_QLKCSVDV_F_QMOEAKCW',
    msb=4,
    lsb=4,
    register=NV_R_QLKCSVDV
)

NV_R_QLKCSVDV_F_QMOEAKCW_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QLKCSVDV_F_QMOEAKCW_V_DOGBFDTH',
    value=0,
    field=NV_R_QLKCSVDV_F_QMOEAKCW
)
NV_R_QLKCSVDV_F_QMOEAKCW_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QLKCSVDV_F_QMOEAKCW_V_PIUJQBKV',
    value=1,
    field=NV_R_QLKCSVDV_F_QMOEAKCW
)

NV_R_QLKCSVDV_F_NYJNHHJQ = FieldMetadata(
    name='NV_R_QLKCSVDV_F_NYJNHHJQ',
    msb=6,
    lsb=6,
    register=NV_R_QLKCSVDV
)

NV_R_QLKCSVDV_F_NYJNHHJQ_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QLKCSVDV_F_NYJNHHJQ_V_DOGBFDTH',
    value=0,
    field=NV_R_QLKCSVDV_F_NYJNHHJQ
)
NV_R_QLKCSVDV_F_NYJNHHJQ_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QLKCSVDV_F_NYJNHHJQ_V_PIUJQBKV',
    value=1,
    field=NV_R_QLKCSVDV_F_NYJNHHJQ
)

NV_R_QLKCSVDV_F_OJXVVOEC = FieldMetadata(
    name='NV_R_QLKCSVDV_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_QLKCSVDV
)

NV_R_QLKCSVDV_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QLKCSVDV_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_QLKCSVDV_F_OJXVVOEC
)
NV_R_QLKCSVDV_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QLKCSVDV_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_QLKCSVDV_F_OJXVVOEC
)

NV_R_QLKCSVDV_F_VEOHPHHB = FieldMetadata(
    name='NV_R_QLKCSVDV_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_QLKCSVDV
)

NV_R_QLKCSVDV_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QLKCSVDV_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_QLKCSVDV_F_VEOHPHHB
)
NV_R_QLKCSVDV_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QLKCSVDV_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_QLKCSVDV_F_VEOHPHHB
)

NV_R_QLKCSVDV_F_KVYOKDXN = FieldMetadata(
    name='NV_R_QLKCSVDV_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_QLKCSVDV
)

NV_R_QLKCSVDV_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_QLKCSVDV_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_QLKCSVDV_F_KVYOKDXN
)
NV_R_QLKCSVDV_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_QLKCSVDV_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_QLKCSVDV_F_KVYOKDXN
)
NV_R_QLKCSVDV_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_QLKCSVDV_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_QLKCSVDV_F_KVYOKDXN
)

NV_R_QLKCSVDV_F_GAFICVXX = FieldMetadata(
    name='NV_R_QLKCSVDV_F_GAFICVXX',
    msb=1,
    lsb=1,
    register=NV_R_QLKCSVDV
)

NV_R_QLKCSVDV_F_GAFICVXX_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QLKCSVDV_F_GAFICVXX_V_DOGBFDTH',
    value=0,
    field=NV_R_QLKCSVDV_F_GAFICVXX
)
NV_R_QLKCSVDV_F_GAFICVXX_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QLKCSVDV_F_GAFICVXX_V_PIUJQBKV',
    value=1,
    field=NV_R_QLKCSVDV_F_GAFICVXX
)

NV_R_QLKCSVDV_F_UQVLNUJK = FieldMetadata(
    name='NV_R_QLKCSVDV_F_UQVLNUJK',
    msb=3,
    lsb=3,
    register=NV_R_QLKCSVDV
)

NV_R_QLKCSVDV_F_UQVLNUJK_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QLKCSVDV_F_UQVLNUJK_V_DOGBFDTH',
    value=0,
    field=NV_R_QLKCSVDV_F_UQVLNUJK
)
NV_R_QLKCSVDV_F_UQVLNUJK_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QLKCSVDV_F_UQVLNUJK_V_PIUJQBKV',
    value=1,
    field=NV_R_QLKCSVDV_F_UQVLNUJK
)

NV_R_QLKCSVDV_F_LDTFOZKB = FieldMetadata(
    name='NV_R_QLKCSVDV_F_LDTFOZKB',
    msb=5,
    lsb=5,
    register=NV_R_QLKCSVDV
)

NV_R_QLKCSVDV_F_LDTFOZKB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QLKCSVDV_F_LDTFOZKB_V_DOGBFDTH',
    value=0,
    field=NV_R_QLKCSVDV_F_LDTFOZKB
)
NV_R_QLKCSVDV_F_LDTFOZKB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QLKCSVDV_F_LDTFOZKB_V_PIUJQBKV',
    value=1,
    field=NV_R_QLKCSVDV_F_LDTFOZKB
)

NV_R_QLKCSVDV_F_EIMZUTEV = FieldMetadata(
    name='NV_R_QLKCSVDV_F_EIMZUTEV',
    msb=7,
    lsb=7,
    register=NV_R_QLKCSVDV
)

NV_R_QLKCSVDV_F_EIMZUTEV_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QLKCSVDV_F_EIMZUTEV_V_DOGBFDTH',
    value=0,
    field=NV_R_QLKCSVDV_F_EIMZUTEV
)
NV_R_QLKCSVDV_F_EIMZUTEV_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QLKCSVDV_F_EIMZUTEV_V_PIUJQBKV',
    value=1,
    field=NV_R_QLKCSVDV_F_EIMZUTEV
)

NV_R_QLKCSVDV_F_WQYWMRVP = FieldMetadata(
    name='NV_R_QLKCSVDV_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_QLKCSVDV
)

NV_R_QLKCSVDV_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QLKCSVDV_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_QLKCSVDV_F_WQYWMRVP
)
NV_R_QLKCSVDV_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QLKCSVDV_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_QLKCSVDV_F_WQYWMRVP
)

NV_R_QLKCSVDV_F_MLEOLJIS = FieldMetadata(
    name='NV_R_QLKCSVDV_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_QLKCSVDV
)

NV_R_QLKCSVDV_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QLKCSVDV_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_QLKCSVDV_F_MLEOLJIS
)
NV_R_QLKCSVDV_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QLKCSVDV_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_QLKCSVDV_F_MLEOLJIS
)

NV_R_QKOOTAJF = RegisterMetadata(
    name='NV_R_QKOOTAJF',
    address=0x590,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_QKOOTAJF_F_CTBCFRUB = FieldMetadata(
    name='NV_R_QKOOTAJF_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_QKOOTAJF
)

NV_R_QKOOTAJF_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_QKOOTAJF_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_QKOOTAJF_F_CTBCFRUB
)

NV_R_QKOOTAJF_F_QVYAARWP = FieldMetadata(
    name='NV_R_QKOOTAJF_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_QKOOTAJF
)

NV_R_QKOOTAJF_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_QKOOTAJF_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_QKOOTAJF_F_QVYAARWP
)

