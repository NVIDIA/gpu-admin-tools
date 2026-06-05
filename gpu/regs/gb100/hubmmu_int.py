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
NV_R_JSOWLDXY = RegisterMetadata(
    name='NV_R_JSOWLDXY',
    address=0xa5a4,
    zero_based=True
)

NV_R_JSOWLDXY_F_IKQZFICP = FieldMetadata(
    name='NV_R_JSOWLDXY_F_IKQZFICP',
    msb=11,
    lsb=0,
    register=NV_R_JSOWLDXY
)

NV_R_JSOWLDXY_F_CLQFSQFY = FieldMetadata(
    name='NV_R_JSOWLDXY_F_CLQFSQFY',
    msb=27,
    lsb=12,
    register=NV_R_JSOWLDXY
)

NV_R_JSOWLDXY_F_OJYQBJXU = FieldMetadata(
    name='NV_R_JSOWLDXY_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_JSOWLDXY
)

NV_R_JSOWLDXY_F_SYSBRQCF = FieldMetadata(
    name='NV_R_JSOWLDXY_F_SYSBRQCF',
    msb=31,
    lsb=28,
    register=NV_R_JSOWLDXY
)

NV_R_JSOWLDXY_F_SYSBRQCF_V_BVGPLGJU = ValueMetadata(
    name='NV_R_JSOWLDXY_F_SYSBRQCF_V_BVGPLGJU',
    value=2,
    field=NV_R_JSOWLDXY_F_SYSBRQCF
)
NV_R_JSOWLDXY_F_SYSBRQCF_V_DRSZNSAZ = ValueMetadata(
    name='NV_R_JSOWLDXY_F_SYSBRQCF_V_DRSZNSAZ',
    value=1,
    field=NV_R_JSOWLDXY_F_SYSBRQCF
)

NV_R_AWKJAKDO = RegisterMetadata(
    name='NV_R_AWKJAKDO',
    address=0xa59c,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_AWKJAKDO_F_CTBCFRUB = FieldMetadata(
    name='NV_R_AWKJAKDO_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_AWKJAKDO
)

NV_R_AWKJAKDO_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_AWKJAKDO_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_AWKJAKDO_F_CTBCFRUB
)

NV_R_AWKJAKDO_F_QVYAARWP = FieldMetadata(
    name='NV_R_AWKJAKDO_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_AWKJAKDO
)

NV_R_AWKJAKDO_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_AWKJAKDO_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_AWKJAKDO_F_QVYAARWP
)

NV_R_EMKSAJYS = RegisterMetadata(
    name='NV_R_EMKSAJYS',
    address=0xa598,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_EMKSAJYS_F_QAUPMBGQ = FieldMetadata(
    name='NV_R_EMKSAJYS_F_QAUPMBGQ',
    msb=2,
    lsb=2,
    register=NV_R_EMKSAJYS
)

NV_R_EMKSAJYS_F_QAUPMBGQ_V_DOGBFDTH = ValueMetadata(
    name='NV_R_EMKSAJYS_F_QAUPMBGQ_V_DOGBFDTH',
    value=0,
    field=NV_R_EMKSAJYS_F_QAUPMBGQ
)
NV_R_EMKSAJYS_F_QAUPMBGQ_V_PIUJQBKV = ValueMetadata(
    name='NV_R_EMKSAJYS_F_QAUPMBGQ_V_PIUJQBKV',
    value=1,
    field=NV_R_EMKSAJYS_F_QAUPMBGQ
)

NV_R_EMKSAJYS_F_CPTNFTON = FieldMetadata(
    name='NV_R_EMKSAJYS_F_CPTNFTON',
    msb=0,
    lsb=0,
    register=NV_R_EMKSAJYS
)

NV_R_EMKSAJYS_F_CPTNFTON_V_DOGBFDTH = ValueMetadata(
    name='NV_R_EMKSAJYS_F_CPTNFTON_V_DOGBFDTH',
    value=0,
    field=NV_R_EMKSAJYS_F_CPTNFTON
)
NV_R_EMKSAJYS_F_CPTNFTON_V_PIUJQBKV = ValueMetadata(
    name='NV_R_EMKSAJYS_F_CPTNFTON_V_PIUJQBKV',
    value=1,
    field=NV_R_EMKSAJYS_F_CPTNFTON
)

NV_R_EMKSAJYS_F_OJXVVOEC = FieldMetadata(
    name='NV_R_EMKSAJYS_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_EMKSAJYS
)

NV_R_EMKSAJYS_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_EMKSAJYS_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_EMKSAJYS_F_OJXVVOEC
)
NV_R_EMKSAJYS_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_EMKSAJYS_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_EMKSAJYS_F_OJXVVOEC
)

NV_R_EMKSAJYS_F_VEOHPHHB = FieldMetadata(
    name='NV_R_EMKSAJYS_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_EMKSAJYS
)

NV_R_EMKSAJYS_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_EMKSAJYS_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_EMKSAJYS_F_VEOHPHHB
)
NV_R_EMKSAJYS_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_EMKSAJYS_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_EMKSAJYS_F_VEOHPHHB
)

NV_R_EMKSAJYS_F_KVYOKDXN = FieldMetadata(
    name='NV_R_EMKSAJYS_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_EMKSAJYS
)

NV_R_EMKSAJYS_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_EMKSAJYS_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_EMKSAJYS_F_KVYOKDXN
)
NV_R_EMKSAJYS_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_EMKSAJYS_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_EMKSAJYS_F_KVYOKDXN
)
NV_R_EMKSAJYS_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_EMKSAJYS_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_EMKSAJYS_F_KVYOKDXN
)

NV_R_EMKSAJYS_F_LEDOESZC = FieldMetadata(
    name='NV_R_EMKSAJYS_F_LEDOESZC',
    msb=3,
    lsb=3,
    register=NV_R_EMKSAJYS
)

NV_R_EMKSAJYS_F_LEDOESZC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_EMKSAJYS_F_LEDOESZC_V_DOGBFDTH',
    value=0,
    field=NV_R_EMKSAJYS_F_LEDOESZC
)
NV_R_EMKSAJYS_F_LEDOESZC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_EMKSAJYS_F_LEDOESZC_V_PIUJQBKV',
    value=1,
    field=NV_R_EMKSAJYS_F_LEDOESZC
)

NV_R_EMKSAJYS_F_PIJLJNAU = FieldMetadata(
    name='NV_R_EMKSAJYS_F_PIJLJNAU',
    msb=1,
    lsb=1,
    register=NV_R_EMKSAJYS
)

NV_R_EMKSAJYS_F_PIJLJNAU_V_DOGBFDTH = ValueMetadata(
    name='NV_R_EMKSAJYS_F_PIJLJNAU_V_DOGBFDTH',
    value=0,
    field=NV_R_EMKSAJYS_F_PIJLJNAU
)
NV_R_EMKSAJYS_F_PIJLJNAU_V_PIUJQBKV = ValueMetadata(
    name='NV_R_EMKSAJYS_F_PIJLJNAU_V_PIUJQBKV',
    value=1,
    field=NV_R_EMKSAJYS_F_PIJLJNAU
)

NV_R_EMKSAJYS_F_WQYWMRVP = FieldMetadata(
    name='NV_R_EMKSAJYS_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_EMKSAJYS
)

NV_R_EMKSAJYS_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_EMKSAJYS_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_EMKSAJYS_F_WQYWMRVP
)
NV_R_EMKSAJYS_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_EMKSAJYS_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_EMKSAJYS_F_WQYWMRVP
)

NV_R_EMKSAJYS_F_MLEOLJIS = FieldMetadata(
    name='NV_R_EMKSAJYS_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_EMKSAJYS
)

NV_R_EMKSAJYS_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_EMKSAJYS_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_EMKSAJYS_F_MLEOLJIS
)
NV_R_EMKSAJYS_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_EMKSAJYS_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_EMKSAJYS_F_MLEOLJIS
)

NV_R_CKUMFEPI = RegisterMetadata(
    name='NV_R_CKUMFEPI',
    address=0xa5a0,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_CKUMFEPI_F_CTBCFRUB = FieldMetadata(
    name='NV_R_CKUMFEPI_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_CKUMFEPI
)

NV_R_CKUMFEPI_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_CKUMFEPI_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_CKUMFEPI_F_CTBCFRUB
)

NV_R_CKUMFEPI_F_QVYAARWP = FieldMetadata(
    name='NV_R_CKUMFEPI_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_CKUMFEPI
)

NV_R_CKUMFEPI_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_CKUMFEPI_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_CKUMFEPI_F_QVYAARWP
)

NV_R_GHZBWDAT = RegisterMetadata(
    name='NV_R_GHZBWDAT',
    address=0xa590,
    zero_based=True
)

NV_R_GHZBWDAT_F_IKQZFICP = FieldMetadata(
    name='NV_R_GHZBWDAT_F_IKQZFICP',
    msb=11,
    lsb=0,
    register=NV_R_GHZBWDAT
)

NV_R_GHZBWDAT_F_CLQFSQFY = FieldMetadata(
    name='NV_R_GHZBWDAT_F_CLQFSQFY',
    msb=27,
    lsb=12,
    register=NV_R_GHZBWDAT
)

NV_R_GHZBWDAT_F_OJYQBJXU = FieldMetadata(
    name='NV_R_GHZBWDAT_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_GHZBWDAT
)

NV_R_GHZBWDAT_F_SYSBRQCF = FieldMetadata(
    name='NV_R_GHZBWDAT_F_SYSBRQCF',
    msb=31,
    lsb=28,
    register=NV_R_GHZBWDAT
)

NV_R_GHZBWDAT_F_SYSBRQCF_V_TZVRGSRL = ValueMetadata(
    name='NV_R_GHZBWDAT_F_SYSBRQCF_V_TZVRGSRL',
    value=1,
    field=NV_R_GHZBWDAT_F_SYSBRQCF
)

NV_R_MDRZYXMY = RegisterMetadata(
    name='NV_R_MDRZYXMY',
    address=0xa588,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_MDRZYXMY_F_CTBCFRUB = FieldMetadata(
    name='NV_R_MDRZYXMY_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_MDRZYXMY
)

NV_R_MDRZYXMY_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_MDRZYXMY_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_MDRZYXMY_F_CTBCFRUB
)

NV_R_MDRZYXMY_F_QVYAARWP = FieldMetadata(
    name='NV_R_MDRZYXMY_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_MDRZYXMY
)

NV_R_MDRZYXMY_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_MDRZYXMY_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_MDRZYXMY_F_QVYAARWP
)

NV_R_UDJQMCAC = RegisterMetadata(
    name='NV_R_UDJQMCAC',
    address=0xa584,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_UDJQMCAC_F_JLYSGAKE = FieldMetadata(
    name='NV_R_UDJQMCAC_F_JLYSGAKE',
    msb=0,
    lsb=0,
    register=NV_R_UDJQMCAC
)

NV_R_UDJQMCAC_F_JLYSGAKE_V_DOGBFDTH = ValueMetadata(
    name='NV_R_UDJQMCAC_F_JLYSGAKE_V_DOGBFDTH',
    value=0,
    field=NV_R_UDJQMCAC_F_JLYSGAKE
)
NV_R_UDJQMCAC_F_JLYSGAKE_V_PIUJQBKV = ValueMetadata(
    name='NV_R_UDJQMCAC_F_JLYSGAKE_V_PIUJQBKV',
    value=1,
    field=NV_R_UDJQMCAC_F_JLYSGAKE
)

NV_R_UDJQMCAC_F_OJXVVOEC = FieldMetadata(
    name='NV_R_UDJQMCAC_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_UDJQMCAC
)

NV_R_UDJQMCAC_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_UDJQMCAC_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_UDJQMCAC_F_OJXVVOEC
)
NV_R_UDJQMCAC_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_UDJQMCAC_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_UDJQMCAC_F_OJXVVOEC
)

NV_R_UDJQMCAC_F_VEOHPHHB = FieldMetadata(
    name='NV_R_UDJQMCAC_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_UDJQMCAC
)

NV_R_UDJQMCAC_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_UDJQMCAC_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_UDJQMCAC_F_VEOHPHHB
)
NV_R_UDJQMCAC_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_UDJQMCAC_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_UDJQMCAC_F_VEOHPHHB
)

NV_R_UDJQMCAC_F_KVYOKDXN = FieldMetadata(
    name='NV_R_UDJQMCAC_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_UDJQMCAC
)

NV_R_UDJQMCAC_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_UDJQMCAC_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_UDJQMCAC_F_KVYOKDXN
)
NV_R_UDJQMCAC_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_UDJQMCAC_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_UDJQMCAC_F_KVYOKDXN
)
NV_R_UDJQMCAC_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_UDJQMCAC_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_UDJQMCAC_F_KVYOKDXN
)

NV_R_UDJQMCAC_F_KWJKGHXW = FieldMetadata(
    name='NV_R_UDJQMCAC_F_KWJKGHXW',
    msb=1,
    lsb=1,
    register=NV_R_UDJQMCAC
)

NV_R_UDJQMCAC_F_KWJKGHXW_V_DOGBFDTH = ValueMetadata(
    name='NV_R_UDJQMCAC_F_KWJKGHXW_V_DOGBFDTH',
    value=0,
    field=NV_R_UDJQMCAC_F_KWJKGHXW
)
NV_R_UDJQMCAC_F_KWJKGHXW_V_PIUJQBKV = ValueMetadata(
    name='NV_R_UDJQMCAC_F_KWJKGHXW_V_PIUJQBKV',
    value=1,
    field=NV_R_UDJQMCAC_F_KWJKGHXW
)

NV_R_UDJQMCAC_F_WQYWMRVP = FieldMetadata(
    name='NV_R_UDJQMCAC_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_UDJQMCAC
)

NV_R_UDJQMCAC_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_UDJQMCAC_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_UDJQMCAC_F_WQYWMRVP
)
NV_R_UDJQMCAC_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_UDJQMCAC_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_UDJQMCAC_F_WQYWMRVP
)

NV_R_UDJQMCAC_F_MLEOLJIS = FieldMetadata(
    name='NV_R_UDJQMCAC_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_UDJQMCAC
)

NV_R_UDJQMCAC_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_UDJQMCAC_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_UDJQMCAC_F_MLEOLJIS
)
NV_R_UDJQMCAC_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_UDJQMCAC_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_UDJQMCAC_F_MLEOLJIS
)

NV_R_TPRPESMP = RegisterMetadata(
    name='NV_R_TPRPESMP',
    address=0xa58c,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_TPRPESMP_F_CTBCFRUB = FieldMetadata(
    name='NV_R_TPRPESMP_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_TPRPESMP
)

NV_R_TPRPESMP_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_TPRPESMP_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_TPRPESMP_F_CTBCFRUB
)

NV_R_TPRPESMP_F_QVYAARWP = FieldMetadata(
    name='NV_R_TPRPESMP_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_TPRPESMP
)

NV_R_TPRPESMP_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_TPRPESMP_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_TPRPESMP_F_QVYAARWP
)

NV_R_ACGALQVG = RegisterMetadata(
    name='NV_R_ACGALQVG',
    address=0xa57c,
    zero_based=True
)

NV_R_ACGALQVG_F_IKQZFICP = FieldMetadata(
    name='NV_R_ACGALQVG_F_IKQZFICP',
    msb=15,
    lsb=0,
    register=NV_R_ACGALQVG
)

NV_R_ACGALQVG_F_CLQFSQFY = FieldMetadata(
    name='NV_R_ACGALQVG_F_CLQFSQFY',
    msb=27,
    lsb=16,
    register=NV_R_ACGALQVG
)

NV_R_ACGALQVG_F_OJYQBJXU = FieldMetadata(
    name='NV_R_ACGALQVG_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_ACGALQVG
)

NV_R_ACGALQVG_F_SYSBRQCF = FieldMetadata(
    name='NV_R_ACGALQVG_F_SYSBRQCF',
    msb=31,
    lsb=28,
    register=NV_R_ACGALQVG
)

NV_R_ACGALQVG_F_SYSBRQCF_V_NUTEJVBM = ValueMetadata(
    name='NV_R_ACGALQVG_F_SYSBRQCF_V_NUTEJVBM',
    value=1,
    field=NV_R_ACGALQVG_F_SYSBRQCF
)
NV_R_ACGALQVG_F_SYSBRQCF_V_STRFAYDJ = ValueMetadata(
    name='NV_R_ACGALQVG_F_SYSBRQCF_V_STRFAYDJ',
    value=2,
    field=NV_R_ACGALQVG_F_SYSBRQCF
)

NV_R_OZJKYHQG = RegisterMetadata(
    name='NV_R_OZJKYHQG',
    address=0xa574,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_OZJKYHQG_F_CTBCFRUB = FieldMetadata(
    name='NV_R_OZJKYHQG_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_OZJKYHQG
)

NV_R_OZJKYHQG_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_OZJKYHQG_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_OZJKYHQG_F_CTBCFRUB
)

NV_R_OZJKYHQG_F_QVYAARWP = FieldMetadata(
    name='NV_R_OZJKYHQG_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_OZJKYHQG
)

NV_R_OZJKYHQG_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_OZJKYHQG_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_OZJKYHQG_F_QVYAARWP
)

NV_R_OBBTRPQV = RegisterMetadata(
    name='NV_R_OBBTRPQV',
    address=0xa570,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_OBBTRPQV_F_AKLEWUFG = FieldMetadata(
    name='NV_R_OBBTRPQV_F_AKLEWUFG',
    msb=0,
    lsb=0,
    register=NV_R_OBBTRPQV
)

NV_R_OBBTRPQV_F_AKLEWUFG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_OBBTRPQV_F_AKLEWUFG_V_DOGBFDTH',
    value=0,
    field=NV_R_OBBTRPQV_F_AKLEWUFG
)
NV_R_OBBTRPQV_F_AKLEWUFG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_OBBTRPQV_F_AKLEWUFG_V_PIUJQBKV',
    value=1,
    field=NV_R_OBBTRPQV_F_AKLEWUFG
)

NV_R_OBBTRPQV_F_BLAOEDCT = FieldMetadata(
    name='NV_R_OBBTRPQV_F_BLAOEDCT',
    msb=2,
    lsb=2,
    register=NV_R_OBBTRPQV
)

NV_R_OBBTRPQV_F_BLAOEDCT_V_DOGBFDTH = ValueMetadata(
    name='NV_R_OBBTRPQV_F_BLAOEDCT_V_DOGBFDTH',
    value=0,
    field=NV_R_OBBTRPQV_F_BLAOEDCT
)
NV_R_OBBTRPQV_F_BLAOEDCT_V_PIUJQBKV = ValueMetadata(
    name='NV_R_OBBTRPQV_F_BLAOEDCT_V_PIUJQBKV',
    value=1,
    field=NV_R_OBBTRPQV_F_BLAOEDCT
)

NV_R_OBBTRPQV_F_UVPLRDMG = FieldMetadata(
    name='NV_R_OBBTRPQV_F_UVPLRDMG',
    msb=0,
    lsb=0,
    register=NV_R_OBBTRPQV
)

NV_R_OBBTRPQV_F_UVPLRDMG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_OBBTRPQV_F_UVPLRDMG_V_DOGBFDTH',
    value=0,
    field=NV_R_OBBTRPQV_F_UVPLRDMG
)
NV_R_OBBTRPQV_F_UVPLRDMG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_OBBTRPQV_F_UVPLRDMG_V_PIUJQBKV',
    value=1,
    field=NV_R_OBBTRPQV_F_UVPLRDMG
)

NV_R_OBBTRPQV_F_OJXVVOEC = FieldMetadata(
    name='NV_R_OBBTRPQV_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_OBBTRPQV
)

NV_R_OBBTRPQV_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_OBBTRPQV_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_OBBTRPQV_F_OJXVVOEC
)
NV_R_OBBTRPQV_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_OBBTRPQV_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_OBBTRPQV_F_OJXVVOEC
)

NV_R_OBBTRPQV_F_VEOHPHHB = FieldMetadata(
    name='NV_R_OBBTRPQV_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_OBBTRPQV
)

NV_R_OBBTRPQV_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_OBBTRPQV_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_OBBTRPQV_F_VEOHPHHB
)
NV_R_OBBTRPQV_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_OBBTRPQV_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_OBBTRPQV_F_VEOHPHHB
)

NV_R_OBBTRPQV_F_KVYOKDXN = FieldMetadata(
    name='NV_R_OBBTRPQV_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_OBBTRPQV
)

NV_R_OBBTRPQV_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_OBBTRPQV_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_OBBTRPQV_F_KVYOKDXN
)
NV_R_OBBTRPQV_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_OBBTRPQV_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_OBBTRPQV_F_KVYOKDXN
)
NV_R_OBBTRPQV_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_OBBTRPQV_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_OBBTRPQV_F_KVYOKDXN
)

NV_R_OBBTRPQV_F_CRVGSVEG = FieldMetadata(
    name='NV_R_OBBTRPQV_F_CRVGSVEG',
    msb=1,
    lsb=1,
    register=NV_R_OBBTRPQV
)

NV_R_OBBTRPQV_F_CRVGSVEG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_OBBTRPQV_F_CRVGSVEG_V_DOGBFDTH',
    value=0,
    field=NV_R_OBBTRPQV_F_CRVGSVEG
)
NV_R_OBBTRPQV_F_CRVGSVEG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_OBBTRPQV_F_CRVGSVEG_V_PIUJQBKV',
    value=1,
    field=NV_R_OBBTRPQV_F_CRVGSVEG
)

NV_R_OBBTRPQV_F_IPXTTKZM = FieldMetadata(
    name='NV_R_OBBTRPQV_F_IPXTTKZM',
    msb=3,
    lsb=3,
    register=NV_R_OBBTRPQV
)

NV_R_OBBTRPQV_F_IPXTTKZM_V_DOGBFDTH = ValueMetadata(
    name='NV_R_OBBTRPQV_F_IPXTTKZM_V_DOGBFDTH',
    value=0,
    field=NV_R_OBBTRPQV_F_IPXTTKZM
)
NV_R_OBBTRPQV_F_IPXTTKZM_V_PIUJQBKV = ValueMetadata(
    name='NV_R_OBBTRPQV_F_IPXTTKZM_V_PIUJQBKV',
    value=1,
    field=NV_R_OBBTRPQV_F_IPXTTKZM
)

NV_R_OBBTRPQV_F_QBNGJNXY = FieldMetadata(
    name='NV_R_OBBTRPQV_F_QBNGJNXY',
    msb=1,
    lsb=1,
    register=NV_R_OBBTRPQV
)

NV_R_OBBTRPQV_F_QBNGJNXY_V_DOGBFDTH = ValueMetadata(
    name='NV_R_OBBTRPQV_F_QBNGJNXY_V_DOGBFDTH',
    value=0,
    field=NV_R_OBBTRPQV_F_QBNGJNXY
)
NV_R_OBBTRPQV_F_QBNGJNXY_V_PIUJQBKV = ValueMetadata(
    name='NV_R_OBBTRPQV_F_QBNGJNXY_V_PIUJQBKV',
    value=1,
    field=NV_R_OBBTRPQV_F_QBNGJNXY
)

NV_R_OBBTRPQV_F_WQYWMRVP = FieldMetadata(
    name='NV_R_OBBTRPQV_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_OBBTRPQV
)

NV_R_OBBTRPQV_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_OBBTRPQV_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_OBBTRPQV_F_WQYWMRVP
)
NV_R_OBBTRPQV_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_OBBTRPQV_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_OBBTRPQV_F_WQYWMRVP
)

NV_R_OBBTRPQV_F_MLEOLJIS = FieldMetadata(
    name='NV_R_OBBTRPQV_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_OBBTRPQV
)

NV_R_OBBTRPQV_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_OBBTRPQV_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_OBBTRPQV_F_MLEOLJIS
)
NV_R_OBBTRPQV_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_OBBTRPQV_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_OBBTRPQV_F_MLEOLJIS
)

NV_R_KHWTVBQG = RegisterMetadata(
    name='NV_R_KHWTVBQG',
    address=0xa578,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_KHWTVBQG_F_CTBCFRUB = FieldMetadata(
    name='NV_R_KHWTVBQG_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_KHWTVBQG
)

NV_R_KHWTVBQG_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_KHWTVBQG_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_KHWTVBQG_F_CTBCFRUB
)

NV_R_KHWTVBQG_F_QVYAARWP = FieldMetadata(
    name='NV_R_KHWTVBQG_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_KHWTVBQG
)

NV_R_KHWTVBQG_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_KHWTVBQG_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_KHWTVBQG_F_QVYAARWP
)

