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
from gpu.regs.gb100.hubmmu_int import (
    NV_R_JSOWLDXY,
    NV_R_JSOWLDXY_F_IKQZFICP,
    NV_R_JSOWLDXY_F_CLQFSQFY,
    NV_R_JSOWLDXY_F_OJYQBJXU,
    NV_R_JSOWLDXY_F_SYSBRQCF,
    NV_R_JSOWLDXY_F_SYSBRQCF_V_BVGPLGJU,
    NV_R_JSOWLDXY_F_SYSBRQCF_V_DRSZNSAZ,
    NV_R_AWKJAKDO,
    NV_R_AWKJAKDO_F_CTBCFRUB,
    NV_R_AWKJAKDO_F_CTBCFRUB_V_GQHYEKHS,
    NV_R_AWKJAKDO_F_QVYAARWP,
    NV_R_AWKJAKDO_F_QVYAARWP_V_GQHYEKHS,
    NV_R_EMKSAJYS,
    NV_R_EMKSAJYS_F_QAUPMBGQ,
    NV_R_EMKSAJYS_F_QAUPMBGQ_V_DOGBFDTH,
    NV_R_EMKSAJYS_F_QAUPMBGQ_V_PIUJQBKV,
    NV_R_EMKSAJYS_F_CPTNFTON,
    NV_R_EMKSAJYS_F_CPTNFTON_V_DOGBFDTH,
    NV_R_EMKSAJYS_F_CPTNFTON_V_PIUJQBKV,
    NV_R_EMKSAJYS_F_OJXVVOEC,
    NV_R_EMKSAJYS_F_OJXVVOEC_V_DOGBFDTH,
    NV_R_EMKSAJYS_F_OJXVVOEC_V_PIUJQBKV,
    NV_R_EMKSAJYS_F_VEOHPHHB,
    NV_R_EMKSAJYS_F_VEOHPHHB_V_DOGBFDTH,
    NV_R_EMKSAJYS_F_VEOHPHHB_V_PIUJQBKV,
    NV_R_EMKSAJYS_F_KVYOKDXN,
    NV_R_EMKSAJYS_F_KVYOKDXN_V_ACRDDIMM,
    NV_R_EMKSAJYS_F_KVYOKDXN_V_GQHYEKHS,
    NV_R_EMKSAJYS_F_KVYOKDXN_V_CIHFEZTE,
    NV_R_EMKSAJYS_F_LEDOESZC,
    NV_R_EMKSAJYS_F_LEDOESZC_V_DOGBFDTH,
    NV_R_EMKSAJYS_F_LEDOESZC_V_PIUJQBKV,
    NV_R_EMKSAJYS_F_PIJLJNAU,
    NV_R_EMKSAJYS_F_PIJLJNAU_V_DOGBFDTH,
    NV_R_EMKSAJYS_F_PIJLJNAU_V_PIUJQBKV,
    NV_R_EMKSAJYS_F_WQYWMRVP,
    NV_R_EMKSAJYS_F_WQYWMRVP_V_DOGBFDTH,
    NV_R_EMKSAJYS_F_WQYWMRVP_V_PIUJQBKV,
    NV_R_EMKSAJYS_F_MLEOLJIS,
    NV_R_EMKSAJYS_F_MLEOLJIS_V_DOGBFDTH,
    NV_R_EMKSAJYS_F_MLEOLJIS_V_PIUJQBKV,
    NV_R_CKUMFEPI,
    NV_R_CKUMFEPI_F_CTBCFRUB,
    NV_R_CKUMFEPI_F_CTBCFRUB_V_GQHYEKHS,
    NV_R_CKUMFEPI_F_QVYAARWP,
    NV_R_CKUMFEPI_F_QVYAARWP_V_GQHYEKHS,
    NV_R_GHZBWDAT,
    NV_R_GHZBWDAT_F_IKQZFICP,
    NV_R_GHZBWDAT_F_CLQFSQFY,
    NV_R_GHZBWDAT_F_OJYQBJXU,
    NV_R_GHZBWDAT_F_SYSBRQCF,
    NV_R_GHZBWDAT_F_SYSBRQCF_V_TZVRGSRL,
    NV_R_MDRZYXMY,
    NV_R_MDRZYXMY_F_CTBCFRUB,
    NV_R_MDRZYXMY_F_CTBCFRUB_V_GQHYEKHS,
    NV_R_MDRZYXMY_F_QVYAARWP,
    NV_R_MDRZYXMY_F_QVYAARWP_V_GQHYEKHS,
    NV_R_UDJQMCAC,
    NV_R_UDJQMCAC_F_JLYSGAKE,
    NV_R_UDJQMCAC_F_JLYSGAKE_V_DOGBFDTH,
    NV_R_UDJQMCAC_F_JLYSGAKE_V_PIUJQBKV,
    NV_R_UDJQMCAC_F_OJXVVOEC,
    NV_R_UDJQMCAC_F_OJXVVOEC_V_DOGBFDTH,
    NV_R_UDJQMCAC_F_OJXVVOEC_V_PIUJQBKV,
    NV_R_UDJQMCAC_F_VEOHPHHB,
    NV_R_UDJQMCAC_F_VEOHPHHB_V_DOGBFDTH,
    NV_R_UDJQMCAC_F_VEOHPHHB_V_PIUJQBKV,
    NV_R_UDJQMCAC_F_KVYOKDXN,
    NV_R_UDJQMCAC_F_KVYOKDXN_V_ACRDDIMM,
    NV_R_UDJQMCAC_F_KVYOKDXN_V_GQHYEKHS,
    NV_R_UDJQMCAC_F_KVYOKDXN_V_CIHFEZTE,
    NV_R_UDJQMCAC_F_KWJKGHXW,
    NV_R_UDJQMCAC_F_KWJKGHXW_V_DOGBFDTH,
    NV_R_UDJQMCAC_F_KWJKGHXW_V_PIUJQBKV,
    NV_R_UDJQMCAC_F_WQYWMRVP,
    NV_R_UDJQMCAC_F_WQYWMRVP_V_DOGBFDTH,
    NV_R_UDJQMCAC_F_WQYWMRVP_V_PIUJQBKV,
    NV_R_UDJQMCAC_F_MLEOLJIS,
    NV_R_UDJQMCAC_F_MLEOLJIS_V_DOGBFDTH,
    NV_R_UDJQMCAC_F_MLEOLJIS_V_PIUJQBKV,
    NV_R_TPRPESMP,
    NV_R_TPRPESMP_F_CTBCFRUB,
    NV_R_TPRPESMP_F_CTBCFRUB_V_GQHYEKHS,
    NV_R_TPRPESMP_F_QVYAARWP,
    NV_R_TPRPESMP_F_QVYAARWP_V_GQHYEKHS,
    NV_R_ACGALQVG,
    NV_R_ACGALQVG_F_IKQZFICP,
    NV_R_ACGALQVG_F_CLQFSQFY,
    NV_R_ACGALQVG_F_OJYQBJXU,
    NV_R_ACGALQVG_F_SYSBRQCF,
    NV_R_ACGALQVG_F_SYSBRQCF_V_NUTEJVBM,
    NV_R_ACGALQVG_F_SYSBRQCF_V_STRFAYDJ,
    NV_R_OZJKYHQG,
    NV_R_OZJKYHQG_F_CTBCFRUB,
    NV_R_OZJKYHQG_F_CTBCFRUB_V_GQHYEKHS,
    NV_R_OZJKYHQG_F_QVYAARWP,
    NV_R_OZJKYHQG_F_QVYAARWP_V_GQHYEKHS,
    NV_R_OBBTRPQV,
    NV_R_OBBTRPQV_F_AKLEWUFG,
    NV_R_OBBTRPQV_F_AKLEWUFG_V_DOGBFDTH,
    NV_R_OBBTRPQV_F_AKLEWUFG_V_PIUJQBKV,
    NV_R_OBBTRPQV_F_BLAOEDCT,
    NV_R_OBBTRPQV_F_BLAOEDCT_V_DOGBFDTH,
    NV_R_OBBTRPQV_F_BLAOEDCT_V_PIUJQBKV,
    NV_R_OBBTRPQV_F_UVPLRDMG,
    NV_R_OBBTRPQV_F_UVPLRDMG_V_DOGBFDTH,
    NV_R_OBBTRPQV_F_UVPLRDMG_V_PIUJQBKV,
    NV_R_OBBTRPQV_F_OJXVVOEC,
    NV_R_OBBTRPQV_F_OJXVVOEC_V_DOGBFDTH,
    NV_R_OBBTRPQV_F_OJXVVOEC_V_PIUJQBKV,
    NV_R_OBBTRPQV_F_VEOHPHHB,
    NV_R_OBBTRPQV_F_VEOHPHHB_V_DOGBFDTH,
    NV_R_OBBTRPQV_F_VEOHPHHB_V_PIUJQBKV,
    NV_R_OBBTRPQV_F_KVYOKDXN,
    NV_R_OBBTRPQV_F_KVYOKDXN_V_ACRDDIMM,
    NV_R_OBBTRPQV_F_KVYOKDXN_V_GQHYEKHS,
    NV_R_OBBTRPQV_F_KVYOKDXN_V_CIHFEZTE,
    NV_R_OBBTRPQV_F_CRVGSVEG,
    NV_R_OBBTRPQV_F_CRVGSVEG_V_DOGBFDTH,
    NV_R_OBBTRPQV_F_CRVGSVEG_V_PIUJQBKV,
    NV_R_OBBTRPQV_F_IPXTTKZM,
    NV_R_OBBTRPQV_F_IPXTTKZM_V_DOGBFDTH,
    NV_R_OBBTRPQV_F_IPXTTKZM_V_PIUJQBKV,
    NV_R_OBBTRPQV_F_QBNGJNXY,
    NV_R_OBBTRPQV_F_QBNGJNXY_V_DOGBFDTH,
    NV_R_OBBTRPQV_F_QBNGJNXY_V_PIUJQBKV,
    NV_R_OBBTRPQV_F_WQYWMRVP,
    NV_R_OBBTRPQV_F_WQYWMRVP_V_DOGBFDTH,
    NV_R_OBBTRPQV_F_WQYWMRVP_V_PIUJQBKV,
    NV_R_OBBTRPQV_F_MLEOLJIS,
    NV_R_OBBTRPQV_F_MLEOLJIS_V_DOGBFDTH,
    NV_R_OBBTRPQV_F_MLEOLJIS_V_PIUJQBKV,
    NV_R_KHWTVBQG,
    NV_R_KHWTVBQG_F_CTBCFRUB,
    NV_R_KHWTVBQG_F_CTBCFRUB_V_GQHYEKHS,
    NV_R_KHWTVBQG_F_QVYAARWP,
    NV_R_KHWTVBQG_F_QVYAARWP_V_GQHYEKHS,
)

# Register definitions
NV_R_RBERYCUQ = RegisterMetadata(
    name='NV_R_RBERYCUQ',
    address=0xa354,
    zero_based=True
)

NV_R_RBERYCUQ_F_IKQZFICP = FieldMetadata(
    name='NV_R_RBERYCUQ_F_IKQZFICP',
    msb=15,
    lsb=0,
    register=NV_R_RBERYCUQ
)

NV_R_RBERYCUQ_F_CLQFSQFY = FieldMetadata(
    name='NV_R_RBERYCUQ_F_CLQFSQFY',
    msb=27,
    lsb=16,
    register=NV_R_RBERYCUQ
)

NV_R_RBERYCUQ_F_OJYQBJXU = FieldMetadata(
    name='NV_R_RBERYCUQ_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_RBERYCUQ
)

NV_R_RBERYCUQ_F_SYSBRQCF = FieldMetadata(
    name='NV_R_RBERYCUQ_F_SYSBRQCF',
    msb=31,
    lsb=28,
    register=NV_R_RBERYCUQ
)

NV_R_RBERYCUQ_F_SYSBRQCF_V_EZDDFVAS = ValueMetadata(
    name='NV_R_RBERYCUQ_F_SYSBRQCF_V_EZDDFVAS',
    value=1,
    field=NV_R_RBERYCUQ_F_SYSBRQCF
)

NV_R_JIPALPWD = RegisterMetadata(
    name='NV_R_JIPALPWD',
    address=0xa34c,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_JIPALPWD_F_CTBCFRUB = FieldMetadata(
    name='NV_R_JIPALPWD_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_JIPALPWD
)

NV_R_JIPALPWD_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_JIPALPWD_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_JIPALPWD_F_CTBCFRUB
)

NV_R_JIPALPWD_F_QVYAARWP = FieldMetadata(
    name='NV_R_JIPALPWD_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_JIPALPWD
)

NV_R_JIPALPWD_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_JIPALPWD_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_JIPALPWD_F_QVYAARWP
)

NV_R_UKFHOXYN = RegisterMetadata(
    name='NV_R_UKFHOXYN',
    address=0xa348,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_UKFHOXYN_F_UVPLRDMG = FieldMetadata(
    name='NV_R_UKFHOXYN_F_UVPLRDMG',
    msb=0,
    lsb=0,
    register=NV_R_UKFHOXYN
)

NV_R_UKFHOXYN_F_UVPLRDMG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_UKFHOXYN_F_UVPLRDMG_V_DOGBFDTH',
    value=0,
    field=NV_R_UKFHOXYN_F_UVPLRDMG
)
NV_R_UKFHOXYN_F_UVPLRDMG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_UKFHOXYN_F_UVPLRDMG_V_PIUJQBKV',
    value=1,
    field=NV_R_UKFHOXYN_F_UVPLRDMG
)

NV_R_UKFHOXYN_F_OJXVVOEC = FieldMetadata(
    name='NV_R_UKFHOXYN_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_UKFHOXYN
)

NV_R_UKFHOXYN_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_UKFHOXYN_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_UKFHOXYN_F_OJXVVOEC
)
NV_R_UKFHOXYN_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_UKFHOXYN_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_UKFHOXYN_F_OJXVVOEC
)

NV_R_UKFHOXYN_F_VEOHPHHB = FieldMetadata(
    name='NV_R_UKFHOXYN_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_UKFHOXYN
)

NV_R_UKFHOXYN_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_UKFHOXYN_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_UKFHOXYN_F_VEOHPHHB
)
NV_R_UKFHOXYN_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_UKFHOXYN_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_UKFHOXYN_F_VEOHPHHB
)

NV_R_UKFHOXYN_F_KVYOKDXN = FieldMetadata(
    name='NV_R_UKFHOXYN_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_UKFHOXYN
)

NV_R_UKFHOXYN_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_UKFHOXYN_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_UKFHOXYN_F_KVYOKDXN
)
NV_R_UKFHOXYN_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_UKFHOXYN_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_UKFHOXYN_F_KVYOKDXN
)
NV_R_UKFHOXYN_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_UKFHOXYN_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_UKFHOXYN_F_KVYOKDXN
)

NV_R_UKFHOXYN_F_QBNGJNXY = FieldMetadata(
    name='NV_R_UKFHOXYN_F_QBNGJNXY',
    msb=1,
    lsb=1,
    register=NV_R_UKFHOXYN
)

NV_R_UKFHOXYN_F_QBNGJNXY_V_DOGBFDTH = ValueMetadata(
    name='NV_R_UKFHOXYN_F_QBNGJNXY_V_DOGBFDTH',
    value=0,
    field=NV_R_UKFHOXYN_F_QBNGJNXY
)
NV_R_UKFHOXYN_F_QBNGJNXY_V_PIUJQBKV = ValueMetadata(
    name='NV_R_UKFHOXYN_F_QBNGJNXY_V_PIUJQBKV',
    value=1,
    field=NV_R_UKFHOXYN_F_QBNGJNXY
)

NV_R_UKFHOXYN_F_WQYWMRVP = FieldMetadata(
    name='NV_R_UKFHOXYN_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_UKFHOXYN
)

NV_R_UKFHOXYN_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_UKFHOXYN_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_UKFHOXYN_F_WQYWMRVP
)
NV_R_UKFHOXYN_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_UKFHOXYN_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_UKFHOXYN_F_WQYWMRVP
)

NV_R_UKFHOXYN_F_MLEOLJIS = FieldMetadata(
    name='NV_R_UKFHOXYN_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_UKFHOXYN
)

NV_R_UKFHOXYN_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_UKFHOXYN_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_UKFHOXYN_F_MLEOLJIS
)
NV_R_UKFHOXYN_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_UKFHOXYN_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_UKFHOXYN_F_MLEOLJIS
)

NV_R_CGHXJOSE = RegisterMetadata(
    name='NV_R_CGHXJOSE',
    address=0xa350,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_CGHXJOSE_F_CTBCFRUB = FieldMetadata(
    name='NV_R_CGHXJOSE_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_CGHXJOSE
)

NV_R_CGHXJOSE_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_CGHXJOSE_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_CGHXJOSE_F_CTBCFRUB
)

NV_R_CGHXJOSE_F_QVYAARWP = FieldMetadata(
    name='NV_R_CGHXJOSE_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_CGHXJOSE
)

NV_R_CGHXJOSE_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_CGHXJOSE_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_CGHXJOSE_F_QVYAARWP
)

NV_R_SUQWCWUA = RegisterMetadata(
    name='NV_R_SUQWCWUA',
    address=0xa368,
    zero_based=True
)

NV_R_SUQWCWUA_F_IKQZFICP = FieldMetadata(
    name='NV_R_SUQWCWUA_F_IKQZFICP',
    msb=15,
    lsb=0,
    register=NV_R_SUQWCWUA
)

NV_R_SUQWCWUA_F_CLQFSQFY = FieldMetadata(
    name='NV_R_SUQWCWUA_F_CLQFSQFY',
    msb=27,
    lsb=16,
    register=NV_R_SUQWCWUA
)

NV_R_SUQWCWUA_F_OJYQBJXU = FieldMetadata(
    name='NV_R_SUQWCWUA_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_SUQWCWUA
)

NV_R_SUQWCWUA_F_SYSBRQCF = FieldMetadata(
    name='NV_R_SUQWCWUA_F_SYSBRQCF',
    msb=31,
    lsb=28,
    register=NV_R_SUQWCWUA
)

NV_R_SUQWCWUA_F_SYSBRQCF_V_EZDDFVAS = ValueMetadata(
    name='NV_R_SUQWCWUA_F_SYSBRQCF_V_EZDDFVAS',
    value=1,
    field=NV_R_SUQWCWUA_F_SYSBRQCF
)

NV_R_WNVOCQAJ = RegisterMetadata(
    name='NV_R_WNVOCQAJ',
    address=0xa360,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_WNVOCQAJ_F_CTBCFRUB = FieldMetadata(
    name='NV_R_WNVOCQAJ_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_WNVOCQAJ
)

NV_R_WNVOCQAJ_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_WNVOCQAJ_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_WNVOCQAJ_F_CTBCFRUB
)

NV_R_WNVOCQAJ_F_QVYAARWP = FieldMetadata(
    name='NV_R_WNVOCQAJ_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_WNVOCQAJ
)

NV_R_WNVOCQAJ_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_WNVOCQAJ_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_WNVOCQAJ_F_QVYAARWP
)

NV_R_ZMPSWYVE = RegisterMetadata(
    name='NV_R_ZMPSWYVE',
    address=0xa35c,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_ZMPSWYVE_F_UVPLRDMG = FieldMetadata(
    name='NV_R_ZMPSWYVE_F_UVPLRDMG',
    msb=0,
    lsb=0,
    register=NV_R_ZMPSWYVE
)

NV_R_ZMPSWYVE_F_UVPLRDMG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ZMPSWYVE_F_UVPLRDMG_V_DOGBFDTH',
    value=0,
    field=NV_R_ZMPSWYVE_F_UVPLRDMG
)
NV_R_ZMPSWYVE_F_UVPLRDMG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ZMPSWYVE_F_UVPLRDMG_V_PIUJQBKV',
    value=1,
    field=NV_R_ZMPSWYVE_F_UVPLRDMG
)

NV_R_ZMPSWYVE_F_OJXVVOEC = FieldMetadata(
    name='NV_R_ZMPSWYVE_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_ZMPSWYVE
)

NV_R_ZMPSWYVE_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ZMPSWYVE_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_ZMPSWYVE_F_OJXVVOEC
)
NV_R_ZMPSWYVE_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ZMPSWYVE_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_ZMPSWYVE_F_OJXVVOEC
)

NV_R_ZMPSWYVE_F_VEOHPHHB = FieldMetadata(
    name='NV_R_ZMPSWYVE_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_ZMPSWYVE
)

NV_R_ZMPSWYVE_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ZMPSWYVE_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_ZMPSWYVE_F_VEOHPHHB
)
NV_R_ZMPSWYVE_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ZMPSWYVE_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_ZMPSWYVE_F_VEOHPHHB
)

NV_R_ZMPSWYVE_F_KVYOKDXN = FieldMetadata(
    name='NV_R_ZMPSWYVE_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_ZMPSWYVE
)

NV_R_ZMPSWYVE_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_ZMPSWYVE_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_ZMPSWYVE_F_KVYOKDXN
)
NV_R_ZMPSWYVE_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_ZMPSWYVE_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_ZMPSWYVE_F_KVYOKDXN
)
NV_R_ZMPSWYVE_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_ZMPSWYVE_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_ZMPSWYVE_F_KVYOKDXN
)

NV_R_ZMPSWYVE_F_QBNGJNXY = FieldMetadata(
    name='NV_R_ZMPSWYVE_F_QBNGJNXY',
    msb=1,
    lsb=1,
    register=NV_R_ZMPSWYVE
)

NV_R_ZMPSWYVE_F_QBNGJNXY_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ZMPSWYVE_F_QBNGJNXY_V_DOGBFDTH',
    value=0,
    field=NV_R_ZMPSWYVE_F_QBNGJNXY
)
NV_R_ZMPSWYVE_F_QBNGJNXY_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ZMPSWYVE_F_QBNGJNXY_V_PIUJQBKV',
    value=1,
    field=NV_R_ZMPSWYVE_F_QBNGJNXY
)

NV_R_ZMPSWYVE_F_WQYWMRVP = FieldMetadata(
    name='NV_R_ZMPSWYVE_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_ZMPSWYVE
)

NV_R_ZMPSWYVE_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ZMPSWYVE_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_ZMPSWYVE_F_WQYWMRVP
)
NV_R_ZMPSWYVE_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ZMPSWYVE_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_ZMPSWYVE_F_WQYWMRVP
)

NV_R_ZMPSWYVE_F_MLEOLJIS = FieldMetadata(
    name='NV_R_ZMPSWYVE_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_ZMPSWYVE
)

NV_R_ZMPSWYVE_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ZMPSWYVE_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_ZMPSWYVE_F_MLEOLJIS
)
NV_R_ZMPSWYVE_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ZMPSWYVE_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_ZMPSWYVE_F_MLEOLJIS
)

NV_R_YXFCWNYE = RegisterMetadata(
    name='NV_R_YXFCWNYE',
    address=0xa364,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_YXFCWNYE_F_CTBCFRUB = FieldMetadata(
    name='NV_R_YXFCWNYE_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_YXFCWNYE
)

NV_R_YXFCWNYE_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_YXFCWNYE_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_YXFCWNYE_F_CTBCFRUB
)

NV_R_YXFCWNYE_F_QVYAARWP = FieldMetadata(
    name='NV_R_YXFCWNYE_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_YXFCWNYE
)

NV_R_YXFCWNYE_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_YXFCWNYE_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_YXFCWNYE_F_QVYAARWP
)

NV_R_HELGJAON = RegisterMetadata(
    name='NV_R_HELGJAON',
    address=0xa37c,
    zero_based=True
)

NV_R_HELGJAON_F_IKQZFICP = FieldMetadata(
    name='NV_R_HELGJAON_F_IKQZFICP',
    msb=15,
    lsb=0,
    register=NV_R_HELGJAON
)

NV_R_HELGJAON_F_CLQFSQFY = FieldMetadata(
    name='NV_R_HELGJAON_F_CLQFSQFY',
    msb=27,
    lsb=16,
    register=NV_R_HELGJAON
)

NV_R_HELGJAON_F_OJYQBJXU = FieldMetadata(
    name='NV_R_HELGJAON_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_HELGJAON
)

NV_R_HELGJAON_F_SYSBRQCF = FieldMetadata(
    name='NV_R_HELGJAON_F_SYSBRQCF',
    msb=31,
    lsb=28,
    register=NV_R_HELGJAON
)

NV_R_HELGJAON_F_SYSBRQCF_V_EZDDFVAS = ValueMetadata(
    name='NV_R_HELGJAON_F_SYSBRQCF_V_EZDDFVAS',
    value=1,
    field=NV_R_HELGJAON_F_SYSBRQCF
)

NV_R_PXQIJKMJ = RegisterMetadata(
    name='NV_R_PXQIJKMJ',
    address=0xa374,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_PXQIJKMJ_F_CTBCFRUB = FieldMetadata(
    name='NV_R_PXQIJKMJ_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_PXQIJKMJ
)

NV_R_PXQIJKMJ_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_PXQIJKMJ_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_PXQIJKMJ_F_CTBCFRUB
)

NV_R_PXQIJKMJ_F_QVYAARWP = FieldMetadata(
    name='NV_R_PXQIJKMJ_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_PXQIJKMJ
)

NV_R_PXQIJKMJ_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_PXQIJKMJ_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_PXQIJKMJ_F_QVYAARWP
)

NV_R_LANOIZYM = RegisterMetadata(
    name='NV_R_LANOIZYM',
    address=0xa370,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_LANOIZYM_F_UVPLRDMG = FieldMetadata(
    name='NV_R_LANOIZYM_F_UVPLRDMG',
    msb=0,
    lsb=0,
    register=NV_R_LANOIZYM
)

NV_R_LANOIZYM_F_UVPLRDMG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_LANOIZYM_F_UVPLRDMG_V_DOGBFDTH',
    value=0,
    field=NV_R_LANOIZYM_F_UVPLRDMG
)
NV_R_LANOIZYM_F_UVPLRDMG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_LANOIZYM_F_UVPLRDMG_V_PIUJQBKV',
    value=1,
    field=NV_R_LANOIZYM_F_UVPLRDMG
)

NV_R_LANOIZYM_F_OJXVVOEC = FieldMetadata(
    name='NV_R_LANOIZYM_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_LANOIZYM
)

NV_R_LANOIZYM_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_LANOIZYM_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_LANOIZYM_F_OJXVVOEC
)
NV_R_LANOIZYM_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_LANOIZYM_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_LANOIZYM_F_OJXVVOEC
)

NV_R_LANOIZYM_F_VEOHPHHB = FieldMetadata(
    name='NV_R_LANOIZYM_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_LANOIZYM
)

NV_R_LANOIZYM_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_LANOIZYM_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_LANOIZYM_F_VEOHPHHB
)
NV_R_LANOIZYM_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_LANOIZYM_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_LANOIZYM_F_VEOHPHHB
)

NV_R_LANOIZYM_F_KVYOKDXN = FieldMetadata(
    name='NV_R_LANOIZYM_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_LANOIZYM
)

NV_R_LANOIZYM_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_LANOIZYM_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_LANOIZYM_F_KVYOKDXN
)
NV_R_LANOIZYM_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LANOIZYM_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_LANOIZYM_F_KVYOKDXN
)
NV_R_LANOIZYM_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_LANOIZYM_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_LANOIZYM_F_KVYOKDXN
)

NV_R_LANOIZYM_F_QBNGJNXY = FieldMetadata(
    name='NV_R_LANOIZYM_F_QBNGJNXY',
    msb=1,
    lsb=1,
    register=NV_R_LANOIZYM
)

NV_R_LANOIZYM_F_QBNGJNXY_V_DOGBFDTH = ValueMetadata(
    name='NV_R_LANOIZYM_F_QBNGJNXY_V_DOGBFDTH',
    value=0,
    field=NV_R_LANOIZYM_F_QBNGJNXY
)
NV_R_LANOIZYM_F_QBNGJNXY_V_PIUJQBKV = ValueMetadata(
    name='NV_R_LANOIZYM_F_QBNGJNXY_V_PIUJQBKV',
    value=1,
    field=NV_R_LANOIZYM_F_QBNGJNXY
)

NV_R_LANOIZYM_F_WQYWMRVP = FieldMetadata(
    name='NV_R_LANOIZYM_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_LANOIZYM
)

NV_R_LANOIZYM_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_LANOIZYM_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_LANOIZYM_F_WQYWMRVP
)
NV_R_LANOIZYM_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_LANOIZYM_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_LANOIZYM_F_WQYWMRVP
)

NV_R_LANOIZYM_F_MLEOLJIS = FieldMetadata(
    name='NV_R_LANOIZYM_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_LANOIZYM
)

NV_R_LANOIZYM_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_LANOIZYM_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_LANOIZYM_F_MLEOLJIS
)
NV_R_LANOIZYM_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_LANOIZYM_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_LANOIZYM_F_MLEOLJIS
)

NV_R_RPKAYUKR = RegisterMetadata(
    name='NV_R_RPKAYUKR',
    address=0xa378,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_RPKAYUKR_F_CTBCFRUB = FieldMetadata(
    name='NV_R_RPKAYUKR_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_RPKAYUKR
)

NV_R_RPKAYUKR_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_RPKAYUKR_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_RPKAYUKR_F_CTBCFRUB
)

NV_R_RPKAYUKR_F_QVYAARWP = FieldMetadata(
    name='NV_R_RPKAYUKR_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_RPKAYUKR
)

NV_R_RPKAYUKR_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_RPKAYUKR_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_RPKAYUKR_F_QVYAARWP
)

