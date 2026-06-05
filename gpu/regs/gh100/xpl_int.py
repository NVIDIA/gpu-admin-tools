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
NV_R_BNNSDDXF = RegisterMetadata(
    name='NV_R_BNNSDDXF',
    address=0xa54,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_BNNSDDXF_F_AKTRDZCQ = FieldMetadata(
    name='NV_R_BNNSDDXF_F_AKTRDZCQ',
    msb=15,
    lsb=0,
    register=NV_R_BNNSDDXF
)

NV_R_BNNSDDXF_F_AKTRDZCQ_V_GQHYEKHS = ValueMetadata(
    name='NV_R_BNNSDDXF_F_AKTRDZCQ_V_GQHYEKHS',
    value=0,
    field=NV_R_BNNSDDXF_F_AKTRDZCQ
)

NV_R_BNNSDDXF_F_ANILTANB = FieldMetadata(
    name='NV_R_BNNSDDXF_F_ANILTANB',
    msb=31,
    lsb=16,
    register=NV_R_BNNSDDXF
)

NV_R_BNNSDDXF_F_ANILTANB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_BNNSDDXF_F_ANILTANB_V_GQHYEKHS',
    value=0,
    field=NV_R_BNNSDDXF_F_ANILTANB
)

NV_R_XZFMHPDU = RegisterMetadata(
    name='NV_R_XZFMHPDU',
    address=0xa58,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_XZFMHPDU_F_AKTRDZCQ = FieldMetadata(
    name='NV_R_XZFMHPDU_F_AKTRDZCQ',
    msb=15,
    lsb=0,
    register=NV_R_XZFMHPDU
)

NV_R_XZFMHPDU_F_AKTRDZCQ_V_GQHYEKHS = ValueMetadata(
    name='NV_R_XZFMHPDU_F_AKTRDZCQ_V_GQHYEKHS',
    value=0,
    field=NV_R_XZFMHPDU_F_AKTRDZCQ
)

NV_R_XZFMHPDU_F_ANILTANB = FieldMetadata(
    name='NV_R_XZFMHPDU_F_ANILTANB',
    msb=31,
    lsb=16,
    register=NV_R_XZFMHPDU
)

NV_R_XZFMHPDU_F_ANILTANB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_XZFMHPDU_F_ANILTANB_V_GQHYEKHS',
    value=0,
    field=NV_R_XZFMHPDU_F_ANILTANB
)

NV_R_NWMGSRVP = RegisterMetadata(
    name='NV_R_NWMGSRVP',
    address=0xa50,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_NWMGSRVP_F_MSZTXVYR = FieldMetadata(
    name='NV_R_NWMGSRVP_F_MSZTXVYR',
    msb=0,
    lsb=0,
    register=NV_R_NWMGSRVP
)

NV_R_NWMGSRVP_F_MSZTXVYR_V_AABSSBNA = ValueMetadata(
    name='NV_R_NWMGSRVP_F_MSZTXVYR_V_AABSSBNA',
    value=1,
    field=NV_R_NWMGSRVP_F_MSZTXVYR
)
NV_R_NWMGSRVP_F_MSZTXVYR_V_DDVZAPYQ = ValueMetadata(
    name='NV_R_NWMGSRVP_F_MSZTXVYR_V_DDVZAPYQ',
    value=0,
    field=NV_R_NWMGSRVP_F_MSZTXVYR
)

NV_R_NWMGSRVP_F_IRNMTOGK = FieldMetadata(
    name='NV_R_NWMGSRVP_F_IRNMTOGK',
    msb=16,
    lsb=16,
    register=NV_R_NWMGSRVP
)

NV_R_NWMGSRVP_F_IRNMTOGK_V_AABSSBNA = ValueMetadata(
    name='NV_R_NWMGSRVP_F_IRNMTOGK_V_AABSSBNA',
    value=1,
    field=NV_R_NWMGSRVP_F_IRNMTOGK
)
NV_R_NWMGSRVP_F_IRNMTOGK_V_DDVZAPYQ = ValueMetadata(
    name='NV_R_NWMGSRVP_F_IRNMTOGK_V_DDVZAPYQ',
    value=0,
    field=NV_R_NWMGSRVP_F_IRNMTOGK
)

NV_R_NWMGSRVP_F_HLOPLUKX = FieldMetadata(
    name='NV_R_NWMGSRVP_F_HLOPLUKX',
    msb=1,
    lsb=1,
    register=NV_R_NWMGSRVP
)

NV_R_NWMGSRVP_F_HLOPLUKX_V_AABSSBNA = ValueMetadata(
    name='NV_R_NWMGSRVP_F_HLOPLUKX_V_AABSSBNA',
    value=1,
    field=NV_R_NWMGSRVP_F_HLOPLUKX
)
NV_R_NWMGSRVP_F_HLOPLUKX_V_DDVZAPYQ = ValueMetadata(
    name='NV_R_NWMGSRVP_F_HLOPLUKX_V_DDVZAPYQ',
    value=0,
    field=NV_R_NWMGSRVP_F_HLOPLUKX
)

NV_R_NWMGSRVP_F_IXIMDEIL = FieldMetadata(
    name='NV_R_NWMGSRVP_F_IXIMDEIL',
    msb=17,
    lsb=17,
    register=NV_R_NWMGSRVP
)

NV_R_NWMGSRVP_F_IXIMDEIL_V_AABSSBNA = ValueMetadata(
    name='NV_R_NWMGSRVP_F_IXIMDEIL_V_AABSSBNA',
    value=1,
    field=NV_R_NWMGSRVP_F_IXIMDEIL
)
NV_R_NWMGSRVP_F_IXIMDEIL_V_DDVZAPYQ = ValueMetadata(
    name='NV_R_NWMGSRVP_F_IXIMDEIL_V_DDVZAPYQ',
    value=0,
    field=NV_R_NWMGSRVP_F_IXIMDEIL
)

NV_R_TJDJJXDH = RegisterMetadata(
    name='NV_R_TJDJJXDH',
    address=0x8c8,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_TJDJJXDH_F_AHLCMKXX = FieldMetadata(
    name='NV_R_TJDJJXDH_F_AHLCMKXX',
    msb=0,
    lsb=0,
    register=NV_R_TJDJJXDH
)

NV_R_TJDJJXDH_F_AHLCMKXX_V_AABSSBNA = ValueMetadata(
    name='NV_R_TJDJJXDH_F_AHLCMKXX_V_AABSSBNA',
    value=1,
    field=NV_R_TJDJJXDH_F_AHLCMKXX
)
NV_R_TJDJJXDH_F_AHLCMKXX_V_DDVZAPYQ = ValueMetadata(
    name='NV_R_TJDJJXDH_F_AHLCMKXX_V_DDVZAPYQ',
    value=0,
    field=NV_R_TJDJJXDH_F_AHLCMKXX
)

NV_R_TJDJJXDH_F_YAKCLFUX = FieldMetadata(
    name='NV_R_TJDJJXDH_F_YAKCLFUX',
    msb=1,
    lsb=1,
    register=NV_R_TJDJJXDH
)

NV_R_TJDJJXDH_F_YAKCLFUX_V_AABSSBNA = ValueMetadata(
    name='NV_R_TJDJJXDH_F_YAKCLFUX_V_AABSSBNA',
    value=1,
    field=NV_R_TJDJJXDH_F_YAKCLFUX
)
NV_R_TJDJJXDH_F_YAKCLFUX_V_DDVZAPYQ = ValueMetadata(
    name='NV_R_TJDJJXDH_F_YAKCLFUX_V_DDVZAPYQ',
    value=0,
    field=NV_R_TJDJJXDH_F_YAKCLFUX
)

NV_R_TJDJJXDH_F_ZCZOCUFJ = FieldMetadata(
    name='NV_R_TJDJJXDH_F_ZCZOCUFJ',
    msb=2,
    lsb=2,
    register=NV_R_TJDJJXDH
)

NV_R_TJDJJXDH_F_ZCZOCUFJ_V_AABSSBNA = ValueMetadata(
    name='NV_R_TJDJJXDH_F_ZCZOCUFJ_V_AABSSBNA',
    value=1,
    field=NV_R_TJDJJXDH_F_ZCZOCUFJ
)
NV_R_TJDJJXDH_F_ZCZOCUFJ_V_DDVZAPYQ = ValueMetadata(
    name='NV_R_TJDJJXDH_F_ZCZOCUFJ_V_DDVZAPYQ',
    value=0,
    field=NV_R_TJDJJXDH_F_ZCZOCUFJ
)

NV_R_TLULSDNR = RegisterMetadata(
    name='NV_R_TLULSDNR',
    address=0x1930,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_TLULSDNR_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_TLULSDNR_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_TLULSDNR
)

NV_R_TLULSDNR_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TLULSDNR_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_TLULSDNR_F_JHGJMNYQ
)
NV_R_TLULSDNR_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TLULSDNR_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_TLULSDNR_F_JHGJMNYQ
)

NV_R_TLULSDNR_F_TWVSOJZO = FieldMetadata(
    name='NV_R_TLULSDNR_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_TLULSDNR
)

NV_R_TLULSDNR_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TLULSDNR_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_TLULSDNR_F_TWVSOJZO
)
NV_R_TLULSDNR_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TLULSDNR_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_TLULSDNR_F_TWVSOJZO
)

NV_R_TLULSDNR_F_VWSCRVRH = FieldMetadata(
    name='NV_R_TLULSDNR_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_TLULSDNR
)

NV_R_TLULSDNR_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TLULSDNR_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_TLULSDNR_F_VWSCRVRH
)
NV_R_TLULSDNR_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TLULSDNR_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_TLULSDNR_F_VWSCRVRH
)

NV_R_TLULSDNR_F_THIGODWY = FieldMetadata(
    name='NV_R_TLULSDNR_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_TLULSDNR
)

NV_R_TLULSDNR_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TLULSDNR_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_TLULSDNR_F_THIGODWY
)
NV_R_TLULSDNR_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TLULSDNR_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_TLULSDNR_F_THIGODWY
)

NV_R_TLULSDNR_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_TLULSDNR_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_TLULSDNR
)

NV_R_TLULSDNR_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TLULSDNR_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_TLULSDNR_F_SQRXFRHZ
)
NV_R_TLULSDNR_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TLULSDNR_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_TLULSDNR_F_SQRXFRHZ
)

NV_R_TLULSDNR_F_NMTHBTXS = FieldMetadata(
    name='NV_R_TLULSDNR_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_TLULSDNR
)

NV_R_TLULSDNR_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TLULSDNR_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_TLULSDNR_F_NMTHBTXS
)
NV_R_TLULSDNR_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TLULSDNR_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_TLULSDNR_F_NMTHBTXS
)

NV_R_TLULSDNR_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_TLULSDNR_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_TLULSDNR
)

NV_R_TLULSDNR_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TLULSDNR_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_TLULSDNR_F_DSYFNOVJ
)
NV_R_TLULSDNR_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TLULSDNR_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_TLULSDNR_F_DSYFNOVJ
)

NV_R_TLULSDNR_F_OQDTLZES = FieldMetadata(
    name='NV_R_TLULSDNR_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_TLULSDNR
)

NV_R_TLULSDNR_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TLULSDNR_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_TLULSDNR_F_OQDTLZES
)
NV_R_TLULSDNR_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TLULSDNR_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_TLULSDNR_F_OQDTLZES
)

NV_R_TLULSDNR_F_IXDOHJET = FieldMetadata(
    name='NV_R_TLULSDNR_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_TLULSDNR
)

NV_R_TLULSDNR_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TLULSDNR_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_TLULSDNR_F_IXDOHJET
)
NV_R_TLULSDNR_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TLULSDNR_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_TLULSDNR_F_IXDOHJET
)

NV_R_TLULSDNR_F_RNQUIIKV = FieldMetadata(
    name='NV_R_TLULSDNR_F_RNQUIIKV',
    msb=5,
    lsb=5,
    register=NV_R_TLULSDNR
)

NV_R_TLULSDNR_F_RNQUIIKV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TLULSDNR_F_RNQUIIKV_V_TYLKUYOV',
    value=1,
    field=NV_R_TLULSDNR_F_RNQUIIKV
)
NV_R_TLULSDNR_F_RNQUIIKV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TLULSDNR_F_RNQUIIKV_V_SQJQPVVW',
    value=0,
    field=NV_R_TLULSDNR_F_RNQUIIKV
)

NV_R_TLULSDNR_F_WYFDZNYR = FieldMetadata(
    name='NV_R_TLULSDNR_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_TLULSDNR
)

NV_R_TLULSDNR_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TLULSDNR_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_TLULSDNR_F_WYFDZNYR
)
NV_R_TLULSDNR_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TLULSDNR_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_TLULSDNR_F_WYFDZNYR
)

NV_R_TLULSDNR_F_PEZFMQOX = FieldMetadata(
    name='NV_R_TLULSDNR_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_TLULSDNR
)

NV_R_TLULSDNR_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TLULSDNR_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_TLULSDNR_F_PEZFMQOX
)
NV_R_TLULSDNR_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TLULSDNR_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_TLULSDNR_F_PEZFMQOX
)

NV_R_TLULSDNR_F_NSKTUIIC = FieldMetadata(
    name='NV_R_TLULSDNR_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_TLULSDNR
)

NV_R_TLULSDNR_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TLULSDNR_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_TLULSDNR_F_NSKTUIIC
)
NV_R_TLULSDNR_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TLULSDNR_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_TLULSDNR_F_NSKTUIIC
)

NV_R_TLULSDNR_F_ASYSFFPX = FieldMetadata(
    name='NV_R_TLULSDNR_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_TLULSDNR
)

NV_R_TLULSDNR_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TLULSDNR_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_TLULSDNR_F_ASYSFFPX
)
NV_R_TLULSDNR_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TLULSDNR_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_TLULSDNR_F_ASYSFFPX
)

NV_R_TLULSDNR_F_CUYZXDTI = FieldMetadata(
    name='NV_R_TLULSDNR_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_TLULSDNR
)

NV_R_TLULSDNR_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TLULSDNR_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_TLULSDNR_F_CUYZXDTI
)
NV_R_TLULSDNR_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TLULSDNR_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_TLULSDNR_F_CUYZXDTI
)

NV_R_TLULSDNR_F_PWTBFXDT = FieldMetadata(
    name='NV_R_TLULSDNR_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_TLULSDNR
)

NV_R_TLULSDNR_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TLULSDNR_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_TLULSDNR_F_PWTBFXDT
)
NV_R_TLULSDNR_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TLULSDNR_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_TLULSDNR_F_PWTBFXDT
)

