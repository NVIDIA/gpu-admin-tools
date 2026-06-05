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
NV_R_LVWMJUPP = RegisterMetadata(
    name='NV_R_LVWMJUPP',
    address=0x1a6c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_LVWMJUPP_F_MNSWOTMH = FieldMetadata(
    name='NV_R_LVWMJUPP_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_MNSWOTMH
)
NV_R_LVWMJUPP_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_MNSWOTMH
)

NV_R_LVWMJUPP_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_LVWMJUPP_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_JHGJMNYQ
)
NV_R_LVWMJUPP_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_JHGJMNYQ
)

NV_R_LVWMJUPP_F_TWVSOJZO = FieldMetadata(
    name='NV_R_LVWMJUPP_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_TWVSOJZO
)
NV_R_LVWMJUPP_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_TWVSOJZO
)

NV_R_LVWMJUPP_F_VWSCRVRH = FieldMetadata(
    name='NV_R_LVWMJUPP_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_VWSCRVRH
)
NV_R_LVWMJUPP_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_VWSCRVRH
)

NV_R_LVWMJUPP_F_THIGODWY = FieldMetadata(
    name='NV_R_LVWMJUPP_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_THIGODWY
)
NV_R_LVWMJUPP_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_THIGODWY
)

NV_R_LVWMJUPP_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_LVWMJUPP_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_SQRXFRHZ
)
NV_R_LVWMJUPP_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_SQRXFRHZ
)

NV_R_LVWMJUPP_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_LVWMJUPP_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_ZJMMMYFA
)
NV_R_LVWMJUPP_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_ZJMMMYFA
)

NV_R_LVWMJUPP_F_NMTHBTXS = FieldMetadata(
    name='NV_R_LVWMJUPP_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_NMTHBTXS
)
NV_R_LVWMJUPP_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_NMTHBTXS
)

NV_R_LVWMJUPP_F_JEAVMSHI = FieldMetadata(
    name='NV_R_LVWMJUPP_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_JEAVMSHI
)
NV_R_LVWMJUPP_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_JEAVMSHI
)

NV_R_LVWMJUPP_F_OLIZCBYA = FieldMetadata(
    name='NV_R_LVWMJUPP_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_OLIZCBYA
)
NV_R_LVWMJUPP_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_OLIZCBYA
)

NV_R_LVWMJUPP_F_XSZMCGXI = FieldMetadata(
    name='NV_R_LVWMJUPP_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_XSZMCGXI
)
NV_R_LVWMJUPP_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_XSZMCGXI
)

NV_R_LVWMJUPP_F_JOBZBTMV = FieldMetadata(
    name='NV_R_LVWMJUPP_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_JOBZBTMV
)
NV_R_LVWMJUPP_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_JOBZBTMV
)

NV_R_LVWMJUPP_F_WRDSLEDI = FieldMetadata(
    name='NV_R_LVWMJUPP_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_WRDSLEDI
)
NV_R_LVWMJUPP_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_WRDSLEDI
)

NV_R_LVWMJUPP_F_UKWLUGKU = FieldMetadata(
    name='NV_R_LVWMJUPP_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_UKWLUGKU
)
NV_R_LVWMJUPP_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_UKWLUGKU
)

NV_R_LVWMJUPP_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_LVWMJUPP_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_TRYWMIGQ
)
NV_R_LVWMJUPP_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_TRYWMIGQ
)

NV_R_LVWMJUPP_F_LHCOGDWA = FieldMetadata(
    name='NV_R_LVWMJUPP_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_LHCOGDWA
)
NV_R_LVWMJUPP_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_LHCOGDWA
)

NV_R_LVWMJUPP_F_IHJHOKZP = FieldMetadata(
    name='NV_R_LVWMJUPP_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_IHJHOKZP
)
NV_R_LVWMJUPP_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_IHJHOKZP
)

NV_R_LVWMJUPP_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_LVWMJUPP_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_DSYFNOVJ
)
NV_R_LVWMJUPP_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_DSYFNOVJ
)

NV_R_LVWMJUPP_F_SAUMSBDG = FieldMetadata(
    name='NV_R_LVWMJUPP_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_SAUMSBDG
)
NV_R_LVWMJUPP_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_SAUMSBDG
)

NV_R_LVWMJUPP_F_ISSIIJSS = FieldMetadata(
    name='NV_R_LVWMJUPP_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_ISSIIJSS
)
NV_R_LVWMJUPP_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_ISSIIJSS
)

NV_R_LVWMJUPP_F_OQDTLZES = FieldMetadata(
    name='NV_R_LVWMJUPP_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_OQDTLZES
)
NV_R_LVWMJUPP_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_OQDTLZES
)

NV_R_LVWMJUPP_F_IXDOHJET = FieldMetadata(
    name='NV_R_LVWMJUPP_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_IXDOHJET
)
NV_R_LVWMJUPP_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_IXDOHJET
)

NV_R_LVWMJUPP_F_WYFDZNYR = FieldMetadata(
    name='NV_R_LVWMJUPP_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_WYFDZNYR
)
NV_R_LVWMJUPP_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_WYFDZNYR
)

NV_R_LVWMJUPP_F_PEZFMQOX = FieldMetadata(
    name='NV_R_LVWMJUPP_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_PEZFMQOX
)
NV_R_LVWMJUPP_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_PEZFMQOX
)

NV_R_LVWMJUPP_F_GATBHVDW = FieldMetadata(
    name='NV_R_LVWMJUPP_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_GATBHVDW
)
NV_R_LVWMJUPP_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_GATBHVDW
)

NV_R_LVWMJUPP_F_NSKTUIIC = FieldMetadata(
    name='NV_R_LVWMJUPP_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_NSKTUIIC
)
NV_R_LVWMJUPP_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_NSKTUIIC
)

NV_R_LVWMJUPP_F_ASYSFFPX = FieldMetadata(
    name='NV_R_LVWMJUPP_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_ASYSFFPX
)
NV_R_LVWMJUPP_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_ASYSFFPX
)

NV_R_LVWMJUPP_F_CUYZXDTI = FieldMetadata(
    name='NV_R_LVWMJUPP_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_CUYZXDTI
)
NV_R_LVWMJUPP_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_CUYZXDTI
)

NV_R_LVWMJUPP_F_PWTBFXDT = FieldMetadata(
    name='NV_R_LVWMJUPP_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_PWTBFXDT
)
NV_R_LVWMJUPP_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_PWTBFXDT
)

NV_R_LVWMJUPP_F_IEYDLCEU = FieldMetadata(
    name='NV_R_LVWMJUPP_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_LVWMJUPP
)

NV_R_LVWMJUPP_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_LVWMJUPP_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_LVWMJUPP_F_IEYDLCEU
)
NV_R_LVWMJUPP_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_LVWMJUPP_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_LVWMJUPP_F_IEYDLCEU
)

NV_R_UXKBJTLP = RegisterMetadata(
    name='NV_R_UXKBJTLP',
    address=0x246c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_UXKBJTLP_F_MNSWOTMH = FieldMetadata(
    name='NV_R_UXKBJTLP_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_MNSWOTMH
)
NV_R_UXKBJTLP_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_MNSWOTMH
)

NV_R_UXKBJTLP_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_UXKBJTLP_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_JHGJMNYQ
)
NV_R_UXKBJTLP_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_JHGJMNYQ
)

NV_R_UXKBJTLP_F_TWVSOJZO = FieldMetadata(
    name='NV_R_UXKBJTLP_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_TWVSOJZO
)
NV_R_UXKBJTLP_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_TWVSOJZO
)

NV_R_UXKBJTLP_F_VWSCRVRH = FieldMetadata(
    name='NV_R_UXKBJTLP_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_VWSCRVRH
)
NV_R_UXKBJTLP_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_VWSCRVRH
)

NV_R_UXKBJTLP_F_THIGODWY = FieldMetadata(
    name='NV_R_UXKBJTLP_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_THIGODWY
)
NV_R_UXKBJTLP_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_THIGODWY
)

NV_R_UXKBJTLP_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_UXKBJTLP_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_SQRXFRHZ
)
NV_R_UXKBJTLP_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_SQRXFRHZ
)

NV_R_UXKBJTLP_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_UXKBJTLP_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_ZJMMMYFA
)
NV_R_UXKBJTLP_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_ZJMMMYFA
)

NV_R_UXKBJTLP_F_NMTHBTXS = FieldMetadata(
    name='NV_R_UXKBJTLP_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_NMTHBTXS
)
NV_R_UXKBJTLP_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_NMTHBTXS
)

NV_R_UXKBJTLP_F_JEAVMSHI = FieldMetadata(
    name='NV_R_UXKBJTLP_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_JEAVMSHI
)
NV_R_UXKBJTLP_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_JEAVMSHI
)

NV_R_UXKBJTLP_F_OLIZCBYA = FieldMetadata(
    name='NV_R_UXKBJTLP_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_OLIZCBYA
)
NV_R_UXKBJTLP_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_OLIZCBYA
)

NV_R_UXKBJTLP_F_XSZMCGXI = FieldMetadata(
    name='NV_R_UXKBJTLP_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_XSZMCGXI
)
NV_R_UXKBJTLP_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_XSZMCGXI
)

NV_R_UXKBJTLP_F_JOBZBTMV = FieldMetadata(
    name='NV_R_UXKBJTLP_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_JOBZBTMV
)
NV_R_UXKBJTLP_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_JOBZBTMV
)

NV_R_UXKBJTLP_F_WRDSLEDI = FieldMetadata(
    name='NV_R_UXKBJTLP_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_WRDSLEDI
)
NV_R_UXKBJTLP_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_WRDSLEDI
)

NV_R_UXKBJTLP_F_UKWLUGKU = FieldMetadata(
    name='NV_R_UXKBJTLP_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_UKWLUGKU
)
NV_R_UXKBJTLP_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_UKWLUGKU
)

NV_R_UXKBJTLP_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_UXKBJTLP_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_TRYWMIGQ
)
NV_R_UXKBJTLP_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_TRYWMIGQ
)

NV_R_UXKBJTLP_F_LHCOGDWA = FieldMetadata(
    name='NV_R_UXKBJTLP_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_LHCOGDWA
)
NV_R_UXKBJTLP_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_LHCOGDWA
)

NV_R_UXKBJTLP_F_IHJHOKZP = FieldMetadata(
    name='NV_R_UXKBJTLP_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_IHJHOKZP
)
NV_R_UXKBJTLP_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_IHJHOKZP
)

NV_R_UXKBJTLP_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_UXKBJTLP_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_DSYFNOVJ
)
NV_R_UXKBJTLP_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_DSYFNOVJ
)

NV_R_UXKBJTLP_F_SAUMSBDG = FieldMetadata(
    name='NV_R_UXKBJTLP_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_SAUMSBDG
)
NV_R_UXKBJTLP_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_SAUMSBDG
)

NV_R_UXKBJTLP_F_ISSIIJSS = FieldMetadata(
    name='NV_R_UXKBJTLP_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_ISSIIJSS
)
NV_R_UXKBJTLP_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_ISSIIJSS
)

NV_R_UXKBJTLP_F_OQDTLZES = FieldMetadata(
    name='NV_R_UXKBJTLP_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_OQDTLZES
)
NV_R_UXKBJTLP_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_OQDTLZES
)

NV_R_UXKBJTLP_F_IXDOHJET = FieldMetadata(
    name='NV_R_UXKBJTLP_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_IXDOHJET
)
NV_R_UXKBJTLP_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_IXDOHJET
)

NV_R_UXKBJTLP_F_WYFDZNYR = FieldMetadata(
    name='NV_R_UXKBJTLP_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_WYFDZNYR
)
NV_R_UXKBJTLP_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_WYFDZNYR
)

NV_R_UXKBJTLP_F_PEZFMQOX = FieldMetadata(
    name='NV_R_UXKBJTLP_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_PEZFMQOX
)
NV_R_UXKBJTLP_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_PEZFMQOX
)

NV_R_UXKBJTLP_F_GATBHVDW = FieldMetadata(
    name='NV_R_UXKBJTLP_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_GATBHVDW
)
NV_R_UXKBJTLP_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_GATBHVDW
)

NV_R_UXKBJTLP_F_NSKTUIIC = FieldMetadata(
    name='NV_R_UXKBJTLP_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_NSKTUIIC
)
NV_R_UXKBJTLP_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_NSKTUIIC
)

NV_R_UXKBJTLP_F_ASYSFFPX = FieldMetadata(
    name='NV_R_UXKBJTLP_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_ASYSFFPX
)
NV_R_UXKBJTLP_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_ASYSFFPX
)

NV_R_UXKBJTLP_F_CUYZXDTI = FieldMetadata(
    name='NV_R_UXKBJTLP_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_CUYZXDTI
)
NV_R_UXKBJTLP_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_CUYZXDTI
)

NV_R_UXKBJTLP_F_PWTBFXDT = FieldMetadata(
    name='NV_R_UXKBJTLP_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_PWTBFXDT
)
NV_R_UXKBJTLP_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_PWTBFXDT
)

NV_R_UXKBJTLP_F_IEYDLCEU = FieldMetadata(
    name='NV_R_UXKBJTLP_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_UXKBJTLP
)

NV_R_UXKBJTLP_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UXKBJTLP_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_UXKBJTLP_F_IEYDLCEU
)
NV_R_UXKBJTLP_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UXKBJTLP_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_UXKBJTLP_F_IEYDLCEU
)

NV_R_BIOEIXEV = RegisterMetadata(
    name='NV_R_BIOEIXEV',
    address=0x256c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_BIOEIXEV_F_MNSWOTMH = FieldMetadata(
    name='NV_R_BIOEIXEV_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_MNSWOTMH
)
NV_R_BIOEIXEV_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_MNSWOTMH
)

NV_R_BIOEIXEV_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_BIOEIXEV_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_JHGJMNYQ
)
NV_R_BIOEIXEV_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_JHGJMNYQ
)

NV_R_BIOEIXEV_F_TWVSOJZO = FieldMetadata(
    name='NV_R_BIOEIXEV_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_TWVSOJZO
)
NV_R_BIOEIXEV_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_TWVSOJZO
)

NV_R_BIOEIXEV_F_VWSCRVRH = FieldMetadata(
    name='NV_R_BIOEIXEV_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_VWSCRVRH
)
NV_R_BIOEIXEV_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_VWSCRVRH
)

NV_R_BIOEIXEV_F_THIGODWY = FieldMetadata(
    name='NV_R_BIOEIXEV_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_THIGODWY
)
NV_R_BIOEIXEV_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_THIGODWY
)

NV_R_BIOEIXEV_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_BIOEIXEV_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_SQRXFRHZ
)
NV_R_BIOEIXEV_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_SQRXFRHZ
)

NV_R_BIOEIXEV_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_BIOEIXEV_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_ZJMMMYFA
)
NV_R_BIOEIXEV_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_ZJMMMYFA
)

NV_R_BIOEIXEV_F_NMTHBTXS = FieldMetadata(
    name='NV_R_BIOEIXEV_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_NMTHBTXS
)
NV_R_BIOEIXEV_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_NMTHBTXS
)

NV_R_BIOEIXEV_F_JEAVMSHI = FieldMetadata(
    name='NV_R_BIOEIXEV_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_JEAVMSHI
)
NV_R_BIOEIXEV_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_JEAVMSHI
)

NV_R_BIOEIXEV_F_OLIZCBYA = FieldMetadata(
    name='NV_R_BIOEIXEV_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_OLIZCBYA
)
NV_R_BIOEIXEV_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_OLIZCBYA
)

NV_R_BIOEIXEV_F_XSZMCGXI = FieldMetadata(
    name='NV_R_BIOEIXEV_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_XSZMCGXI
)
NV_R_BIOEIXEV_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_XSZMCGXI
)

NV_R_BIOEIXEV_F_JOBZBTMV = FieldMetadata(
    name='NV_R_BIOEIXEV_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_JOBZBTMV
)
NV_R_BIOEIXEV_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_JOBZBTMV
)

NV_R_BIOEIXEV_F_WRDSLEDI = FieldMetadata(
    name='NV_R_BIOEIXEV_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_WRDSLEDI
)
NV_R_BIOEIXEV_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_WRDSLEDI
)

NV_R_BIOEIXEV_F_UKWLUGKU = FieldMetadata(
    name='NV_R_BIOEIXEV_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_UKWLUGKU
)
NV_R_BIOEIXEV_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_UKWLUGKU
)

NV_R_BIOEIXEV_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_BIOEIXEV_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_TRYWMIGQ
)
NV_R_BIOEIXEV_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_TRYWMIGQ
)

NV_R_BIOEIXEV_F_LHCOGDWA = FieldMetadata(
    name='NV_R_BIOEIXEV_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_LHCOGDWA
)
NV_R_BIOEIXEV_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_LHCOGDWA
)

NV_R_BIOEIXEV_F_IHJHOKZP = FieldMetadata(
    name='NV_R_BIOEIXEV_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_IHJHOKZP
)
NV_R_BIOEIXEV_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_IHJHOKZP
)

NV_R_BIOEIXEV_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_BIOEIXEV_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_DSYFNOVJ
)
NV_R_BIOEIXEV_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_DSYFNOVJ
)

NV_R_BIOEIXEV_F_SAUMSBDG = FieldMetadata(
    name='NV_R_BIOEIXEV_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_SAUMSBDG
)
NV_R_BIOEIXEV_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_SAUMSBDG
)

NV_R_BIOEIXEV_F_ISSIIJSS = FieldMetadata(
    name='NV_R_BIOEIXEV_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_ISSIIJSS
)
NV_R_BIOEIXEV_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_ISSIIJSS
)

NV_R_BIOEIXEV_F_OQDTLZES = FieldMetadata(
    name='NV_R_BIOEIXEV_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_OQDTLZES
)
NV_R_BIOEIXEV_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_OQDTLZES
)

NV_R_BIOEIXEV_F_IXDOHJET = FieldMetadata(
    name='NV_R_BIOEIXEV_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_IXDOHJET
)
NV_R_BIOEIXEV_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_IXDOHJET
)

NV_R_BIOEIXEV_F_WYFDZNYR = FieldMetadata(
    name='NV_R_BIOEIXEV_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_WYFDZNYR
)
NV_R_BIOEIXEV_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_WYFDZNYR
)

NV_R_BIOEIXEV_F_PEZFMQOX = FieldMetadata(
    name='NV_R_BIOEIXEV_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_PEZFMQOX
)
NV_R_BIOEIXEV_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_PEZFMQOX
)

NV_R_BIOEIXEV_F_GATBHVDW = FieldMetadata(
    name='NV_R_BIOEIXEV_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_GATBHVDW
)
NV_R_BIOEIXEV_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_GATBHVDW
)

NV_R_BIOEIXEV_F_NSKTUIIC = FieldMetadata(
    name='NV_R_BIOEIXEV_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_NSKTUIIC
)
NV_R_BIOEIXEV_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_NSKTUIIC
)

NV_R_BIOEIXEV_F_ASYSFFPX = FieldMetadata(
    name='NV_R_BIOEIXEV_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_ASYSFFPX
)
NV_R_BIOEIXEV_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_ASYSFFPX
)

NV_R_BIOEIXEV_F_CUYZXDTI = FieldMetadata(
    name='NV_R_BIOEIXEV_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_CUYZXDTI
)
NV_R_BIOEIXEV_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_CUYZXDTI
)

NV_R_BIOEIXEV_F_PWTBFXDT = FieldMetadata(
    name='NV_R_BIOEIXEV_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_PWTBFXDT
)
NV_R_BIOEIXEV_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_PWTBFXDT
)

NV_R_BIOEIXEV_F_IEYDLCEU = FieldMetadata(
    name='NV_R_BIOEIXEV_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_BIOEIXEV
)

NV_R_BIOEIXEV_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BIOEIXEV_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_BIOEIXEV_F_IEYDLCEU
)
NV_R_BIOEIXEV_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BIOEIXEV_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_BIOEIXEV_F_IEYDLCEU
)

NV_R_VADVECUF = RegisterMetadata(
    name='NV_R_VADVECUF',
    address=0x266c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_VADVECUF_F_MNSWOTMH = FieldMetadata(
    name='NV_R_VADVECUF_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_MNSWOTMH
)
NV_R_VADVECUF_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_MNSWOTMH
)

NV_R_VADVECUF_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_VADVECUF_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_JHGJMNYQ
)
NV_R_VADVECUF_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_JHGJMNYQ
)

NV_R_VADVECUF_F_TWVSOJZO = FieldMetadata(
    name='NV_R_VADVECUF_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_TWVSOJZO
)
NV_R_VADVECUF_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_TWVSOJZO
)

NV_R_VADVECUF_F_VWSCRVRH = FieldMetadata(
    name='NV_R_VADVECUF_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_VWSCRVRH
)
NV_R_VADVECUF_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_VWSCRVRH
)

NV_R_VADVECUF_F_THIGODWY = FieldMetadata(
    name='NV_R_VADVECUF_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_THIGODWY
)
NV_R_VADVECUF_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_THIGODWY
)

NV_R_VADVECUF_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_VADVECUF_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_SQRXFRHZ
)
NV_R_VADVECUF_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_SQRXFRHZ
)

NV_R_VADVECUF_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_VADVECUF_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_ZJMMMYFA
)
NV_R_VADVECUF_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_ZJMMMYFA
)

NV_R_VADVECUF_F_NMTHBTXS = FieldMetadata(
    name='NV_R_VADVECUF_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_NMTHBTXS
)
NV_R_VADVECUF_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_NMTHBTXS
)

NV_R_VADVECUF_F_JEAVMSHI = FieldMetadata(
    name='NV_R_VADVECUF_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_JEAVMSHI
)
NV_R_VADVECUF_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_JEAVMSHI
)

NV_R_VADVECUF_F_OLIZCBYA = FieldMetadata(
    name='NV_R_VADVECUF_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_OLIZCBYA
)
NV_R_VADVECUF_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_OLIZCBYA
)

NV_R_VADVECUF_F_XSZMCGXI = FieldMetadata(
    name='NV_R_VADVECUF_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_XSZMCGXI
)
NV_R_VADVECUF_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_XSZMCGXI
)

NV_R_VADVECUF_F_JOBZBTMV = FieldMetadata(
    name='NV_R_VADVECUF_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_JOBZBTMV
)
NV_R_VADVECUF_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_JOBZBTMV
)

NV_R_VADVECUF_F_WRDSLEDI = FieldMetadata(
    name='NV_R_VADVECUF_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_WRDSLEDI
)
NV_R_VADVECUF_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_WRDSLEDI
)

NV_R_VADVECUF_F_UKWLUGKU = FieldMetadata(
    name='NV_R_VADVECUF_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_UKWLUGKU
)
NV_R_VADVECUF_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_UKWLUGKU
)

NV_R_VADVECUF_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_VADVECUF_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_TRYWMIGQ
)
NV_R_VADVECUF_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_TRYWMIGQ
)

NV_R_VADVECUF_F_LHCOGDWA = FieldMetadata(
    name='NV_R_VADVECUF_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_LHCOGDWA
)
NV_R_VADVECUF_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_LHCOGDWA
)

NV_R_VADVECUF_F_IHJHOKZP = FieldMetadata(
    name='NV_R_VADVECUF_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_IHJHOKZP
)
NV_R_VADVECUF_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_IHJHOKZP
)

NV_R_VADVECUF_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_VADVECUF_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_DSYFNOVJ
)
NV_R_VADVECUF_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_DSYFNOVJ
)

NV_R_VADVECUF_F_SAUMSBDG = FieldMetadata(
    name='NV_R_VADVECUF_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_SAUMSBDG
)
NV_R_VADVECUF_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_SAUMSBDG
)

NV_R_VADVECUF_F_ISSIIJSS = FieldMetadata(
    name='NV_R_VADVECUF_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_ISSIIJSS
)
NV_R_VADVECUF_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_ISSIIJSS
)

NV_R_VADVECUF_F_OQDTLZES = FieldMetadata(
    name='NV_R_VADVECUF_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_OQDTLZES
)
NV_R_VADVECUF_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_OQDTLZES
)

NV_R_VADVECUF_F_IXDOHJET = FieldMetadata(
    name='NV_R_VADVECUF_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_IXDOHJET
)
NV_R_VADVECUF_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_IXDOHJET
)

NV_R_VADVECUF_F_WYFDZNYR = FieldMetadata(
    name='NV_R_VADVECUF_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_WYFDZNYR
)
NV_R_VADVECUF_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_WYFDZNYR
)

NV_R_VADVECUF_F_PEZFMQOX = FieldMetadata(
    name='NV_R_VADVECUF_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_PEZFMQOX
)
NV_R_VADVECUF_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_PEZFMQOX
)

NV_R_VADVECUF_F_GATBHVDW = FieldMetadata(
    name='NV_R_VADVECUF_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_GATBHVDW
)
NV_R_VADVECUF_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_GATBHVDW
)

NV_R_VADVECUF_F_NSKTUIIC = FieldMetadata(
    name='NV_R_VADVECUF_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_NSKTUIIC
)
NV_R_VADVECUF_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_NSKTUIIC
)

NV_R_VADVECUF_F_ASYSFFPX = FieldMetadata(
    name='NV_R_VADVECUF_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_ASYSFFPX
)
NV_R_VADVECUF_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_ASYSFFPX
)

NV_R_VADVECUF_F_CUYZXDTI = FieldMetadata(
    name='NV_R_VADVECUF_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_CUYZXDTI
)
NV_R_VADVECUF_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_CUYZXDTI
)

NV_R_VADVECUF_F_PWTBFXDT = FieldMetadata(
    name='NV_R_VADVECUF_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_PWTBFXDT
)
NV_R_VADVECUF_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_PWTBFXDT
)

NV_R_VADVECUF_F_IEYDLCEU = FieldMetadata(
    name='NV_R_VADVECUF_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_VADVECUF
)

NV_R_VADVECUF_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_VADVECUF_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_VADVECUF_F_IEYDLCEU
)
NV_R_VADVECUF_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_VADVECUF_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_VADVECUF_F_IEYDLCEU
)

NV_R_WNYTMMPV = RegisterMetadata(
    name='NV_R_WNYTMMPV',
    address=0x276c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_WNYTMMPV_F_MNSWOTMH = FieldMetadata(
    name='NV_R_WNYTMMPV_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_MNSWOTMH
)
NV_R_WNYTMMPV_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_MNSWOTMH
)

NV_R_WNYTMMPV_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_WNYTMMPV_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_JHGJMNYQ
)
NV_R_WNYTMMPV_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_JHGJMNYQ
)

NV_R_WNYTMMPV_F_TWVSOJZO = FieldMetadata(
    name='NV_R_WNYTMMPV_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_TWVSOJZO
)
NV_R_WNYTMMPV_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_TWVSOJZO
)

NV_R_WNYTMMPV_F_VWSCRVRH = FieldMetadata(
    name='NV_R_WNYTMMPV_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_VWSCRVRH
)
NV_R_WNYTMMPV_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_VWSCRVRH
)

NV_R_WNYTMMPV_F_THIGODWY = FieldMetadata(
    name='NV_R_WNYTMMPV_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_THIGODWY
)
NV_R_WNYTMMPV_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_THIGODWY
)

NV_R_WNYTMMPV_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_WNYTMMPV_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_SQRXFRHZ
)
NV_R_WNYTMMPV_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_SQRXFRHZ
)

NV_R_WNYTMMPV_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_WNYTMMPV_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_ZJMMMYFA
)
NV_R_WNYTMMPV_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_ZJMMMYFA
)

NV_R_WNYTMMPV_F_NMTHBTXS = FieldMetadata(
    name='NV_R_WNYTMMPV_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_NMTHBTXS
)
NV_R_WNYTMMPV_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_NMTHBTXS
)

NV_R_WNYTMMPV_F_JEAVMSHI = FieldMetadata(
    name='NV_R_WNYTMMPV_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_JEAVMSHI
)
NV_R_WNYTMMPV_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_JEAVMSHI
)

NV_R_WNYTMMPV_F_OLIZCBYA = FieldMetadata(
    name='NV_R_WNYTMMPV_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_OLIZCBYA
)
NV_R_WNYTMMPV_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_OLIZCBYA
)

NV_R_WNYTMMPV_F_XSZMCGXI = FieldMetadata(
    name='NV_R_WNYTMMPV_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_XSZMCGXI
)
NV_R_WNYTMMPV_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_XSZMCGXI
)

NV_R_WNYTMMPV_F_JOBZBTMV = FieldMetadata(
    name='NV_R_WNYTMMPV_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_JOBZBTMV
)
NV_R_WNYTMMPV_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_JOBZBTMV
)

NV_R_WNYTMMPV_F_WRDSLEDI = FieldMetadata(
    name='NV_R_WNYTMMPV_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_WRDSLEDI
)
NV_R_WNYTMMPV_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_WRDSLEDI
)

NV_R_WNYTMMPV_F_UKWLUGKU = FieldMetadata(
    name='NV_R_WNYTMMPV_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_UKWLUGKU
)
NV_R_WNYTMMPV_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_UKWLUGKU
)

NV_R_WNYTMMPV_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_WNYTMMPV_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_TRYWMIGQ
)
NV_R_WNYTMMPV_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_TRYWMIGQ
)

NV_R_WNYTMMPV_F_LHCOGDWA = FieldMetadata(
    name='NV_R_WNYTMMPV_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_LHCOGDWA
)
NV_R_WNYTMMPV_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_LHCOGDWA
)

NV_R_WNYTMMPV_F_IHJHOKZP = FieldMetadata(
    name='NV_R_WNYTMMPV_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_IHJHOKZP
)
NV_R_WNYTMMPV_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_IHJHOKZP
)

NV_R_WNYTMMPV_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_WNYTMMPV_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_DSYFNOVJ
)
NV_R_WNYTMMPV_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_DSYFNOVJ
)

NV_R_WNYTMMPV_F_SAUMSBDG = FieldMetadata(
    name='NV_R_WNYTMMPV_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_SAUMSBDG
)
NV_R_WNYTMMPV_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_SAUMSBDG
)

NV_R_WNYTMMPV_F_ISSIIJSS = FieldMetadata(
    name='NV_R_WNYTMMPV_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_ISSIIJSS
)
NV_R_WNYTMMPV_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_ISSIIJSS
)

NV_R_WNYTMMPV_F_OQDTLZES = FieldMetadata(
    name='NV_R_WNYTMMPV_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_OQDTLZES
)
NV_R_WNYTMMPV_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_OQDTLZES
)

NV_R_WNYTMMPV_F_IXDOHJET = FieldMetadata(
    name='NV_R_WNYTMMPV_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_IXDOHJET
)
NV_R_WNYTMMPV_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_IXDOHJET
)

NV_R_WNYTMMPV_F_WYFDZNYR = FieldMetadata(
    name='NV_R_WNYTMMPV_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_WYFDZNYR
)
NV_R_WNYTMMPV_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_WYFDZNYR
)

NV_R_WNYTMMPV_F_PEZFMQOX = FieldMetadata(
    name='NV_R_WNYTMMPV_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_PEZFMQOX
)
NV_R_WNYTMMPV_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_PEZFMQOX
)

NV_R_WNYTMMPV_F_GATBHVDW = FieldMetadata(
    name='NV_R_WNYTMMPV_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_GATBHVDW
)
NV_R_WNYTMMPV_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_GATBHVDW
)

NV_R_WNYTMMPV_F_NSKTUIIC = FieldMetadata(
    name='NV_R_WNYTMMPV_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_NSKTUIIC
)
NV_R_WNYTMMPV_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_NSKTUIIC
)

NV_R_WNYTMMPV_F_ASYSFFPX = FieldMetadata(
    name='NV_R_WNYTMMPV_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_ASYSFFPX
)
NV_R_WNYTMMPV_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_ASYSFFPX
)

NV_R_WNYTMMPV_F_CUYZXDTI = FieldMetadata(
    name='NV_R_WNYTMMPV_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_CUYZXDTI
)
NV_R_WNYTMMPV_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_CUYZXDTI
)

NV_R_WNYTMMPV_F_PWTBFXDT = FieldMetadata(
    name='NV_R_WNYTMMPV_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_PWTBFXDT
)
NV_R_WNYTMMPV_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_PWTBFXDT
)

NV_R_WNYTMMPV_F_IEYDLCEU = FieldMetadata(
    name='NV_R_WNYTMMPV_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_WNYTMMPV
)

NV_R_WNYTMMPV_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WNYTMMPV_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_WNYTMMPV_F_IEYDLCEU
)
NV_R_WNYTMMPV_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WNYTMMPV_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_WNYTMMPV_F_IEYDLCEU
)

NV_R_TJAPEFLJ = RegisterMetadata(
    name='NV_R_TJAPEFLJ',
    address=0x286c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_TJAPEFLJ_F_MNSWOTMH = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_MNSWOTMH
)
NV_R_TJAPEFLJ_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_MNSWOTMH
)

NV_R_TJAPEFLJ_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_JHGJMNYQ
)
NV_R_TJAPEFLJ_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_JHGJMNYQ
)

NV_R_TJAPEFLJ_F_TWVSOJZO = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_TWVSOJZO
)
NV_R_TJAPEFLJ_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_TWVSOJZO
)

NV_R_TJAPEFLJ_F_VWSCRVRH = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_VWSCRVRH
)
NV_R_TJAPEFLJ_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_VWSCRVRH
)

NV_R_TJAPEFLJ_F_THIGODWY = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_THIGODWY
)
NV_R_TJAPEFLJ_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_THIGODWY
)

NV_R_TJAPEFLJ_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_SQRXFRHZ
)
NV_R_TJAPEFLJ_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_SQRXFRHZ
)

NV_R_TJAPEFLJ_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_ZJMMMYFA
)
NV_R_TJAPEFLJ_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_ZJMMMYFA
)

NV_R_TJAPEFLJ_F_NMTHBTXS = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_NMTHBTXS
)
NV_R_TJAPEFLJ_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_NMTHBTXS
)

NV_R_TJAPEFLJ_F_JEAVMSHI = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_JEAVMSHI
)
NV_R_TJAPEFLJ_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_JEAVMSHI
)

NV_R_TJAPEFLJ_F_OLIZCBYA = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_OLIZCBYA
)
NV_R_TJAPEFLJ_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_OLIZCBYA
)

NV_R_TJAPEFLJ_F_XSZMCGXI = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_XSZMCGXI
)
NV_R_TJAPEFLJ_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_XSZMCGXI
)

NV_R_TJAPEFLJ_F_JOBZBTMV = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_JOBZBTMV
)
NV_R_TJAPEFLJ_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_JOBZBTMV
)

NV_R_TJAPEFLJ_F_WRDSLEDI = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_WRDSLEDI
)
NV_R_TJAPEFLJ_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_WRDSLEDI
)

NV_R_TJAPEFLJ_F_UKWLUGKU = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_UKWLUGKU
)
NV_R_TJAPEFLJ_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_UKWLUGKU
)

NV_R_TJAPEFLJ_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_TRYWMIGQ
)
NV_R_TJAPEFLJ_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_TRYWMIGQ
)

NV_R_TJAPEFLJ_F_LHCOGDWA = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_LHCOGDWA
)
NV_R_TJAPEFLJ_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_LHCOGDWA
)

NV_R_TJAPEFLJ_F_IHJHOKZP = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_IHJHOKZP
)
NV_R_TJAPEFLJ_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_IHJHOKZP
)

NV_R_TJAPEFLJ_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_DSYFNOVJ
)
NV_R_TJAPEFLJ_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_DSYFNOVJ
)

NV_R_TJAPEFLJ_F_SAUMSBDG = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_SAUMSBDG
)
NV_R_TJAPEFLJ_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_SAUMSBDG
)

NV_R_TJAPEFLJ_F_ISSIIJSS = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_ISSIIJSS
)
NV_R_TJAPEFLJ_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_ISSIIJSS
)

NV_R_TJAPEFLJ_F_OQDTLZES = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_OQDTLZES
)
NV_R_TJAPEFLJ_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_OQDTLZES
)

NV_R_TJAPEFLJ_F_IXDOHJET = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_IXDOHJET
)
NV_R_TJAPEFLJ_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_IXDOHJET
)

NV_R_TJAPEFLJ_F_WYFDZNYR = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_WYFDZNYR
)
NV_R_TJAPEFLJ_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_WYFDZNYR
)

NV_R_TJAPEFLJ_F_PEZFMQOX = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_PEZFMQOX
)
NV_R_TJAPEFLJ_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_PEZFMQOX
)

NV_R_TJAPEFLJ_F_GATBHVDW = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_GATBHVDW
)
NV_R_TJAPEFLJ_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_GATBHVDW
)

NV_R_TJAPEFLJ_F_NSKTUIIC = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_NSKTUIIC
)
NV_R_TJAPEFLJ_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_NSKTUIIC
)

NV_R_TJAPEFLJ_F_ASYSFFPX = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_ASYSFFPX
)
NV_R_TJAPEFLJ_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_ASYSFFPX
)

NV_R_TJAPEFLJ_F_CUYZXDTI = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_CUYZXDTI
)
NV_R_TJAPEFLJ_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_CUYZXDTI
)

NV_R_TJAPEFLJ_F_PWTBFXDT = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_PWTBFXDT
)
NV_R_TJAPEFLJ_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_PWTBFXDT
)

NV_R_TJAPEFLJ_F_IEYDLCEU = FieldMetadata(
    name='NV_R_TJAPEFLJ_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_TJAPEFLJ
)

NV_R_TJAPEFLJ_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_TJAPEFLJ_F_IEYDLCEU
)
NV_R_TJAPEFLJ_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TJAPEFLJ_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_TJAPEFLJ_F_IEYDLCEU
)

NV_R_TRMZKWDQ = RegisterMetadata(
    name='NV_R_TRMZKWDQ',
    address=0x296c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_TRMZKWDQ_F_MNSWOTMH = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_MNSWOTMH
)
NV_R_TRMZKWDQ_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_MNSWOTMH
)

NV_R_TRMZKWDQ_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_JHGJMNYQ
)
NV_R_TRMZKWDQ_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_JHGJMNYQ
)

NV_R_TRMZKWDQ_F_TWVSOJZO = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_TWVSOJZO
)
NV_R_TRMZKWDQ_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_TWVSOJZO
)

NV_R_TRMZKWDQ_F_VWSCRVRH = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_VWSCRVRH
)
NV_R_TRMZKWDQ_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_VWSCRVRH
)

NV_R_TRMZKWDQ_F_THIGODWY = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_THIGODWY
)
NV_R_TRMZKWDQ_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_THIGODWY
)

NV_R_TRMZKWDQ_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_SQRXFRHZ
)
NV_R_TRMZKWDQ_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_SQRXFRHZ
)

NV_R_TRMZKWDQ_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_ZJMMMYFA
)
NV_R_TRMZKWDQ_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_ZJMMMYFA
)

NV_R_TRMZKWDQ_F_NMTHBTXS = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_NMTHBTXS
)
NV_R_TRMZKWDQ_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_NMTHBTXS
)

NV_R_TRMZKWDQ_F_JEAVMSHI = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_JEAVMSHI
)
NV_R_TRMZKWDQ_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_JEAVMSHI
)

NV_R_TRMZKWDQ_F_OLIZCBYA = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_OLIZCBYA
)
NV_R_TRMZKWDQ_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_OLIZCBYA
)

NV_R_TRMZKWDQ_F_XSZMCGXI = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_XSZMCGXI
)
NV_R_TRMZKWDQ_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_XSZMCGXI
)

NV_R_TRMZKWDQ_F_JOBZBTMV = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_JOBZBTMV
)
NV_R_TRMZKWDQ_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_JOBZBTMV
)

NV_R_TRMZKWDQ_F_WRDSLEDI = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_WRDSLEDI
)
NV_R_TRMZKWDQ_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_WRDSLEDI
)

NV_R_TRMZKWDQ_F_UKWLUGKU = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_UKWLUGKU
)
NV_R_TRMZKWDQ_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_UKWLUGKU
)

NV_R_TRMZKWDQ_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_TRYWMIGQ
)
NV_R_TRMZKWDQ_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_TRYWMIGQ
)

NV_R_TRMZKWDQ_F_LHCOGDWA = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_LHCOGDWA
)
NV_R_TRMZKWDQ_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_LHCOGDWA
)

NV_R_TRMZKWDQ_F_IHJHOKZP = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_IHJHOKZP
)
NV_R_TRMZKWDQ_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_IHJHOKZP
)

NV_R_TRMZKWDQ_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_DSYFNOVJ
)
NV_R_TRMZKWDQ_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_DSYFNOVJ
)

NV_R_TRMZKWDQ_F_SAUMSBDG = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_SAUMSBDG
)
NV_R_TRMZKWDQ_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_SAUMSBDG
)

NV_R_TRMZKWDQ_F_ISSIIJSS = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_ISSIIJSS
)
NV_R_TRMZKWDQ_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_ISSIIJSS
)

NV_R_TRMZKWDQ_F_OQDTLZES = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_OQDTLZES
)
NV_R_TRMZKWDQ_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_OQDTLZES
)

NV_R_TRMZKWDQ_F_IXDOHJET = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_IXDOHJET
)
NV_R_TRMZKWDQ_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_IXDOHJET
)

NV_R_TRMZKWDQ_F_WYFDZNYR = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_WYFDZNYR
)
NV_R_TRMZKWDQ_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_WYFDZNYR
)

NV_R_TRMZKWDQ_F_PEZFMQOX = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_PEZFMQOX
)
NV_R_TRMZKWDQ_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_PEZFMQOX
)

NV_R_TRMZKWDQ_F_GATBHVDW = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_GATBHVDW
)
NV_R_TRMZKWDQ_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_GATBHVDW
)

NV_R_TRMZKWDQ_F_NSKTUIIC = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_NSKTUIIC
)
NV_R_TRMZKWDQ_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_NSKTUIIC
)

NV_R_TRMZKWDQ_F_ASYSFFPX = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_ASYSFFPX
)
NV_R_TRMZKWDQ_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_ASYSFFPX
)

NV_R_TRMZKWDQ_F_CUYZXDTI = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_CUYZXDTI
)
NV_R_TRMZKWDQ_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_CUYZXDTI
)

NV_R_TRMZKWDQ_F_PWTBFXDT = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_PWTBFXDT
)
NV_R_TRMZKWDQ_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_PWTBFXDT
)

NV_R_TRMZKWDQ_F_IEYDLCEU = FieldMetadata(
    name='NV_R_TRMZKWDQ_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_TRMZKWDQ
)

NV_R_TRMZKWDQ_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_TRMZKWDQ_F_IEYDLCEU
)
NV_R_TRMZKWDQ_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TRMZKWDQ_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_TRMZKWDQ_F_IEYDLCEU
)

NV_R_IWNZKJKW = RegisterMetadata(
    name='NV_R_IWNZKJKW',
    address=0x1b6c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_IWNZKJKW_F_MNSWOTMH = FieldMetadata(
    name='NV_R_IWNZKJKW_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_MNSWOTMH
)
NV_R_IWNZKJKW_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_MNSWOTMH
)

NV_R_IWNZKJKW_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_IWNZKJKW_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_JHGJMNYQ
)
NV_R_IWNZKJKW_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_JHGJMNYQ
)

NV_R_IWNZKJKW_F_TWVSOJZO = FieldMetadata(
    name='NV_R_IWNZKJKW_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_TWVSOJZO
)
NV_R_IWNZKJKW_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_TWVSOJZO
)

NV_R_IWNZKJKW_F_VWSCRVRH = FieldMetadata(
    name='NV_R_IWNZKJKW_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_VWSCRVRH
)
NV_R_IWNZKJKW_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_VWSCRVRH
)

NV_R_IWNZKJKW_F_THIGODWY = FieldMetadata(
    name='NV_R_IWNZKJKW_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_THIGODWY
)
NV_R_IWNZKJKW_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_THIGODWY
)

NV_R_IWNZKJKW_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_IWNZKJKW_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_SQRXFRHZ
)
NV_R_IWNZKJKW_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_SQRXFRHZ
)

NV_R_IWNZKJKW_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_IWNZKJKW_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_ZJMMMYFA
)
NV_R_IWNZKJKW_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_ZJMMMYFA
)

NV_R_IWNZKJKW_F_NMTHBTXS = FieldMetadata(
    name='NV_R_IWNZKJKW_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_NMTHBTXS
)
NV_R_IWNZKJKW_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_NMTHBTXS
)

NV_R_IWNZKJKW_F_JEAVMSHI = FieldMetadata(
    name='NV_R_IWNZKJKW_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_JEAVMSHI
)
NV_R_IWNZKJKW_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_JEAVMSHI
)

NV_R_IWNZKJKW_F_OLIZCBYA = FieldMetadata(
    name='NV_R_IWNZKJKW_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_OLIZCBYA
)
NV_R_IWNZKJKW_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_OLIZCBYA
)

NV_R_IWNZKJKW_F_XSZMCGXI = FieldMetadata(
    name='NV_R_IWNZKJKW_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_XSZMCGXI
)
NV_R_IWNZKJKW_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_XSZMCGXI
)

NV_R_IWNZKJKW_F_JOBZBTMV = FieldMetadata(
    name='NV_R_IWNZKJKW_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_JOBZBTMV
)
NV_R_IWNZKJKW_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_JOBZBTMV
)

NV_R_IWNZKJKW_F_WRDSLEDI = FieldMetadata(
    name='NV_R_IWNZKJKW_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_WRDSLEDI
)
NV_R_IWNZKJKW_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_WRDSLEDI
)

NV_R_IWNZKJKW_F_UKWLUGKU = FieldMetadata(
    name='NV_R_IWNZKJKW_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_UKWLUGKU
)
NV_R_IWNZKJKW_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_UKWLUGKU
)

NV_R_IWNZKJKW_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_IWNZKJKW_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_TRYWMIGQ
)
NV_R_IWNZKJKW_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_TRYWMIGQ
)

NV_R_IWNZKJKW_F_LHCOGDWA = FieldMetadata(
    name='NV_R_IWNZKJKW_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_LHCOGDWA
)
NV_R_IWNZKJKW_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_LHCOGDWA
)

NV_R_IWNZKJKW_F_IHJHOKZP = FieldMetadata(
    name='NV_R_IWNZKJKW_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_IHJHOKZP
)
NV_R_IWNZKJKW_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_IHJHOKZP
)

NV_R_IWNZKJKW_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_IWNZKJKW_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_DSYFNOVJ
)
NV_R_IWNZKJKW_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_DSYFNOVJ
)

NV_R_IWNZKJKW_F_SAUMSBDG = FieldMetadata(
    name='NV_R_IWNZKJKW_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_SAUMSBDG
)
NV_R_IWNZKJKW_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_SAUMSBDG
)

NV_R_IWNZKJKW_F_ISSIIJSS = FieldMetadata(
    name='NV_R_IWNZKJKW_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_ISSIIJSS
)
NV_R_IWNZKJKW_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_ISSIIJSS
)

NV_R_IWNZKJKW_F_OQDTLZES = FieldMetadata(
    name='NV_R_IWNZKJKW_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_OQDTLZES
)
NV_R_IWNZKJKW_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_OQDTLZES
)

NV_R_IWNZKJKW_F_IXDOHJET = FieldMetadata(
    name='NV_R_IWNZKJKW_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_IXDOHJET
)
NV_R_IWNZKJKW_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_IXDOHJET
)

NV_R_IWNZKJKW_F_WYFDZNYR = FieldMetadata(
    name='NV_R_IWNZKJKW_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_WYFDZNYR
)
NV_R_IWNZKJKW_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_WYFDZNYR
)

NV_R_IWNZKJKW_F_PEZFMQOX = FieldMetadata(
    name='NV_R_IWNZKJKW_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_PEZFMQOX
)
NV_R_IWNZKJKW_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_PEZFMQOX
)

NV_R_IWNZKJKW_F_GATBHVDW = FieldMetadata(
    name='NV_R_IWNZKJKW_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_GATBHVDW
)
NV_R_IWNZKJKW_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_GATBHVDW
)

NV_R_IWNZKJKW_F_NSKTUIIC = FieldMetadata(
    name='NV_R_IWNZKJKW_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_NSKTUIIC
)
NV_R_IWNZKJKW_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_NSKTUIIC
)

NV_R_IWNZKJKW_F_ASYSFFPX = FieldMetadata(
    name='NV_R_IWNZKJKW_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_ASYSFFPX
)
NV_R_IWNZKJKW_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_ASYSFFPX
)

NV_R_IWNZKJKW_F_CUYZXDTI = FieldMetadata(
    name='NV_R_IWNZKJKW_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_CUYZXDTI
)
NV_R_IWNZKJKW_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_CUYZXDTI
)

NV_R_IWNZKJKW_F_PWTBFXDT = FieldMetadata(
    name='NV_R_IWNZKJKW_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_PWTBFXDT
)
NV_R_IWNZKJKW_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_PWTBFXDT
)

NV_R_IWNZKJKW_F_IEYDLCEU = FieldMetadata(
    name='NV_R_IWNZKJKW_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_IWNZKJKW
)

NV_R_IWNZKJKW_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IWNZKJKW_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_IWNZKJKW_F_IEYDLCEU
)
NV_R_IWNZKJKW_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IWNZKJKW_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_IWNZKJKW_F_IEYDLCEU
)

NV_R_YVCKCTZX = RegisterMetadata(
    name='NV_R_YVCKCTZX',
    address=0x1c6c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_YVCKCTZX_F_MNSWOTMH = FieldMetadata(
    name='NV_R_YVCKCTZX_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_MNSWOTMH
)
NV_R_YVCKCTZX_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_MNSWOTMH
)

NV_R_YVCKCTZX_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_YVCKCTZX_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_JHGJMNYQ
)
NV_R_YVCKCTZX_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_JHGJMNYQ
)

NV_R_YVCKCTZX_F_TWVSOJZO = FieldMetadata(
    name='NV_R_YVCKCTZX_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_TWVSOJZO
)
NV_R_YVCKCTZX_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_TWVSOJZO
)

NV_R_YVCKCTZX_F_VWSCRVRH = FieldMetadata(
    name='NV_R_YVCKCTZX_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_VWSCRVRH
)
NV_R_YVCKCTZX_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_VWSCRVRH
)

NV_R_YVCKCTZX_F_THIGODWY = FieldMetadata(
    name='NV_R_YVCKCTZX_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_THIGODWY
)
NV_R_YVCKCTZX_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_THIGODWY
)

NV_R_YVCKCTZX_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_YVCKCTZX_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_SQRXFRHZ
)
NV_R_YVCKCTZX_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_SQRXFRHZ
)

NV_R_YVCKCTZX_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_YVCKCTZX_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_ZJMMMYFA
)
NV_R_YVCKCTZX_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_ZJMMMYFA
)

NV_R_YVCKCTZX_F_NMTHBTXS = FieldMetadata(
    name='NV_R_YVCKCTZX_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_NMTHBTXS
)
NV_R_YVCKCTZX_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_NMTHBTXS
)

NV_R_YVCKCTZX_F_JEAVMSHI = FieldMetadata(
    name='NV_R_YVCKCTZX_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_JEAVMSHI
)
NV_R_YVCKCTZX_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_JEAVMSHI
)

NV_R_YVCKCTZX_F_OLIZCBYA = FieldMetadata(
    name='NV_R_YVCKCTZX_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_OLIZCBYA
)
NV_R_YVCKCTZX_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_OLIZCBYA
)

NV_R_YVCKCTZX_F_XSZMCGXI = FieldMetadata(
    name='NV_R_YVCKCTZX_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_XSZMCGXI
)
NV_R_YVCKCTZX_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_XSZMCGXI
)

NV_R_YVCKCTZX_F_JOBZBTMV = FieldMetadata(
    name='NV_R_YVCKCTZX_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_JOBZBTMV
)
NV_R_YVCKCTZX_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_JOBZBTMV
)

NV_R_YVCKCTZX_F_WRDSLEDI = FieldMetadata(
    name='NV_R_YVCKCTZX_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_WRDSLEDI
)
NV_R_YVCKCTZX_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_WRDSLEDI
)

NV_R_YVCKCTZX_F_UKWLUGKU = FieldMetadata(
    name='NV_R_YVCKCTZX_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_UKWLUGKU
)
NV_R_YVCKCTZX_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_UKWLUGKU
)

NV_R_YVCKCTZX_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_YVCKCTZX_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_TRYWMIGQ
)
NV_R_YVCKCTZX_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_TRYWMIGQ
)

NV_R_YVCKCTZX_F_LHCOGDWA = FieldMetadata(
    name='NV_R_YVCKCTZX_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_LHCOGDWA
)
NV_R_YVCKCTZX_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_LHCOGDWA
)

NV_R_YVCKCTZX_F_IHJHOKZP = FieldMetadata(
    name='NV_R_YVCKCTZX_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_IHJHOKZP
)
NV_R_YVCKCTZX_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_IHJHOKZP
)

NV_R_YVCKCTZX_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_YVCKCTZX_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_DSYFNOVJ
)
NV_R_YVCKCTZX_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_DSYFNOVJ
)

NV_R_YVCKCTZX_F_SAUMSBDG = FieldMetadata(
    name='NV_R_YVCKCTZX_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_SAUMSBDG
)
NV_R_YVCKCTZX_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_SAUMSBDG
)

NV_R_YVCKCTZX_F_ISSIIJSS = FieldMetadata(
    name='NV_R_YVCKCTZX_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_ISSIIJSS
)
NV_R_YVCKCTZX_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_ISSIIJSS
)

NV_R_YVCKCTZX_F_OQDTLZES = FieldMetadata(
    name='NV_R_YVCKCTZX_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_OQDTLZES
)
NV_R_YVCKCTZX_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_OQDTLZES
)

NV_R_YVCKCTZX_F_IXDOHJET = FieldMetadata(
    name='NV_R_YVCKCTZX_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_IXDOHJET
)
NV_R_YVCKCTZX_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_IXDOHJET
)

NV_R_YVCKCTZX_F_WYFDZNYR = FieldMetadata(
    name='NV_R_YVCKCTZX_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_WYFDZNYR
)
NV_R_YVCKCTZX_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_WYFDZNYR
)

NV_R_YVCKCTZX_F_PEZFMQOX = FieldMetadata(
    name='NV_R_YVCKCTZX_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_PEZFMQOX
)
NV_R_YVCKCTZX_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_PEZFMQOX
)

NV_R_YVCKCTZX_F_GATBHVDW = FieldMetadata(
    name='NV_R_YVCKCTZX_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_GATBHVDW
)
NV_R_YVCKCTZX_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_GATBHVDW
)

NV_R_YVCKCTZX_F_NSKTUIIC = FieldMetadata(
    name='NV_R_YVCKCTZX_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_NSKTUIIC
)
NV_R_YVCKCTZX_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_NSKTUIIC
)

NV_R_YVCKCTZX_F_ASYSFFPX = FieldMetadata(
    name='NV_R_YVCKCTZX_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_ASYSFFPX
)
NV_R_YVCKCTZX_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_ASYSFFPX
)

NV_R_YVCKCTZX_F_CUYZXDTI = FieldMetadata(
    name='NV_R_YVCKCTZX_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_CUYZXDTI
)
NV_R_YVCKCTZX_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_CUYZXDTI
)

NV_R_YVCKCTZX_F_PWTBFXDT = FieldMetadata(
    name='NV_R_YVCKCTZX_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_PWTBFXDT
)
NV_R_YVCKCTZX_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_PWTBFXDT
)

NV_R_YVCKCTZX_F_IEYDLCEU = FieldMetadata(
    name='NV_R_YVCKCTZX_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_YVCKCTZX
)

NV_R_YVCKCTZX_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YVCKCTZX_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_YVCKCTZX_F_IEYDLCEU
)
NV_R_YVCKCTZX_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YVCKCTZX_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_YVCKCTZX_F_IEYDLCEU
)

NV_R_UALVCYWW = RegisterMetadata(
    name='NV_R_UALVCYWW',
    address=0x1d6c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_UALVCYWW_F_MNSWOTMH = FieldMetadata(
    name='NV_R_UALVCYWW_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_MNSWOTMH
)
NV_R_UALVCYWW_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_MNSWOTMH
)

NV_R_UALVCYWW_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_UALVCYWW_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_JHGJMNYQ
)
NV_R_UALVCYWW_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_JHGJMNYQ
)

NV_R_UALVCYWW_F_TWVSOJZO = FieldMetadata(
    name='NV_R_UALVCYWW_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_TWVSOJZO
)
NV_R_UALVCYWW_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_TWVSOJZO
)

NV_R_UALVCYWW_F_VWSCRVRH = FieldMetadata(
    name='NV_R_UALVCYWW_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_VWSCRVRH
)
NV_R_UALVCYWW_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_VWSCRVRH
)

NV_R_UALVCYWW_F_THIGODWY = FieldMetadata(
    name='NV_R_UALVCYWW_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_THIGODWY
)
NV_R_UALVCYWW_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_THIGODWY
)

NV_R_UALVCYWW_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_UALVCYWW_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_SQRXFRHZ
)
NV_R_UALVCYWW_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_SQRXFRHZ
)

NV_R_UALVCYWW_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_UALVCYWW_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_ZJMMMYFA
)
NV_R_UALVCYWW_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_ZJMMMYFA
)

NV_R_UALVCYWW_F_NMTHBTXS = FieldMetadata(
    name='NV_R_UALVCYWW_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_NMTHBTXS
)
NV_R_UALVCYWW_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_NMTHBTXS
)

NV_R_UALVCYWW_F_JEAVMSHI = FieldMetadata(
    name='NV_R_UALVCYWW_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_JEAVMSHI
)
NV_R_UALVCYWW_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_JEAVMSHI
)

NV_R_UALVCYWW_F_OLIZCBYA = FieldMetadata(
    name='NV_R_UALVCYWW_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_OLIZCBYA
)
NV_R_UALVCYWW_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_OLIZCBYA
)

NV_R_UALVCYWW_F_XSZMCGXI = FieldMetadata(
    name='NV_R_UALVCYWW_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_XSZMCGXI
)
NV_R_UALVCYWW_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_XSZMCGXI
)

NV_R_UALVCYWW_F_JOBZBTMV = FieldMetadata(
    name='NV_R_UALVCYWW_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_JOBZBTMV
)
NV_R_UALVCYWW_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_JOBZBTMV
)

NV_R_UALVCYWW_F_WRDSLEDI = FieldMetadata(
    name='NV_R_UALVCYWW_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_WRDSLEDI
)
NV_R_UALVCYWW_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_WRDSLEDI
)

NV_R_UALVCYWW_F_UKWLUGKU = FieldMetadata(
    name='NV_R_UALVCYWW_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_UKWLUGKU
)
NV_R_UALVCYWW_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_UKWLUGKU
)

NV_R_UALVCYWW_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_UALVCYWW_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_TRYWMIGQ
)
NV_R_UALVCYWW_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_TRYWMIGQ
)

NV_R_UALVCYWW_F_LHCOGDWA = FieldMetadata(
    name='NV_R_UALVCYWW_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_LHCOGDWA
)
NV_R_UALVCYWW_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_LHCOGDWA
)

NV_R_UALVCYWW_F_IHJHOKZP = FieldMetadata(
    name='NV_R_UALVCYWW_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_IHJHOKZP
)
NV_R_UALVCYWW_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_IHJHOKZP
)

NV_R_UALVCYWW_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_UALVCYWW_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_DSYFNOVJ
)
NV_R_UALVCYWW_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_DSYFNOVJ
)

NV_R_UALVCYWW_F_SAUMSBDG = FieldMetadata(
    name='NV_R_UALVCYWW_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_SAUMSBDG
)
NV_R_UALVCYWW_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_SAUMSBDG
)

NV_R_UALVCYWW_F_ISSIIJSS = FieldMetadata(
    name='NV_R_UALVCYWW_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_ISSIIJSS
)
NV_R_UALVCYWW_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_ISSIIJSS
)

NV_R_UALVCYWW_F_OQDTLZES = FieldMetadata(
    name='NV_R_UALVCYWW_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_OQDTLZES
)
NV_R_UALVCYWW_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_OQDTLZES
)

NV_R_UALVCYWW_F_IXDOHJET = FieldMetadata(
    name='NV_R_UALVCYWW_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_IXDOHJET
)
NV_R_UALVCYWW_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_IXDOHJET
)

NV_R_UALVCYWW_F_WYFDZNYR = FieldMetadata(
    name='NV_R_UALVCYWW_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_WYFDZNYR
)
NV_R_UALVCYWW_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_WYFDZNYR
)

NV_R_UALVCYWW_F_PEZFMQOX = FieldMetadata(
    name='NV_R_UALVCYWW_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_PEZFMQOX
)
NV_R_UALVCYWW_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_PEZFMQOX
)

NV_R_UALVCYWW_F_GATBHVDW = FieldMetadata(
    name='NV_R_UALVCYWW_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_GATBHVDW
)
NV_R_UALVCYWW_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_GATBHVDW
)

NV_R_UALVCYWW_F_NSKTUIIC = FieldMetadata(
    name='NV_R_UALVCYWW_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_NSKTUIIC
)
NV_R_UALVCYWW_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_NSKTUIIC
)

NV_R_UALVCYWW_F_ASYSFFPX = FieldMetadata(
    name='NV_R_UALVCYWW_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_ASYSFFPX
)
NV_R_UALVCYWW_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_ASYSFFPX
)

NV_R_UALVCYWW_F_CUYZXDTI = FieldMetadata(
    name='NV_R_UALVCYWW_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_CUYZXDTI
)
NV_R_UALVCYWW_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_CUYZXDTI
)

NV_R_UALVCYWW_F_PWTBFXDT = FieldMetadata(
    name='NV_R_UALVCYWW_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_PWTBFXDT
)
NV_R_UALVCYWW_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_PWTBFXDT
)

NV_R_UALVCYWW_F_IEYDLCEU = FieldMetadata(
    name='NV_R_UALVCYWW_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_UALVCYWW
)

NV_R_UALVCYWW_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_UALVCYWW_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_UALVCYWW_F_IEYDLCEU
)
NV_R_UALVCYWW_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_UALVCYWW_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_UALVCYWW_F_IEYDLCEU
)

NV_R_WXARETNQ = RegisterMetadata(
    name='NV_R_WXARETNQ',
    address=0x1e6c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_WXARETNQ_F_MNSWOTMH = FieldMetadata(
    name='NV_R_WXARETNQ_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_MNSWOTMH
)
NV_R_WXARETNQ_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_MNSWOTMH
)

NV_R_WXARETNQ_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_WXARETNQ_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_JHGJMNYQ
)
NV_R_WXARETNQ_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_JHGJMNYQ
)

NV_R_WXARETNQ_F_TWVSOJZO = FieldMetadata(
    name='NV_R_WXARETNQ_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_TWVSOJZO
)
NV_R_WXARETNQ_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_TWVSOJZO
)

NV_R_WXARETNQ_F_VWSCRVRH = FieldMetadata(
    name='NV_R_WXARETNQ_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_VWSCRVRH
)
NV_R_WXARETNQ_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_VWSCRVRH
)

NV_R_WXARETNQ_F_THIGODWY = FieldMetadata(
    name='NV_R_WXARETNQ_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_THIGODWY
)
NV_R_WXARETNQ_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_THIGODWY
)

NV_R_WXARETNQ_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_WXARETNQ_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_SQRXFRHZ
)
NV_R_WXARETNQ_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_SQRXFRHZ
)

NV_R_WXARETNQ_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_WXARETNQ_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_ZJMMMYFA
)
NV_R_WXARETNQ_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_ZJMMMYFA
)

NV_R_WXARETNQ_F_NMTHBTXS = FieldMetadata(
    name='NV_R_WXARETNQ_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_NMTHBTXS
)
NV_R_WXARETNQ_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_NMTHBTXS
)

NV_R_WXARETNQ_F_JEAVMSHI = FieldMetadata(
    name='NV_R_WXARETNQ_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_JEAVMSHI
)
NV_R_WXARETNQ_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_JEAVMSHI
)

NV_R_WXARETNQ_F_OLIZCBYA = FieldMetadata(
    name='NV_R_WXARETNQ_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_OLIZCBYA
)
NV_R_WXARETNQ_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_OLIZCBYA
)

NV_R_WXARETNQ_F_XSZMCGXI = FieldMetadata(
    name='NV_R_WXARETNQ_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_XSZMCGXI
)
NV_R_WXARETNQ_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_XSZMCGXI
)

NV_R_WXARETNQ_F_JOBZBTMV = FieldMetadata(
    name='NV_R_WXARETNQ_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_JOBZBTMV
)
NV_R_WXARETNQ_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_JOBZBTMV
)

NV_R_WXARETNQ_F_WRDSLEDI = FieldMetadata(
    name='NV_R_WXARETNQ_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_WRDSLEDI
)
NV_R_WXARETNQ_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_WRDSLEDI
)

NV_R_WXARETNQ_F_UKWLUGKU = FieldMetadata(
    name='NV_R_WXARETNQ_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_UKWLUGKU
)
NV_R_WXARETNQ_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_UKWLUGKU
)

NV_R_WXARETNQ_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_WXARETNQ_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_TRYWMIGQ
)
NV_R_WXARETNQ_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_TRYWMIGQ
)

NV_R_WXARETNQ_F_LHCOGDWA = FieldMetadata(
    name='NV_R_WXARETNQ_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_LHCOGDWA
)
NV_R_WXARETNQ_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_LHCOGDWA
)

NV_R_WXARETNQ_F_IHJHOKZP = FieldMetadata(
    name='NV_R_WXARETNQ_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_IHJHOKZP
)
NV_R_WXARETNQ_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_IHJHOKZP
)

NV_R_WXARETNQ_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_WXARETNQ_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_DSYFNOVJ
)
NV_R_WXARETNQ_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_DSYFNOVJ
)

NV_R_WXARETNQ_F_SAUMSBDG = FieldMetadata(
    name='NV_R_WXARETNQ_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_SAUMSBDG
)
NV_R_WXARETNQ_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_SAUMSBDG
)

NV_R_WXARETNQ_F_ISSIIJSS = FieldMetadata(
    name='NV_R_WXARETNQ_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_ISSIIJSS
)
NV_R_WXARETNQ_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_ISSIIJSS
)

NV_R_WXARETNQ_F_OQDTLZES = FieldMetadata(
    name='NV_R_WXARETNQ_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_OQDTLZES
)
NV_R_WXARETNQ_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_OQDTLZES
)

NV_R_WXARETNQ_F_IXDOHJET = FieldMetadata(
    name='NV_R_WXARETNQ_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_IXDOHJET
)
NV_R_WXARETNQ_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_IXDOHJET
)

NV_R_WXARETNQ_F_WYFDZNYR = FieldMetadata(
    name='NV_R_WXARETNQ_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_WYFDZNYR
)
NV_R_WXARETNQ_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_WYFDZNYR
)

NV_R_WXARETNQ_F_PEZFMQOX = FieldMetadata(
    name='NV_R_WXARETNQ_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_PEZFMQOX
)
NV_R_WXARETNQ_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_PEZFMQOX
)

NV_R_WXARETNQ_F_GATBHVDW = FieldMetadata(
    name='NV_R_WXARETNQ_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_GATBHVDW
)
NV_R_WXARETNQ_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_GATBHVDW
)

NV_R_WXARETNQ_F_NSKTUIIC = FieldMetadata(
    name='NV_R_WXARETNQ_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_NSKTUIIC
)
NV_R_WXARETNQ_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_NSKTUIIC
)

NV_R_WXARETNQ_F_ASYSFFPX = FieldMetadata(
    name='NV_R_WXARETNQ_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_ASYSFFPX
)
NV_R_WXARETNQ_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_ASYSFFPX
)

NV_R_WXARETNQ_F_CUYZXDTI = FieldMetadata(
    name='NV_R_WXARETNQ_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_CUYZXDTI
)
NV_R_WXARETNQ_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_CUYZXDTI
)

NV_R_WXARETNQ_F_PWTBFXDT = FieldMetadata(
    name='NV_R_WXARETNQ_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_PWTBFXDT
)
NV_R_WXARETNQ_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_PWTBFXDT
)

NV_R_WXARETNQ_F_IEYDLCEU = FieldMetadata(
    name='NV_R_WXARETNQ_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_WXARETNQ
)

NV_R_WXARETNQ_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_WXARETNQ_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_WXARETNQ_F_IEYDLCEU
)
NV_R_WXARETNQ_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_WXARETNQ_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_WXARETNQ_F_IEYDLCEU
)

NV_R_TXBKGYDZ = RegisterMetadata(
    name='NV_R_TXBKGYDZ',
    address=0x1f6c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_TXBKGYDZ_F_MNSWOTMH = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_MNSWOTMH
)
NV_R_TXBKGYDZ_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_MNSWOTMH
)

NV_R_TXBKGYDZ_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_JHGJMNYQ
)
NV_R_TXBKGYDZ_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_JHGJMNYQ
)

NV_R_TXBKGYDZ_F_TWVSOJZO = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_TWVSOJZO
)
NV_R_TXBKGYDZ_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_TWVSOJZO
)

NV_R_TXBKGYDZ_F_VWSCRVRH = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_VWSCRVRH
)
NV_R_TXBKGYDZ_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_VWSCRVRH
)

NV_R_TXBKGYDZ_F_THIGODWY = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_THIGODWY
)
NV_R_TXBKGYDZ_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_THIGODWY
)

NV_R_TXBKGYDZ_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_SQRXFRHZ
)
NV_R_TXBKGYDZ_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_SQRXFRHZ
)

NV_R_TXBKGYDZ_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_ZJMMMYFA
)
NV_R_TXBKGYDZ_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_ZJMMMYFA
)

NV_R_TXBKGYDZ_F_NMTHBTXS = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_NMTHBTXS
)
NV_R_TXBKGYDZ_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_NMTHBTXS
)

NV_R_TXBKGYDZ_F_JEAVMSHI = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_JEAVMSHI
)
NV_R_TXBKGYDZ_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_JEAVMSHI
)

NV_R_TXBKGYDZ_F_OLIZCBYA = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_OLIZCBYA
)
NV_R_TXBKGYDZ_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_OLIZCBYA
)

NV_R_TXBKGYDZ_F_XSZMCGXI = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_XSZMCGXI
)
NV_R_TXBKGYDZ_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_XSZMCGXI
)

NV_R_TXBKGYDZ_F_JOBZBTMV = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_JOBZBTMV
)
NV_R_TXBKGYDZ_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_JOBZBTMV
)

NV_R_TXBKGYDZ_F_WRDSLEDI = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_WRDSLEDI
)
NV_R_TXBKGYDZ_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_WRDSLEDI
)

NV_R_TXBKGYDZ_F_UKWLUGKU = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_UKWLUGKU
)
NV_R_TXBKGYDZ_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_UKWLUGKU
)

NV_R_TXBKGYDZ_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_TRYWMIGQ
)
NV_R_TXBKGYDZ_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_TRYWMIGQ
)

NV_R_TXBKGYDZ_F_LHCOGDWA = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_LHCOGDWA
)
NV_R_TXBKGYDZ_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_LHCOGDWA
)

NV_R_TXBKGYDZ_F_IHJHOKZP = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_IHJHOKZP
)
NV_R_TXBKGYDZ_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_IHJHOKZP
)

NV_R_TXBKGYDZ_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_DSYFNOVJ
)
NV_R_TXBKGYDZ_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_DSYFNOVJ
)

NV_R_TXBKGYDZ_F_SAUMSBDG = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_SAUMSBDG
)
NV_R_TXBKGYDZ_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_SAUMSBDG
)

NV_R_TXBKGYDZ_F_ISSIIJSS = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_ISSIIJSS
)
NV_R_TXBKGYDZ_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_ISSIIJSS
)

NV_R_TXBKGYDZ_F_OQDTLZES = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_OQDTLZES
)
NV_R_TXBKGYDZ_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_OQDTLZES
)

NV_R_TXBKGYDZ_F_IXDOHJET = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_IXDOHJET
)
NV_R_TXBKGYDZ_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_IXDOHJET
)

NV_R_TXBKGYDZ_F_WYFDZNYR = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_WYFDZNYR
)
NV_R_TXBKGYDZ_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_WYFDZNYR
)

NV_R_TXBKGYDZ_F_PEZFMQOX = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_PEZFMQOX
)
NV_R_TXBKGYDZ_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_PEZFMQOX
)

NV_R_TXBKGYDZ_F_GATBHVDW = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_GATBHVDW
)
NV_R_TXBKGYDZ_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_GATBHVDW
)

NV_R_TXBKGYDZ_F_NSKTUIIC = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_NSKTUIIC
)
NV_R_TXBKGYDZ_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_NSKTUIIC
)

NV_R_TXBKGYDZ_F_ASYSFFPX = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_ASYSFFPX
)
NV_R_TXBKGYDZ_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_ASYSFFPX
)

NV_R_TXBKGYDZ_F_CUYZXDTI = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_CUYZXDTI
)
NV_R_TXBKGYDZ_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_CUYZXDTI
)

NV_R_TXBKGYDZ_F_PWTBFXDT = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_PWTBFXDT
)
NV_R_TXBKGYDZ_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_PWTBFXDT
)

NV_R_TXBKGYDZ_F_IEYDLCEU = FieldMetadata(
    name='NV_R_TXBKGYDZ_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_TXBKGYDZ
)

NV_R_TXBKGYDZ_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_TXBKGYDZ_F_IEYDLCEU
)
NV_R_TXBKGYDZ_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_TXBKGYDZ_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_TXBKGYDZ_F_IEYDLCEU
)

NV_R_PFUSQTKJ = RegisterMetadata(
    name='NV_R_PFUSQTKJ',
    address=0x206c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_PFUSQTKJ_F_MNSWOTMH = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_MNSWOTMH
)
NV_R_PFUSQTKJ_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_MNSWOTMH
)

NV_R_PFUSQTKJ_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_JHGJMNYQ
)
NV_R_PFUSQTKJ_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_JHGJMNYQ
)

NV_R_PFUSQTKJ_F_TWVSOJZO = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_TWVSOJZO
)
NV_R_PFUSQTKJ_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_TWVSOJZO
)

NV_R_PFUSQTKJ_F_VWSCRVRH = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_VWSCRVRH
)
NV_R_PFUSQTKJ_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_VWSCRVRH
)

NV_R_PFUSQTKJ_F_THIGODWY = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_THIGODWY
)
NV_R_PFUSQTKJ_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_THIGODWY
)

NV_R_PFUSQTKJ_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_SQRXFRHZ
)
NV_R_PFUSQTKJ_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_SQRXFRHZ
)

NV_R_PFUSQTKJ_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_ZJMMMYFA
)
NV_R_PFUSQTKJ_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_ZJMMMYFA
)

NV_R_PFUSQTKJ_F_NMTHBTXS = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_NMTHBTXS
)
NV_R_PFUSQTKJ_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_NMTHBTXS
)

NV_R_PFUSQTKJ_F_JEAVMSHI = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_JEAVMSHI
)
NV_R_PFUSQTKJ_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_JEAVMSHI
)

NV_R_PFUSQTKJ_F_OLIZCBYA = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_OLIZCBYA
)
NV_R_PFUSQTKJ_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_OLIZCBYA
)

NV_R_PFUSQTKJ_F_XSZMCGXI = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_XSZMCGXI
)
NV_R_PFUSQTKJ_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_XSZMCGXI
)

NV_R_PFUSQTKJ_F_JOBZBTMV = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_JOBZBTMV
)
NV_R_PFUSQTKJ_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_JOBZBTMV
)

NV_R_PFUSQTKJ_F_WRDSLEDI = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_WRDSLEDI
)
NV_R_PFUSQTKJ_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_WRDSLEDI
)

NV_R_PFUSQTKJ_F_UKWLUGKU = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_UKWLUGKU
)
NV_R_PFUSQTKJ_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_UKWLUGKU
)

NV_R_PFUSQTKJ_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_TRYWMIGQ
)
NV_R_PFUSQTKJ_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_TRYWMIGQ
)

NV_R_PFUSQTKJ_F_LHCOGDWA = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_LHCOGDWA
)
NV_R_PFUSQTKJ_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_LHCOGDWA
)

NV_R_PFUSQTKJ_F_IHJHOKZP = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_IHJHOKZP
)
NV_R_PFUSQTKJ_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_IHJHOKZP
)

NV_R_PFUSQTKJ_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_DSYFNOVJ
)
NV_R_PFUSQTKJ_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_DSYFNOVJ
)

NV_R_PFUSQTKJ_F_SAUMSBDG = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_SAUMSBDG
)
NV_R_PFUSQTKJ_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_SAUMSBDG
)

NV_R_PFUSQTKJ_F_ISSIIJSS = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_ISSIIJSS
)
NV_R_PFUSQTKJ_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_ISSIIJSS
)

NV_R_PFUSQTKJ_F_OQDTLZES = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_OQDTLZES
)
NV_R_PFUSQTKJ_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_OQDTLZES
)

NV_R_PFUSQTKJ_F_IXDOHJET = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_IXDOHJET
)
NV_R_PFUSQTKJ_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_IXDOHJET
)

NV_R_PFUSQTKJ_F_WYFDZNYR = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_WYFDZNYR
)
NV_R_PFUSQTKJ_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_WYFDZNYR
)

NV_R_PFUSQTKJ_F_PEZFMQOX = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_PEZFMQOX
)
NV_R_PFUSQTKJ_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_PEZFMQOX
)

NV_R_PFUSQTKJ_F_GATBHVDW = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_GATBHVDW
)
NV_R_PFUSQTKJ_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_GATBHVDW
)

NV_R_PFUSQTKJ_F_NSKTUIIC = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_NSKTUIIC
)
NV_R_PFUSQTKJ_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_NSKTUIIC
)

NV_R_PFUSQTKJ_F_ASYSFFPX = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_ASYSFFPX
)
NV_R_PFUSQTKJ_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_ASYSFFPX
)

NV_R_PFUSQTKJ_F_CUYZXDTI = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_CUYZXDTI
)
NV_R_PFUSQTKJ_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_CUYZXDTI
)

NV_R_PFUSQTKJ_F_PWTBFXDT = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_PWTBFXDT
)
NV_R_PFUSQTKJ_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_PWTBFXDT
)

NV_R_PFUSQTKJ_F_IEYDLCEU = FieldMetadata(
    name='NV_R_PFUSQTKJ_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_PFUSQTKJ
)

NV_R_PFUSQTKJ_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_PFUSQTKJ_F_IEYDLCEU
)
NV_R_PFUSQTKJ_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_PFUSQTKJ_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_PFUSQTKJ_F_IEYDLCEU
)

NV_R_NBVGKXJW = RegisterMetadata(
    name='NV_R_NBVGKXJW',
    address=0x216c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_NBVGKXJW_F_MNSWOTMH = FieldMetadata(
    name='NV_R_NBVGKXJW_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_MNSWOTMH
)
NV_R_NBVGKXJW_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_MNSWOTMH
)

NV_R_NBVGKXJW_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_NBVGKXJW_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_JHGJMNYQ
)
NV_R_NBVGKXJW_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_JHGJMNYQ
)

NV_R_NBVGKXJW_F_TWVSOJZO = FieldMetadata(
    name='NV_R_NBVGKXJW_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_TWVSOJZO
)
NV_R_NBVGKXJW_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_TWVSOJZO
)

NV_R_NBVGKXJW_F_VWSCRVRH = FieldMetadata(
    name='NV_R_NBVGKXJW_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_VWSCRVRH
)
NV_R_NBVGKXJW_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_VWSCRVRH
)

NV_R_NBVGKXJW_F_THIGODWY = FieldMetadata(
    name='NV_R_NBVGKXJW_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_THIGODWY
)
NV_R_NBVGKXJW_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_THIGODWY
)

NV_R_NBVGKXJW_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_NBVGKXJW_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_SQRXFRHZ
)
NV_R_NBVGKXJW_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_SQRXFRHZ
)

NV_R_NBVGKXJW_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_NBVGKXJW_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_ZJMMMYFA
)
NV_R_NBVGKXJW_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_ZJMMMYFA
)

NV_R_NBVGKXJW_F_NMTHBTXS = FieldMetadata(
    name='NV_R_NBVGKXJW_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_NMTHBTXS
)
NV_R_NBVGKXJW_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_NMTHBTXS
)

NV_R_NBVGKXJW_F_JEAVMSHI = FieldMetadata(
    name='NV_R_NBVGKXJW_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_JEAVMSHI
)
NV_R_NBVGKXJW_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_JEAVMSHI
)

NV_R_NBVGKXJW_F_OLIZCBYA = FieldMetadata(
    name='NV_R_NBVGKXJW_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_OLIZCBYA
)
NV_R_NBVGKXJW_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_OLIZCBYA
)

NV_R_NBVGKXJW_F_XSZMCGXI = FieldMetadata(
    name='NV_R_NBVGKXJW_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_XSZMCGXI
)
NV_R_NBVGKXJW_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_XSZMCGXI
)

NV_R_NBVGKXJW_F_JOBZBTMV = FieldMetadata(
    name='NV_R_NBVGKXJW_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_JOBZBTMV
)
NV_R_NBVGKXJW_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_JOBZBTMV
)

NV_R_NBVGKXJW_F_WRDSLEDI = FieldMetadata(
    name='NV_R_NBVGKXJW_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_WRDSLEDI
)
NV_R_NBVGKXJW_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_WRDSLEDI
)

NV_R_NBVGKXJW_F_UKWLUGKU = FieldMetadata(
    name='NV_R_NBVGKXJW_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_UKWLUGKU
)
NV_R_NBVGKXJW_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_UKWLUGKU
)

NV_R_NBVGKXJW_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_NBVGKXJW_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_TRYWMIGQ
)
NV_R_NBVGKXJW_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_TRYWMIGQ
)

NV_R_NBVGKXJW_F_LHCOGDWA = FieldMetadata(
    name='NV_R_NBVGKXJW_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_LHCOGDWA
)
NV_R_NBVGKXJW_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_LHCOGDWA
)

NV_R_NBVGKXJW_F_IHJHOKZP = FieldMetadata(
    name='NV_R_NBVGKXJW_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_IHJHOKZP
)
NV_R_NBVGKXJW_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_IHJHOKZP
)

NV_R_NBVGKXJW_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_NBVGKXJW_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_DSYFNOVJ
)
NV_R_NBVGKXJW_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_DSYFNOVJ
)

NV_R_NBVGKXJW_F_SAUMSBDG = FieldMetadata(
    name='NV_R_NBVGKXJW_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_SAUMSBDG
)
NV_R_NBVGKXJW_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_SAUMSBDG
)

NV_R_NBVGKXJW_F_ISSIIJSS = FieldMetadata(
    name='NV_R_NBVGKXJW_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_ISSIIJSS
)
NV_R_NBVGKXJW_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_ISSIIJSS
)

NV_R_NBVGKXJW_F_OQDTLZES = FieldMetadata(
    name='NV_R_NBVGKXJW_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_OQDTLZES
)
NV_R_NBVGKXJW_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_OQDTLZES
)

NV_R_NBVGKXJW_F_IXDOHJET = FieldMetadata(
    name='NV_R_NBVGKXJW_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_IXDOHJET
)
NV_R_NBVGKXJW_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_IXDOHJET
)

NV_R_NBVGKXJW_F_WYFDZNYR = FieldMetadata(
    name='NV_R_NBVGKXJW_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_WYFDZNYR
)
NV_R_NBVGKXJW_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_WYFDZNYR
)

NV_R_NBVGKXJW_F_PEZFMQOX = FieldMetadata(
    name='NV_R_NBVGKXJW_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_PEZFMQOX
)
NV_R_NBVGKXJW_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_PEZFMQOX
)

NV_R_NBVGKXJW_F_GATBHVDW = FieldMetadata(
    name='NV_R_NBVGKXJW_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_GATBHVDW
)
NV_R_NBVGKXJW_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_GATBHVDW
)

NV_R_NBVGKXJW_F_NSKTUIIC = FieldMetadata(
    name='NV_R_NBVGKXJW_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_NSKTUIIC
)
NV_R_NBVGKXJW_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_NSKTUIIC
)

NV_R_NBVGKXJW_F_ASYSFFPX = FieldMetadata(
    name='NV_R_NBVGKXJW_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_ASYSFFPX
)
NV_R_NBVGKXJW_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_ASYSFFPX
)

NV_R_NBVGKXJW_F_CUYZXDTI = FieldMetadata(
    name='NV_R_NBVGKXJW_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_CUYZXDTI
)
NV_R_NBVGKXJW_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_CUYZXDTI
)

NV_R_NBVGKXJW_F_PWTBFXDT = FieldMetadata(
    name='NV_R_NBVGKXJW_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_PWTBFXDT
)
NV_R_NBVGKXJW_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_PWTBFXDT
)

NV_R_NBVGKXJW_F_IEYDLCEU = FieldMetadata(
    name='NV_R_NBVGKXJW_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_NBVGKXJW
)

NV_R_NBVGKXJW_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_NBVGKXJW_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_NBVGKXJW_F_IEYDLCEU
)
NV_R_NBVGKXJW_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_NBVGKXJW_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_NBVGKXJW_F_IEYDLCEU
)

NV_R_IFSGALTJ = RegisterMetadata(
    name='NV_R_IFSGALTJ',
    address=0x226c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_IFSGALTJ_F_MNSWOTMH = FieldMetadata(
    name='NV_R_IFSGALTJ_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_MNSWOTMH
)
NV_R_IFSGALTJ_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_MNSWOTMH
)

NV_R_IFSGALTJ_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_IFSGALTJ_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_JHGJMNYQ
)
NV_R_IFSGALTJ_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_JHGJMNYQ
)

NV_R_IFSGALTJ_F_TWVSOJZO = FieldMetadata(
    name='NV_R_IFSGALTJ_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_TWVSOJZO
)
NV_R_IFSGALTJ_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_TWVSOJZO
)

NV_R_IFSGALTJ_F_VWSCRVRH = FieldMetadata(
    name='NV_R_IFSGALTJ_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_VWSCRVRH
)
NV_R_IFSGALTJ_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_VWSCRVRH
)

NV_R_IFSGALTJ_F_THIGODWY = FieldMetadata(
    name='NV_R_IFSGALTJ_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_THIGODWY
)
NV_R_IFSGALTJ_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_THIGODWY
)

NV_R_IFSGALTJ_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_IFSGALTJ_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_SQRXFRHZ
)
NV_R_IFSGALTJ_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_SQRXFRHZ
)

NV_R_IFSGALTJ_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_IFSGALTJ_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_ZJMMMYFA
)
NV_R_IFSGALTJ_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_ZJMMMYFA
)

NV_R_IFSGALTJ_F_NMTHBTXS = FieldMetadata(
    name='NV_R_IFSGALTJ_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_NMTHBTXS
)
NV_R_IFSGALTJ_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_NMTHBTXS
)

NV_R_IFSGALTJ_F_JEAVMSHI = FieldMetadata(
    name='NV_R_IFSGALTJ_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_JEAVMSHI
)
NV_R_IFSGALTJ_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_JEAVMSHI
)

NV_R_IFSGALTJ_F_OLIZCBYA = FieldMetadata(
    name='NV_R_IFSGALTJ_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_OLIZCBYA
)
NV_R_IFSGALTJ_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_OLIZCBYA
)

NV_R_IFSGALTJ_F_XSZMCGXI = FieldMetadata(
    name='NV_R_IFSGALTJ_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_XSZMCGXI
)
NV_R_IFSGALTJ_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_XSZMCGXI
)

NV_R_IFSGALTJ_F_JOBZBTMV = FieldMetadata(
    name='NV_R_IFSGALTJ_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_JOBZBTMV
)
NV_R_IFSGALTJ_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_JOBZBTMV
)

NV_R_IFSGALTJ_F_WRDSLEDI = FieldMetadata(
    name='NV_R_IFSGALTJ_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_WRDSLEDI
)
NV_R_IFSGALTJ_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_WRDSLEDI
)

NV_R_IFSGALTJ_F_UKWLUGKU = FieldMetadata(
    name='NV_R_IFSGALTJ_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_UKWLUGKU
)
NV_R_IFSGALTJ_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_UKWLUGKU
)

NV_R_IFSGALTJ_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_IFSGALTJ_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_TRYWMIGQ
)
NV_R_IFSGALTJ_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_TRYWMIGQ
)

NV_R_IFSGALTJ_F_LHCOGDWA = FieldMetadata(
    name='NV_R_IFSGALTJ_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_LHCOGDWA
)
NV_R_IFSGALTJ_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_LHCOGDWA
)

NV_R_IFSGALTJ_F_IHJHOKZP = FieldMetadata(
    name='NV_R_IFSGALTJ_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_IHJHOKZP
)
NV_R_IFSGALTJ_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_IHJHOKZP
)

NV_R_IFSGALTJ_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_IFSGALTJ_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_DSYFNOVJ
)
NV_R_IFSGALTJ_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_DSYFNOVJ
)

NV_R_IFSGALTJ_F_SAUMSBDG = FieldMetadata(
    name='NV_R_IFSGALTJ_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_SAUMSBDG
)
NV_R_IFSGALTJ_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_SAUMSBDG
)

NV_R_IFSGALTJ_F_ISSIIJSS = FieldMetadata(
    name='NV_R_IFSGALTJ_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_ISSIIJSS
)
NV_R_IFSGALTJ_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_ISSIIJSS
)

NV_R_IFSGALTJ_F_OQDTLZES = FieldMetadata(
    name='NV_R_IFSGALTJ_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_OQDTLZES
)
NV_R_IFSGALTJ_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_OQDTLZES
)

NV_R_IFSGALTJ_F_IXDOHJET = FieldMetadata(
    name='NV_R_IFSGALTJ_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_IXDOHJET
)
NV_R_IFSGALTJ_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_IXDOHJET
)

NV_R_IFSGALTJ_F_WYFDZNYR = FieldMetadata(
    name='NV_R_IFSGALTJ_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_WYFDZNYR
)
NV_R_IFSGALTJ_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_WYFDZNYR
)

NV_R_IFSGALTJ_F_PEZFMQOX = FieldMetadata(
    name='NV_R_IFSGALTJ_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_PEZFMQOX
)
NV_R_IFSGALTJ_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_PEZFMQOX
)

NV_R_IFSGALTJ_F_GATBHVDW = FieldMetadata(
    name='NV_R_IFSGALTJ_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_GATBHVDW
)
NV_R_IFSGALTJ_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_GATBHVDW
)

NV_R_IFSGALTJ_F_NSKTUIIC = FieldMetadata(
    name='NV_R_IFSGALTJ_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_NSKTUIIC
)
NV_R_IFSGALTJ_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_NSKTUIIC
)

NV_R_IFSGALTJ_F_ASYSFFPX = FieldMetadata(
    name='NV_R_IFSGALTJ_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_ASYSFFPX
)
NV_R_IFSGALTJ_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_ASYSFFPX
)

NV_R_IFSGALTJ_F_CUYZXDTI = FieldMetadata(
    name='NV_R_IFSGALTJ_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_CUYZXDTI
)
NV_R_IFSGALTJ_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_CUYZXDTI
)

NV_R_IFSGALTJ_F_PWTBFXDT = FieldMetadata(
    name='NV_R_IFSGALTJ_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_PWTBFXDT
)
NV_R_IFSGALTJ_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_PWTBFXDT
)

NV_R_IFSGALTJ_F_IEYDLCEU = FieldMetadata(
    name='NV_R_IFSGALTJ_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_IFSGALTJ
)

NV_R_IFSGALTJ_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_IFSGALTJ_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_IFSGALTJ_F_IEYDLCEU
)
NV_R_IFSGALTJ_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_IFSGALTJ_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_IFSGALTJ_F_IEYDLCEU
)

NV_R_BLGTUOOJ = RegisterMetadata(
    name='NV_R_BLGTUOOJ',
    address=0x236c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_BLGTUOOJ_F_MNSWOTMH = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_MNSWOTMH
)
NV_R_BLGTUOOJ_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_MNSWOTMH
)

NV_R_BLGTUOOJ_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_JHGJMNYQ
)
NV_R_BLGTUOOJ_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_JHGJMNYQ
)

NV_R_BLGTUOOJ_F_TWVSOJZO = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_TWVSOJZO
)
NV_R_BLGTUOOJ_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_TWVSOJZO
)

NV_R_BLGTUOOJ_F_VWSCRVRH = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_VWSCRVRH
)
NV_R_BLGTUOOJ_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_VWSCRVRH
)

NV_R_BLGTUOOJ_F_THIGODWY = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_THIGODWY
)
NV_R_BLGTUOOJ_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_THIGODWY
)

NV_R_BLGTUOOJ_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_SQRXFRHZ
)
NV_R_BLGTUOOJ_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_SQRXFRHZ
)

NV_R_BLGTUOOJ_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_ZJMMMYFA
)
NV_R_BLGTUOOJ_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_ZJMMMYFA
)

NV_R_BLGTUOOJ_F_NMTHBTXS = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_NMTHBTXS
)
NV_R_BLGTUOOJ_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_NMTHBTXS
)

NV_R_BLGTUOOJ_F_JEAVMSHI = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_JEAVMSHI
)
NV_R_BLGTUOOJ_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_JEAVMSHI
)

NV_R_BLGTUOOJ_F_OLIZCBYA = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_OLIZCBYA
)
NV_R_BLGTUOOJ_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_OLIZCBYA
)

NV_R_BLGTUOOJ_F_XSZMCGXI = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_XSZMCGXI
)
NV_R_BLGTUOOJ_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_XSZMCGXI
)

NV_R_BLGTUOOJ_F_JOBZBTMV = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_JOBZBTMV
)
NV_R_BLGTUOOJ_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_JOBZBTMV
)

NV_R_BLGTUOOJ_F_WRDSLEDI = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_WRDSLEDI
)
NV_R_BLGTUOOJ_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_WRDSLEDI
)

NV_R_BLGTUOOJ_F_UKWLUGKU = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_UKWLUGKU
)
NV_R_BLGTUOOJ_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_UKWLUGKU
)

NV_R_BLGTUOOJ_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_TRYWMIGQ
)
NV_R_BLGTUOOJ_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_TRYWMIGQ
)

NV_R_BLGTUOOJ_F_LHCOGDWA = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_LHCOGDWA
)
NV_R_BLGTUOOJ_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_LHCOGDWA
)

NV_R_BLGTUOOJ_F_IHJHOKZP = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_IHJHOKZP
)
NV_R_BLGTUOOJ_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_IHJHOKZP
)

NV_R_BLGTUOOJ_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_DSYFNOVJ
)
NV_R_BLGTUOOJ_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_DSYFNOVJ
)

NV_R_BLGTUOOJ_F_SAUMSBDG = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_SAUMSBDG
)
NV_R_BLGTUOOJ_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_SAUMSBDG
)

NV_R_BLGTUOOJ_F_ISSIIJSS = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_ISSIIJSS
)
NV_R_BLGTUOOJ_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_ISSIIJSS
)

NV_R_BLGTUOOJ_F_OQDTLZES = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_OQDTLZES
)
NV_R_BLGTUOOJ_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_OQDTLZES
)

NV_R_BLGTUOOJ_F_IXDOHJET = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_IXDOHJET
)
NV_R_BLGTUOOJ_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_IXDOHJET
)

NV_R_BLGTUOOJ_F_WYFDZNYR = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_WYFDZNYR
)
NV_R_BLGTUOOJ_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_WYFDZNYR
)

NV_R_BLGTUOOJ_F_PEZFMQOX = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_PEZFMQOX
)
NV_R_BLGTUOOJ_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_PEZFMQOX
)

NV_R_BLGTUOOJ_F_GATBHVDW = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_GATBHVDW
)
NV_R_BLGTUOOJ_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_GATBHVDW
)

NV_R_BLGTUOOJ_F_NSKTUIIC = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_NSKTUIIC
)
NV_R_BLGTUOOJ_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_NSKTUIIC
)

NV_R_BLGTUOOJ_F_ASYSFFPX = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_ASYSFFPX
)
NV_R_BLGTUOOJ_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_ASYSFFPX
)

NV_R_BLGTUOOJ_F_CUYZXDTI = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_CUYZXDTI
)
NV_R_BLGTUOOJ_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_CUYZXDTI
)

NV_R_BLGTUOOJ_F_PWTBFXDT = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_PWTBFXDT
)
NV_R_BLGTUOOJ_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_PWTBFXDT
)

NV_R_BLGTUOOJ_F_IEYDLCEU = FieldMetadata(
    name='NV_R_BLGTUOOJ_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_BLGTUOOJ
)

NV_R_BLGTUOOJ_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_BLGTUOOJ_F_IEYDLCEU
)
NV_R_BLGTUOOJ_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_BLGTUOOJ_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_BLGTUOOJ_F_IEYDLCEU
)

NV_R_ONMETVYQ = RegisterMetadata(
    name='NV_R_ONMETVYQ',
    address=0x3930,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_ONMETVYQ_F_CLQYUNYQ = FieldMetadata(
    name='NV_R_ONMETVYQ_F_CLQYUNYQ',
    msb=15,
    lsb=8,
    register=NV_R_ONMETVYQ
)

NV_R_ONMETVYQ_F_CLQYUNYQ_V_GQHYEKHS = ValueMetadata(
    name='NV_R_ONMETVYQ_F_CLQYUNYQ_V_GQHYEKHS',
    value=0,
    field=NV_R_ONMETVYQ_F_CLQYUNYQ
)

NV_R_ONMETVYQ_F_WUIVUAPX = FieldMetadata(
    name='NV_R_ONMETVYQ_F_WUIVUAPX',
    msb=2,
    lsb=0,
    register=NV_R_ONMETVYQ
)

NV_R_ONMETVYQ_F_WUIVUAPX_V_GQHYEKHS = ValueMetadata(
    name='NV_R_ONMETVYQ_F_WUIVUAPX_V_GQHYEKHS',
    value=0,
    field=NV_R_ONMETVYQ_F_WUIVUAPX
)

NV_R_IOBVZPUC = RegisterMetadata(
    name='NV_R_IOBVZPUC',
    address=0x2a5c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_IOBVZPUC_F_CMRFHUES = FieldMetadata(
    name='NV_R_IOBVZPUC_F_CMRFHUES',
    msb=0,
    lsb=0,
    register=NV_R_IOBVZPUC
)

NV_R_IOBVZPUC_F_CMRFHUES_V_WRWAITKO = ValueMetadata(
    name='NV_R_IOBVZPUC_F_CMRFHUES_V_WRWAITKO',
    value=0,
    field=NV_R_IOBVZPUC_F_CMRFHUES
)
NV_R_IOBVZPUC_F_CMRFHUES_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IOBVZPUC_F_CMRFHUES_V_PIUJQBKV',
    value=1,
    field=NV_R_IOBVZPUC_F_CMRFHUES
)

NV_R_IOBVZPUC_F_ZADWJCQJ = FieldMetadata(
    name='NV_R_IOBVZPUC_F_ZADWJCQJ',
    msb=1,
    lsb=1,
    register=NV_R_IOBVZPUC
)

NV_R_IOBVZPUC_F_ZADWJCQJ_V_WRWAITKO = ValueMetadata(
    name='NV_R_IOBVZPUC_F_ZADWJCQJ_V_WRWAITKO',
    value=0,
    field=NV_R_IOBVZPUC_F_ZADWJCQJ
)
NV_R_IOBVZPUC_F_ZADWJCQJ_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IOBVZPUC_F_ZADWJCQJ_V_PIUJQBKV',
    value=1,
    field=NV_R_IOBVZPUC_F_ZADWJCQJ
)

NV_R_IOBVZPUC_F_JZLMVYTL = FieldMetadata(
    name='NV_R_IOBVZPUC_F_JZLMVYTL',
    msb=10,
    lsb=10,
    register=NV_R_IOBVZPUC
)

NV_R_IOBVZPUC_F_JZLMVYTL_V_WRWAITKO = ValueMetadata(
    name='NV_R_IOBVZPUC_F_JZLMVYTL_V_WRWAITKO',
    value=0,
    field=NV_R_IOBVZPUC_F_JZLMVYTL
)
NV_R_IOBVZPUC_F_JZLMVYTL_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IOBVZPUC_F_JZLMVYTL_V_PIUJQBKV',
    value=1,
    field=NV_R_IOBVZPUC_F_JZLMVYTL
)

NV_R_IOBVZPUC_F_ILWKYGOI = FieldMetadata(
    name='NV_R_IOBVZPUC_F_ILWKYGOI',
    msb=11,
    lsb=11,
    register=NV_R_IOBVZPUC
)

NV_R_IOBVZPUC_F_ILWKYGOI_V_WRWAITKO = ValueMetadata(
    name='NV_R_IOBVZPUC_F_ILWKYGOI_V_WRWAITKO',
    value=0,
    field=NV_R_IOBVZPUC_F_ILWKYGOI
)
NV_R_IOBVZPUC_F_ILWKYGOI_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IOBVZPUC_F_ILWKYGOI_V_PIUJQBKV',
    value=1,
    field=NV_R_IOBVZPUC_F_ILWKYGOI
)

NV_R_IOBVZPUC_F_NHIPYYRP = FieldMetadata(
    name='NV_R_IOBVZPUC_F_NHIPYYRP',
    msb=12,
    lsb=12,
    register=NV_R_IOBVZPUC
)

NV_R_IOBVZPUC_F_NHIPYYRP_V_WRWAITKO = ValueMetadata(
    name='NV_R_IOBVZPUC_F_NHIPYYRP_V_WRWAITKO',
    value=0,
    field=NV_R_IOBVZPUC_F_NHIPYYRP
)
NV_R_IOBVZPUC_F_NHIPYYRP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IOBVZPUC_F_NHIPYYRP_V_PIUJQBKV',
    value=1,
    field=NV_R_IOBVZPUC_F_NHIPYYRP
)

NV_R_IOBVZPUC_F_YTCQLNTK = FieldMetadata(
    name='NV_R_IOBVZPUC_F_YTCQLNTK',
    msb=13,
    lsb=13,
    register=NV_R_IOBVZPUC
)

NV_R_IOBVZPUC_F_YTCQLNTK_V_WRWAITKO = ValueMetadata(
    name='NV_R_IOBVZPUC_F_YTCQLNTK_V_WRWAITKO',
    value=0,
    field=NV_R_IOBVZPUC_F_YTCQLNTK
)
NV_R_IOBVZPUC_F_YTCQLNTK_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IOBVZPUC_F_YTCQLNTK_V_PIUJQBKV',
    value=1,
    field=NV_R_IOBVZPUC_F_YTCQLNTK
)

NV_R_IOBVZPUC_F_XMTBCMOU = FieldMetadata(
    name='NV_R_IOBVZPUC_F_XMTBCMOU',
    msb=14,
    lsb=14,
    register=NV_R_IOBVZPUC
)

NV_R_IOBVZPUC_F_XMTBCMOU_V_WRWAITKO = ValueMetadata(
    name='NV_R_IOBVZPUC_F_XMTBCMOU_V_WRWAITKO',
    value=0,
    field=NV_R_IOBVZPUC_F_XMTBCMOU
)
NV_R_IOBVZPUC_F_XMTBCMOU_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IOBVZPUC_F_XMTBCMOU_V_PIUJQBKV',
    value=1,
    field=NV_R_IOBVZPUC_F_XMTBCMOU
)

NV_R_IOBVZPUC_F_OSZUREOY = FieldMetadata(
    name='NV_R_IOBVZPUC_F_OSZUREOY',
    msb=15,
    lsb=15,
    register=NV_R_IOBVZPUC
)

NV_R_IOBVZPUC_F_OSZUREOY_V_WRWAITKO = ValueMetadata(
    name='NV_R_IOBVZPUC_F_OSZUREOY_V_WRWAITKO',
    value=0,
    field=NV_R_IOBVZPUC_F_OSZUREOY
)
NV_R_IOBVZPUC_F_OSZUREOY_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IOBVZPUC_F_OSZUREOY_V_PIUJQBKV',
    value=1,
    field=NV_R_IOBVZPUC_F_OSZUREOY
)

NV_R_IOBVZPUC_F_NONJMGNM = FieldMetadata(
    name='NV_R_IOBVZPUC_F_NONJMGNM',
    msb=2,
    lsb=2,
    register=NV_R_IOBVZPUC
)

NV_R_IOBVZPUC_F_NONJMGNM_V_WRWAITKO = ValueMetadata(
    name='NV_R_IOBVZPUC_F_NONJMGNM_V_WRWAITKO',
    value=0,
    field=NV_R_IOBVZPUC_F_NONJMGNM
)
NV_R_IOBVZPUC_F_NONJMGNM_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IOBVZPUC_F_NONJMGNM_V_PIUJQBKV',
    value=1,
    field=NV_R_IOBVZPUC_F_NONJMGNM
)

NV_R_IOBVZPUC_F_IXEZZHAQ = FieldMetadata(
    name='NV_R_IOBVZPUC_F_IXEZZHAQ',
    msb=3,
    lsb=3,
    register=NV_R_IOBVZPUC
)

NV_R_IOBVZPUC_F_IXEZZHAQ_V_WRWAITKO = ValueMetadata(
    name='NV_R_IOBVZPUC_F_IXEZZHAQ_V_WRWAITKO',
    value=0,
    field=NV_R_IOBVZPUC_F_IXEZZHAQ
)
NV_R_IOBVZPUC_F_IXEZZHAQ_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IOBVZPUC_F_IXEZZHAQ_V_PIUJQBKV',
    value=1,
    field=NV_R_IOBVZPUC_F_IXEZZHAQ
)

NV_R_IOBVZPUC_F_RFSYPPTY = FieldMetadata(
    name='NV_R_IOBVZPUC_F_RFSYPPTY',
    msb=4,
    lsb=4,
    register=NV_R_IOBVZPUC
)

NV_R_IOBVZPUC_F_RFSYPPTY_V_WRWAITKO = ValueMetadata(
    name='NV_R_IOBVZPUC_F_RFSYPPTY_V_WRWAITKO',
    value=0,
    field=NV_R_IOBVZPUC_F_RFSYPPTY
)
NV_R_IOBVZPUC_F_RFSYPPTY_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IOBVZPUC_F_RFSYPPTY_V_PIUJQBKV',
    value=1,
    field=NV_R_IOBVZPUC_F_RFSYPPTY
)

NV_R_IOBVZPUC_F_FLQODFVQ = FieldMetadata(
    name='NV_R_IOBVZPUC_F_FLQODFVQ',
    msb=5,
    lsb=5,
    register=NV_R_IOBVZPUC
)

NV_R_IOBVZPUC_F_FLQODFVQ_V_WRWAITKO = ValueMetadata(
    name='NV_R_IOBVZPUC_F_FLQODFVQ_V_WRWAITKO',
    value=0,
    field=NV_R_IOBVZPUC_F_FLQODFVQ
)
NV_R_IOBVZPUC_F_FLQODFVQ_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IOBVZPUC_F_FLQODFVQ_V_PIUJQBKV',
    value=1,
    field=NV_R_IOBVZPUC_F_FLQODFVQ
)

NV_R_IOBVZPUC_F_QMNZVJUL = FieldMetadata(
    name='NV_R_IOBVZPUC_F_QMNZVJUL',
    msb=6,
    lsb=6,
    register=NV_R_IOBVZPUC
)

NV_R_IOBVZPUC_F_QMNZVJUL_V_WRWAITKO = ValueMetadata(
    name='NV_R_IOBVZPUC_F_QMNZVJUL_V_WRWAITKO',
    value=0,
    field=NV_R_IOBVZPUC_F_QMNZVJUL
)
NV_R_IOBVZPUC_F_QMNZVJUL_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IOBVZPUC_F_QMNZVJUL_V_PIUJQBKV',
    value=1,
    field=NV_R_IOBVZPUC_F_QMNZVJUL
)

NV_R_IOBVZPUC_F_YOHIMBLA = FieldMetadata(
    name='NV_R_IOBVZPUC_F_YOHIMBLA',
    msb=7,
    lsb=7,
    register=NV_R_IOBVZPUC
)

NV_R_IOBVZPUC_F_YOHIMBLA_V_WRWAITKO = ValueMetadata(
    name='NV_R_IOBVZPUC_F_YOHIMBLA_V_WRWAITKO',
    value=0,
    field=NV_R_IOBVZPUC_F_YOHIMBLA
)
NV_R_IOBVZPUC_F_YOHIMBLA_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IOBVZPUC_F_YOHIMBLA_V_PIUJQBKV',
    value=1,
    field=NV_R_IOBVZPUC_F_YOHIMBLA
)

NV_R_IOBVZPUC_F_ATNUPQZB = FieldMetadata(
    name='NV_R_IOBVZPUC_F_ATNUPQZB',
    msb=8,
    lsb=8,
    register=NV_R_IOBVZPUC
)

NV_R_IOBVZPUC_F_ATNUPQZB_V_WRWAITKO = ValueMetadata(
    name='NV_R_IOBVZPUC_F_ATNUPQZB_V_WRWAITKO',
    value=0,
    field=NV_R_IOBVZPUC_F_ATNUPQZB
)
NV_R_IOBVZPUC_F_ATNUPQZB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IOBVZPUC_F_ATNUPQZB_V_PIUJQBKV',
    value=1,
    field=NV_R_IOBVZPUC_F_ATNUPQZB
)

NV_R_IOBVZPUC_F_ALIQBTGP = FieldMetadata(
    name='NV_R_IOBVZPUC_F_ALIQBTGP',
    msb=9,
    lsb=9,
    register=NV_R_IOBVZPUC
)

NV_R_IOBVZPUC_F_ALIQBTGP_V_WRWAITKO = ValueMetadata(
    name='NV_R_IOBVZPUC_F_ALIQBTGP_V_WRWAITKO',
    value=0,
    field=NV_R_IOBVZPUC_F_ALIQBTGP
)
NV_R_IOBVZPUC_F_ALIQBTGP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_IOBVZPUC_F_ALIQBTGP_V_PIUJQBKV',
    value=1,
    field=NV_R_IOBVZPUC_F_ALIQBTGP
)

NV_R_ZYRSFONH = RegisterMetadata(
    name='NV_R_ZYRSFONH',
    address=0x3d48,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_ZYRSFONH_F_MGRBVZVT = FieldMetadata(
    name='NV_R_ZYRSFONH_F_MGRBVZVT',
    msb=0,
    lsb=0,
    register=NV_R_ZYRSFONH
)

NV_R_ZYRSFONH_F_MGRBVZVT_V_GQHYEKHS = ValueMetadata(
    name='NV_R_ZYRSFONH_F_MGRBVZVT_V_GQHYEKHS',
    value=0,
    field=NV_R_ZYRSFONH_F_MGRBVZVT
)

NV_R_ZYRSFONH_F_GRHVXYKZ = FieldMetadata(
    name='NV_R_ZYRSFONH_F_GRHVXYKZ',
    msb=1,
    lsb=1,
    register=NV_R_ZYRSFONH
)

NV_R_ZYRSFONH_F_GRHVXYKZ_V_GQHYEKHS = ValueMetadata(
    name='NV_R_ZYRSFONH_F_GRHVXYKZ_V_GQHYEKHS',
    value=0,
    field=NV_R_ZYRSFONH_F_GRHVXYKZ
)

NV_R_ZYRSFONH_F_XWVXEXKM = FieldMetadata(
    name='NV_R_ZYRSFONH_F_XWVXEXKM',
    msb=2,
    lsb=2,
    register=NV_R_ZYRSFONH
)

NV_R_ZYRSFONH_F_XWVXEXKM_V_GQHYEKHS = ValueMetadata(
    name='NV_R_ZYRSFONH_F_XWVXEXKM_V_GQHYEKHS',
    value=0,
    field=NV_R_ZYRSFONH_F_XWVXEXKM
)

NV_R_ZYRSFONH_F_SHJDZBXE = FieldMetadata(
    name='NV_R_ZYRSFONH_F_SHJDZBXE',
    msb=3,
    lsb=3,
    register=NV_R_ZYRSFONH
)

NV_R_ZYRSFONH_F_SHJDZBXE_V_GQHYEKHS = ValueMetadata(
    name='NV_R_ZYRSFONH_F_SHJDZBXE_V_GQHYEKHS',
    value=0,
    field=NV_R_ZYRSFONH_F_SHJDZBXE
)

NV_R_ZYRSFONH_F_YAGMVJQJ = FieldMetadata(
    name='NV_R_ZYRSFONH_F_YAGMVJQJ',
    msb=4,
    lsb=4,
    register=NV_R_ZYRSFONH
)

NV_R_ZYRSFONH_F_YAGMVJQJ_V_GQHYEKHS = ValueMetadata(
    name='NV_R_ZYRSFONH_F_YAGMVJQJ_V_GQHYEKHS',
    value=0,
    field=NV_R_ZYRSFONH_F_YAGMVJQJ
)

NV_R_LSVFRJCG = RegisterMetadata(
    name='NV_R_LSVFRJCG',
    address=0x3d38,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_LSVFRJCG_F_MGRBVZVT = FieldMetadata(
    name='NV_R_LSVFRJCG_F_MGRBVZVT',
    msb=0,
    lsb=0,
    register=NV_R_LSVFRJCG
)

NV_R_LSVFRJCG_F_MGRBVZVT_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LSVFRJCG_F_MGRBVZVT_V_GQHYEKHS',
    value=0,
    field=NV_R_LSVFRJCG_F_MGRBVZVT
)

NV_R_LSVFRJCG_F_GRHVXYKZ = FieldMetadata(
    name='NV_R_LSVFRJCG_F_GRHVXYKZ',
    msb=1,
    lsb=1,
    register=NV_R_LSVFRJCG
)

NV_R_LSVFRJCG_F_GRHVXYKZ_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LSVFRJCG_F_GRHVXYKZ_V_GQHYEKHS',
    value=0,
    field=NV_R_LSVFRJCG_F_GRHVXYKZ
)

NV_R_LSVFRJCG_F_XWVXEXKM = FieldMetadata(
    name='NV_R_LSVFRJCG_F_XWVXEXKM',
    msb=2,
    lsb=2,
    register=NV_R_LSVFRJCG
)

NV_R_LSVFRJCG_F_XWVXEXKM_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LSVFRJCG_F_XWVXEXKM_V_GQHYEKHS',
    value=0,
    field=NV_R_LSVFRJCG_F_XWVXEXKM
)

NV_R_LSVFRJCG_F_SHJDZBXE = FieldMetadata(
    name='NV_R_LSVFRJCG_F_SHJDZBXE',
    msb=3,
    lsb=3,
    register=NV_R_LSVFRJCG
)

NV_R_LSVFRJCG_F_SHJDZBXE_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LSVFRJCG_F_SHJDZBXE_V_GQHYEKHS',
    value=0,
    field=NV_R_LSVFRJCG_F_SHJDZBXE
)

NV_R_LSVFRJCG_F_YAGMVJQJ = FieldMetadata(
    name='NV_R_LSVFRJCG_F_YAGMVJQJ',
    msb=4,
    lsb=4,
    register=NV_R_LSVFRJCG
)

NV_R_LSVFRJCG_F_YAGMVJQJ_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LSVFRJCG_F_YAGMVJQJ_V_GQHYEKHS',
    value=0,
    field=NV_R_LSVFRJCG_F_YAGMVJQJ
)

NV_R_FZWFVVEC = RegisterMetadata(
    name='NV_R_FZWFVVEC',
    address=0x3b70,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_FZWFVVEC_F_CLQYUNYQ = FieldMetadata(
    name='NV_R_FZWFVVEC_F_CLQYUNYQ',
    msb=15,
    lsb=8,
    register=NV_R_FZWFVVEC
)

NV_R_FZWFVVEC_F_CLQYUNYQ_V_GQHYEKHS = ValueMetadata(
    name='NV_R_FZWFVVEC_F_CLQYUNYQ_V_GQHYEKHS',
    value=0,
    field=NV_R_FZWFVVEC_F_CLQYUNYQ
)

NV_R_FZWFVVEC_F_WUIVUAPX = FieldMetadata(
    name='NV_R_FZWFVVEC_F_WUIVUAPX',
    msb=2,
    lsb=0,
    register=NV_R_FZWFVVEC
)

NV_R_FZWFVVEC_F_WUIVUAPX_V_GQHYEKHS = ValueMetadata(
    name='NV_R_FZWFVVEC_F_WUIVUAPX_V_GQHYEKHS',
    value=0,
    field=NV_R_FZWFVVEC_F_WUIVUAPX
)

NV_R_FNCQVNLR = RegisterMetadata(
    name='NV_R_FNCQVNLR',
    address=0x3b78,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_FNCQVNLR_F_CLQYUNYQ = FieldMetadata(
    name='NV_R_FNCQVNLR_F_CLQYUNYQ',
    msb=15,
    lsb=8,
    register=NV_R_FNCQVNLR
)

NV_R_FNCQVNLR_F_CLQYUNYQ_V_GQHYEKHS = ValueMetadata(
    name='NV_R_FNCQVNLR_F_CLQYUNYQ_V_GQHYEKHS',
    value=0,
    field=NV_R_FNCQVNLR_F_CLQYUNYQ
)

NV_R_FNCQVNLR_F_WUIVUAPX = FieldMetadata(
    name='NV_R_FNCQVNLR_F_WUIVUAPX',
    msb=2,
    lsb=0,
    register=NV_R_FNCQVNLR
)

NV_R_FNCQVNLR_F_WUIVUAPX_V_GQHYEKHS = ValueMetadata(
    name='NV_R_FNCQVNLR_F_WUIVUAPX_V_GQHYEKHS',
    value=0,
    field=NV_R_FNCQVNLR_F_WUIVUAPX
)

NV_R_FHEDDMVN = RegisterMetadata(
    name='NV_R_FHEDDMVN',
    address=0x3b9c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_FHEDDMVN_F_TXDAMMUM = FieldMetadata(
    name='NV_R_FHEDDMVN_F_TXDAMMUM',
    msb=1,
    lsb=1,
    register=NV_R_FHEDDMVN
)

NV_R_FHEDDMVN_F_TXDAMMUM_V_TYLKUYOV = ValueMetadata(
    name='NV_R_FHEDDMVN_F_TXDAMMUM_V_TYLKUYOV',
    value=1,
    field=NV_R_FHEDDMVN_F_TXDAMMUM
)
NV_R_FHEDDMVN_F_TXDAMMUM_V_SQJQPVVW = ValueMetadata(
    name='NV_R_FHEDDMVN_F_TXDAMMUM_V_SQJQPVVW',
    value=0,
    field=NV_R_FHEDDMVN_F_TXDAMMUM
)

NV_R_FHEDDMVN_F_KZJIWLOD = FieldMetadata(
    name='NV_R_FHEDDMVN_F_KZJIWLOD',
    msb=0,
    lsb=0,
    register=NV_R_FHEDDMVN
)

NV_R_FHEDDMVN_F_KZJIWLOD_V_TYLKUYOV = ValueMetadata(
    name='NV_R_FHEDDMVN_F_KZJIWLOD_V_TYLKUYOV',
    value=1,
    field=NV_R_FHEDDMVN_F_KZJIWLOD
)
NV_R_FHEDDMVN_F_KZJIWLOD_V_SQJQPVVW = ValueMetadata(
    name='NV_R_FHEDDMVN_F_KZJIWLOD_V_SQJQPVVW',
    value=0,
    field=NV_R_FHEDDMVN_F_KZJIWLOD
)

NV_R_YEGQOACV = RegisterMetadata(
    name='NV_R_YEGQOACV',
    address=0x3b98,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_YEGQOACV_F_SNTBRTOI = FieldMetadata(
    name='NV_R_YEGQOACV_F_SNTBRTOI',
    msb=6,
    lsb=6,
    register=NV_R_YEGQOACV
)

NV_R_YEGQOACV_F_SNTBRTOI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YEGQOACV_F_SNTBRTOI_V_TYLKUYOV',
    value=1,
    field=NV_R_YEGQOACV_F_SNTBRTOI
)
NV_R_YEGQOACV_F_SNTBRTOI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YEGQOACV_F_SNTBRTOI_V_SQJQPVVW',
    value=0,
    field=NV_R_YEGQOACV_F_SNTBRTOI
)

NV_R_YEGQOACV_F_DRDBUAML = FieldMetadata(
    name='NV_R_YEGQOACV_F_DRDBUAML',
    msb=3,
    lsb=3,
    register=NV_R_YEGQOACV
)

NV_R_YEGQOACV_F_DRDBUAML_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YEGQOACV_F_DRDBUAML_V_TYLKUYOV',
    value=1,
    field=NV_R_YEGQOACV_F_DRDBUAML
)
NV_R_YEGQOACV_F_DRDBUAML_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YEGQOACV_F_DRDBUAML_V_SQJQPVVW',
    value=0,
    field=NV_R_YEGQOACV_F_DRDBUAML
)

NV_R_YEGQOACV_F_WWENSDYL = FieldMetadata(
    name='NV_R_YEGQOACV_F_WWENSDYL',
    msb=4,
    lsb=4,
    register=NV_R_YEGQOACV
)

NV_R_YEGQOACV_F_WWENSDYL_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YEGQOACV_F_WWENSDYL_V_TYLKUYOV',
    value=1,
    field=NV_R_YEGQOACV_F_WWENSDYL
)
NV_R_YEGQOACV_F_WWENSDYL_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YEGQOACV_F_WWENSDYL_V_SQJQPVVW',
    value=0,
    field=NV_R_YEGQOACV_F_WWENSDYL
)

NV_R_YEGQOACV_F_PHAHUOEK = FieldMetadata(
    name='NV_R_YEGQOACV_F_PHAHUOEK',
    msb=5,
    lsb=5,
    register=NV_R_YEGQOACV
)

NV_R_YEGQOACV_F_PHAHUOEK_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YEGQOACV_F_PHAHUOEK_V_TYLKUYOV',
    value=1,
    field=NV_R_YEGQOACV_F_PHAHUOEK
)
NV_R_YEGQOACV_F_PHAHUOEK_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YEGQOACV_F_PHAHUOEK_V_SQJQPVVW',
    value=0,
    field=NV_R_YEGQOACV_F_PHAHUOEK
)

NV_R_YEGQOACV_F_NLZVRXSI = FieldMetadata(
    name='NV_R_YEGQOACV_F_NLZVRXSI',
    msb=2,
    lsb=2,
    register=NV_R_YEGQOACV
)

NV_R_YEGQOACV_F_NLZVRXSI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YEGQOACV_F_NLZVRXSI_V_TYLKUYOV',
    value=1,
    field=NV_R_YEGQOACV_F_NLZVRXSI
)
NV_R_YEGQOACV_F_NLZVRXSI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YEGQOACV_F_NLZVRXSI_V_SQJQPVVW',
    value=0,
    field=NV_R_YEGQOACV_F_NLZVRXSI
)

NV_R_YEGQOACV_F_ANTNPEQQ = FieldMetadata(
    name='NV_R_YEGQOACV_F_ANTNPEQQ',
    msb=1,
    lsb=1,
    register=NV_R_YEGQOACV
)

NV_R_YEGQOACV_F_ANTNPEQQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YEGQOACV_F_ANTNPEQQ_V_TYLKUYOV',
    value=1,
    field=NV_R_YEGQOACV_F_ANTNPEQQ
)
NV_R_YEGQOACV_F_ANTNPEQQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YEGQOACV_F_ANTNPEQQ_V_SQJQPVVW',
    value=0,
    field=NV_R_YEGQOACV_F_ANTNPEQQ
)

NV_R_YEGQOACV_F_PNAAHXLR = FieldMetadata(
    name='NV_R_YEGQOACV_F_PNAAHXLR',
    msb=7,
    lsb=7,
    register=NV_R_YEGQOACV
)

NV_R_YEGQOACV_F_PNAAHXLR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YEGQOACV_F_PNAAHXLR_V_TYLKUYOV',
    value=1,
    field=NV_R_YEGQOACV_F_PNAAHXLR
)
NV_R_YEGQOACV_F_PNAAHXLR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YEGQOACV_F_PNAAHXLR_V_SQJQPVVW',
    value=0,
    field=NV_R_YEGQOACV_F_PNAAHXLR
)

NV_R_YEGQOACV_F_VVJEDIXA = FieldMetadata(
    name='NV_R_YEGQOACV_F_VVJEDIXA',
    msb=10,
    lsb=10,
    register=NV_R_YEGQOACV
)

NV_R_YEGQOACV_F_VVJEDIXA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YEGQOACV_F_VVJEDIXA_V_TYLKUYOV',
    value=1,
    field=NV_R_YEGQOACV_F_VVJEDIXA
)
NV_R_YEGQOACV_F_VVJEDIXA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YEGQOACV_F_VVJEDIXA_V_SQJQPVVW',
    value=0,
    field=NV_R_YEGQOACV_F_VVJEDIXA
)

NV_R_YEGQOACV_F_BNNCMYAS = FieldMetadata(
    name='NV_R_YEGQOACV_F_BNNCMYAS',
    msb=11,
    lsb=11,
    register=NV_R_YEGQOACV
)

NV_R_YEGQOACV_F_BNNCMYAS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YEGQOACV_F_BNNCMYAS_V_TYLKUYOV',
    value=1,
    field=NV_R_YEGQOACV_F_BNNCMYAS
)
NV_R_YEGQOACV_F_BNNCMYAS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YEGQOACV_F_BNNCMYAS_V_SQJQPVVW',
    value=0,
    field=NV_R_YEGQOACV_F_BNNCMYAS
)

NV_R_YEGQOACV_F_HIGBDZXJ = FieldMetadata(
    name='NV_R_YEGQOACV_F_HIGBDZXJ',
    msb=8,
    lsb=8,
    register=NV_R_YEGQOACV
)

NV_R_YEGQOACV_F_HIGBDZXJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YEGQOACV_F_HIGBDZXJ_V_TYLKUYOV',
    value=1,
    field=NV_R_YEGQOACV_F_HIGBDZXJ
)
NV_R_YEGQOACV_F_HIGBDZXJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YEGQOACV_F_HIGBDZXJ_V_SQJQPVVW',
    value=0,
    field=NV_R_YEGQOACV_F_HIGBDZXJ
)

NV_R_YEGQOACV_F_NQQWPJZF = FieldMetadata(
    name='NV_R_YEGQOACV_F_NQQWPJZF',
    msb=0,
    lsb=0,
    register=NV_R_YEGQOACV
)

NV_R_YEGQOACV_F_NQQWPJZF_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YEGQOACV_F_NQQWPJZF_V_TYLKUYOV',
    value=1,
    field=NV_R_YEGQOACV_F_NQQWPJZF
)
NV_R_YEGQOACV_F_NQQWPJZF_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YEGQOACV_F_NQQWPJZF_V_SQJQPVVW',
    value=0,
    field=NV_R_YEGQOACV_F_NQQWPJZF
)

NV_R_YEGQOACV_F_XCZTFHWL = FieldMetadata(
    name='NV_R_YEGQOACV_F_XCZTFHWL',
    msb=9,
    lsb=9,
    register=NV_R_YEGQOACV
)

NV_R_YEGQOACV_F_XCZTFHWL_V_TYLKUYOV = ValueMetadata(
    name='NV_R_YEGQOACV_F_XCZTFHWL_V_TYLKUYOV',
    value=1,
    field=NV_R_YEGQOACV_F_XCZTFHWL
)
NV_R_YEGQOACV_F_XCZTFHWL_V_SQJQPVVW = ValueMetadata(
    name='NV_R_YEGQOACV_F_XCZTFHWL_V_SQJQPVVW',
    value=0,
    field=NV_R_YEGQOACV_F_XCZTFHWL
)

NV_R_NYQEGMBB = RegisterMetadata(
    name='NV_R_NYQEGMBB',
    address=0x3a88,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_NYQEGMBB_F_AHLCMKXX = FieldMetadata(
    name='NV_R_NYQEGMBB_F_AHLCMKXX',
    msb=0,
    lsb=0,
    register=NV_R_NYQEGMBB
)

NV_R_NYQEGMBB_F_AHLCMKXX_V_AABSSBNA = ValueMetadata(
    name='NV_R_NYQEGMBB_F_AHLCMKXX_V_AABSSBNA',
    value=1,
    field=NV_R_NYQEGMBB_F_AHLCMKXX
)
NV_R_NYQEGMBB_F_AHLCMKXX_V_DDVZAPYQ = ValueMetadata(
    name='NV_R_NYQEGMBB_F_AHLCMKXX_V_DDVZAPYQ',
    value=0,
    field=NV_R_NYQEGMBB_F_AHLCMKXX
)

NV_R_NYQEGMBB_F_YAKCLFUX = FieldMetadata(
    name='NV_R_NYQEGMBB_F_YAKCLFUX',
    msb=1,
    lsb=1,
    register=NV_R_NYQEGMBB
)

NV_R_NYQEGMBB_F_YAKCLFUX_V_AABSSBNA = ValueMetadata(
    name='NV_R_NYQEGMBB_F_YAKCLFUX_V_AABSSBNA',
    value=1,
    field=NV_R_NYQEGMBB_F_YAKCLFUX
)
NV_R_NYQEGMBB_F_YAKCLFUX_V_DDVZAPYQ = ValueMetadata(
    name='NV_R_NYQEGMBB_F_YAKCLFUX_V_DDVZAPYQ',
    value=0,
    field=NV_R_NYQEGMBB_F_YAKCLFUX
)

NV_R_NYQEGMBB_F_ZCZOCUFJ = FieldMetadata(
    name='NV_R_NYQEGMBB_F_ZCZOCUFJ',
    msb=2,
    lsb=2,
    register=NV_R_NYQEGMBB
)

NV_R_NYQEGMBB_F_ZCZOCUFJ_V_AABSSBNA = ValueMetadata(
    name='NV_R_NYQEGMBB_F_ZCZOCUFJ_V_AABSSBNA',
    value=1,
    field=NV_R_NYQEGMBB_F_ZCZOCUFJ
)
NV_R_NYQEGMBB_F_ZCZOCUFJ_V_DDVZAPYQ = ValueMetadata(
    name='NV_R_NYQEGMBB_F_ZCZOCUFJ_V_DDVZAPYQ',
    value=0,
    field=NV_R_NYQEGMBB_F_ZCZOCUFJ
)

NV_R_AHKAHBOE = RegisterMetadata(
    name='NV_R_AHKAHBOE',
    address=0x3394,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_AHKAHBOE_F_MNSWOTMH = FieldMetadata(
    name='NV_R_AHKAHBOE_F_MNSWOTMH',
    msb=19,
    lsb=19,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_MNSWOTMH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_MNSWOTMH_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_MNSWOTMH
)
NV_R_AHKAHBOE_F_MNSWOTMH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_MNSWOTMH_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_MNSWOTMH
)

NV_R_AHKAHBOE_F_JHGJMNYQ = FieldMetadata(
    name='NV_R_AHKAHBOE_F_JHGJMNYQ',
    msb=12,
    lsb=12,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_JHGJMNYQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_JHGJMNYQ_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_JHGJMNYQ
)
NV_R_AHKAHBOE_F_JHGJMNYQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_JHGJMNYQ_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_JHGJMNYQ
)

NV_R_AHKAHBOE_F_TWVSOJZO = FieldMetadata(
    name='NV_R_AHKAHBOE_F_TWVSOJZO',
    msb=11,
    lsb=11,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_TWVSOJZO_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_TWVSOJZO_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_TWVSOJZO
)
NV_R_AHKAHBOE_F_TWVSOJZO_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_TWVSOJZO_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_TWVSOJZO
)

NV_R_AHKAHBOE_F_VWSCRVRH = FieldMetadata(
    name='NV_R_AHKAHBOE_F_VWSCRVRH',
    msb=14,
    lsb=14,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_VWSCRVRH_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_VWSCRVRH_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_VWSCRVRH
)
NV_R_AHKAHBOE_F_VWSCRVRH_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_VWSCRVRH_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_VWSCRVRH
)

NV_R_AHKAHBOE_F_THIGODWY = FieldMetadata(
    name='NV_R_AHKAHBOE_F_THIGODWY',
    msb=15,
    lsb=15,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_THIGODWY_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_THIGODWY_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_THIGODWY
)
NV_R_AHKAHBOE_F_THIGODWY_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_THIGODWY_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_THIGODWY
)

NV_R_AHKAHBOE_F_SQRXFRHZ = FieldMetadata(
    name='NV_R_AHKAHBOE_F_SQRXFRHZ',
    msb=13,
    lsb=13,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_SQRXFRHZ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_SQRXFRHZ_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_SQRXFRHZ
)
NV_R_AHKAHBOE_F_SQRXFRHZ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_SQRXFRHZ_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_SQRXFRHZ
)

NV_R_AHKAHBOE_F_ZJMMMYFA = FieldMetadata(
    name='NV_R_AHKAHBOE_F_ZJMMMYFA',
    msb=30,
    lsb=30,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_ZJMMMYFA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_ZJMMMYFA_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_ZJMMMYFA
)
NV_R_AHKAHBOE_F_ZJMMMYFA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_ZJMMMYFA_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_ZJMMMYFA
)

NV_R_AHKAHBOE_F_NMTHBTXS = FieldMetadata(
    name='NV_R_AHKAHBOE_F_NMTHBTXS',
    msb=6,
    lsb=6,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_NMTHBTXS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_NMTHBTXS_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_NMTHBTXS
)
NV_R_AHKAHBOE_F_NMTHBTXS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_NMTHBTXS_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_NMTHBTXS
)

NV_R_AHKAHBOE_F_JEAVMSHI = FieldMetadata(
    name='NV_R_AHKAHBOE_F_JEAVMSHI',
    msb=24,
    lsb=24,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_JEAVMSHI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_JEAVMSHI_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_JEAVMSHI
)
NV_R_AHKAHBOE_F_JEAVMSHI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_JEAVMSHI_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_JEAVMSHI
)

NV_R_AHKAHBOE_F_OLIZCBYA = FieldMetadata(
    name='NV_R_AHKAHBOE_F_OLIZCBYA',
    msb=23,
    lsb=23,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_OLIZCBYA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_OLIZCBYA_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_OLIZCBYA
)
NV_R_AHKAHBOE_F_OLIZCBYA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_OLIZCBYA_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_OLIZCBYA
)

NV_R_AHKAHBOE_F_XSZMCGXI = FieldMetadata(
    name='NV_R_AHKAHBOE_F_XSZMCGXI',
    msb=16,
    lsb=16,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_XSZMCGXI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_XSZMCGXI_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_XSZMCGXI
)
NV_R_AHKAHBOE_F_XSZMCGXI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_XSZMCGXI_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_XSZMCGXI
)

NV_R_AHKAHBOE_F_JOBZBTMV = FieldMetadata(
    name='NV_R_AHKAHBOE_F_JOBZBTMV',
    msb=17,
    lsb=17,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_JOBZBTMV_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_JOBZBTMV_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_JOBZBTMV
)
NV_R_AHKAHBOE_F_JOBZBTMV_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_JOBZBTMV_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_JOBZBTMV
)

NV_R_AHKAHBOE_F_WRDSLEDI = FieldMetadata(
    name='NV_R_AHKAHBOE_F_WRDSLEDI',
    msb=27,
    lsb=27,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_WRDSLEDI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_WRDSLEDI_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_WRDSLEDI
)
NV_R_AHKAHBOE_F_WRDSLEDI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_WRDSLEDI_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_WRDSLEDI
)

NV_R_AHKAHBOE_F_UKWLUGKU = FieldMetadata(
    name='NV_R_AHKAHBOE_F_UKWLUGKU',
    msb=28,
    lsb=28,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_UKWLUGKU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_UKWLUGKU_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_UKWLUGKU
)
NV_R_AHKAHBOE_F_UKWLUGKU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_UKWLUGKU_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_UKWLUGKU
)

NV_R_AHKAHBOE_F_TRYWMIGQ = FieldMetadata(
    name='NV_R_AHKAHBOE_F_TRYWMIGQ',
    msb=25,
    lsb=25,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_TRYWMIGQ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_TRYWMIGQ_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_TRYWMIGQ
)
NV_R_AHKAHBOE_F_TRYWMIGQ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_TRYWMIGQ_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_TRYWMIGQ
)

NV_R_AHKAHBOE_F_LHCOGDWA = FieldMetadata(
    name='NV_R_AHKAHBOE_F_LHCOGDWA',
    msb=21,
    lsb=21,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_LHCOGDWA_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_LHCOGDWA_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_LHCOGDWA
)
NV_R_AHKAHBOE_F_LHCOGDWA_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_LHCOGDWA_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_LHCOGDWA
)

NV_R_AHKAHBOE_F_IHJHOKZP = FieldMetadata(
    name='NV_R_AHKAHBOE_F_IHJHOKZP',
    msb=29,
    lsb=29,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_IHJHOKZP_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_IHJHOKZP_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_IHJHOKZP
)
NV_R_AHKAHBOE_F_IHJHOKZP_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_IHJHOKZP_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_IHJHOKZP
)

NV_R_AHKAHBOE_F_DSYFNOVJ = FieldMetadata(
    name='NV_R_AHKAHBOE_F_DSYFNOVJ',
    msb=2,
    lsb=2,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_DSYFNOVJ_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_DSYFNOVJ_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_DSYFNOVJ
)
NV_R_AHKAHBOE_F_DSYFNOVJ_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_DSYFNOVJ_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_DSYFNOVJ
)

NV_R_AHKAHBOE_F_SAUMSBDG = FieldMetadata(
    name='NV_R_AHKAHBOE_F_SAUMSBDG',
    msb=26,
    lsb=26,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_SAUMSBDG_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_SAUMSBDG_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_SAUMSBDG
)
NV_R_AHKAHBOE_F_SAUMSBDG_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_SAUMSBDG_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_SAUMSBDG
)

NV_R_AHKAHBOE_F_ISSIIJSS = FieldMetadata(
    name='NV_R_AHKAHBOE_F_ISSIIJSS',
    msb=20,
    lsb=20,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_ISSIIJSS_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_ISSIIJSS_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_ISSIIJSS
)
NV_R_AHKAHBOE_F_ISSIIJSS_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_ISSIIJSS_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_ISSIIJSS
)

NV_R_AHKAHBOE_F_OQDTLZES = FieldMetadata(
    name='NV_R_AHKAHBOE_F_OQDTLZES',
    msb=8,
    lsb=8,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_OQDTLZES_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_OQDTLZES_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_OQDTLZES
)
NV_R_AHKAHBOE_F_OQDTLZES_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_OQDTLZES_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_OQDTLZES
)

NV_R_AHKAHBOE_F_IXDOHJET = FieldMetadata(
    name='NV_R_AHKAHBOE_F_IXDOHJET',
    msb=7,
    lsb=7,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_IXDOHJET_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_IXDOHJET_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_IXDOHJET
)
NV_R_AHKAHBOE_F_IXDOHJET_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_IXDOHJET_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_IXDOHJET
)

NV_R_AHKAHBOE_F_WYFDZNYR = FieldMetadata(
    name='NV_R_AHKAHBOE_F_WYFDZNYR',
    msb=4,
    lsb=4,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_WYFDZNYR_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_WYFDZNYR_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_WYFDZNYR
)
NV_R_AHKAHBOE_F_WYFDZNYR_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_WYFDZNYR_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_WYFDZNYR
)

NV_R_AHKAHBOE_F_PEZFMQOX = FieldMetadata(
    name='NV_R_AHKAHBOE_F_PEZFMQOX',
    msb=10,
    lsb=10,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_PEZFMQOX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_PEZFMQOX_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_PEZFMQOX
)
NV_R_AHKAHBOE_F_PEZFMQOX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_PEZFMQOX_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_PEZFMQOX
)

NV_R_AHKAHBOE_F_GATBHVDW = FieldMetadata(
    name='NV_R_AHKAHBOE_F_GATBHVDW',
    msb=31,
    lsb=31,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_GATBHVDW_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_GATBHVDW_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_GATBHVDW
)
NV_R_AHKAHBOE_F_GATBHVDW_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_GATBHVDW_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_GATBHVDW
)

NV_R_AHKAHBOE_F_NSKTUIIC = FieldMetadata(
    name='NV_R_AHKAHBOE_F_NSKTUIIC',
    msb=9,
    lsb=9,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_NSKTUIIC_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_NSKTUIIC_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_NSKTUIIC
)
NV_R_AHKAHBOE_F_NSKTUIIC_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_NSKTUIIC_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_NSKTUIIC
)

NV_R_AHKAHBOE_F_ASYSFFPX = FieldMetadata(
    name='NV_R_AHKAHBOE_F_ASYSFFPX',
    msb=0,
    lsb=0,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_ASYSFFPX_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_ASYSFFPX_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_ASYSFFPX
)
NV_R_AHKAHBOE_F_ASYSFFPX_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_ASYSFFPX_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_ASYSFFPX
)

NV_R_AHKAHBOE_F_CUYZXDTI = FieldMetadata(
    name='NV_R_AHKAHBOE_F_CUYZXDTI',
    msb=1,
    lsb=1,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_CUYZXDTI_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_CUYZXDTI_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_CUYZXDTI
)
NV_R_AHKAHBOE_F_CUYZXDTI_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_CUYZXDTI_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_CUYZXDTI
)

NV_R_AHKAHBOE_F_PWTBFXDT = FieldMetadata(
    name='NV_R_AHKAHBOE_F_PWTBFXDT',
    msb=3,
    lsb=3,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_PWTBFXDT_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_PWTBFXDT_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_PWTBFXDT
)
NV_R_AHKAHBOE_F_PWTBFXDT_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_PWTBFXDT_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_PWTBFXDT
)

NV_R_AHKAHBOE_F_IEYDLCEU = FieldMetadata(
    name='NV_R_AHKAHBOE_F_IEYDLCEU',
    msb=18,
    lsb=18,
    register=NV_R_AHKAHBOE
)

NV_R_AHKAHBOE_F_IEYDLCEU_V_TYLKUYOV = ValueMetadata(
    name='NV_R_AHKAHBOE_F_IEYDLCEU_V_TYLKUYOV',
    value=1,
    field=NV_R_AHKAHBOE_F_IEYDLCEU
)
NV_R_AHKAHBOE_F_IEYDLCEU_V_SQJQPVVW = ValueMetadata(
    name='NV_R_AHKAHBOE_F_IEYDLCEU_V_SQJQPVVW',
    value=0,
    field=NV_R_AHKAHBOE_F_IEYDLCEU
)

