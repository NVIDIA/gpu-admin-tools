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
NV_R_BMZQXPLY = RegisterMetadata(
    name='NV_R_BMZQXPLY',
    address=0x8,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_BMZQXPLY_F_FVXHVRTA = FieldMetadata(
    name='NV_R_BMZQXPLY_F_FVXHVRTA',
    msb=3,
    lsb=3,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_FVXHVRTA_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_FVXHVRTA_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_FVXHVRTA
)
NV_R_BMZQXPLY_F_FVXHVRTA_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_FVXHVRTA_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_FVXHVRTA
)

NV_R_BMZQXPLY_F_BIHMJACV = FieldMetadata(
    name='NV_R_BMZQXPLY_F_BIHMJACV',
    msb=19,
    lsb=19,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_BIHMJACV_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_BIHMJACV_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_BIHMJACV
)
NV_R_BMZQXPLY_F_BIHMJACV_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_BIHMJACV_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_BIHMJACV
)

NV_R_BMZQXPLY_F_LTQGMMAR = FieldMetadata(
    name='NV_R_BMZQXPLY_F_LTQGMMAR',
    msb=16,
    lsb=16,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_LTQGMMAR_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_LTQGMMAR_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_LTQGMMAR
)
NV_R_BMZQXPLY_F_LTQGMMAR_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_LTQGMMAR_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_LTQGMMAR
)

NV_R_BMZQXPLY_F_EMGZLTGZ = FieldMetadata(
    name='NV_R_BMZQXPLY_F_EMGZLTGZ',
    msb=15,
    lsb=8,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_DSGOUXIZ = FieldMetadata(
    name='NV_R_BMZQXPLY_F_DSGOUXIZ',
    msb=5,
    lsb=5,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_DSGOUXIZ_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_DSGOUXIZ_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_DSGOUXIZ
)
NV_R_BMZQXPLY_F_DSGOUXIZ_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_DSGOUXIZ_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_DSGOUXIZ
)

NV_R_BMZQXPLY_F_XZHYIMCX = FieldMetadata(
    name='NV_R_BMZQXPLY_F_XZHYIMCX',
    msb=8,
    lsb=8,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_XZHYIMCX_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_XZHYIMCX_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_XZHYIMCX
)
NV_R_BMZQXPLY_F_XZHYIMCX_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_XZHYIMCX_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_XZHYIMCX
)

NV_R_BMZQXPLY_F_VHSTKTQI = FieldMetadata(
    name='NV_R_BMZQXPLY_F_VHSTKTQI',
    msb=9,
    lsb=9,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_VHSTKTQI_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_VHSTKTQI_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_VHSTKTQI
)
NV_R_BMZQXPLY_F_VHSTKTQI_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_VHSTKTQI_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_VHSTKTQI
)

NV_R_BMZQXPLY_F_EZJCUWDO = FieldMetadata(
    name='NV_R_BMZQXPLY_F_EZJCUWDO',
    msb=10,
    lsb=10,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_EZJCUWDO_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_EZJCUWDO_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_EZJCUWDO
)
NV_R_BMZQXPLY_F_EZJCUWDO_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_EZJCUWDO_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_EZJCUWDO
)

NV_R_BMZQXPLY_F_EQKZSBRA = FieldMetadata(
    name='NV_R_BMZQXPLY_F_EQKZSBRA',
    msb=11,
    lsb=11,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_EQKZSBRA_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_EQKZSBRA_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_EQKZSBRA
)
NV_R_BMZQXPLY_F_EQKZSBRA_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_EQKZSBRA_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_EQKZSBRA
)

NV_R_BMZQXPLY_F_LWKMSBBC = FieldMetadata(
    name='NV_R_BMZQXPLY_F_LWKMSBBC',
    msb=12,
    lsb=12,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_LWKMSBBC_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_LWKMSBBC_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_LWKMSBBC
)
NV_R_BMZQXPLY_F_LWKMSBBC_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_LWKMSBBC_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_LWKMSBBC
)

NV_R_BMZQXPLY_F_OLSUXPVR = FieldMetadata(
    name='NV_R_BMZQXPLY_F_OLSUXPVR',
    msb=13,
    lsb=13,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_OLSUXPVR_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_OLSUXPVR_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_OLSUXPVR
)
NV_R_BMZQXPLY_F_OLSUXPVR_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_OLSUXPVR_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_OLSUXPVR
)

NV_R_BMZQXPLY_F_TNLGDFFZ = FieldMetadata(
    name='NV_R_BMZQXPLY_F_TNLGDFFZ',
    msb=14,
    lsb=14,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_TNLGDFFZ_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_TNLGDFFZ_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_TNLGDFFZ
)
NV_R_BMZQXPLY_F_TNLGDFFZ_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_TNLGDFFZ_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_TNLGDFFZ
)

NV_R_BMZQXPLY_F_RUNXVSFR = FieldMetadata(
    name='NV_R_BMZQXPLY_F_RUNXVSFR',
    msb=15,
    lsb=15,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_RUNXVSFR_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_RUNXVSFR_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_RUNXVSFR
)
NV_R_BMZQXPLY_F_RUNXVSFR_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_RUNXVSFR_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_RUNXVSFR
)

NV_R_BMZQXPLY_F_STDKQCIO = FieldMetadata(
    name='NV_R_BMZQXPLY_F_STDKQCIO',
    msb=24,
    lsb=24,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_STDKQCIO_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_STDKQCIO_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_STDKQCIO
)
NV_R_BMZQXPLY_F_STDKQCIO_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_STDKQCIO_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_STDKQCIO
)

NV_R_BMZQXPLY_F_CGRUAMSF = FieldMetadata(
    name='NV_R_BMZQXPLY_F_CGRUAMSF',
    msb=0,
    lsb=0,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_CGRUAMSF_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_CGRUAMSF_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_CGRUAMSF
)
NV_R_BMZQXPLY_F_CGRUAMSF_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_CGRUAMSF_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_CGRUAMSF
)

NV_R_BMZQXPLY_F_XBOVAYUG = FieldMetadata(
    name='NV_R_BMZQXPLY_F_XBOVAYUG',
    msb=4,
    lsb=4,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_XBOVAYUG_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_XBOVAYUG_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_XBOVAYUG
)
NV_R_BMZQXPLY_F_XBOVAYUG_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_XBOVAYUG_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_XBOVAYUG
)

NV_R_BMZQXPLY_F_JBHWIYMF = FieldMetadata(
    name='NV_R_BMZQXPLY_F_JBHWIYMF',
    msb=22,
    lsb=22,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_JBHWIYMF_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_JBHWIYMF_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_JBHWIYMF
)
NV_R_BMZQXPLY_F_JBHWIYMF_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_JBHWIYMF_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_JBHWIYMF
)

NV_R_BMZQXPLY_F_JEXQSETX = FieldMetadata(
    name='NV_R_BMZQXPLY_F_JEXQSETX',
    msb=23,
    lsb=23,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_JEXQSETX_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_JEXQSETX_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_JEXQSETX
)
NV_R_BMZQXPLY_F_JEXQSETX_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_JEXQSETX_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_JEXQSETX
)

NV_R_BMZQXPLY_F_DHDDCIHI = FieldMetadata(
    name='NV_R_BMZQXPLY_F_DHDDCIHI',
    msb=18,
    lsb=18,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_DHDDCIHI_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_DHDDCIHI_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_DHDDCIHI
)
NV_R_BMZQXPLY_F_DHDDCIHI_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_DHDDCIHI_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_DHDDCIHI
)

NV_R_BMZQXPLY_F_YMGTERLV = FieldMetadata(
    name='NV_R_BMZQXPLY_F_YMGTERLV',
    msb=2,
    lsb=2,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_YMGTERLV_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_YMGTERLV_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_YMGTERLV
)
NV_R_BMZQXPLY_F_YMGTERLV_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_YMGTERLV_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_YMGTERLV
)

NV_R_BMZQXPLY_F_SOSEMTLM = FieldMetadata(
    name='NV_R_BMZQXPLY_F_SOSEMTLM',
    msb=6,
    lsb=6,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_SOSEMTLM_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_SOSEMTLM_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_SOSEMTLM
)
NV_R_BMZQXPLY_F_SOSEMTLM_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_SOSEMTLM_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_SOSEMTLM
)

NV_R_BMZQXPLY_F_KYBUHXRW = FieldMetadata(
    name='NV_R_BMZQXPLY_F_KYBUHXRW',
    msb=7,
    lsb=7,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_KYBUHXRW_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_KYBUHXRW_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_KYBUHXRW
)
NV_R_BMZQXPLY_F_KYBUHXRW_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_KYBUHXRW_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_KYBUHXRW
)

NV_R_BMZQXPLY_F_WXLURVGK = FieldMetadata(
    name='NV_R_BMZQXPLY_F_WXLURVGK',
    msb=30,
    lsb=30,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_WXLURVGK_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_WXLURVGK_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_WXLURVGK
)
NV_R_BMZQXPLY_F_WXLURVGK_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_WXLURVGK_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_WXLURVGK
)

NV_R_BMZQXPLY_F_MDDGHXQZ = FieldMetadata(
    name='NV_R_BMZQXPLY_F_MDDGHXQZ',
    msb=31,
    lsb=31,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_MDDGHXQZ_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_MDDGHXQZ_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_MDDGHXQZ
)
NV_R_BMZQXPLY_F_MDDGHXQZ_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_MDDGHXQZ_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_MDDGHXQZ
)

NV_R_BMZQXPLY_F_GTWXCFNU = FieldMetadata(
    name='NV_R_BMZQXPLY_F_GTWXCFNU',
    msb=1,
    lsb=1,
    register=NV_R_BMZQXPLY
)

NV_R_BMZQXPLY_F_GTWXCFNU_FALSE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_GTWXCFNU_FALSE',
    value=0,
    field=NV_R_BMZQXPLY_F_GTWXCFNU
)
NV_R_BMZQXPLY_F_GTWXCFNU_TRUE = ValueMetadata(
    name='NV_R_BMZQXPLY_F_GTWXCFNU_TRUE',
    value=1,
    field=NV_R_BMZQXPLY_F_GTWXCFNU
)

NV_R_SFYGKRXX = RegisterMetadata(
    name='NV_R_SFYGKRXX',
    address=0x948,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_SFYGKRXX_F_EMGZLTGZ = FieldMetadata(
    name='NV_R_SFYGKRXX_F_EMGZLTGZ',
    msb=31,
    lsb=0,
    register=NV_R_SFYGKRXX
)

NV_R_SFYGKRXX_F_EMGZLTGZ_V_GQHYEKHS = ValueMetadata(
    name='NV_R_SFYGKRXX_F_EMGZLTGZ_V_GQHYEKHS',
    value=0,
    field=NV_R_SFYGKRXX_F_EMGZLTGZ
)
NV_R_SFYGKRXX_F_EMGZLTGZ_V_DCMKEEMG = ValueMetadata(
    name='NV_R_SFYGKRXX_F_EMGZLTGZ_V_DCMKEEMG',
    value=0,
    field=NV_R_SFYGKRXX_F_EMGZLTGZ
)
NV_R_SFYGKRXX_F_EMGZLTGZ_V_ZKHRIEUO = ValueMetadata(
    name='NV_R_SFYGKRXX_F_EMGZLTGZ_V_ZKHRIEUO',
    value=1,
    field=NV_R_SFYGKRXX_F_EMGZLTGZ
)

NV_R_EHYXPRVO = RegisterMetadata(
    name='NV_R_EHYXPRVO',
    address=0x910,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_EHYXPRVO_F_PXIMDYUO = FieldMetadata(
    name='NV_R_EHYXPRVO_F_PXIMDYUO',
    msb=30,
    lsb=0,
    register=NV_R_EHYXPRVO
)

NV_R_EHYXPRVO_F_PXIMDYUO_V_GQHYEKHS = ValueMetadata(
    name='NV_R_EHYXPRVO_F_PXIMDYUO_V_GQHYEKHS',
    value=0,
    field=NV_R_EHYXPRVO_F_PXIMDYUO
)

NV_R_EHYXPRVO_VALID = FieldMetadata(
    name='NV_R_EHYXPRVO_VALID',
    msb=31,
    lsb=31,
    register=NV_R_EHYXPRVO
)

NV_R_EHYXPRVO_VALID_FALSE = ValueMetadata(
    name='NV_R_EHYXPRVO_VALID_FALSE',
    value=0,
    field=NV_R_EHYXPRVO_VALID
)
NV_R_EHYXPRVO_VALID_TRUE = ValueMetadata(
    name='NV_R_EHYXPRVO_VALID_TRUE',
    value=1,
    field=NV_R_EHYXPRVO_VALID
)

NV_R_QDLZFFKQ = RegisterMetadata(
    name='NV_R_QDLZFFKQ',
    address=0x794,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_QDLZFFKQ_F_AQFWRERE = FieldMetadata(
    name='NV_R_QDLZFFKQ_F_AQFWRERE',
    msb=1,
    lsb=1,
    register=NV_R_QDLZFFKQ
)

NV_R_QDLZFFKQ_F_AQFWRERE_V_JMNURMPL = ValueMetadata(
    name='NV_R_QDLZFFKQ_F_AQFWRERE_V_JMNURMPL',
    value=0,
    field=NV_R_QDLZFFKQ_F_AQFWRERE
)

NV_R_QDLZFFKQ_F_NQSKSSVT = FieldMetadata(
    name='NV_R_QDLZFFKQ_F_NQSKSSVT',
    msb=1,
    lsb=1,
    register=NV_R_QDLZFFKQ
)

NV_R_QDLZFFKQ_F_NQSKSSVT_V_JMNURMPL = ValueMetadata(
    name='NV_R_QDLZFFKQ_F_NQSKSSVT_V_JMNURMPL',
    value=0,
    field=NV_R_QDLZFFKQ_F_NQSKSSVT
)

NV_R_QDLZFFKQ_F_CBSZTSEY = FieldMetadata(
    name='NV_R_QDLZFFKQ_F_CBSZTSEY',
    msb=2,
    lsb=2,
    register=NV_R_QDLZFFKQ
)

NV_R_QDLZFFKQ_F_CBSZTSEY_V_GQHYEKHS = ValueMetadata(
    name='NV_R_QDLZFFKQ_F_CBSZTSEY_V_GQHYEKHS',
    value=0,
    field=NV_R_QDLZFFKQ_F_CBSZTSEY
)

NV_R_QDLZFFKQ_F_MAKIJQIB = FieldMetadata(
    name='NV_R_QDLZFFKQ_F_MAKIJQIB',
    msb=0,
    lsb=0,
    register=NV_R_QDLZFFKQ
)

NV_R_QDLZFFKQ_F_MAKIJQIB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_QDLZFFKQ_F_MAKIJQIB_V_GQHYEKHS',
    value=0,
    field=NV_R_QDLZFFKQ_F_MAKIJQIB
)

NV_R_QDLZFFKQ_F_WXNDRGMV = FieldMetadata(
    name='NV_R_QDLZFFKQ_F_WXNDRGMV',
    msb=2,
    lsb=2,
    register=NV_R_QDLZFFKQ
)

NV_R_QDLZFFKQ_F_WXNDRGMV_V_GQHYEKHS = ValueMetadata(
    name='NV_R_QDLZFFKQ_F_WXNDRGMV_V_GQHYEKHS',
    value=0,
    field=NV_R_QDLZFFKQ_F_WXNDRGMV
)

NV_R_QDLZFFKQ_F_HGPEBRTD = FieldMetadata(
    name='NV_R_QDLZFFKQ_F_HGPEBRTD',
    msb=4,
    lsb=4,
    register=NV_R_QDLZFFKQ
)

NV_R_QDLZFFKQ_F_HGPEBRTD_V_JMNURMPL = ValueMetadata(
    name='NV_R_QDLZFFKQ_F_HGPEBRTD_V_JMNURMPL',
    value=0,
    field=NV_R_QDLZFFKQ_F_HGPEBRTD
)

NV_R_QDLZFFKQ_F_TTDEBOJC = FieldMetadata(
    name='NV_R_QDLZFFKQ_F_TTDEBOJC',
    msb=5,
    lsb=5,
    register=NV_R_QDLZFFKQ
)

NV_R_QDLZFFKQ_F_TTDEBOJC_V_GQHYEKHS = ValueMetadata(
    name='NV_R_QDLZFFKQ_F_TTDEBOJC_V_GQHYEKHS',
    value=0,
    field=NV_R_QDLZFFKQ_F_TTDEBOJC
)

NV_R_QDLZFFKQ_F_WXLVGPLS = FieldMetadata(
    name='NV_R_QDLZFFKQ_F_WXLVGPLS',
    msb=3,
    lsb=3,
    register=NV_R_QDLZFFKQ
)

NV_R_QDLZFFKQ_F_WXLVGPLS_V_GQHYEKHS = ValueMetadata(
    name='NV_R_QDLZFFKQ_F_WXLVGPLS_V_GQHYEKHS',
    value=0,
    field=NV_R_QDLZFFKQ_F_WXLVGPLS
)

NV_R_QDLZFFKQ_F_GZUNECFA = FieldMetadata(
    name='NV_R_QDLZFFKQ_F_GZUNECFA',
    msb=0,
    lsb=0,
    register=NV_R_QDLZFFKQ
)

NV_R_QDLZFFKQ_F_GZUNECFA_V_GQHYEKHS = ValueMetadata(
    name='NV_R_QDLZFFKQ_F_GZUNECFA_V_GQHYEKHS',
    value=0,
    field=NV_R_QDLZFFKQ_F_GZUNECFA
)

NV_R_FASQCAHH = RegisterMetadata(
    name='NV_R_FASQCAHH',
    address=0x9e0,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_FASQCAHH_VALID = FieldMetadata(
    name='NV_R_FASQCAHH_VALID',
    msb=31,
    lsb=31,
    register=NV_R_FASQCAHH
)

NV_R_FASQCAHH_VALID_FALSE = ValueMetadata(
    name='NV_R_FASQCAHH_VALID_FALSE',
    value=0,
    field=NV_R_FASQCAHH_VALID
)
NV_R_FASQCAHH_VALID_V_MDRPRBDC = ValueMetadata(
    name='NV_R_FASQCAHH_VALID_V_MDRPRBDC',
    value=1,
    field=NV_R_FASQCAHH_VALID
)

NV_R_EVIISCRZ = RegisterMetadata(
    name='NV_R_EVIISCRZ',
    address=0x900,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_EVIISCRZ_F_PXIMDYUO = FieldMetadata(
    name='NV_R_EVIISCRZ_F_PXIMDYUO',
    msb=29,
    lsb=0,
    register=NV_R_EVIISCRZ
)

NV_R_EVIISCRZ_F_PXIMDYUO_V_GQHYEKHS = ValueMetadata(
    name='NV_R_EVIISCRZ_F_PXIMDYUO_V_GQHYEKHS',
    value=0,
    field=NV_R_EVIISCRZ_F_PXIMDYUO
)

NV_R_EVIISCRZ_VALID = FieldMetadata(
    name='NV_R_EVIISCRZ_VALID',
    msb=31,
    lsb=31,
    register=NV_R_EVIISCRZ
)

NV_R_EVIISCRZ_VALID_FALSE = ValueMetadata(
    name='NV_R_EVIISCRZ_VALID_FALSE',
    value=0,
    field=NV_R_EVIISCRZ_VALID
)
NV_R_EVIISCRZ_VALID_TRUE = ValueMetadata(
    name='NV_R_EVIISCRZ_VALID_TRUE',
    value=1,
    field=NV_R_EVIISCRZ_VALID
)

NV_R_EVIISCRZ_WRITE = FieldMetadata(
    name='NV_R_EVIISCRZ_WRITE',
    msb=30,
    lsb=30,
    register=NV_R_EVIISCRZ
)

NV_R_EVIISCRZ_WRITE_FALSE = ValueMetadata(
    name='NV_R_EVIISCRZ_WRITE_FALSE',
    value=0,
    field=NV_R_EVIISCRZ_WRITE
)
NV_R_EVIISCRZ_WRITE_TRUE = ValueMetadata(
    name='NV_R_EVIISCRZ_WRITE_TRUE',
    value=1,
    field=NV_R_EVIISCRZ_WRITE
)

