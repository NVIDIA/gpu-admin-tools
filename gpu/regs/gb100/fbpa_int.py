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
from gpu.regs.gh100.fbpa_int import (
    NV_A_KSKYOZUD,
    NV_A_KSKYOZUD_F_ONWVXYRY,
    NV_A_KSKYOZUD_F_ONWVXYRY_V_GQHYEKHS,
    NV_A_KSKYOZUD_F_SNZGXGUL,
    NV_A_KSKYOZUD_F_SNZGXGUL_V_GQHYEKHS,
    NV_A_XSYHDVCL,
    NV_A_XSYHDVCL_F_ONWVXYRY,
    NV_A_XSYHDVCL_F_ONWVXYRY_V_GQHYEKHS,
    NV_A_XSYHDVCL_F_SNZGXGUL,
    NV_A_XSYHDVCL_F_SNZGXGUL_V_GQHYEKHS,
    NV_A_XSYHDVCL_F_WMYUJFVG,
    NV_A_XSYHDVCL_F_WMYUJFVG_V_GQHYEKHS,
    NV_A_RFGFXXJP,
    NV_A_RFGFXXJP_F_SNZGXGUL,
    NV_A_RFGFXXJP_F_SNZGXGUL_V_GQHYEKHS,
    NV_A_RXIHRXWD,
    NV_A_RXIHRXWD_F_SNZGXGUL,
    NV_A_RXIHRXWD_F_SNZGXGUL_V_GQHYEKHS,
    NV_A_RXIHRXWD_F_YAHKBQSG,
    NV_A_RXIHRXWD_F_YAHKBQSG_V_GQHYEKHS,
    NV_A_RXIHRXWD_F_VADPLWWG,
    NV_A_RXIHRXWD_F_VADPLWWG_V_GQHYEKHS,
    NV_A_RXIHRXWD_F_SLHBJMTE,
    NV_A_RXIHRXWD_F_SLHBJMTE_V_GQHYEKHS,
    NV_A_RXIHRXWD_F_CLSFQYAN,
    NV_A_RXIHRXWD_F_CLSFQYAN_V_GQHYEKHS,
    NV_A_RFGFXXJP_F_YAHKBQSG,
    NV_A_RFGFXXJP_F_YAHKBQSG_V_GQHYEKHS,
    NV_A_SGDNNLKJ,
    NV_A_SGDNNLKJ_F_SNZGXGUL,
    NV_A_SGDNNLKJ_F_SNZGXGUL_V_GQHYEKHS,
    NV_A_SGDNNLKJ_F_YAHKBQSG,
    NV_A_SGDNNLKJ_F_YAHKBQSG_V_GQHYEKHS,
    NV_A_SGDNNLKJ_F_VADPLWWG,
    NV_A_SGDNNLKJ_F_VADPLWWG_V_GQHYEKHS,
    NV_A_SGDNNLKJ_F_SLHBJMTE,
    NV_A_SGDNNLKJ_F_SLHBJMTE_V_GQHYEKHS,
    NV_A_SGDNNLKJ_F_CLSFQYAN,
    NV_A_SGDNNLKJ_F_CLSFQYAN_V_GQHYEKHS,
    NV_A_RFGFXXJP_F_VADPLWWG,
    NV_A_RFGFXXJP_F_VADPLWWG_V_GQHYEKHS,
    NV_A_RFGFXXJP_F_SLHBJMTE,
    NV_A_RFGFXXJP_F_SLHBJMTE_V_GQHYEKHS,
    NV_A_RFGFXXJP_F_CLSFQYAN,
    NV_A_RFGFXXJP_F_CLSFQYAN_V_GQHYEKHS,
    NV_A_MALLPWTC,
    NV_A_MALLPWTC_F_SNZGXGUL,
    NV_A_MALLPWTC_F_SNZGXGUL_V_GQHYEKHS,
    NV_A_MALLPWTC_F_YAHKBQSG,
    NV_A_MALLPWTC_F_YAHKBQSG_V_GQHYEKHS,
    NV_A_MALLPWTC_F_VADPLWWG,
    NV_A_MALLPWTC_F_VADPLWWG_V_GQHYEKHS,
    NV_A_MALLPWTC_F_SLHBJMTE,
    NV_A_MALLPWTC_F_SLHBJMTE_V_GQHYEKHS,
    NV_A_MALLPWTC_F_CLSFQYAN,
    NV_A_MALLPWTC_F_CLSFQYAN_V_GQHYEKHS,
    NV_A_RDZIUWKU,
    NV_A_RDZIUWKU_F_ONWVXYRY,
    NV_A_RDZIUWKU_F_ONWVXYRY_V_GQHYEKHS,
    NV_A_RDZIUWKU_F_SNZGXGUL,
    NV_A_RDZIUWKU_F_SNZGXGUL_V_GQHYEKHS,
    NV_A_RDZIUWKU_F_WMYUJFVG,
    NV_A_RDZIUWKU_F_WMYUJFVG_V_GQHYEKHS,
    NV_A_KSKYOZUD_F_WMYUJFVG,
    NV_A_KSKYOZUD_F_WMYUJFVG_V_GQHYEKHS,
    NV_A_UXHPZDSJ,
    NV_A_UXHPZDSJ_F_ONWVXYRY,
    NV_A_UXHPZDSJ_F_ONWVXYRY_V_GQHYEKHS,
    NV_A_UXHPZDSJ_F_SNZGXGUL,
    NV_A_UXHPZDSJ_F_SNZGXGUL_V_GQHYEKHS,
    NV_A_UXHPZDSJ_F_WMYUJFVG,
    NV_A_UXHPZDSJ_F_WMYUJFVG_V_GQHYEKHS,
    NV_A_AWNLVVOC,
    NV_A_AWNLVVOC_F_HWFYBREO,
    NV_A_AWNLVVOC_F_HWFYBREO_V_GQHYEKHS,
    NV_A_HCOTFNDZ,
    NV_A_HCOTFNDZ_F_NUEJUWHY,
    NV_A_HCOTFNDZ_F_NUEJUWHY_V_GQHYEKHS,
    NV_A_HCOTFNDZ_F_SBDIUQGA,
    NV_A_HCOTFNDZ_F_SBDIUQGA_V_GQHYEKHS,
    NV_A_HCOTFNDZ_F_GUBRWLLH,
    NV_A_HCOTFNDZ_F_GUBRWLLH_V_GQHYEKHS,
    NV_A_HCOTFNDZ_F_VUHHTIYD,
    NV_A_HCOTFNDZ_F_AKNLWQOP,
    NV_A_HCOTFNDZ_F_AKNLWQOP_V_GQHYEKHS,
    NV_A_HCOTFNDZ_F_VUHHTIYD_V_GQHYEKHS,
    NV_A_HCOTFNDZ_F_JGDYKRTC,
    NV_A_HCOTFNDZ_F_JGDYKRTC_V_NMZBWDAB,
    NV_A_HCOTFNDZ_F_JGDYKRTC_V_RHDXTSMA,
    NV_A_HCOTFNDZ_F_JGDYKRTC_V_NSDPVTAV,
    NV_A_NYBWZNTJ,
    NV_A_NYBWZNTJ_F_HWFYBREO,
    NV_A_NYBWZNTJ_F_HWFYBREO_V_GQHYEKHS,
    NV_A_DDAHJXIM,
    NV_A_DDAHJXIM_F_AJYGETZL,
    NV_A_DDAHJXIM_F_AJYGETZL_V_DOGBFDTH,
    NV_A_DDAHJXIM_F_AJYGETZL_V_PIUJQBKV,
    NV_A_DDAHJXIM_F_JFFCFQPN,
    NV_A_DDAHJXIM_F_JFFCFQPN_V_DOGBFDTH,
    NV_A_DDAHJXIM_F_JFFCFQPN_V_PIUJQBKV,
    NV_A_DDAHJXIM_F_EMORFZBS,
    NV_A_DDAHJXIM_F_EMORFZBS_V_DOGBFDTH,
    NV_A_DDAHJXIM_F_EMORFZBS_V_PIUJQBKV,
    NV_A_DDAHJXIM_F_RYPFIBFW,
    NV_A_DDAHJXIM_F_RYPFIBFW_V_DOGBFDTH,
    NV_A_DDAHJXIM_F_RYPFIBFW_V_PIUJQBKV,
    NV_A_DDAHJXIM_F_WKXUQINF,
    NV_A_DDAHJXIM_F_WKXUQINF_V_DOGBFDTH,
    NV_A_DDAHJXIM_F_WKXUQINF_V_PIUJQBKV,
)

# Device definitions
NV_D_BXFCDUCX = DeviceMetadata(
    name='NV_D_BXFCDUCX',
    start_address=0x900000,
    end_address=0x903fff
)

# Array definitions
NV_A_KAKYCVUL = ArrayMetadata(
    name='NV_A_KAKYCVUL',
    base_address=0x24a0,
    stride=4,
    size=4,
    zero_based=True
)

NV_A_KAKYCVUL_F_RPLZNTNW = FieldMetadata(
    name='NV_A_KAKYCVUL_F_RPLZNTNW',
    msb=3,
    lsb=3,
    register=NV_A_KAKYCVUL
)

NV_A_KAKYCVUL_F_RPLZNTNW_V_GQHYEKHS = ValueMetadata(
    name='NV_A_KAKYCVUL_F_RPLZNTNW_V_GQHYEKHS',
    value=0,
    field=NV_A_KAKYCVUL_F_RPLZNTNW
)

NV_A_KAKYCVUL_F_QDKQSLJS = FieldMetadata(
    name='NV_A_KAKYCVUL_F_QDKQSLJS',
    msb=9,
    lsb=9,
    register=NV_A_KAKYCVUL
)

NV_A_KAKYCVUL_F_QDKQSLJS_V_GQHYEKHS = ValueMetadata(
    name='NV_A_KAKYCVUL_F_QDKQSLJS_V_GQHYEKHS',
    value=0,
    field=NV_A_KAKYCVUL_F_QDKQSLJS
)

NV_A_KAKYCVUL_F_MYDGBKIO = FieldMetadata(
    name='NV_A_KAKYCVUL_F_MYDGBKIO',
    msb=2,
    lsb=0,
    register=NV_A_KAKYCVUL
)

NV_A_KAKYCVUL_F_MYDGBKIO_V_GQHYEKHS = ValueMetadata(
    name='NV_A_KAKYCVUL_F_MYDGBKIO_V_GQHYEKHS',
    value=0,
    field=NV_A_KAKYCVUL_F_MYDGBKIO
)

NV_A_KAKYCVUL_F_AATGDVNZ = FieldMetadata(
    name='NV_A_KAKYCVUL_F_AATGDVNZ',
    msb=7,
    lsb=6,
    register=NV_A_KAKYCVUL
)

NV_A_KAKYCVUL_F_AATGDVNZ_V_JKXSHNLV = ValueMetadata(
    name='NV_A_KAKYCVUL_F_AATGDVNZ_V_JKXSHNLV',
    value=1,
    field=NV_A_KAKYCVUL_F_AATGDVNZ
)
NV_A_KAKYCVUL_F_AATGDVNZ_V_SDCCMLBB = ValueMetadata(
    name='NV_A_KAKYCVUL_F_AATGDVNZ_V_SDCCMLBB',
    value=3,
    field=NV_A_KAKYCVUL_F_AATGDVNZ
)
NV_A_KAKYCVUL_F_AATGDVNZ_V_BLJQBDXK = ValueMetadata(
    name='NV_A_KAKYCVUL_F_AATGDVNZ_V_BLJQBDXK',
    value=2,
    field=NV_A_KAKYCVUL_F_AATGDVNZ
)
NV_A_KAKYCVUL_F_AATGDVNZ_V_DDVZAPYQ = ValueMetadata(
    name='NV_A_KAKYCVUL_F_AATGDVNZ_V_DDVZAPYQ',
    value=0,
    field=NV_A_KAKYCVUL_F_AATGDVNZ
)
NV_A_KAKYCVUL_F_AATGDVNZ_V_YKKIGEDW = ValueMetadata(
    name='NV_A_KAKYCVUL_F_AATGDVNZ_V_YKKIGEDW',
    value=0,
    field=NV_A_KAKYCVUL_F_AATGDVNZ
)

NV_A_KAKYCVUL_F_JBSRGFLD = FieldMetadata(
    name='NV_A_KAKYCVUL_F_JBSRGFLD',
    msb=8,
    lsb=8,
    register=NV_A_KAKYCVUL
)

NV_A_KAKYCVUL_F_JBSRGFLD_V_GQHYEKHS = ValueMetadata(
    name='NV_A_KAKYCVUL_F_JBSRGFLD_V_GQHYEKHS',
    value=0,
    field=NV_A_KAKYCVUL_F_JBSRGFLD
)

NV_A_OCRCKSGD = ArrayMetadata(
    name='NV_A_OCRCKSGD',
    base_address=0x22e0,
    stride=4,
    size=4,
    zero_based=True
)

NV_A_OCRCKSGD_F_RPLZNTNW = FieldMetadata(
    name='NV_A_OCRCKSGD_F_RPLZNTNW',
    msb=3,
    lsb=3,
    register=NV_A_OCRCKSGD
)

NV_A_OCRCKSGD_F_RPLZNTNW_V_GQHYEKHS = ValueMetadata(
    name='NV_A_OCRCKSGD_F_RPLZNTNW_V_GQHYEKHS',
    value=0,
    field=NV_A_OCRCKSGD_F_RPLZNTNW
)

NV_A_OCRCKSGD_F_QDKQSLJS = FieldMetadata(
    name='NV_A_OCRCKSGD_F_QDKQSLJS',
    msb=9,
    lsb=9,
    register=NV_A_OCRCKSGD
)

NV_A_OCRCKSGD_F_QDKQSLJS_V_GQHYEKHS = ValueMetadata(
    name='NV_A_OCRCKSGD_F_QDKQSLJS_V_GQHYEKHS',
    value=0,
    field=NV_A_OCRCKSGD_F_QDKQSLJS
)

NV_A_OCRCKSGD_F_MYDGBKIO = FieldMetadata(
    name='NV_A_OCRCKSGD_F_MYDGBKIO',
    msb=2,
    lsb=0,
    register=NV_A_OCRCKSGD
)

NV_A_OCRCKSGD_F_MYDGBKIO_V_GQHYEKHS = ValueMetadata(
    name='NV_A_OCRCKSGD_F_MYDGBKIO_V_GQHYEKHS',
    value=0,
    field=NV_A_OCRCKSGD_F_MYDGBKIO
)

NV_A_OCRCKSGD_F_AATGDVNZ = FieldMetadata(
    name='NV_A_OCRCKSGD_F_AATGDVNZ',
    msb=7,
    lsb=6,
    register=NV_A_OCRCKSGD
)

NV_A_OCRCKSGD_F_AATGDVNZ_V_JKXSHNLV = ValueMetadata(
    name='NV_A_OCRCKSGD_F_AATGDVNZ_V_JKXSHNLV',
    value=1,
    field=NV_A_OCRCKSGD_F_AATGDVNZ
)
NV_A_OCRCKSGD_F_AATGDVNZ_V_SDCCMLBB = ValueMetadata(
    name='NV_A_OCRCKSGD_F_AATGDVNZ_V_SDCCMLBB',
    value=3,
    field=NV_A_OCRCKSGD_F_AATGDVNZ
)
NV_A_OCRCKSGD_F_AATGDVNZ_V_BLJQBDXK = ValueMetadata(
    name='NV_A_OCRCKSGD_F_AATGDVNZ_V_BLJQBDXK',
    value=2,
    field=NV_A_OCRCKSGD_F_AATGDVNZ
)
NV_A_OCRCKSGD_F_AATGDVNZ_V_DDVZAPYQ = ValueMetadata(
    name='NV_A_OCRCKSGD_F_AATGDVNZ_V_DDVZAPYQ',
    value=0,
    field=NV_A_OCRCKSGD_F_AATGDVNZ
)
NV_A_OCRCKSGD_F_AATGDVNZ_V_YKKIGEDW = ValueMetadata(
    name='NV_A_OCRCKSGD_F_AATGDVNZ_V_YKKIGEDW',
    value=0,
    field=NV_A_OCRCKSGD_F_AATGDVNZ
)

NV_A_OCRCKSGD_F_JBSRGFLD = FieldMetadata(
    name='NV_A_OCRCKSGD_F_JBSRGFLD',
    msb=8,
    lsb=8,
    register=NV_A_OCRCKSGD
)

NV_A_OCRCKSGD_F_JBSRGFLD_V_GQHYEKHS = ValueMetadata(
    name='NV_A_OCRCKSGD_F_JBSRGFLD_V_GQHYEKHS',
    value=0,
    field=NV_A_OCRCKSGD_F_JBSRGFLD
)

NV_A_ZRHMNQOZ = ArrayMetadata(
    name='NV_A_ZRHMNQOZ',
    base_address=0x24b0,
    stride=4,
    size=4,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_A_ZRHMNQOZ_F_RPLZNTNW = FieldMetadata(
    name='NV_A_ZRHMNQOZ_F_RPLZNTNW',
    msb=3,
    lsb=3,
    register=NV_A_ZRHMNQOZ
)

NV_A_ZRHMNQOZ_F_RPLZNTNW_V_GQHYEKHS = ValueMetadata(
    name='NV_A_ZRHMNQOZ_F_RPLZNTNW_V_GQHYEKHS',
    value=0,
    field=NV_A_ZRHMNQOZ_F_RPLZNTNW
)

NV_A_ZRHMNQOZ_F_QDKQSLJS = FieldMetadata(
    name='NV_A_ZRHMNQOZ_F_QDKQSLJS',
    msb=5,
    lsb=5,
    register=NV_A_ZRHMNQOZ
)

NV_A_ZRHMNQOZ_F_QDKQSLJS_V_GQHYEKHS = ValueMetadata(
    name='NV_A_ZRHMNQOZ_F_QDKQSLJS_V_GQHYEKHS',
    value=0,
    field=NV_A_ZRHMNQOZ_F_QDKQSLJS
)

NV_A_ZRHMNQOZ_F_MYDGBKIO = FieldMetadata(
    name='NV_A_ZRHMNQOZ_F_MYDGBKIO',
    msb=2,
    lsb=0,
    register=NV_A_ZRHMNQOZ
)

NV_A_ZRHMNQOZ_F_MYDGBKIO_V_GQHYEKHS = ValueMetadata(
    name='NV_A_ZRHMNQOZ_F_MYDGBKIO_V_GQHYEKHS',
    value=0,
    field=NV_A_ZRHMNQOZ_F_MYDGBKIO
)

NV_A_ZRHMNQOZ_F_JBSRGFLD = FieldMetadata(
    name='NV_A_ZRHMNQOZ_F_JBSRGFLD',
    msb=4,
    lsb=4,
    register=NV_A_ZRHMNQOZ
)

NV_A_ZRHMNQOZ_F_JBSRGFLD_V_GQHYEKHS = ValueMetadata(
    name='NV_A_ZRHMNQOZ_F_JBSRGFLD_V_GQHYEKHS',
    value=0,
    field=NV_A_ZRHMNQOZ_F_JBSRGFLD
)

NV_A_GQNBAIWL = ArrayMetadata(
    name='NV_A_GQNBAIWL',
    base_address=0x22d0,
    stride=4,
    size=4,
    zero_based=True
)

NV_A_GQNBAIWL_F_RPLZNTNW = FieldMetadata(
    name='NV_A_GQNBAIWL_F_RPLZNTNW',
    msb=3,
    lsb=3,
    register=NV_A_GQNBAIWL
)

NV_A_GQNBAIWL_F_RPLZNTNW_V_GQHYEKHS = ValueMetadata(
    name='NV_A_GQNBAIWL_F_RPLZNTNW_V_GQHYEKHS',
    value=0,
    field=NV_A_GQNBAIWL_F_RPLZNTNW
)

NV_A_GQNBAIWL_F_QDKQSLJS = FieldMetadata(
    name='NV_A_GQNBAIWL_F_QDKQSLJS',
    msb=9,
    lsb=9,
    register=NV_A_GQNBAIWL
)

NV_A_GQNBAIWL_F_QDKQSLJS_V_GQHYEKHS = ValueMetadata(
    name='NV_A_GQNBAIWL_F_QDKQSLJS_V_GQHYEKHS',
    value=0,
    field=NV_A_GQNBAIWL_F_QDKQSLJS
)

NV_A_GQNBAIWL_F_MYDGBKIO = FieldMetadata(
    name='NV_A_GQNBAIWL_F_MYDGBKIO',
    msb=2,
    lsb=0,
    register=NV_A_GQNBAIWL
)

NV_A_GQNBAIWL_F_MYDGBKIO_V_GQHYEKHS = ValueMetadata(
    name='NV_A_GQNBAIWL_F_MYDGBKIO_V_GQHYEKHS',
    value=0,
    field=NV_A_GQNBAIWL_F_MYDGBKIO
)

NV_A_GQNBAIWL_F_AATGDVNZ = FieldMetadata(
    name='NV_A_GQNBAIWL_F_AATGDVNZ',
    msb=7,
    lsb=6,
    register=NV_A_GQNBAIWL
)

NV_A_GQNBAIWL_F_AATGDVNZ_V_JKXSHNLV = ValueMetadata(
    name='NV_A_GQNBAIWL_F_AATGDVNZ_V_JKXSHNLV',
    value=1,
    field=NV_A_GQNBAIWL_F_AATGDVNZ
)
NV_A_GQNBAIWL_F_AATGDVNZ_V_SDCCMLBB = ValueMetadata(
    name='NV_A_GQNBAIWL_F_AATGDVNZ_V_SDCCMLBB',
    value=3,
    field=NV_A_GQNBAIWL_F_AATGDVNZ
)
NV_A_GQNBAIWL_F_AATGDVNZ_V_BLJQBDXK = ValueMetadata(
    name='NV_A_GQNBAIWL_F_AATGDVNZ_V_BLJQBDXK',
    value=2,
    field=NV_A_GQNBAIWL_F_AATGDVNZ
)
NV_A_GQNBAIWL_F_AATGDVNZ_V_DDVZAPYQ = ValueMetadata(
    name='NV_A_GQNBAIWL_F_AATGDVNZ_V_DDVZAPYQ',
    value=0,
    field=NV_A_GQNBAIWL_F_AATGDVNZ
)
NV_A_GQNBAIWL_F_AATGDVNZ_V_YKKIGEDW = ValueMetadata(
    name='NV_A_GQNBAIWL_F_AATGDVNZ_V_YKKIGEDW',
    value=0,
    field=NV_A_GQNBAIWL_F_AATGDVNZ
)

NV_A_GQNBAIWL_F_JBSRGFLD = FieldMetadata(
    name='NV_A_GQNBAIWL_F_JBSRGFLD',
    msb=8,
    lsb=8,
    register=NV_A_GQNBAIWL
)

NV_A_GQNBAIWL_F_JBSRGFLD_V_GQHYEKHS = ValueMetadata(
    name='NV_A_GQNBAIWL_F_JBSRGFLD_V_GQHYEKHS',
    value=0,
    field=NV_A_GQNBAIWL_F_JBSRGFLD
)

NV_A_HACMEFIF = ArrayMetadata(
    name='NV_A_HACMEFIF',
    base_address=0x2280,
    stride=4,
    size=4,
    zero_based=True
)

NV_A_HACMEFIF_F_ONWVXYRY = FieldMetadata(
    name='NV_A_HACMEFIF_F_ONWVXYRY',
    msb=31,
    lsb=28,
    register=NV_A_HACMEFIF
)

NV_A_HACMEFIF_F_ONWVXYRY_V_GQHYEKHS = ValueMetadata(
    name='NV_A_HACMEFIF_F_ONWVXYRY_V_GQHYEKHS',
    value=0,
    field=NV_A_HACMEFIF_F_ONWVXYRY
)

NV_A_HACMEFIF_F_SNZGXGUL = FieldMetadata(
    name='NV_A_HACMEFIF_F_SNZGXGUL',
    msb=10,
    lsb=0,
    register=NV_A_HACMEFIF
)

NV_A_HACMEFIF_F_SNZGXGUL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_HACMEFIF_F_SNZGXGUL_V_GQHYEKHS',
    value=0,
    field=NV_A_HACMEFIF_F_SNZGXGUL
)

NV_A_HACMEFIF_F_WMYUJFVG = FieldMetadata(
    name='NV_A_HACMEFIF_F_WMYUJFVG',
    msb=27,
    lsb=11,
    register=NV_A_HACMEFIF
)

NV_A_HACMEFIF_F_WMYUJFVG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_HACMEFIF_F_WMYUJFVG_V_GQHYEKHS',
    value=0,
    field=NV_A_HACMEFIF_F_WMYUJFVG
)

NV_A_QZWGGPTT = ArrayMetadata(
    name='NV_A_QZWGGPTT',
    base_address=0x2290,
    stride=4,
    size=4,
    zero_based=True
)

NV_A_QZWGGPTT_F_RPLZNTNW = FieldMetadata(
    name='NV_A_QZWGGPTT_F_RPLZNTNW',
    msb=3,
    lsb=3,
    register=NV_A_QZWGGPTT
)

NV_A_QZWGGPTT_F_RPLZNTNW_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QZWGGPTT_F_RPLZNTNW_V_GQHYEKHS',
    value=0,
    field=NV_A_QZWGGPTT_F_RPLZNTNW
)

NV_A_QZWGGPTT_F_QDKQSLJS = FieldMetadata(
    name='NV_A_QZWGGPTT_F_QDKQSLJS',
    msb=9,
    lsb=9,
    register=NV_A_QZWGGPTT
)

NV_A_QZWGGPTT_F_QDKQSLJS_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QZWGGPTT_F_QDKQSLJS_V_GQHYEKHS',
    value=0,
    field=NV_A_QZWGGPTT_F_QDKQSLJS
)

NV_A_QZWGGPTT_F_MYDGBKIO = FieldMetadata(
    name='NV_A_QZWGGPTT_F_MYDGBKIO',
    msb=2,
    lsb=0,
    register=NV_A_QZWGGPTT
)

NV_A_QZWGGPTT_F_MYDGBKIO_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QZWGGPTT_F_MYDGBKIO_V_GQHYEKHS',
    value=0,
    field=NV_A_QZWGGPTT_F_MYDGBKIO
)

NV_A_QZWGGPTT_F_AATGDVNZ = FieldMetadata(
    name='NV_A_QZWGGPTT_F_AATGDVNZ',
    msb=7,
    lsb=6,
    register=NV_A_QZWGGPTT
)

NV_A_QZWGGPTT_F_AATGDVNZ_V_JKXSHNLV = ValueMetadata(
    name='NV_A_QZWGGPTT_F_AATGDVNZ_V_JKXSHNLV',
    value=1,
    field=NV_A_QZWGGPTT_F_AATGDVNZ
)
NV_A_QZWGGPTT_F_AATGDVNZ_V_SDCCMLBB = ValueMetadata(
    name='NV_A_QZWGGPTT_F_AATGDVNZ_V_SDCCMLBB',
    value=3,
    field=NV_A_QZWGGPTT_F_AATGDVNZ
)
NV_A_QZWGGPTT_F_AATGDVNZ_V_BLJQBDXK = ValueMetadata(
    name='NV_A_QZWGGPTT_F_AATGDVNZ_V_BLJQBDXK',
    value=2,
    field=NV_A_QZWGGPTT_F_AATGDVNZ
)
NV_A_QZWGGPTT_F_AATGDVNZ_V_DDVZAPYQ = ValueMetadata(
    name='NV_A_QZWGGPTT_F_AATGDVNZ_V_DDVZAPYQ',
    value=0,
    field=NV_A_QZWGGPTT_F_AATGDVNZ
)
NV_A_QZWGGPTT_F_AATGDVNZ_V_YKKIGEDW = ValueMetadata(
    name='NV_A_QZWGGPTT_F_AATGDVNZ_V_YKKIGEDW',
    value=0,
    field=NV_A_QZWGGPTT_F_AATGDVNZ
)

NV_A_QZWGGPTT_F_YLJFJGHP = FieldMetadata(
    name='NV_A_QZWGGPTT_F_YLJFJGHP',
    msb=31,
    lsb=31,
    register=NV_A_QZWGGPTT
)

NV_A_QZWGGPTT_F_YLJFJGHP_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QZWGGPTT_F_YLJFJGHP_V_GQHYEKHS',
    value=0,
    field=NV_A_QZWGGPTT_F_YLJFJGHP
)

NV_A_QZWGGPTT_F_JBSRGFLD = FieldMetadata(
    name='NV_A_QZWGGPTT_F_JBSRGFLD',
    msb=8,
    lsb=8,
    register=NV_A_QZWGGPTT
)

NV_A_QZWGGPTT_F_JBSRGFLD_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QZWGGPTT_F_JBSRGFLD_V_GQHYEKHS',
    value=0,
    field=NV_A_QZWGGPTT_F_JBSRGFLD
)

NV_A_CKSPGNMV = ArrayMetadata(
    name='NV_A_CKSPGNMV',
    base_address=0x270,
    stride=4,
    size=4,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_A_CKSPGNMV_F_RPLZNTNW = FieldMetadata(
    name='NV_A_CKSPGNMV_F_RPLZNTNW',
    msb=3,
    lsb=3,
    register=NV_A_CKSPGNMV
)

NV_A_CKSPGNMV_F_RPLZNTNW_V_GQHYEKHS = ValueMetadata(
    name='NV_A_CKSPGNMV_F_RPLZNTNW_V_GQHYEKHS',
    value=0,
    field=NV_A_CKSPGNMV_F_RPLZNTNW
)

NV_A_CKSPGNMV_F_QDKQSLJS = FieldMetadata(
    name='NV_A_CKSPGNMV_F_QDKQSLJS',
    msb=5,
    lsb=5,
    register=NV_A_CKSPGNMV
)

NV_A_CKSPGNMV_F_QDKQSLJS_V_GQHYEKHS = ValueMetadata(
    name='NV_A_CKSPGNMV_F_QDKQSLJS_V_GQHYEKHS',
    value=0,
    field=NV_A_CKSPGNMV_F_QDKQSLJS
)

NV_A_CKSPGNMV_F_MYDGBKIO = FieldMetadata(
    name='NV_A_CKSPGNMV_F_MYDGBKIO',
    msb=2,
    lsb=0,
    register=NV_A_CKSPGNMV
)

NV_A_CKSPGNMV_F_MYDGBKIO_V_GQHYEKHS = ValueMetadata(
    name='NV_A_CKSPGNMV_F_MYDGBKIO_V_GQHYEKHS',
    value=0,
    field=NV_A_CKSPGNMV_F_MYDGBKIO
)

NV_A_CKSPGNMV_F_YLJFJGHP = FieldMetadata(
    name='NV_A_CKSPGNMV_F_YLJFJGHP',
    msb=31,
    lsb=31,
    register=NV_A_CKSPGNMV
)

NV_A_CKSPGNMV_F_YLJFJGHP_V_GQHYEKHS = ValueMetadata(
    name='NV_A_CKSPGNMV_F_YLJFJGHP_V_GQHYEKHS',
    value=0,
    field=NV_A_CKSPGNMV_F_YLJFJGHP
)

NV_A_CKSPGNMV_F_JBSRGFLD = FieldMetadata(
    name='NV_A_CKSPGNMV_F_JBSRGFLD',
    msb=4,
    lsb=4,
    register=NV_A_CKSPGNMV
)

NV_A_CKSPGNMV_F_JBSRGFLD_V_GQHYEKHS = ValueMetadata(
    name='NV_A_CKSPGNMV_F_JBSRGFLD_V_GQHYEKHS',
    value=0,
    field=NV_A_CKSPGNMV_F_JBSRGFLD
)

NV_A_DATQLVAB = ArrayMetadata(
    name='NV_A_DATQLVAB',
    base_address=0x2270,
    stride=4,
    size=4,
    zero_based=True
)

NV_A_DATQLVAB_F_SNZGXGUL = FieldMetadata(
    name='NV_A_DATQLVAB_F_SNZGXGUL',
    msb=7,
    lsb=0,
    register=NV_A_DATQLVAB
)

NV_A_DATQLVAB_F_SNZGXGUL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_DATQLVAB_F_SNZGXGUL_V_GQHYEKHS',
    value=0,
    field=NV_A_DATQLVAB_F_SNZGXGUL
)

NV_A_DATQLVAB_F_YAHKBQSG = FieldMetadata(
    name='NV_A_DATQLVAB_F_YAHKBQSG',
    msb=31,
    lsb=31,
    register=NV_A_DATQLVAB
)

NV_A_DATQLVAB_F_YAHKBQSG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_DATQLVAB_F_YAHKBQSG_V_GQHYEKHS',
    value=0,
    field=NV_A_DATQLVAB_F_YAHKBQSG
)

NV_A_DATQLVAB_F_VADPLWWG = FieldMetadata(
    name='NV_A_DATQLVAB_F_VADPLWWG',
    msb=30,
    lsb=30,
    register=NV_A_DATQLVAB
)

NV_A_DATQLVAB_F_VADPLWWG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_DATQLVAB_F_VADPLWWG_V_GQHYEKHS',
    value=0,
    field=NV_A_DATQLVAB_F_VADPLWWG
)

NV_A_DATQLVAB_F_SLHBJMTE = FieldMetadata(
    name='NV_A_DATQLVAB_F_SLHBJMTE',
    msb=29,
    lsb=29,
    register=NV_A_DATQLVAB
)

NV_A_DATQLVAB_F_SLHBJMTE_V_GQHYEKHS = ValueMetadata(
    name='NV_A_DATQLVAB_F_SLHBJMTE_V_GQHYEKHS',
    value=0,
    field=NV_A_DATQLVAB_F_SLHBJMTE
)

NV_A_DATQLVAB_F_CLSFQYAN = FieldMetadata(
    name='NV_A_DATQLVAB_F_CLSFQYAN',
    msb=26,
    lsb=10,
    register=NV_A_DATQLVAB
)

NV_A_DATQLVAB_F_CLSFQYAN_V_GQHYEKHS = ValueMetadata(
    name='NV_A_DATQLVAB_F_CLSFQYAN_V_GQHYEKHS',
    value=0,
    field=NV_A_DATQLVAB_F_CLSFQYAN
)

NV_A_HCMCBNUE = ArrayMetadata(
    name='NV_A_HCMCBNUE',
    base_address=0x530,
    stride=4,
    size=4,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_A_HCMCBNUE_F_SNZGXGUL = FieldMetadata(
    name='NV_A_HCMCBNUE_F_SNZGXGUL',
    msb=7,
    lsb=0,
    register=NV_A_HCMCBNUE
)

NV_A_HCMCBNUE_F_SNZGXGUL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_HCMCBNUE_F_SNZGXGUL_V_GQHYEKHS',
    value=0,
    field=NV_A_HCMCBNUE_F_SNZGXGUL
)

NV_A_HCMCBNUE_F_YAHKBQSG = FieldMetadata(
    name='NV_A_HCMCBNUE_F_YAHKBQSG',
    msb=31,
    lsb=31,
    register=NV_A_HCMCBNUE
)

NV_A_HCMCBNUE_F_YAHKBQSG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_HCMCBNUE_F_YAHKBQSG_V_GQHYEKHS',
    value=0,
    field=NV_A_HCMCBNUE_F_YAHKBQSG
)

NV_A_HCMCBNUE_F_VADPLWWG = FieldMetadata(
    name='NV_A_HCMCBNUE_F_VADPLWWG',
    msb=30,
    lsb=30,
    register=NV_A_HCMCBNUE
)

NV_A_HCMCBNUE_F_VADPLWWG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_HCMCBNUE_F_VADPLWWG_V_GQHYEKHS',
    value=0,
    field=NV_A_HCMCBNUE_F_VADPLWWG
)

NV_A_HCMCBNUE_F_SLHBJMTE = FieldMetadata(
    name='NV_A_HCMCBNUE_F_SLHBJMTE',
    msb=29,
    lsb=29,
    register=NV_A_HCMCBNUE
)

NV_A_HCMCBNUE_F_SLHBJMTE_V_GQHYEKHS = ValueMetadata(
    name='NV_A_HCMCBNUE_F_SLHBJMTE_V_GQHYEKHS',
    value=0,
    field=NV_A_HCMCBNUE_F_SLHBJMTE
)

NV_A_HCMCBNUE_F_CLSFQYAN = FieldMetadata(
    name='NV_A_HCMCBNUE_F_CLSFQYAN',
    msb=26,
    lsb=10,
    register=NV_A_HCMCBNUE
)

NV_A_HCMCBNUE_F_CLSFQYAN_V_GQHYEKHS = ValueMetadata(
    name='NV_A_HCMCBNUE_F_CLSFQYAN_V_GQHYEKHS',
    value=0,
    field=NV_A_HCMCBNUE_F_CLSFQYAN
)

NV_A_XYVLFUGI = ArrayMetadata(
    name='NV_A_XYVLFUGI',
    base_address=0x540,
    stride=4,
    size=4,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_A_XYVLFUGI_F_ONWVXYRY = FieldMetadata(
    name='NV_A_XYVLFUGI_F_ONWVXYRY',
    msb=31,
    lsb=28,
    register=NV_A_XYVLFUGI
)

NV_A_XYVLFUGI_F_ONWVXYRY_V_GQHYEKHS = ValueMetadata(
    name='NV_A_XYVLFUGI_F_ONWVXYRY_V_GQHYEKHS',
    value=0,
    field=NV_A_XYVLFUGI_F_ONWVXYRY
)

NV_A_XYVLFUGI_F_SNZGXGUL = FieldMetadata(
    name='NV_A_XYVLFUGI_F_SNZGXGUL',
    msb=10,
    lsb=0,
    register=NV_A_XYVLFUGI
)

NV_A_XYVLFUGI_F_SNZGXGUL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_XYVLFUGI_F_SNZGXGUL_V_GQHYEKHS',
    value=0,
    field=NV_A_XYVLFUGI_F_SNZGXGUL
)

NV_A_XYVLFUGI_F_WMYUJFVG = FieldMetadata(
    name='NV_A_XYVLFUGI_F_WMYUJFVG',
    msb=27,
    lsb=11,
    register=NV_A_XYVLFUGI
)

NV_A_XYVLFUGI_F_WMYUJFVG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_XYVLFUGI_F_WMYUJFVG_V_GQHYEKHS',
    value=0,
    field=NV_A_XYVLFUGI_F_WMYUJFVG
)

