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
from gpu.regs.gb100.fbpa_int import (
    NV_A_HACMEFIF,
    NV_A_HACMEFIF_F_ONWVXYRY,
    NV_A_HACMEFIF_F_ONWVXYRY_V_GQHYEKHS,
    NV_A_HACMEFIF_F_SNZGXGUL,
    NV_A_HACMEFIF_F_SNZGXGUL_V_GQHYEKHS,
    NV_A_HACMEFIF_F_WMYUJFVG,
    NV_A_HACMEFIF_F_WMYUJFVG_V_GQHYEKHS,
    NV_A_XYVLFUGI,
    NV_A_XYVLFUGI_F_ONWVXYRY,
    NV_A_XYVLFUGI_F_ONWVXYRY_V_GQHYEKHS,
    NV_A_XYVLFUGI_F_SNZGXGUL,
    NV_A_XYVLFUGI_F_SNZGXGUL_V_GQHYEKHS,
    NV_A_XYVLFUGI_F_WMYUJFVG,
    NV_A_XYVLFUGI_F_WMYUJFVG_V_GQHYEKHS,
)

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
    NV_A_NYBWZNTJ,
    NV_A_NYBWZNTJ_F_HWFYBREO,
    NV_A_NYBWZNTJ_F_HWFYBREO_V_GQHYEKHS,
)

# Device definitions
NV_D_BXFCDUCX = DeviceMetadata(
    name='NV_D_BXFCDUCX',
    start_address=0x7000000,
    end_address=0x7003fff
)

# Array definitions
NV_A_RFGFXXJP = ArrayMetadata(
    name='NV_A_RFGFXXJP',
    base_address=0x25c0,
    stride=4,
    size=4,
    zero_based=True
)

NV_A_RFGFXXJP_F_SNZGXGUL = FieldMetadata(
    name='NV_A_RFGFXXJP_F_SNZGXGUL',
    msb=7,
    lsb=0,
    register=NV_A_RFGFXXJP
)

NV_A_RFGFXXJP_F_SNZGXGUL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_RFGFXXJP_F_SNZGXGUL_V_GQHYEKHS',
    value=0,
    field=NV_A_RFGFXXJP_F_SNZGXGUL
)

NV_A_RFGFXXJP_F_YAHKBQSG = FieldMetadata(
    name='NV_A_RFGFXXJP_F_YAHKBQSG',
    msb=31,
    lsb=31,
    register=NV_A_RFGFXXJP
)

NV_A_RFGFXXJP_F_YAHKBQSG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_RFGFXXJP_F_YAHKBQSG_V_GQHYEKHS',
    value=0,
    field=NV_A_RFGFXXJP_F_YAHKBQSG
)

NV_A_RFGFXXJP_F_VADPLWWG = FieldMetadata(
    name='NV_A_RFGFXXJP_F_VADPLWWG',
    msb=30,
    lsb=30,
    register=NV_A_RFGFXXJP
)

NV_A_RFGFXXJP_F_VADPLWWG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_RFGFXXJP_F_VADPLWWG_V_GQHYEKHS',
    value=0,
    field=NV_A_RFGFXXJP_F_VADPLWWG
)

NV_A_RFGFXXJP_F_SLHBJMTE = FieldMetadata(
    name='NV_A_RFGFXXJP_F_SLHBJMTE',
    msb=29,
    lsb=29,
    register=NV_A_RFGFXXJP
)

NV_A_RFGFXXJP_F_CLSFQYAN = FieldMetadata(
    name='NV_A_RFGFXXJP_F_CLSFQYAN',
    msb=26,
    lsb=10,
    register=NV_A_RFGFXXJP
)

NV_A_RFGFXXJP_F_TGABQZSP = FieldMetadata(
    name='NV_A_RFGFXXJP_F_TGABQZSP',
    msb=26,
    lsb=10,
    register=NV_A_RFGFXXJP
)

NV_A_RFGFXXJP_F_TGABQZSP_V_GQHYEKHS = ValueMetadata(
    name='NV_A_RFGFXXJP_F_TGABQZSP_V_GQHYEKHS',
    value=0,
    field=NV_A_RFGFXXJP_F_TGABQZSP
)

NV_A_RFGFXXJP_F_RWWIKXPU = FieldMetadata(
    name='NV_A_RFGFXXJP_F_RWWIKXPU',
    msb=29,
    lsb=29,
    register=NV_A_RFGFXXJP
)

NV_A_RFGFXXJP_F_RWWIKXPU_V_GQHYEKHS = ValueMetadata(
    name='NV_A_RFGFXXJP_F_RWWIKXPU_V_GQHYEKHS',
    value=0,
    field=NV_A_RFGFXXJP_F_RWWIKXPU
)

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

NV_A_KAKYCVUL_F_MFZSBPQY = FieldMetadata(
    name='NV_A_KAKYCVUL_F_MFZSBPQY',
    msb=12,
    lsb=9,
    register=NV_A_KAKYCVUL
)

NV_A_KAKYCVUL_F_MFZSBPQY_V_GQHYEKHS = ValueMetadata(
    name='NV_A_KAKYCVUL_F_MFZSBPQY_V_GQHYEKHS',
    value=0,
    field=NV_A_KAKYCVUL_F_MFZSBPQY
)

NV_A_KAKYCVUL_F_YMDJMHFS = FieldMetadata(
    name='NV_A_KAKYCVUL_F_YMDJMHFS',
    msb=13,
    lsb=13,
    register=NV_A_KAKYCVUL
)

NV_A_KAKYCVUL_F_YMDJMHFS_V_GQHYEKHS = ValueMetadata(
    name='NV_A_KAKYCVUL_F_YMDJMHFS_V_GQHYEKHS',
    value=0,
    field=NV_A_KAKYCVUL_F_YMDJMHFS
)

NV_A_KAKYCVUL_F_HEMOBSWQ = FieldMetadata(
    name='NV_A_KAKYCVUL_F_HEMOBSWQ',
    msb=16,
    lsb=14,
    register=NV_A_KAKYCVUL
)

NV_A_KAKYCVUL_F_HEMOBSWQ_V_GQHYEKHS = ValueMetadata(
    name='NV_A_KAKYCVUL_F_HEMOBSWQ_V_GQHYEKHS',
    value=0,
    field=NV_A_KAKYCVUL_F_HEMOBSWQ
)

NV_A_KAKYCVUL_F_LWXXOUIL = FieldMetadata(
    name='NV_A_KAKYCVUL_F_LWXXOUIL',
    msb=8,
    lsb=8,
    register=NV_A_KAKYCVUL
)

NV_A_KAKYCVUL_F_LWXXOUIL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_KAKYCVUL_F_LWXXOUIL_V_GQHYEKHS',
    value=0,
    field=NV_A_KAKYCVUL_F_LWXXOUIL
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

NV_A_OCRCKSGD_F_MFZSBPQY = FieldMetadata(
    name='NV_A_OCRCKSGD_F_MFZSBPQY',
    msb=12,
    lsb=9,
    register=NV_A_OCRCKSGD
)

NV_A_OCRCKSGD_F_MFZSBPQY_V_GQHYEKHS = ValueMetadata(
    name='NV_A_OCRCKSGD_F_MFZSBPQY_V_GQHYEKHS',
    value=0,
    field=NV_A_OCRCKSGD_F_MFZSBPQY
)

NV_A_OCRCKSGD_F_YMDJMHFS = FieldMetadata(
    name='NV_A_OCRCKSGD_F_YMDJMHFS',
    msb=13,
    lsb=13,
    register=NV_A_OCRCKSGD
)

NV_A_OCRCKSGD_F_YMDJMHFS_V_GQHYEKHS = ValueMetadata(
    name='NV_A_OCRCKSGD_F_YMDJMHFS_V_GQHYEKHS',
    value=0,
    field=NV_A_OCRCKSGD_F_YMDJMHFS
)

NV_A_OCRCKSGD_F_HEMOBSWQ = FieldMetadata(
    name='NV_A_OCRCKSGD_F_HEMOBSWQ',
    msb=16,
    lsb=14,
    register=NV_A_OCRCKSGD
)

NV_A_OCRCKSGD_F_HEMOBSWQ_V_GQHYEKHS = ValueMetadata(
    name='NV_A_OCRCKSGD_F_HEMOBSWQ_V_GQHYEKHS',
    value=0,
    field=NV_A_OCRCKSGD_F_HEMOBSWQ
)

NV_A_OCRCKSGD_F_LWXXOUIL = FieldMetadata(
    name='NV_A_OCRCKSGD_F_LWXXOUIL',
    msb=8,
    lsb=8,
    register=NV_A_OCRCKSGD
)

NV_A_OCRCKSGD_F_LWXXOUIL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_OCRCKSGD_F_LWXXOUIL_V_GQHYEKHS',
    value=0,
    field=NV_A_OCRCKSGD_F_LWXXOUIL
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

NV_A_ZRHMNQOZ_F_MFZSBPQY = FieldMetadata(
    name='NV_A_ZRHMNQOZ_F_MFZSBPQY',
    msb=12,
    lsb=9,
    register=NV_A_ZRHMNQOZ
)

NV_A_ZRHMNQOZ_F_MFZSBPQY_V_GQHYEKHS = ValueMetadata(
    name='NV_A_ZRHMNQOZ_F_MFZSBPQY_V_GQHYEKHS',
    value=0,
    field=NV_A_ZRHMNQOZ_F_MFZSBPQY
)

NV_A_ZRHMNQOZ_F_YMDJMHFS = FieldMetadata(
    name='NV_A_ZRHMNQOZ_F_YMDJMHFS',
    msb=13,
    lsb=13,
    register=NV_A_ZRHMNQOZ
)

NV_A_ZRHMNQOZ_F_YMDJMHFS_V_GQHYEKHS = ValueMetadata(
    name='NV_A_ZRHMNQOZ_F_YMDJMHFS_V_GQHYEKHS',
    value=0,
    field=NV_A_ZRHMNQOZ_F_YMDJMHFS
)

NV_A_ZRHMNQOZ_F_HEMOBSWQ = FieldMetadata(
    name='NV_A_ZRHMNQOZ_F_HEMOBSWQ',
    msb=16,
    lsb=14,
    register=NV_A_ZRHMNQOZ
)

NV_A_ZRHMNQOZ_F_HEMOBSWQ_V_GQHYEKHS = ValueMetadata(
    name='NV_A_ZRHMNQOZ_F_HEMOBSWQ_V_GQHYEKHS',
    value=0,
    field=NV_A_ZRHMNQOZ_F_HEMOBSWQ
)

NV_A_ZRHMNQOZ_F_LWXXOUIL = FieldMetadata(
    name='NV_A_ZRHMNQOZ_F_LWXXOUIL',
    msb=8,
    lsb=8,
    register=NV_A_ZRHMNQOZ
)

NV_A_ZRHMNQOZ_F_LWXXOUIL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_ZRHMNQOZ_F_LWXXOUIL_V_GQHYEKHS',
    value=0,
    field=NV_A_ZRHMNQOZ_F_LWXXOUIL
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

NV_A_OFFJTAZO = ArrayMetadata(
    name='NV_A_OFFJTAZO',
    base_address=0x1490,
    stride=4,
    size=4,
    zero_based=True
)

NV_A_OFFJTAZO_F_RPLZNTNW = FieldMetadata(
    name='NV_A_OFFJTAZO_F_RPLZNTNW',
    msb=3,
    lsb=3,
    register=NV_A_OFFJTAZO
)

NV_A_OFFJTAZO_F_RPLZNTNW_V_GQHYEKHS = ValueMetadata(
    name='NV_A_OFFJTAZO_F_RPLZNTNW_V_GQHYEKHS',
    value=0,
    field=NV_A_OFFJTAZO_F_RPLZNTNW
)

NV_A_OFFJTAZO_F_MFZSBPQY = FieldMetadata(
    name='NV_A_OFFJTAZO_F_MFZSBPQY',
    msb=12,
    lsb=9,
    register=NV_A_OFFJTAZO
)

NV_A_OFFJTAZO_F_MFZSBPQY_V_GQHYEKHS = ValueMetadata(
    name='NV_A_OFFJTAZO_F_MFZSBPQY_V_GQHYEKHS',
    value=0,
    field=NV_A_OFFJTAZO_F_MFZSBPQY
)

NV_A_OFFJTAZO_F_YMDJMHFS = FieldMetadata(
    name='NV_A_OFFJTAZO_F_YMDJMHFS',
    msb=13,
    lsb=13,
    register=NV_A_OFFJTAZO
)

NV_A_OFFJTAZO_F_YMDJMHFS_V_GQHYEKHS = ValueMetadata(
    name='NV_A_OFFJTAZO_F_YMDJMHFS_V_GQHYEKHS',
    value=0,
    field=NV_A_OFFJTAZO_F_YMDJMHFS
)

NV_A_OFFJTAZO_F_HEMOBSWQ = FieldMetadata(
    name='NV_A_OFFJTAZO_F_HEMOBSWQ',
    msb=16,
    lsb=14,
    register=NV_A_OFFJTAZO
)

NV_A_OFFJTAZO_F_HEMOBSWQ_V_GQHYEKHS = ValueMetadata(
    name='NV_A_OFFJTAZO_F_HEMOBSWQ_V_GQHYEKHS',
    value=0,
    field=NV_A_OFFJTAZO_F_HEMOBSWQ
)

NV_A_OFFJTAZO_F_LWXXOUIL = FieldMetadata(
    name='NV_A_OFFJTAZO_F_LWXXOUIL',
    msb=8,
    lsb=8,
    register=NV_A_OFFJTAZO
)

NV_A_OFFJTAZO_F_LWXXOUIL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_OFFJTAZO_F_LWXXOUIL_V_GQHYEKHS',
    value=0,
    field=NV_A_OFFJTAZO_F_LWXXOUIL
)

NV_A_OFFJTAZO_F_MYDGBKIO = FieldMetadata(
    name='NV_A_OFFJTAZO_F_MYDGBKIO',
    msb=2,
    lsb=0,
    register=NV_A_OFFJTAZO
)

NV_A_OFFJTAZO_F_MYDGBKIO_V_GQHYEKHS = ValueMetadata(
    name='NV_A_OFFJTAZO_F_MYDGBKIO_V_GQHYEKHS',
    value=0,
    field=NV_A_OFFJTAZO_F_MYDGBKIO
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

NV_A_GQNBAIWL_F_MFZSBPQY = FieldMetadata(
    name='NV_A_GQNBAIWL_F_MFZSBPQY',
    msb=12,
    lsb=9,
    register=NV_A_GQNBAIWL
)

NV_A_GQNBAIWL_F_MFZSBPQY_V_GQHYEKHS = ValueMetadata(
    name='NV_A_GQNBAIWL_F_MFZSBPQY_V_GQHYEKHS',
    value=0,
    field=NV_A_GQNBAIWL_F_MFZSBPQY
)

NV_A_GQNBAIWL_F_YMDJMHFS = FieldMetadata(
    name='NV_A_GQNBAIWL_F_YMDJMHFS',
    msb=13,
    lsb=13,
    register=NV_A_GQNBAIWL
)

NV_A_GQNBAIWL_F_YMDJMHFS_V_GQHYEKHS = ValueMetadata(
    name='NV_A_GQNBAIWL_F_YMDJMHFS_V_GQHYEKHS',
    value=0,
    field=NV_A_GQNBAIWL_F_YMDJMHFS
)

NV_A_GQNBAIWL_F_HEMOBSWQ = FieldMetadata(
    name='NV_A_GQNBAIWL_F_HEMOBSWQ',
    msb=16,
    lsb=14,
    register=NV_A_GQNBAIWL
)

NV_A_GQNBAIWL_F_HEMOBSWQ_V_GQHYEKHS = ValueMetadata(
    name='NV_A_GQNBAIWL_F_HEMOBSWQ_V_GQHYEKHS',
    value=0,
    field=NV_A_GQNBAIWL_F_HEMOBSWQ
)

NV_A_GQNBAIWL_F_LWXXOUIL = FieldMetadata(
    name='NV_A_GQNBAIWL_F_LWXXOUIL',
    msb=8,
    lsb=8,
    register=NV_A_GQNBAIWL
)

NV_A_GQNBAIWL_F_LWXXOUIL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_GQNBAIWL_F_LWXXOUIL_V_GQHYEKHS',
    value=0,
    field=NV_A_GQNBAIWL_F_LWXXOUIL
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

NV_A_RXIHRXWD = ArrayMetadata(
    name='NV_A_RXIHRXWD',
    base_address=0x25d0,
    stride=4,
    size=4,
    zero_based=True
)

NV_A_RXIHRXWD_F_SNZGXGUL = FieldMetadata(
    name='NV_A_RXIHRXWD_F_SNZGXGUL',
    msb=7,
    lsb=0,
    register=NV_A_RXIHRXWD
)

NV_A_RXIHRXWD_F_SNZGXGUL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_RXIHRXWD_F_SNZGXGUL_V_GQHYEKHS',
    value=0,
    field=NV_A_RXIHRXWD_F_SNZGXGUL
)

NV_A_RXIHRXWD_F_YAHKBQSG = FieldMetadata(
    name='NV_A_RXIHRXWD_F_YAHKBQSG',
    msb=31,
    lsb=31,
    register=NV_A_RXIHRXWD
)

NV_A_RXIHRXWD_F_YAHKBQSG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_RXIHRXWD_F_YAHKBQSG_V_GQHYEKHS',
    value=0,
    field=NV_A_RXIHRXWD_F_YAHKBQSG
)

NV_A_RXIHRXWD_F_VADPLWWG = FieldMetadata(
    name='NV_A_RXIHRXWD_F_VADPLWWG',
    msb=30,
    lsb=30,
    register=NV_A_RXIHRXWD
)

NV_A_RXIHRXWD_F_VADPLWWG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_RXIHRXWD_F_VADPLWWG_V_GQHYEKHS',
    value=0,
    field=NV_A_RXIHRXWD_F_VADPLWWG
)

NV_A_RXIHRXWD_F_SLHBJMTE = FieldMetadata(
    name='NV_A_RXIHRXWD_F_SLHBJMTE',
    msb=29,
    lsb=29,
    register=NV_A_RXIHRXWD
)

NV_A_RXIHRXWD_F_CLSFQYAN = FieldMetadata(
    name='NV_A_RXIHRXWD_F_CLSFQYAN',
    msb=26,
    lsb=10,
    register=NV_A_RXIHRXWD
)

NV_A_RXIHRXWD_F_TGABQZSP = FieldMetadata(
    name='NV_A_RXIHRXWD_F_TGABQZSP',
    msb=26,
    lsb=10,
    register=NV_A_RXIHRXWD
)

NV_A_RXIHRXWD_F_TGABQZSP_V_GQHYEKHS = ValueMetadata(
    name='NV_A_RXIHRXWD_F_TGABQZSP_V_GQHYEKHS',
    value=0,
    field=NV_A_RXIHRXWD_F_TGABQZSP
)

NV_A_RXIHRXWD_F_RWWIKXPU = FieldMetadata(
    name='NV_A_RXIHRXWD_F_RWWIKXPU',
    msb=29,
    lsb=29,
    register=NV_A_RXIHRXWD
)

NV_A_RXIHRXWD_F_RWWIKXPU_V_GQHYEKHS = ValueMetadata(
    name='NV_A_RXIHRXWD_F_RWWIKXPU_V_GQHYEKHS',
    value=0,
    field=NV_A_RXIHRXWD_F_RWWIKXPU
)

NV_A_SGDNNLKJ = ArrayMetadata(
    name='NV_A_SGDNNLKJ',
    base_address=0x25f0,
    stride=4,
    size=4,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_A_SGDNNLKJ_F_SNZGXGUL = FieldMetadata(
    name='NV_A_SGDNNLKJ_F_SNZGXGUL',
    msb=7,
    lsb=0,
    register=NV_A_SGDNNLKJ
)

NV_A_SGDNNLKJ_F_SNZGXGUL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_SGDNNLKJ_F_SNZGXGUL_V_GQHYEKHS',
    value=0,
    field=NV_A_SGDNNLKJ_F_SNZGXGUL
)

NV_A_SGDNNLKJ_F_YAHKBQSG = FieldMetadata(
    name='NV_A_SGDNNLKJ_F_YAHKBQSG',
    msb=31,
    lsb=31,
    register=NV_A_SGDNNLKJ
)

NV_A_SGDNNLKJ_F_YAHKBQSG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_SGDNNLKJ_F_YAHKBQSG_V_GQHYEKHS',
    value=0,
    field=NV_A_SGDNNLKJ_F_YAHKBQSG
)

NV_A_SGDNNLKJ_F_VADPLWWG = FieldMetadata(
    name='NV_A_SGDNNLKJ_F_VADPLWWG',
    msb=30,
    lsb=30,
    register=NV_A_SGDNNLKJ
)

NV_A_SGDNNLKJ_F_VADPLWWG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_SGDNNLKJ_F_VADPLWWG_V_GQHYEKHS',
    value=0,
    field=NV_A_SGDNNLKJ_F_VADPLWWG
)

NV_A_SGDNNLKJ_F_SLHBJMTE = FieldMetadata(
    name='NV_A_SGDNNLKJ_F_SLHBJMTE',
    msb=29,
    lsb=29,
    register=NV_A_SGDNNLKJ
)

NV_A_SGDNNLKJ_F_CLSFQYAN = FieldMetadata(
    name='NV_A_SGDNNLKJ_F_CLSFQYAN',
    msb=26,
    lsb=10,
    register=NV_A_SGDNNLKJ
)

NV_A_SGDNNLKJ_F_TGABQZSP = FieldMetadata(
    name='NV_A_SGDNNLKJ_F_TGABQZSP',
    msb=26,
    lsb=10,
    register=NV_A_SGDNNLKJ
)

NV_A_SGDNNLKJ_F_TGABQZSP_V_GQHYEKHS = ValueMetadata(
    name='NV_A_SGDNNLKJ_F_TGABQZSP_V_GQHYEKHS',
    value=0,
    field=NV_A_SGDNNLKJ_F_TGABQZSP
)

NV_A_SGDNNLKJ_F_RWWIKXPU = FieldMetadata(
    name='NV_A_SGDNNLKJ_F_RWWIKXPU',
    msb=29,
    lsb=29,
    register=NV_A_SGDNNLKJ
)

NV_A_SGDNNLKJ_F_RWWIKXPU_V_GQHYEKHS = ValueMetadata(
    name='NV_A_SGDNNLKJ_F_RWWIKXPU_V_GQHYEKHS',
    value=0,
    field=NV_A_SGDNNLKJ_F_RWWIKXPU
)

NV_A_QCZRBWVL = ArrayMetadata(
    name='NV_A_QCZRBWVL',
    base_address=0x1480,
    stride=4,
    size=4,
    zero_based=True
)

NV_A_QCZRBWVL_F_SNZGXGUL = FieldMetadata(
    name='NV_A_QCZRBWVL_F_SNZGXGUL',
    msb=7,
    lsb=0,
    register=NV_A_QCZRBWVL
)

NV_A_QCZRBWVL_F_SNZGXGUL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QCZRBWVL_F_SNZGXGUL_V_GQHYEKHS',
    value=0,
    field=NV_A_QCZRBWVL_F_SNZGXGUL
)

NV_A_QCZRBWVL_F_YAHKBQSG = FieldMetadata(
    name='NV_A_QCZRBWVL_F_YAHKBQSG',
    msb=31,
    lsb=31,
    register=NV_A_QCZRBWVL
)

NV_A_QCZRBWVL_F_YAHKBQSG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QCZRBWVL_F_YAHKBQSG_V_GQHYEKHS',
    value=0,
    field=NV_A_QCZRBWVL_F_YAHKBQSG
)

NV_A_QCZRBWVL_F_VADPLWWG = FieldMetadata(
    name='NV_A_QCZRBWVL_F_VADPLWWG',
    msb=30,
    lsb=30,
    register=NV_A_QCZRBWVL
)

NV_A_QCZRBWVL_F_VADPLWWG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QCZRBWVL_F_VADPLWWG_V_GQHYEKHS',
    value=0,
    field=NV_A_QCZRBWVL_F_VADPLWWG
)

NV_A_QCZRBWVL_F_SLHBJMTE = FieldMetadata(
    name='NV_A_QCZRBWVL_F_SLHBJMTE',
    msb=29,
    lsb=29,
    register=NV_A_QCZRBWVL
)

NV_A_QCZRBWVL_F_CLSFQYAN = FieldMetadata(
    name='NV_A_QCZRBWVL_F_CLSFQYAN',
    msb=26,
    lsb=10,
    register=NV_A_QCZRBWVL
)

NV_A_QCZRBWVL_F_TGABQZSP = FieldMetadata(
    name='NV_A_QCZRBWVL_F_TGABQZSP',
    msb=26,
    lsb=10,
    register=NV_A_QCZRBWVL
)

NV_A_QCZRBWVL_F_TGABQZSP_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QCZRBWVL_F_TGABQZSP_V_GQHYEKHS',
    value=0,
    field=NV_A_QCZRBWVL_F_TGABQZSP
)

NV_A_QCZRBWVL_F_RWWIKXPU = FieldMetadata(
    name='NV_A_QCZRBWVL_F_RWWIKXPU',
    msb=29,
    lsb=29,
    register=NV_A_QCZRBWVL
)

NV_A_QCZRBWVL_F_RWWIKXPU_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QCZRBWVL_F_RWWIKXPU_V_GQHYEKHS',
    value=0,
    field=NV_A_QCZRBWVL_F_RWWIKXPU
)

NV_A_MALLPWTC = ArrayMetadata(
    name='NV_A_MALLPWTC',
    base_address=0x25e0,
    stride=4,
    size=4,
    zero_based=True
)

NV_A_MALLPWTC_F_SNZGXGUL = FieldMetadata(
    name='NV_A_MALLPWTC_F_SNZGXGUL',
    msb=7,
    lsb=0,
    register=NV_A_MALLPWTC
)

NV_A_MALLPWTC_F_SNZGXGUL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_MALLPWTC_F_SNZGXGUL_V_GQHYEKHS',
    value=0,
    field=NV_A_MALLPWTC_F_SNZGXGUL
)

NV_A_MALLPWTC_F_YAHKBQSG = FieldMetadata(
    name='NV_A_MALLPWTC_F_YAHKBQSG',
    msb=31,
    lsb=31,
    register=NV_A_MALLPWTC
)

NV_A_MALLPWTC_F_YAHKBQSG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_MALLPWTC_F_YAHKBQSG_V_GQHYEKHS',
    value=0,
    field=NV_A_MALLPWTC_F_YAHKBQSG
)

NV_A_MALLPWTC_F_VADPLWWG = FieldMetadata(
    name='NV_A_MALLPWTC_F_VADPLWWG',
    msb=30,
    lsb=30,
    register=NV_A_MALLPWTC
)

NV_A_MALLPWTC_F_VADPLWWG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_MALLPWTC_F_VADPLWWG_V_GQHYEKHS',
    value=0,
    field=NV_A_MALLPWTC_F_VADPLWWG
)

NV_A_MALLPWTC_F_SLHBJMTE = FieldMetadata(
    name='NV_A_MALLPWTC_F_SLHBJMTE',
    msb=29,
    lsb=29,
    register=NV_A_MALLPWTC
)

NV_A_MALLPWTC_F_CLSFQYAN = FieldMetadata(
    name='NV_A_MALLPWTC_F_CLSFQYAN',
    msb=26,
    lsb=10,
    register=NV_A_MALLPWTC
)

NV_A_MALLPWTC_F_TGABQZSP = FieldMetadata(
    name='NV_A_MALLPWTC_F_TGABQZSP',
    msb=26,
    lsb=10,
    register=NV_A_MALLPWTC
)

NV_A_MALLPWTC_F_TGABQZSP_V_GQHYEKHS = ValueMetadata(
    name='NV_A_MALLPWTC_F_TGABQZSP_V_GQHYEKHS',
    value=0,
    field=NV_A_MALLPWTC_F_TGABQZSP
)

NV_A_MALLPWTC_F_RWWIKXPU = FieldMetadata(
    name='NV_A_MALLPWTC_F_RWWIKXPU',
    msb=29,
    lsb=29,
    register=NV_A_MALLPWTC
)

NV_A_MALLPWTC_F_RWWIKXPU_V_GQHYEKHS = ValueMetadata(
    name='NV_A_MALLPWTC_F_RWWIKXPU_V_GQHYEKHS',
    value=0,
    field=NV_A_MALLPWTC_F_RWWIKXPU
)

NV_A_SASCVVCP = ArrayMetadata(
    name='NV_A_SASCVVCP',
    base_address=0x1470,
    stride=4,
    size=4,
    zero_based=True
)

NV_A_SASCVVCP_F_ONWVXYRY = FieldMetadata(
    name='NV_A_SASCVVCP_F_ONWVXYRY',
    msb=31,
    lsb=28,
    register=NV_A_SASCVVCP
)

NV_A_SASCVVCP_F_ONWVXYRY_V_GQHYEKHS = ValueMetadata(
    name='NV_A_SASCVVCP_F_ONWVXYRY_V_GQHYEKHS',
    value=0,
    field=NV_A_SASCVVCP_F_ONWVXYRY
)

NV_A_SASCVVCP_F_SNZGXGUL = FieldMetadata(
    name='NV_A_SASCVVCP_F_SNZGXGUL',
    msb=10,
    lsb=0,
    register=NV_A_SASCVVCP
)

NV_A_SASCVVCP_F_SNZGXGUL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_SASCVVCP_F_SNZGXGUL_V_GQHYEKHS',
    value=0,
    field=NV_A_SASCVVCP_F_SNZGXGUL
)

NV_A_SASCVVCP_F_WMYUJFVG = FieldMetadata(
    name='NV_A_SASCVVCP_F_WMYUJFVG',
    msb=27,
    lsb=11,
    register=NV_A_SASCVVCP
)

NV_A_SASCVVCP_F_WMYUJFVG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_SASCVVCP_F_WMYUJFVG_V_GQHYEKHS',
    value=0,
    field=NV_A_SASCVVCP_F_WMYUJFVG
)

NV_A_HCOTFNDZ = ArrayMetadata(
    name='NV_A_HCOTFNDZ',
    base_address=0x490,
    stride=4,
    size=4,
    zero_based=True
)

NV_A_HCOTFNDZ_F_NUEJUWHY = FieldMetadata(
    name='NV_A_HCOTFNDZ_F_NUEJUWHY',
    msb=2,
    lsb=2,
    register=NV_A_HCOTFNDZ
)

NV_A_HCOTFNDZ_F_NUEJUWHY_V_GQHYEKHS = ValueMetadata(
    name='NV_A_HCOTFNDZ_F_NUEJUWHY_V_GQHYEKHS',
    value=0,
    field=NV_A_HCOTFNDZ_F_NUEJUWHY
)

NV_A_HCOTFNDZ_F_SBDIUQGA = FieldMetadata(
    name='NV_A_HCOTFNDZ_F_SBDIUQGA',
    msb=5,
    lsb=5,
    register=NV_A_HCOTFNDZ
)

NV_A_HCOTFNDZ_F_SBDIUQGA_V_GQHYEKHS = ValueMetadata(
    name='NV_A_HCOTFNDZ_F_SBDIUQGA_V_GQHYEKHS',
    value=0,
    field=NV_A_HCOTFNDZ_F_SBDIUQGA
)

NV_A_HCOTFNDZ_F_GUBRWLLH = FieldMetadata(
    name='NV_A_HCOTFNDZ_F_GUBRWLLH',
    msb=4,
    lsb=4,
    register=NV_A_HCOTFNDZ
)

NV_A_HCOTFNDZ_F_GUBRWLLH_V_GQHYEKHS = ValueMetadata(
    name='NV_A_HCOTFNDZ_F_GUBRWLLH_V_GQHYEKHS',
    value=0,
    field=NV_A_HCOTFNDZ_F_GUBRWLLH
)

NV_A_HCOTFNDZ_F_CXBZSWTF = FieldMetadata(
    name='NV_A_HCOTFNDZ_F_CXBZSWTF',
    msb=6,
    lsb=6,
    register=NV_A_HCOTFNDZ
)

NV_A_HCOTFNDZ_F_CXBZSWTF_V_GQHYEKHS = ValueMetadata(
    name='NV_A_HCOTFNDZ_F_CXBZSWTF_V_GQHYEKHS',
    value=0,
    field=NV_A_HCOTFNDZ_F_CXBZSWTF
)

NV_A_HCOTFNDZ_F_VUHHTIYD = FieldMetadata(
    name='NV_A_HCOTFNDZ_F_VUHHTIYD',
    msb=3,
    lsb=3,
    register=NV_A_HCOTFNDZ
)

NV_A_HCOTFNDZ_F_VUHHTIYD_V_GQHYEKHS = ValueMetadata(
    name='NV_A_HCOTFNDZ_F_VUHHTIYD_V_GQHYEKHS',
    value=0,
    field=NV_A_HCOTFNDZ_F_VUHHTIYD
)

NV_A_HCOTFNDZ_F_AKNLWQOP = FieldMetadata(
    name='NV_A_HCOTFNDZ_F_AKNLWQOP',
    msb=16,
    lsb=8,
    register=NV_A_HCOTFNDZ
)

NV_A_HCOTFNDZ_F_AKNLWQOP_V_GQHYEKHS = ValueMetadata(
    name='NV_A_HCOTFNDZ_F_AKNLWQOP_V_GQHYEKHS',
    value=0,
    field=NV_A_HCOTFNDZ_F_AKNLWQOP
)

NV_A_HCOTFNDZ_F_JGDYKRTC = FieldMetadata(
    name='NV_A_HCOTFNDZ_F_JGDYKRTC',
    msb=1,
    lsb=0,
    register=NV_A_HCOTFNDZ
)

NV_A_HCOTFNDZ_F_JGDYKRTC_V_NMZBWDAB = ValueMetadata(
    name='NV_A_HCOTFNDZ_F_JGDYKRTC_V_NMZBWDAB',
    value=1,
    field=NV_A_HCOTFNDZ_F_JGDYKRTC
)
NV_A_HCOTFNDZ_F_JGDYKRTC_V_RHDXTSMA = ValueMetadata(
    name='NV_A_HCOTFNDZ_F_JGDYKRTC_V_RHDXTSMA',
    value=2,
    field=NV_A_HCOTFNDZ_F_JGDYKRTC
)
NV_A_HCOTFNDZ_F_JGDYKRTC_V_NSDPVTAV = ValueMetadata(
    name='NV_A_HCOTFNDZ_F_JGDYKRTC_V_NSDPVTAV',
    value=0,
    field=NV_A_HCOTFNDZ_F_JGDYKRTC
)

NV_A_DDAHJXIM = ArrayMetadata(
    name='NV_A_DDAHJXIM',
    base_address=0x478,
    stride=4,
    size=4,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_A_DDAHJXIM_F_AJYGETZL = FieldMetadata(
    name='NV_A_DDAHJXIM_F_AJYGETZL',
    msb=18,
    lsb=18,
    register=NV_A_DDAHJXIM
)

NV_A_DDAHJXIM_F_AJYGETZL_V_DOGBFDTH = ValueMetadata(
    name='NV_A_DDAHJXIM_F_AJYGETZL_V_DOGBFDTH',
    value=0,
    field=NV_A_DDAHJXIM_F_AJYGETZL
)
NV_A_DDAHJXIM_F_AJYGETZL_V_PIUJQBKV = ValueMetadata(
    name='NV_A_DDAHJXIM_F_AJYGETZL_V_PIUJQBKV',
    value=1,
    field=NV_A_DDAHJXIM_F_AJYGETZL
)

NV_A_DDAHJXIM_F_JFFCFQPN = FieldMetadata(
    name='NV_A_DDAHJXIM_F_JFFCFQPN',
    msb=2,
    lsb=2,
    register=NV_A_DDAHJXIM
)

NV_A_DDAHJXIM_F_JFFCFQPN_V_DOGBFDTH = ValueMetadata(
    name='NV_A_DDAHJXIM_F_JFFCFQPN_V_DOGBFDTH',
    value=0,
    field=NV_A_DDAHJXIM_F_JFFCFQPN
)
NV_A_DDAHJXIM_F_JFFCFQPN_V_PIUJQBKV = ValueMetadata(
    name='NV_A_DDAHJXIM_F_JFFCFQPN_V_PIUJQBKV',
    value=1,
    field=NV_A_DDAHJXIM_F_JFFCFQPN
)

NV_A_DDAHJXIM_F_EMORFZBS = FieldMetadata(
    name='NV_A_DDAHJXIM_F_EMORFZBS',
    msb=3,
    lsb=3,
    register=NV_A_DDAHJXIM
)

NV_A_DDAHJXIM_F_EMORFZBS_V_DOGBFDTH = ValueMetadata(
    name='NV_A_DDAHJXIM_F_EMORFZBS_V_DOGBFDTH',
    value=0,
    field=NV_A_DDAHJXIM_F_EMORFZBS
)
NV_A_DDAHJXIM_F_EMORFZBS_V_PIUJQBKV = ValueMetadata(
    name='NV_A_DDAHJXIM_F_EMORFZBS_V_PIUJQBKV',
    value=1,
    field=NV_A_DDAHJXIM_F_EMORFZBS
)

NV_A_DDAHJXIM_F_VYWFLWGM = FieldMetadata(
    name='NV_A_DDAHJXIM_F_VYWFLWGM',
    msb=22,
    lsb=22,
    register=NV_A_DDAHJXIM
)

NV_A_DDAHJXIM_F_VYWFLWGM_V_DOGBFDTH = ValueMetadata(
    name='NV_A_DDAHJXIM_F_VYWFLWGM_V_DOGBFDTH',
    value=0,
    field=NV_A_DDAHJXIM_F_VYWFLWGM
)
NV_A_DDAHJXIM_F_VYWFLWGM_V_PIUJQBKV = ValueMetadata(
    name='NV_A_DDAHJXIM_F_VYWFLWGM_V_PIUJQBKV',
    value=1,
    field=NV_A_DDAHJXIM_F_VYWFLWGM
)

NV_A_DDAHJXIM_F_EUADEKBG = FieldMetadata(
    name='NV_A_DDAHJXIM_F_EUADEKBG',
    msb=24,
    lsb=24,
    register=NV_A_DDAHJXIM
)

NV_A_DDAHJXIM_F_EUADEKBG_V_DOGBFDTH = ValueMetadata(
    name='NV_A_DDAHJXIM_F_EUADEKBG_V_DOGBFDTH',
    value=0,
    field=NV_A_DDAHJXIM_F_EUADEKBG
)
NV_A_DDAHJXIM_F_EUADEKBG_V_PIUJQBKV = ValueMetadata(
    name='NV_A_DDAHJXIM_F_EUADEKBG_V_PIUJQBKV',
    value=1,
    field=NV_A_DDAHJXIM_F_EUADEKBG
)

NV_A_DDAHJXIM_F_LQISPZHE = FieldMetadata(
    name='NV_A_DDAHJXIM_F_LQISPZHE',
    msb=23,
    lsb=23,
    register=NV_A_DDAHJXIM
)

NV_A_DDAHJXIM_F_LQISPZHE_V_DOGBFDTH = ValueMetadata(
    name='NV_A_DDAHJXIM_F_LQISPZHE_V_DOGBFDTH',
    value=0,
    field=NV_A_DDAHJXIM_F_LQISPZHE
)
NV_A_DDAHJXIM_F_LQISPZHE_V_PIUJQBKV = ValueMetadata(
    name='NV_A_DDAHJXIM_F_LQISPZHE_V_PIUJQBKV',
    value=1,
    field=NV_A_DDAHJXIM_F_LQISPZHE
)

NV_A_DDAHJXIM_F_RYPFIBFW = FieldMetadata(
    name='NV_A_DDAHJXIM_F_RYPFIBFW',
    msb=17,
    lsb=17,
    register=NV_A_DDAHJXIM
)

NV_A_DDAHJXIM_F_RYPFIBFW_V_DOGBFDTH = ValueMetadata(
    name='NV_A_DDAHJXIM_F_RYPFIBFW_V_DOGBFDTH',
    value=0,
    field=NV_A_DDAHJXIM_F_RYPFIBFW
)
NV_A_DDAHJXIM_F_RYPFIBFW_V_PIUJQBKV = ValueMetadata(
    name='NV_A_DDAHJXIM_F_RYPFIBFW_V_PIUJQBKV',
    value=1,
    field=NV_A_DDAHJXIM_F_RYPFIBFW
)

NV_A_DDAHJXIM_F_WKXUQINF = FieldMetadata(
    name='NV_A_DDAHJXIM_F_WKXUQINF',
    msb=1,
    lsb=1,
    register=NV_A_DDAHJXIM
)

NV_A_DDAHJXIM_F_WKXUQINF_V_DOGBFDTH = ValueMetadata(
    name='NV_A_DDAHJXIM_F_WKXUQINF_V_DOGBFDTH',
    value=0,
    field=NV_A_DDAHJXIM_F_WKXUQINF
)
NV_A_DDAHJXIM_F_WKXUQINF_V_PIUJQBKV = ValueMetadata(
    name='NV_A_DDAHJXIM_F_WKXUQINF_V_PIUJQBKV',
    value=1,
    field=NV_A_DDAHJXIM_F_WKXUQINF
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

NV_A_QZWGGPTT_F_MFZSBPQY = FieldMetadata(
    name='NV_A_QZWGGPTT_F_MFZSBPQY',
    msb=12,
    lsb=9,
    register=NV_A_QZWGGPTT
)

NV_A_QZWGGPTT_F_MFZSBPQY_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QZWGGPTT_F_MFZSBPQY_V_GQHYEKHS',
    value=0,
    field=NV_A_QZWGGPTT_F_MFZSBPQY
)

NV_A_QZWGGPTT_F_YMDJMHFS = FieldMetadata(
    name='NV_A_QZWGGPTT_F_YMDJMHFS',
    msb=13,
    lsb=13,
    register=NV_A_QZWGGPTT
)

NV_A_QZWGGPTT_F_YMDJMHFS_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QZWGGPTT_F_YMDJMHFS_V_GQHYEKHS',
    value=0,
    field=NV_A_QZWGGPTT_F_YMDJMHFS
)

NV_A_QZWGGPTT_F_HEMOBSWQ = FieldMetadata(
    name='NV_A_QZWGGPTT_F_HEMOBSWQ',
    msb=16,
    lsb=14,
    register=NV_A_QZWGGPTT
)

NV_A_QZWGGPTT_F_HEMOBSWQ_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QZWGGPTT_F_HEMOBSWQ_V_GQHYEKHS',
    value=0,
    field=NV_A_QZWGGPTT_F_HEMOBSWQ
)

NV_A_QZWGGPTT_F_LWXXOUIL = FieldMetadata(
    name='NV_A_QZWGGPTT_F_LWXXOUIL',
    msb=8,
    lsb=8,
    register=NV_A_QZWGGPTT
)

NV_A_QZWGGPTT_F_LWXXOUIL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QZWGGPTT_F_LWXXOUIL_V_GQHYEKHS',
    value=0,
    field=NV_A_QZWGGPTT_F_LWXXOUIL
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

NV_A_CKSPGNMV_F_MFZSBPQY = FieldMetadata(
    name='NV_A_CKSPGNMV_F_MFZSBPQY',
    msb=12,
    lsb=9,
    register=NV_A_CKSPGNMV
)

NV_A_CKSPGNMV_F_MFZSBPQY_V_GQHYEKHS = ValueMetadata(
    name='NV_A_CKSPGNMV_F_MFZSBPQY_V_GQHYEKHS',
    value=0,
    field=NV_A_CKSPGNMV_F_MFZSBPQY
)

NV_A_CKSPGNMV_F_YMDJMHFS = FieldMetadata(
    name='NV_A_CKSPGNMV_F_YMDJMHFS',
    msb=13,
    lsb=13,
    register=NV_A_CKSPGNMV
)

NV_A_CKSPGNMV_F_YMDJMHFS_V_GQHYEKHS = ValueMetadata(
    name='NV_A_CKSPGNMV_F_YMDJMHFS_V_GQHYEKHS',
    value=0,
    field=NV_A_CKSPGNMV_F_YMDJMHFS
)

NV_A_CKSPGNMV_F_HEMOBSWQ = FieldMetadata(
    name='NV_A_CKSPGNMV_F_HEMOBSWQ',
    msb=16,
    lsb=14,
    register=NV_A_CKSPGNMV
)

NV_A_CKSPGNMV_F_HEMOBSWQ_V_GQHYEKHS = ValueMetadata(
    name='NV_A_CKSPGNMV_F_HEMOBSWQ_V_GQHYEKHS',
    value=0,
    field=NV_A_CKSPGNMV_F_HEMOBSWQ
)

NV_A_CKSPGNMV_F_LWXXOUIL = FieldMetadata(
    name='NV_A_CKSPGNMV_F_LWXXOUIL',
    msb=8,
    lsb=8,
    register=NV_A_CKSPGNMV
)

NV_A_CKSPGNMV_F_LWXXOUIL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_CKSPGNMV_F_LWXXOUIL_V_GQHYEKHS',
    value=0,
    field=NV_A_CKSPGNMV_F_LWXXOUIL
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

NV_A_EUQIINVE = ArrayMetadata(
    name='NV_A_EUQIINVE',
    base_address=0x14c0,
    stride=4,
    size=4,
    zero_based=True
)

NV_A_EUQIINVE_F_RPLZNTNW = FieldMetadata(
    name='NV_A_EUQIINVE_F_RPLZNTNW',
    msb=3,
    lsb=3,
    register=NV_A_EUQIINVE
)

NV_A_EUQIINVE_F_RPLZNTNW_V_GQHYEKHS = ValueMetadata(
    name='NV_A_EUQIINVE_F_RPLZNTNW_V_GQHYEKHS',
    value=0,
    field=NV_A_EUQIINVE_F_RPLZNTNW
)

NV_A_EUQIINVE_F_MFZSBPQY = FieldMetadata(
    name='NV_A_EUQIINVE_F_MFZSBPQY',
    msb=12,
    lsb=9,
    register=NV_A_EUQIINVE
)

NV_A_EUQIINVE_F_MFZSBPQY_V_GQHYEKHS = ValueMetadata(
    name='NV_A_EUQIINVE_F_MFZSBPQY_V_GQHYEKHS',
    value=0,
    field=NV_A_EUQIINVE_F_MFZSBPQY
)

NV_A_EUQIINVE_F_YMDJMHFS = FieldMetadata(
    name='NV_A_EUQIINVE_F_YMDJMHFS',
    msb=13,
    lsb=13,
    register=NV_A_EUQIINVE
)

NV_A_EUQIINVE_F_YMDJMHFS_V_GQHYEKHS = ValueMetadata(
    name='NV_A_EUQIINVE_F_YMDJMHFS_V_GQHYEKHS',
    value=0,
    field=NV_A_EUQIINVE_F_YMDJMHFS
)

NV_A_EUQIINVE_F_HEMOBSWQ = FieldMetadata(
    name='NV_A_EUQIINVE_F_HEMOBSWQ',
    msb=16,
    lsb=14,
    register=NV_A_EUQIINVE
)

NV_A_EUQIINVE_F_HEMOBSWQ_V_GQHYEKHS = ValueMetadata(
    name='NV_A_EUQIINVE_F_HEMOBSWQ_V_GQHYEKHS',
    value=0,
    field=NV_A_EUQIINVE_F_HEMOBSWQ
)

NV_A_EUQIINVE_F_LWXXOUIL = FieldMetadata(
    name='NV_A_EUQIINVE_F_LWXXOUIL',
    msb=8,
    lsb=8,
    register=NV_A_EUQIINVE
)

NV_A_EUQIINVE_F_LWXXOUIL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_EUQIINVE_F_LWXXOUIL_V_GQHYEKHS',
    value=0,
    field=NV_A_EUQIINVE_F_LWXXOUIL
)

NV_A_EUQIINVE_F_MYDGBKIO = FieldMetadata(
    name='NV_A_EUQIINVE_F_MYDGBKIO',
    msb=2,
    lsb=0,
    register=NV_A_EUQIINVE
)

NV_A_EUQIINVE_F_MYDGBKIO_V_GQHYEKHS = ValueMetadata(
    name='NV_A_EUQIINVE_F_MYDGBKIO_V_GQHYEKHS',
    value=0,
    field=NV_A_EUQIINVE_F_MYDGBKIO
)

NV_A_EUQIINVE_F_YLJFJGHP = FieldMetadata(
    name='NV_A_EUQIINVE_F_YLJFJGHP',
    msb=31,
    lsb=31,
    register=NV_A_EUQIINVE
)

NV_A_EUQIINVE_F_YLJFJGHP_V_GQHYEKHS = ValueMetadata(
    name='NV_A_EUQIINVE_F_YLJFJGHP_V_GQHYEKHS',
    value=0,
    field=NV_A_EUQIINVE_F_YLJFJGHP
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

NV_A_DATQLVAB_F_CLSFQYAN = FieldMetadata(
    name='NV_A_DATQLVAB_F_CLSFQYAN',
    msb=26,
    lsb=10,
    register=NV_A_DATQLVAB
)

NV_A_DATQLVAB_F_TGABQZSP = FieldMetadata(
    name='NV_A_DATQLVAB_F_TGABQZSP',
    msb=26,
    lsb=10,
    register=NV_A_DATQLVAB
)

NV_A_DATQLVAB_F_TGABQZSP_V_GQHYEKHS = ValueMetadata(
    name='NV_A_DATQLVAB_F_TGABQZSP_V_GQHYEKHS',
    value=0,
    field=NV_A_DATQLVAB_F_TGABQZSP
)

NV_A_DATQLVAB_F_RWWIKXPU = FieldMetadata(
    name='NV_A_DATQLVAB_F_RWWIKXPU',
    msb=29,
    lsb=29,
    register=NV_A_DATQLVAB
)

NV_A_DATQLVAB_F_RWWIKXPU_V_GQHYEKHS = ValueMetadata(
    name='NV_A_DATQLVAB_F_RWWIKXPU_V_GQHYEKHS',
    value=0,
    field=NV_A_DATQLVAB_F_RWWIKXPU
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

NV_A_HCMCBNUE_F_CLSFQYAN = FieldMetadata(
    name='NV_A_HCMCBNUE_F_CLSFQYAN',
    msb=26,
    lsb=10,
    register=NV_A_HCMCBNUE
)

NV_A_HCMCBNUE_F_TGABQZSP = FieldMetadata(
    name='NV_A_HCMCBNUE_F_TGABQZSP',
    msb=26,
    lsb=10,
    register=NV_A_HCMCBNUE
)

NV_A_HCMCBNUE_F_TGABQZSP_V_GQHYEKHS = ValueMetadata(
    name='NV_A_HCMCBNUE_F_TGABQZSP_V_GQHYEKHS',
    value=0,
    field=NV_A_HCMCBNUE_F_TGABQZSP
)

NV_A_HCMCBNUE_F_RWWIKXPU = FieldMetadata(
    name='NV_A_HCMCBNUE_F_RWWIKXPU',
    msb=29,
    lsb=29,
    register=NV_A_HCMCBNUE
)

NV_A_HCMCBNUE_F_RWWIKXPU_V_GQHYEKHS = ValueMetadata(
    name='NV_A_HCMCBNUE_F_RWWIKXPU_V_GQHYEKHS',
    value=0,
    field=NV_A_HCMCBNUE_F_RWWIKXPU
)

NV_A_RVBOBCFY = ArrayMetadata(
    name='NV_A_RVBOBCFY',
    base_address=0x14b0,
    stride=4,
    size=4,
    zero_based=True
)

NV_A_RVBOBCFY_F_SNZGXGUL = FieldMetadata(
    name='NV_A_RVBOBCFY_F_SNZGXGUL',
    msb=7,
    lsb=0,
    register=NV_A_RVBOBCFY
)

NV_A_RVBOBCFY_F_SNZGXGUL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_RVBOBCFY_F_SNZGXGUL_V_GQHYEKHS',
    value=0,
    field=NV_A_RVBOBCFY_F_SNZGXGUL
)

NV_A_RVBOBCFY_F_YAHKBQSG = FieldMetadata(
    name='NV_A_RVBOBCFY_F_YAHKBQSG',
    msb=31,
    lsb=31,
    register=NV_A_RVBOBCFY
)

NV_A_RVBOBCFY_F_YAHKBQSG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_RVBOBCFY_F_YAHKBQSG_V_GQHYEKHS',
    value=0,
    field=NV_A_RVBOBCFY_F_YAHKBQSG
)

NV_A_RVBOBCFY_F_VADPLWWG = FieldMetadata(
    name='NV_A_RVBOBCFY_F_VADPLWWG',
    msb=30,
    lsb=30,
    register=NV_A_RVBOBCFY
)

NV_A_RVBOBCFY_F_VADPLWWG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_RVBOBCFY_F_VADPLWWG_V_GQHYEKHS',
    value=0,
    field=NV_A_RVBOBCFY_F_VADPLWWG
)

NV_A_RVBOBCFY_F_SLHBJMTE = FieldMetadata(
    name='NV_A_RVBOBCFY_F_SLHBJMTE',
    msb=29,
    lsb=29,
    register=NV_A_RVBOBCFY
)

NV_A_RVBOBCFY_F_CLSFQYAN = FieldMetadata(
    name='NV_A_RVBOBCFY_F_CLSFQYAN',
    msb=26,
    lsb=10,
    register=NV_A_RVBOBCFY
)

NV_A_RVBOBCFY_F_TGABQZSP = FieldMetadata(
    name='NV_A_RVBOBCFY_F_TGABQZSP',
    msb=26,
    lsb=10,
    register=NV_A_RVBOBCFY
)

NV_A_RVBOBCFY_F_TGABQZSP_V_GQHYEKHS = ValueMetadata(
    name='NV_A_RVBOBCFY_F_TGABQZSP_V_GQHYEKHS',
    value=0,
    field=NV_A_RVBOBCFY_F_TGABQZSP
)

NV_A_RVBOBCFY_F_RWWIKXPU = FieldMetadata(
    name='NV_A_RVBOBCFY_F_RWWIKXPU',
    msb=29,
    lsb=29,
    register=NV_A_RVBOBCFY
)

NV_A_RVBOBCFY_F_RWWIKXPU_V_GQHYEKHS = ValueMetadata(
    name='NV_A_RVBOBCFY_F_RWWIKXPU_V_GQHYEKHS',
    value=0,
    field=NV_A_RVBOBCFY_F_RWWIKXPU
)

NV_A_QKGXBXMW = ArrayMetadata(
    name='NV_A_QKGXBXMW',
    base_address=0x14a0,
    stride=4,
    size=4,
    zero_based=True
)

NV_A_QKGXBXMW_F_ONWVXYRY = FieldMetadata(
    name='NV_A_QKGXBXMW_F_ONWVXYRY',
    msb=31,
    lsb=28,
    register=NV_A_QKGXBXMW
)

NV_A_QKGXBXMW_F_ONWVXYRY_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QKGXBXMW_F_ONWVXYRY_V_GQHYEKHS',
    value=0,
    field=NV_A_QKGXBXMW_F_ONWVXYRY
)

NV_A_QKGXBXMW_F_SNZGXGUL = FieldMetadata(
    name='NV_A_QKGXBXMW_F_SNZGXGUL',
    msb=10,
    lsb=0,
    register=NV_A_QKGXBXMW
)

NV_A_QKGXBXMW_F_SNZGXGUL_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QKGXBXMW_F_SNZGXGUL_V_GQHYEKHS',
    value=0,
    field=NV_A_QKGXBXMW_F_SNZGXGUL
)

NV_A_QKGXBXMW_F_WMYUJFVG = FieldMetadata(
    name='NV_A_QKGXBXMW_F_WMYUJFVG',
    msb=27,
    lsb=11,
    register=NV_A_QKGXBXMW
)

NV_A_QKGXBXMW_F_WMYUJFVG_V_GQHYEKHS = ValueMetadata(
    name='NV_A_QKGXBXMW_F_WMYUJFVG_V_GQHYEKHS',
    value=0,
    field=NV_A_QKGXBXMW_F_WMYUJFVG
)

