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
from gpu.regs.gh100.gsp_int import (
    NV_R_YXCJNNGA,
    NV_R_YXCJNNGA_F_PXCBEXUQ,
    NV_R_YXCJNNGA_F_PXCBEXUQ_V_ACRDDIMM,
    NV_R_YXCJNNGA_F_PXCBEXUQ_FALSE,
    NV_R_YXCJNNGA_F_PXCBEXUQ_TRUE,
    NV_R_YXCJNNGA_F_IXCWUGWK,
    NV_R_YXCJNNGA_F_IXCWUGWK_V_ACRDDIMM,
    NV_R_YXCJNNGA_F_IXCWUGWK_FALSE,
    NV_R_YXCJNNGA_F_IXCWUGWK_TRUE,
    NV_R_YXCJNNGA_F_VTCPIJDF,
    NV_R_YXCJNNGA_F_VTCPIJDF_V_ACRDDIMM,
    NV_R_YXCJNNGA_F_VTCPIJDF_FALSE,
    NV_R_YXCJNNGA_F_VTCPIJDF_TRUE,
    NV_R_YXCJNNGA_F_JECCSHZH,
    NV_R_YXCJNNGA_F_JECCSHZH_V_ACRDDIMM,
    NV_R_YXCJNNGA_F_JECCSHZH_FALSE,
    NV_R_YXCJNNGA_F_JECCSHZH_TRUE,
    NV_R_YXCJNNGA_F_SLJISBLK,
    NV_R_YXCJNNGA_F_SLJISBLK_V_ACRDDIMM,
    NV_R_YXCJNNGA_F_SLJISBLK_FALSE,
    NV_R_YXCJNNGA_F_SLJISBLK_TRUE,
    NV_R_YXCJNNGA_F_BZCZRXGZ,
    NV_R_YXCJNNGA_F_BZCZRXGZ_V_ACRDDIMM,
    NV_R_YXCJNNGA_F_BZCZRXGZ_FALSE,
    NV_R_YXCJNNGA_F_BZCZRXGZ_TRUE,
)

# Register definitions
NV_R_SODTRZWY = RegisterMetadata(
    name='NV_R_SODTRZWY',
    address=0x11087c
)

NV_R_SODTRZWY_F_OJYQBJXU = FieldMetadata(
    name='NV_R_SODTRZWY_F_OJYQBJXU',
    msb=23,
    lsb=0,
    register=NV_R_SODTRZWY
)

NV_R_SODTRZWY_F_XFMUTNQZ = FieldMetadata(
    name='NV_R_SODTRZWY_F_XFMUTNQZ',
    msb=15,
    lsb=0,
    register=NV_R_SODTRZWY
)

NV_R_SODTRZWY_F_JGDYKRTC = FieldMetadata(
    name='NV_R_SODTRZWY_F_JGDYKRTC',
    msb=23,
    lsb=20,
    register=NV_R_SODTRZWY
)

NV_R_SODTRZWY_F_JGDYKRTC_V_WCVPXXHE = ValueMetadata(
    name='NV_R_SODTRZWY_F_JGDYKRTC_V_WCVPXXHE',
    value=7,
    field=NV_R_SODTRZWY_F_JGDYKRTC
)
NV_R_SODTRZWY_F_JGDYKRTC_V_PNYOQAQM = ValueMetadata(
    name='NV_R_SODTRZWY_F_JGDYKRTC_V_PNYOQAQM',
    value=4,
    field=NV_R_SODTRZWY_F_JGDYKRTC
)
NV_R_SODTRZWY_F_JGDYKRTC_V_KDIWUXVJ = ValueMetadata(
    name='NV_R_SODTRZWY_F_JGDYKRTC_V_KDIWUXVJ',
    value=1,
    field=NV_R_SODTRZWY_F_JGDYKRTC
)
NV_R_SODTRZWY_F_JGDYKRTC_V_XMGVQXQS = ValueMetadata(
    name='NV_R_SODTRZWY_F_JGDYKRTC_V_XMGVQXQS',
    value=2,
    field=NV_R_SODTRZWY_F_JGDYKRTC
)
NV_R_SODTRZWY_F_JGDYKRTC_V_AIDLSZMJ = ValueMetadata(
    name='NV_R_SODTRZWY_F_JGDYKRTC_V_AIDLSZMJ',
    value=6,
    field=NV_R_SODTRZWY_F_JGDYKRTC
)
NV_R_SODTRZWY_F_JGDYKRTC_V_VUSAYYPI = ValueMetadata(
    name='NV_R_SODTRZWY_F_JGDYKRTC_V_VUSAYYPI',
    value=0,
    field=NV_R_SODTRZWY_F_JGDYKRTC
)
NV_R_SODTRZWY_F_JGDYKRTC_V_MZYWYLOG = ValueMetadata(
    name='NV_R_SODTRZWY_F_JGDYKRTC_V_MZYWYLOG',
    value=3,
    field=NV_R_SODTRZWY_F_JGDYKRTC
)
NV_R_SODTRZWY_F_JGDYKRTC_V_EVNHGWNF = ValueMetadata(
    name='NV_R_SODTRZWY_F_JGDYKRTC_V_EVNHGWNF',
    value=5,
    field=NV_R_SODTRZWY_F_JGDYKRTC
)

NV_R_LMUBENNI = RegisterMetadata(
    name='NV_R_LMUBENNI',
    address=0x110880,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_LMUBENNI_F_CTBCFRUB = FieldMetadata(
    name='NV_R_LMUBENNI_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_LMUBENNI
)

NV_R_LMUBENNI_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LMUBENNI_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_LMUBENNI_F_CTBCFRUB
)

NV_R_LMUBENNI_F_QVYAARWP = FieldMetadata(
    name='NV_R_LMUBENNI_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_LMUBENNI
)

NV_R_LMUBENNI_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LMUBENNI_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_LMUBENNI_F_QVYAARWP
)

NV_R_WPJJMZJN = RegisterMetadata(
    name='NV_R_WPJJMZJN',
    address=0x110878,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_WPJJMZJN_F_MXXPKPJR = FieldMetadata(
    name='NV_R_WPJJMZJN_F_MXXPKPJR',
    msb=7,
    lsb=7,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_MXXPKPJR_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_MXXPKPJR_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_MXXPKPJR
)
NV_R_WPJJMZJN_F_MXXPKPJR_V_BYFONLES = ValueMetadata(
    name='NV_R_WPJJMZJN_F_MXXPKPJR_V_BYFONLES',
    value=1,
    field=NV_R_WPJJMZJN_F_MXXPKPJR
)
NV_R_WPJJMZJN_F_MXXPKPJR_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_MXXPKPJR_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_MXXPKPJR
)

NV_R_WPJJMZJN_F_LNUAVCHH = FieldMetadata(
    name='NV_R_WPJJMZJN_F_LNUAVCHH',
    msb=3,
    lsb=3,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_LNUAVCHH_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_LNUAVCHH_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_LNUAVCHH
)
NV_R_WPJJMZJN_F_LNUAVCHH_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_LNUAVCHH_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_LNUAVCHH
)
NV_R_WPJJMZJN_F_LNUAVCHH_V_PIUJQBKV = ValueMetadata(
    name='NV_R_WPJJMZJN_F_LNUAVCHH_V_PIUJQBKV',
    value=1,
    field=NV_R_WPJJMZJN_F_LNUAVCHH
)

NV_R_WPJJMZJN_F_YRPERXJI = FieldMetadata(
    name='NV_R_WPJJMZJN_F_YRPERXJI',
    msb=1,
    lsb=1,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_YRPERXJI_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_YRPERXJI_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_YRPERXJI
)
NV_R_WPJJMZJN_F_YRPERXJI_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_YRPERXJI_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_YRPERXJI
)
NV_R_WPJJMZJN_F_YRPERXJI_V_PIUJQBKV = ValueMetadata(
    name='NV_R_WPJJMZJN_F_YRPERXJI_V_PIUJQBKV',
    value=1,
    field=NV_R_WPJJMZJN_F_YRPERXJI
)

NV_R_WPJJMZJN_F_RUOLVTCC = FieldMetadata(
    name='NV_R_WPJJMZJN_F_RUOLVTCC',
    msb=5,
    lsb=5,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_RUOLVTCC_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_RUOLVTCC_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_RUOLVTCC
)
NV_R_WPJJMZJN_F_RUOLVTCC_V_BYFONLES = ValueMetadata(
    name='NV_R_WPJJMZJN_F_RUOLVTCC_V_BYFONLES',
    value=1,
    field=NV_R_WPJJMZJN_F_RUOLVTCC
)
NV_R_WPJJMZJN_F_RUOLVTCC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_RUOLVTCC_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_RUOLVTCC
)

NV_R_WPJJMZJN_F_JMCISOGY = FieldMetadata(
    name='NV_R_WPJJMZJN_F_JMCISOGY',
    msb=6,
    lsb=6,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_JMCISOGY_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_JMCISOGY_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_JMCISOGY
)
NV_R_WPJJMZJN_F_JMCISOGY_V_BYFONLES = ValueMetadata(
    name='NV_R_WPJJMZJN_F_JMCISOGY_V_BYFONLES',
    value=1,
    field=NV_R_WPJJMZJN_F_JMCISOGY
)
NV_R_WPJJMZJN_F_JMCISOGY_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_JMCISOGY_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_JMCISOGY
)

NV_R_WPJJMZJN_F_JLLZSLAE = FieldMetadata(
    name='NV_R_WPJJMZJN_F_JLLZSLAE',
    msb=0,
    lsb=0,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_JLLZSLAE_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_JLLZSLAE_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_JLLZSLAE
)
NV_R_WPJJMZJN_F_JLLZSLAE_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_JLLZSLAE_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_JLLZSLAE
)
NV_R_WPJJMZJN_F_JLLZSLAE_V_PIUJQBKV = ValueMetadata(
    name='NV_R_WPJJMZJN_F_JLLZSLAE_V_PIUJQBKV',
    value=1,
    field=NV_R_WPJJMZJN_F_JLLZSLAE
)

NV_R_WPJJMZJN_F_YZFIWFYH = FieldMetadata(
    name='NV_R_WPJJMZJN_F_YZFIWFYH',
    msb=2,
    lsb=2,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_YZFIWFYH_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_YZFIWFYH_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_YZFIWFYH
)
NV_R_WPJJMZJN_F_YZFIWFYH_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_YZFIWFYH_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_YZFIWFYH
)
NV_R_WPJJMZJN_F_YZFIWFYH_V_PIUJQBKV = ValueMetadata(
    name='NV_R_WPJJMZJN_F_YZFIWFYH_V_PIUJQBKV',
    value=1,
    field=NV_R_WPJJMZJN_F_YZFIWFYH
)

NV_R_WPJJMZJN_F_HEPGGPHE = FieldMetadata(
    name='NV_R_WPJJMZJN_F_HEPGGPHE',
    msb=4,
    lsb=4,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_HEPGGPHE_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_HEPGGPHE_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_HEPGGPHE
)
NV_R_WPJJMZJN_F_HEPGGPHE_V_BYFONLES = ValueMetadata(
    name='NV_R_WPJJMZJN_F_HEPGGPHE_V_BYFONLES',
    value=1,
    field=NV_R_WPJJMZJN_F_HEPGGPHE
)
NV_R_WPJJMZJN_F_HEPGGPHE_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_HEPGGPHE_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_HEPGGPHE
)

NV_R_WPJJMZJN_F_OJXVVOEC = FieldMetadata(
    name='NV_R_WPJJMZJN_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_OJXVVOEC_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_OJXVVOEC_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_OJXVVOEC
)
NV_R_WPJJMZJN_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_OJXVVOEC
)
NV_R_WPJJMZJN_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_WPJJMZJN_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_WPJJMZJN_F_OJXVVOEC
)

NV_R_WPJJMZJN_F_VEOHPHHB = FieldMetadata(
    name='NV_R_WPJJMZJN_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_VEOHPHHB_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_VEOHPHHB_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_VEOHPHHB
)
NV_R_WPJJMZJN_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_VEOHPHHB
)
NV_R_WPJJMZJN_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_WPJJMZJN_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_WPJJMZJN_F_VEOHPHHB
)

NV_R_WPJJMZJN_F_KVYOKDXN = FieldMetadata(
    name='NV_R_WPJJMZJN_F_KVYOKDXN',
    msb=31,
    lsb=31,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_WPJJMZJN_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_WPJJMZJN_F_KVYOKDXN
)
NV_R_WPJJMZJN_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_WPJJMZJN_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_WPJJMZJN_F_KVYOKDXN
)

NV_R_WPJJMZJN_F_GCVXAKOX = FieldMetadata(
    name='NV_R_WPJJMZJN_F_GCVXAKOX',
    msb=15,
    lsb=15,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_GCVXAKOX_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_GCVXAKOX_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_GCVXAKOX
)
NV_R_WPJJMZJN_F_GCVXAKOX_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_GCVXAKOX_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_GCVXAKOX
)
NV_R_WPJJMZJN_F_GCVXAKOX_V_PIUJQBKV = ValueMetadata(
    name='NV_R_WPJJMZJN_F_GCVXAKOX_V_PIUJQBKV',
    value=1,
    field=NV_R_WPJJMZJN_F_GCVXAKOX
)

NV_R_WPJJMZJN_F_YYABRASA = FieldMetadata(
    name='NV_R_WPJJMZJN_F_YYABRASA',
    msb=11,
    lsb=11,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_YYABRASA_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_YYABRASA_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_YYABRASA
)
NV_R_WPJJMZJN_F_YYABRASA_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_YYABRASA_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_YYABRASA
)
NV_R_WPJJMZJN_F_YYABRASA_V_PIUJQBKV = ValueMetadata(
    name='NV_R_WPJJMZJN_F_YYABRASA_V_PIUJQBKV',
    value=1,
    field=NV_R_WPJJMZJN_F_YYABRASA
)

NV_R_WPJJMZJN_F_MFLONDNB = FieldMetadata(
    name='NV_R_WPJJMZJN_F_MFLONDNB',
    msb=9,
    lsb=9,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_MFLONDNB_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_MFLONDNB_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_MFLONDNB
)
NV_R_WPJJMZJN_F_MFLONDNB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_MFLONDNB_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_MFLONDNB
)
NV_R_WPJJMZJN_F_MFLONDNB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_WPJJMZJN_F_MFLONDNB_V_PIUJQBKV',
    value=1,
    field=NV_R_WPJJMZJN_F_MFLONDNB
)

NV_R_WPJJMZJN_F_BFXAIXWT = FieldMetadata(
    name='NV_R_WPJJMZJN_F_BFXAIXWT',
    msb=13,
    lsb=13,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_BFXAIXWT_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_BFXAIXWT_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_BFXAIXWT
)
NV_R_WPJJMZJN_F_BFXAIXWT_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_BFXAIXWT_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_BFXAIXWT
)
NV_R_WPJJMZJN_F_BFXAIXWT_V_PIUJQBKV = ValueMetadata(
    name='NV_R_WPJJMZJN_F_BFXAIXWT_V_PIUJQBKV',
    value=1,
    field=NV_R_WPJJMZJN_F_BFXAIXWT
)

NV_R_WPJJMZJN_F_ZOVMXOOA = FieldMetadata(
    name='NV_R_WPJJMZJN_F_ZOVMXOOA',
    msb=14,
    lsb=14,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_ZOVMXOOA_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_ZOVMXOOA_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_ZOVMXOOA
)
NV_R_WPJJMZJN_F_ZOVMXOOA_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_ZOVMXOOA_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_ZOVMXOOA
)
NV_R_WPJJMZJN_F_ZOVMXOOA_V_PIUJQBKV = ValueMetadata(
    name='NV_R_WPJJMZJN_F_ZOVMXOOA_V_PIUJQBKV',
    value=1,
    field=NV_R_WPJJMZJN_F_ZOVMXOOA
)

NV_R_WPJJMZJN_F_VYVKXHMZ = FieldMetadata(
    name='NV_R_WPJJMZJN_F_VYVKXHMZ',
    msb=8,
    lsb=8,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_VYVKXHMZ_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_VYVKXHMZ_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_VYVKXHMZ
)
NV_R_WPJJMZJN_F_VYVKXHMZ_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_VYVKXHMZ_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_VYVKXHMZ
)
NV_R_WPJJMZJN_F_VYVKXHMZ_V_PIUJQBKV = ValueMetadata(
    name='NV_R_WPJJMZJN_F_VYVKXHMZ_V_PIUJQBKV',
    value=1,
    field=NV_R_WPJJMZJN_F_VYVKXHMZ
)

NV_R_WPJJMZJN_F_EPICIITL = FieldMetadata(
    name='NV_R_WPJJMZJN_F_EPICIITL',
    msb=10,
    lsb=10,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_EPICIITL_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_EPICIITL_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_EPICIITL
)
NV_R_WPJJMZJN_F_EPICIITL_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_EPICIITL_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_EPICIITL
)
NV_R_WPJJMZJN_F_EPICIITL_V_PIUJQBKV = ValueMetadata(
    name='NV_R_WPJJMZJN_F_EPICIITL_V_PIUJQBKV',
    value=1,
    field=NV_R_WPJJMZJN_F_EPICIITL
)

NV_R_WPJJMZJN_F_JAAGIRNF = FieldMetadata(
    name='NV_R_WPJJMZJN_F_JAAGIRNF',
    msb=12,
    lsb=12,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_JAAGIRNF_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_JAAGIRNF_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_JAAGIRNF
)
NV_R_WPJJMZJN_F_JAAGIRNF_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_JAAGIRNF_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_JAAGIRNF
)
NV_R_WPJJMZJN_F_JAAGIRNF_V_PIUJQBKV = ValueMetadata(
    name='NV_R_WPJJMZJN_F_JAAGIRNF_V_PIUJQBKV',
    value=1,
    field=NV_R_WPJJMZJN_F_JAAGIRNF
)

NV_R_WPJJMZJN_F_WQYWMRVP = FieldMetadata(
    name='NV_R_WPJJMZJN_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_WQYWMRVP_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_WQYWMRVP_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_WQYWMRVP
)
NV_R_WPJJMZJN_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_WQYWMRVP
)
NV_R_WPJJMZJN_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_WPJJMZJN_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_WPJJMZJN_F_WQYWMRVP
)

NV_R_WPJJMZJN_F_MLEOLJIS = FieldMetadata(
    name='NV_R_WPJJMZJN_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_WPJJMZJN
)

NV_R_WPJJMZJN_F_MLEOLJIS_V_ACRDDIMM = ValueMetadata(
    name='NV_R_WPJJMZJN_F_MLEOLJIS_V_ACRDDIMM',
    value=1,
    field=NV_R_WPJJMZJN_F_MLEOLJIS
)
NV_R_WPJJMZJN_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_WPJJMZJN_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_WPJJMZJN_F_MLEOLJIS
)
NV_R_WPJJMZJN_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_WPJJMZJN_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_WPJJMZJN_F_MLEOLJIS
)

NV_R_JWSZDUPO = RegisterMetadata(
    name='NV_R_JWSZDUPO',
    address=0x110884,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_JWSZDUPO_F_CTBCFRUB = FieldMetadata(
    name='NV_R_JWSZDUPO_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_JWSZDUPO
)

NV_R_JWSZDUPO_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_JWSZDUPO_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_JWSZDUPO_F_CTBCFRUB
)

NV_R_JWSZDUPO_F_QVYAARWP = FieldMetadata(
    name='NV_R_JWSZDUPO_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_JWSZDUPO
)

NV_R_JWSZDUPO_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_JWSZDUPO_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_JWSZDUPO_F_QVYAARWP
)

NV_R_UTAXXXPZ = RegisterMetadata(
    name='NV_R_UTAXXXPZ',
    address=0x111700,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_UTAXXXPZ_F_XSITITWP = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_XSITITWP',
    msb=5,
    lsb=5,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_XSITITWP_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_XSITITWP_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_XSITITWP
)
NV_R_UTAXXXPZ_F_XSITITWP_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_XSITITWP_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_XSITITWP
)

NV_R_UTAXXXPZ_F_NTFGMFRV = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_NTFGMFRV',
    msb=3,
    lsb=3,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_NTFGMFRV_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_NTFGMFRV_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_NTFGMFRV
)
NV_R_UTAXXXPZ_F_NTFGMFRV_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_NTFGMFRV_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_NTFGMFRV
)

NV_R_UTAXXXPZ_F_ERDYHKGJ = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_ERDYHKGJ',
    msb=16,
    lsb=16,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_ERDYHKGJ_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_ERDYHKGJ_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_ERDYHKGJ
)
NV_R_UTAXXXPZ_F_ERDYHKGJ_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_ERDYHKGJ_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_ERDYHKGJ
)

NV_R_UTAXXXPZ_F_XYSQFRPO = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_XYSQFRPO',
    msb=17,
    lsb=17,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_XYSQFRPO_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_XYSQFRPO_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_XYSQFRPO
)
NV_R_UTAXXXPZ_F_XYSQFRPO_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_XYSQFRPO_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_XYSQFRPO
)

NV_R_UTAXXXPZ_F_WKWRATQY = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_WKWRATQY',
    msb=18,
    lsb=18,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_WKWRATQY_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_WKWRATQY_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_WKWRATQY
)
NV_R_UTAXXXPZ_F_WKWRATQY_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_WKWRATQY_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_WKWRATQY
)

NV_R_UTAXXXPZ_F_MOZMIYSP = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_MOZMIYSP',
    msb=19,
    lsb=19,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_MOZMIYSP_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_MOZMIYSP_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_MOZMIYSP
)
NV_R_UTAXXXPZ_F_MOZMIYSP_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_MOZMIYSP_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_MOZMIYSP
)

NV_R_UTAXXXPZ_F_WBEEKQHD = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_WBEEKQHD',
    msb=20,
    lsb=20,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_WBEEKQHD_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_WBEEKQHD_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_WBEEKQHD
)
NV_R_UTAXXXPZ_F_WBEEKQHD_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_WBEEKQHD_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_WBEEKQHD
)

NV_R_UTAXXXPZ_F_ICTIMCRO = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_ICTIMCRO',
    msb=21,
    lsb=21,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_ICTIMCRO_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_ICTIMCRO_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_ICTIMCRO
)
NV_R_UTAXXXPZ_F_ICTIMCRO_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_ICTIMCRO_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_ICTIMCRO
)

NV_R_UTAXXXPZ_F_RUWZQWRJ = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_RUWZQWRJ',
    msb=22,
    lsb=22,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_RUWZQWRJ_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_RUWZQWRJ_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_RUWZQWRJ
)
NV_R_UTAXXXPZ_F_RUWZQWRJ_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_RUWZQWRJ_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_RUWZQWRJ
)

NV_R_UTAXXXPZ_F_YXYIKXGL = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_YXYIKXGL',
    msb=23,
    lsb=23,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_YXYIKXGL_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_YXYIKXGL_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_YXYIKXGL
)
NV_R_UTAXXXPZ_F_YXYIKXGL_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_YXYIKXGL_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_YXYIKXGL
)

NV_R_UTAXXXPZ_F_OZEFPLWY = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_OZEFPLWY',
    msb=11,
    lsb=11,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_OZEFPLWY_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_OZEFPLWY_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_OZEFPLWY
)
NV_R_UTAXXXPZ_F_OZEFPLWY_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_OZEFPLWY_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_OZEFPLWY
)

NV_R_UTAXXXPZ_F_DYUSITHC = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_DYUSITHC',
    msb=0,
    lsb=0,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_DYUSITHC_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_DYUSITHC_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_DYUSITHC
)
NV_R_UTAXXXPZ_F_DYUSITHC_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_DYUSITHC_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_DYUSITHC
)

NV_R_UTAXXXPZ_F_RDNJIHYP = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_RDNJIHYP',
    msb=4,
    lsb=4,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_RDNJIHYP_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_RDNJIHYP_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_RDNJIHYP
)
NV_R_UTAXXXPZ_F_RDNJIHYP_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_RDNJIHYP_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_RDNJIHYP
)

NV_R_UTAXXXPZ_F_WEPURKBS = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_WEPURKBS',
    msb=2,
    lsb=2,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_WEPURKBS_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_WEPURKBS_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_WEPURKBS
)
NV_R_UTAXXXPZ_F_WEPURKBS_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_WEPURKBS_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_WEPURKBS
)

NV_R_UTAXXXPZ_F_FEEUZQIV = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_FEEUZQIV',
    msb=12,
    lsb=12,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_FEEUZQIV_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_FEEUZQIV_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_FEEUZQIV
)
NV_R_UTAXXXPZ_F_FEEUZQIV_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_FEEUZQIV_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_FEEUZQIV
)

NV_R_UTAXXXPZ_F_BYZDKTIS = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_BYZDKTIS',
    msb=7,
    lsb=7,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_BYZDKTIS_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_BYZDKTIS_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_BYZDKTIS
)
NV_R_UTAXXXPZ_F_BYZDKTIS_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_BYZDKTIS_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_BYZDKTIS
)

NV_R_UTAXXXPZ_F_KVZRLISQ = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_KVZRLISQ',
    msb=1,
    lsb=1,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_KVZRLISQ_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_KVZRLISQ_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_KVZRLISQ
)
NV_R_UTAXXXPZ_F_KVZRLISQ_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_KVZRLISQ_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_KVZRLISQ
)

NV_R_UTAXXXPZ_F_OHMYUUQB = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_OHMYUUQB',
    msb=6,
    lsb=6,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_OHMYUUQB_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_OHMYUUQB_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_OHMYUUQB
)
NV_R_UTAXXXPZ_F_OHMYUUQB_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_OHMYUUQB_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_OHMYUUQB
)

NV_R_UTAXXXPZ_F_EIKVWFOG = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_EIKVWFOG',
    msb=9,
    lsb=9,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_EIKVWFOG_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_EIKVWFOG_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_EIKVWFOG
)
NV_R_UTAXXXPZ_F_EIKVWFOG_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_EIKVWFOG_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_EIKVWFOG
)

NV_R_UTAXXXPZ_F_FYAHGRZR = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_FYAHGRZR',
    msb=8,
    lsb=8,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_FYAHGRZR_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_FYAHGRZR_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_FYAHGRZR
)
NV_R_UTAXXXPZ_F_FYAHGRZR_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_FYAHGRZR_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_FYAHGRZR
)

NV_R_UTAXXXPZ_F_BUIAEGFB = FieldMetadata(
    name='NV_R_UTAXXXPZ_F_BUIAEGFB',
    msb=10,
    lsb=10,
    register=NV_R_UTAXXXPZ
)

NV_R_UTAXXXPZ_F_BUIAEGFB_V_SGMWIISS = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_BUIAEGFB_V_SGMWIISS',
    value=1,
    field=NV_R_UTAXXXPZ_F_BUIAEGFB
)
NV_R_UTAXXXPZ_F_BUIAEGFB_V_IMCKYHSN = ValueMetadata(
    name='NV_R_UTAXXXPZ_F_BUIAEGFB_V_IMCKYHSN',
    value=0,
    field=NV_R_UTAXXXPZ_F_BUIAEGFB
)

