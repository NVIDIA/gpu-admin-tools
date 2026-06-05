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
NV_R_VRCXWNKJ = RegisterMetadata(
    name='NV_R_VRCXWNKJ',
    address=0x520,
    zero_based=True
)

NV_R_VRCXWNKJ_F_ZIGJNGSD = FieldMetadata(
    name='NV_R_VRCXWNKJ_F_ZIGJNGSD',
    msb=22,
    lsb=0,
    register=NV_R_VRCXWNKJ
)

NV_R_VRCXWNKJ_F_SEXGFUSZ = FieldMetadata(
    name='NV_R_VRCXWNKJ_F_SEXGFUSZ',
    msb=21,
    lsb=20,
    register=NV_R_VRCXWNKJ
)

NV_R_VRCXWNKJ_F_GNCFTOVU = FieldMetadata(
    name='NV_R_VRCXWNKJ_F_GNCFTOVU',
    msb=2,
    lsb=0,
    register=NV_R_VRCXWNKJ
)

NV_R_VRCXWNKJ_F_WDHNBQNK = FieldMetadata(
    name='NV_R_VRCXWNKJ_F_WDHNBQNK',
    msb=19,
    lsb=8,
    register=NV_R_VRCXWNKJ
)

NV_R_VRCXWNKJ_F_HMBHUSZP = FieldMetadata(
    name='NV_R_VRCXWNKJ_F_HMBHUSZP',
    msb=22,
    lsb=22,
    register=NV_R_VRCXWNKJ
)

NV_R_VRCXWNKJ_F_FEERLBNG = FieldMetadata(
    name='NV_R_VRCXWNKJ_F_FEERLBNG',
    msb=7,
    lsb=3,
    register=NV_R_VRCXWNKJ
)

NV_R_VRCXWNKJ_F_EXMOKFEV = FieldMetadata(
    name='NV_R_VRCXWNKJ_F_EXMOKFEV',
    msb=7,
    lsb=6,
    register=NV_R_VRCXWNKJ
)

NV_R_BDWDWOOR = RegisterMetadata(
    name='NV_R_BDWDWOOR',
    address=0x4fc,
    zero_based=True
)

NV_R_BDWDWOOR_F_OJYQBJXU = FieldMetadata(
    name='NV_R_BDWDWOOR_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_SPGCMIVK = FieldMetadata(
    name='NV_R_BDWDWOOR_F_SPGCMIVK',
    msb=21,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_CMIRIWSV = FieldMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV',
    msb=29,
    lsb=22,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_CMIRIWSV_V_IZNDWFDP = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_IZNDWFDP',
    value=8,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_JSAHBBTM = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_JSAHBBTM',
    value=9,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_SEBMZEME = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_SEBMZEME',
    value=10,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_DTQCFPNA = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_DTQCFPNA',
    value=11,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_KYVQOIDU = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_KYVQOIDU',
    value=0,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_GHVVYIOF = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_GHVVYIOF',
    value=1,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_JGXXVPER = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_JGXXVPER',
    value=2,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_OKQHAWBB = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_OKQHAWBB',
    value=3,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_IGVRDHSA = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_IGVRDHSA',
    value=4,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_BJJBUDSL = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_BJJBUDSL',
    value=5,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_NRGNBBPQ = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_NRGNBBPQ',
    value=6,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_EKNISTBR = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_EKNISTBR',
    value=7,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_PGSYHHVE = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_PGSYHHVE',
    value=13,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_UXJGURAX = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_UXJGURAX',
    value=12,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_QWWBEQMS = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_QWWBEQMS',
    value=0,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_NBBWYRIG = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_NBBWYRIG',
    value=1,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_SWVVTNLK = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_SWVVTNLK',
    value=2,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_OTSWZFCR = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_OTSWZFCR',
    value=3,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_CSRKWDGO = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_CSRKWDGO',
    value=39,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_ORGICIHS = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_ORGICIHS',
    value=40,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_EYIXRFGX = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_EYIXRFGX',
    value=35,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_KFPOQASI = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_KFPOQASI',
    value=34,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_WPWBKHJE = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_WPWBKHJE',
    value=37,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_EEUGWZSG = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_EEUGWZSG',
    value=0,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_QUUGDQPL = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_QUUGDQPL',
    value=1,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_LKELNZMU = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_LKELNZMU',
    value=36,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_RSFATZWE = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_RSFATZWE',
    value=41,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_VGDLATGR = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_VGDLATGR',
    value=18,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_AWDMJNRQ = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_AWDMJNRQ',
    value=19,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_GDTZIIUX = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_GDTZIIUX',
    value=20,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_RGDBBIRS = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_RGDBBIRS',
    value=21,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_PYYFFBZF = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_PYYFFBZF',
    value=22,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_ZDAXSZOC = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_ZDAXSZOC',
    value=23,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_ZGKOZVLG = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_ZGKOZVLG',
    value=24,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_LXFZIIVC = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_LXFZIIVC',
    value=25,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_LIYVCEVC = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_LIYVCEVC',
    value=10,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_TMZYYQRE = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_TMZYYQRE',
    value=11,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_LITMRWHR = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_LITMRWHR',
    value=12,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_SUXZYCLN = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_SUXZYCLN',
    value=13,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_JKGEPTXY = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_JKGEPTXY',
    value=14,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_BIJQQCBV = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_BIJQQCBV',
    value=15,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_RCXNOXVH = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_RCXNOXVH',
    value=16,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_UPMZYBHK = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_UPMZYBHK',
    value=17,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_JKIFIDZS = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_JKIFIDZS',
    value=2,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_CWMIEYYA = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_CWMIEYYA',
    value=3,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_PTVPYSBH = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_PTVPYSBH',
    value=4,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_IGQUZTXI = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_IGQUZTXI',
    value=5,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_IPJEHZXT = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_IPJEHZXT',
    value=6,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_IBHISRTF = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_IBHISRTF',
    value=7,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_AHLHOHQY = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_AHLHOHQY',
    value=8,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_MAPDVIBO = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_MAPDVIBO',
    value=9,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_JFLKUSVU = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_JFLKUSVU',
    value=33,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_XZPNLKQC = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_XZPNLKQC',
    value=32,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_GZQSWWSJ = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_GZQSWWSJ',
    value=31,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_KHINLTNU = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_KHINLTNU',
    value=26,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_AYPUIJOI = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_AYPUIJOI',
    value=27,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_MDSIVZHL = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_MDSIVZHL',
    value=43,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_CFWZRRWE = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_CFWZRRWE',
    value=28,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_JSOPIVBC = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_JSOPIVBC',
    value=38,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_ZSEHHQVS = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_ZSEHHQVS',
    value=30,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_OOWFZASJ = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_OOWFZASJ',
    value=42,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)
NV_R_BDWDWOOR_F_CMIRIWSV_V_USZOAVIE = ValueMetadata(
    name='NV_R_BDWDWOOR_F_CMIRIWSV_V_USZOAVIE',
    value=29,
    field=NV_R_BDWDWOOR_F_CMIRIWSV
)

NV_R_BDWDWOOR_F_NVKIPOLT = FieldMetadata(
    name='NV_R_BDWDWOOR_F_NVKIPOLT',
    msb=11,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_ZOUDCMGL = FieldMetadata(
    name='NV_R_BDWDWOOR_F_ZOUDCMGL',
    msb=11,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_NOARSXQM = FieldMetadata(
    name='NV_R_BDWDWOOR_F_NOARSXQM',
    msb=11,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_OEUAITQM = FieldMetadata(
    name='NV_R_BDWDWOOR_F_OEUAITQM',
    msb=11,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_RBGFPZWZ = FieldMetadata(
    name='NV_R_BDWDWOOR_F_RBGFPZWZ',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_IMJFHBST = FieldMetadata(
    name='NV_R_BDWDWOOR_F_IMJFHBST',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_HYLTFYZK = FieldMetadata(
    name='NV_R_BDWDWOOR_F_HYLTFYZK',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_HBTOFJFB = FieldMetadata(
    name='NV_R_BDWDWOOR_F_HBTOFJFB',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_HHGTSGOX = FieldMetadata(
    name='NV_R_BDWDWOOR_F_HHGTSGOX',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_NBECFVTR = FieldMetadata(
    name='NV_R_BDWDWOOR_F_NBECFVTR',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_CPXQSYZB = FieldMetadata(
    name='NV_R_BDWDWOOR_F_CPXQSYZB',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_MBJISBNZ = FieldMetadata(
    name='NV_R_BDWDWOOR_F_MBJISBNZ',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_PDZWCDXW = FieldMetadata(
    name='NV_R_BDWDWOOR_F_PDZWCDXW',
    msb=9,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_OMCFLBAL = FieldMetadata(
    name='NV_R_BDWDWOOR_F_OMCFLBAL',
    msb=9,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_NKLXJYMD = FieldMetadata(
    name='NV_R_BDWDWOOR_F_NKLXJYMD',
    msb=9,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_IHQOCTPV = FieldMetadata(
    name='NV_R_BDWDWOOR_F_IHQOCTPV',
    msb=9,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_YJVWXNNE = FieldMetadata(
    name='NV_R_BDWDWOOR_F_YJVWXNNE',
    msb=9,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_VHLUWKFL = FieldMetadata(
    name='NV_R_BDWDWOOR_F_VHLUWKFL',
    msb=9,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_ZJAXCAQG = FieldMetadata(
    name='NV_R_BDWDWOOR_F_ZJAXCAQG',
    msb=8,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_OBEWYHPX = FieldMetadata(
    name='NV_R_BDWDWOOR_F_OBEWYHPX',
    msb=8,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_IRSEKWPQ = FieldMetadata(
    name='NV_R_BDWDWOOR_F_IRSEKWPQ',
    msb=16,
    lsb=9,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_VCXMQJEV = FieldMetadata(
    name='NV_R_BDWDWOOR_F_VCXMQJEV',
    msb=11,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_JXKCEYNM = FieldMetadata(
    name='NV_R_BDWDWOOR_F_JXKCEYNM',
    msb=8,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_DURSWDLP = FieldMetadata(
    name='NV_R_BDWDWOOR_F_DURSWDLP',
    msb=8,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_BJBZERKJ = FieldMetadata(
    name='NV_R_BDWDWOOR_F_BJBZERKJ',
    msb=8,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_MVPRIXOL = FieldMetadata(
    name='NV_R_BDWDWOOR_F_MVPRIXOL',
    msb=15,
    lsb=10,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_TDCDFVOD = FieldMetadata(
    name='NV_R_BDWDWOOR_F_TDCDFVOD',
    msb=9,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_XOCPWSRT = FieldMetadata(
    name='NV_R_BDWDWOOR_F_XOCPWSRT',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_INSVKLRN = FieldMetadata(
    name='NV_R_BDWDWOOR_F_INSVKLRN',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_NQFZWAHO = FieldMetadata(
    name='NV_R_BDWDWOOR_F_NQFZWAHO',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_DSQEGPFA = FieldMetadata(
    name='NV_R_BDWDWOOR_F_DSQEGPFA',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_IKVTHDBZ = FieldMetadata(
    name='NV_R_BDWDWOOR_F_IKVTHDBZ',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_CIWVBJVE = FieldMetadata(
    name='NV_R_BDWDWOOR_F_CIWVBJVE',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_LWLGMGAQ = FieldMetadata(
    name='NV_R_BDWDWOOR_F_LWLGMGAQ',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_OWIZHHHO = FieldMetadata(
    name='NV_R_BDWDWOOR_F_OWIZHHHO',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_MBHCYZPH = FieldMetadata(
    name='NV_R_BDWDWOOR_F_MBHCYZPH',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_YVRYVWIH = FieldMetadata(
    name='NV_R_BDWDWOOR_F_YVRYVWIH',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_UFKZEDCN = FieldMetadata(
    name='NV_R_BDWDWOOR_F_UFKZEDCN',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_QPRFLYFC = FieldMetadata(
    name='NV_R_BDWDWOOR_F_QPRFLYFC',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_KRYCNSZJ = FieldMetadata(
    name='NV_R_BDWDWOOR_F_KRYCNSZJ',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_GDDJPIHT = FieldMetadata(
    name='NV_R_BDWDWOOR_F_GDDJPIHT',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_PYWRHSKK = FieldMetadata(
    name='NV_R_BDWDWOOR_F_PYWRHSKK',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_SMFYIMNS = FieldMetadata(
    name='NV_R_BDWDWOOR_F_SMFYIMNS',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_HDYAZFOA = FieldMetadata(
    name='NV_R_BDWDWOOR_F_HDYAZFOA',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_UHBEOVYM = FieldMetadata(
    name='NV_R_BDWDWOOR_F_UHBEOVYM',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_ZICNZMHT = FieldMetadata(
    name='NV_R_BDWDWOOR_F_ZICNZMHT',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_ZJCFMQNK = FieldMetadata(
    name='NV_R_BDWDWOOR_F_ZJCFMQNK',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_DEBPYEVL = FieldMetadata(
    name='NV_R_BDWDWOOR_F_DEBPYEVL',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_AHTIHXXI = FieldMetadata(
    name='NV_R_BDWDWOOR_F_AHTIHXXI',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_RHTNASDF = FieldMetadata(
    name='NV_R_BDWDWOOR_F_RHTNASDF',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_DVLAWAYM = FieldMetadata(
    name='NV_R_BDWDWOOR_F_DVLAWAYM',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_IHLFYVUA = FieldMetadata(
    name='NV_R_BDWDWOOR_F_IHLFYVUA',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_VJZXXNRN = FieldMetadata(
    name='NV_R_BDWDWOOR_F_VJZXXNRN',
    msb=7,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_WPOKVHWB = FieldMetadata(
    name='NV_R_BDWDWOOR_F_WPOKVHWB',
    msb=11,
    lsb=4,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_QAWEVVQS = FieldMetadata(
    name='NV_R_BDWDWOOR_F_QAWEVVQS',
    msb=3,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_ECKYQVQO = FieldMetadata(
    name='NV_R_BDWDWOOR_F_ECKYQVQO',
    msb=11,
    lsb=4,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_AUKLBKOY = FieldMetadata(
    name='NV_R_BDWDWOOR_F_AUKLBKOY',
    msb=3,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_AZTHXCVL = FieldMetadata(
    name='NV_R_BDWDWOOR_F_AZTHXCVL',
    msb=11,
    lsb=4,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_FWIQTCRQ = FieldMetadata(
    name='NV_R_BDWDWOOR_F_FWIQTCRQ',
    msb=3,
    lsb=0,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_OFGCIMFI = FieldMetadata(
    name='NV_R_BDWDWOOR_F_OFGCIMFI',
    msb=31,
    lsb=30,
    register=NV_R_BDWDWOOR
)

NV_R_BDWDWOOR_F_OFGCIMFI_V_FQNJWQHN = ValueMetadata(
    name='NV_R_BDWDWOOR_F_OFGCIMFI_V_FQNJWQHN',
    value=2,
    field=NV_R_BDWDWOOR_F_OFGCIMFI
)
NV_R_BDWDWOOR_F_OFGCIMFI_V_CMVOBGUP = ValueMetadata(
    name='NV_R_BDWDWOOR_F_OFGCIMFI_V_CMVOBGUP',
    value=0,
    field=NV_R_BDWDWOOR_F_OFGCIMFI
)
NV_R_BDWDWOOR_F_OFGCIMFI_V_TVGHFSAG = ValueMetadata(
    name='NV_R_BDWDWOOR_F_OFGCIMFI_V_TVGHFSAG',
    value=1,
    field=NV_R_BDWDWOOR_F_OFGCIMFI
)

NV_R_FQBKROOJ = RegisterMetadata(
    name='NV_R_FQBKROOJ',
    address=0x4f4,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_FQBKROOJ_F_CTBCFRUB = FieldMetadata(
    name='NV_R_FQBKROOJ_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_FQBKROOJ
)

NV_R_FQBKROOJ_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_FQBKROOJ_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_FQBKROOJ_F_CTBCFRUB
)

NV_R_FQBKROOJ_F_QVYAARWP = FieldMetadata(
    name='NV_R_FQBKROOJ_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_FQBKROOJ
)

NV_R_FQBKROOJ_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_FQBKROOJ_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_FQBKROOJ_F_QVYAARWP
)

NV_R_NOISJRGU = RegisterMetadata(
    name='NV_R_NOISJRGU',
    address=0x4f0,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_NOISJRGU_F_KWDQPURW = FieldMetadata(
    name='NV_R_NOISJRGU_F_KWDQPURW',
    msb=5,
    lsb=5,
    register=NV_R_NOISJRGU
)

NV_R_NOISJRGU_F_KWDQPURW_V_DOGBFDTH = ValueMetadata(
    name='NV_R_NOISJRGU_F_KWDQPURW_V_DOGBFDTH',
    value=0,
    field=NV_R_NOISJRGU_F_KWDQPURW
)
NV_R_NOISJRGU_F_KWDQPURW_V_PIUJQBKV = ValueMetadata(
    name='NV_R_NOISJRGU_F_KWDQPURW_V_PIUJQBKV',
    value=1,
    field=NV_R_NOISJRGU_F_KWDQPURW
)

NV_R_NOISJRGU_F_ELAIAXBC = FieldMetadata(
    name='NV_R_NOISJRGU_F_ELAIAXBC',
    msb=1,
    lsb=1,
    register=NV_R_NOISJRGU
)

NV_R_NOISJRGU_F_ELAIAXBC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_NOISJRGU_F_ELAIAXBC_V_DOGBFDTH',
    value=0,
    field=NV_R_NOISJRGU_F_ELAIAXBC
)
NV_R_NOISJRGU_F_ELAIAXBC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_NOISJRGU_F_ELAIAXBC_V_PIUJQBKV',
    value=1,
    field=NV_R_NOISJRGU_F_ELAIAXBC
)

NV_R_NOISJRGU_F_OJXVVOEC = FieldMetadata(
    name='NV_R_NOISJRGU_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_NOISJRGU
)

NV_R_NOISJRGU_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_NOISJRGU_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_NOISJRGU_F_OJXVVOEC
)
NV_R_NOISJRGU_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_NOISJRGU_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_NOISJRGU_F_OJXVVOEC
)

NV_R_NOISJRGU_F_PBAFTNZG = FieldMetadata(
    name='NV_R_NOISJRGU_F_PBAFTNZG',
    msb=3,
    lsb=3,
    register=NV_R_NOISJRGU
)

NV_R_NOISJRGU_F_PBAFTNZG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_NOISJRGU_F_PBAFTNZG_V_DOGBFDTH',
    value=0,
    field=NV_R_NOISJRGU_F_PBAFTNZG
)
NV_R_NOISJRGU_F_PBAFTNZG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_NOISJRGU_F_PBAFTNZG_V_PIUJQBKV',
    value=1,
    field=NV_R_NOISJRGU_F_PBAFTNZG
)

NV_R_NOISJRGU_F_VEOHPHHB = FieldMetadata(
    name='NV_R_NOISJRGU_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_NOISJRGU
)

NV_R_NOISJRGU_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_NOISJRGU_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_NOISJRGU_F_VEOHPHHB
)
NV_R_NOISJRGU_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_NOISJRGU_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_NOISJRGU_F_VEOHPHHB
)

NV_R_NOISJRGU_F_KVYOKDXN = FieldMetadata(
    name='NV_R_NOISJRGU_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_NOISJRGU
)

NV_R_NOISJRGU_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_NOISJRGU_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_NOISJRGU_F_KVYOKDXN
)
NV_R_NOISJRGU_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_NOISJRGU_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_NOISJRGU_F_KVYOKDXN
)
NV_R_NOISJRGU_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_NOISJRGU_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_NOISJRGU_F_KVYOKDXN
)

NV_R_NOISJRGU_F_IOJLTETI = FieldMetadata(
    name='NV_R_NOISJRGU_F_IOJLTETI',
    msb=4,
    lsb=4,
    register=NV_R_NOISJRGU
)

NV_R_NOISJRGU_F_IOJLTETI_V_DOGBFDTH = ValueMetadata(
    name='NV_R_NOISJRGU_F_IOJLTETI_V_DOGBFDTH',
    value=0,
    field=NV_R_NOISJRGU_F_IOJLTETI
)
NV_R_NOISJRGU_F_IOJLTETI_V_PIUJQBKV = ValueMetadata(
    name='NV_R_NOISJRGU_F_IOJLTETI_V_PIUJQBKV',
    value=1,
    field=NV_R_NOISJRGU_F_IOJLTETI
)

NV_R_NOISJRGU_F_TZSQEAIB = FieldMetadata(
    name='NV_R_NOISJRGU_F_TZSQEAIB',
    msb=0,
    lsb=0,
    register=NV_R_NOISJRGU
)

NV_R_NOISJRGU_F_TZSQEAIB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_NOISJRGU_F_TZSQEAIB_V_DOGBFDTH',
    value=0,
    field=NV_R_NOISJRGU_F_TZSQEAIB
)
NV_R_NOISJRGU_F_TZSQEAIB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_NOISJRGU_F_TZSQEAIB_V_PIUJQBKV',
    value=1,
    field=NV_R_NOISJRGU_F_TZSQEAIB
)

NV_R_NOISJRGU_F_WQYWMRVP = FieldMetadata(
    name='NV_R_NOISJRGU_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_NOISJRGU
)

NV_R_NOISJRGU_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_NOISJRGU_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_NOISJRGU_F_WQYWMRVP
)
NV_R_NOISJRGU_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_NOISJRGU_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_NOISJRGU_F_WQYWMRVP
)

NV_R_NOISJRGU_F_PUPDHQBL = FieldMetadata(
    name='NV_R_NOISJRGU_F_PUPDHQBL',
    msb=2,
    lsb=2,
    register=NV_R_NOISJRGU
)

NV_R_NOISJRGU_F_PUPDHQBL_V_DOGBFDTH = ValueMetadata(
    name='NV_R_NOISJRGU_F_PUPDHQBL_V_DOGBFDTH',
    value=0,
    field=NV_R_NOISJRGU_F_PUPDHQBL
)
NV_R_NOISJRGU_F_PUPDHQBL_V_PIUJQBKV = ValueMetadata(
    name='NV_R_NOISJRGU_F_PUPDHQBL_V_PIUJQBKV',
    value=1,
    field=NV_R_NOISJRGU_F_PUPDHQBL
)

NV_R_NOISJRGU_F_MLEOLJIS = FieldMetadata(
    name='NV_R_NOISJRGU_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_NOISJRGU
)

NV_R_NOISJRGU_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_NOISJRGU_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_NOISJRGU_F_MLEOLJIS
)
NV_R_NOISJRGU_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_NOISJRGU_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_NOISJRGU_F_MLEOLJIS
)

NV_R_YUYBXWDP = RegisterMetadata(
    name='NV_R_YUYBXWDP',
    address=0x4f8,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_YUYBXWDP_F_CTBCFRUB = FieldMetadata(
    name='NV_R_YUYBXWDP_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_YUYBXWDP
)

NV_R_YUYBXWDP_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_YUYBXWDP_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_YUYBXWDP_F_CTBCFRUB
)

NV_R_YUYBXWDP_F_QVYAARWP = FieldMetadata(
    name='NV_R_YUYBXWDP_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_YUYBXWDP
)

NV_R_YUYBXWDP_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_YUYBXWDP_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_YUYBXWDP_F_QVYAARWP
)

NV_R_VUDLSKSF = RegisterMetadata(
    name='NV_R_VUDLSKSF',
    address=0x720,
    zero_based=True
)

NV_R_VUDLSKSF_F_ZIGJNGSD = FieldMetadata(
    name='NV_R_VUDLSKSF_F_ZIGJNGSD',
    msb=22,
    lsb=0,
    register=NV_R_VUDLSKSF
)

NV_R_VUDLSKSF_F_SEXGFUSZ = FieldMetadata(
    name='NV_R_VUDLSKSF_F_SEXGFUSZ',
    msb=21,
    lsb=20,
    register=NV_R_VUDLSKSF
)

NV_R_VUDLSKSF_F_GNCFTOVU = FieldMetadata(
    name='NV_R_VUDLSKSF_F_GNCFTOVU',
    msb=2,
    lsb=0,
    register=NV_R_VUDLSKSF
)

NV_R_VUDLSKSF_F_WDHNBQNK = FieldMetadata(
    name='NV_R_VUDLSKSF_F_WDHNBQNK',
    msb=19,
    lsb=8,
    register=NV_R_VUDLSKSF
)

NV_R_VUDLSKSF_F_HMBHUSZP = FieldMetadata(
    name='NV_R_VUDLSKSF_F_HMBHUSZP',
    msb=22,
    lsb=22,
    register=NV_R_VUDLSKSF
)

NV_R_VUDLSKSF_F_FEERLBNG = FieldMetadata(
    name='NV_R_VUDLSKSF_F_FEERLBNG',
    msb=7,
    lsb=3,
    register=NV_R_VUDLSKSF
)

NV_R_VUDLSKSF_F_EXMOKFEV = FieldMetadata(
    name='NV_R_VUDLSKSF_F_EXMOKFEV',
    msb=7,
    lsb=6,
    register=NV_R_VUDLSKSF
)

NV_R_EXNTVVJC = RegisterMetadata(
    name='NV_R_EXNTVVJC',
    address=0x6fc,
    zero_based=True
)

NV_R_EXNTVVJC_F_OJYQBJXU = FieldMetadata(
    name='NV_R_EXNTVVJC_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_SPGCMIVK = FieldMetadata(
    name='NV_R_EXNTVVJC_F_SPGCMIVK',
    msb=21,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_CMIRIWSV = FieldMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV',
    msb=29,
    lsb=22,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_CMIRIWSV_V_IZNDWFDP = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_IZNDWFDP',
    value=8,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_JSAHBBTM = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_JSAHBBTM',
    value=9,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_SEBMZEME = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_SEBMZEME',
    value=10,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_DTQCFPNA = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_DTQCFPNA',
    value=11,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_KYVQOIDU = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_KYVQOIDU',
    value=0,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_GHVVYIOF = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_GHVVYIOF',
    value=1,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_JGXXVPER = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_JGXXVPER',
    value=2,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_OKQHAWBB = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_OKQHAWBB',
    value=3,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_IGVRDHSA = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_IGVRDHSA',
    value=4,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_BJJBUDSL = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_BJJBUDSL',
    value=5,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_NRGNBBPQ = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_NRGNBBPQ',
    value=6,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_EKNISTBR = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_EKNISTBR',
    value=7,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_PGSYHHVE = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_PGSYHHVE',
    value=13,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_UXJGURAX = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_UXJGURAX',
    value=12,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_QWWBEQMS = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_QWWBEQMS',
    value=0,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_NBBWYRIG = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_NBBWYRIG',
    value=1,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_SWVVTNLK = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_SWVVTNLK',
    value=2,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_OTSWZFCR = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_OTSWZFCR',
    value=3,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_CSRKWDGO = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_CSRKWDGO',
    value=39,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_ORGICIHS = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_ORGICIHS',
    value=40,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_EYIXRFGX = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_EYIXRFGX',
    value=35,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_KFPOQASI = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_KFPOQASI',
    value=34,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_WPWBKHJE = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_WPWBKHJE',
    value=37,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_EEUGWZSG = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_EEUGWZSG',
    value=0,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_QUUGDQPL = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_QUUGDQPL',
    value=1,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_LKELNZMU = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_LKELNZMU',
    value=36,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_RSFATZWE = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_RSFATZWE',
    value=41,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_VGDLATGR = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_VGDLATGR',
    value=18,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_AWDMJNRQ = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_AWDMJNRQ',
    value=19,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_GDTZIIUX = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_GDTZIIUX',
    value=20,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_RGDBBIRS = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_RGDBBIRS',
    value=21,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_PYYFFBZF = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_PYYFFBZF',
    value=22,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_ZDAXSZOC = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_ZDAXSZOC',
    value=23,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_ZGKOZVLG = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_ZGKOZVLG',
    value=24,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_LXFZIIVC = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_LXFZIIVC',
    value=25,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_LIYVCEVC = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_LIYVCEVC',
    value=10,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_TMZYYQRE = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_TMZYYQRE',
    value=11,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_LITMRWHR = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_LITMRWHR',
    value=12,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_SUXZYCLN = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_SUXZYCLN',
    value=13,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_JKGEPTXY = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_JKGEPTXY',
    value=14,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_BIJQQCBV = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_BIJQQCBV',
    value=15,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_RCXNOXVH = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_RCXNOXVH',
    value=16,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_UPMZYBHK = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_UPMZYBHK',
    value=17,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_JKIFIDZS = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_JKIFIDZS',
    value=2,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_CWMIEYYA = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_CWMIEYYA',
    value=3,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_PTVPYSBH = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_PTVPYSBH',
    value=4,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_IGQUZTXI = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_IGQUZTXI',
    value=5,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_IPJEHZXT = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_IPJEHZXT',
    value=6,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_IBHISRTF = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_IBHISRTF',
    value=7,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_AHLHOHQY = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_AHLHOHQY',
    value=8,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_MAPDVIBO = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_MAPDVIBO',
    value=9,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_JFLKUSVU = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_JFLKUSVU',
    value=33,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_XZPNLKQC = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_XZPNLKQC',
    value=32,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_GZQSWWSJ = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_GZQSWWSJ',
    value=31,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_KHINLTNU = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_KHINLTNU',
    value=26,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_AYPUIJOI = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_AYPUIJOI',
    value=27,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_MDSIVZHL = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_MDSIVZHL',
    value=43,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_CFWZRRWE = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_CFWZRRWE',
    value=28,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_JSOPIVBC = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_JSOPIVBC',
    value=38,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_ZSEHHQVS = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_ZSEHHQVS',
    value=30,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_OOWFZASJ = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_OOWFZASJ',
    value=42,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)
NV_R_EXNTVVJC_F_CMIRIWSV_V_USZOAVIE = ValueMetadata(
    name='NV_R_EXNTVVJC_F_CMIRIWSV_V_USZOAVIE',
    value=29,
    field=NV_R_EXNTVVJC_F_CMIRIWSV
)

NV_R_EXNTVVJC_F_NVKIPOLT = FieldMetadata(
    name='NV_R_EXNTVVJC_F_NVKIPOLT',
    msb=11,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_ZOUDCMGL = FieldMetadata(
    name='NV_R_EXNTVVJC_F_ZOUDCMGL',
    msb=11,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_NOARSXQM = FieldMetadata(
    name='NV_R_EXNTVVJC_F_NOARSXQM',
    msb=11,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_OEUAITQM = FieldMetadata(
    name='NV_R_EXNTVVJC_F_OEUAITQM',
    msb=11,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_RBGFPZWZ = FieldMetadata(
    name='NV_R_EXNTVVJC_F_RBGFPZWZ',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_IMJFHBST = FieldMetadata(
    name='NV_R_EXNTVVJC_F_IMJFHBST',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_HYLTFYZK = FieldMetadata(
    name='NV_R_EXNTVVJC_F_HYLTFYZK',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_HBTOFJFB = FieldMetadata(
    name='NV_R_EXNTVVJC_F_HBTOFJFB',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_HHGTSGOX = FieldMetadata(
    name='NV_R_EXNTVVJC_F_HHGTSGOX',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_NBECFVTR = FieldMetadata(
    name='NV_R_EXNTVVJC_F_NBECFVTR',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_CPXQSYZB = FieldMetadata(
    name='NV_R_EXNTVVJC_F_CPXQSYZB',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_MBJISBNZ = FieldMetadata(
    name='NV_R_EXNTVVJC_F_MBJISBNZ',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_PDZWCDXW = FieldMetadata(
    name='NV_R_EXNTVVJC_F_PDZWCDXW',
    msb=9,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_OMCFLBAL = FieldMetadata(
    name='NV_R_EXNTVVJC_F_OMCFLBAL',
    msb=9,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_NKLXJYMD = FieldMetadata(
    name='NV_R_EXNTVVJC_F_NKLXJYMD',
    msb=9,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_IHQOCTPV = FieldMetadata(
    name='NV_R_EXNTVVJC_F_IHQOCTPV',
    msb=9,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_YJVWXNNE = FieldMetadata(
    name='NV_R_EXNTVVJC_F_YJVWXNNE',
    msb=9,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_VHLUWKFL = FieldMetadata(
    name='NV_R_EXNTVVJC_F_VHLUWKFL',
    msb=9,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_ZJAXCAQG = FieldMetadata(
    name='NV_R_EXNTVVJC_F_ZJAXCAQG',
    msb=8,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_OBEWYHPX = FieldMetadata(
    name='NV_R_EXNTVVJC_F_OBEWYHPX',
    msb=8,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_IRSEKWPQ = FieldMetadata(
    name='NV_R_EXNTVVJC_F_IRSEKWPQ',
    msb=16,
    lsb=9,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_VCXMQJEV = FieldMetadata(
    name='NV_R_EXNTVVJC_F_VCXMQJEV',
    msb=11,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_JXKCEYNM = FieldMetadata(
    name='NV_R_EXNTVVJC_F_JXKCEYNM',
    msb=8,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_DURSWDLP = FieldMetadata(
    name='NV_R_EXNTVVJC_F_DURSWDLP',
    msb=8,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_BJBZERKJ = FieldMetadata(
    name='NV_R_EXNTVVJC_F_BJBZERKJ',
    msb=8,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_MVPRIXOL = FieldMetadata(
    name='NV_R_EXNTVVJC_F_MVPRIXOL',
    msb=15,
    lsb=10,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_TDCDFVOD = FieldMetadata(
    name='NV_R_EXNTVVJC_F_TDCDFVOD',
    msb=9,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_XOCPWSRT = FieldMetadata(
    name='NV_R_EXNTVVJC_F_XOCPWSRT',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_INSVKLRN = FieldMetadata(
    name='NV_R_EXNTVVJC_F_INSVKLRN',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_NQFZWAHO = FieldMetadata(
    name='NV_R_EXNTVVJC_F_NQFZWAHO',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_DSQEGPFA = FieldMetadata(
    name='NV_R_EXNTVVJC_F_DSQEGPFA',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_IKVTHDBZ = FieldMetadata(
    name='NV_R_EXNTVVJC_F_IKVTHDBZ',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_CIWVBJVE = FieldMetadata(
    name='NV_R_EXNTVVJC_F_CIWVBJVE',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_LWLGMGAQ = FieldMetadata(
    name='NV_R_EXNTVVJC_F_LWLGMGAQ',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_OWIZHHHO = FieldMetadata(
    name='NV_R_EXNTVVJC_F_OWIZHHHO',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_MBHCYZPH = FieldMetadata(
    name='NV_R_EXNTVVJC_F_MBHCYZPH',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_YVRYVWIH = FieldMetadata(
    name='NV_R_EXNTVVJC_F_YVRYVWIH',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_UFKZEDCN = FieldMetadata(
    name='NV_R_EXNTVVJC_F_UFKZEDCN',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_QPRFLYFC = FieldMetadata(
    name='NV_R_EXNTVVJC_F_QPRFLYFC',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_KRYCNSZJ = FieldMetadata(
    name='NV_R_EXNTVVJC_F_KRYCNSZJ',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_GDDJPIHT = FieldMetadata(
    name='NV_R_EXNTVVJC_F_GDDJPIHT',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_PYWRHSKK = FieldMetadata(
    name='NV_R_EXNTVVJC_F_PYWRHSKK',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_SMFYIMNS = FieldMetadata(
    name='NV_R_EXNTVVJC_F_SMFYIMNS',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_HDYAZFOA = FieldMetadata(
    name='NV_R_EXNTVVJC_F_HDYAZFOA',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_UHBEOVYM = FieldMetadata(
    name='NV_R_EXNTVVJC_F_UHBEOVYM',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_ZICNZMHT = FieldMetadata(
    name='NV_R_EXNTVVJC_F_ZICNZMHT',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_ZJCFMQNK = FieldMetadata(
    name='NV_R_EXNTVVJC_F_ZJCFMQNK',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_DEBPYEVL = FieldMetadata(
    name='NV_R_EXNTVVJC_F_DEBPYEVL',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_AHTIHXXI = FieldMetadata(
    name='NV_R_EXNTVVJC_F_AHTIHXXI',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_RHTNASDF = FieldMetadata(
    name='NV_R_EXNTVVJC_F_RHTNASDF',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_DVLAWAYM = FieldMetadata(
    name='NV_R_EXNTVVJC_F_DVLAWAYM',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_IHLFYVUA = FieldMetadata(
    name='NV_R_EXNTVVJC_F_IHLFYVUA',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_VJZXXNRN = FieldMetadata(
    name='NV_R_EXNTVVJC_F_VJZXXNRN',
    msb=7,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_WPOKVHWB = FieldMetadata(
    name='NV_R_EXNTVVJC_F_WPOKVHWB',
    msb=11,
    lsb=4,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_QAWEVVQS = FieldMetadata(
    name='NV_R_EXNTVVJC_F_QAWEVVQS',
    msb=3,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_ECKYQVQO = FieldMetadata(
    name='NV_R_EXNTVVJC_F_ECKYQVQO',
    msb=11,
    lsb=4,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_AUKLBKOY = FieldMetadata(
    name='NV_R_EXNTVVJC_F_AUKLBKOY',
    msb=3,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_AZTHXCVL = FieldMetadata(
    name='NV_R_EXNTVVJC_F_AZTHXCVL',
    msb=11,
    lsb=4,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_FWIQTCRQ = FieldMetadata(
    name='NV_R_EXNTVVJC_F_FWIQTCRQ',
    msb=3,
    lsb=0,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_OFGCIMFI = FieldMetadata(
    name='NV_R_EXNTVVJC_F_OFGCIMFI',
    msb=31,
    lsb=30,
    register=NV_R_EXNTVVJC
)

NV_R_EXNTVVJC_F_OFGCIMFI_V_FQNJWQHN = ValueMetadata(
    name='NV_R_EXNTVVJC_F_OFGCIMFI_V_FQNJWQHN',
    value=2,
    field=NV_R_EXNTVVJC_F_OFGCIMFI
)
NV_R_EXNTVVJC_F_OFGCIMFI_V_CMVOBGUP = ValueMetadata(
    name='NV_R_EXNTVVJC_F_OFGCIMFI_V_CMVOBGUP',
    value=0,
    field=NV_R_EXNTVVJC_F_OFGCIMFI
)
NV_R_EXNTVVJC_F_OFGCIMFI_V_TVGHFSAG = ValueMetadata(
    name='NV_R_EXNTVVJC_F_OFGCIMFI_V_TVGHFSAG',
    value=1,
    field=NV_R_EXNTVVJC_F_OFGCIMFI
)

NV_R_ABQHSVUT = RegisterMetadata(
    name='NV_R_ABQHSVUT',
    address=0x6f4,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_ABQHSVUT_F_CTBCFRUB = FieldMetadata(
    name='NV_R_ABQHSVUT_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_ABQHSVUT
)

NV_R_ABQHSVUT_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_ABQHSVUT_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_ABQHSVUT_F_CTBCFRUB
)

NV_R_ABQHSVUT_F_QVYAARWP = FieldMetadata(
    name='NV_R_ABQHSVUT_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_ABQHSVUT
)

NV_R_ABQHSVUT_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_ABQHSVUT_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_ABQHSVUT_F_QVYAARWP
)

NV_R_SSSJZHWR = RegisterMetadata(
    name='NV_R_SSSJZHWR',
    address=0x6f0,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_SSSJZHWR_F_KWDQPURW = FieldMetadata(
    name='NV_R_SSSJZHWR_F_KWDQPURW',
    msb=5,
    lsb=5,
    register=NV_R_SSSJZHWR
)

NV_R_SSSJZHWR_F_KWDQPURW_V_DOGBFDTH = ValueMetadata(
    name='NV_R_SSSJZHWR_F_KWDQPURW_V_DOGBFDTH',
    value=0,
    field=NV_R_SSSJZHWR_F_KWDQPURW
)
NV_R_SSSJZHWR_F_KWDQPURW_V_PIUJQBKV = ValueMetadata(
    name='NV_R_SSSJZHWR_F_KWDQPURW_V_PIUJQBKV',
    value=1,
    field=NV_R_SSSJZHWR_F_KWDQPURW
)

NV_R_SSSJZHWR_F_ELAIAXBC = FieldMetadata(
    name='NV_R_SSSJZHWR_F_ELAIAXBC',
    msb=1,
    lsb=1,
    register=NV_R_SSSJZHWR
)

NV_R_SSSJZHWR_F_ELAIAXBC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_SSSJZHWR_F_ELAIAXBC_V_DOGBFDTH',
    value=0,
    field=NV_R_SSSJZHWR_F_ELAIAXBC
)
NV_R_SSSJZHWR_F_ELAIAXBC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_SSSJZHWR_F_ELAIAXBC_V_PIUJQBKV',
    value=1,
    field=NV_R_SSSJZHWR_F_ELAIAXBC
)

NV_R_SSSJZHWR_F_OJXVVOEC = FieldMetadata(
    name='NV_R_SSSJZHWR_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_SSSJZHWR
)

NV_R_SSSJZHWR_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_SSSJZHWR_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_SSSJZHWR_F_OJXVVOEC
)
NV_R_SSSJZHWR_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_SSSJZHWR_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_SSSJZHWR_F_OJXVVOEC
)

NV_R_SSSJZHWR_F_PBAFTNZG = FieldMetadata(
    name='NV_R_SSSJZHWR_F_PBAFTNZG',
    msb=3,
    lsb=3,
    register=NV_R_SSSJZHWR
)

NV_R_SSSJZHWR_F_PBAFTNZG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_SSSJZHWR_F_PBAFTNZG_V_DOGBFDTH',
    value=0,
    field=NV_R_SSSJZHWR_F_PBAFTNZG
)
NV_R_SSSJZHWR_F_PBAFTNZG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_SSSJZHWR_F_PBAFTNZG_V_PIUJQBKV',
    value=1,
    field=NV_R_SSSJZHWR_F_PBAFTNZG
)

NV_R_SSSJZHWR_F_VEOHPHHB = FieldMetadata(
    name='NV_R_SSSJZHWR_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_SSSJZHWR
)

NV_R_SSSJZHWR_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_SSSJZHWR_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_SSSJZHWR_F_VEOHPHHB
)
NV_R_SSSJZHWR_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_SSSJZHWR_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_SSSJZHWR_F_VEOHPHHB
)

NV_R_SSSJZHWR_F_KVYOKDXN = FieldMetadata(
    name='NV_R_SSSJZHWR_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_SSSJZHWR
)

NV_R_SSSJZHWR_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_SSSJZHWR_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_SSSJZHWR_F_KVYOKDXN
)
NV_R_SSSJZHWR_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_SSSJZHWR_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_SSSJZHWR_F_KVYOKDXN
)
NV_R_SSSJZHWR_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_SSSJZHWR_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_SSSJZHWR_F_KVYOKDXN
)

NV_R_SSSJZHWR_F_IOJLTETI = FieldMetadata(
    name='NV_R_SSSJZHWR_F_IOJLTETI',
    msb=4,
    lsb=4,
    register=NV_R_SSSJZHWR
)

NV_R_SSSJZHWR_F_IOJLTETI_V_DOGBFDTH = ValueMetadata(
    name='NV_R_SSSJZHWR_F_IOJLTETI_V_DOGBFDTH',
    value=0,
    field=NV_R_SSSJZHWR_F_IOJLTETI
)
NV_R_SSSJZHWR_F_IOJLTETI_V_PIUJQBKV = ValueMetadata(
    name='NV_R_SSSJZHWR_F_IOJLTETI_V_PIUJQBKV',
    value=1,
    field=NV_R_SSSJZHWR_F_IOJLTETI
)

NV_R_SSSJZHWR_F_TZSQEAIB = FieldMetadata(
    name='NV_R_SSSJZHWR_F_TZSQEAIB',
    msb=0,
    lsb=0,
    register=NV_R_SSSJZHWR
)

NV_R_SSSJZHWR_F_TZSQEAIB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_SSSJZHWR_F_TZSQEAIB_V_DOGBFDTH',
    value=0,
    field=NV_R_SSSJZHWR_F_TZSQEAIB
)
NV_R_SSSJZHWR_F_TZSQEAIB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_SSSJZHWR_F_TZSQEAIB_V_PIUJQBKV',
    value=1,
    field=NV_R_SSSJZHWR_F_TZSQEAIB
)

NV_R_SSSJZHWR_F_WQYWMRVP = FieldMetadata(
    name='NV_R_SSSJZHWR_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_SSSJZHWR
)

NV_R_SSSJZHWR_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_SSSJZHWR_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_SSSJZHWR_F_WQYWMRVP
)
NV_R_SSSJZHWR_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_SSSJZHWR_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_SSSJZHWR_F_WQYWMRVP
)

NV_R_SSSJZHWR_F_PUPDHQBL = FieldMetadata(
    name='NV_R_SSSJZHWR_F_PUPDHQBL',
    msb=2,
    lsb=2,
    register=NV_R_SSSJZHWR
)

NV_R_SSSJZHWR_F_PUPDHQBL_V_DOGBFDTH = ValueMetadata(
    name='NV_R_SSSJZHWR_F_PUPDHQBL_V_DOGBFDTH',
    value=0,
    field=NV_R_SSSJZHWR_F_PUPDHQBL
)
NV_R_SSSJZHWR_F_PUPDHQBL_V_PIUJQBKV = ValueMetadata(
    name='NV_R_SSSJZHWR_F_PUPDHQBL_V_PIUJQBKV',
    value=1,
    field=NV_R_SSSJZHWR_F_PUPDHQBL
)

NV_R_SSSJZHWR_F_MLEOLJIS = FieldMetadata(
    name='NV_R_SSSJZHWR_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_SSSJZHWR
)

NV_R_SSSJZHWR_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_SSSJZHWR_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_SSSJZHWR_F_MLEOLJIS
)
NV_R_SSSJZHWR_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_SSSJZHWR_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_SSSJZHWR_F_MLEOLJIS
)

NV_R_VGYGCVFQ = RegisterMetadata(
    name='NV_R_VGYGCVFQ',
    address=0x6f8,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_VGYGCVFQ_F_CTBCFRUB = FieldMetadata(
    name='NV_R_VGYGCVFQ_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_VGYGCVFQ
)

NV_R_VGYGCVFQ_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_VGYGCVFQ_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_VGYGCVFQ_F_CTBCFRUB
)

NV_R_VGYGCVFQ_F_QVYAARWP = FieldMetadata(
    name='NV_R_VGYGCVFQ_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_VGYGCVFQ
)

NV_R_VGYGCVFQ_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_VGYGCVFQ_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_VGYGCVFQ_F_QVYAARWP
)

NV_R_QBOLOAXZ = RegisterMetadata(
    name='NV_R_QBOLOAXZ',
    address=0x920,
    zero_based=True
)

NV_R_QBOLOAXZ_F_ZIGJNGSD = FieldMetadata(
    name='NV_R_QBOLOAXZ_F_ZIGJNGSD',
    msb=22,
    lsb=0,
    register=NV_R_QBOLOAXZ
)

NV_R_QBOLOAXZ_F_SEXGFUSZ = FieldMetadata(
    name='NV_R_QBOLOAXZ_F_SEXGFUSZ',
    msb=21,
    lsb=20,
    register=NV_R_QBOLOAXZ
)

NV_R_QBOLOAXZ_F_GNCFTOVU = FieldMetadata(
    name='NV_R_QBOLOAXZ_F_GNCFTOVU',
    msb=2,
    lsb=0,
    register=NV_R_QBOLOAXZ
)

NV_R_QBOLOAXZ_F_WDHNBQNK = FieldMetadata(
    name='NV_R_QBOLOAXZ_F_WDHNBQNK',
    msb=19,
    lsb=8,
    register=NV_R_QBOLOAXZ
)

NV_R_QBOLOAXZ_F_HMBHUSZP = FieldMetadata(
    name='NV_R_QBOLOAXZ_F_HMBHUSZP',
    msb=22,
    lsb=22,
    register=NV_R_QBOLOAXZ
)

NV_R_QBOLOAXZ_F_FEERLBNG = FieldMetadata(
    name='NV_R_QBOLOAXZ_F_FEERLBNG',
    msb=7,
    lsb=3,
    register=NV_R_QBOLOAXZ
)

NV_R_QBOLOAXZ_F_EXMOKFEV = FieldMetadata(
    name='NV_R_QBOLOAXZ_F_EXMOKFEV',
    msb=7,
    lsb=6,
    register=NV_R_QBOLOAXZ
)

NV_R_GQMUSMWE = RegisterMetadata(
    name='NV_R_GQMUSMWE',
    address=0x8fc,
    zero_based=True
)

NV_R_GQMUSMWE_F_OJYQBJXU = FieldMetadata(
    name='NV_R_GQMUSMWE_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_SPGCMIVK = FieldMetadata(
    name='NV_R_GQMUSMWE_F_SPGCMIVK',
    msb=21,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_CMIRIWSV = FieldMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV',
    msb=29,
    lsb=22,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_CMIRIWSV_V_IZNDWFDP = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_IZNDWFDP',
    value=8,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_JSAHBBTM = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_JSAHBBTM',
    value=9,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_SEBMZEME = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_SEBMZEME',
    value=10,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_DTQCFPNA = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_DTQCFPNA',
    value=11,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_KYVQOIDU = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_KYVQOIDU',
    value=0,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_GHVVYIOF = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_GHVVYIOF',
    value=1,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_JGXXVPER = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_JGXXVPER',
    value=2,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_OKQHAWBB = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_OKQHAWBB',
    value=3,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_IGVRDHSA = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_IGVRDHSA',
    value=4,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_BJJBUDSL = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_BJJBUDSL',
    value=5,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_NRGNBBPQ = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_NRGNBBPQ',
    value=6,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_EKNISTBR = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_EKNISTBR',
    value=7,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_PGSYHHVE = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_PGSYHHVE',
    value=13,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_UXJGURAX = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_UXJGURAX',
    value=12,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_QWWBEQMS = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_QWWBEQMS',
    value=0,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_NBBWYRIG = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_NBBWYRIG',
    value=1,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_SWVVTNLK = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_SWVVTNLK',
    value=2,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_OTSWZFCR = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_OTSWZFCR',
    value=3,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_CSRKWDGO = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_CSRKWDGO',
    value=39,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_ORGICIHS = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_ORGICIHS',
    value=40,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_EYIXRFGX = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_EYIXRFGX',
    value=35,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_KFPOQASI = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_KFPOQASI',
    value=34,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_WPWBKHJE = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_WPWBKHJE',
    value=37,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_EEUGWZSG = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_EEUGWZSG',
    value=0,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_QUUGDQPL = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_QUUGDQPL',
    value=1,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_LKELNZMU = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_LKELNZMU',
    value=36,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_RSFATZWE = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_RSFATZWE',
    value=41,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_VGDLATGR = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_VGDLATGR',
    value=18,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_AWDMJNRQ = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_AWDMJNRQ',
    value=19,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_GDTZIIUX = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_GDTZIIUX',
    value=20,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_RGDBBIRS = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_RGDBBIRS',
    value=21,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_PYYFFBZF = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_PYYFFBZF',
    value=22,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_ZDAXSZOC = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_ZDAXSZOC',
    value=23,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_ZGKOZVLG = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_ZGKOZVLG',
    value=24,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_LXFZIIVC = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_LXFZIIVC',
    value=25,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_LIYVCEVC = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_LIYVCEVC',
    value=10,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_TMZYYQRE = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_TMZYYQRE',
    value=11,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_LITMRWHR = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_LITMRWHR',
    value=12,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_SUXZYCLN = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_SUXZYCLN',
    value=13,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_JKGEPTXY = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_JKGEPTXY',
    value=14,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_BIJQQCBV = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_BIJQQCBV',
    value=15,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_RCXNOXVH = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_RCXNOXVH',
    value=16,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_UPMZYBHK = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_UPMZYBHK',
    value=17,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_JKIFIDZS = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_JKIFIDZS',
    value=2,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_CWMIEYYA = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_CWMIEYYA',
    value=3,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_PTVPYSBH = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_PTVPYSBH',
    value=4,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_IGQUZTXI = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_IGQUZTXI',
    value=5,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_IPJEHZXT = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_IPJEHZXT',
    value=6,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_IBHISRTF = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_IBHISRTF',
    value=7,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_AHLHOHQY = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_AHLHOHQY',
    value=8,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_MAPDVIBO = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_MAPDVIBO',
    value=9,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_JFLKUSVU = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_JFLKUSVU',
    value=33,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_XZPNLKQC = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_XZPNLKQC',
    value=32,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_GZQSWWSJ = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_GZQSWWSJ',
    value=31,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_KHINLTNU = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_KHINLTNU',
    value=26,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_AYPUIJOI = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_AYPUIJOI',
    value=27,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_MDSIVZHL = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_MDSIVZHL',
    value=43,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_CFWZRRWE = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_CFWZRRWE',
    value=28,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_JSOPIVBC = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_JSOPIVBC',
    value=38,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_ZSEHHQVS = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_ZSEHHQVS',
    value=30,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_OOWFZASJ = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_OOWFZASJ',
    value=42,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)
NV_R_GQMUSMWE_F_CMIRIWSV_V_USZOAVIE = ValueMetadata(
    name='NV_R_GQMUSMWE_F_CMIRIWSV_V_USZOAVIE',
    value=29,
    field=NV_R_GQMUSMWE_F_CMIRIWSV
)

NV_R_GQMUSMWE_F_NVKIPOLT = FieldMetadata(
    name='NV_R_GQMUSMWE_F_NVKIPOLT',
    msb=11,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_ZOUDCMGL = FieldMetadata(
    name='NV_R_GQMUSMWE_F_ZOUDCMGL',
    msb=11,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_NOARSXQM = FieldMetadata(
    name='NV_R_GQMUSMWE_F_NOARSXQM',
    msb=11,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_OEUAITQM = FieldMetadata(
    name='NV_R_GQMUSMWE_F_OEUAITQM',
    msb=11,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_RBGFPZWZ = FieldMetadata(
    name='NV_R_GQMUSMWE_F_RBGFPZWZ',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_IMJFHBST = FieldMetadata(
    name='NV_R_GQMUSMWE_F_IMJFHBST',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_HYLTFYZK = FieldMetadata(
    name='NV_R_GQMUSMWE_F_HYLTFYZK',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_HBTOFJFB = FieldMetadata(
    name='NV_R_GQMUSMWE_F_HBTOFJFB',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_HHGTSGOX = FieldMetadata(
    name='NV_R_GQMUSMWE_F_HHGTSGOX',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_NBECFVTR = FieldMetadata(
    name='NV_R_GQMUSMWE_F_NBECFVTR',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_CPXQSYZB = FieldMetadata(
    name='NV_R_GQMUSMWE_F_CPXQSYZB',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_MBJISBNZ = FieldMetadata(
    name='NV_R_GQMUSMWE_F_MBJISBNZ',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_PDZWCDXW = FieldMetadata(
    name='NV_R_GQMUSMWE_F_PDZWCDXW',
    msb=9,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_OMCFLBAL = FieldMetadata(
    name='NV_R_GQMUSMWE_F_OMCFLBAL',
    msb=9,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_NKLXJYMD = FieldMetadata(
    name='NV_R_GQMUSMWE_F_NKLXJYMD',
    msb=9,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_IHQOCTPV = FieldMetadata(
    name='NV_R_GQMUSMWE_F_IHQOCTPV',
    msb=9,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_YJVWXNNE = FieldMetadata(
    name='NV_R_GQMUSMWE_F_YJVWXNNE',
    msb=9,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_VHLUWKFL = FieldMetadata(
    name='NV_R_GQMUSMWE_F_VHLUWKFL',
    msb=9,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_ZJAXCAQG = FieldMetadata(
    name='NV_R_GQMUSMWE_F_ZJAXCAQG',
    msb=8,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_OBEWYHPX = FieldMetadata(
    name='NV_R_GQMUSMWE_F_OBEWYHPX',
    msb=8,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_IRSEKWPQ = FieldMetadata(
    name='NV_R_GQMUSMWE_F_IRSEKWPQ',
    msb=16,
    lsb=9,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_VCXMQJEV = FieldMetadata(
    name='NV_R_GQMUSMWE_F_VCXMQJEV',
    msb=11,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_JXKCEYNM = FieldMetadata(
    name='NV_R_GQMUSMWE_F_JXKCEYNM',
    msb=8,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_DURSWDLP = FieldMetadata(
    name='NV_R_GQMUSMWE_F_DURSWDLP',
    msb=8,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_BJBZERKJ = FieldMetadata(
    name='NV_R_GQMUSMWE_F_BJBZERKJ',
    msb=8,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_MVPRIXOL = FieldMetadata(
    name='NV_R_GQMUSMWE_F_MVPRIXOL',
    msb=15,
    lsb=10,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_TDCDFVOD = FieldMetadata(
    name='NV_R_GQMUSMWE_F_TDCDFVOD',
    msb=9,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_XOCPWSRT = FieldMetadata(
    name='NV_R_GQMUSMWE_F_XOCPWSRT',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_INSVKLRN = FieldMetadata(
    name='NV_R_GQMUSMWE_F_INSVKLRN',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_NQFZWAHO = FieldMetadata(
    name='NV_R_GQMUSMWE_F_NQFZWAHO',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_DSQEGPFA = FieldMetadata(
    name='NV_R_GQMUSMWE_F_DSQEGPFA',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_IKVTHDBZ = FieldMetadata(
    name='NV_R_GQMUSMWE_F_IKVTHDBZ',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_CIWVBJVE = FieldMetadata(
    name='NV_R_GQMUSMWE_F_CIWVBJVE',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_LWLGMGAQ = FieldMetadata(
    name='NV_R_GQMUSMWE_F_LWLGMGAQ',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_OWIZHHHO = FieldMetadata(
    name='NV_R_GQMUSMWE_F_OWIZHHHO',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_MBHCYZPH = FieldMetadata(
    name='NV_R_GQMUSMWE_F_MBHCYZPH',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_YVRYVWIH = FieldMetadata(
    name='NV_R_GQMUSMWE_F_YVRYVWIH',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_UFKZEDCN = FieldMetadata(
    name='NV_R_GQMUSMWE_F_UFKZEDCN',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_QPRFLYFC = FieldMetadata(
    name='NV_R_GQMUSMWE_F_QPRFLYFC',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_KRYCNSZJ = FieldMetadata(
    name='NV_R_GQMUSMWE_F_KRYCNSZJ',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_GDDJPIHT = FieldMetadata(
    name='NV_R_GQMUSMWE_F_GDDJPIHT',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_PYWRHSKK = FieldMetadata(
    name='NV_R_GQMUSMWE_F_PYWRHSKK',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_SMFYIMNS = FieldMetadata(
    name='NV_R_GQMUSMWE_F_SMFYIMNS',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_HDYAZFOA = FieldMetadata(
    name='NV_R_GQMUSMWE_F_HDYAZFOA',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_UHBEOVYM = FieldMetadata(
    name='NV_R_GQMUSMWE_F_UHBEOVYM',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_ZICNZMHT = FieldMetadata(
    name='NV_R_GQMUSMWE_F_ZICNZMHT',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_ZJCFMQNK = FieldMetadata(
    name='NV_R_GQMUSMWE_F_ZJCFMQNK',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_DEBPYEVL = FieldMetadata(
    name='NV_R_GQMUSMWE_F_DEBPYEVL',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_AHTIHXXI = FieldMetadata(
    name='NV_R_GQMUSMWE_F_AHTIHXXI',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_RHTNASDF = FieldMetadata(
    name='NV_R_GQMUSMWE_F_RHTNASDF',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_DVLAWAYM = FieldMetadata(
    name='NV_R_GQMUSMWE_F_DVLAWAYM',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_IHLFYVUA = FieldMetadata(
    name='NV_R_GQMUSMWE_F_IHLFYVUA',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_VJZXXNRN = FieldMetadata(
    name='NV_R_GQMUSMWE_F_VJZXXNRN',
    msb=7,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_WPOKVHWB = FieldMetadata(
    name='NV_R_GQMUSMWE_F_WPOKVHWB',
    msb=11,
    lsb=4,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_QAWEVVQS = FieldMetadata(
    name='NV_R_GQMUSMWE_F_QAWEVVQS',
    msb=3,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_ECKYQVQO = FieldMetadata(
    name='NV_R_GQMUSMWE_F_ECKYQVQO',
    msb=11,
    lsb=4,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_AUKLBKOY = FieldMetadata(
    name='NV_R_GQMUSMWE_F_AUKLBKOY',
    msb=3,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_AZTHXCVL = FieldMetadata(
    name='NV_R_GQMUSMWE_F_AZTHXCVL',
    msb=11,
    lsb=4,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_FWIQTCRQ = FieldMetadata(
    name='NV_R_GQMUSMWE_F_FWIQTCRQ',
    msb=3,
    lsb=0,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_OFGCIMFI = FieldMetadata(
    name='NV_R_GQMUSMWE_F_OFGCIMFI',
    msb=31,
    lsb=30,
    register=NV_R_GQMUSMWE
)

NV_R_GQMUSMWE_F_OFGCIMFI_V_FQNJWQHN = ValueMetadata(
    name='NV_R_GQMUSMWE_F_OFGCIMFI_V_FQNJWQHN',
    value=2,
    field=NV_R_GQMUSMWE_F_OFGCIMFI
)
NV_R_GQMUSMWE_F_OFGCIMFI_V_CMVOBGUP = ValueMetadata(
    name='NV_R_GQMUSMWE_F_OFGCIMFI_V_CMVOBGUP',
    value=0,
    field=NV_R_GQMUSMWE_F_OFGCIMFI
)
NV_R_GQMUSMWE_F_OFGCIMFI_V_TVGHFSAG = ValueMetadata(
    name='NV_R_GQMUSMWE_F_OFGCIMFI_V_TVGHFSAG',
    value=1,
    field=NV_R_GQMUSMWE_F_OFGCIMFI
)

NV_R_GRVJYLUC = RegisterMetadata(
    name='NV_R_GRVJYLUC',
    address=0x8f4,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_GRVJYLUC_F_CTBCFRUB = FieldMetadata(
    name='NV_R_GRVJYLUC_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_GRVJYLUC
)

NV_R_GRVJYLUC_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_GRVJYLUC_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_GRVJYLUC_F_CTBCFRUB
)

NV_R_GRVJYLUC_F_QVYAARWP = FieldMetadata(
    name='NV_R_GRVJYLUC_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_GRVJYLUC
)

NV_R_GRVJYLUC_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_GRVJYLUC_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_GRVJYLUC_F_QVYAARWP
)

NV_R_PYQJLPSK = RegisterMetadata(
    name='NV_R_PYQJLPSK',
    address=0x8f0,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_PYQJLPSK_F_KWDQPURW = FieldMetadata(
    name='NV_R_PYQJLPSK_F_KWDQPURW',
    msb=5,
    lsb=5,
    register=NV_R_PYQJLPSK
)

NV_R_PYQJLPSK_F_KWDQPURW_V_DOGBFDTH = ValueMetadata(
    name='NV_R_PYQJLPSK_F_KWDQPURW_V_DOGBFDTH',
    value=0,
    field=NV_R_PYQJLPSK_F_KWDQPURW
)
NV_R_PYQJLPSK_F_KWDQPURW_V_PIUJQBKV = ValueMetadata(
    name='NV_R_PYQJLPSK_F_KWDQPURW_V_PIUJQBKV',
    value=1,
    field=NV_R_PYQJLPSK_F_KWDQPURW
)

NV_R_PYQJLPSK_F_ELAIAXBC = FieldMetadata(
    name='NV_R_PYQJLPSK_F_ELAIAXBC',
    msb=1,
    lsb=1,
    register=NV_R_PYQJLPSK
)

NV_R_PYQJLPSK_F_ELAIAXBC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_PYQJLPSK_F_ELAIAXBC_V_DOGBFDTH',
    value=0,
    field=NV_R_PYQJLPSK_F_ELAIAXBC
)
NV_R_PYQJLPSK_F_ELAIAXBC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_PYQJLPSK_F_ELAIAXBC_V_PIUJQBKV',
    value=1,
    field=NV_R_PYQJLPSK_F_ELAIAXBC
)

NV_R_PYQJLPSK_F_OJXVVOEC = FieldMetadata(
    name='NV_R_PYQJLPSK_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_PYQJLPSK
)

NV_R_PYQJLPSK_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_PYQJLPSK_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_PYQJLPSK_F_OJXVVOEC
)
NV_R_PYQJLPSK_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_PYQJLPSK_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_PYQJLPSK_F_OJXVVOEC
)

NV_R_PYQJLPSK_F_PBAFTNZG = FieldMetadata(
    name='NV_R_PYQJLPSK_F_PBAFTNZG',
    msb=3,
    lsb=3,
    register=NV_R_PYQJLPSK
)

NV_R_PYQJLPSK_F_PBAFTNZG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_PYQJLPSK_F_PBAFTNZG_V_DOGBFDTH',
    value=0,
    field=NV_R_PYQJLPSK_F_PBAFTNZG
)
NV_R_PYQJLPSK_F_PBAFTNZG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_PYQJLPSK_F_PBAFTNZG_V_PIUJQBKV',
    value=1,
    field=NV_R_PYQJLPSK_F_PBAFTNZG
)

NV_R_PYQJLPSK_F_VEOHPHHB = FieldMetadata(
    name='NV_R_PYQJLPSK_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_PYQJLPSK
)

NV_R_PYQJLPSK_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_PYQJLPSK_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_PYQJLPSK_F_VEOHPHHB
)
NV_R_PYQJLPSK_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_PYQJLPSK_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_PYQJLPSK_F_VEOHPHHB
)

NV_R_PYQJLPSK_F_KVYOKDXN = FieldMetadata(
    name='NV_R_PYQJLPSK_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_PYQJLPSK
)

NV_R_PYQJLPSK_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_PYQJLPSK_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_PYQJLPSK_F_KVYOKDXN
)
NV_R_PYQJLPSK_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_PYQJLPSK_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_PYQJLPSK_F_KVYOKDXN
)
NV_R_PYQJLPSK_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_PYQJLPSK_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_PYQJLPSK_F_KVYOKDXN
)

NV_R_PYQJLPSK_F_IOJLTETI = FieldMetadata(
    name='NV_R_PYQJLPSK_F_IOJLTETI',
    msb=4,
    lsb=4,
    register=NV_R_PYQJLPSK
)

NV_R_PYQJLPSK_F_IOJLTETI_V_DOGBFDTH = ValueMetadata(
    name='NV_R_PYQJLPSK_F_IOJLTETI_V_DOGBFDTH',
    value=0,
    field=NV_R_PYQJLPSK_F_IOJLTETI
)
NV_R_PYQJLPSK_F_IOJLTETI_V_PIUJQBKV = ValueMetadata(
    name='NV_R_PYQJLPSK_F_IOJLTETI_V_PIUJQBKV',
    value=1,
    field=NV_R_PYQJLPSK_F_IOJLTETI
)

NV_R_PYQJLPSK_F_TZSQEAIB = FieldMetadata(
    name='NV_R_PYQJLPSK_F_TZSQEAIB',
    msb=0,
    lsb=0,
    register=NV_R_PYQJLPSK
)

NV_R_PYQJLPSK_F_TZSQEAIB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_PYQJLPSK_F_TZSQEAIB_V_DOGBFDTH',
    value=0,
    field=NV_R_PYQJLPSK_F_TZSQEAIB
)
NV_R_PYQJLPSK_F_TZSQEAIB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_PYQJLPSK_F_TZSQEAIB_V_PIUJQBKV',
    value=1,
    field=NV_R_PYQJLPSK_F_TZSQEAIB
)

NV_R_PYQJLPSK_F_WQYWMRVP = FieldMetadata(
    name='NV_R_PYQJLPSK_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_PYQJLPSK
)

NV_R_PYQJLPSK_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_PYQJLPSK_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_PYQJLPSK_F_WQYWMRVP
)
NV_R_PYQJLPSK_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_PYQJLPSK_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_PYQJLPSK_F_WQYWMRVP
)

NV_R_PYQJLPSK_F_PUPDHQBL = FieldMetadata(
    name='NV_R_PYQJLPSK_F_PUPDHQBL',
    msb=2,
    lsb=2,
    register=NV_R_PYQJLPSK
)

NV_R_PYQJLPSK_F_PUPDHQBL_V_DOGBFDTH = ValueMetadata(
    name='NV_R_PYQJLPSK_F_PUPDHQBL_V_DOGBFDTH',
    value=0,
    field=NV_R_PYQJLPSK_F_PUPDHQBL
)
NV_R_PYQJLPSK_F_PUPDHQBL_V_PIUJQBKV = ValueMetadata(
    name='NV_R_PYQJLPSK_F_PUPDHQBL_V_PIUJQBKV',
    value=1,
    field=NV_R_PYQJLPSK_F_PUPDHQBL
)

NV_R_PYQJLPSK_F_MLEOLJIS = FieldMetadata(
    name='NV_R_PYQJLPSK_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_PYQJLPSK
)

NV_R_PYQJLPSK_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_PYQJLPSK_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_PYQJLPSK_F_MLEOLJIS
)
NV_R_PYQJLPSK_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_PYQJLPSK_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_PYQJLPSK_F_MLEOLJIS
)

NV_R_LVNJASUC = RegisterMetadata(
    name='NV_R_LVNJASUC',
    address=0x8f8,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_LVNJASUC_F_CTBCFRUB = FieldMetadata(
    name='NV_R_LVNJASUC_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_LVNJASUC
)

NV_R_LVNJASUC_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LVNJASUC_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_LVNJASUC_F_CTBCFRUB
)

NV_R_LVNJASUC_F_QVYAARWP = FieldMetadata(
    name='NV_R_LVNJASUC_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_LVNJASUC
)

NV_R_LVNJASUC_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LVNJASUC_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_LVNJASUC_F_QVYAARWP
)

NV_R_EGCNXFWH = RegisterMetadata(
    name='NV_R_EGCNXFWH',
    address=0xb20,
    zero_based=True
)

NV_R_EGCNXFWH_F_ZIGJNGSD = FieldMetadata(
    name='NV_R_EGCNXFWH_F_ZIGJNGSD',
    msb=22,
    lsb=0,
    register=NV_R_EGCNXFWH
)

NV_R_EGCNXFWH_F_SEXGFUSZ = FieldMetadata(
    name='NV_R_EGCNXFWH_F_SEXGFUSZ',
    msb=21,
    lsb=20,
    register=NV_R_EGCNXFWH
)

NV_R_EGCNXFWH_F_GNCFTOVU = FieldMetadata(
    name='NV_R_EGCNXFWH_F_GNCFTOVU',
    msb=2,
    lsb=0,
    register=NV_R_EGCNXFWH
)

NV_R_EGCNXFWH_F_WDHNBQNK = FieldMetadata(
    name='NV_R_EGCNXFWH_F_WDHNBQNK',
    msb=19,
    lsb=8,
    register=NV_R_EGCNXFWH
)

NV_R_EGCNXFWH_F_HMBHUSZP = FieldMetadata(
    name='NV_R_EGCNXFWH_F_HMBHUSZP',
    msb=22,
    lsb=22,
    register=NV_R_EGCNXFWH
)

NV_R_EGCNXFWH_F_FEERLBNG = FieldMetadata(
    name='NV_R_EGCNXFWH_F_FEERLBNG',
    msb=7,
    lsb=3,
    register=NV_R_EGCNXFWH
)

NV_R_EGCNXFWH_F_EXMOKFEV = FieldMetadata(
    name='NV_R_EGCNXFWH_F_EXMOKFEV',
    msb=7,
    lsb=6,
    register=NV_R_EGCNXFWH
)

NV_R_JPQUZGIF = RegisterMetadata(
    name='NV_R_JPQUZGIF',
    address=0xafc,
    zero_based=True
)

NV_R_JPQUZGIF_F_OJYQBJXU = FieldMetadata(
    name='NV_R_JPQUZGIF_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_SPGCMIVK = FieldMetadata(
    name='NV_R_JPQUZGIF_F_SPGCMIVK',
    msb=21,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_CMIRIWSV = FieldMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV',
    msb=29,
    lsb=22,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_CMIRIWSV_V_IZNDWFDP = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_IZNDWFDP',
    value=8,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_JSAHBBTM = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_JSAHBBTM',
    value=9,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_SEBMZEME = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_SEBMZEME',
    value=10,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_DTQCFPNA = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_DTQCFPNA',
    value=11,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_KYVQOIDU = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_KYVQOIDU',
    value=0,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_GHVVYIOF = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_GHVVYIOF',
    value=1,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_JGXXVPER = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_JGXXVPER',
    value=2,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_OKQHAWBB = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_OKQHAWBB',
    value=3,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_IGVRDHSA = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_IGVRDHSA',
    value=4,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_BJJBUDSL = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_BJJBUDSL',
    value=5,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_NRGNBBPQ = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_NRGNBBPQ',
    value=6,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_EKNISTBR = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_EKNISTBR',
    value=7,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_PGSYHHVE = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_PGSYHHVE',
    value=13,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_UXJGURAX = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_UXJGURAX',
    value=12,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_QWWBEQMS = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_QWWBEQMS',
    value=0,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_NBBWYRIG = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_NBBWYRIG',
    value=1,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_SWVVTNLK = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_SWVVTNLK',
    value=2,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_OTSWZFCR = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_OTSWZFCR',
    value=3,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_CSRKWDGO = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_CSRKWDGO',
    value=39,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_ORGICIHS = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_ORGICIHS',
    value=40,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_EYIXRFGX = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_EYIXRFGX',
    value=35,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_KFPOQASI = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_KFPOQASI',
    value=34,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_WPWBKHJE = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_WPWBKHJE',
    value=37,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_EEUGWZSG = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_EEUGWZSG',
    value=0,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_QUUGDQPL = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_QUUGDQPL',
    value=1,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_LKELNZMU = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_LKELNZMU',
    value=36,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_RSFATZWE = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_RSFATZWE',
    value=41,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_VGDLATGR = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_VGDLATGR',
    value=18,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_AWDMJNRQ = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_AWDMJNRQ',
    value=19,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_GDTZIIUX = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_GDTZIIUX',
    value=20,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_RGDBBIRS = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_RGDBBIRS',
    value=21,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_PYYFFBZF = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_PYYFFBZF',
    value=22,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_ZDAXSZOC = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_ZDAXSZOC',
    value=23,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_ZGKOZVLG = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_ZGKOZVLG',
    value=24,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_LXFZIIVC = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_LXFZIIVC',
    value=25,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_LIYVCEVC = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_LIYVCEVC',
    value=10,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_TMZYYQRE = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_TMZYYQRE',
    value=11,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_LITMRWHR = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_LITMRWHR',
    value=12,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_SUXZYCLN = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_SUXZYCLN',
    value=13,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_JKGEPTXY = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_JKGEPTXY',
    value=14,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_BIJQQCBV = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_BIJQQCBV',
    value=15,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_RCXNOXVH = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_RCXNOXVH',
    value=16,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_UPMZYBHK = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_UPMZYBHK',
    value=17,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_JKIFIDZS = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_JKIFIDZS',
    value=2,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_CWMIEYYA = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_CWMIEYYA',
    value=3,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_PTVPYSBH = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_PTVPYSBH',
    value=4,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_IGQUZTXI = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_IGQUZTXI',
    value=5,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_IPJEHZXT = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_IPJEHZXT',
    value=6,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_IBHISRTF = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_IBHISRTF',
    value=7,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_AHLHOHQY = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_AHLHOHQY',
    value=8,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_MAPDVIBO = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_MAPDVIBO',
    value=9,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_JFLKUSVU = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_JFLKUSVU',
    value=33,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_XZPNLKQC = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_XZPNLKQC',
    value=32,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_GZQSWWSJ = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_GZQSWWSJ',
    value=31,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_KHINLTNU = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_KHINLTNU',
    value=26,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_AYPUIJOI = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_AYPUIJOI',
    value=27,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_MDSIVZHL = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_MDSIVZHL',
    value=43,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_CFWZRRWE = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_CFWZRRWE',
    value=28,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_JSOPIVBC = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_JSOPIVBC',
    value=38,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_ZSEHHQVS = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_ZSEHHQVS',
    value=30,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_OOWFZASJ = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_OOWFZASJ',
    value=42,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)
NV_R_JPQUZGIF_F_CMIRIWSV_V_USZOAVIE = ValueMetadata(
    name='NV_R_JPQUZGIF_F_CMIRIWSV_V_USZOAVIE',
    value=29,
    field=NV_R_JPQUZGIF_F_CMIRIWSV
)

NV_R_JPQUZGIF_F_NVKIPOLT = FieldMetadata(
    name='NV_R_JPQUZGIF_F_NVKIPOLT',
    msb=11,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_ZOUDCMGL = FieldMetadata(
    name='NV_R_JPQUZGIF_F_ZOUDCMGL',
    msb=11,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_NOARSXQM = FieldMetadata(
    name='NV_R_JPQUZGIF_F_NOARSXQM',
    msb=11,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_OEUAITQM = FieldMetadata(
    name='NV_R_JPQUZGIF_F_OEUAITQM',
    msb=11,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_RBGFPZWZ = FieldMetadata(
    name='NV_R_JPQUZGIF_F_RBGFPZWZ',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_IMJFHBST = FieldMetadata(
    name='NV_R_JPQUZGIF_F_IMJFHBST',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_HYLTFYZK = FieldMetadata(
    name='NV_R_JPQUZGIF_F_HYLTFYZK',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_HBTOFJFB = FieldMetadata(
    name='NV_R_JPQUZGIF_F_HBTOFJFB',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_HHGTSGOX = FieldMetadata(
    name='NV_R_JPQUZGIF_F_HHGTSGOX',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_NBECFVTR = FieldMetadata(
    name='NV_R_JPQUZGIF_F_NBECFVTR',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_CPXQSYZB = FieldMetadata(
    name='NV_R_JPQUZGIF_F_CPXQSYZB',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_MBJISBNZ = FieldMetadata(
    name='NV_R_JPQUZGIF_F_MBJISBNZ',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_PDZWCDXW = FieldMetadata(
    name='NV_R_JPQUZGIF_F_PDZWCDXW',
    msb=9,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_OMCFLBAL = FieldMetadata(
    name='NV_R_JPQUZGIF_F_OMCFLBAL',
    msb=9,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_NKLXJYMD = FieldMetadata(
    name='NV_R_JPQUZGIF_F_NKLXJYMD',
    msb=9,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_IHQOCTPV = FieldMetadata(
    name='NV_R_JPQUZGIF_F_IHQOCTPV',
    msb=9,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_YJVWXNNE = FieldMetadata(
    name='NV_R_JPQUZGIF_F_YJVWXNNE',
    msb=9,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_VHLUWKFL = FieldMetadata(
    name='NV_R_JPQUZGIF_F_VHLUWKFL',
    msb=9,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_ZJAXCAQG = FieldMetadata(
    name='NV_R_JPQUZGIF_F_ZJAXCAQG',
    msb=8,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_OBEWYHPX = FieldMetadata(
    name='NV_R_JPQUZGIF_F_OBEWYHPX',
    msb=8,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_IRSEKWPQ = FieldMetadata(
    name='NV_R_JPQUZGIF_F_IRSEKWPQ',
    msb=16,
    lsb=9,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_VCXMQJEV = FieldMetadata(
    name='NV_R_JPQUZGIF_F_VCXMQJEV',
    msb=11,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_JXKCEYNM = FieldMetadata(
    name='NV_R_JPQUZGIF_F_JXKCEYNM',
    msb=8,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_DURSWDLP = FieldMetadata(
    name='NV_R_JPQUZGIF_F_DURSWDLP',
    msb=8,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_BJBZERKJ = FieldMetadata(
    name='NV_R_JPQUZGIF_F_BJBZERKJ',
    msb=8,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_MVPRIXOL = FieldMetadata(
    name='NV_R_JPQUZGIF_F_MVPRIXOL',
    msb=15,
    lsb=10,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_TDCDFVOD = FieldMetadata(
    name='NV_R_JPQUZGIF_F_TDCDFVOD',
    msb=9,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_XOCPWSRT = FieldMetadata(
    name='NV_R_JPQUZGIF_F_XOCPWSRT',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_INSVKLRN = FieldMetadata(
    name='NV_R_JPQUZGIF_F_INSVKLRN',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_NQFZWAHO = FieldMetadata(
    name='NV_R_JPQUZGIF_F_NQFZWAHO',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_DSQEGPFA = FieldMetadata(
    name='NV_R_JPQUZGIF_F_DSQEGPFA',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_IKVTHDBZ = FieldMetadata(
    name='NV_R_JPQUZGIF_F_IKVTHDBZ',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_CIWVBJVE = FieldMetadata(
    name='NV_R_JPQUZGIF_F_CIWVBJVE',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_LWLGMGAQ = FieldMetadata(
    name='NV_R_JPQUZGIF_F_LWLGMGAQ',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_OWIZHHHO = FieldMetadata(
    name='NV_R_JPQUZGIF_F_OWIZHHHO',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_MBHCYZPH = FieldMetadata(
    name='NV_R_JPQUZGIF_F_MBHCYZPH',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_YVRYVWIH = FieldMetadata(
    name='NV_R_JPQUZGIF_F_YVRYVWIH',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_UFKZEDCN = FieldMetadata(
    name='NV_R_JPQUZGIF_F_UFKZEDCN',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_QPRFLYFC = FieldMetadata(
    name='NV_R_JPQUZGIF_F_QPRFLYFC',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_KRYCNSZJ = FieldMetadata(
    name='NV_R_JPQUZGIF_F_KRYCNSZJ',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_GDDJPIHT = FieldMetadata(
    name='NV_R_JPQUZGIF_F_GDDJPIHT',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_PYWRHSKK = FieldMetadata(
    name='NV_R_JPQUZGIF_F_PYWRHSKK',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_SMFYIMNS = FieldMetadata(
    name='NV_R_JPQUZGIF_F_SMFYIMNS',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_HDYAZFOA = FieldMetadata(
    name='NV_R_JPQUZGIF_F_HDYAZFOA',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_UHBEOVYM = FieldMetadata(
    name='NV_R_JPQUZGIF_F_UHBEOVYM',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_ZICNZMHT = FieldMetadata(
    name='NV_R_JPQUZGIF_F_ZICNZMHT',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_ZJCFMQNK = FieldMetadata(
    name='NV_R_JPQUZGIF_F_ZJCFMQNK',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_DEBPYEVL = FieldMetadata(
    name='NV_R_JPQUZGIF_F_DEBPYEVL',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_AHTIHXXI = FieldMetadata(
    name='NV_R_JPQUZGIF_F_AHTIHXXI',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_RHTNASDF = FieldMetadata(
    name='NV_R_JPQUZGIF_F_RHTNASDF',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_DVLAWAYM = FieldMetadata(
    name='NV_R_JPQUZGIF_F_DVLAWAYM',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_IHLFYVUA = FieldMetadata(
    name='NV_R_JPQUZGIF_F_IHLFYVUA',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_VJZXXNRN = FieldMetadata(
    name='NV_R_JPQUZGIF_F_VJZXXNRN',
    msb=7,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_WPOKVHWB = FieldMetadata(
    name='NV_R_JPQUZGIF_F_WPOKVHWB',
    msb=11,
    lsb=4,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_QAWEVVQS = FieldMetadata(
    name='NV_R_JPQUZGIF_F_QAWEVVQS',
    msb=3,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_ECKYQVQO = FieldMetadata(
    name='NV_R_JPQUZGIF_F_ECKYQVQO',
    msb=11,
    lsb=4,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_AUKLBKOY = FieldMetadata(
    name='NV_R_JPQUZGIF_F_AUKLBKOY',
    msb=3,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_AZTHXCVL = FieldMetadata(
    name='NV_R_JPQUZGIF_F_AZTHXCVL',
    msb=11,
    lsb=4,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_FWIQTCRQ = FieldMetadata(
    name='NV_R_JPQUZGIF_F_FWIQTCRQ',
    msb=3,
    lsb=0,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_OFGCIMFI = FieldMetadata(
    name='NV_R_JPQUZGIF_F_OFGCIMFI',
    msb=31,
    lsb=30,
    register=NV_R_JPQUZGIF
)

NV_R_JPQUZGIF_F_OFGCIMFI_V_FQNJWQHN = ValueMetadata(
    name='NV_R_JPQUZGIF_F_OFGCIMFI_V_FQNJWQHN',
    value=2,
    field=NV_R_JPQUZGIF_F_OFGCIMFI
)
NV_R_JPQUZGIF_F_OFGCIMFI_V_CMVOBGUP = ValueMetadata(
    name='NV_R_JPQUZGIF_F_OFGCIMFI_V_CMVOBGUP',
    value=0,
    field=NV_R_JPQUZGIF_F_OFGCIMFI
)
NV_R_JPQUZGIF_F_OFGCIMFI_V_TVGHFSAG = ValueMetadata(
    name='NV_R_JPQUZGIF_F_OFGCIMFI_V_TVGHFSAG',
    value=1,
    field=NV_R_JPQUZGIF_F_OFGCIMFI
)

NV_R_PDBACWYG = RegisterMetadata(
    name='NV_R_PDBACWYG',
    address=0xaf4,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_PDBACWYG_F_CTBCFRUB = FieldMetadata(
    name='NV_R_PDBACWYG_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_PDBACWYG
)

NV_R_PDBACWYG_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_PDBACWYG_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_PDBACWYG_F_CTBCFRUB
)

NV_R_PDBACWYG_F_QVYAARWP = FieldMetadata(
    name='NV_R_PDBACWYG_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_PDBACWYG
)

NV_R_PDBACWYG_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_PDBACWYG_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_PDBACWYG_F_QVYAARWP
)

NV_R_KYKXQQJX = RegisterMetadata(
    name='NV_R_KYKXQQJX',
    address=0xaf0,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_KYKXQQJX_F_KWDQPURW = FieldMetadata(
    name='NV_R_KYKXQQJX_F_KWDQPURW',
    msb=5,
    lsb=5,
    register=NV_R_KYKXQQJX
)

NV_R_KYKXQQJX_F_KWDQPURW_V_DOGBFDTH = ValueMetadata(
    name='NV_R_KYKXQQJX_F_KWDQPURW_V_DOGBFDTH',
    value=0,
    field=NV_R_KYKXQQJX_F_KWDQPURW
)
NV_R_KYKXQQJX_F_KWDQPURW_V_PIUJQBKV = ValueMetadata(
    name='NV_R_KYKXQQJX_F_KWDQPURW_V_PIUJQBKV',
    value=1,
    field=NV_R_KYKXQQJX_F_KWDQPURW
)

NV_R_KYKXQQJX_F_ELAIAXBC = FieldMetadata(
    name='NV_R_KYKXQQJX_F_ELAIAXBC',
    msb=1,
    lsb=1,
    register=NV_R_KYKXQQJX
)

NV_R_KYKXQQJX_F_ELAIAXBC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_KYKXQQJX_F_ELAIAXBC_V_DOGBFDTH',
    value=0,
    field=NV_R_KYKXQQJX_F_ELAIAXBC
)
NV_R_KYKXQQJX_F_ELAIAXBC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_KYKXQQJX_F_ELAIAXBC_V_PIUJQBKV',
    value=1,
    field=NV_R_KYKXQQJX_F_ELAIAXBC
)

NV_R_KYKXQQJX_F_OJXVVOEC = FieldMetadata(
    name='NV_R_KYKXQQJX_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_KYKXQQJX
)

NV_R_KYKXQQJX_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_KYKXQQJX_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_KYKXQQJX_F_OJXVVOEC
)
NV_R_KYKXQQJX_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_KYKXQQJX_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_KYKXQQJX_F_OJXVVOEC
)

NV_R_KYKXQQJX_F_PBAFTNZG = FieldMetadata(
    name='NV_R_KYKXQQJX_F_PBAFTNZG',
    msb=3,
    lsb=3,
    register=NV_R_KYKXQQJX
)

NV_R_KYKXQQJX_F_PBAFTNZG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_KYKXQQJX_F_PBAFTNZG_V_DOGBFDTH',
    value=0,
    field=NV_R_KYKXQQJX_F_PBAFTNZG
)
NV_R_KYKXQQJX_F_PBAFTNZG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_KYKXQQJX_F_PBAFTNZG_V_PIUJQBKV',
    value=1,
    field=NV_R_KYKXQQJX_F_PBAFTNZG
)

NV_R_KYKXQQJX_F_VEOHPHHB = FieldMetadata(
    name='NV_R_KYKXQQJX_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_KYKXQQJX
)

NV_R_KYKXQQJX_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_KYKXQQJX_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_KYKXQQJX_F_VEOHPHHB
)
NV_R_KYKXQQJX_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_KYKXQQJX_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_KYKXQQJX_F_VEOHPHHB
)

NV_R_KYKXQQJX_F_KVYOKDXN = FieldMetadata(
    name='NV_R_KYKXQQJX_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_KYKXQQJX
)

NV_R_KYKXQQJX_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_KYKXQQJX_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_KYKXQQJX_F_KVYOKDXN
)
NV_R_KYKXQQJX_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_KYKXQQJX_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_KYKXQQJX_F_KVYOKDXN
)
NV_R_KYKXQQJX_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_KYKXQQJX_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_KYKXQQJX_F_KVYOKDXN
)

NV_R_KYKXQQJX_F_IOJLTETI = FieldMetadata(
    name='NV_R_KYKXQQJX_F_IOJLTETI',
    msb=4,
    lsb=4,
    register=NV_R_KYKXQQJX
)

NV_R_KYKXQQJX_F_IOJLTETI_V_DOGBFDTH = ValueMetadata(
    name='NV_R_KYKXQQJX_F_IOJLTETI_V_DOGBFDTH',
    value=0,
    field=NV_R_KYKXQQJX_F_IOJLTETI
)
NV_R_KYKXQQJX_F_IOJLTETI_V_PIUJQBKV = ValueMetadata(
    name='NV_R_KYKXQQJX_F_IOJLTETI_V_PIUJQBKV',
    value=1,
    field=NV_R_KYKXQQJX_F_IOJLTETI
)

NV_R_KYKXQQJX_F_TZSQEAIB = FieldMetadata(
    name='NV_R_KYKXQQJX_F_TZSQEAIB',
    msb=0,
    lsb=0,
    register=NV_R_KYKXQQJX
)

NV_R_KYKXQQJX_F_TZSQEAIB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_KYKXQQJX_F_TZSQEAIB_V_DOGBFDTH',
    value=0,
    field=NV_R_KYKXQQJX_F_TZSQEAIB
)
NV_R_KYKXQQJX_F_TZSQEAIB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_KYKXQQJX_F_TZSQEAIB_V_PIUJQBKV',
    value=1,
    field=NV_R_KYKXQQJX_F_TZSQEAIB
)

NV_R_KYKXQQJX_F_WQYWMRVP = FieldMetadata(
    name='NV_R_KYKXQQJX_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_KYKXQQJX
)

NV_R_KYKXQQJX_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_KYKXQQJX_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_KYKXQQJX_F_WQYWMRVP
)
NV_R_KYKXQQJX_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_KYKXQQJX_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_KYKXQQJX_F_WQYWMRVP
)

NV_R_KYKXQQJX_F_PUPDHQBL = FieldMetadata(
    name='NV_R_KYKXQQJX_F_PUPDHQBL',
    msb=2,
    lsb=2,
    register=NV_R_KYKXQQJX
)

NV_R_KYKXQQJX_F_PUPDHQBL_V_DOGBFDTH = ValueMetadata(
    name='NV_R_KYKXQQJX_F_PUPDHQBL_V_DOGBFDTH',
    value=0,
    field=NV_R_KYKXQQJX_F_PUPDHQBL
)
NV_R_KYKXQQJX_F_PUPDHQBL_V_PIUJQBKV = ValueMetadata(
    name='NV_R_KYKXQQJX_F_PUPDHQBL_V_PIUJQBKV',
    value=1,
    field=NV_R_KYKXQQJX_F_PUPDHQBL
)

NV_R_KYKXQQJX_F_MLEOLJIS = FieldMetadata(
    name='NV_R_KYKXQQJX_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_KYKXQQJX
)

NV_R_KYKXQQJX_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_KYKXQQJX_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_KYKXQQJX_F_MLEOLJIS
)
NV_R_KYKXQQJX_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_KYKXQQJX_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_KYKXQQJX_F_MLEOLJIS
)

NV_R_ILSVNUDL = RegisterMetadata(
    name='NV_R_ILSVNUDL',
    address=0xaf8,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_ILSVNUDL_F_CTBCFRUB = FieldMetadata(
    name='NV_R_ILSVNUDL_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_ILSVNUDL
)

NV_R_ILSVNUDL_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_ILSVNUDL_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_ILSVNUDL_F_CTBCFRUB
)

NV_R_ILSVNUDL_F_QVYAARWP = FieldMetadata(
    name='NV_R_ILSVNUDL_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_ILSVNUDL
)

NV_R_ILSVNUDL_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_ILSVNUDL_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_ILSVNUDL_F_QVYAARWP
)

NV_R_KKJEEHZK = RegisterMetadata(
    name='NV_R_KKJEEHZK',
    address=0xd20,
    zero_based=True
)

NV_R_KKJEEHZK_F_ZIGJNGSD = FieldMetadata(
    name='NV_R_KKJEEHZK_F_ZIGJNGSD',
    msb=22,
    lsb=0,
    register=NV_R_KKJEEHZK
)

NV_R_KKJEEHZK_F_SEXGFUSZ = FieldMetadata(
    name='NV_R_KKJEEHZK_F_SEXGFUSZ',
    msb=21,
    lsb=20,
    register=NV_R_KKJEEHZK
)

NV_R_KKJEEHZK_F_GNCFTOVU = FieldMetadata(
    name='NV_R_KKJEEHZK_F_GNCFTOVU',
    msb=2,
    lsb=0,
    register=NV_R_KKJEEHZK
)

NV_R_KKJEEHZK_F_WDHNBQNK = FieldMetadata(
    name='NV_R_KKJEEHZK_F_WDHNBQNK',
    msb=19,
    lsb=8,
    register=NV_R_KKJEEHZK
)

NV_R_KKJEEHZK_F_HMBHUSZP = FieldMetadata(
    name='NV_R_KKJEEHZK_F_HMBHUSZP',
    msb=22,
    lsb=22,
    register=NV_R_KKJEEHZK
)

NV_R_KKJEEHZK_F_FEERLBNG = FieldMetadata(
    name='NV_R_KKJEEHZK_F_FEERLBNG',
    msb=7,
    lsb=3,
    register=NV_R_KKJEEHZK
)

NV_R_KKJEEHZK_F_EXMOKFEV = FieldMetadata(
    name='NV_R_KKJEEHZK_F_EXMOKFEV',
    msb=7,
    lsb=6,
    register=NV_R_KKJEEHZK
)

NV_R_JUVPUKFZ = RegisterMetadata(
    name='NV_R_JUVPUKFZ',
    address=0xcfc,
    zero_based=True
)

NV_R_JUVPUKFZ_F_OJYQBJXU = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_SPGCMIVK = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_SPGCMIVK',
    msb=21,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_CMIRIWSV = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV',
    msb=29,
    lsb=22,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_CMIRIWSV_V_IZNDWFDP = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_IZNDWFDP',
    value=8,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_JSAHBBTM = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_JSAHBBTM',
    value=9,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_SEBMZEME = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_SEBMZEME',
    value=10,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_DTQCFPNA = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_DTQCFPNA',
    value=11,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_KYVQOIDU = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_KYVQOIDU',
    value=0,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_GHVVYIOF = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_GHVVYIOF',
    value=1,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_JGXXVPER = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_JGXXVPER',
    value=2,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_OKQHAWBB = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_OKQHAWBB',
    value=3,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_IGVRDHSA = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_IGVRDHSA',
    value=4,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_BJJBUDSL = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_BJJBUDSL',
    value=5,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_NRGNBBPQ = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_NRGNBBPQ',
    value=6,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_EKNISTBR = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_EKNISTBR',
    value=7,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_PGSYHHVE = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_PGSYHHVE',
    value=13,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_UXJGURAX = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_UXJGURAX',
    value=12,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_QWWBEQMS = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_QWWBEQMS',
    value=0,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_NBBWYRIG = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_NBBWYRIG',
    value=1,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_SWVVTNLK = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_SWVVTNLK',
    value=2,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_OTSWZFCR = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_OTSWZFCR',
    value=3,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_CSRKWDGO = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_CSRKWDGO',
    value=39,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_ORGICIHS = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_ORGICIHS',
    value=40,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_EYIXRFGX = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_EYIXRFGX',
    value=35,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_KFPOQASI = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_KFPOQASI',
    value=34,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_WPWBKHJE = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_WPWBKHJE',
    value=37,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_EEUGWZSG = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_EEUGWZSG',
    value=0,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_QUUGDQPL = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_QUUGDQPL',
    value=1,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_LKELNZMU = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_LKELNZMU',
    value=36,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_RSFATZWE = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_RSFATZWE',
    value=41,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_VGDLATGR = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_VGDLATGR',
    value=18,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_AWDMJNRQ = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_AWDMJNRQ',
    value=19,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_GDTZIIUX = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_GDTZIIUX',
    value=20,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_RGDBBIRS = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_RGDBBIRS',
    value=21,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_PYYFFBZF = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_PYYFFBZF',
    value=22,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_ZDAXSZOC = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_ZDAXSZOC',
    value=23,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_ZGKOZVLG = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_ZGKOZVLG',
    value=24,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_LXFZIIVC = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_LXFZIIVC',
    value=25,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_LIYVCEVC = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_LIYVCEVC',
    value=10,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_TMZYYQRE = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_TMZYYQRE',
    value=11,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_LITMRWHR = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_LITMRWHR',
    value=12,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_SUXZYCLN = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_SUXZYCLN',
    value=13,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_JKGEPTXY = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_JKGEPTXY',
    value=14,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_BIJQQCBV = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_BIJQQCBV',
    value=15,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_RCXNOXVH = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_RCXNOXVH',
    value=16,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_UPMZYBHK = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_UPMZYBHK',
    value=17,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_JKIFIDZS = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_JKIFIDZS',
    value=2,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_CWMIEYYA = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_CWMIEYYA',
    value=3,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_PTVPYSBH = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_PTVPYSBH',
    value=4,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_IGQUZTXI = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_IGQUZTXI',
    value=5,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_IPJEHZXT = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_IPJEHZXT',
    value=6,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_IBHISRTF = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_IBHISRTF',
    value=7,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_AHLHOHQY = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_AHLHOHQY',
    value=8,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_MAPDVIBO = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_MAPDVIBO',
    value=9,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_JFLKUSVU = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_JFLKUSVU',
    value=33,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_XZPNLKQC = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_XZPNLKQC',
    value=32,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_GZQSWWSJ = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_GZQSWWSJ',
    value=31,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_KHINLTNU = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_KHINLTNU',
    value=26,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_AYPUIJOI = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_AYPUIJOI',
    value=27,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_MDSIVZHL = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_MDSIVZHL',
    value=43,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_CFWZRRWE = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_CFWZRRWE',
    value=28,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_JSOPIVBC = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_JSOPIVBC',
    value=38,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_ZSEHHQVS = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_ZSEHHQVS',
    value=30,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_OOWFZASJ = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_OOWFZASJ',
    value=42,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)
NV_R_JUVPUKFZ_F_CMIRIWSV_V_USZOAVIE = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_CMIRIWSV_V_USZOAVIE',
    value=29,
    field=NV_R_JUVPUKFZ_F_CMIRIWSV
)

NV_R_JUVPUKFZ_F_NVKIPOLT = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_NVKIPOLT',
    msb=11,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_ZOUDCMGL = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_ZOUDCMGL',
    msb=11,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_NOARSXQM = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_NOARSXQM',
    msb=11,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_OEUAITQM = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_OEUAITQM',
    msb=11,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_RBGFPZWZ = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_RBGFPZWZ',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_IMJFHBST = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_IMJFHBST',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_HYLTFYZK = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_HYLTFYZK',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_HBTOFJFB = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_HBTOFJFB',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_HHGTSGOX = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_HHGTSGOX',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_NBECFVTR = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_NBECFVTR',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_CPXQSYZB = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_CPXQSYZB',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_MBJISBNZ = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_MBJISBNZ',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_PDZWCDXW = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_PDZWCDXW',
    msb=9,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_OMCFLBAL = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_OMCFLBAL',
    msb=9,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_NKLXJYMD = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_NKLXJYMD',
    msb=9,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_IHQOCTPV = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_IHQOCTPV',
    msb=9,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_YJVWXNNE = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_YJVWXNNE',
    msb=9,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_VHLUWKFL = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_VHLUWKFL',
    msb=9,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_ZJAXCAQG = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_ZJAXCAQG',
    msb=8,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_OBEWYHPX = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_OBEWYHPX',
    msb=8,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_IRSEKWPQ = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_IRSEKWPQ',
    msb=16,
    lsb=9,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_VCXMQJEV = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_VCXMQJEV',
    msb=11,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_JXKCEYNM = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_JXKCEYNM',
    msb=8,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_DURSWDLP = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_DURSWDLP',
    msb=8,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_BJBZERKJ = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_BJBZERKJ',
    msb=8,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_MVPRIXOL = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_MVPRIXOL',
    msb=15,
    lsb=10,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_TDCDFVOD = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_TDCDFVOD',
    msb=9,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_XOCPWSRT = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_XOCPWSRT',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_INSVKLRN = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_INSVKLRN',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_NQFZWAHO = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_NQFZWAHO',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_DSQEGPFA = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_DSQEGPFA',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_IKVTHDBZ = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_IKVTHDBZ',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_CIWVBJVE = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_CIWVBJVE',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_LWLGMGAQ = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_LWLGMGAQ',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_OWIZHHHO = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_OWIZHHHO',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_MBHCYZPH = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_MBHCYZPH',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_YVRYVWIH = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_YVRYVWIH',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_UFKZEDCN = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_UFKZEDCN',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_QPRFLYFC = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_QPRFLYFC',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_KRYCNSZJ = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_KRYCNSZJ',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_GDDJPIHT = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_GDDJPIHT',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_PYWRHSKK = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_PYWRHSKK',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_SMFYIMNS = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_SMFYIMNS',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_HDYAZFOA = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_HDYAZFOA',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_UHBEOVYM = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_UHBEOVYM',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_ZICNZMHT = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_ZICNZMHT',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_ZJCFMQNK = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_ZJCFMQNK',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_DEBPYEVL = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_DEBPYEVL',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_AHTIHXXI = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_AHTIHXXI',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_RHTNASDF = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_RHTNASDF',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_DVLAWAYM = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_DVLAWAYM',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_IHLFYVUA = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_IHLFYVUA',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_VJZXXNRN = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_VJZXXNRN',
    msb=7,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_WPOKVHWB = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_WPOKVHWB',
    msb=11,
    lsb=4,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_QAWEVVQS = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_QAWEVVQS',
    msb=3,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_ECKYQVQO = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_ECKYQVQO',
    msb=11,
    lsb=4,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_AUKLBKOY = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_AUKLBKOY',
    msb=3,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_AZTHXCVL = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_AZTHXCVL',
    msb=11,
    lsb=4,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_FWIQTCRQ = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_FWIQTCRQ',
    msb=3,
    lsb=0,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_OFGCIMFI = FieldMetadata(
    name='NV_R_JUVPUKFZ_F_OFGCIMFI',
    msb=31,
    lsb=30,
    register=NV_R_JUVPUKFZ
)

NV_R_JUVPUKFZ_F_OFGCIMFI_V_FQNJWQHN = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_OFGCIMFI_V_FQNJWQHN',
    value=2,
    field=NV_R_JUVPUKFZ_F_OFGCIMFI
)
NV_R_JUVPUKFZ_F_OFGCIMFI_V_CMVOBGUP = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_OFGCIMFI_V_CMVOBGUP',
    value=0,
    field=NV_R_JUVPUKFZ_F_OFGCIMFI
)
NV_R_JUVPUKFZ_F_OFGCIMFI_V_TVGHFSAG = ValueMetadata(
    name='NV_R_JUVPUKFZ_F_OFGCIMFI_V_TVGHFSAG',
    value=1,
    field=NV_R_JUVPUKFZ_F_OFGCIMFI
)

NV_R_HBDACULW = RegisterMetadata(
    name='NV_R_HBDACULW',
    address=0xcf4,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_HBDACULW_F_CTBCFRUB = FieldMetadata(
    name='NV_R_HBDACULW_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_HBDACULW
)

NV_R_HBDACULW_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_HBDACULW_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_HBDACULW_F_CTBCFRUB
)

NV_R_HBDACULW_F_QVYAARWP = FieldMetadata(
    name='NV_R_HBDACULW_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_HBDACULW
)

NV_R_HBDACULW_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_HBDACULW_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_HBDACULW_F_QVYAARWP
)

NV_R_TURRIXCA = RegisterMetadata(
    name='NV_R_TURRIXCA',
    address=0xcf0,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_TURRIXCA_F_KWDQPURW = FieldMetadata(
    name='NV_R_TURRIXCA_F_KWDQPURW',
    msb=5,
    lsb=5,
    register=NV_R_TURRIXCA
)

NV_R_TURRIXCA_F_KWDQPURW_V_DOGBFDTH = ValueMetadata(
    name='NV_R_TURRIXCA_F_KWDQPURW_V_DOGBFDTH',
    value=0,
    field=NV_R_TURRIXCA_F_KWDQPURW
)
NV_R_TURRIXCA_F_KWDQPURW_V_PIUJQBKV = ValueMetadata(
    name='NV_R_TURRIXCA_F_KWDQPURW_V_PIUJQBKV',
    value=1,
    field=NV_R_TURRIXCA_F_KWDQPURW
)

NV_R_TURRIXCA_F_ELAIAXBC = FieldMetadata(
    name='NV_R_TURRIXCA_F_ELAIAXBC',
    msb=1,
    lsb=1,
    register=NV_R_TURRIXCA
)

NV_R_TURRIXCA_F_ELAIAXBC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_TURRIXCA_F_ELAIAXBC_V_DOGBFDTH',
    value=0,
    field=NV_R_TURRIXCA_F_ELAIAXBC
)
NV_R_TURRIXCA_F_ELAIAXBC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_TURRIXCA_F_ELAIAXBC_V_PIUJQBKV',
    value=1,
    field=NV_R_TURRIXCA_F_ELAIAXBC
)

NV_R_TURRIXCA_F_OJXVVOEC = FieldMetadata(
    name='NV_R_TURRIXCA_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_TURRIXCA
)

NV_R_TURRIXCA_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_TURRIXCA_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_TURRIXCA_F_OJXVVOEC
)
NV_R_TURRIXCA_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_TURRIXCA_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_TURRIXCA_F_OJXVVOEC
)

NV_R_TURRIXCA_F_PBAFTNZG = FieldMetadata(
    name='NV_R_TURRIXCA_F_PBAFTNZG',
    msb=3,
    lsb=3,
    register=NV_R_TURRIXCA
)

NV_R_TURRIXCA_F_PBAFTNZG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_TURRIXCA_F_PBAFTNZG_V_DOGBFDTH',
    value=0,
    field=NV_R_TURRIXCA_F_PBAFTNZG
)
NV_R_TURRIXCA_F_PBAFTNZG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_TURRIXCA_F_PBAFTNZG_V_PIUJQBKV',
    value=1,
    field=NV_R_TURRIXCA_F_PBAFTNZG
)

NV_R_TURRIXCA_F_VEOHPHHB = FieldMetadata(
    name='NV_R_TURRIXCA_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_TURRIXCA
)

NV_R_TURRIXCA_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_TURRIXCA_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_TURRIXCA_F_VEOHPHHB
)
NV_R_TURRIXCA_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_TURRIXCA_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_TURRIXCA_F_VEOHPHHB
)

NV_R_TURRIXCA_F_KVYOKDXN = FieldMetadata(
    name='NV_R_TURRIXCA_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_TURRIXCA
)

NV_R_TURRIXCA_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_TURRIXCA_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_TURRIXCA_F_KVYOKDXN
)
NV_R_TURRIXCA_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_TURRIXCA_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_TURRIXCA_F_KVYOKDXN
)
NV_R_TURRIXCA_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_TURRIXCA_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_TURRIXCA_F_KVYOKDXN
)

NV_R_TURRIXCA_F_IOJLTETI = FieldMetadata(
    name='NV_R_TURRIXCA_F_IOJLTETI',
    msb=4,
    lsb=4,
    register=NV_R_TURRIXCA
)

NV_R_TURRIXCA_F_IOJLTETI_V_DOGBFDTH = ValueMetadata(
    name='NV_R_TURRIXCA_F_IOJLTETI_V_DOGBFDTH',
    value=0,
    field=NV_R_TURRIXCA_F_IOJLTETI
)
NV_R_TURRIXCA_F_IOJLTETI_V_PIUJQBKV = ValueMetadata(
    name='NV_R_TURRIXCA_F_IOJLTETI_V_PIUJQBKV',
    value=1,
    field=NV_R_TURRIXCA_F_IOJLTETI
)

NV_R_TURRIXCA_F_TZSQEAIB = FieldMetadata(
    name='NV_R_TURRIXCA_F_TZSQEAIB',
    msb=0,
    lsb=0,
    register=NV_R_TURRIXCA
)

NV_R_TURRIXCA_F_TZSQEAIB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_TURRIXCA_F_TZSQEAIB_V_DOGBFDTH',
    value=0,
    field=NV_R_TURRIXCA_F_TZSQEAIB
)
NV_R_TURRIXCA_F_TZSQEAIB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_TURRIXCA_F_TZSQEAIB_V_PIUJQBKV',
    value=1,
    field=NV_R_TURRIXCA_F_TZSQEAIB
)

NV_R_TURRIXCA_F_WQYWMRVP = FieldMetadata(
    name='NV_R_TURRIXCA_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_TURRIXCA
)

NV_R_TURRIXCA_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_TURRIXCA_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_TURRIXCA_F_WQYWMRVP
)
NV_R_TURRIXCA_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_TURRIXCA_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_TURRIXCA_F_WQYWMRVP
)

NV_R_TURRIXCA_F_PUPDHQBL = FieldMetadata(
    name='NV_R_TURRIXCA_F_PUPDHQBL',
    msb=2,
    lsb=2,
    register=NV_R_TURRIXCA
)

NV_R_TURRIXCA_F_PUPDHQBL_V_DOGBFDTH = ValueMetadata(
    name='NV_R_TURRIXCA_F_PUPDHQBL_V_DOGBFDTH',
    value=0,
    field=NV_R_TURRIXCA_F_PUPDHQBL
)
NV_R_TURRIXCA_F_PUPDHQBL_V_PIUJQBKV = ValueMetadata(
    name='NV_R_TURRIXCA_F_PUPDHQBL_V_PIUJQBKV',
    value=1,
    field=NV_R_TURRIXCA_F_PUPDHQBL
)

NV_R_TURRIXCA_F_MLEOLJIS = FieldMetadata(
    name='NV_R_TURRIXCA_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_TURRIXCA
)

NV_R_TURRIXCA_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_TURRIXCA_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_TURRIXCA_F_MLEOLJIS
)
NV_R_TURRIXCA_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_TURRIXCA_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_TURRIXCA_F_MLEOLJIS
)

NV_R_RJBPWEFQ = RegisterMetadata(
    name='NV_R_RJBPWEFQ',
    address=0xcf8,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_RJBPWEFQ_F_CTBCFRUB = FieldMetadata(
    name='NV_R_RJBPWEFQ_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_RJBPWEFQ
)

NV_R_RJBPWEFQ_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_RJBPWEFQ_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_RJBPWEFQ_F_CTBCFRUB
)

NV_R_RJBPWEFQ_F_QVYAARWP = FieldMetadata(
    name='NV_R_RJBPWEFQ_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_RJBPWEFQ
)

NV_R_RJBPWEFQ_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_RJBPWEFQ_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_RJBPWEFQ_F_QVYAARWP
)

NV_R_DMHSYORT = RegisterMetadata(
    name='NV_R_DMHSYORT',
    address=0xf20,
    zero_based=True
)

NV_R_DMHSYORT_F_ZIGJNGSD = FieldMetadata(
    name='NV_R_DMHSYORT_F_ZIGJNGSD',
    msb=22,
    lsb=0,
    register=NV_R_DMHSYORT
)

NV_R_DMHSYORT_F_SEXGFUSZ = FieldMetadata(
    name='NV_R_DMHSYORT_F_SEXGFUSZ',
    msb=21,
    lsb=20,
    register=NV_R_DMHSYORT
)

NV_R_DMHSYORT_F_GNCFTOVU = FieldMetadata(
    name='NV_R_DMHSYORT_F_GNCFTOVU',
    msb=2,
    lsb=0,
    register=NV_R_DMHSYORT
)

NV_R_DMHSYORT_F_WDHNBQNK = FieldMetadata(
    name='NV_R_DMHSYORT_F_WDHNBQNK',
    msb=19,
    lsb=8,
    register=NV_R_DMHSYORT
)

NV_R_DMHSYORT_F_HMBHUSZP = FieldMetadata(
    name='NV_R_DMHSYORT_F_HMBHUSZP',
    msb=22,
    lsb=22,
    register=NV_R_DMHSYORT
)

NV_R_DMHSYORT_F_FEERLBNG = FieldMetadata(
    name='NV_R_DMHSYORT_F_FEERLBNG',
    msb=7,
    lsb=3,
    register=NV_R_DMHSYORT
)

NV_R_DMHSYORT_F_EXMOKFEV = FieldMetadata(
    name='NV_R_DMHSYORT_F_EXMOKFEV',
    msb=7,
    lsb=6,
    register=NV_R_DMHSYORT
)

NV_R_KLHYZSPH = RegisterMetadata(
    name='NV_R_KLHYZSPH',
    address=0xefc,
    zero_based=True
)

NV_R_KLHYZSPH_F_OJYQBJXU = FieldMetadata(
    name='NV_R_KLHYZSPH_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_SPGCMIVK = FieldMetadata(
    name='NV_R_KLHYZSPH_F_SPGCMIVK',
    msb=21,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_CMIRIWSV = FieldMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV',
    msb=29,
    lsb=22,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_CMIRIWSV_V_IZNDWFDP = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_IZNDWFDP',
    value=8,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_JSAHBBTM = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_JSAHBBTM',
    value=9,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_SEBMZEME = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_SEBMZEME',
    value=10,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_DTQCFPNA = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_DTQCFPNA',
    value=11,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_KYVQOIDU = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_KYVQOIDU',
    value=0,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_GHVVYIOF = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_GHVVYIOF',
    value=1,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_JGXXVPER = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_JGXXVPER',
    value=2,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_OKQHAWBB = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_OKQHAWBB',
    value=3,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_IGVRDHSA = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_IGVRDHSA',
    value=4,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_BJJBUDSL = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_BJJBUDSL',
    value=5,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_NRGNBBPQ = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_NRGNBBPQ',
    value=6,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_EKNISTBR = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_EKNISTBR',
    value=7,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_PGSYHHVE = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_PGSYHHVE',
    value=13,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_UXJGURAX = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_UXJGURAX',
    value=12,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_QWWBEQMS = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_QWWBEQMS',
    value=0,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_NBBWYRIG = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_NBBWYRIG',
    value=1,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_SWVVTNLK = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_SWVVTNLK',
    value=2,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_OTSWZFCR = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_OTSWZFCR',
    value=3,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_CSRKWDGO = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_CSRKWDGO',
    value=39,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_ORGICIHS = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_ORGICIHS',
    value=40,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_EYIXRFGX = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_EYIXRFGX',
    value=35,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_KFPOQASI = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_KFPOQASI',
    value=34,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_WPWBKHJE = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_WPWBKHJE',
    value=37,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_EEUGWZSG = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_EEUGWZSG',
    value=0,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_QUUGDQPL = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_QUUGDQPL',
    value=1,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_LKELNZMU = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_LKELNZMU',
    value=36,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_RSFATZWE = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_RSFATZWE',
    value=41,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_VGDLATGR = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_VGDLATGR',
    value=18,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_AWDMJNRQ = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_AWDMJNRQ',
    value=19,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_GDTZIIUX = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_GDTZIIUX',
    value=20,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_RGDBBIRS = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_RGDBBIRS',
    value=21,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_PYYFFBZF = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_PYYFFBZF',
    value=22,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_ZDAXSZOC = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_ZDAXSZOC',
    value=23,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_ZGKOZVLG = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_ZGKOZVLG',
    value=24,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_LXFZIIVC = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_LXFZIIVC',
    value=25,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_LIYVCEVC = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_LIYVCEVC',
    value=10,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_TMZYYQRE = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_TMZYYQRE',
    value=11,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_LITMRWHR = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_LITMRWHR',
    value=12,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_SUXZYCLN = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_SUXZYCLN',
    value=13,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_JKGEPTXY = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_JKGEPTXY',
    value=14,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_BIJQQCBV = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_BIJQQCBV',
    value=15,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_RCXNOXVH = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_RCXNOXVH',
    value=16,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_UPMZYBHK = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_UPMZYBHK',
    value=17,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_JKIFIDZS = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_JKIFIDZS',
    value=2,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_CWMIEYYA = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_CWMIEYYA',
    value=3,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_PTVPYSBH = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_PTVPYSBH',
    value=4,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_IGQUZTXI = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_IGQUZTXI',
    value=5,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_IPJEHZXT = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_IPJEHZXT',
    value=6,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_IBHISRTF = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_IBHISRTF',
    value=7,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_AHLHOHQY = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_AHLHOHQY',
    value=8,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_MAPDVIBO = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_MAPDVIBO',
    value=9,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_JFLKUSVU = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_JFLKUSVU',
    value=33,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_XZPNLKQC = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_XZPNLKQC',
    value=32,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_GZQSWWSJ = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_GZQSWWSJ',
    value=31,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_KHINLTNU = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_KHINLTNU',
    value=26,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_AYPUIJOI = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_AYPUIJOI',
    value=27,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_MDSIVZHL = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_MDSIVZHL',
    value=43,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_CFWZRRWE = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_CFWZRRWE',
    value=28,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_JSOPIVBC = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_JSOPIVBC',
    value=38,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_ZSEHHQVS = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_ZSEHHQVS',
    value=30,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_OOWFZASJ = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_OOWFZASJ',
    value=42,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)
NV_R_KLHYZSPH_F_CMIRIWSV_V_USZOAVIE = ValueMetadata(
    name='NV_R_KLHYZSPH_F_CMIRIWSV_V_USZOAVIE',
    value=29,
    field=NV_R_KLHYZSPH_F_CMIRIWSV
)

NV_R_KLHYZSPH_F_NVKIPOLT = FieldMetadata(
    name='NV_R_KLHYZSPH_F_NVKIPOLT',
    msb=11,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_ZOUDCMGL = FieldMetadata(
    name='NV_R_KLHYZSPH_F_ZOUDCMGL',
    msb=11,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_NOARSXQM = FieldMetadata(
    name='NV_R_KLHYZSPH_F_NOARSXQM',
    msb=11,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_OEUAITQM = FieldMetadata(
    name='NV_R_KLHYZSPH_F_OEUAITQM',
    msb=11,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_RBGFPZWZ = FieldMetadata(
    name='NV_R_KLHYZSPH_F_RBGFPZWZ',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_IMJFHBST = FieldMetadata(
    name='NV_R_KLHYZSPH_F_IMJFHBST',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_HYLTFYZK = FieldMetadata(
    name='NV_R_KLHYZSPH_F_HYLTFYZK',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_HBTOFJFB = FieldMetadata(
    name='NV_R_KLHYZSPH_F_HBTOFJFB',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_HHGTSGOX = FieldMetadata(
    name='NV_R_KLHYZSPH_F_HHGTSGOX',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_NBECFVTR = FieldMetadata(
    name='NV_R_KLHYZSPH_F_NBECFVTR',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_CPXQSYZB = FieldMetadata(
    name='NV_R_KLHYZSPH_F_CPXQSYZB',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_MBJISBNZ = FieldMetadata(
    name='NV_R_KLHYZSPH_F_MBJISBNZ',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_PDZWCDXW = FieldMetadata(
    name='NV_R_KLHYZSPH_F_PDZWCDXW',
    msb=9,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_OMCFLBAL = FieldMetadata(
    name='NV_R_KLHYZSPH_F_OMCFLBAL',
    msb=9,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_NKLXJYMD = FieldMetadata(
    name='NV_R_KLHYZSPH_F_NKLXJYMD',
    msb=9,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_IHQOCTPV = FieldMetadata(
    name='NV_R_KLHYZSPH_F_IHQOCTPV',
    msb=9,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_YJVWXNNE = FieldMetadata(
    name='NV_R_KLHYZSPH_F_YJVWXNNE',
    msb=9,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_VHLUWKFL = FieldMetadata(
    name='NV_R_KLHYZSPH_F_VHLUWKFL',
    msb=9,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_ZJAXCAQG = FieldMetadata(
    name='NV_R_KLHYZSPH_F_ZJAXCAQG',
    msb=8,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_OBEWYHPX = FieldMetadata(
    name='NV_R_KLHYZSPH_F_OBEWYHPX',
    msb=8,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_IRSEKWPQ = FieldMetadata(
    name='NV_R_KLHYZSPH_F_IRSEKWPQ',
    msb=16,
    lsb=9,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_VCXMQJEV = FieldMetadata(
    name='NV_R_KLHYZSPH_F_VCXMQJEV',
    msb=11,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_JXKCEYNM = FieldMetadata(
    name='NV_R_KLHYZSPH_F_JXKCEYNM',
    msb=8,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_DURSWDLP = FieldMetadata(
    name='NV_R_KLHYZSPH_F_DURSWDLP',
    msb=8,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_BJBZERKJ = FieldMetadata(
    name='NV_R_KLHYZSPH_F_BJBZERKJ',
    msb=8,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_MVPRIXOL = FieldMetadata(
    name='NV_R_KLHYZSPH_F_MVPRIXOL',
    msb=15,
    lsb=10,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_TDCDFVOD = FieldMetadata(
    name='NV_R_KLHYZSPH_F_TDCDFVOD',
    msb=9,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_XOCPWSRT = FieldMetadata(
    name='NV_R_KLHYZSPH_F_XOCPWSRT',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_INSVKLRN = FieldMetadata(
    name='NV_R_KLHYZSPH_F_INSVKLRN',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_NQFZWAHO = FieldMetadata(
    name='NV_R_KLHYZSPH_F_NQFZWAHO',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_DSQEGPFA = FieldMetadata(
    name='NV_R_KLHYZSPH_F_DSQEGPFA',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_IKVTHDBZ = FieldMetadata(
    name='NV_R_KLHYZSPH_F_IKVTHDBZ',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_CIWVBJVE = FieldMetadata(
    name='NV_R_KLHYZSPH_F_CIWVBJVE',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_LWLGMGAQ = FieldMetadata(
    name='NV_R_KLHYZSPH_F_LWLGMGAQ',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_OWIZHHHO = FieldMetadata(
    name='NV_R_KLHYZSPH_F_OWIZHHHO',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_MBHCYZPH = FieldMetadata(
    name='NV_R_KLHYZSPH_F_MBHCYZPH',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_YVRYVWIH = FieldMetadata(
    name='NV_R_KLHYZSPH_F_YVRYVWIH',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_UFKZEDCN = FieldMetadata(
    name='NV_R_KLHYZSPH_F_UFKZEDCN',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_QPRFLYFC = FieldMetadata(
    name='NV_R_KLHYZSPH_F_QPRFLYFC',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_KRYCNSZJ = FieldMetadata(
    name='NV_R_KLHYZSPH_F_KRYCNSZJ',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_GDDJPIHT = FieldMetadata(
    name='NV_R_KLHYZSPH_F_GDDJPIHT',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_PYWRHSKK = FieldMetadata(
    name='NV_R_KLHYZSPH_F_PYWRHSKK',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_SMFYIMNS = FieldMetadata(
    name='NV_R_KLHYZSPH_F_SMFYIMNS',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_HDYAZFOA = FieldMetadata(
    name='NV_R_KLHYZSPH_F_HDYAZFOA',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_UHBEOVYM = FieldMetadata(
    name='NV_R_KLHYZSPH_F_UHBEOVYM',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_ZICNZMHT = FieldMetadata(
    name='NV_R_KLHYZSPH_F_ZICNZMHT',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_ZJCFMQNK = FieldMetadata(
    name='NV_R_KLHYZSPH_F_ZJCFMQNK',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_DEBPYEVL = FieldMetadata(
    name='NV_R_KLHYZSPH_F_DEBPYEVL',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_AHTIHXXI = FieldMetadata(
    name='NV_R_KLHYZSPH_F_AHTIHXXI',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_RHTNASDF = FieldMetadata(
    name='NV_R_KLHYZSPH_F_RHTNASDF',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_DVLAWAYM = FieldMetadata(
    name='NV_R_KLHYZSPH_F_DVLAWAYM',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_IHLFYVUA = FieldMetadata(
    name='NV_R_KLHYZSPH_F_IHLFYVUA',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_VJZXXNRN = FieldMetadata(
    name='NV_R_KLHYZSPH_F_VJZXXNRN',
    msb=7,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_WPOKVHWB = FieldMetadata(
    name='NV_R_KLHYZSPH_F_WPOKVHWB',
    msb=11,
    lsb=4,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_QAWEVVQS = FieldMetadata(
    name='NV_R_KLHYZSPH_F_QAWEVVQS',
    msb=3,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_ECKYQVQO = FieldMetadata(
    name='NV_R_KLHYZSPH_F_ECKYQVQO',
    msb=11,
    lsb=4,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_AUKLBKOY = FieldMetadata(
    name='NV_R_KLHYZSPH_F_AUKLBKOY',
    msb=3,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_AZTHXCVL = FieldMetadata(
    name='NV_R_KLHYZSPH_F_AZTHXCVL',
    msb=11,
    lsb=4,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_FWIQTCRQ = FieldMetadata(
    name='NV_R_KLHYZSPH_F_FWIQTCRQ',
    msb=3,
    lsb=0,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_OFGCIMFI = FieldMetadata(
    name='NV_R_KLHYZSPH_F_OFGCIMFI',
    msb=31,
    lsb=30,
    register=NV_R_KLHYZSPH
)

NV_R_KLHYZSPH_F_OFGCIMFI_V_FQNJWQHN = ValueMetadata(
    name='NV_R_KLHYZSPH_F_OFGCIMFI_V_FQNJWQHN',
    value=2,
    field=NV_R_KLHYZSPH_F_OFGCIMFI
)
NV_R_KLHYZSPH_F_OFGCIMFI_V_CMVOBGUP = ValueMetadata(
    name='NV_R_KLHYZSPH_F_OFGCIMFI_V_CMVOBGUP',
    value=0,
    field=NV_R_KLHYZSPH_F_OFGCIMFI
)
NV_R_KLHYZSPH_F_OFGCIMFI_V_TVGHFSAG = ValueMetadata(
    name='NV_R_KLHYZSPH_F_OFGCIMFI_V_TVGHFSAG',
    value=1,
    field=NV_R_KLHYZSPH_F_OFGCIMFI
)

NV_R_LNELYPMO = RegisterMetadata(
    name='NV_R_LNELYPMO',
    address=0xef4,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_LNELYPMO_F_CTBCFRUB = FieldMetadata(
    name='NV_R_LNELYPMO_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_LNELYPMO
)

NV_R_LNELYPMO_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LNELYPMO_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_LNELYPMO_F_CTBCFRUB
)

NV_R_LNELYPMO_F_QVYAARWP = FieldMetadata(
    name='NV_R_LNELYPMO_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_LNELYPMO
)

NV_R_LNELYPMO_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LNELYPMO_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_LNELYPMO_F_QVYAARWP
)

NV_R_YJIZWFPE = RegisterMetadata(
    name='NV_R_YJIZWFPE',
    address=0xef0,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_YJIZWFPE_F_KWDQPURW = FieldMetadata(
    name='NV_R_YJIZWFPE_F_KWDQPURW',
    msb=5,
    lsb=5,
    register=NV_R_YJIZWFPE
)

NV_R_YJIZWFPE_F_KWDQPURW_V_DOGBFDTH = ValueMetadata(
    name='NV_R_YJIZWFPE_F_KWDQPURW_V_DOGBFDTH',
    value=0,
    field=NV_R_YJIZWFPE_F_KWDQPURW
)
NV_R_YJIZWFPE_F_KWDQPURW_V_PIUJQBKV = ValueMetadata(
    name='NV_R_YJIZWFPE_F_KWDQPURW_V_PIUJQBKV',
    value=1,
    field=NV_R_YJIZWFPE_F_KWDQPURW
)

NV_R_YJIZWFPE_F_ELAIAXBC = FieldMetadata(
    name='NV_R_YJIZWFPE_F_ELAIAXBC',
    msb=1,
    lsb=1,
    register=NV_R_YJIZWFPE
)

NV_R_YJIZWFPE_F_ELAIAXBC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_YJIZWFPE_F_ELAIAXBC_V_DOGBFDTH',
    value=0,
    field=NV_R_YJIZWFPE_F_ELAIAXBC
)
NV_R_YJIZWFPE_F_ELAIAXBC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_YJIZWFPE_F_ELAIAXBC_V_PIUJQBKV',
    value=1,
    field=NV_R_YJIZWFPE_F_ELAIAXBC
)

NV_R_YJIZWFPE_F_OJXVVOEC = FieldMetadata(
    name='NV_R_YJIZWFPE_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_YJIZWFPE
)

NV_R_YJIZWFPE_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_YJIZWFPE_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_YJIZWFPE_F_OJXVVOEC
)
NV_R_YJIZWFPE_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_YJIZWFPE_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_YJIZWFPE_F_OJXVVOEC
)

NV_R_YJIZWFPE_F_PBAFTNZG = FieldMetadata(
    name='NV_R_YJIZWFPE_F_PBAFTNZG',
    msb=3,
    lsb=3,
    register=NV_R_YJIZWFPE
)

NV_R_YJIZWFPE_F_PBAFTNZG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_YJIZWFPE_F_PBAFTNZG_V_DOGBFDTH',
    value=0,
    field=NV_R_YJIZWFPE_F_PBAFTNZG
)
NV_R_YJIZWFPE_F_PBAFTNZG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_YJIZWFPE_F_PBAFTNZG_V_PIUJQBKV',
    value=1,
    field=NV_R_YJIZWFPE_F_PBAFTNZG
)

NV_R_YJIZWFPE_F_VEOHPHHB = FieldMetadata(
    name='NV_R_YJIZWFPE_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_YJIZWFPE
)

NV_R_YJIZWFPE_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_YJIZWFPE_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_YJIZWFPE_F_VEOHPHHB
)
NV_R_YJIZWFPE_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_YJIZWFPE_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_YJIZWFPE_F_VEOHPHHB
)

NV_R_YJIZWFPE_F_KVYOKDXN = FieldMetadata(
    name='NV_R_YJIZWFPE_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_YJIZWFPE
)

NV_R_YJIZWFPE_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_YJIZWFPE_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_YJIZWFPE_F_KVYOKDXN
)
NV_R_YJIZWFPE_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_YJIZWFPE_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_YJIZWFPE_F_KVYOKDXN
)
NV_R_YJIZWFPE_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_YJIZWFPE_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_YJIZWFPE_F_KVYOKDXN
)

NV_R_YJIZWFPE_F_IOJLTETI = FieldMetadata(
    name='NV_R_YJIZWFPE_F_IOJLTETI',
    msb=4,
    lsb=4,
    register=NV_R_YJIZWFPE
)

NV_R_YJIZWFPE_F_IOJLTETI_V_DOGBFDTH = ValueMetadata(
    name='NV_R_YJIZWFPE_F_IOJLTETI_V_DOGBFDTH',
    value=0,
    field=NV_R_YJIZWFPE_F_IOJLTETI
)
NV_R_YJIZWFPE_F_IOJLTETI_V_PIUJQBKV = ValueMetadata(
    name='NV_R_YJIZWFPE_F_IOJLTETI_V_PIUJQBKV',
    value=1,
    field=NV_R_YJIZWFPE_F_IOJLTETI
)

NV_R_YJIZWFPE_F_TZSQEAIB = FieldMetadata(
    name='NV_R_YJIZWFPE_F_TZSQEAIB',
    msb=0,
    lsb=0,
    register=NV_R_YJIZWFPE
)

NV_R_YJIZWFPE_F_TZSQEAIB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_YJIZWFPE_F_TZSQEAIB_V_DOGBFDTH',
    value=0,
    field=NV_R_YJIZWFPE_F_TZSQEAIB
)
NV_R_YJIZWFPE_F_TZSQEAIB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_YJIZWFPE_F_TZSQEAIB_V_PIUJQBKV',
    value=1,
    field=NV_R_YJIZWFPE_F_TZSQEAIB
)

NV_R_YJIZWFPE_F_WQYWMRVP = FieldMetadata(
    name='NV_R_YJIZWFPE_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_YJIZWFPE
)

NV_R_YJIZWFPE_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_YJIZWFPE_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_YJIZWFPE_F_WQYWMRVP
)
NV_R_YJIZWFPE_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_YJIZWFPE_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_YJIZWFPE_F_WQYWMRVP
)

NV_R_YJIZWFPE_F_PUPDHQBL = FieldMetadata(
    name='NV_R_YJIZWFPE_F_PUPDHQBL',
    msb=2,
    lsb=2,
    register=NV_R_YJIZWFPE
)

NV_R_YJIZWFPE_F_PUPDHQBL_V_DOGBFDTH = ValueMetadata(
    name='NV_R_YJIZWFPE_F_PUPDHQBL_V_DOGBFDTH',
    value=0,
    field=NV_R_YJIZWFPE_F_PUPDHQBL
)
NV_R_YJIZWFPE_F_PUPDHQBL_V_PIUJQBKV = ValueMetadata(
    name='NV_R_YJIZWFPE_F_PUPDHQBL_V_PIUJQBKV',
    value=1,
    field=NV_R_YJIZWFPE_F_PUPDHQBL
)

NV_R_YJIZWFPE_F_MLEOLJIS = FieldMetadata(
    name='NV_R_YJIZWFPE_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_YJIZWFPE
)

NV_R_YJIZWFPE_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_YJIZWFPE_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_YJIZWFPE_F_MLEOLJIS
)
NV_R_YJIZWFPE_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_YJIZWFPE_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_YJIZWFPE_F_MLEOLJIS
)

NV_R_JMORBECA = RegisterMetadata(
    name='NV_R_JMORBECA',
    address=0xef8,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_JMORBECA_F_CTBCFRUB = FieldMetadata(
    name='NV_R_JMORBECA_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_JMORBECA
)

NV_R_JMORBECA_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_JMORBECA_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_JMORBECA_F_CTBCFRUB
)

NV_R_JMORBECA_F_QVYAARWP = FieldMetadata(
    name='NV_R_JMORBECA_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_JMORBECA
)

NV_R_JMORBECA_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_JMORBECA_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_JMORBECA_F_QVYAARWP
)

NV_R_XBNRZXCR = RegisterMetadata(
    name='NV_R_XBNRZXCR',
    address=0x1120,
    zero_based=True
)

NV_R_XBNRZXCR_F_ZIGJNGSD = FieldMetadata(
    name='NV_R_XBNRZXCR_F_ZIGJNGSD',
    msb=22,
    lsb=0,
    register=NV_R_XBNRZXCR
)

NV_R_XBNRZXCR_F_SEXGFUSZ = FieldMetadata(
    name='NV_R_XBNRZXCR_F_SEXGFUSZ',
    msb=21,
    lsb=20,
    register=NV_R_XBNRZXCR
)

NV_R_XBNRZXCR_F_GNCFTOVU = FieldMetadata(
    name='NV_R_XBNRZXCR_F_GNCFTOVU',
    msb=2,
    lsb=0,
    register=NV_R_XBNRZXCR
)

NV_R_XBNRZXCR_F_WDHNBQNK = FieldMetadata(
    name='NV_R_XBNRZXCR_F_WDHNBQNK',
    msb=19,
    lsb=8,
    register=NV_R_XBNRZXCR
)

NV_R_XBNRZXCR_F_HMBHUSZP = FieldMetadata(
    name='NV_R_XBNRZXCR_F_HMBHUSZP',
    msb=22,
    lsb=22,
    register=NV_R_XBNRZXCR
)

NV_R_XBNRZXCR_F_FEERLBNG = FieldMetadata(
    name='NV_R_XBNRZXCR_F_FEERLBNG',
    msb=7,
    lsb=3,
    register=NV_R_XBNRZXCR
)

NV_R_XBNRZXCR_F_EXMOKFEV = FieldMetadata(
    name='NV_R_XBNRZXCR_F_EXMOKFEV',
    msb=7,
    lsb=6,
    register=NV_R_XBNRZXCR
)

NV_R_JSIAWHPU = RegisterMetadata(
    name='NV_R_JSIAWHPU',
    address=0x10fc,
    zero_based=True
)

NV_R_JSIAWHPU_F_OJYQBJXU = FieldMetadata(
    name='NV_R_JSIAWHPU_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_SPGCMIVK = FieldMetadata(
    name='NV_R_JSIAWHPU_F_SPGCMIVK',
    msb=21,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_CMIRIWSV = FieldMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV',
    msb=29,
    lsb=22,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_CMIRIWSV_V_IZNDWFDP = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_IZNDWFDP',
    value=8,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_JSAHBBTM = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_JSAHBBTM',
    value=9,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_SEBMZEME = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_SEBMZEME',
    value=10,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_DTQCFPNA = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_DTQCFPNA',
    value=11,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_KYVQOIDU = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_KYVQOIDU',
    value=0,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_GHVVYIOF = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_GHVVYIOF',
    value=1,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_JGXXVPER = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_JGXXVPER',
    value=2,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_OKQHAWBB = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_OKQHAWBB',
    value=3,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_IGVRDHSA = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_IGVRDHSA',
    value=4,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_BJJBUDSL = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_BJJBUDSL',
    value=5,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_NRGNBBPQ = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_NRGNBBPQ',
    value=6,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_EKNISTBR = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_EKNISTBR',
    value=7,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_PGSYHHVE = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_PGSYHHVE',
    value=13,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_UXJGURAX = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_UXJGURAX',
    value=12,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_QWWBEQMS = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_QWWBEQMS',
    value=0,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_NBBWYRIG = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_NBBWYRIG',
    value=1,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_SWVVTNLK = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_SWVVTNLK',
    value=2,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_OTSWZFCR = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_OTSWZFCR',
    value=3,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_CSRKWDGO = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_CSRKWDGO',
    value=39,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_ORGICIHS = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_ORGICIHS',
    value=40,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_EYIXRFGX = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_EYIXRFGX',
    value=35,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_KFPOQASI = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_KFPOQASI',
    value=34,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_WPWBKHJE = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_WPWBKHJE',
    value=37,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_EEUGWZSG = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_EEUGWZSG',
    value=0,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_QUUGDQPL = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_QUUGDQPL',
    value=1,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_LKELNZMU = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_LKELNZMU',
    value=36,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_RSFATZWE = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_RSFATZWE',
    value=41,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_VGDLATGR = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_VGDLATGR',
    value=18,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_AWDMJNRQ = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_AWDMJNRQ',
    value=19,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_GDTZIIUX = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_GDTZIIUX',
    value=20,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_RGDBBIRS = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_RGDBBIRS',
    value=21,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_PYYFFBZF = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_PYYFFBZF',
    value=22,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_ZDAXSZOC = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_ZDAXSZOC',
    value=23,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_ZGKOZVLG = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_ZGKOZVLG',
    value=24,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_LXFZIIVC = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_LXFZIIVC',
    value=25,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_LIYVCEVC = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_LIYVCEVC',
    value=10,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_TMZYYQRE = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_TMZYYQRE',
    value=11,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_LITMRWHR = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_LITMRWHR',
    value=12,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_SUXZYCLN = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_SUXZYCLN',
    value=13,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_JKGEPTXY = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_JKGEPTXY',
    value=14,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_BIJQQCBV = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_BIJQQCBV',
    value=15,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_RCXNOXVH = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_RCXNOXVH',
    value=16,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_UPMZYBHK = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_UPMZYBHK',
    value=17,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_JKIFIDZS = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_JKIFIDZS',
    value=2,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_CWMIEYYA = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_CWMIEYYA',
    value=3,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_PTVPYSBH = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_PTVPYSBH',
    value=4,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_IGQUZTXI = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_IGQUZTXI',
    value=5,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_IPJEHZXT = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_IPJEHZXT',
    value=6,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_IBHISRTF = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_IBHISRTF',
    value=7,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_AHLHOHQY = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_AHLHOHQY',
    value=8,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_MAPDVIBO = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_MAPDVIBO',
    value=9,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_JFLKUSVU = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_JFLKUSVU',
    value=33,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_XZPNLKQC = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_XZPNLKQC',
    value=32,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_GZQSWWSJ = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_GZQSWWSJ',
    value=31,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_KHINLTNU = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_KHINLTNU',
    value=26,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_AYPUIJOI = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_AYPUIJOI',
    value=27,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_MDSIVZHL = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_MDSIVZHL',
    value=43,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_CFWZRRWE = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_CFWZRRWE',
    value=28,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_JSOPIVBC = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_JSOPIVBC',
    value=38,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_ZSEHHQVS = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_ZSEHHQVS',
    value=30,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_OOWFZASJ = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_OOWFZASJ',
    value=42,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)
NV_R_JSIAWHPU_F_CMIRIWSV_V_USZOAVIE = ValueMetadata(
    name='NV_R_JSIAWHPU_F_CMIRIWSV_V_USZOAVIE',
    value=29,
    field=NV_R_JSIAWHPU_F_CMIRIWSV
)

NV_R_JSIAWHPU_F_NVKIPOLT = FieldMetadata(
    name='NV_R_JSIAWHPU_F_NVKIPOLT',
    msb=11,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_ZOUDCMGL = FieldMetadata(
    name='NV_R_JSIAWHPU_F_ZOUDCMGL',
    msb=11,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_NOARSXQM = FieldMetadata(
    name='NV_R_JSIAWHPU_F_NOARSXQM',
    msb=11,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_OEUAITQM = FieldMetadata(
    name='NV_R_JSIAWHPU_F_OEUAITQM',
    msb=11,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_RBGFPZWZ = FieldMetadata(
    name='NV_R_JSIAWHPU_F_RBGFPZWZ',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_IMJFHBST = FieldMetadata(
    name='NV_R_JSIAWHPU_F_IMJFHBST',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_HYLTFYZK = FieldMetadata(
    name='NV_R_JSIAWHPU_F_HYLTFYZK',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_HBTOFJFB = FieldMetadata(
    name='NV_R_JSIAWHPU_F_HBTOFJFB',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_HHGTSGOX = FieldMetadata(
    name='NV_R_JSIAWHPU_F_HHGTSGOX',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_NBECFVTR = FieldMetadata(
    name='NV_R_JSIAWHPU_F_NBECFVTR',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_CPXQSYZB = FieldMetadata(
    name='NV_R_JSIAWHPU_F_CPXQSYZB',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_MBJISBNZ = FieldMetadata(
    name='NV_R_JSIAWHPU_F_MBJISBNZ',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_PDZWCDXW = FieldMetadata(
    name='NV_R_JSIAWHPU_F_PDZWCDXW',
    msb=9,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_OMCFLBAL = FieldMetadata(
    name='NV_R_JSIAWHPU_F_OMCFLBAL',
    msb=9,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_NKLXJYMD = FieldMetadata(
    name='NV_R_JSIAWHPU_F_NKLXJYMD',
    msb=9,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_IHQOCTPV = FieldMetadata(
    name='NV_R_JSIAWHPU_F_IHQOCTPV',
    msb=9,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_YJVWXNNE = FieldMetadata(
    name='NV_R_JSIAWHPU_F_YJVWXNNE',
    msb=9,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_VHLUWKFL = FieldMetadata(
    name='NV_R_JSIAWHPU_F_VHLUWKFL',
    msb=9,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_ZJAXCAQG = FieldMetadata(
    name='NV_R_JSIAWHPU_F_ZJAXCAQG',
    msb=8,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_OBEWYHPX = FieldMetadata(
    name='NV_R_JSIAWHPU_F_OBEWYHPX',
    msb=8,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_IRSEKWPQ = FieldMetadata(
    name='NV_R_JSIAWHPU_F_IRSEKWPQ',
    msb=16,
    lsb=9,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_VCXMQJEV = FieldMetadata(
    name='NV_R_JSIAWHPU_F_VCXMQJEV',
    msb=11,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_JXKCEYNM = FieldMetadata(
    name='NV_R_JSIAWHPU_F_JXKCEYNM',
    msb=8,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_DURSWDLP = FieldMetadata(
    name='NV_R_JSIAWHPU_F_DURSWDLP',
    msb=8,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_BJBZERKJ = FieldMetadata(
    name='NV_R_JSIAWHPU_F_BJBZERKJ',
    msb=8,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_MVPRIXOL = FieldMetadata(
    name='NV_R_JSIAWHPU_F_MVPRIXOL',
    msb=15,
    lsb=10,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_TDCDFVOD = FieldMetadata(
    name='NV_R_JSIAWHPU_F_TDCDFVOD',
    msb=9,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_XOCPWSRT = FieldMetadata(
    name='NV_R_JSIAWHPU_F_XOCPWSRT',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_INSVKLRN = FieldMetadata(
    name='NV_R_JSIAWHPU_F_INSVKLRN',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_NQFZWAHO = FieldMetadata(
    name='NV_R_JSIAWHPU_F_NQFZWAHO',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_DSQEGPFA = FieldMetadata(
    name='NV_R_JSIAWHPU_F_DSQEGPFA',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_IKVTHDBZ = FieldMetadata(
    name='NV_R_JSIAWHPU_F_IKVTHDBZ',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_CIWVBJVE = FieldMetadata(
    name='NV_R_JSIAWHPU_F_CIWVBJVE',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_LWLGMGAQ = FieldMetadata(
    name='NV_R_JSIAWHPU_F_LWLGMGAQ',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_OWIZHHHO = FieldMetadata(
    name='NV_R_JSIAWHPU_F_OWIZHHHO',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_MBHCYZPH = FieldMetadata(
    name='NV_R_JSIAWHPU_F_MBHCYZPH',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_YVRYVWIH = FieldMetadata(
    name='NV_R_JSIAWHPU_F_YVRYVWIH',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_UFKZEDCN = FieldMetadata(
    name='NV_R_JSIAWHPU_F_UFKZEDCN',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_QPRFLYFC = FieldMetadata(
    name='NV_R_JSIAWHPU_F_QPRFLYFC',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_KRYCNSZJ = FieldMetadata(
    name='NV_R_JSIAWHPU_F_KRYCNSZJ',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_GDDJPIHT = FieldMetadata(
    name='NV_R_JSIAWHPU_F_GDDJPIHT',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_PYWRHSKK = FieldMetadata(
    name='NV_R_JSIAWHPU_F_PYWRHSKK',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_SMFYIMNS = FieldMetadata(
    name='NV_R_JSIAWHPU_F_SMFYIMNS',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_HDYAZFOA = FieldMetadata(
    name='NV_R_JSIAWHPU_F_HDYAZFOA',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_UHBEOVYM = FieldMetadata(
    name='NV_R_JSIAWHPU_F_UHBEOVYM',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_ZICNZMHT = FieldMetadata(
    name='NV_R_JSIAWHPU_F_ZICNZMHT',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_ZJCFMQNK = FieldMetadata(
    name='NV_R_JSIAWHPU_F_ZJCFMQNK',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_DEBPYEVL = FieldMetadata(
    name='NV_R_JSIAWHPU_F_DEBPYEVL',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_AHTIHXXI = FieldMetadata(
    name='NV_R_JSIAWHPU_F_AHTIHXXI',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_RHTNASDF = FieldMetadata(
    name='NV_R_JSIAWHPU_F_RHTNASDF',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_DVLAWAYM = FieldMetadata(
    name='NV_R_JSIAWHPU_F_DVLAWAYM',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_IHLFYVUA = FieldMetadata(
    name='NV_R_JSIAWHPU_F_IHLFYVUA',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_VJZXXNRN = FieldMetadata(
    name='NV_R_JSIAWHPU_F_VJZXXNRN',
    msb=7,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_WPOKVHWB = FieldMetadata(
    name='NV_R_JSIAWHPU_F_WPOKVHWB',
    msb=11,
    lsb=4,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_QAWEVVQS = FieldMetadata(
    name='NV_R_JSIAWHPU_F_QAWEVVQS',
    msb=3,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_ECKYQVQO = FieldMetadata(
    name='NV_R_JSIAWHPU_F_ECKYQVQO',
    msb=11,
    lsb=4,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_AUKLBKOY = FieldMetadata(
    name='NV_R_JSIAWHPU_F_AUKLBKOY',
    msb=3,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_AZTHXCVL = FieldMetadata(
    name='NV_R_JSIAWHPU_F_AZTHXCVL',
    msb=11,
    lsb=4,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_FWIQTCRQ = FieldMetadata(
    name='NV_R_JSIAWHPU_F_FWIQTCRQ',
    msb=3,
    lsb=0,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_OFGCIMFI = FieldMetadata(
    name='NV_R_JSIAWHPU_F_OFGCIMFI',
    msb=31,
    lsb=30,
    register=NV_R_JSIAWHPU
)

NV_R_JSIAWHPU_F_OFGCIMFI_V_FQNJWQHN = ValueMetadata(
    name='NV_R_JSIAWHPU_F_OFGCIMFI_V_FQNJWQHN',
    value=2,
    field=NV_R_JSIAWHPU_F_OFGCIMFI
)
NV_R_JSIAWHPU_F_OFGCIMFI_V_CMVOBGUP = ValueMetadata(
    name='NV_R_JSIAWHPU_F_OFGCIMFI_V_CMVOBGUP',
    value=0,
    field=NV_R_JSIAWHPU_F_OFGCIMFI
)
NV_R_JSIAWHPU_F_OFGCIMFI_V_TVGHFSAG = ValueMetadata(
    name='NV_R_JSIAWHPU_F_OFGCIMFI_V_TVGHFSAG',
    value=1,
    field=NV_R_JSIAWHPU_F_OFGCIMFI
)

NV_R_YDGQQOBE = RegisterMetadata(
    name='NV_R_YDGQQOBE',
    address=0x10f4,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_YDGQQOBE_F_CTBCFRUB = FieldMetadata(
    name='NV_R_YDGQQOBE_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_YDGQQOBE
)

NV_R_YDGQQOBE_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_YDGQQOBE_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_YDGQQOBE_F_CTBCFRUB
)

NV_R_YDGQQOBE_F_QVYAARWP = FieldMetadata(
    name='NV_R_YDGQQOBE_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_YDGQQOBE
)

NV_R_YDGQQOBE_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_YDGQQOBE_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_YDGQQOBE_F_QVYAARWP
)

NV_R_ITCGQRNP = RegisterMetadata(
    name='NV_R_ITCGQRNP',
    address=0x10f0,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_ITCGQRNP_F_KWDQPURW = FieldMetadata(
    name='NV_R_ITCGQRNP_F_KWDQPURW',
    msb=5,
    lsb=5,
    register=NV_R_ITCGQRNP
)

NV_R_ITCGQRNP_F_KWDQPURW_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ITCGQRNP_F_KWDQPURW_V_DOGBFDTH',
    value=0,
    field=NV_R_ITCGQRNP_F_KWDQPURW
)
NV_R_ITCGQRNP_F_KWDQPURW_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ITCGQRNP_F_KWDQPURW_V_PIUJQBKV',
    value=1,
    field=NV_R_ITCGQRNP_F_KWDQPURW
)

NV_R_ITCGQRNP_F_ELAIAXBC = FieldMetadata(
    name='NV_R_ITCGQRNP_F_ELAIAXBC',
    msb=1,
    lsb=1,
    register=NV_R_ITCGQRNP
)

NV_R_ITCGQRNP_F_ELAIAXBC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ITCGQRNP_F_ELAIAXBC_V_DOGBFDTH',
    value=0,
    field=NV_R_ITCGQRNP_F_ELAIAXBC
)
NV_R_ITCGQRNP_F_ELAIAXBC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ITCGQRNP_F_ELAIAXBC_V_PIUJQBKV',
    value=1,
    field=NV_R_ITCGQRNP_F_ELAIAXBC
)

NV_R_ITCGQRNP_F_OJXVVOEC = FieldMetadata(
    name='NV_R_ITCGQRNP_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_ITCGQRNP
)

NV_R_ITCGQRNP_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ITCGQRNP_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_ITCGQRNP_F_OJXVVOEC
)
NV_R_ITCGQRNP_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ITCGQRNP_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_ITCGQRNP_F_OJXVVOEC
)

NV_R_ITCGQRNP_F_PBAFTNZG = FieldMetadata(
    name='NV_R_ITCGQRNP_F_PBAFTNZG',
    msb=3,
    lsb=3,
    register=NV_R_ITCGQRNP
)

NV_R_ITCGQRNP_F_PBAFTNZG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ITCGQRNP_F_PBAFTNZG_V_DOGBFDTH',
    value=0,
    field=NV_R_ITCGQRNP_F_PBAFTNZG
)
NV_R_ITCGQRNP_F_PBAFTNZG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ITCGQRNP_F_PBAFTNZG_V_PIUJQBKV',
    value=1,
    field=NV_R_ITCGQRNP_F_PBAFTNZG
)

NV_R_ITCGQRNP_F_VEOHPHHB = FieldMetadata(
    name='NV_R_ITCGQRNP_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_ITCGQRNP
)

NV_R_ITCGQRNP_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ITCGQRNP_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_ITCGQRNP_F_VEOHPHHB
)
NV_R_ITCGQRNP_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ITCGQRNP_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_ITCGQRNP_F_VEOHPHHB
)

NV_R_ITCGQRNP_F_KVYOKDXN = FieldMetadata(
    name='NV_R_ITCGQRNP_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_ITCGQRNP
)

NV_R_ITCGQRNP_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_ITCGQRNP_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_ITCGQRNP_F_KVYOKDXN
)
NV_R_ITCGQRNP_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_ITCGQRNP_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_ITCGQRNP_F_KVYOKDXN
)
NV_R_ITCGQRNP_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_ITCGQRNP_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_ITCGQRNP_F_KVYOKDXN
)

NV_R_ITCGQRNP_F_IOJLTETI = FieldMetadata(
    name='NV_R_ITCGQRNP_F_IOJLTETI',
    msb=4,
    lsb=4,
    register=NV_R_ITCGQRNP
)

NV_R_ITCGQRNP_F_IOJLTETI_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ITCGQRNP_F_IOJLTETI_V_DOGBFDTH',
    value=0,
    field=NV_R_ITCGQRNP_F_IOJLTETI
)
NV_R_ITCGQRNP_F_IOJLTETI_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ITCGQRNP_F_IOJLTETI_V_PIUJQBKV',
    value=1,
    field=NV_R_ITCGQRNP_F_IOJLTETI
)

NV_R_ITCGQRNP_F_TZSQEAIB = FieldMetadata(
    name='NV_R_ITCGQRNP_F_TZSQEAIB',
    msb=0,
    lsb=0,
    register=NV_R_ITCGQRNP
)

NV_R_ITCGQRNP_F_TZSQEAIB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ITCGQRNP_F_TZSQEAIB_V_DOGBFDTH',
    value=0,
    field=NV_R_ITCGQRNP_F_TZSQEAIB
)
NV_R_ITCGQRNP_F_TZSQEAIB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ITCGQRNP_F_TZSQEAIB_V_PIUJQBKV',
    value=1,
    field=NV_R_ITCGQRNP_F_TZSQEAIB
)

NV_R_ITCGQRNP_F_WQYWMRVP = FieldMetadata(
    name='NV_R_ITCGQRNP_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_ITCGQRNP
)

NV_R_ITCGQRNP_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ITCGQRNP_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_ITCGQRNP_F_WQYWMRVP
)
NV_R_ITCGQRNP_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ITCGQRNP_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_ITCGQRNP_F_WQYWMRVP
)

NV_R_ITCGQRNP_F_PUPDHQBL = FieldMetadata(
    name='NV_R_ITCGQRNP_F_PUPDHQBL',
    msb=2,
    lsb=2,
    register=NV_R_ITCGQRNP
)

NV_R_ITCGQRNP_F_PUPDHQBL_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ITCGQRNP_F_PUPDHQBL_V_DOGBFDTH',
    value=0,
    field=NV_R_ITCGQRNP_F_PUPDHQBL
)
NV_R_ITCGQRNP_F_PUPDHQBL_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ITCGQRNP_F_PUPDHQBL_V_PIUJQBKV',
    value=1,
    field=NV_R_ITCGQRNP_F_PUPDHQBL
)

NV_R_ITCGQRNP_F_MLEOLJIS = FieldMetadata(
    name='NV_R_ITCGQRNP_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_ITCGQRNP
)

NV_R_ITCGQRNP_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_ITCGQRNP_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_ITCGQRNP_F_MLEOLJIS
)
NV_R_ITCGQRNP_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_ITCGQRNP_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_ITCGQRNP_F_MLEOLJIS
)

NV_R_QUWAZECT = RegisterMetadata(
    name='NV_R_QUWAZECT',
    address=0x10f8,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_QUWAZECT_F_CTBCFRUB = FieldMetadata(
    name='NV_R_QUWAZECT_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_QUWAZECT
)

NV_R_QUWAZECT_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_QUWAZECT_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_QUWAZECT_F_CTBCFRUB
)

NV_R_QUWAZECT_F_QVYAARWP = FieldMetadata(
    name='NV_R_QUWAZECT_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_QUWAZECT
)

NV_R_QUWAZECT_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_QUWAZECT_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_QUWAZECT_F_QVYAARWP
)

NV_R_PYZGKULO = RegisterMetadata(
    name='NV_R_PYZGKULO',
    address=0x1320,
    zero_based=True
)

NV_R_PYZGKULO_F_ZIGJNGSD = FieldMetadata(
    name='NV_R_PYZGKULO_F_ZIGJNGSD',
    msb=22,
    lsb=0,
    register=NV_R_PYZGKULO
)

NV_R_PYZGKULO_F_SEXGFUSZ = FieldMetadata(
    name='NV_R_PYZGKULO_F_SEXGFUSZ',
    msb=21,
    lsb=20,
    register=NV_R_PYZGKULO
)

NV_R_PYZGKULO_F_GNCFTOVU = FieldMetadata(
    name='NV_R_PYZGKULO_F_GNCFTOVU',
    msb=2,
    lsb=0,
    register=NV_R_PYZGKULO
)

NV_R_PYZGKULO_F_WDHNBQNK = FieldMetadata(
    name='NV_R_PYZGKULO_F_WDHNBQNK',
    msb=19,
    lsb=8,
    register=NV_R_PYZGKULO
)

NV_R_PYZGKULO_F_HMBHUSZP = FieldMetadata(
    name='NV_R_PYZGKULO_F_HMBHUSZP',
    msb=22,
    lsb=22,
    register=NV_R_PYZGKULO
)

NV_R_PYZGKULO_F_FEERLBNG = FieldMetadata(
    name='NV_R_PYZGKULO_F_FEERLBNG',
    msb=7,
    lsb=3,
    register=NV_R_PYZGKULO
)

NV_R_PYZGKULO_F_EXMOKFEV = FieldMetadata(
    name='NV_R_PYZGKULO_F_EXMOKFEV',
    msb=7,
    lsb=6,
    register=NV_R_PYZGKULO
)

NV_R_ORDQEXFV = RegisterMetadata(
    name='NV_R_ORDQEXFV',
    address=0x12fc,
    zero_based=True
)

NV_R_ORDQEXFV_F_OJYQBJXU = FieldMetadata(
    name='NV_R_ORDQEXFV_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_SPGCMIVK = FieldMetadata(
    name='NV_R_ORDQEXFV_F_SPGCMIVK',
    msb=21,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_CMIRIWSV = FieldMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV',
    msb=29,
    lsb=22,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_CMIRIWSV_V_IZNDWFDP = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_IZNDWFDP',
    value=8,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_JSAHBBTM = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_JSAHBBTM',
    value=9,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_SEBMZEME = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_SEBMZEME',
    value=10,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_DTQCFPNA = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_DTQCFPNA',
    value=11,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_KYVQOIDU = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_KYVQOIDU',
    value=0,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_GHVVYIOF = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_GHVVYIOF',
    value=1,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_JGXXVPER = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_JGXXVPER',
    value=2,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_OKQHAWBB = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_OKQHAWBB',
    value=3,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_IGVRDHSA = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_IGVRDHSA',
    value=4,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_BJJBUDSL = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_BJJBUDSL',
    value=5,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_NRGNBBPQ = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_NRGNBBPQ',
    value=6,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_EKNISTBR = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_EKNISTBR',
    value=7,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_PGSYHHVE = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_PGSYHHVE',
    value=13,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_UXJGURAX = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_UXJGURAX',
    value=12,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_QWWBEQMS = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_QWWBEQMS',
    value=0,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_NBBWYRIG = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_NBBWYRIG',
    value=1,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_SWVVTNLK = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_SWVVTNLK',
    value=2,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_OTSWZFCR = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_OTSWZFCR',
    value=3,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_CSRKWDGO = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_CSRKWDGO',
    value=39,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_ORGICIHS = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_ORGICIHS',
    value=40,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_EYIXRFGX = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_EYIXRFGX',
    value=35,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_KFPOQASI = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_KFPOQASI',
    value=34,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_WPWBKHJE = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_WPWBKHJE',
    value=37,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_EEUGWZSG = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_EEUGWZSG',
    value=0,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_QUUGDQPL = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_QUUGDQPL',
    value=1,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_LKELNZMU = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_LKELNZMU',
    value=36,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_RSFATZWE = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_RSFATZWE',
    value=41,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_VGDLATGR = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_VGDLATGR',
    value=18,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_AWDMJNRQ = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_AWDMJNRQ',
    value=19,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_GDTZIIUX = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_GDTZIIUX',
    value=20,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_RGDBBIRS = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_RGDBBIRS',
    value=21,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_PYYFFBZF = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_PYYFFBZF',
    value=22,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_ZDAXSZOC = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_ZDAXSZOC',
    value=23,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_ZGKOZVLG = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_ZGKOZVLG',
    value=24,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_LXFZIIVC = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_LXFZIIVC',
    value=25,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_LIYVCEVC = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_LIYVCEVC',
    value=10,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_TMZYYQRE = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_TMZYYQRE',
    value=11,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_LITMRWHR = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_LITMRWHR',
    value=12,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_SUXZYCLN = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_SUXZYCLN',
    value=13,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_JKGEPTXY = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_JKGEPTXY',
    value=14,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_BIJQQCBV = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_BIJQQCBV',
    value=15,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_RCXNOXVH = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_RCXNOXVH',
    value=16,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_UPMZYBHK = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_UPMZYBHK',
    value=17,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_JKIFIDZS = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_JKIFIDZS',
    value=2,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_CWMIEYYA = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_CWMIEYYA',
    value=3,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_PTVPYSBH = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_PTVPYSBH',
    value=4,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_IGQUZTXI = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_IGQUZTXI',
    value=5,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_IPJEHZXT = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_IPJEHZXT',
    value=6,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_IBHISRTF = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_IBHISRTF',
    value=7,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_AHLHOHQY = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_AHLHOHQY',
    value=8,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_MAPDVIBO = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_MAPDVIBO',
    value=9,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_JFLKUSVU = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_JFLKUSVU',
    value=33,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_XZPNLKQC = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_XZPNLKQC',
    value=32,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_GZQSWWSJ = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_GZQSWWSJ',
    value=31,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_KHINLTNU = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_KHINLTNU',
    value=26,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_AYPUIJOI = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_AYPUIJOI',
    value=27,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_MDSIVZHL = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_MDSIVZHL',
    value=43,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_CFWZRRWE = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_CFWZRRWE',
    value=28,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_JSOPIVBC = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_JSOPIVBC',
    value=38,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_ZSEHHQVS = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_ZSEHHQVS',
    value=30,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_OOWFZASJ = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_OOWFZASJ',
    value=42,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)
NV_R_ORDQEXFV_F_CMIRIWSV_V_USZOAVIE = ValueMetadata(
    name='NV_R_ORDQEXFV_F_CMIRIWSV_V_USZOAVIE',
    value=29,
    field=NV_R_ORDQEXFV_F_CMIRIWSV
)

NV_R_ORDQEXFV_F_NVKIPOLT = FieldMetadata(
    name='NV_R_ORDQEXFV_F_NVKIPOLT',
    msb=11,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_ZOUDCMGL = FieldMetadata(
    name='NV_R_ORDQEXFV_F_ZOUDCMGL',
    msb=11,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_NOARSXQM = FieldMetadata(
    name='NV_R_ORDQEXFV_F_NOARSXQM',
    msb=11,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_OEUAITQM = FieldMetadata(
    name='NV_R_ORDQEXFV_F_OEUAITQM',
    msb=11,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_RBGFPZWZ = FieldMetadata(
    name='NV_R_ORDQEXFV_F_RBGFPZWZ',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_IMJFHBST = FieldMetadata(
    name='NV_R_ORDQEXFV_F_IMJFHBST',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_HYLTFYZK = FieldMetadata(
    name='NV_R_ORDQEXFV_F_HYLTFYZK',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_HBTOFJFB = FieldMetadata(
    name='NV_R_ORDQEXFV_F_HBTOFJFB',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_HHGTSGOX = FieldMetadata(
    name='NV_R_ORDQEXFV_F_HHGTSGOX',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_NBECFVTR = FieldMetadata(
    name='NV_R_ORDQEXFV_F_NBECFVTR',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_CPXQSYZB = FieldMetadata(
    name='NV_R_ORDQEXFV_F_CPXQSYZB',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_MBJISBNZ = FieldMetadata(
    name='NV_R_ORDQEXFV_F_MBJISBNZ',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_PDZWCDXW = FieldMetadata(
    name='NV_R_ORDQEXFV_F_PDZWCDXW',
    msb=9,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_OMCFLBAL = FieldMetadata(
    name='NV_R_ORDQEXFV_F_OMCFLBAL',
    msb=9,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_NKLXJYMD = FieldMetadata(
    name='NV_R_ORDQEXFV_F_NKLXJYMD',
    msb=9,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_IHQOCTPV = FieldMetadata(
    name='NV_R_ORDQEXFV_F_IHQOCTPV',
    msb=9,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_YJVWXNNE = FieldMetadata(
    name='NV_R_ORDQEXFV_F_YJVWXNNE',
    msb=9,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_VHLUWKFL = FieldMetadata(
    name='NV_R_ORDQEXFV_F_VHLUWKFL',
    msb=9,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_ZJAXCAQG = FieldMetadata(
    name='NV_R_ORDQEXFV_F_ZJAXCAQG',
    msb=8,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_OBEWYHPX = FieldMetadata(
    name='NV_R_ORDQEXFV_F_OBEWYHPX',
    msb=8,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_IRSEKWPQ = FieldMetadata(
    name='NV_R_ORDQEXFV_F_IRSEKWPQ',
    msb=16,
    lsb=9,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_VCXMQJEV = FieldMetadata(
    name='NV_R_ORDQEXFV_F_VCXMQJEV',
    msb=11,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_JXKCEYNM = FieldMetadata(
    name='NV_R_ORDQEXFV_F_JXKCEYNM',
    msb=8,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_DURSWDLP = FieldMetadata(
    name='NV_R_ORDQEXFV_F_DURSWDLP',
    msb=8,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_BJBZERKJ = FieldMetadata(
    name='NV_R_ORDQEXFV_F_BJBZERKJ',
    msb=8,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_MVPRIXOL = FieldMetadata(
    name='NV_R_ORDQEXFV_F_MVPRIXOL',
    msb=15,
    lsb=10,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_TDCDFVOD = FieldMetadata(
    name='NV_R_ORDQEXFV_F_TDCDFVOD',
    msb=9,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_XOCPWSRT = FieldMetadata(
    name='NV_R_ORDQEXFV_F_XOCPWSRT',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_INSVKLRN = FieldMetadata(
    name='NV_R_ORDQEXFV_F_INSVKLRN',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_NQFZWAHO = FieldMetadata(
    name='NV_R_ORDQEXFV_F_NQFZWAHO',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_DSQEGPFA = FieldMetadata(
    name='NV_R_ORDQEXFV_F_DSQEGPFA',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_IKVTHDBZ = FieldMetadata(
    name='NV_R_ORDQEXFV_F_IKVTHDBZ',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_CIWVBJVE = FieldMetadata(
    name='NV_R_ORDQEXFV_F_CIWVBJVE',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_LWLGMGAQ = FieldMetadata(
    name='NV_R_ORDQEXFV_F_LWLGMGAQ',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_OWIZHHHO = FieldMetadata(
    name='NV_R_ORDQEXFV_F_OWIZHHHO',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_MBHCYZPH = FieldMetadata(
    name='NV_R_ORDQEXFV_F_MBHCYZPH',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_YVRYVWIH = FieldMetadata(
    name='NV_R_ORDQEXFV_F_YVRYVWIH',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_UFKZEDCN = FieldMetadata(
    name='NV_R_ORDQEXFV_F_UFKZEDCN',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_QPRFLYFC = FieldMetadata(
    name='NV_R_ORDQEXFV_F_QPRFLYFC',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_KRYCNSZJ = FieldMetadata(
    name='NV_R_ORDQEXFV_F_KRYCNSZJ',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_GDDJPIHT = FieldMetadata(
    name='NV_R_ORDQEXFV_F_GDDJPIHT',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_PYWRHSKK = FieldMetadata(
    name='NV_R_ORDQEXFV_F_PYWRHSKK',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_SMFYIMNS = FieldMetadata(
    name='NV_R_ORDQEXFV_F_SMFYIMNS',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_HDYAZFOA = FieldMetadata(
    name='NV_R_ORDQEXFV_F_HDYAZFOA',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_UHBEOVYM = FieldMetadata(
    name='NV_R_ORDQEXFV_F_UHBEOVYM',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_ZICNZMHT = FieldMetadata(
    name='NV_R_ORDQEXFV_F_ZICNZMHT',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_ZJCFMQNK = FieldMetadata(
    name='NV_R_ORDQEXFV_F_ZJCFMQNK',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_DEBPYEVL = FieldMetadata(
    name='NV_R_ORDQEXFV_F_DEBPYEVL',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_AHTIHXXI = FieldMetadata(
    name='NV_R_ORDQEXFV_F_AHTIHXXI',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_RHTNASDF = FieldMetadata(
    name='NV_R_ORDQEXFV_F_RHTNASDF',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_DVLAWAYM = FieldMetadata(
    name='NV_R_ORDQEXFV_F_DVLAWAYM',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_IHLFYVUA = FieldMetadata(
    name='NV_R_ORDQEXFV_F_IHLFYVUA',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_VJZXXNRN = FieldMetadata(
    name='NV_R_ORDQEXFV_F_VJZXXNRN',
    msb=7,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_WPOKVHWB = FieldMetadata(
    name='NV_R_ORDQEXFV_F_WPOKVHWB',
    msb=11,
    lsb=4,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_QAWEVVQS = FieldMetadata(
    name='NV_R_ORDQEXFV_F_QAWEVVQS',
    msb=3,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_ECKYQVQO = FieldMetadata(
    name='NV_R_ORDQEXFV_F_ECKYQVQO',
    msb=11,
    lsb=4,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_AUKLBKOY = FieldMetadata(
    name='NV_R_ORDQEXFV_F_AUKLBKOY',
    msb=3,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_AZTHXCVL = FieldMetadata(
    name='NV_R_ORDQEXFV_F_AZTHXCVL',
    msb=11,
    lsb=4,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_FWIQTCRQ = FieldMetadata(
    name='NV_R_ORDQEXFV_F_FWIQTCRQ',
    msb=3,
    lsb=0,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_OFGCIMFI = FieldMetadata(
    name='NV_R_ORDQEXFV_F_OFGCIMFI',
    msb=31,
    lsb=30,
    register=NV_R_ORDQEXFV
)

NV_R_ORDQEXFV_F_OFGCIMFI_V_FQNJWQHN = ValueMetadata(
    name='NV_R_ORDQEXFV_F_OFGCIMFI_V_FQNJWQHN',
    value=2,
    field=NV_R_ORDQEXFV_F_OFGCIMFI
)
NV_R_ORDQEXFV_F_OFGCIMFI_V_CMVOBGUP = ValueMetadata(
    name='NV_R_ORDQEXFV_F_OFGCIMFI_V_CMVOBGUP',
    value=0,
    field=NV_R_ORDQEXFV_F_OFGCIMFI
)
NV_R_ORDQEXFV_F_OFGCIMFI_V_TVGHFSAG = ValueMetadata(
    name='NV_R_ORDQEXFV_F_OFGCIMFI_V_TVGHFSAG',
    value=1,
    field=NV_R_ORDQEXFV_F_OFGCIMFI
)

NV_R_LTFLNIUJ = RegisterMetadata(
    name='NV_R_LTFLNIUJ',
    address=0x12f4,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_LTFLNIUJ_F_CTBCFRUB = FieldMetadata(
    name='NV_R_LTFLNIUJ_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_LTFLNIUJ
)

NV_R_LTFLNIUJ_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LTFLNIUJ_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_LTFLNIUJ_F_CTBCFRUB
)

NV_R_LTFLNIUJ_F_QVYAARWP = FieldMetadata(
    name='NV_R_LTFLNIUJ_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_LTFLNIUJ
)

NV_R_LTFLNIUJ_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_LTFLNIUJ_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_LTFLNIUJ_F_QVYAARWP
)

NV_R_QBHFYLFD = RegisterMetadata(
    name='NV_R_QBHFYLFD',
    address=0x12f0,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_QBHFYLFD_F_KWDQPURW = FieldMetadata(
    name='NV_R_QBHFYLFD_F_KWDQPURW',
    msb=5,
    lsb=5,
    register=NV_R_QBHFYLFD
)

NV_R_QBHFYLFD_F_KWDQPURW_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QBHFYLFD_F_KWDQPURW_V_DOGBFDTH',
    value=0,
    field=NV_R_QBHFYLFD_F_KWDQPURW
)
NV_R_QBHFYLFD_F_KWDQPURW_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QBHFYLFD_F_KWDQPURW_V_PIUJQBKV',
    value=1,
    field=NV_R_QBHFYLFD_F_KWDQPURW
)

NV_R_QBHFYLFD_F_ELAIAXBC = FieldMetadata(
    name='NV_R_QBHFYLFD_F_ELAIAXBC',
    msb=1,
    lsb=1,
    register=NV_R_QBHFYLFD
)

NV_R_QBHFYLFD_F_ELAIAXBC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QBHFYLFD_F_ELAIAXBC_V_DOGBFDTH',
    value=0,
    field=NV_R_QBHFYLFD_F_ELAIAXBC
)
NV_R_QBHFYLFD_F_ELAIAXBC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QBHFYLFD_F_ELAIAXBC_V_PIUJQBKV',
    value=1,
    field=NV_R_QBHFYLFD_F_ELAIAXBC
)

NV_R_QBHFYLFD_F_OJXVVOEC = FieldMetadata(
    name='NV_R_QBHFYLFD_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_QBHFYLFD
)

NV_R_QBHFYLFD_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QBHFYLFD_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_QBHFYLFD_F_OJXVVOEC
)
NV_R_QBHFYLFD_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QBHFYLFD_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_QBHFYLFD_F_OJXVVOEC
)

NV_R_QBHFYLFD_F_PBAFTNZG = FieldMetadata(
    name='NV_R_QBHFYLFD_F_PBAFTNZG',
    msb=3,
    lsb=3,
    register=NV_R_QBHFYLFD
)

NV_R_QBHFYLFD_F_PBAFTNZG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QBHFYLFD_F_PBAFTNZG_V_DOGBFDTH',
    value=0,
    field=NV_R_QBHFYLFD_F_PBAFTNZG
)
NV_R_QBHFYLFD_F_PBAFTNZG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QBHFYLFD_F_PBAFTNZG_V_PIUJQBKV',
    value=1,
    field=NV_R_QBHFYLFD_F_PBAFTNZG
)

NV_R_QBHFYLFD_F_VEOHPHHB = FieldMetadata(
    name='NV_R_QBHFYLFD_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_QBHFYLFD
)

NV_R_QBHFYLFD_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QBHFYLFD_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_QBHFYLFD_F_VEOHPHHB
)
NV_R_QBHFYLFD_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QBHFYLFD_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_QBHFYLFD_F_VEOHPHHB
)

NV_R_QBHFYLFD_F_KVYOKDXN = FieldMetadata(
    name='NV_R_QBHFYLFD_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_QBHFYLFD
)

NV_R_QBHFYLFD_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_QBHFYLFD_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_QBHFYLFD_F_KVYOKDXN
)
NV_R_QBHFYLFD_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_QBHFYLFD_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_QBHFYLFD_F_KVYOKDXN
)
NV_R_QBHFYLFD_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_QBHFYLFD_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_QBHFYLFD_F_KVYOKDXN
)

NV_R_QBHFYLFD_F_IOJLTETI = FieldMetadata(
    name='NV_R_QBHFYLFD_F_IOJLTETI',
    msb=4,
    lsb=4,
    register=NV_R_QBHFYLFD
)

NV_R_QBHFYLFD_F_IOJLTETI_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QBHFYLFD_F_IOJLTETI_V_DOGBFDTH',
    value=0,
    field=NV_R_QBHFYLFD_F_IOJLTETI
)
NV_R_QBHFYLFD_F_IOJLTETI_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QBHFYLFD_F_IOJLTETI_V_PIUJQBKV',
    value=1,
    field=NV_R_QBHFYLFD_F_IOJLTETI
)

NV_R_QBHFYLFD_F_TZSQEAIB = FieldMetadata(
    name='NV_R_QBHFYLFD_F_TZSQEAIB',
    msb=0,
    lsb=0,
    register=NV_R_QBHFYLFD
)

NV_R_QBHFYLFD_F_TZSQEAIB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QBHFYLFD_F_TZSQEAIB_V_DOGBFDTH',
    value=0,
    field=NV_R_QBHFYLFD_F_TZSQEAIB
)
NV_R_QBHFYLFD_F_TZSQEAIB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QBHFYLFD_F_TZSQEAIB_V_PIUJQBKV',
    value=1,
    field=NV_R_QBHFYLFD_F_TZSQEAIB
)

NV_R_QBHFYLFD_F_WQYWMRVP = FieldMetadata(
    name='NV_R_QBHFYLFD_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_QBHFYLFD
)

NV_R_QBHFYLFD_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QBHFYLFD_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_QBHFYLFD_F_WQYWMRVP
)
NV_R_QBHFYLFD_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QBHFYLFD_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_QBHFYLFD_F_WQYWMRVP
)

NV_R_QBHFYLFD_F_PUPDHQBL = FieldMetadata(
    name='NV_R_QBHFYLFD_F_PUPDHQBL',
    msb=2,
    lsb=2,
    register=NV_R_QBHFYLFD
)

NV_R_QBHFYLFD_F_PUPDHQBL_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QBHFYLFD_F_PUPDHQBL_V_DOGBFDTH',
    value=0,
    field=NV_R_QBHFYLFD_F_PUPDHQBL
)
NV_R_QBHFYLFD_F_PUPDHQBL_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QBHFYLFD_F_PUPDHQBL_V_PIUJQBKV',
    value=1,
    field=NV_R_QBHFYLFD_F_PUPDHQBL
)

NV_R_QBHFYLFD_F_MLEOLJIS = FieldMetadata(
    name='NV_R_QBHFYLFD_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_QBHFYLFD
)

NV_R_QBHFYLFD_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_QBHFYLFD_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_QBHFYLFD_F_MLEOLJIS
)
NV_R_QBHFYLFD_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_QBHFYLFD_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_QBHFYLFD_F_MLEOLJIS
)

NV_R_UVVXSNLT = RegisterMetadata(
    name='NV_R_UVVXSNLT',
    address=0x12f8,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_UVVXSNLT_F_CTBCFRUB = FieldMetadata(
    name='NV_R_UVVXSNLT_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_UVVXSNLT
)

NV_R_UVVXSNLT_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_UVVXSNLT_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_UVVXSNLT_F_CTBCFRUB
)

NV_R_UVVXSNLT_F_QVYAARWP = FieldMetadata(
    name='NV_R_UVVXSNLT_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_UVVXSNLT
)

NV_R_UVVXSNLT_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_UVVXSNLT_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_UVVXSNLT_F_QVYAARWP
)

NV_R_WFDUZVSD = RegisterMetadata(
    name='NV_R_WFDUZVSD',
    address=0x320,
    zero_based=True
)

NV_R_WFDUZVSD_F_ZIGJNGSD = FieldMetadata(
    name='NV_R_WFDUZVSD_F_ZIGJNGSD',
    msb=22,
    lsb=0,
    register=NV_R_WFDUZVSD
)

NV_R_WFDUZVSD_F_SEXGFUSZ = FieldMetadata(
    name='NV_R_WFDUZVSD_F_SEXGFUSZ',
    msb=21,
    lsb=20,
    register=NV_R_WFDUZVSD
)

NV_R_WFDUZVSD_F_GNCFTOVU = FieldMetadata(
    name='NV_R_WFDUZVSD_F_GNCFTOVU',
    msb=2,
    lsb=0,
    register=NV_R_WFDUZVSD
)

NV_R_WFDUZVSD_F_WDHNBQNK = FieldMetadata(
    name='NV_R_WFDUZVSD_F_WDHNBQNK',
    msb=19,
    lsb=8,
    register=NV_R_WFDUZVSD
)

NV_R_WFDUZVSD_F_HMBHUSZP = FieldMetadata(
    name='NV_R_WFDUZVSD_F_HMBHUSZP',
    msb=22,
    lsb=22,
    register=NV_R_WFDUZVSD
)

NV_R_WFDUZVSD_F_FEERLBNG = FieldMetadata(
    name='NV_R_WFDUZVSD_F_FEERLBNG',
    msb=7,
    lsb=3,
    register=NV_R_WFDUZVSD
)

NV_R_WFDUZVSD_F_EXMOKFEV = FieldMetadata(
    name='NV_R_WFDUZVSD_F_EXMOKFEV',
    msb=7,
    lsb=6,
    register=NV_R_WFDUZVSD
)

NV_R_NSYQFIAP = RegisterMetadata(
    name='NV_R_NSYQFIAP',
    address=0x2fc,
    zero_based=True
)

NV_R_NSYQFIAP_F_OJYQBJXU = FieldMetadata(
    name='NV_R_NSYQFIAP_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_SPGCMIVK = FieldMetadata(
    name='NV_R_NSYQFIAP_F_SPGCMIVK',
    msb=21,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_CMIRIWSV = FieldMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV',
    msb=29,
    lsb=22,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_CMIRIWSV_V_IZNDWFDP = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_IZNDWFDP',
    value=8,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_JSAHBBTM = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_JSAHBBTM',
    value=9,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_SEBMZEME = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_SEBMZEME',
    value=10,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_DTQCFPNA = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_DTQCFPNA',
    value=11,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_KYVQOIDU = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_KYVQOIDU',
    value=0,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_GHVVYIOF = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_GHVVYIOF',
    value=1,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_JGXXVPER = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_JGXXVPER',
    value=2,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_OKQHAWBB = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_OKQHAWBB',
    value=3,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_IGVRDHSA = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_IGVRDHSA',
    value=4,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_BJJBUDSL = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_BJJBUDSL',
    value=5,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_NRGNBBPQ = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_NRGNBBPQ',
    value=6,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_EKNISTBR = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_EKNISTBR',
    value=7,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_PGSYHHVE = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_PGSYHHVE',
    value=13,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_UXJGURAX = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_UXJGURAX',
    value=12,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_QWWBEQMS = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_QWWBEQMS',
    value=0,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_NBBWYRIG = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_NBBWYRIG',
    value=1,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_SWVVTNLK = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_SWVVTNLK',
    value=2,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_OTSWZFCR = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_OTSWZFCR',
    value=3,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_CSRKWDGO = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_CSRKWDGO',
    value=39,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_ORGICIHS = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_ORGICIHS',
    value=40,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_EYIXRFGX = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_EYIXRFGX',
    value=35,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_KFPOQASI = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_KFPOQASI',
    value=34,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_WPWBKHJE = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_WPWBKHJE',
    value=37,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_EEUGWZSG = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_EEUGWZSG',
    value=0,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_QUUGDQPL = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_QUUGDQPL',
    value=1,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_LKELNZMU = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_LKELNZMU',
    value=36,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_RSFATZWE = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_RSFATZWE',
    value=41,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_VGDLATGR = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_VGDLATGR',
    value=18,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_AWDMJNRQ = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_AWDMJNRQ',
    value=19,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_GDTZIIUX = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_GDTZIIUX',
    value=20,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_RGDBBIRS = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_RGDBBIRS',
    value=21,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_PYYFFBZF = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_PYYFFBZF',
    value=22,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_ZDAXSZOC = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_ZDAXSZOC',
    value=23,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_ZGKOZVLG = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_ZGKOZVLG',
    value=24,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_LXFZIIVC = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_LXFZIIVC',
    value=25,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_LIYVCEVC = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_LIYVCEVC',
    value=10,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_TMZYYQRE = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_TMZYYQRE',
    value=11,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_LITMRWHR = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_LITMRWHR',
    value=12,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_SUXZYCLN = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_SUXZYCLN',
    value=13,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_JKGEPTXY = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_JKGEPTXY',
    value=14,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_BIJQQCBV = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_BIJQQCBV',
    value=15,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_RCXNOXVH = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_RCXNOXVH',
    value=16,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_UPMZYBHK = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_UPMZYBHK',
    value=17,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_JKIFIDZS = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_JKIFIDZS',
    value=2,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_CWMIEYYA = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_CWMIEYYA',
    value=3,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_PTVPYSBH = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_PTVPYSBH',
    value=4,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_IGQUZTXI = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_IGQUZTXI',
    value=5,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_IPJEHZXT = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_IPJEHZXT',
    value=6,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_IBHISRTF = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_IBHISRTF',
    value=7,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_AHLHOHQY = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_AHLHOHQY',
    value=8,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_MAPDVIBO = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_MAPDVIBO',
    value=9,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_JFLKUSVU = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_JFLKUSVU',
    value=33,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_XZPNLKQC = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_XZPNLKQC',
    value=32,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_GZQSWWSJ = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_GZQSWWSJ',
    value=31,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_KHINLTNU = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_KHINLTNU',
    value=26,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_AYPUIJOI = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_AYPUIJOI',
    value=27,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_MDSIVZHL = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_MDSIVZHL',
    value=43,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_CFWZRRWE = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_CFWZRRWE',
    value=28,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_JSOPIVBC = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_JSOPIVBC',
    value=38,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_ZSEHHQVS = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_ZSEHHQVS',
    value=30,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_OOWFZASJ = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_OOWFZASJ',
    value=42,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)
NV_R_NSYQFIAP_F_CMIRIWSV_V_USZOAVIE = ValueMetadata(
    name='NV_R_NSYQFIAP_F_CMIRIWSV_V_USZOAVIE',
    value=29,
    field=NV_R_NSYQFIAP_F_CMIRIWSV
)

NV_R_NSYQFIAP_F_NVKIPOLT = FieldMetadata(
    name='NV_R_NSYQFIAP_F_NVKIPOLT',
    msb=11,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_ZOUDCMGL = FieldMetadata(
    name='NV_R_NSYQFIAP_F_ZOUDCMGL',
    msb=11,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_NOARSXQM = FieldMetadata(
    name='NV_R_NSYQFIAP_F_NOARSXQM',
    msb=11,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_OEUAITQM = FieldMetadata(
    name='NV_R_NSYQFIAP_F_OEUAITQM',
    msb=11,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_RBGFPZWZ = FieldMetadata(
    name='NV_R_NSYQFIAP_F_RBGFPZWZ',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_IMJFHBST = FieldMetadata(
    name='NV_R_NSYQFIAP_F_IMJFHBST',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_HYLTFYZK = FieldMetadata(
    name='NV_R_NSYQFIAP_F_HYLTFYZK',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_HBTOFJFB = FieldMetadata(
    name='NV_R_NSYQFIAP_F_HBTOFJFB',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_HHGTSGOX = FieldMetadata(
    name='NV_R_NSYQFIAP_F_HHGTSGOX',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_NBECFVTR = FieldMetadata(
    name='NV_R_NSYQFIAP_F_NBECFVTR',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_CPXQSYZB = FieldMetadata(
    name='NV_R_NSYQFIAP_F_CPXQSYZB',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_MBJISBNZ = FieldMetadata(
    name='NV_R_NSYQFIAP_F_MBJISBNZ',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_PDZWCDXW = FieldMetadata(
    name='NV_R_NSYQFIAP_F_PDZWCDXW',
    msb=9,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_OMCFLBAL = FieldMetadata(
    name='NV_R_NSYQFIAP_F_OMCFLBAL',
    msb=9,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_NKLXJYMD = FieldMetadata(
    name='NV_R_NSYQFIAP_F_NKLXJYMD',
    msb=9,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_IHQOCTPV = FieldMetadata(
    name='NV_R_NSYQFIAP_F_IHQOCTPV',
    msb=9,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_YJVWXNNE = FieldMetadata(
    name='NV_R_NSYQFIAP_F_YJVWXNNE',
    msb=9,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_VHLUWKFL = FieldMetadata(
    name='NV_R_NSYQFIAP_F_VHLUWKFL',
    msb=9,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_ZJAXCAQG = FieldMetadata(
    name='NV_R_NSYQFIAP_F_ZJAXCAQG',
    msb=8,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_OBEWYHPX = FieldMetadata(
    name='NV_R_NSYQFIAP_F_OBEWYHPX',
    msb=8,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_IRSEKWPQ = FieldMetadata(
    name='NV_R_NSYQFIAP_F_IRSEKWPQ',
    msb=16,
    lsb=9,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_VCXMQJEV = FieldMetadata(
    name='NV_R_NSYQFIAP_F_VCXMQJEV',
    msb=11,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_JXKCEYNM = FieldMetadata(
    name='NV_R_NSYQFIAP_F_JXKCEYNM',
    msb=8,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_DURSWDLP = FieldMetadata(
    name='NV_R_NSYQFIAP_F_DURSWDLP',
    msb=8,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_BJBZERKJ = FieldMetadata(
    name='NV_R_NSYQFIAP_F_BJBZERKJ',
    msb=8,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_MVPRIXOL = FieldMetadata(
    name='NV_R_NSYQFIAP_F_MVPRIXOL',
    msb=15,
    lsb=10,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_TDCDFVOD = FieldMetadata(
    name='NV_R_NSYQFIAP_F_TDCDFVOD',
    msb=9,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_XOCPWSRT = FieldMetadata(
    name='NV_R_NSYQFIAP_F_XOCPWSRT',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_INSVKLRN = FieldMetadata(
    name='NV_R_NSYQFIAP_F_INSVKLRN',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_NQFZWAHO = FieldMetadata(
    name='NV_R_NSYQFIAP_F_NQFZWAHO',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_DSQEGPFA = FieldMetadata(
    name='NV_R_NSYQFIAP_F_DSQEGPFA',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_IKVTHDBZ = FieldMetadata(
    name='NV_R_NSYQFIAP_F_IKVTHDBZ',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_CIWVBJVE = FieldMetadata(
    name='NV_R_NSYQFIAP_F_CIWVBJVE',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_LWLGMGAQ = FieldMetadata(
    name='NV_R_NSYQFIAP_F_LWLGMGAQ',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_OWIZHHHO = FieldMetadata(
    name='NV_R_NSYQFIAP_F_OWIZHHHO',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_MBHCYZPH = FieldMetadata(
    name='NV_R_NSYQFIAP_F_MBHCYZPH',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_YVRYVWIH = FieldMetadata(
    name='NV_R_NSYQFIAP_F_YVRYVWIH',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_UFKZEDCN = FieldMetadata(
    name='NV_R_NSYQFIAP_F_UFKZEDCN',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_QPRFLYFC = FieldMetadata(
    name='NV_R_NSYQFIAP_F_QPRFLYFC',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_KRYCNSZJ = FieldMetadata(
    name='NV_R_NSYQFIAP_F_KRYCNSZJ',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_GDDJPIHT = FieldMetadata(
    name='NV_R_NSYQFIAP_F_GDDJPIHT',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_PYWRHSKK = FieldMetadata(
    name='NV_R_NSYQFIAP_F_PYWRHSKK',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_SMFYIMNS = FieldMetadata(
    name='NV_R_NSYQFIAP_F_SMFYIMNS',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_HDYAZFOA = FieldMetadata(
    name='NV_R_NSYQFIAP_F_HDYAZFOA',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_UHBEOVYM = FieldMetadata(
    name='NV_R_NSYQFIAP_F_UHBEOVYM',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_ZICNZMHT = FieldMetadata(
    name='NV_R_NSYQFIAP_F_ZICNZMHT',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_ZJCFMQNK = FieldMetadata(
    name='NV_R_NSYQFIAP_F_ZJCFMQNK',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_DEBPYEVL = FieldMetadata(
    name='NV_R_NSYQFIAP_F_DEBPYEVL',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_AHTIHXXI = FieldMetadata(
    name='NV_R_NSYQFIAP_F_AHTIHXXI',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_RHTNASDF = FieldMetadata(
    name='NV_R_NSYQFIAP_F_RHTNASDF',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_DVLAWAYM = FieldMetadata(
    name='NV_R_NSYQFIAP_F_DVLAWAYM',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_IHLFYVUA = FieldMetadata(
    name='NV_R_NSYQFIAP_F_IHLFYVUA',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_VJZXXNRN = FieldMetadata(
    name='NV_R_NSYQFIAP_F_VJZXXNRN',
    msb=7,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_WPOKVHWB = FieldMetadata(
    name='NV_R_NSYQFIAP_F_WPOKVHWB',
    msb=11,
    lsb=4,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_QAWEVVQS = FieldMetadata(
    name='NV_R_NSYQFIAP_F_QAWEVVQS',
    msb=3,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_ECKYQVQO = FieldMetadata(
    name='NV_R_NSYQFIAP_F_ECKYQVQO',
    msb=11,
    lsb=4,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_AUKLBKOY = FieldMetadata(
    name='NV_R_NSYQFIAP_F_AUKLBKOY',
    msb=3,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_AZTHXCVL = FieldMetadata(
    name='NV_R_NSYQFIAP_F_AZTHXCVL',
    msb=11,
    lsb=4,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_FWIQTCRQ = FieldMetadata(
    name='NV_R_NSYQFIAP_F_FWIQTCRQ',
    msb=3,
    lsb=0,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_OFGCIMFI = FieldMetadata(
    name='NV_R_NSYQFIAP_F_OFGCIMFI',
    msb=31,
    lsb=30,
    register=NV_R_NSYQFIAP
)

NV_R_NSYQFIAP_F_OFGCIMFI_V_FQNJWQHN = ValueMetadata(
    name='NV_R_NSYQFIAP_F_OFGCIMFI_V_FQNJWQHN',
    value=2,
    field=NV_R_NSYQFIAP_F_OFGCIMFI
)
NV_R_NSYQFIAP_F_OFGCIMFI_V_CMVOBGUP = ValueMetadata(
    name='NV_R_NSYQFIAP_F_OFGCIMFI_V_CMVOBGUP',
    value=0,
    field=NV_R_NSYQFIAP_F_OFGCIMFI
)
NV_R_NSYQFIAP_F_OFGCIMFI_V_TVGHFSAG = ValueMetadata(
    name='NV_R_NSYQFIAP_F_OFGCIMFI_V_TVGHFSAG',
    value=1,
    field=NV_R_NSYQFIAP_F_OFGCIMFI
)

NV_R_SXQTDICY = RegisterMetadata(
    name='NV_R_SXQTDICY',
    address=0x2f4,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_SXQTDICY_F_CTBCFRUB = FieldMetadata(
    name='NV_R_SXQTDICY_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_SXQTDICY
)

NV_R_SXQTDICY_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_SXQTDICY_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_SXQTDICY_F_CTBCFRUB
)

NV_R_SXQTDICY_F_QVYAARWP = FieldMetadata(
    name='NV_R_SXQTDICY_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_SXQTDICY
)

NV_R_SXQTDICY_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_SXQTDICY_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_SXQTDICY_F_QVYAARWP
)

NV_R_GQOFCLQY = RegisterMetadata(
    name='NV_R_GQOFCLQY',
    address=0x2f0,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_GQOFCLQY_F_KWDQPURW = FieldMetadata(
    name='NV_R_GQOFCLQY_F_KWDQPURW',
    msb=5,
    lsb=5,
    register=NV_R_GQOFCLQY
)

NV_R_GQOFCLQY_F_KWDQPURW_V_DOGBFDTH = ValueMetadata(
    name='NV_R_GQOFCLQY_F_KWDQPURW_V_DOGBFDTH',
    value=0,
    field=NV_R_GQOFCLQY_F_KWDQPURW
)
NV_R_GQOFCLQY_F_KWDQPURW_V_PIUJQBKV = ValueMetadata(
    name='NV_R_GQOFCLQY_F_KWDQPURW_V_PIUJQBKV',
    value=1,
    field=NV_R_GQOFCLQY_F_KWDQPURW
)

NV_R_GQOFCLQY_F_ELAIAXBC = FieldMetadata(
    name='NV_R_GQOFCLQY_F_ELAIAXBC',
    msb=1,
    lsb=1,
    register=NV_R_GQOFCLQY
)

NV_R_GQOFCLQY_F_ELAIAXBC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_GQOFCLQY_F_ELAIAXBC_V_DOGBFDTH',
    value=0,
    field=NV_R_GQOFCLQY_F_ELAIAXBC
)
NV_R_GQOFCLQY_F_ELAIAXBC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_GQOFCLQY_F_ELAIAXBC_V_PIUJQBKV',
    value=1,
    field=NV_R_GQOFCLQY_F_ELAIAXBC
)

NV_R_GQOFCLQY_F_OJXVVOEC = FieldMetadata(
    name='NV_R_GQOFCLQY_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_GQOFCLQY
)

NV_R_GQOFCLQY_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_GQOFCLQY_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_GQOFCLQY_F_OJXVVOEC
)
NV_R_GQOFCLQY_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_GQOFCLQY_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_GQOFCLQY_F_OJXVVOEC
)

NV_R_GQOFCLQY_F_PBAFTNZG = FieldMetadata(
    name='NV_R_GQOFCLQY_F_PBAFTNZG',
    msb=3,
    lsb=3,
    register=NV_R_GQOFCLQY
)

NV_R_GQOFCLQY_F_PBAFTNZG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_GQOFCLQY_F_PBAFTNZG_V_DOGBFDTH',
    value=0,
    field=NV_R_GQOFCLQY_F_PBAFTNZG
)
NV_R_GQOFCLQY_F_PBAFTNZG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_GQOFCLQY_F_PBAFTNZG_V_PIUJQBKV',
    value=1,
    field=NV_R_GQOFCLQY_F_PBAFTNZG
)

NV_R_GQOFCLQY_F_VEOHPHHB = FieldMetadata(
    name='NV_R_GQOFCLQY_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_GQOFCLQY
)

NV_R_GQOFCLQY_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_GQOFCLQY_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_GQOFCLQY_F_VEOHPHHB
)
NV_R_GQOFCLQY_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_GQOFCLQY_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_GQOFCLQY_F_VEOHPHHB
)

NV_R_GQOFCLQY_F_KVYOKDXN = FieldMetadata(
    name='NV_R_GQOFCLQY_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_GQOFCLQY
)

NV_R_GQOFCLQY_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_GQOFCLQY_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_GQOFCLQY_F_KVYOKDXN
)
NV_R_GQOFCLQY_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_GQOFCLQY_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_GQOFCLQY_F_KVYOKDXN
)
NV_R_GQOFCLQY_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_GQOFCLQY_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_GQOFCLQY_F_KVYOKDXN
)

NV_R_GQOFCLQY_F_IOJLTETI = FieldMetadata(
    name='NV_R_GQOFCLQY_F_IOJLTETI',
    msb=4,
    lsb=4,
    register=NV_R_GQOFCLQY
)

NV_R_GQOFCLQY_F_IOJLTETI_V_DOGBFDTH = ValueMetadata(
    name='NV_R_GQOFCLQY_F_IOJLTETI_V_DOGBFDTH',
    value=0,
    field=NV_R_GQOFCLQY_F_IOJLTETI
)
NV_R_GQOFCLQY_F_IOJLTETI_V_PIUJQBKV = ValueMetadata(
    name='NV_R_GQOFCLQY_F_IOJLTETI_V_PIUJQBKV',
    value=1,
    field=NV_R_GQOFCLQY_F_IOJLTETI
)

NV_R_GQOFCLQY_F_TZSQEAIB = FieldMetadata(
    name='NV_R_GQOFCLQY_F_TZSQEAIB',
    msb=0,
    lsb=0,
    register=NV_R_GQOFCLQY
)

NV_R_GQOFCLQY_F_TZSQEAIB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_GQOFCLQY_F_TZSQEAIB_V_DOGBFDTH',
    value=0,
    field=NV_R_GQOFCLQY_F_TZSQEAIB
)
NV_R_GQOFCLQY_F_TZSQEAIB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_GQOFCLQY_F_TZSQEAIB_V_PIUJQBKV',
    value=1,
    field=NV_R_GQOFCLQY_F_TZSQEAIB
)

NV_R_GQOFCLQY_F_WQYWMRVP = FieldMetadata(
    name='NV_R_GQOFCLQY_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_GQOFCLQY
)

NV_R_GQOFCLQY_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_GQOFCLQY_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_GQOFCLQY_F_WQYWMRVP
)
NV_R_GQOFCLQY_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_GQOFCLQY_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_GQOFCLQY_F_WQYWMRVP
)

NV_R_GQOFCLQY_F_PUPDHQBL = FieldMetadata(
    name='NV_R_GQOFCLQY_F_PUPDHQBL',
    msb=2,
    lsb=2,
    register=NV_R_GQOFCLQY
)

NV_R_GQOFCLQY_F_PUPDHQBL_V_DOGBFDTH = ValueMetadata(
    name='NV_R_GQOFCLQY_F_PUPDHQBL_V_DOGBFDTH',
    value=0,
    field=NV_R_GQOFCLQY_F_PUPDHQBL
)
NV_R_GQOFCLQY_F_PUPDHQBL_V_PIUJQBKV = ValueMetadata(
    name='NV_R_GQOFCLQY_F_PUPDHQBL_V_PIUJQBKV',
    value=1,
    field=NV_R_GQOFCLQY_F_PUPDHQBL
)

NV_R_GQOFCLQY_F_MLEOLJIS = FieldMetadata(
    name='NV_R_GQOFCLQY_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_GQOFCLQY
)

NV_R_GQOFCLQY_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_GQOFCLQY_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_GQOFCLQY_F_MLEOLJIS
)
NV_R_GQOFCLQY_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_GQOFCLQY_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_GQOFCLQY_F_MLEOLJIS
)

NV_R_XBNBBKSI = RegisterMetadata(
    name='NV_R_XBNBBKSI',
    address=0x2f8,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_XBNBBKSI_F_CTBCFRUB = FieldMetadata(
    name='NV_R_XBNBBKSI_F_CTBCFRUB',
    msb=15,
    lsb=0,
    register=NV_R_XBNBBKSI
)

NV_R_XBNBBKSI_F_CTBCFRUB_V_GQHYEKHS = ValueMetadata(
    name='NV_R_XBNBBKSI_F_CTBCFRUB_V_GQHYEKHS',
    value=0,
    field=NV_R_XBNBBKSI_F_CTBCFRUB
)

NV_R_XBNBBKSI_F_QVYAARWP = FieldMetadata(
    name='NV_R_XBNBBKSI_F_QVYAARWP',
    msb=31,
    lsb=16,
    register=NV_R_XBNBBKSI
)

NV_R_XBNBBKSI_F_QVYAARWP_V_GQHYEKHS = ValueMetadata(
    name='NV_R_XBNBBKSI_F_QVYAARWP_V_GQHYEKHS',
    value=0,
    field=NV_R_XBNBBKSI_F_QVYAARWP
)

