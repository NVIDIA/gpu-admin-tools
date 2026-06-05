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
from gpu.regs.gh100.ltc_int import (
    NV_R_IDQJZRRR,
    NV_R_IDQJZRRR_F_ZIGJNGSD,
    NV_R_IDQJZRRR_F_SEXGFUSZ,
    NV_R_IDQJZRRR_F_GNCFTOVU,
    NV_R_IDQJZRRR_F_WDHNBQNK,
    NV_R_IDQJZRRR_F_HMBHUSZP,
    NV_R_IDQJZRRR_F_FEERLBNG,
    NV_R_IDQJZRRR_F_EXMOKFEV,
    NV_R_ETBBRWEH,
    NV_R_ETBBRWEH_F_CTBCFRUB,
    NV_R_ETBBRWEH_F_CTBCFRUB_V_GQHYEKHS,
    NV_R_ETBBRWEH_F_QVYAARWP,
    NV_R_ETBBRWEH_F_QVYAARWP_V_GQHYEKHS,
    NV_R_QBXKTXZY,
    NV_R_QBXKTXZY_F_CTBCFRUB,
    NV_R_QBXKTXZY_F_CTBCFRUB_V_GQHYEKHS,
    NV_R_QBXKTXZY_F_QVYAARWP,
    NV_R_QBXKTXZY_F_QVYAARWP_V_GQHYEKHS,
)

# Register definitions
NV_R_UOFHMCOW = RegisterMetadata(
    name='NV_R_UOFHMCOW',
    address=0x4fc,
    zero_based=True
)

NV_R_UOFHMCOW_F_OJYQBJXU = FieldMetadata(
    name='NV_R_UOFHMCOW_F_OJYQBJXU',
    msb=31,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_SPGCMIVK = FieldMetadata(
    name='NV_R_UOFHMCOW_F_SPGCMIVK',
    msb=21,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_CMIRIWSV = FieldMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV',
    msb=29,
    lsb=22,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_CMIRIWSV_V_IZNDWFDP = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_IZNDWFDP',
    value=8,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_JSAHBBTM = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_JSAHBBTM',
    value=9,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_SEBMZEME = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_SEBMZEME',
    value=10,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_DTQCFPNA = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_DTQCFPNA',
    value=11,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_KYVQOIDU = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_KYVQOIDU',
    value=0,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_GHVVYIOF = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_GHVVYIOF',
    value=1,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_JGXXVPER = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_JGXXVPER',
    value=2,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_OKQHAWBB = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_OKQHAWBB',
    value=3,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_IGVRDHSA = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_IGVRDHSA',
    value=4,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_BJJBUDSL = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_BJJBUDSL',
    value=5,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_NRGNBBPQ = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_NRGNBBPQ',
    value=6,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_EKNISTBR = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_EKNISTBR',
    value=7,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_PGSYHHVE = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_PGSYHHVE',
    value=13,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_UXJGURAX = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_UXJGURAX',
    value=12,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_QWWBEQMS = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_QWWBEQMS',
    value=0,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_NBBWYRIG = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_NBBWYRIG',
    value=1,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_SWVVTNLK = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_SWVVTNLK',
    value=2,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_OTSWZFCR = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_OTSWZFCR',
    value=3,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_CSRKWDGO = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_CSRKWDGO',
    value=39,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_ORGICIHS = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_ORGICIHS',
    value=40,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_EYIXRFGX = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_EYIXRFGX',
    value=35,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_KFPOQASI = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_KFPOQASI',
    value=34,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_WPWBKHJE = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_WPWBKHJE',
    value=37,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_EEUGWZSG = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_EEUGWZSG',
    value=0,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_QUUGDQPL = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_QUUGDQPL',
    value=1,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_LKELNZMU = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_LKELNZMU',
    value=36,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_RSFATZWE = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_RSFATZWE',
    value=41,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_VGDLATGR = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_VGDLATGR',
    value=18,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_AWDMJNRQ = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_AWDMJNRQ',
    value=19,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_GDTZIIUX = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_GDTZIIUX',
    value=20,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_RGDBBIRS = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_RGDBBIRS',
    value=21,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_PYYFFBZF = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_PYYFFBZF',
    value=22,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_ZDAXSZOC = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_ZDAXSZOC',
    value=23,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_ZGKOZVLG = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_ZGKOZVLG',
    value=24,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_LXFZIIVC = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_LXFZIIVC',
    value=25,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_LIYVCEVC = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_LIYVCEVC',
    value=10,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_TMZYYQRE = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_TMZYYQRE',
    value=11,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_LITMRWHR = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_LITMRWHR',
    value=12,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_SUXZYCLN = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_SUXZYCLN',
    value=13,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_JKGEPTXY = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_JKGEPTXY',
    value=14,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_BIJQQCBV = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_BIJQQCBV',
    value=15,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_RCXNOXVH = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_RCXNOXVH',
    value=16,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_UPMZYBHK = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_UPMZYBHK',
    value=17,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_JKIFIDZS = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_JKIFIDZS',
    value=2,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_CWMIEYYA = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_CWMIEYYA',
    value=3,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_PTVPYSBH = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_PTVPYSBH',
    value=4,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_IGQUZTXI = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_IGQUZTXI',
    value=5,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_IPJEHZXT = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_IPJEHZXT',
    value=6,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_IBHISRTF = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_IBHISRTF',
    value=7,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_AHLHOHQY = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_AHLHOHQY',
    value=8,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_MAPDVIBO = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_MAPDVIBO',
    value=9,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_JFLKUSVU = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_JFLKUSVU',
    value=33,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_XZPNLKQC = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_XZPNLKQC',
    value=32,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_GZQSWWSJ = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_GZQSWWSJ',
    value=31,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_KHINLTNU = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_KHINLTNU',
    value=26,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_AYPUIJOI = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_AYPUIJOI',
    value=27,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_CFWZRRWE = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_CFWZRRWE',
    value=28,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_JSOPIVBC = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_JSOPIVBC',
    value=38,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_ZSEHHQVS = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_ZSEHHQVS',
    value=30,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_OOWFZASJ = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_OOWFZASJ',
    value=42,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)
NV_R_UOFHMCOW_F_CMIRIWSV_V_USZOAVIE = ValueMetadata(
    name='NV_R_UOFHMCOW_F_CMIRIWSV_V_USZOAVIE',
    value=29,
    field=NV_R_UOFHMCOW_F_CMIRIWSV
)

NV_R_UOFHMCOW_F_NVKIPOLT = FieldMetadata(
    name='NV_R_UOFHMCOW_F_NVKIPOLT',
    msb=11,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_ZOUDCMGL = FieldMetadata(
    name='NV_R_UOFHMCOW_F_ZOUDCMGL',
    msb=11,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_NOARSXQM = FieldMetadata(
    name='NV_R_UOFHMCOW_F_NOARSXQM',
    msb=11,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_OEUAITQM = FieldMetadata(
    name='NV_R_UOFHMCOW_F_OEUAITQM',
    msb=11,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_RBGFPZWZ = FieldMetadata(
    name='NV_R_UOFHMCOW_F_RBGFPZWZ',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_IMJFHBST = FieldMetadata(
    name='NV_R_UOFHMCOW_F_IMJFHBST',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_HYLTFYZK = FieldMetadata(
    name='NV_R_UOFHMCOW_F_HYLTFYZK',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_HBTOFJFB = FieldMetadata(
    name='NV_R_UOFHMCOW_F_HBTOFJFB',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_HHGTSGOX = FieldMetadata(
    name='NV_R_UOFHMCOW_F_HHGTSGOX',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_NBECFVTR = FieldMetadata(
    name='NV_R_UOFHMCOW_F_NBECFVTR',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_CPXQSYZB = FieldMetadata(
    name='NV_R_UOFHMCOW_F_CPXQSYZB',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_MBJISBNZ = FieldMetadata(
    name='NV_R_UOFHMCOW_F_MBJISBNZ',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_PDZWCDXW = FieldMetadata(
    name='NV_R_UOFHMCOW_F_PDZWCDXW',
    msb=9,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_OMCFLBAL = FieldMetadata(
    name='NV_R_UOFHMCOW_F_OMCFLBAL',
    msb=9,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_NKLXJYMD = FieldMetadata(
    name='NV_R_UOFHMCOW_F_NKLXJYMD',
    msb=9,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_IHQOCTPV = FieldMetadata(
    name='NV_R_UOFHMCOW_F_IHQOCTPV',
    msb=9,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_YJVWXNNE = FieldMetadata(
    name='NV_R_UOFHMCOW_F_YJVWXNNE',
    msb=9,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_VHLUWKFL = FieldMetadata(
    name='NV_R_UOFHMCOW_F_VHLUWKFL',
    msb=9,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_ZJAXCAQG = FieldMetadata(
    name='NV_R_UOFHMCOW_F_ZJAXCAQG',
    msb=8,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_OBEWYHPX = FieldMetadata(
    name='NV_R_UOFHMCOW_F_OBEWYHPX',
    msb=8,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_IRSEKWPQ = FieldMetadata(
    name='NV_R_UOFHMCOW_F_IRSEKWPQ',
    msb=16,
    lsb=9,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_VCXMQJEV = FieldMetadata(
    name='NV_R_UOFHMCOW_F_VCXMQJEV',
    msb=11,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_JXKCEYNM = FieldMetadata(
    name='NV_R_UOFHMCOW_F_JXKCEYNM',
    msb=8,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_DURSWDLP = FieldMetadata(
    name='NV_R_UOFHMCOW_F_DURSWDLP',
    msb=8,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_BJBZERKJ = FieldMetadata(
    name='NV_R_UOFHMCOW_F_BJBZERKJ',
    msb=8,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_MVPRIXOL = FieldMetadata(
    name='NV_R_UOFHMCOW_F_MVPRIXOL',
    msb=15,
    lsb=10,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_TDCDFVOD = FieldMetadata(
    name='NV_R_UOFHMCOW_F_TDCDFVOD',
    msb=9,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_XOCPWSRT = FieldMetadata(
    name='NV_R_UOFHMCOW_F_XOCPWSRT',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_INSVKLRN = FieldMetadata(
    name='NV_R_UOFHMCOW_F_INSVKLRN',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_NQFZWAHO = FieldMetadata(
    name='NV_R_UOFHMCOW_F_NQFZWAHO',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_DSQEGPFA = FieldMetadata(
    name='NV_R_UOFHMCOW_F_DSQEGPFA',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_IKVTHDBZ = FieldMetadata(
    name='NV_R_UOFHMCOW_F_IKVTHDBZ',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_CIWVBJVE = FieldMetadata(
    name='NV_R_UOFHMCOW_F_CIWVBJVE',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_LWLGMGAQ = FieldMetadata(
    name='NV_R_UOFHMCOW_F_LWLGMGAQ',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_OWIZHHHO = FieldMetadata(
    name='NV_R_UOFHMCOW_F_OWIZHHHO',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_MBHCYZPH = FieldMetadata(
    name='NV_R_UOFHMCOW_F_MBHCYZPH',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_YVRYVWIH = FieldMetadata(
    name='NV_R_UOFHMCOW_F_YVRYVWIH',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_UFKZEDCN = FieldMetadata(
    name='NV_R_UOFHMCOW_F_UFKZEDCN',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_QPRFLYFC = FieldMetadata(
    name='NV_R_UOFHMCOW_F_QPRFLYFC',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_KRYCNSZJ = FieldMetadata(
    name='NV_R_UOFHMCOW_F_KRYCNSZJ',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_GDDJPIHT = FieldMetadata(
    name='NV_R_UOFHMCOW_F_GDDJPIHT',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_PYWRHSKK = FieldMetadata(
    name='NV_R_UOFHMCOW_F_PYWRHSKK',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_SMFYIMNS = FieldMetadata(
    name='NV_R_UOFHMCOW_F_SMFYIMNS',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_HDYAZFOA = FieldMetadata(
    name='NV_R_UOFHMCOW_F_HDYAZFOA',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_UHBEOVYM = FieldMetadata(
    name='NV_R_UOFHMCOW_F_UHBEOVYM',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_ZICNZMHT = FieldMetadata(
    name='NV_R_UOFHMCOW_F_ZICNZMHT',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_ZJCFMQNK = FieldMetadata(
    name='NV_R_UOFHMCOW_F_ZJCFMQNK',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_DEBPYEVL = FieldMetadata(
    name='NV_R_UOFHMCOW_F_DEBPYEVL',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_AHTIHXXI = FieldMetadata(
    name='NV_R_UOFHMCOW_F_AHTIHXXI',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_RHTNASDF = FieldMetadata(
    name='NV_R_UOFHMCOW_F_RHTNASDF',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_DVLAWAYM = FieldMetadata(
    name='NV_R_UOFHMCOW_F_DVLAWAYM',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_IHLFYVUA = FieldMetadata(
    name='NV_R_UOFHMCOW_F_IHLFYVUA',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_VJZXXNRN = FieldMetadata(
    name='NV_R_UOFHMCOW_F_VJZXXNRN',
    msb=7,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_WPOKVHWB = FieldMetadata(
    name='NV_R_UOFHMCOW_F_WPOKVHWB',
    msb=11,
    lsb=4,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_QAWEVVQS = FieldMetadata(
    name='NV_R_UOFHMCOW_F_QAWEVVQS',
    msb=3,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_ECKYQVQO = FieldMetadata(
    name='NV_R_UOFHMCOW_F_ECKYQVQO',
    msb=11,
    lsb=4,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_AUKLBKOY = FieldMetadata(
    name='NV_R_UOFHMCOW_F_AUKLBKOY',
    msb=3,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_AZTHXCVL = FieldMetadata(
    name='NV_R_UOFHMCOW_F_AZTHXCVL',
    msb=11,
    lsb=4,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_FWIQTCRQ = FieldMetadata(
    name='NV_R_UOFHMCOW_F_FWIQTCRQ',
    msb=3,
    lsb=0,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_OFGCIMFI = FieldMetadata(
    name='NV_R_UOFHMCOW_F_OFGCIMFI',
    msb=31,
    lsb=30,
    register=NV_R_UOFHMCOW
)

NV_R_UOFHMCOW_F_OFGCIMFI_V_FQNJWQHN = ValueMetadata(
    name='NV_R_UOFHMCOW_F_OFGCIMFI_V_FQNJWQHN',
    value=2,
    field=NV_R_UOFHMCOW_F_OFGCIMFI
)
NV_R_UOFHMCOW_F_OFGCIMFI_V_CMVOBGUP = ValueMetadata(
    name='NV_R_UOFHMCOW_F_OFGCIMFI_V_CMVOBGUP',
    value=0,
    field=NV_R_UOFHMCOW_F_OFGCIMFI
)
NV_R_UOFHMCOW_F_OFGCIMFI_V_TVGHFSAG = ValueMetadata(
    name='NV_R_UOFHMCOW_F_OFGCIMFI_V_TVGHFSAG',
    value=1,
    field=NV_R_UOFHMCOW_F_OFGCIMFI
)

NV_R_XJNHUSAW = RegisterMetadata(
    name='NV_R_XJNHUSAW',
    address=0x4f0,
    zero_based=True,
    debug_dump={'tags': ['error', 'ecc'], 'interesting': {'rule': 'nonzero', 'reason': 'ECC status nonzero'}}
)

NV_R_XJNHUSAW_F_KWDQPURW = FieldMetadata(
    name='NV_R_XJNHUSAW_F_KWDQPURW',
    msb=5,
    lsb=5,
    register=NV_R_XJNHUSAW
)

NV_R_XJNHUSAW_F_KWDQPURW_V_DOGBFDTH = ValueMetadata(
    name='NV_R_XJNHUSAW_F_KWDQPURW_V_DOGBFDTH',
    value=0,
    field=NV_R_XJNHUSAW_F_KWDQPURW
)
NV_R_XJNHUSAW_F_KWDQPURW_V_PIUJQBKV = ValueMetadata(
    name='NV_R_XJNHUSAW_F_KWDQPURW_V_PIUJQBKV',
    value=1,
    field=NV_R_XJNHUSAW_F_KWDQPURW
)

NV_R_XJNHUSAW_F_ELAIAXBC = FieldMetadata(
    name='NV_R_XJNHUSAW_F_ELAIAXBC',
    msb=1,
    lsb=1,
    register=NV_R_XJNHUSAW
)

NV_R_XJNHUSAW_F_ELAIAXBC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_XJNHUSAW_F_ELAIAXBC_V_DOGBFDTH',
    value=0,
    field=NV_R_XJNHUSAW_F_ELAIAXBC
)
NV_R_XJNHUSAW_F_ELAIAXBC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_XJNHUSAW_F_ELAIAXBC_V_PIUJQBKV',
    value=1,
    field=NV_R_XJNHUSAW_F_ELAIAXBC
)

NV_R_XJNHUSAW_F_OJXVVOEC = FieldMetadata(
    name='NV_R_XJNHUSAW_F_OJXVVOEC',
    msb=16,
    lsb=16,
    register=NV_R_XJNHUSAW
)

NV_R_XJNHUSAW_F_OJXVVOEC_V_DOGBFDTH = ValueMetadata(
    name='NV_R_XJNHUSAW_F_OJXVVOEC_V_DOGBFDTH',
    value=0,
    field=NV_R_XJNHUSAW_F_OJXVVOEC
)
NV_R_XJNHUSAW_F_OJXVVOEC_V_PIUJQBKV = ValueMetadata(
    name='NV_R_XJNHUSAW_F_OJXVVOEC_V_PIUJQBKV',
    value=1,
    field=NV_R_XJNHUSAW_F_OJXVVOEC
)

NV_R_XJNHUSAW_F_PBAFTNZG = FieldMetadata(
    name='NV_R_XJNHUSAW_F_PBAFTNZG',
    msb=3,
    lsb=3,
    register=NV_R_XJNHUSAW
)

NV_R_XJNHUSAW_F_PBAFTNZG_V_DOGBFDTH = ValueMetadata(
    name='NV_R_XJNHUSAW_F_PBAFTNZG_V_DOGBFDTH',
    value=0,
    field=NV_R_XJNHUSAW_F_PBAFTNZG
)
NV_R_XJNHUSAW_F_PBAFTNZG_V_PIUJQBKV = ValueMetadata(
    name='NV_R_XJNHUSAW_F_PBAFTNZG_V_PIUJQBKV',
    value=1,
    field=NV_R_XJNHUSAW_F_PBAFTNZG
)

NV_R_XJNHUSAW_F_VEOHPHHB = FieldMetadata(
    name='NV_R_XJNHUSAW_F_VEOHPHHB',
    msb=17,
    lsb=17,
    register=NV_R_XJNHUSAW
)

NV_R_XJNHUSAW_F_VEOHPHHB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_XJNHUSAW_F_VEOHPHHB_V_DOGBFDTH',
    value=0,
    field=NV_R_XJNHUSAW_F_VEOHPHHB
)
NV_R_XJNHUSAW_F_VEOHPHHB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_XJNHUSAW_F_VEOHPHHB_V_PIUJQBKV',
    value=1,
    field=NV_R_XJNHUSAW_F_VEOHPHHB
)

NV_R_XJNHUSAW_F_KVYOKDXN = FieldMetadata(
    name='NV_R_XJNHUSAW_F_KVYOKDXN',
    msb=30,
    lsb=30,
    register=NV_R_XJNHUSAW
)

NV_R_XJNHUSAW_F_KVYOKDXN_V_ACRDDIMM = ValueMetadata(
    name='NV_R_XJNHUSAW_F_KVYOKDXN_V_ACRDDIMM',
    value=1,
    field=NV_R_XJNHUSAW_F_KVYOKDXN
)
NV_R_XJNHUSAW_F_KVYOKDXN_V_GQHYEKHS = ValueMetadata(
    name='NV_R_XJNHUSAW_F_KVYOKDXN_V_GQHYEKHS',
    value=0,
    field=NV_R_XJNHUSAW_F_KVYOKDXN
)
NV_R_XJNHUSAW_F_KVYOKDXN_V_CIHFEZTE = ValueMetadata(
    name='NV_R_XJNHUSAW_F_KVYOKDXN_V_CIHFEZTE',
    value=1,
    field=NV_R_XJNHUSAW_F_KVYOKDXN
)

NV_R_XJNHUSAW_F_IOJLTETI = FieldMetadata(
    name='NV_R_XJNHUSAW_F_IOJLTETI',
    msb=4,
    lsb=4,
    register=NV_R_XJNHUSAW
)

NV_R_XJNHUSAW_F_IOJLTETI_V_DOGBFDTH = ValueMetadata(
    name='NV_R_XJNHUSAW_F_IOJLTETI_V_DOGBFDTH',
    value=0,
    field=NV_R_XJNHUSAW_F_IOJLTETI
)
NV_R_XJNHUSAW_F_IOJLTETI_V_PIUJQBKV = ValueMetadata(
    name='NV_R_XJNHUSAW_F_IOJLTETI_V_PIUJQBKV',
    value=1,
    field=NV_R_XJNHUSAW_F_IOJLTETI
)

NV_R_XJNHUSAW_F_TZSQEAIB = FieldMetadata(
    name='NV_R_XJNHUSAW_F_TZSQEAIB',
    msb=0,
    lsb=0,
    register=NV_R_XJNHUSAW
)

NV_R_XJNHUSAW_F_TZSQEAIB_V_DOGBFDTH = ValueMetadata(
    name='NV_R_XJNHUSAW_F_TZSQEAIB_V_DOGBFDTH',
    value=0,
    field=NV_R_XJNHUSAW_F_TZSQEAIB
)
NV_R_XJNHUSAW_F_TZSQEAIB_V_PIUJQBKV = ValueMetadata(
    name='NV_R_XJNHUSAW_F_TZSQEAIB_V_PIUJQBKV',
    value=1,
    field=NV_R_XJNHUSAW_F_TZSQEAIB
)

NV_R_XJNHUSAW_F_WQYWMRVP = FieldMetadata(
    name='NV_R_XJNHUSAW_F_WQYWMRVP',
    msb=18,
    lsb=18,
    register=NV_R_XJNHUSAW
)

NV_R_XJNHUSAW_F_WQYWMRVP_V_DOGBFDTH = ValueMetadata(
    name='NV_R_XJNHUSAW_F_WQYWMRVP_V_DOGBFDTH',
    value=0,
    field=NV_R_XJNHUSAW_F_WQYWMRVP
)
NV_R_XJNHUSAW_F_WQYWMRVP_V_PIUJQBKV = ValueMetadata(
    name='NV_R_XJNHUSAW_F_WQYWMRVP_V_PIUJQBKV',
    value=1,
    field=NV_R_XJNHUSAW_F_WQYWMRVP
)

NV_R_XJNHUSAW_F_PUPDHQBL = FieldMetadata(
    name='NV_R_XJNHUSAW_F_PUPDHQBL',
    msb=2,
    lsb=2,
    register=NV_R_XJNHUSAW
)

NV_R_XJNHUSAW_F_PUPDHQBL_V_DOGBFDTH = ValueMetadata(
    name='NV_R_XJNHUSAW_F_PUPDHQBL_V_DOGBFDTH',
    value=0,
    field=NV_R_XJNHUSAW_F_PUPDHQBL
)
NV_R_XJNHUSAW_F_PUPDHQBL_V_PIUJQBKV = ValueMetadata(
    name='NV_R_XJNHUSAW_F_PUPDHQBL_V_PIUJQBKV',
    value=1,
    field=NV_R_XJNHUSAW_F_PUPDHQBL
)

NV_R_XJNHUSAW_F_MLEOLJIS = FieldMetadata(
    name='NV_R_XJNHUSAW_F_MLEOLJIS',
    msb=19,
    lsb=19,
    register=NV_R_XJNHUSAW
)

NV_R_XJNHUSAW_F_MLEOLJIS_V_DOGBFDTH = ValueMetadata(
    name='NV_R_XJNHUSAW_F_MLEOLJIS_V_DOGBFDTH',
    value=0,
    field=NV_R_XJNHUSAW_F_MLEOLJIS
)
NV_R_XJNHUSAW_F_MLEOLJIS_V_PIUJQBKV = ValueMetadata(
    name='NV_R_XJNHUSAW_F_MLEOLJIS_V_PIUJQBKV',
    value=1,
    field=NV_R_XJNHUSAW_F_MLEOLJIS
)

