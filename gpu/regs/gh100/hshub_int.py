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
NV_R_ARPDSYWJ = RegisterMetadata(
    name='NV_R_ARPDSYWJ',
    address=0x674,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_ARPDSYWJ_F_UIEBWIOP = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_UIEBWIOP',
    msb=0,
    lsb=0,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_FAEUEZAW = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_FAEUEZAW',
    msb=24,
    lsb=24,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_BYVHZPQX = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_BYVHZPQX',
    msb=16,
    lsb=16,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_OVHIEUWZ = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_OVHIEUWZ',
    msb=8,
    lsb=8,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_RVUCMMQW = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_RVUCMMQW',
    msb=1,
    lsb=1,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_UBSXHLEI = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_UBSXHLEI',
    msb=25,
    lsb=25,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_QIAPVCZD = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_QIAPVCZD',
    msb=17,
    lsb=17,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_USAVQDXC = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_USAVQDXC',
    msb=9,
    lsb=9,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_QOSJJLDM = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_QOSJJLDM',
    msb=2,
    lsb=2,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_BHUBJKFV = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_BHUBJKFV',
    msb=26,
    lsb=26,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_ABWDODUN = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_ABWDODUN',
    msb=18,
    lsb=18,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_MHNEZFRQ = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_MHNEZFRQ',
    msb=10,
    lsb=10,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_VYFBOXTE = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_VYFBOXTE',
    msb=3,
    lsb=3,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_MUQTRHDF = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_MUQTRHDF',
    msb=27,
    lsb=27,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_XOXIJWVV = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_XOXIJWVV',
    msb=19,
    lsb=19,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_LTGXRNQG = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_LTGXRNQG',
    msb=11,
    lsb=11,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_EFDTLHDG = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_EFDTLHDG',
    msb=4,
    lsb=4,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_TMQXZMBG = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_TMQXZMBG',
    msb=28,
    lsb=28,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_SJDTJBJN = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_SJDTJBJN',
    msb=20,
    lsb=20,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_PFFGLBNM = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_PFFGLBNM',
    msb=12,
    lsb=12,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_PLZQIACE = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_PLZQIACE',
    msb=5,
    lsb=5,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_WIPYMAKE = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_WIPYMAKE',
    msb=29,
    lsb=29,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_ZKBAFHLI = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_ZKBAFHLI',
    msb=21,
    lsb=21,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_XWFBCFRJ = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_XWFBCFRJ',
    msb=13,
    lsb=13,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_TCLCZDDH = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_TCLCZDDH',
    msb=6,
    lsb=6,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_EKPVCEPD = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_EKPVCEPD',
    msb=30,
    lsb=30,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_DWPQFURP = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_DWPQFURP',
    msb=22,
    lsb=22,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_BUPQWFXM = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_BUPQWFXM',
    msb=14,
    lsb=14,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_FRJXMXJQ = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_FRJXMXJQ',
    msb=7,
    lsb=7,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_ORKXRJUG = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_ORKXRJUG',
    msb=31,
    lsb=31,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_YSWEGNLP = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_YSWEGNLP',
    msb=23,
    lsb=23,
    register=NV_R_ARPDSYWJ
)

NV_R_ARPDSYWJ_F_JEQVOZNZ = FieldMetadata(
    name='NV_R_ARPDSYWJ_F_JEQVOZNZ',
    msb=15,
    lsb=15,
    register=NV_R_ARPDSYWJ
)

NV_R_UKKYSIAF = RegisterMetadata(
    name='NV_R_UKKYSIAF',
    address=0x678,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_UKKYSIAF_F_GHADRSVE = FieldMetadata(
    name='NV_R_UKKYSIAF_F_GHADRSVE',
    msb=16,
    lsb=16,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_VKZSMGDV = FieldMetadata(
    name='NV_R_UKKYSIAF_F_VKZSMGDV',
    msb=0,
    lsb=0,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_RFNOQRKU = FieldMetadata(
    name='NV_R_UKKYSIAF_F_RFNOQRKU',
    msb=24,
    lsb=24,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_MDFSQBEN = FieldMetadata(
    name='NV_R_UKKYSIAF_F_MDFSQBEN',
    msb=17,
    lsb=17,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_BHJSHXVD = FieldMetadata(
    name='NV_R_UKKYSIAF_F_BHJSHXVD',
    msb=1,
    lsb=1,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_WCWOZBSP = FieldMetadata(
    name='NV_R_UKKYSIAF_F_WCWOZBSP',
    msb=25,
    lsb=25,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_FGONZWSR = FieldMetadata(
    name='NV_R_UKKYSIAF_F_FGONZWSR',
    msb=18,
    lsb=18,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_CYZESORM = FieldMetadata(
    name='NV_R_UKKYSIAF_F_CYZESORM',
    msb=2,
    lsb=2,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_LGJGRYRP = FieldMetadata(
    name='NV_R_UKKYSIAF_F_LGJGRYRP',
    msb=26,
    lsb=26,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_GDMIBLMG = FieldMetadata(
    name='NV_R_UKKYSIAF_F_GDMIBLMG',
    msb=19,
    lsb=19,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_KSUINJNF = FieldMetadata(
    name='NV_R_UKKYSIAF_F_KSUINJNF',
    msb=3,
    lsb=3,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_XESRATBZ = FieldMetadata(
    name='NV_R_UKKYSIAF_F_XESRATBZ',
    msb=27,
    lsb=27,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_VNPLXLTZ = FieldMetadata(
    name='NV_R_UKKYSIAF_F_VNPLXLTZ',
    msb=20,
    lsb=20,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_GLRHQTMZ = FieldMetadata(
    name='NV_R_UKKYSIAF_F_GLRHQTMZ',
    msb=4,
    lsb=4,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_SCZMFHQZ = FieldMetadata(
    name='NV_R_UKKYSIAF_F_SCZMFHQZ',
    msb=28,
    lsb=28,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_UGYRKLJM = FieldMetadata(
    name='NV_R_UKKYSIAF_F_UGYRKLJM',
    msb=21,
    lsb=21,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_HCNWPCOD = FieldMetadata(
    name='NV_R_UKKYSIAF_F_HCNWPCOD',
    msb=5,
    lsb=5,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_USJEEPOD = FieldMetadata(
    name='NV_R_UKKYSIAF_F_USJEEPOD',
    msb=29,
    lsb=29,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_MBJACMOM = FieldMetadata(
    name='NV_R_UKKYSIAF_F_MBJACMOM',
    msb=22,
    lsb=22,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_WZBWTNQM = FieldMetadata(
    name='NV_R_UKKYSIAF_F_WZBWTNQM',
    msb=6,
    lsb=6,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_MVLNUVFD = FieldMetadata(
    name='NV_R_UKKYSIAF_F_MVLNUVFD',
    msb=30,
    lsb=30,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_ROWAGQRG = FieldMetadata(
    name='NV_R_UKKYSIAF_F_ROWAGQRG',
    msb=23,
    lsb=23,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_XDTAJKAN = FieldMetadata(
    name='NV_R_UKKYSIAF_F_XDTAJKAN',
    msb=7,
    lsb=7,
    register=NV_R_UKKYSIAF
)

NV_R_UKKYSIAF_F_DCQQYUYD = FieldMetadata(
    name='NV_R_UKKYSIAF_F_DCQQYUYD',
    msb=31,
    lsb=31,
    register=NV_R_UKKYSIAF
)

NV_R_YIGWUDUH = RegisterMetadata(
    name='NV_R_YIGWUDUH',
    address=0x67c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_YIGWUDUH_F_NNFCGTYP = FieldMetadata(
    name='NV_R_YIGWUDUH_F_NNFCGTYP',
    msb=16,
    lsb=16,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_CXEEAICP = FieldMetadata(
    name='NV_R_YIGWUDUH_F_CXEEAICP',
    msb=8,
    lsb=8,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_HPIQFWIX = FieldMetadata(
    name='NV_R_YIGWUDUH_F_HPIQFWIX',
    msb=0,
    lsb=0,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_SQTOOUAI = FieldMetadata(
    name='NV_R_YIGWUDUH_F_SQTOOUAI',
    msb=17,
    lsb=17,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_CEJVDJOX = FieldMetadata(
    name='NV_R_YIGWUDUH_F_CEJVDJOX',
    msb=9,
    lsb=9,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_ZRPTCQAP = FieldMetadata(
    name='NV_R_YIGWUDUH_F_ZRPTCQAP',
    msb=1,
    lsb=1,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_NYBSYXIM = FieldMetadata(
    name='NV_R_YIGWUDUH_F_NYBSYXIM',
    msb=18,
    lsb=18,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_ZBAJVGWI = FieldMetadata(
    name='NV_R_YIGWUDUH_F_ZBAJVGWI',
    msb=10,
    lsb=10,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_UXTSLKKL = FieldMetadata(
    name='NV_R_YIGWUDUH_F_UXTSLKKL',
    msb=2,
    lsb=2,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_QYOXNJZP = FieldMetadata(
    name='NV_R_YIGWUDUH_F_QYOXNJZP',
    msb=19,
    lsb=19,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_RJPDPMUF = FieldMetadata(
    name='NV_R_YIGWUDUH_F_RJPDPMUF',
    msb=11,
    lsb=11,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_UGWQGWBC = FieldMetadata(
    name='NV_R_YIGWUDUH_F_UGWQGWBC',
    msb=3,
    lsb=3,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_ZHZYTJOF = FieldMetadata(
    name='NV_R_YIGWUDUH_F_ZHZYTJOF',
    msb=20,
    lsb=20,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_ULSVLGBK = FieldMetadata(
    name='NV_R_YIGWUDUH_F_ULSVLGBK',
    msb=12,
    lsb=12,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_UIIZGDCH = FieldMetadata(
    name='NV_R_YIGWUDUH_F_UIIZGDCH',
    msb=4,
    lsb=4,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_HUYKHIAQ = FieldMetadata(
    name='NV_R_YIGWUDUH_F_HUYKHIAQ',
    msb=21,
    lsb=21,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_CUCRWJAI = FieldMetadata(
    name='NV_R_YIGWUDUH_F_CUCRWJAI',
    msb=13,
    lsb=13,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_XBEYGJRZ = FieldMetadata(
    name='NV_R_YIGWUDUH_F_XBEYGJRZ',
    msb=5,
    lsb=5,
    register=NV_R_YIGWUDUH
)

NV_R_VHMXVILL = RegisterMetadata(
    name='NV_R_VHMXVILL',
    address=0x684,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_VHMXVILL_F_NHWQWYKB = FieldMetadata(
    name='NV_R_VHMXVILL_F_NHWQWYKB',
    msb=8,
    lsb=8,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_JEOILTCQ = FieldMetadata(
    name='NV_R_VHMXVILL_F_JEOILTCQ',
    msb=0,
    lsb=0,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_VCNJDYQL = FieldMetadata(
    name='NV_R_VHMXVILL_F_VCNJDYQL',
    msb=16,
    lsb=16,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_ZZFUQOWD = FieldMetadata(
    name='NV_R_VHMXVILL_F_ZZFUQOWD',
    msb=9,
    lsb=9,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_BXZRWUSZ = FieldMetadata(
    name='NV_R_VHMXVILL_F_BXZRWUSZ',
    msb=1,
    lsb=1,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_JQYWCMFK = FieldMetadata(
    name='NV_R_VHMXVILL_F_JQYWCMFK',
    msb=17,
    lsb=17,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_OLUWWATW = FieldMetadata(
    name='NV_R_VHMXVILL_F_OLUWWATW',
    msb=10,
    lsb=10,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_XSGCOOMD = FieldMetadata(
    name='NV_R_VHMXVILL_F_XSGCOOMD',
    msb=2,
    lsb=2,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_IUUVKLNG = FieldMetadata(
    name='NV_R_VHMXVILL_F_IUUVKLNG',
    msb=18,
    lsb=18,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_ZVJJIQUB = FieldMetadata(
    name='NV_R_VHMXVILL_F_ZVJJIQUB',
    msb=11,
    lsb=11,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_KOSYPJGT = FieldMetadata(
    name='NV_R_VHMXVILL_F_KOSYPJGT',
    msb=3,
    lsb=3,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_ORBFRNVC = FieldMetadata(
    name='NV_R_VHMXVILL_F_ORBFRNVC',
    msb=19,
    lsb=19,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_TJMANTCF = FieldMetadata(
    name='NV_R_VHMXVILL_F_TJMANTCF',
    msb=12,
    lsb=12,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_ZJRYUNWL = FieldMetadata(
    name='NV_R_VHMXVILL_F_ZJRYUNWL',
    msb=4,
    lsb=4,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_BGXRTXWK = FieldMetadata(
    name='NV_R_VHMXVILL_F_BGXRTXWK',
    msb=20,
    lsb=20,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_ZIDYUYEN = FieldMetadata(
    name='NV_R_VHMXVILL_F_ZIDYUYEN',
    msb=13,
    lsb=13,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_XTLDZEWY = FieldMetadata(
    name='NV_R_VHMXVILL_F_XTLDZEWY',
    msb=5,
    lsb=5,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_QCIHZJMY = FieldMetadata(
    name='NV_R_VHMXVILL_F_QCIHZJMY',
    msb=21,
    lsb=21,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_PLUELXER = FieldMetadata(
    name='NV_R_VHMXVILL_F_PLUELXER',
    msb=14,
    lsb=14,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_IRRZWHUK = FieldMetadata(
    name='NV_R_VHMXVILL_F_IRRZWHUK',
    msb=6,
    lsb=6,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_QELVSHSQ = FieldMetadata(
    name='NV_R_VHMXVILL_F_QELVSHSQ',
    msb=22,
    lsb=22,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_WXTBBGPW = FieldMetadata(
    name='NV_R_VHMXVILL_F_WXTBBGPW',
    msb=15,
    lsb=15,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_HJCIHHCP = FieldMetadata(
    name='NV_R_VHMXVILL_F_HJCIHHCP',
    msb=7,
    lsb=7,
    register=NV_R_VHMXVILL
)

NV_R_VHMXVILL_F_GINEOITN = FieldMetadata(
    name='NV_R_VHMXVILL_F_GINEOITN',
    msb=23,
    lsb=23,
    register=NV_R_VHMXVILL
)

NV_R_CCGPKYYA = RegisterMetadata(
    name='NV_R_CCGPKYYA',
    address=0x688,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_CCGPKYYA_F_BITPXHUP = FieldMetadata(
    name='NV_R_CCGPKYYA_F_BITPXHUP',
    msb=0,
    lsb=0,
    register=NV_R_CCGPKYYA
)

NV_R_CCGPKYYA_F_EJYBLZYR = FieldMetadata(
    name='NV_R_CCGPKYYA_F_EJYBLZYR',
    msb=1,
    lsb=1,
    register=NV_R_CCGPKYYA
)

NV_R_CCGPKYYA_F_SFSHPVKH = FieldMetadata(
    name='NV_R_CCGPKYYA_F_SFSHPVKH',
    msb=2,
    lsb=2,
    register=NV_R_CCGPKYYA
)

NV_R_CCGPKYYA_F_UJAARHKP = FieldMetadata(
    name='NV_R_CCGPKYYA_F_UJAARHKP',
    msb=3,
    lsb=3,
    register=NV_R_CCGPKYYA
)

NV_R_CCGPKYYA_F_NWFCKICN = FieldMetadata(
    name='NV_R_CCGPKYYA_F_NWFCKICN',
    msb=4,
    lsb=4,
    register=NV_R_CCGPKYYA
)

NV_R_CCGPKYYA_F_TPIKPMNQ = FieldMetadata(
    name='NV_R_CCGPKYYA_F_TPIKPMNQ',
    msb=5,
    lsb=5,
    register=NV_R_CCGPKYYA
)

NV_R_IZEOAHGN = RegisterMetadata(
    name='NV_R_IZEOAHGN',
    address=0x1c8,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_IZEOAHGN_F_SEQHVADV = FieldMetadata(
    name='NV_R_IZEOAHGN_F_SEQHVADV',
    msb=6,
    lsb=6,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_UNQBRXUU = FieldMetadata(
    name='NV_R_IZEOAHGN_F_UNQBRXUU',
    msb=2,
    lsb=2,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_CMPLSAEF = FieldMetadata(
    name='NV_R_IZEOAHGN_F_CMPLSAEF',
    msb=5,
    lsb=5,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_KSWZIZDW = FieldMetadata(
    name='NV_R_IZEOAHGN_F_KSWZIZDW',
    msb=0,
    lsb=0,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_ARVINKLE = FieldMetadata(
    name='NV_R_IZEOAHGN_F_ARVINKLE',
    msb=7,
    lsb=7,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_NSOEEUAM = FieldMetadata(
    name='NV_R_IZEOAHGN_F_NSOEEUAM',
    msb=3,
    lsb=3,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_SRZWQELT = FieldMetadata(
    name='NV_R_IZEOAHGN_F_SRZWQELT',
    msb=4,
    lsb=4,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_ZGCQCLLK = FieldMetadata(
    name='NV_R_IZEOAHGN_F_ZGCQCLLK',
    msb=1,
    lsb=1,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_CLUWFOJL = FieldMetadata(
    name='NV_R_IZEOAHGN_F_CLUWFOJL',
    msb=20,
    lsb=20,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_GHWDDLSU = FieldMetadata(
    name='NV_R_IZEOAHGN_F_GHWDDLSU',
    msb=19,
    lsb=19,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_HORPGXDM = FieldMetadata(
    name='NV_R_IZEOAHGN_F_HORPGXDM',
    msb=10,
    lsb=10,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_ZJSPVULX = FieldMetadata(
    name='NV_R_IZEOAHGN_F_ZJSPVULX',
    msb=11,
    lsb=11,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_TSOFRVEJ = FieldMetadata(
    name='NV_R_IZEOAHGN_F_TSOFRVEJ',
    msb=15,
    lsb=15,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_QPXJXFKE = FieldMetadata(
    name='NV_R_IZEOAHGN_F_QPXJXFKE',
    msb=17,
    lsb=17,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_PAQRWLDD = FieldMetadata(
    name='NV_R_IZEOAHGN_F_PAQRWLDD',
    msb=14,
    lsb=14,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_RDKYZRZZ = FieldMetadata(
    name='NV_R_IZEOAHGN_F_RDKYZRZZ',
    msb=16,
    lsb=16,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_EYLWIOJU = FieldMetadata(
    name='NV_R_IZEOAHGN_F_EYLWIOJU',
    msb=13,
    lsb=13,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_PFOTNRKY = FieldMetadata(
    name='NV_R_IZEOAHGN_F_PFOTNRKY',
    msb=12,
    lsb=12,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_ELXFWGGQ = FieldMetadata(
    name='NV_R_IZEOAHGN_F_ELXFWGGQ',
    msb=18,
    lsb=18,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_DLPPSKET = FieldMetadata(
    name='NV_R_IZEOAHGN_F_DLPPSKET',
    msb=8,
    lsb=8,
    register=NV_R_IZEOAHGN
)

NV_R_IZEOAHGN_F_ILOUIJKE = FieldMetadata(
    name='NV_R_IZEOAHGN_F_ILOUIJKE',
    msb=9,
    lsb=9,
    register=NV_R_IZEOAHGN
)

NV_R_ESKJFLST = RegisterMetadata(
    name='NV_R_ESKJFLST',
    address=0x1cc,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_ESKJFLST_F_VCGGYIVK = FieldMetadata(
    name='NV_R_ESKJFLST_F_VCGGYIVK',
    msb=12,
    lsb=12,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_QLRDKHUP = FieldMetadata(
    name='NV_R_ESKJFLST_F_QLRDKHUP',
    msb=6,
    lsb=6,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_HNSZUXUD = FieldMetadata(
    name='NV_R_ESKJFLST_F_HNSZUXUD',
    msb=18,
    lsb=18,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_MKLAZDQM = FieldMetadata(
    name='NV_R_ESKJFLST_F_MKLAZDQM',
    msb=0,
    lsb=0,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_TBQCPYQI = FieldMetadata(
    name='NV_R_ESKJFLST_F_TBQCPYQI',
    msb=24,
    lsb=24,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_PHXCTNID = FieldMetadata(
    name='NV_R_ESKJFLST_F_PHXCTNID',
    msb=13,
    lsb=13,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_XIEMEZCU = FieldMetadata(
    name='NV_R_ESKJFLST_F_XIEMEZCU',
    msb=7,
    lsb=7,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_TZDPUEFQ = FieldMetadata(
    name='NV_R_ESKJFLST_F_TZDPUEFQ',
    msb=19,
    lsb=19,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_DGOFWVOK = FieldMetadata(
    name='NV_R_ESKJFLST_F_DGOFWVOK',
    msb=1,
    lsb=1,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_DHRYWLZN = FieldMetadata(
    name='NV_R_ESKJFLST_F_DHRYWLZN',
    msb=25,
    lsb=25,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_GHRIOYVA = FieldMetadata(
    name='NV_R_ESKJFLST_F_GHRIOYVA',
    msb=14,
    lsb=14,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_WHWESNAV = FieldMetadata(
    name='NV_R_ESKJFLST_F_WHWESNAV',
    msb=8,
    lsb=8,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_TWYNOCWN = FieldMetadata(
    name='NV_R_ESKJFLST_F_TWYNOCWN',
    msb=20,
    lsb=20,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_AFBITBKN = FieldMetadata(
    name='NV_R_ESKJFLST_F_AFBITBKN',
    msb=2,
    lsb=2,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_LIRFOWMW = FieldMetadata(
    name='NV_R_ESKJFLST_F_LIRFOWMW',
    msb=26,
    lsb=26,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_BDXPAPPT = FieldMetadata(
    name='NV_R_ESKJFLST_F_BDXPAPPT',
    msb=15,
    lsb=15,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_BGPHLQUY = FieldMetadata(
    name='NV_R_ESKJFLST_F_BGPHLQUY',
    msb=9,
    lsb=9,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_PPCDQOOF = FieldMetadata(
    name='NV_R_ESKJFLST_F_PPCDQOOF',
    msb=21,
    lsb=21,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_ZEBIGWCX = FieldMetadata(
    name='NV_R_ESKJFLST_F_ZEBIGWCX',
    msb=3,
    lsb=3,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_PZKRGPLQ = FieldMetadata(
    name='NV_R_ESKJFLST_F_PZKRGPLQ',
    msb=27,
    lsb=27,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_QLGMWTBS = FieldMetadata(
    name='NV_R_ESKJFLST_F_QLGMWTBS',
    msb=16,
    lsb=16,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_OCTEAQBY = FieldMetadata(
    name='NV_R_ESKJFLST_F_OCTEAQBY',
    msb=10,
    lsb=10,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_UVDXWMSV = FieldMetadata(
    name='NV_R_ESKJFLST_F_UVDXWMSV',
    msb=22,
    lsb=22,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_MHCUVNCJ = FieldMetadata(
    name='NV_R_ESKJFLST_F_MHCUVNCJ',
    msb=4,
    lsb=4,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_GZKODAIV = FieldMetadata(
    name='NV_R_ESKJFLST_F_GZKODAIV',
    msb=28,
    lsb=28,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_ANNMRPQZ = FieldMetadata(
    name='NV_R_ESKJFLST_F_ANNMRPQZ',
    msb=17,
    lsb=17,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_ZFIPTBSX = FieldMetadata(
    name='NV_R_ESKJFLST_F_ZFIPTBSX',
    msb=11,
    lsb=11,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_IVNPCVVE = FieldMetadata(
    name='NV_R_ESKJFLST_F_IVNPCVVE',
    msb=23,
    lsb=23,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_NEFZXJFM = FieldMetadata(
    name='NV_R_ESKJFLST_F_NEFZXJFM',
    msb=5,
    lsb=5,
    register=NV_R_ESKJFLST
)

NV_R_ESKJFLST_F_CHSOJTFO = FieldMetadata(
    name='NV_R_ESKJFLST_F_CHSOJTFO',
    msb=29,
    lsb=29,
    register=NV_R_ESKJFLST
)

NV_R_EPIENTQD = RegisterMetadata(
    name='NV_R_EPIENTQD',
    address=0x1a4,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_EPIENTQD_F_KUMVAQJD = FieldMetadata(
    name='NV_R_EPIENTQD_F_KUMVAQJD',
    msb=0,
    lsb=0,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_TYOXZGWM = FieldMetadata(
    name='NV_R_EPIENTQD_F_TYOXZGWM',
    msb=24,
    lsb=24,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_WYLFHULE = FieldMetadata(
    name='NV_R_EPIENTQD_F_WYLFHULE',
    msb=8,
    lsb=8,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_XHHTFXBF = FieldMetadata(
    name='NV_R_EPIENTQD_F_XHHTFXBF',
    msb=16,
    lsb=16,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_FNYCRITA = FieldMetadata(
    name='NV_R_EPIENTQD_F_FNYCRITA',
    msb=1,
    lsb=1,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_UZOYGOIP = FieldMetadata(
    name='NV_R_EPIENTQD_F_UZOYGOIP',
    msb=25,
    lsb=25,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_HBAEMGIR = FieldMetadata(
    name='NV_R_EPIENTQD_F_HBAEMGIR',
    msb=9,
    lsb=9,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_RCCGDQEN = FieldMetadata(
    name='NV_R_EPIENTQD_F_RCCGDQEN',
    msb=17,
    lsb=17,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_UYWXPQFA = FieldMetadata(
    name='NV_R_EPIENTQD_F_UYWXPQFA',
    msb=2,
    lsb=2,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_JPCMENDU = FieldMetadata(
    name='NV_R_EPIENTQD_F_JPCMENDU',
    msb=26,
    lsb=26,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_AYMLSYVB = FieldMetadata(
    name='NV_R_EPIENTQD_F_AYMLSYVB',
    msb=10,
    lsb=10,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_RMHZUBHC = FieldMetadata(
    name='NV_R_EPIENTQD_F_RMHZUBHC',
    msb=18,
    lsb=18,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_TLNBSLMX = FieldMetadata(
    name='NV_R_EPIENTQD_F_TLNBSLMX',
    msb=3,
    lsb=3,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_WPCNGHRP = FieldMetadata(
    name='NV_R_EPIENTQD_F_WPCNGHRP',
    msb=27,
    lsb=27,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_CJUKDVAT = FieldMetadata(
    name='NV_R_EPIENTQD_F_CJUKDVAT',
    msb=11,
    lsb=11,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_LRKQTPQE = FieldMetadata(
    name='NV_R_EPIENTQD_F_LRKQTPQE',
    msb=19,
    lsb=19,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_TWDGLYIX = FieldMetadata(
    name='NV_R_EPIENTQD_F_TWDGLYIX',
    msb=4,
    lsb=4,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_MJKRSJDN = FieldMetadata(
    name='NV_R_EPIENTQD_F_MJKRSJDN',
    msb=28,
    lsb=28,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_FCOVROJD = FieldMetadata(
    name='NV_R_EPIENTQD_F_FCOVROJD',
    msb=12,
    lsb=12,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_HWNZIONS = FieldMetadata(
    name='NV_R_EPIENTQD_F_HWNZIONS',
    msb=20,
    lsb=20,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_FULXGPBC = FieldMetadata(
    name='NV_R_EPIENTQD_F_FULXGPBC',
    msb=5,
    lsb=5,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_QFPONURS = FieldMetadata(
    name='NV_R_EPIENTQD_F_QFPONURS',
    msb=29,
    lsb=29,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_FGICFQEE = FieldMetadata(
    name='NV_R_EPIENTQD_F_FGICFQEE',
    msb=13,
    lsb=13,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_CXUGBGVX = FieldMetadata(
    name='NV_R_EPIENTQD_F_CXUGBGVX',
    msb=21,
    lsb=21,
    register=NV_R_EPIENTQD
)

NV_R_ANIRSEGB = RegisterMetadata(
    name='NV_R_ANIRSEGB',
    address=0x1a8,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_ANIRSEGB_F_TLLINTEJ = FieldMetadata(
    name='NV_R_ANIRSEGB_F_TLLINTEJ',
    msb=0,
    lsb=0,
    register=NV_R_ANIRSEGB
)

NV_R_ANIRSEGB_F_KUUPKMMF = FieldMetadata(
    name='NV_R_ANIRSEGB_F_KUUPKMMF',
    msb=8,
    lsb=8,
    register=NV_R_ANIRSEGB
)

NV_R_ANIRSEGB_F_JDHZSXQO = FieldMetadata(
    name='NV_R_ANIRSEGB_F_JDHZSXQO',
    msb=1,
    lsb=1,
    register=NV_R_ANIRSEGB
)

NV_R_ANIRSEGB_F_GUGTSKRC = FieldMetadata(
    name='NV_R_ANIRSEGB_F_GUGTSKRC',
    msb=9,
    lsb=9,
    register=NV_R_ANIRSEGB
)

NV_R_ANIRSEGB_F_CQBDJOCV = FieldMetadata(
    name='NV_R_ANIRSEGB_F_CQBDJOCV',
    msb=2,
    lsb=2,
    register=NV_R_ANIRSEGB
)

NV_R_ANIRSEGB_F_MPZNDMYO = FieldMetadata(
    name='NV_R_ANIRSEGB_F_MPZNDMYO',
    msb=10,
    lsb=10,
    register=NV_R_ANIRSEGB
)

NV_R_ANIRSEGB_F_TYYCARNV = FieldMetadata(
    name='NV_R_ANIRSEGB_F_TYYCARNV',
    msb=3,
    lsb=3,
    register=NV_R_ANIRSEGB
)

NV_R_ANIRSEGB_F_EXGFLTYW = FieldMetadata(
    name='NV_R_ANIRSEGB_F_EXGFLTYW',
    msb=11,
    lsb=11,
    register=NV_R_ANIRSEGB
)

NV_R_ANIRSEGB_F_KSALYIKH = FieldMetadata(
    name='NV_R_ANIRSEGB_F_KSALYIKH',
    msb=4,
    lsb=4,
    register=NV_R_ANIRSEGB
)

NV_R_ANIRSEGB_F_FOSWUJUM = FieldMetadata(
    name='NV_R_ANIRSEGB_F_FOSWUJUM',
    msb=12,
    lsb=12,
    register=NV_R_ANIRSEGB
)

NV_R_ANIRSEGB_F_RROOQHFY = FieldMetadata(
    name='NV_R_ANIRSEGB_F_RROOQHFY',
    msb=5,
    lsb=5,
    register=NV_R_ANIRSEGB
)

NV_R_ANIRSEGB_F_XGYPKBFE = FieldMetadata(
    name='NV_R_ANIRSEGB_F_XGYPKBFE',
    msb=13,
    lsb=13,
    register=NV_R_ANIRSEGB
)

NV_R_ETZZJOQG = RegisterMetadata(
    name='NV_R_ETZZJOQG',
    address=0x1ac,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_ETZZJOQG_F_CXEEAICP = FieldMetadata(
    name='NV_R_ETZZJOQG_F_CXEEAICP',
    msb=8,
    lsb=8,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_BCQVWXDT = FieldMetadata(
    name='NV_R_ETZZJOQG_F_BCQVWXDT',
    msb=0,
    lsb=0,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_TOKUHOAG = FieldMetadata(
    name='NV_R_ETZZJOQG_F_TOKUHOAG',
    msb=24,
    lsb=24,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_UDQJPFVT = FieldMetadata(
    name='NV_R_ETZZJOQG_F_UDQJPFVT',
    msb=16,
    lsb=16,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_CEJVDJOX = FieldMetadata(
    name='NV_R_ETZZJOQG_F_CEJVDJOX',
    msb=9,
    lsb=9,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_WBURGLQI = FieldMetadata(
    name='NV_R_ETZZJOQG_F_WBURGLQI',
    msb=1,
    lsb=1,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_SFDQVYNT = FieldMetadata(
    name='NV_R_ETZZJOQG_F_SFDQVYNT',
    msb=25,
    lsb=25,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_NGPSNCWL = FieldMetadata(
    name='NV_R_ETZZJOQG_F_NGPSNCWL',
    msb=17,
    lsb=17,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_ZBAJVGWI = FieldMetadata(
    name='NV_R_ETZZJOQG_F_ZBAJVGWI',
    msb=10,
    lsb=10,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_RWIAGUCH = FieldMetadata(
    name='NV_R_ETZZJOQG_F_RWIAGUCH',
    msb=2,
    lsb=2,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_MZHRVSFZ = FieldMetadata(
    name='NV_R_ETZZJOQG_F_MZHRVSFZ',
    msb=26,
    lsb=26,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_GXPQKYBO = FieldMetadata(
    name='NV_R_ETZZJOQG_F_GXPQKYBO',
    msb=18,
    lsb=18,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_RJPDPMUF = FieldMetadata(
    name='NV_R_ETZZJOQG_F_RJPDPMUF',
    msb=11,
    lsb=11,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_BCJRXJGU = FieldMetadata(
    name='NV_R_ETZZJOQG_F_BCJRXJGU',
    msb=3,
    lsb=3,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_TGFGGTXD = FieldMetadata(
    name='NV_R_ETZZJOQG_F_TGFGGTXD',
    msb=27,
    lsb=27,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_NSEWASIT = FieldMetadata(
    name='NV_R_ETZZJOQG_F_NSEWASIT',
    msb=19,
    lsb=19,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_ULSVLGBK = FieldMetadata(
    name='NV_R_ETZZJOQG_F_ULSVLGBK',
    msb=12,
    lsb=12,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_LXUJUSGI = FieldMetadata(
    name='NV_R_ETZZJOQG_F_LXUJUSGI',
    msb=4,
    lsb=4,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_FGRXETDZ = FieldMetadata(
    name='NV_R_ETZZJOQG_F_FGRXETDZ',
    msb=28,
    lsb=28,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_XJUQYTOQ = FieldMetadata(
    name='NV_R_ETZZJOQG_F_XJUQYTOQ',
    msb=20,
    lsb=20,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_CUCRWJAI = FieldMetadata(
    name='NV_R_ETZZJOQG_F_CUCRWJAI',
    msb=13,
    lsb=13,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_XNCNAQWD = FieldMetadata(
    name='NV_R_ETZZJOQG_F_XNCNAQWD',
    msb=5,
    lsb=5,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_JFFTYMAV = FieldMetadata(
    name='NV_R_ETZZJOQG_F_JFFTYMAV',
    msb=29,
    lsb=29,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_NKUZYUQG = FieldMetadata(
    name='NV_R_ETZZJOQG_F_NKUZYUQG',
    msb=21,
    lsb=21,
    register=NV_R_ETZZJOQG
)

NV_R_VHJZHKLK = RegisterMetadata(
    name='NV_R_VHJZHKLK',
    address=0x1b0,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_VHJZHKLK_F_NNFCGTYP = FieldMetadata(
    name='NV_R_VHJZHKLK_F_NNFCGTYP',
    msb=16,
    lsb=16,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_TRFBQKUD = FieldMetadata(
    name='NV_R_VHJZHKLK_F_TRFBQKUD',
    msb=0,
    lsb=0,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_DHJTXMCU = FieldMetadata(
    name='NV_R_VHJZHKLK_F_DHJTXMCU',
    msb=8,
    lsb=8,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_RZAUKMRK = FieldMetadata(
    name='NV_R_VHJZHKLK_F_RZAUKMRK',
    msb=24,
    lsb=24,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_SQTOOUAI = FieldMetadata(
    name='NV_R_VHJZHKLK_F_SQTOOUAI',
    msb=17,
    lsb=17,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_BXHUMUQN = FieldMetadata(
    name='NV_R_VHJZHKLK_F_BXHUMUQN',
    msb=1,
    lsb=1,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_SXOSCRCN = FieldMetadata(
    name='NV_R_VHJZHKLK_F_SXOSCRCN',
    msb=9,
    lsb=9,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_UTZQRJNK = FieldMetadata(
    name='NV_R_VHJZHKLK_F_UTZQRJNK',
    msb=25,
    lsb=25,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_NYBSYXIM = FieldMetadata(
    name='NV_R_VHJZHKLK_F_NYBSYXIM',
    msb=18,
    lsb=18,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_NCLKHMPI = FieldMetadata(
    name='NV_R_VHJZHKLK_F_NCLKHMPI',
    msb=2,
    lsb=2,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_CUMBFKMJ = FieldMetadata(
    name='NV_R_VHJZHKLK_F_CUMBFKMJ',
    msb=10,
    lsb=10,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_EMLDVAGV = FieldMetadata(
    name='NV_R_VHJZHKLK_F_EMLDVAGV',
    msb=26,
    lsb=26,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_QYOXNJZP = FieldMetadata(
    name='NV_R_VHJZHKLK_F_QYOXNJZP',
    msb=19,
    lsb=19,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_RJPSEBRK = FieldMetadata(
    name='NV_R_VHJZHKLK_F_RJPSEBRK',
    msb=3,
    lsb=3,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_NEKYEWMZ = FieldMetadata(
    name='NV_R_VHJZHKLK_F_NEKYEWMZ',
    msb=11,
    lsb=11,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_NCZDPOSB = FieldMetadata(
    name='NV_R_VHJZHKLK_F_NCZDPOSB',
    msb=27,
    lsb=27,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_ZHZYTJOF = FieldMetadata(
    name='NV_R_VHJZHKLK_F_ZHZYTJOF',
    msb=20,
    lsb=20,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_EYLHPQDX = FieldMetadata(
    name='NV_R_VHJZHKLK_F_EYLHPQDX',
    msb=4,
    lsb=4,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_OHUKQOFX = FieldMetadata(
    name='NV_R_VHJZHKLK_F_OHUKQOFX',
    msb=12,
    lsb=12,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_CSKSIVJM = FieldMetadata(
    name='NV_R_VHJZHKLK_F_CSKSIVJM',
    msb=28,
    lsb=28,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_HUYKHIAQ = FieldMetadata(
    name='NV_R_VHJZHKLK_F_HUYKHIAQ',
    msb=21,
    lsb=21,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_HZZTMIST = FieldMetadata(
    name='NV_R_VHJZHKLK_F_HZZTMIST',
    msb=5,
    lsb=5,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_QJDGDJIH = FieldMetadata(
    name='NV_R_VHJZHKLK_F_QJDGDJIH',
    msb=13,
    lsb=13,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_CMRTSQSB = FieldMetadata(
    name='NV_R_VHJZHKLK_F_CMRTSQSB',
    msb=29,
    lsb=29,
    register=NV_R_VHJZHKLK
)

NV_R_KTLFSNMO = RegisterMetadata(
    name='NV_R_KTLFSNMO',
    address=0x1b4,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_KTLFSNMO_F_KHOESNRQ = FieldMetadata(
    name='NV_R_KTLFSNMO_F_KHOESNRQ',
    msb=8,
    lsb=8,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_ZPNLKDEN = FieldMetadata(
    name='NV_R_KTLFSNMO_F_ZPNLKDEN',
    msb=0,
    lsb=0,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_SUSNCSAX = FieldMetadata(
    name='NV_R_KTLFSNMO_F_SUSNCSAX',
    msb=24,
    lsb=24,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_PODVEVSN = FieldMetadata(
    name='NV_R_KTLFSNMO_F_PODVEVSN',
    msb=9,
    lsb=9,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_OXZAIRWH = FieldMetadata(
    name='NV_R_KTLFSNMO_F_OXZAIRWH',
    msb=1,
    lsb=1,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_RUGAKIAI = FieldMetadata(
    name='NV_R_KTLFSNMO_F_RUGAKIAI',
    msb=25,
    lsb=25,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_OBAGFZLV = FieldMetadata(
    name='NV_R_KTLFSNMO_F_OBAGFZLV',
    msb=10,
    lsb=10,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_NXJQECPI = FieldMetadata(
    name='NV_R_KTLFSNMO_F_NXJQECPI',
    msb=2,
    lsb=2,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_MNCCEEIB = FieldMetadata(
    name='NV_R_KTLFSNMO_F_MNCCEEIB',
    msb=26,
    lsb=26,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_AISBJVUR = FieldMetadata(
    name='NV_R_KTLFSNMO_F_AISBJVUR',
    msb=11,
    lsb=11,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_FUTPTAPB = FieldMetadata(
    name='NV_R_KTLFSNMO_F_FUTPTAPB',
    msb=3,
    lsb=3,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_YHBZFVBF = FieldMetadata(
    name='NV_R_KTLFSNMO_F_YHBZFVBF',
    msb=27,
    lsb=27,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_XUKVUKZW = FieldMetadata(
    name='NV_R_KTLFSNMO_F_XUKVUKZW',
    msb=12,
    lsb=12,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_RXUXVUVA = FieldMetadata(
    name='NV_R_KTLFSNMO_F_RXUXVUVA',
    msb=4,
    lsb=4,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_NVIEZVTT = FieldMetadata(
    name='NV_R_KTLFSNMO_F_NVIEZVTT',
    msb=28,
    lsb=28,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_DGUIPCAF = FieldMetadata(
    name='NV_R_KTLFSNMO_F_DGUIPCAF',
    msb=13,
    lsb=13,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_CFDJFETZ = FieldMetadata(
    name='NV_R_KTLFSNMO_F_CFDJFETZ',
    msb=5,
    lsb=5,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_CBMKLNXI = FieldMetadata(
    name='NV_R_KTLFSNMO_F_CBMKLNXI',
    msb=29,
    lsb=29,
    register=NV_R_KTLFSNMO
)

NV_R_FNOEGFHF = RegisterMetadata(
    name='NV_R_FNOEGFHF',
    address=0x1b8,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_FNOEGFHF_F_SSZLBTBR = FieldMetadata(
    name='NV_R_FNOEGFHF_F_SSZLBTBR',
    msb=2,
    lsb=2,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_KLUFQYAI = FieldMetadata(
    name='NV_R_FNOEGFHF_F_KLUFQYAI',
    msb=0,
    lsb=0,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_OTZTXNMA = FieldMetadata(
    name='NV_R_FNOEGFHF_F_OTZTXNMA',
    msb=26,
    lsb=26,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_ZPTMFEZU = FieldMetadata(
    name='NV_R_FNOEGFHF_F_ZPTMFEZU',
    msb=20,
    lsb=20,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_JULXAXCU = FieldMetadata(
    name='NV_R_FNOEGFHF_F_JULXAXCU',
    msb=12,
    lsb=12,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_FAFXSATM = FieldMetadata(
    name='NV_R_FNOEGFHF_F_FAFXSATM',
    msb=16,
    lsb=16,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_MBGRXNCB = FieldMetadata(
    name='NV_R_FNOEGFHF_F_MBGRXNCB',
    msb=8,
    lsb=8,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_RMKXKYPS = FieldMetadata(
    name='NV_R_FNOEGFHF_F_RMKXKYPS',
    msb=4,
    lsb=4,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_SLERAOFN = FieldMetadata(
    name='NV_R_FNOEGFHF_F_SLERAOFN',
    msb=24,
    lsb=24,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_PKDDTIRB = FieldMetadata(
    name='NV_R_FNOEGFHF_F_PKDDTIRB',
    msb=3,
    lsb=3,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_YRVXNZTC = FieldMetadata(
    name='NV_R_FNOEGFHF_F_YRVXNZTC',
    msb=1,
    lsb=1,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_OHLYGUFA = FieldMetadata(
    name='NV_R_FNOEGFHF_F_OHLYGUFA',
    msb=27,
    lsb=27,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_JFBZTFNH = FieldMetadata(
    name='NV_R_FNOEGFHF_F_JFBZTFNH',
    msb=21,
    lsb=21,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_FWYMYWIS = FieldMetadata(
    name='NV_R_FNOEGFHF_F_FWYMYWIS',
    msb=13,
    lsb=13,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_JAHTFFTS = FieldMetadata(
    name='NV_R_FNOEGFHF_F_JAHTFFTS',
    msb=17,
    lsb=17,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_WLEDNNBK = FieldMetadata(
    name='NV_R_FNOEGFHF_F_WLEDNNBK',
    msb=9,
    lsb=9,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_HZYWMTCE = FieldMetadata(
    name='NV_R_FNOEGFHF_F_HZYWMTCE',
    msb=5,
    lsb=5,
    register=NV_R_FNOEGFHF
)

NV_R_FNOEGFHF_F_YKYKPXPR = FieldMetadata(
    name='NV_R_FNOEGFHF_F_YKYKPXPR',
    msb=25,
    lsb=25,
    register=NV_R_FNOEGFHF
)

NV_R_JBKARAJO = RegisterMetadata(
    name='NV_R_JBKARAJO',
    address=0x1bc,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_JBKARAJO_F_EQLQFTGH = FieldMetadata(
    name='NV_R_JBKARAJO_F_EQLQFTGH',
    msb=8,
    lsb=8,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_FKCQGEXL = FieldMetadata(
    name='NV_R_JBKARAJO_F_FKCQGEXL',
    msb=24,
    lsb=24,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_JOHJYAWP = FieldMetadata(
    name='NV_R_JBKARAJO_F_JOHJYAWP',
    msb=0,
    lsb=0,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_KBHOOAYF = FieldMetadata(
    name='NV_R_JBKARAJO_F_KBHOOAYF',
    msb=16,
    lsb=16,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_ZPMGXEGS = FieldMetadata(
    name='NV_R_JBKARAJO_F_ZPMGXEGS',
    msb=9,
    lsb=9,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_YLQFUGYI = FieldMetadata(
    name='NV_R_JBKARAJO_F_YLQFUGYI',
    msb=25,
    lsb=25,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_HSSCRQVH = FieldMetadata(
    name='NV_R_JBKARAJO_F_HSSCRQVH',
    msb=1,
    lsb=1,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_QQNSDYPH = FieldMetadata(
    name='NV_R_JBKARAJO_F_QQNSDYPH',
    msb=17,
    lsb=17,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_GHSJEOPH = FieldMetadata(
    name='NV_R_JBKARAJO_F_GHSJEOPH',
    msb=10,
    lsb=10,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_HWPKKPOS = FieldMetadata(
    name='NV_R_JBKARAJO_F_HWPKKPOS',
    msb=26,
    lsb=26,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_BZGIJGTC = FieldMetadata(
    name='NV_R_JBKARAJO_F_BZGIJGTC',
    msb=2,
    lsb=2,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_PQYKPSKJ = FieldMetadata(
    name='NV_R_JBKARAJO_F_PQYKPSKJ',
    msb=18,
    lsb=18,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_HAYDREJP = FieldMetadata(
    name='NV_R_JBKARAJO_F_HAYDREJP',
    msb=11,
    lsb=11,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_YONASMHB = FieldMetadata(
    name='NV_R_JBKARAJO_F_YONASMHB',
    msb=27,
    lsb=27,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_UHWMFIRQ = FieldMetadata(
    name='NV_R_JBKARAJO_F_UHWMFIRQ',
    msb=3,
    lsb=3,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_FRKPLMUI = FieldMetadata(
    name='NV_R_JBKARAJO_F_FRKPLMUI',
    msb=19,
    lsb=19,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_VSIKVIVJ = FieldMetadata(
    name='NV_R_JBKARAJO_F_VSIKVIVJ',
    msb=12,
    lsb=12,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_GTNHTLPC = FieldMetadata(
    name='NV_R_JBKARAJO_F_GTNHTLPC',
    msb=28,
    lsb=28,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_PHTDXUBN = FieldMetadata(
    name='NV_R_JBKARAJO_F_PHTDXUBN',
    msb=4,
    lsb=4,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_DZRYQXUL = FieldMetadata(
    name='NV_R_JBKARAJO_F_DZRYQXUL',
    msb=20,
    lsb=20,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_SDQIJEBM = FieldMetadata(
    name='NV_R_JBKARAJO_F_SDQIJEBM',
    msb=13,
    lsb=13,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_WKNXUYEN = FieldMetadata(
    name='NV_R_JBKARAJO_F_WKNXUYEN',
    msb=29,
    lsb=29,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_HSWSYGNF = FieldMetadata(
    name='NV_R_JBKARAJO_F_HSWSYGNF',
    msb=5,
    lsb=5,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_NECPJIYP = FieldMetadata(
    name='NV_R_JBKARAJO_F_NECPJIYP',
    msb=21,
    lsb=21,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_AAQXITLF = FieldMetadata(
    name='NV_R_JBKARAJO_F_AAQXITLF',
    msb=14,
    lsb=14,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_PVTUWAZI = FieldMetadata(
    name='NV_R_JBKARAJO_F_PVTUWAZI',
    msb=30,
    lsb=30,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_OTMSOGWT = FieldMetadata(
    name='NV_R_JBKARAJO_F_OTMSOGWT',
    msb=6,
    lsb=6,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_KZUAUTMT = FieldMetadata(
    name='NV_R_JBKARAJO_F_KZUAUTMT',
    msb=22,
    lsb=22,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_ZBZWDAGO = FieldMetadata(
    name='NV_R_JBKARAJO_F_ZBZWDAGO',
    msb=15,
    lsb=15,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_XGPRZMOL = FieldMetadata(
    name='NV_R_JBKARAJO_F_XGPRZMOL',
    msb=31,
    lsb=31,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_RKVLBNOK = FieldMetadata(
    name='NV_R_JBKARAJO_F_RKVLBNOK',
    msb=7,
    lsb=7,
    register=NV_R_JBKARAJO
)

NV_R_JBKARAJO_F_WZOVEWYN = FieldMetadata(
    name='NV_R_JBKARAJO_F_WZOVEWYN',
    msb=23,
    lsb=23,
    register=NV_R_JBKARAJO
)

NV_R_FJIYUNNQ = RegisterMetadata(
    name='NV_R_FJIYUNNQ',
    address=0x1c0,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_FJIYUNNQ_F_NPNHWLCB = FieldMetadata(
    name='NV_R_FJIYUNNQ_F_NPNHWLCB',
    msb=15,
    lsb=15,
    register=NV_R_FJIYUNNQ
)

NV_R_FJIYUNNQ_F_IBAPEUDK = FieldMetadata(
    name='NV_R_FJIYUNNQ_F_IBAPEUDK',
    msb=0,
    lsb=0,
    register=NV_R_FJIYUNNQ
)

NV_R_FJIYUNNQ_F_ENYWDIMG = FieldMetadata(
    name='NV_R_FJIYUNNQ_F_ENYWDIMG',
    msb=1,
    lsb=1,
    register=NV_R_FJIYUNNQ
)

NV_R_FJIYUNNQ_F_LEFFWSZZ = FieldMetadata(
    name='NV_R_FJIYUNNQ_F_LEFFWSZZ',
    msb=2,
    lsb=2,
    register=NV_R_FJIYUNNQ
)

NV_R_TFXQVLAN = RegisterMetadata(
    name='NV_R_TFXQVLAN',
    address=0xa70,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_TFXQVLAN_F_BYVHZPQX = FieldMetadata(
    name='NV_R_TFXQVLAN_F_BYVHZPQX',
    msb=0,
    lsb=0,
    register=NV_R_TFXQVLAN
)

NV_R_TFXQVLAN_F_UFBJNCSP = FieldMetadata(
    name='NV_R_TFXQVLAN_F_UFBJNCSP',
    msb=8,
    lsb=8,
    register=NV_R_TFXQVLAN
)

NV_R_TFXQVLAN_F_QIAPVCZD = FieldMetadata(
    name='NV_R_TFXQVLAN_F_QIAPVCZD',
    msb=1,
    lsb=1,
    register=NV_R_TFXQVLAN
)

NV_R_TFXQVLAN_F_KMAXVQYV = FieldMetadata(
    name='NV_R_TFXQVLAN_F_KMAXVQYV',
    msb=9,
    lsb=9,
    register=NV_R_TFXQVLAN
)

NV_R_TFXQVLAN_F_ABWDODUN = FieldMetadata(
    name='NV_R_TFXQVLAN_F_ABWDODUN',
    msb=2,
    lsb=2,
    register=NV_R_TFXQVLAN
)

NV_R_TFXQVLAN_F_RLTCBIRF = FieldMetadata(
    name='NV_R_TFXQVLAN_F_RLTCBIRF',
    msb=10,
    lsb=10,
    register=NV_R_TFXQVLAN
)

NV_R_TFXQVLAN_F_XOXIJWVV = FieldMetadata(
    name='NV_R_TFXQVLAN_F_XOXIJWVV',
    msb=3,
    lsb=3,
    register=NV_R_TFXQVLAN
)

NV_R_TFXQVLAN_F_ZGNJYOPN = FieldMetadata(
    name='NV_R_TFXQVLAN_F_ZGNJYOPN',
    msb=11,
    lsb=11,
    register=NV_R_TFXQVLAN
)

NV_R_TFXQVLAN_F_SJDTJBJN = FieldMetadata(
    name='NV_R_TFXQVLAN_F_SJDTJBJN',
    msb=4,
    lsb=4,
    register=NV_R_TFXQVLAN
)

NV_R_TFXQVLAN_F_TUJFZICR = FieldMetadata(
    name='NV_R_TFXQVLAN_F_TUJFZICR',
    msb=12,
    lsb=12,
    register=NV_R_TFXQVLAN
)

NV_R_TFXQVLAN_F_ZKBAFHLI = FieldMetadata(
    name='NV_R_TFXQVLAN_F_ZKBAFHLI',
    msb=5,
    lsb=5,
    register=NV_R_TFXQVLAN
)

NV_R_TFXQVLAN_F_GHJPEMCH = FieldMetadata(
    name='NV_R_TFXQVLAN_F_GHJPEMCH',
    msb=13,
    lsb=13,
    register=NV_R_TFXQVLAN
)

NV_R_TFXQVLAN_F_DWPQFURP = FieldMetadata(
    name='NV_R_TFXQVLAN_F_DWPQFURP',
    msb=6,
    lsb=6,
    register=NV_R_TFXQVLAN
)

NV_R_TFXQVLAN_F_ORNINPVJ = FieldMetadata(
    name='NV_R_TFXQVLAN_F_ORNINPVJ',
    msb=14,
    lsb=14,
    register=NV_R_TFXQVLAN
)

NV_R_TFXQVLAN_F_YSWEGNLP = FieldMetadata(
    name='NV_R_TFXQVLAN_F_YSWEGNLP',
    msb=7,
    lsb=7,
    register=NV_R_TFXQVLAN
)

NV_R_TFXQVLAN_F_HWIDPMKJ = FieldMetadata(
    name='NV_R_TFXQVLAN_F_HWIDPMKJ',
    msb=15,
    lsb=15,
    register=NV_R_TFXQVLAN
)

NV_R_VZBXLZCT = RegisterMetadata(
    name='NV_R_VZBXLZCT',
    address=0xa74,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_VZBXLZCT_F_MVCHRIIC = FieldMetadata(
    name='NV_R_VZBXLZCT_F_MVCHRIIC',
    msb=8,
    lsb=8,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_VXKVFDHV = FieldMetadata(
    name='NV_R_VZBXLZCT_F_VXKVFDHV',
    msb=0,
    lsb=0,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_SLVVMPNI = FieldMetadata(
    name='NV_R_VZBXLZCT_F_SLVVMPNI',
    msb=16,
    lsb=16,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_ZZDMIFVR = FieldMetadata(
    name='NV_R_VZBXLZCT_F_ZZDMIFVR',
    msb=9,
    lsb=9,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_AWDPQRWV = FieldMetadata(
    name='NV_R_VZBXLZCT_F_AWDPQRWV',
    msb=1,
    lsb=1,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_JOFIQYJY = FieldMetadata(
    name='NV_R_VZBXLZCT_F_JOFIQYJY',
    msb=17,
    lsb=17,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_CSEIMAGX = FieldMetadata(
    name='NV_R_VZBXLZCT_F_CSEIMAGX',
    msb=10,
    lsb=10,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_UUMZCYVZ = FieldMetadata(
    name='NV_R_VZBXLZCT_F_UUMZCYVZ',
    msb=2,
    lsb=2,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_NGWZDWXR = FieldMetadata(
    name='NV_R_VZBXLZCT_F_NGWZDWXR',
    msb=18,
    lsb=18,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_TOACAHUV = FieldMetadata(
    name='NV_R_VZBXLZCT_F_TOACAHUV',
    msb=11,
    lsb=11,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_IWCUIWSQ = FieldMetadata(
    name='NV_R_VZBXLZCT_F_IWCUIWSQ',
    msb=3,
    lsb=3,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_AZPGZVJI = FieldMetadata(
    name='NV_R_VZBXLZCT_F_AZPGZVJI',
    msb=19,
    lsb=19,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_PJVGMLHM = FieldMetadata(
    name='NV_R_VZBXLZCT_F_PJVGMLHM',
    msb=12,
    lsb=12,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_PZQDUIPA = FieldMetadata(
    name='NV_R_VZBXLZCT_F_PZQDUIPA',
    msb=4,
    lsb=4,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_AZTIZHJX = FieldMetadata(
    name='NV_R_VZBXLZCT_F_AZTIZHJX',
    msb=20,
    lsb=20,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_RYGOFPTC = FieldMetadata(
    name='NV_R_VZBXLZCT_F_RYGOFPTC',
    msb=13,
    lsb=13,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_XXPPJYOF = FieldMetadata(
    name='NV_R_VZBXLZCT_F_XXPPJYOF',
    msb=5,
    lsb=5,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_MORUAPFT = FieldMetadata(
    name='NV_R_VZBXLZCT_F_MORUAPFT',
    msb=21,
    lsb=21,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_WYXYQPAB = FieldMetadata(
    name='NV_R_VZBXLZCT_F_WYXYQPAB',
    msb=14,
    lsb=14,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_WSWMUGTJ = FieldMetadata(
    name='NV_R_VZBXLZCT_F_WSWMUGTJ',
    msb=6,
    lsb=6,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_SPNDDMHM = FieldMetadata(
    name='NV_R_VZBXLZCT_F_SPNDDMHM',
    msb=22,
    lsb=22,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_OGIZWIPZ = FieldMetadata(
    name='NV_R_VZBXLZCT_F_OGIZWIPZ',
    msb=15,
    lsb=15,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_RGYRBELI = FieldMetadata(
    name='NV_R_VZBXLZCT_F_RGYRBELI',
    msb=7,
    lsb=7,
    register=NV_R_VZBXLZCT
)

NV_R_VZBXLZCT_F_QULPMVIN = FieldMetadata(
    name='NV_R_VZBXLZCT_F_QULPMVIN',
    msb=23,
    lsb=23,
    register=NV_R_VZBXLZCT
)

NV_R_WWYHZXYY = RegisterMetadata(
    name='NV_R_WWYHZXYY',
    address=0xa78,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_WWYHZXYY_F_ADVIPOBT = FieldMetadata(
    name='NV_R_WWYHZXYY_F_ADVIPOBT',
    msb=0,
    lsb=0,
    register=NV_R_WWYHZXYY
)

NV_R_WWYHZXYY_F_SMZZYMVN = FieldMetadata(
    name='NV_R_WWYHZXYY_F_SMZZYMVN',
    msb=1,
    lsb=1,
    register=NV_R_WWYHZXYY
)

NV_R_WWYHZXYY_F_EOBWXOYT = FieldMetadata(
    name='NV_R_WWYHZXYY_F_EOBWXOYT',
    msb=2,
    lsb=2,
    register=NV_R_WWYHZXYY
)

NV_R_WWYHZXYY_F_CEDPJRUO = FieldMetadata(
    name='NV_R_WWYHZXYY_F_CEDPJRUO',
    msb=3,
    lsb=3,
    register=NV_R_WWYHZXYY
)

NV_R_WWYHZXYY_F_PSYXUVXN = FieldMetadata(
    name='NV_R_WWYHZXYY_F_PSYXUVXN',
    msb=4,
    lsb=4,
    register=NV_R_WWYHZXYY
)

NV_R_WWYHZXYY_F_NICBIVHT = FieldMetadata(
    name='NV_R_WWYHZXYY_F_NICBIVHT',
    msb=5,
    lsb=5,
    register=NV_R_WWYHZXYY
)

NV_R_GIYEYJDF = RegisterMetadata(
    name='NV_R_GIYEYJDF',
    address=0xa7c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_GIYEYJDF_F_HIHAFTKL = FieldMetadata(
    name='NV_R_GIYEYJDF_F_HIHAFTKL',
    msb=0,
    lsb=0,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_MAFIKAWU = FieldMetadata(
    name='NV_R_GIYEYJDF_F_MAFIKAWU',
    msb=1,
    lsb=1,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_ZMKFKZMR = FieldMetadata(
    name='NV_R_GIYEYJDF_F_ZMKFKZMR',
    msb=2,
    lsb=2,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_KTPNHEDL = FieldMetadata(
    name='NV_R_GIYEYJDF_F_KTPNHEDL',
    msb=3,
    lsb=3,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_IASRQXAA = FieldMetadata(
    name='NV_R_GIYEYJDF_F_IASRQXAA',
    msb=4,
    lsb=4,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_LEVUINXN = FieldMetadata(
    name='NV_R_GIYEYJDF_F_LEVUINXN',
    msb=5,
    lsb=5,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_AXTKUHUL = FieldMetadata(
    name='NV_R_GIYEYJDF_F_AXTKUHUL',
    msb=30,
    lsb=30,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_KGDURNQY = FieldMetadata(
    name='NV_R_GIYEYJDF_F_KGDURNQY',
    msb=31,
    lsb=31,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_FTRTNUVX = FieldMetadata(
    name='NV_R_GIYEYJDF_F_FTRTNUVX',
    msb=6,
    lsb=6,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_QLLFJRSD = FieldMetadata(
    name='NV_R_GIYEYJDF_F_QLLFJRSD',
    msb=12,
    lsb=12,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_RTOEXJZB = FieldMetadata(
    name='NV_R_GIYEYJDF_F_RTOEXJZB',
    msb=18,
    lsb=18,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_IFLRDQDB = FieldMetadata(
    name='NV_R_GIYEYJDF_F_IFLRDQDB',
    msb=24,
    lsb=24,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_WOJYUHZQ = FieldMetadata(
    name='NV_R_GIYEYJDF_F_WOJYUHZQ',
    msb=7,
    lsb=7,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_DEIWFIXK = FieldMetadata(
    name='NV_R_GIYEYJDF_F_DEIWFIXK',
    msb=13,
    lsb=13,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_IOCHURJF = FieldMetadata(
    name='NV_R_GIYEYJDF_F_IOCHURJF',
    msb=19,
    lsb=19,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_AWHQUNZW = FieldMetadata(
    name='NV_R_GIYEYJDF_F_AWHQUNZW',
    msb=25,
    lsb=25,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_VAIRXJAY = FieldMetadata(
    name='NV_R_GIYEYJDF_F_VAIRXJAY',
    msb=8,
    lsb=8,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_RSEPHDRQ = FieldMetadata(
    name='NV_R_GIYEYJDF_F_RSEPHDRQ',
    msb=14,
    lsb=14,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_PVPIYWOY = FieldMetadata(
    name='NV_R_GIYEYJDF_F_PVPIYWOY',
    msb=20,
    lsb=20,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_AEHJJNUU = FieldMetadata(
    name='NV_R_GIYEYJDF_F_AEHJJNUU',
    msb=26,
    lsb=26,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_MYBLZEWY = FieldMetadata(
    name='NV_R_GIYEYJDF_F_MYBLZEWY',
    msb=9,
    lsb=9,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_OMFFCENF = FieldMetadata(
    name='NV_R_GIYEYJDF_F_OMFFCENF',
    msb=15,
    lsb=15,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_TCGCOFJE = FieldMetadata(
    name='NV_R_GIYEYJDF_F_TCGCOFJE',
    msb=21,
    lsb=21,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_ZCYLKBSV = FieldMetadata(
    name='NV_R_GIYEYJDF_F_ZCYLKBSV',
    msb=27,
    lsb=27,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_WXFBSLUS = FieldMetadata(
    name='NV_R_GIYEYJDF_F_WXFBSLUS',
    msb=10,
    lsb=10,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_VZHCHNXF = FieldMetadata(
    name='NV_R_GIYEYJDF_F_VZHCHNXF',
    msb=16,
    lsb=16,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_MOSLRSDR = FieldMetadata(
    name='NV_R_GIYEYJDF_F_MOSLRSDR',
    msb=22,
    lsb=22,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_JZSLCCSZ = FieldMetadata(
    name='NV_R_GIYEYJDF_F_JZSLCCSZ',
    msb=28,
    lsb=28,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_DIQKXZEF = FieldMetadata(
    name='NV_R_GIYEYJDF_F_DIQKXZEF',
    msb=11,
    lsb=11,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_JYNILTFR = FieldMetadata(
    name='NV_R_GIYEYJDF_F_JYNILTFR',
    msb=17,
    lsb=17,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_NCTKVOJU = FieldMetadata(
    name='NV_R_GIYEYJDF_F_NCTKVOJU',
    msb=23,
    lsb=23,
    register=NV_R_GIYEYJDF
)

NV_R_GIYEYJDF_F_DAETHFNV = FieldMetadata(
    name='NV_R_GIYEYJDF_F_DAETHFNV',
    msb=29,
    lsb=29,
    register=NV_R_GIYEYJDF
)

