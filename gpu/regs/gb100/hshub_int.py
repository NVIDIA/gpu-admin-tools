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
from gpu.regs.gh100.hshub_int import (
    NV_R_ARPDSYWJ,
    NV_R_ARPDSYWJ_F_UIEBWIOP,
    NV_R_ARPDSYWJ_F_FAEUEZAW,
    NV_R_ARPDSYWJ_F_BYVHZPQX,
    NV_R_ARPDSYWJ_F_OVHIEUWZ,
    NV_R_ARPDSYWJ_F_RVUCMMQW,
    NV_R_ARPDSYWJ_F_UBSXHLEI,
    NV_R_ARPDSYWJ_F_QIAPVCZD,
    NV_R_ARPDSYWJ_F_USAVQDXC,
    NV_R_ARPDSYWJ_F_QOSJJLDM,
    NV_R_ARPDSYWJ_F_BHUBJKFV,
    NV_R_ARPDSYWJ_F_ABWDODUN,
    NV_R_ARPDSYWJ_F_MHNEZFRQ,
    NV_R_ARPDSYWJ_F_VYFBOXTE,
    NV_R_ARPDSYWJ_F_MUQTRHDF,
    NV_R_ARPDSYWJ_F_XOXIJWVV,
    NV_R_ARPDSYWJ_F_LTGXRNQG,
    NV_R_ARPDSYWJ_F_EFDTLHDG,
    NV_R_ARPDSYWJ_F_TMQXZMBG,
    NV_R_ARPDSYWJ_F_SJDTJBJN,
    NV_R_ARPDSYWJ_F_PFFGLBNM,
    NV_R_ARPDSYWJ_F_PLZQIACE,
    NV_R_ARPDSYWJ_F_WIPYMAKE,
    NV_R_ARPDSYWJ_F_ZKBAFHLI,
    NV_R_ARPDSYWJ_F_XWFBCFRJ,
    NV_R_ARPDSYWJ_F_TCLCZDDH,
    NV_R_ARPDSYWJ_F_EKPVCEPD,
    NV_R_ARPDSYWJ_F_DWPQFURP,
    NV_R_ARPDSYWJ_F_BUPQWFXM,
    NV_R_ARPDSYWJ_F_FRJXMXJQ,
    NV_R_ARPDSYWJ_F_ORKXRJUG,
    NV_R_ARPDSYWJ_F_YSWEGNLP,
    NV_R_ARPDSYWJ_F_JEQVOZNZ,
    NV_R_UKKYSIAF,
    NV_R_UKKYSIAF_F_GHADRSVE,
    NV_R_UKKYSIAF_F_VKZSMGDV,
    NV_R_UKKYSIAF_F_RFNOQRKU,
    NV_R_UKKYSIAF_F_MDFSQBEN,
    NV_R_UKKYSIAF_F_BHJSHXVD,
    NV_R_UKKYSIAF_F_WCWOZBSP,
    NV_R_UKKYSIAF_F_FGONZWSR,
    NV_R_UKKYSIAF_F_CYZESORM,
    NV_R_UKKYSIAF_F_LGJGRYRP,
    NV_R_UKKYSIAF_F_GDMIBLMG,
    NV_R_UKKYSIAF_F_KSUINJNF,
    NV_R_UKKYSIAF_F_XESRATBZ,
    NV_R_UKKYSIAF_F_VNPLXLTZ,
    NV_R_UKKYSIAF_F_GLRHQTMZ,
    NV_R_UKKYSIAF_F_SCZMFHQZ,
    NV_R_UKKYSIAF_F_UGYRKLJM,
    NV_R_UKKYSIAF_F_HCNWPCOD,
    NV_R_UKKYSIAF_F_USJEEPOD,
    NV_R_UKKYSIAF_F_MBJACMOM,
    NV_R_UKKYSIAF_F_WZBWTNQM,
    NV_R_UKKYSIAF_F_MVLNUVFD,
    NV_R_UKKYSIAF_F_ROWAGQRG,
    NV_R_UKKYSIAF_F_XDTAJKAN,
    NV_R_UKKYSIAF_F_DCQQYUYD,
    NV_R_VHMXVILL,
    NV_R_VHMXVILL_F_NHWQWYKB,
    NV_R_VHMXVILL_F_JEOILTCQ,
    NV_R_VHMXVILL_F_VCNJDYQL,
    NV_R_VHMXVILL_F_ZZFUQOWD,
    NV_R_VHMXVILL_F_BXZRWUSZ,
    NV_R_VHMXVILL_F_JQYWCMFK,
    NV_R_VHMXVILL_F_OLUWWATW,
    NV_R_VHMXVILL_F_XSGCOOMD,
    NV_R_VHMXVILL_F_IUUVKLNG,
    NV_R_VHMXVILL_F_ZVJJIQUB,
    NV_R_VHMXVILL_F_KOSYPJGT,
    NV_R_VHMXVILL_F_ORBFRNVC,
    NV_R_VHMXVILL_F_TJMANTCF,
    NV_R_VHMXVILL_F_ZJRYUNWL,
    NV_R_VHMXVILL_F_BGXRTXWK,
    NV_R_VHMXVILL_F_ZIDYUYEN,
    NV_R_VHMXVILL_F_XTLDZEWY,
    NV_R_VHMXVILL_F_QCIHZJMY,
    NV_R_VHMXVILL_F_PLUELXER,
    NV_R_VHMXVILL_F_IRRZWHUK,
    NV_R_VHMXVILL_F_QELVSHSQ,
    NV_R_VHMXVILL_F_WXTBBGPW,
    NV_R_VHMXVILL_F_HJCIHHCP,
    NV_R_VHMXVILL_F_GINEOITN,
    NV_R_IZEOAHGN,
    NV_R_IZEOAHGN_F_SEQHVADV,
    NV_R_IZEOAHGN_F_UNQBRXUU,
    NV_R_IZEOAHGN_F_CMPLSAEF,
    NV_R_IZEOAHGN_F_KSWZIZDW,
    NV_R_IZEOAHGN_F_ARVINKLE,
    NV_R_IZEOAHGN_F_NSOEEUAM,
    NV_R_IZEOAHGN_F_SRZWQELT,
    NV_R_IZEOAHGN_F_ZGCQCLLK,
    NV_R_IZEOAHGN_F_CLUWFOJL,
    NV_R_IZEOAHGN_F_GHWDDLSU,
    NV_R_IZEOAHGN_F_HORPGXDM,
    NV_R_IZEOAHGN_F_ZJSPVULX,
    NV_R_IZEOAHGN_F_TSOFRVEJ,
    NV_R_IZEOAHGN_F_QPXJXFKE,
    NV_R_IZEOAHGN_F_PAQRWLDD,
    NV_R_IZEOAHGN_F_RDKYZRZZ,
    NV_R_IZEOAHGN_F_EYLWIOJU,
    NV_R_IZEOAHGN_F_PFOTNRKY,
    NV_R_IZEOAHGN_F_ELXFWGGQ,
    NV_R_IZEOAHGN_F_DLPPSKET,
    NV_R_IZEOAHGN_F_ILOUIJKE,
    NV_R_ESKJFLST,
    NV_R_ESKJFLST_F_VCGGYIVK,
    NV_R_ESKJFLST_F_QLRDKHUP,
    NV_R_ESKJFLST_F_HNSZUXUD,
    NV_R_ESKJFLST_F_MKLAZDQM,
    NV_R_ESKJFLST_F_TBQCPYQI,
    NV_R_ESKJFLST_F_PHXCTNID,
    NV_R_ESKJFLST_F_XIEMEZCU,
    NV_R_ESKJFLST_F_TZDPUEFQ,
    NV_R_ESKJFLST_F_DGOFWVOK,
    NV_R_ESKJFLST_F_DHRYWLZN,
    NV_R_ESKJFLST_F_GHRIOYVA,
    NV_R_ESKJFLST_F_WHWESNAV,
    NV_R_ESKJFLST_F_TWYNOCWN,
    NV_R_ESKJFLST_F_AFBITBKN,
    NV_R_ESKJFLST_F_LIRFOWMW,
    NV_R_ESKJFLST_F_BDXPAPPT,
    NV_R_ESKJFLST_F_BGPHLQUY,
    NV_R_ESKJFLST_F_PPCDQOOF,
    NV_R_ESKJFLST_F_ZEBIGWCX,
    NV_R_ESKJFLST_F_PZKRGPLQ,
    NV_R_ESKJFLST_F_QLGMWTBS,
    NV_R_ESKJFLST_F_OCTEAQBY,
    NV_R_ESKJFLST_F_UVDXWMSV,
    NV_R_ESKJFLST_F_MHCUVNCJ,
    NV_R_ESKJFLST_F_GZKODAIV,
    NV_R_ESKJFLST_F_ANNMRPQZ,
    NV_R_ESKJFLST_F_ZFIPTBSX,
    NV_R_ESKJFLST_F_IVNPCVVE,
    NV_R_ESKJFLST_F_NEFZXJFM,
    NV_R_ESKJFLST_F_CHSOJTFO,
    NV_R_FNOEGFHF,
    NV_R_FNOEGFHF_F_SSZLBTBR,
    NV_R_FNOEGFHF_F_KLUFQYAI,
    NV_R_FNOEGFHF_F_OTZTXNMA,
    NV_R_FNOEGFHF_F_ZPTMFEZU,
    NV_R_FNOEGFHF_F_JULXAXCU,
    NV_R_FNOEGFHF_F_FAFXSATM,
    NV_R_FNOEGFHF_F_MBGRXNCB,
    NV_R_FNOEGFHF_F_RMKXKYPS,
    NV_R_FNOEGFHF_F_SLERAOFN,
    NV_R_FNOEGFHF_F_PKDDTIRB,
    NV_R_FNOEGFHF_F_YRVXNZTC,
    NV_R_FNOEGFHF_F_OHLYGUFA,
    NV_R_FNOEGFHF_F_JFBZTFNH,
    NV_R_FNOEGFHF_F_FWYMYWIS,
    NV_R_FNOEGFHF_F_JAHTFFTS,
    NV_R_FNOEGFHF_F_WLEDNNBK,
    NV_R_FNOEGFHF_F_HZYWMTCE,
    NV_R_FNOEGFHF_F_YKYKPXPR,
    NV_R_JBKARAJO,
    NV_R_JBKARAJO_F_EQLQFTGH,
    NV_R_JBKARAJO_F_FKCQGEXL,
    NV_R_JBKARAJO_F_JOHJYAWP,
    NV_R_JBKARAJO_F_KBHOOAYF,
    NV_R_JBKARAJO_F_ZPMGXEGS,
    NV_R_JBKARAJO_F_YLQFUGYI,
    NV_R_JBKARAJO_F_HSSCRQVH,
    NV_R_JBKARAJO_F_QQNSDYPH,
    NV_R_JBKARAJO_F_GHSJEOPH,
    NV_R_JBKARAJO_F_HWPKKPOS,
    NV_R_JBKARAJO_F_BZGIJGTC,
    NV_R_JBKARAJO_F_PQYKPSKJ,
    NV_R_JBKARAJO_F_HAYDREJP,
    NV_R_JBKARAJO_F_YONASMHB,
    NV_R_JBKARAJO_F_UHWMFIRQ,
    NV_R_JBKARAJO_F_FRKPLMUI,
    NV_R_JBKARAJO_F_VSIKVIVJ,
    NV_R_JBKARAJO_F_GTNHTLPC,
    NV_R_JBKARAJO_F_PHTDXUBN,
    NV_R_JBKARAJO_F_DZRYQXUL,
    NV_R_JBKARAJO_F_SDQIJEBM,
    NV_R_JBKARAJO_F_WKNXUYEN,
    NV_R_JBKARAJO_F_HSWSYGNF,
    NV_R_JBKARAJO_F_NECPJIYP,
    NV_R_JBKARAJO_F_AAQXITLF,
    NV_R_JBKARAJO_F_PVTUWAZI,
    NV_R_JBKARAJO_F_OTMSOGWT,
    NV_R_JBKARAJO_F_KZUAUTMT,
    NV_R_JBKARAJO_F_ZBZWDAGO,
    NV_R_JBKARAJO_F_XGPRZMOL,
    NV_R_JBKARAJO_F_RKVLBNOK,
    NV_R_JBKARAJO_F_WZOVEWYN,
    NV_R_FJIYUNNQ,
    NV_R_FJIYUNNQ_F_NPNHWLCB,
    NV_R_FJIYUNNQ_F_IBAPEUDK,
    NV_R_FJIYUNNQ_F_ENYWDIMG,
    NV_R_FJIYUNNQ_F_LEFFWSZZ,
    NV_R_TFXQVLAN,
    NV_R_TFXQVLAN_F_BYVHZPQX,
    NV_R_TFXQVLAN_F_UFBJNCSP,
    NV_R_TFXQVLAN_F_QIAPVCZD,
    NV_R_TFXQVLAN_F_KMAXVQYV,
    NV_R_TFXQVLAN_F_ABWDODUN,
    NV_R_TFXQVLAN_F_RLTCBIRF,
    NV_R_TFXQVLAN_F_XOXIJWVV,
    NV_R_TFXQVLAN_F_ZGNJYOPN,
    NV_R_TFXQVLAN_F_SJDTJBJN,
    NV_R_TFXQVLAN_F_TUJFZICR,
    NV_R_TFXQVLAN_F_ZKBAFHLI,
    NV_R_TFXQVLAN_F_GHJPEMCH,
    NV_R_TFXQVLAN_F_DWPQFURP,
    NV_R_TFXQVLAN_F_ORNINPVJ,
    NV_R_TFXQVLAN_F_YSWEGNLP,
    NV_R_TFXQVLAN_F_HWIDPMKJ,
    NV_R_VZBXLZCT,
    NV_R_VZBXLZCT_F_MVCHRIIC,
    NV_R_VZBXLZCT_F_VXKVFDHV,
    NV_R_VZBXLZCT_F_SLVVMPNI,
    NV_R_VZBXLZCT_F_ZZDMIFVR,
    NV_R_VZBXLZCT_F_AWDPQRWV,
    NV_R_VZBXLZCT_F_JOFIQYJY,
    NV_R_VZBXLZCT_F_CSEIMAGX,
    NV_R_VZBXLZCT_F_UUMZCYVZ,
    NV_R_VZBXLZCT_F_NGWZDWXR,
    NV_R_VZBXLZCT_F_TOACAHUV,
    NV_R_VZBXLZCT_F_IWCUIWSQ,
    NV_R_VZBXLZCT_F_AZPGZVJI,
    NV_R_VZBXLZCT_F_PJVGMLHM,
    NV_R_VZBXLZCT_F_PZQDUIPA,
    NV_R_VZBXLZCT_F_AZTIZHJX,
    NV_R_VZBXLZCT_F_RYGOFPTC,
    NV_R_VZBXLZCT_F_XXPPJYOF,
    NV_R_VZBXLZCT_F_MORUAPFT,
    NV_R_VZBXLZCT_F_WYXYQPAB,
    NV_R_VZBXLZCT_F_WSWMUGTJ,
    NV_R_VZBXLZCT_F_SPNDDMHM,
    NV_R_VZBXLZCT_F_OGIZWIPZ,
    NV_R_VZBXLZCT_F_RGYRBELI,
    NV_R_VZBXLZCT_F_QULPMVIN,
    NV_R_WWYHZXYY,
    NV_R_WWYHZXYY_F_ADVIPOBT,
    NV_R_WWYHZXYY_F_SMZZYMVN,
    NV_R_WWYHZXYY_F_EOBWXOYT,
    NV_R_WWYHZXYY_F_CEDPJRUO,
    NV_R_WWYHZXYY_F_PSYXUVXN,
    NV_R_WWYHZXYY_F_NICBIVHT,
    NV_R_GIYEYJDF,
    NV_R_GIYEYJDF_F_HIHAFTKL,
    NV_R_GIYEYJDF_F_MAFIKAWU,
    NV_R_GIYEYJDF_F_ZMKFKZMR,
    NV_R_GIYEYJDF_F_KTPNHEDL,
    NV_R_GIYEYJDF_F_IASRQXAA,
    NV_R_GIYEYJDF_F_LEVUINXN,
    NV_R_GIYEYJDF_F_AXTKUHUL,
    NV_R_GIYEYJDF_F_KGDURNQY,
    NV_R_GIYEYJDF_F_FTRTNUVX,
    NV_R_GIYEYJDF_F_QLLFJRSD,
    NV_R_GIYEYJDF_F_RTOEXJZB,
    NV_R_GIYEYJDF_F_IFLRDQDB,
    NV_R_GIYEYJDF_F_WOJYUHZQ,
    NV_R_GIYEYJDF_F_DEIWFIXK,
    NV_R_GIYEYJDF_F_IOCHURJF,
    NV_R_GIYEYJDF_F_AWHQUNZW,
    NV_R_GIYEYJDF_F_VAIRXJAY,
    NV_R_GIYEYJDF_F_RSEPHDRQ,
    NV_R_GIYEYJDF_F_PVPIYWOY,
    NV_R_GIYEYJDF_F_AEHJJNUU,
    NV_R_GIYEYJDF_F_MYBLZEWY,
    NV_R_GIYEYJDF_F_OMFFCENF,
    NV_R_GIYEYJDF_F_TCGCOFJE,
    NV_R_GIYEYJDF_F_ZCYLKBSV,
    NV_R_GIYEYJDF_F_WXFBSLUS,
    NV_R_GIYEYJDF_F_VZHCHNXF,
    NV_R_GIYEYJDF_F_MOSLRSDR,
    NV_R_GIYEYJDF_F_JZSLCCSZ,
    NV_R_GIYEYJDF_F_DIQKXZEF,
    NV_R_GIYEYJDF_F_JYNILTFR,
    NV_R_GIYEYJDF_F_NCTKVOJU,
    NV_R_GIYEYJDF_F_DAETHFNV,
)

# Register definitions
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

NV_R_YIGWUDUH_F_CKVFCUZC = FieldMetadata(
    name='NV_R_YIGWUDUH_F_CKVFCUZC',
    msb=22,
    lsb=22,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_SSEPMUCC = FieldMetadata(
    name='NV_R_YIGWUDUH_F_SSEPMUCC',
    msb=14,
    lsb=14,
    register=NV_R_YIGWUDUH
)

NV_R_YIGWUDUH_F_NDAVUIAS = FieldMetadata(
    name='NV_R_YIGWUDUH_F_NDAVUIAS',
    msb=6,
    lsb=6,
    register=NV_R_YIGWUDUH
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

NV_R_CCGPKYYA_F_ZZHPUETU = FieldMetadata(
    name='NV_R_CCGPKYYA_F_ZZHPUETU',
    msb=6,
    lsb=6,
    register=NV_R_CCGPKYYA
)

NV_R_QUVQUPWB = RegisterMetadata(
    name='NV_R_QUVQUPWB',
    address=0x638,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_QUVQUPWB_F_WIQYAUTG = FieldMetadata(
    name='NV_R_QUVQUPWB_F_WIQYAUTG',
    msb=0,
    lsb=0,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_IMVWSWMC = FieldMetadata(
    name='NV_R_QUVQUPWB_F_IMVWSWMC',
    msb=16,
    lsb=16,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_ZUQRAMEI = FieldMetadata(
    name='NV_R_QUVQUPWB_F_ZUQRAMEI',
    msb=1,
    lsb=1,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_EJMRXLLB = FieldMetadata(
    name='NV_R_QUVQUPWB_F_EJMRXLLB',
    msb=17,
    lsb=17,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_TNRXBLAJ = FieldMetadata(
    name='NV_R_QUVQUPWB_F_TNRXBLAJ',
    msb=2,
    lsb=2,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_OTIIOJRE = FieldMetadata(
    name='NV_R_QUVQUPWB_F_OTIIOJRE',
    msb=18,
    lsb=18,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_KMVCGWPC = FieldMetadata(
    name='NV_R_QUVQUPWB_F_KMVCGWPC',
    msb=3,
    lsb=3,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_WATNYSMF = FieldMetadata(
    name='NV_R_QUVQUPWB_F_WATNYSMF',
    msb=19,
    lsb=19,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_IYOVZRDD = FieldMetadata(
    name='NV_R_QUVQUPWB_F_IYOVZRDD',
    msb=4,
    lsb=4,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_QLUEIMZI = FieldMetadata(
    name='NV_R_QUVQUPWB_F_QLUEIMZI',
    msb=20,
    lsb=20,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_AXTOXWRI = FieldMetadata(
    name='NV_R_QUVQUPWB_F_AXTOXWRI',
    msb=5,
    lsb=5,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_RKPHJIZG = FieldMetadata(
    name='NV_R_QUVQUPWB_F_RKPHJIZG',
    msb=21,
    lsb=21,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_USDTRVWY = FieldMetadata(
    name='NV_R_QUVQUPWB_F_USDTRVWY',
    msb=6,
    lsb=6,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_RKIOSCXE = FieldMetadata(
    name='NV_R_QUVQUPWB_F_RKIOSCXE',
    msb=22,
    lsb=22,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_IHXDLTAC = FieldMetadata(
    name='NV_R_QUVQUPWB_F_IHXDLTAC',
    msb=8,
    lsb=8,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_UOBXRRTO = FieldMetadata(
    name='NV_R_QUVQUPWB_F_UOBXRRTO',
    msb=24,
    lsb=24,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_CZOYZZNH = FieldMetadata(
    name='NV_R_QUVQUPWB_F_CZOYZZNH',
    msb=9,
    lsb=9,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_NTROPNLV = FieldMetadata(
    name='NV_R_QUVQUPWB_F_NTROPNLV',
    msb=25,
    lsb=25,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_NCUCLHBI = FieldMetadata(
    name='NV_R_QUVQUPWB_F_NCUCLHBI',
    msb=10,
    lsb=10,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_TONBFYBM = FieldMetadata(
    name='NV_R_QUVQUPWB_F_TONBFYBM',
    msb=26,
    lsb=26,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_QBPYKSPB = FieldMetadata(
    name='NV_R_QUVQUPWB_F_QBPYKSPB',
    msb=11,
    lsb=11,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_EMBORNAQ = FieldMetadata(
    name='NV_R_QUVQUPWB_F_EMBORNAQ',
    msb=27,
    lsb=27,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_NOXIKTRY = FieldMetadata(
    name='NV_R_QUVQUPWB_F_NOXIKTRY',
    msb=12,
    lsb=12,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_LPFHVXKX = FieldMetadata(
    name='NV_R_QUVQUPWB_F_LPFHVXKX',
    msb=28,
    lsb=28,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_SSRWXEYF = FieldMetadata(
    name='NV_R_QUVQUPWB_F_SSRWXEYF',
    msb=13,
    lsb=13,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_XHYYVIOW = FieldMetadata(
    name='NV_R_QUVQUPWB_F_XHYYVIOW',
    msb=29,
    lsb=29,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_PAHJXJEM = FieldMetadata(
    name='NV_R_QUVQUPWB_F_PAHJXJEM',
    msb=14,
    lsb=14,
    register=NV_R_QUVQUPWB
)

NV_R_QUVQUPWB_F_LUXJJNDP = FieldMetadata(
    name='NV_R_QUVQUPWB_F_LUXJJNDP',
    msb=30,
    lsb=30,
    register=NV_R_QUVQUPWB
)

NV_R_SOOJLMNX = RegisterMetadata(
    name='NV_R_SOOJLMNX',
    address=0x63c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_SOOJLMNX_F_PJGLONTU = FieldMetadata(
    name='NV_R_SOOJLMNX_F_PJGLONTU',
    msb=16,
    lsb=16,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_QTOMYCXN = FieldMetadata(
    name='NV_R_SOOJLMNX_F_QTOMYCXN',
    msb=24,
    lsb=24,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_FVRREFSB = FieldMetadata(
    name='NV_R_SOOJLMNX_F_FVRREFSB',
    msb=17,
    lsb=17,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_QRCGJVZC = FieldMetadata(
    name='NV_R_SOOJLMNX_F_QRCGJVZC',
    msb=25,
    lsb=25,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_JXJMYLYI = FieldMetadata(
    name='NV_R_SOOJLMNX_F_JXJMYLYI',
    msb=18,
    lsb=18,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_RCOHLGQP = FieldMetadata(
    name='NV_R_SOOJLMNX_F_RCOHLGQP',
    msb=26,
    lsb=26,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_NWJGYSGM = FieldMetadata(
    name='NV_R_SOOJLMNX_F_NWJGYSGM',
    msb=19,
    lsb=19,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_CHTZCNQV = FieldMetadata(
    name='NV_R_SOOJLMNX_F_CHTZCNQV',
    msb=27,
    lsb=27,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_ABWKHQVV = FieldMetadata(
    name='NV_R_SOOJLMNX_F_ABWKHQVV',
    msb=20,
    lsb=20,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_QZEZPABN = FieldMetadata(
    name='NV_R_SOOJLMNX_F_QZEZPABN',
    msb=28,
    lsb=28,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_LUFCKCUN = FieldMetadata(
    name='NV_R_SOOJLMNX_F_LUFCKCUN',
    msb=21,
    lsb=21,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_TYVWXMGK = FieldMetadata(
    name='NV_R_SOOJLMNX_F_TYVWXMGK',
    msb=29,
    lsb=29,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_LSBZLQJB = FieldMetadata(
    name='NV_R_SOOJLMNX_F_LSBZLQJB',
    msb=22,
    lsb=22,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_TMUOCGEX = FieldMetadata(
    name='NV_R_SOOJLMNX_F_TMUOCGEX',
    msb=30,
    lsb=30,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_XZWGPEQW = FieldMetadata(
    name='NV_R_SOOJLMNX_F_XZWGPEQW',
    msb=0,
    lsb=0,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_OTRCTJNH = FieldMetadata(
    name='NV_R_SOOJLMNX_F_OTRCTJNH',
    msb=1,
    lsb=1,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_OPRJIHTH = FieldMetadata(
    name='NV_R_SOOJLMNX_F_OPRJIHTH',
    msb=2,
    lsb=2,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_RHOSRWOJ = FieldMetadata(
    name='NV_R_SOOJLMNX_F_RHOSRWOJ',
    msb=3,
    lsb=3,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_UNUKIFFK = FieldMetadata(
    name='NV_R_SOOJLMNX_F_UNUKIFFK',
    msb=4,
    lsb=4,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_MBVUYMBH = FieldMetadata(
    name='NV_R_SOOJLMNX_F_MBVUYMBH',
    msb=5,
    lsb=5,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_OSHVWILX = FieldMetadata(
    name='NV_R_SOOJLMNX_F_OSHVWILX',
    msb=6,
    lsb=6,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_IDMEDOVI = FieldMetadata(
    name='NV_R_SOOJLMNX_F_IDMEDOVI',
    msb=8,
    lsb=8,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_XNWUVRYB = FieldMetadata(
    name='NV_R_SOOJLMNX_F_XNWUVRYB',
    msb=9,
    lsb=9,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_JHPPFRTR = FieldMetadata(
    name='NV_R_SOOJLMNX_F_JHPPFRTR',
    msb=10,
    lsb=10,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_JUEMQAVI = FieldMetadata(
    name='NV_R_SOOJLMNX_F_JUEMQAVI',
    msb=11,
    lsb=11,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_HJHMPXKT = FieldMetadata(
    name='NV_R_SOOJLMNX_F_HJHMPXKT',
    msb=12,
    lsb=12,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_QWBWTMYU = FieldMetadata(
    name='NV_R_SOOJLMNX_F_QWBWTMYU',
    msb=13,
    lsb=13,
    register=NV_R_SOOJLMNX
)

NV_R_SOOJLMNX_F_WPJEDLQH = FieldMetadata(
    name='NV_R_SOOJLMNX_F_WPJEDLQH',
    msb=14,
    lsb=14,
    register=NV_R_SOOJLMNX
)

NV_R_USHTVBFS = RegisterMetadata(
    name='NV_R_USHTVBFS',
    address=0x640,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_USHTVBFS_F_SLEIIBOU = FieldMetadata(
    name='NV_R_USHTVBFS_F_SLEIIBOU',
    msb=0,
    lsb=0,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_RKAFJFOP = FieldMetadata(
    name='NV_R_USHTVBFS_F_RKAFJFOP',
    msb=16,
    lsb=16,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_QKMUAVOQ = FieldMetadata(
    name='NV_R_USHTVBFS_F_QKMUAVOQ',
    msb=8,
    lsb=8,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_RKLJOFIX = FieldMetadata(
    name='NV_R_USHTVBFS_F_RKLJOFIX',
    msb=1,
    lsb=1,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_KAFYYCUA = FieldMetadata(
    name='NV_R_USHTVBFS_F_KAFYYCUA',
    msb=17,
    lsb=17,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_THKEMYSL = FieldMetadata(
    name='NV_R_USHTVBFS_F_THKEMYSL',
    msb=9,
    lsb=9,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_USPCMMKT = FieldMetadata(
    name='NV_R_USHTVBFS_F_USPCMMKT',
    msb=2,
    lsb=2,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_EHOUPSWV = FieldMetadata(
    name='NV_R_USHTVBFS_F_EHOUPSWV',
    msb=18,
    lsb=18,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_HNKJTLDT = FieldMetadata(
    name='NV_R_USHTVBFS_F_HNKJTLDT',
    msb=10,
    lsb=10,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_LSVITFOC = FieldMetadata(
    name='NV_R_USHTVBFS_F_LSVITFOC',
    msb=3,
    lsb=3,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_TYTDUCHH = FieldMetadata(
    name='NV_R_USHTVBFS_F_TYTDUCHH',
    msb=19,
    lsb=19,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_JXKTHJOA = FieldMetadata(
    name='NV_R_USHTVBFS_F_JXKTHJOA',
    msb=11,
    lsb=11,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_ATNKXMLT = FieldMetadata(
    name='NV_R_USHTVBFS_F_ATNKXMLT',
    msb=4,
    lsb=4,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_YDPHLRTM = FieldMetadata(
    name='NV_R_USHTVBFS_F_YDPHLRTM',
    msb=20,
    lsb=20,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_QIXBFBUF = FieldMetadata(
    name='NV_R_USHTVBFS_F_QIXBFBUF',
    msb=12,
    lsb=12,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_AMYYRZDN = FieldMetadata(
    name='NV_R_USHTVBFS_F_AMYYRZDN',
    msb=5,
    lsb=5,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_TYUZIMFA = FieldMetadata(
    name='NV_R_USHTVBFS_F_TYUZIMFA',
    msb=21,
    lsb=21,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_ITRCCABI = FieldMetadata(
    name='NV_R_USHTVBFS_F_ITRCCABI',
    msb=13,
    lsb=13,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_FBSMUEON = FieldMetadata(
    name='NV_R_USHTVBFS_F_FBSMUEON',
    msb=6,
    lsb=6,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_SPNFDCZM = FieldMetadata(
    name='NV_R_USHTVBFS_F_SPNFDCZM',
    msb=22,
    lsb=22,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_DRXXULAJ = FieldMetadata(
    name='NV_R_USHTVBFS_F_DRXXULAJ',
    msb=14,
    lsb=14,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_YGBPPKMV = FieldMetadata(
    name='NV_R_USHTVBFS_F_YGBPPKMV',
    msb=26,
    lsb=26,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_BEMIDIOQ = FieldMetadata(
    name='NV_R_USHTVBFS_F_BEMIDIOQ',
    msb=24,
    lsb=24,
    register=NV_R_USHTVBFS
)

NV_R_USHTVBFS_F_IUKBWJEG = FieldMetadata(
    name='NV_R_USHTVBFS_F_IUKBWJEG',
    msb=25,
    lsb=25,
    register=NV_R_USHTVBFS
)

NV_R_TXUCVHRP = RegisterMetadata(
    name='NV_R_TXUCVHRP',
    address=0x644,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_TXUCVHRP_F_AEOEXXTI = FieldMetadata(
    name='NV_R_TXUCVHRP_F_AEOEXXTI',
    msb=16,
    lsb=16,
    register=NV_R_TXUCVHRP
)

NV_R_TXUCVHRP_F_PDDCNVAL = FieldMetadata(
    name='NV_R_TXUCVHRP_F_PDDCNVAL',
    msb=0,
    lsb=0,
    register=NV_R_TXUCVHRP
)

NV_R_TXUCVHRP_F_GSHGPXTW = FieldMetadata(
    name='NV_R_TXUCVHRP_F_GSHGPXTW',
    msb=17,
    lsb=17,
    register=NV_R_TXUCVHRP
)

NV_R_TXUCVHRP_F_FYXEOLOI = FieldMetadata(
    name='NV_R_TXUCVHRP_F_FYXEOLOI',
    msb=1,
    lsb=1,
    register=NV_R_TXUCVHRP
)

NV_R_TXUCVHRP_F_UECGYRHG = FieldMetadata(
    name='NV_R_TXUCVHRP_F_UECGYRHG',
    msb=18,
    lsb=18,
    register=NV_R_TXUCVHRP
)

NV_R_TXUCVHRP_F_UNRCNUPF = FieldMetadata(
    name='NV_R_TXUCVHRP_F_UNRCNUPF',
    msb=2,
    lsb=2,
    register=NV_R_TXUCVHRP
)

NV_R_TXUCVHRP_F_BGAQAWUR = FieldMetadata(
    name='NV_R_TXUCVHRP_F_BGAQAWUR',
    msb=19,
    lsb=19,
    register=NV_R_TXUCVHRP
)

NV_R_TXUCVHRP_F_BSJFMBBJ = FieldMetadata(
    name='NV_R_TXUCVHRP_F_BSJFMBBJ',
    msb=3,
    lsb=3,
    register=NV_R_TXUCVHRP
)

NV_R_TXUCVHRP_F_TBVKWJVC = FieldMetadata(
    name='NV_R_TXUCVHRP_F_TBVKWJVC',
    msb=20,
    lsb=20,
    register=NV_R_TXUCVHRP
)

NV_R_TXUCVHRP_F_PPNOPFMF = FieldMetadata(
    name='NV_R_TXUCVHRP_F_PPNOPFMF',
    msb=4,
    lsb=4,
    register=NV_R_TXUCVHRP
)

NV_R_TXUCVHRP_F_SHZDJMKW = FieldMetadata(
    name='NV_R_TXUCVHRP_F_SHZDJMKW',
    msb=21,
    lsb=21,
    register=NV_R_TXUCVHRP
)

NV_R_TXUCVHRP_F_SUZIODBV = FieldMetadata(
    name='NV_R_TXUCVHRP_F_SUZIODBV',
    msb=5,
    lsb=5,
    register=NV_R_TXUCVHRP
)

NV_R_TXUCVHRP_F_DOBCVIIR = FieldMetadata(
    name='NV_R_TXUCVHRP_F_DOBCVIIR',
    msb=22,
    lsb=22,
    register=NV_R_TXUCVHRP
)

NV_R_TXUCVHRP_F_JWGBZBYK = FieldMetadata(
    name='NV_R_TXUCVHRP_F_JWGBZBYK',
    msb=6,
    lsb=6,
    register=NV_R_TXUCVHRP
)

NV_R_TXUCVHRP_F_IBTQZNBI = FieldMetadata(
    name='NV_R_TXUCVHRP_F_IBTQZNBI',
    msb=23,
    lsb=23,
    register=NV_R_TXUCVHRP
)

NV_R_TXUCVHRP_F_LQCPFHBN = FieldMetadata(
    name='NV_R_TXUCVHRP_F_LQCPFHBN',
    msb=7,
    lsb=7,
    register=NV_R_TXUCVHRP
)

NV_R_OOOWULGZ = RegisterMetadata(
    name='NV_R_OOOWULGZ',
    address=0x648,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_OOOWULGZ_F_GMLKXVPK = FieldMetadata(
    name='NV_R_OOOWULGZ_F_GMLKXVPK',
    msb=0,
    lsb=0,
    register=NV_R_OOOWULGZ
)

NV_R_OOOWULGZ_F_KLZJQJIL = FieldMetadata(
    name='NV_R_OOOWULGZ_F_KLZJQJIL',
    msb=16,
    lsb=16,
    register=NV_R_OOOWULGZ
)

NV_R_OOOWULGZ_F_WFGSECYG = FieldMetadata(
    name='NV_R_OOOWULGZ_F_WFGSECYG',
    msb=1,
    lsb=1,
    register=NV_R_OOOWULGZ
)

NV_R_OOOWULGZ_F_JRVWZFHN = FieldMetadata(
    name='NV_R_OOOWULGZ_F_JRVWZFHN',
    msb=17,
    lsb=17,
    register=NV_R_OOOWULGZ
)

NV_R_OOOWULGZ_F_NJOAWENL = FieldMetadata(
    name='NV_R_OOOWULGZ_F_NJOAWENL',
    msb=2,
    lsb=2,
    register=NV_R_OOOWULGZ
)

NV_R_OOOWULGZ_F_ZGDFPAHD = FieldMetadata(
    name='NV_R_OOOWULGZ_F_ZGDFPAHD',
    msb=18,
    lsb=18,
    register=NV_R_OOOWULGZ
)

NV_R_OOOWULGZ_F_TWKIHMRO = FieldMetadata(
    name='NV_R_OOOWULGZ_F_TWKIHMRO',
    msb=3,
    lsb=3,
    register=NV_R_OOOWULGZ
)

NV_R_OOOWULGZ_F_IWGXANVI = FieldMetadata(
    name='NV_R_OOOWULGZ_F_IWGXANVI',
    msb=19,
    lsb=19,
    register=NV_R_OOOWULGZ
)

NV_R_OOOWULGZ_F_ODERJZJQ = FieldMetadata(
    name='NV_R_OOOWULGZ_F_ODERJZJQ',
    msb=4,
    lsb=4,
    register=NV_R_OOOWULGZ
)

NV_R_OOOWULGZ_F_GDAIEDQX = FieldMetadata(
    name='NV_R_OOOWULGZ_F_GDAIEDQX',
    msb=20,
    lsb=20,
    register=NV_R_OOOWULGZ
)

NV_R_OOOWULGZ_F_QLHZBZGT = FieldMetadata(
    name='NV_R_OOOWULGZ_F_QLHZBZGT',
    msb=5,
    lsb=5,
    register=NV_R_OOOWULGZ
)

NV_R_OOOWULGZ_F_PQQMWGCO = FieldMetadata(
    name='NV_R_OOOWULGZ_F_PQQMWGCO',
    msb=21,
    lsb=21,
    register=NV_R_OOOWULGZ
)

NV_R_OOOWULGZ_F_DZBIJKFT = FieldMetadata(
    name='NV_R_OOOWULGZ_F_DZBIJKFT',
    msb=6,
    lsb=6,
    register=NV_R_OOOWULGZ
)

NV_R_OOOWULGZ_F_GXEWEQKY = FieldMetadata(
    name='NV_R_OOOWULGZ_F_GXEWEQKY',
    msb=22,
    lsb=22,
    register=NV_R_OOOWULGZ
)

NV_R_OOOWULGZ_F_IGUGRDBW = FieldMetadata(
    name='NV_R_OOOWULGZ_F_IGUGRDBW',
    msb=7,
    lsb=7,
    register=NV_R_OOOWULGZ
)

NV_R_OOOWULGZ_F_LIOBCLNL = FieldMetadata(
    name='NV_R_OOOWULGZ_F_LIOBCLNL',
    msb=23,
    lsb=23,
    register=NV_R_OOOWULGZ
)

NV_R_VNWZRJFS = RegisterMetadata(
    name='NV_R_VNWZRJFS',
    address=0x64c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_VNWZRJFS_F_ISAWNROL = FieldMetadata(
    name='NV_R_VNWZRJFS_F_ISAWNROL',
    msb=0,
    lsb=0,
    register=NV_R_VNWZRJFS
)

NV_R_VNWZRJFS_F_BVQUQPCB = FieldMetadata(
    name='NV_R_VNWZRJFS_F_BVQUQPCB',
    msb=1,
    lsb=1,
    register=NV_R_VNWZRJFS
)

NV_R_VNWZRJFS_F_HBOFUHKI = FieldMetadata(
    name='NV_R_VNWZRJFS_F_HBOFUHKI',
    msb=2,
    lsb=2,
    register=NV_R_VNWZRJFS
)

NV_R_VNWZRJFS_F_EKPZEMGL = FieldMetadata(
    name='NV_R_VNWZRJFS_F_EKPZEMGL',
    msb=3,
    lsb=3,
    register=NV_R_VNWZRJFS
)

NV_R_VNWZRJFS_F_HWHRXAYQ = FieldMetadata(
    name='NV_R_VNWZRJFS_F_HWHRXAYQ',
    msb=4,
    lsb=4,
    register=NV_R_VNWZRJFS
)

NV_R_VNWZRJFS_F_NGAMZWXM = FieldMetadata(
    name='NV_R_VNWZRJFS_F_NGAMZWXM',
    msb=5,
    lsb=5,
    register=NV_R_VNWZRJFS
)

NV_R_VNWZRJFS_F_BWDKTOIB = FieldMetadata(
    name='NV_R_VNWZRJFS_F_BWDKTOIB',
    msb=6,
    lsb=6,
    register=NV_R_VNWZRJFS
)

NV_R_VNWZRJFS_F_EUOLZXXB = FieldMetadata(
    name='NV_R_VNWZRJFS_F_EUOLZXXB',
    msb=7,
    lsb=7,
    register=NV_R_VNWZRJFS
)

NV_R_VNWZRJFS_F_IMDQVOVC = FieldMetadata(
    name='NV_R_VNWZRJFS_F_IMDQVOVC',
    msb=16,
    lsb=16,
    register=NV_R_VNWZRJFS
)

NV_R_VNWZRJFS_F_KALLMQVS = FieldMetadata(
    name='NV_R_VNWZRJFS_F_KALLMQVS',
    msb=17,
    lsb=17,
    register=NV_R_VNWZRJFS
)

NV_R_VNWZRJFS_F_NPTASYMU = FieldMetadata(
    name='NV_R_VNWZRJFS_F_NPTASYMU',
    msb=18,
    lsb=18,
    register=NV_R_VNWZRJFS
)

NV_R_VNWZRJFS_F_SBJDPUXE = FieldMetadata(
    name='NV_R_VNWZRJFS_F_SBJDPUXE',
    msb=19,
    lsb=19,
    register=NV_R_VNWZRJFS
)

NV_R_VNWZRJFS_F_IGFQMODB = FieldMetadata(
    name='NV_R_VNWZRJFS_F_IGFQMODB',
    msb=20,
    lsb=20,
    register=NV_R_VNWZRJFS
)

NV_R_VNWZRJFS_F_OMEFKFNK = FieldMetadata(
    name='NV_R_VNWZRJFS_F_OMEFKFNK',
    msb=21,
    lsb=21,
    register=NV_R_VNWZRJFS
)

NV_R_VNWZRJFS_F_JXXSTZJP = FieldMetadata(
    name='NV_R_VNWZRJFS_F_JXXSTZJP',
    msb=22,
    lsb=22,
    register=NV_R_VNWZRJFS
)

NV_R_VNWZRJFS_F_JSAZMAOX = FieldMetadata(
    name='NV_R_VNWZRJFS_F_JSAZMAOX',
    msb=23,
    lsb=23,
    register=NV_R_VNWZRJFS
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

NV_R_EPIENTQD_F_GPAFCECN = FieldMetadata(
    name='NV_R_EPIENTQD_F_GPAFCECN',
    msb=6,
    lsb=6,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_ZRNNEPKE = FieldMetadata(
    name='NV_R_EPIENTQD_F_ZRNNEPKE',
    msb=30,
    lsb=30,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_DGJQCFOB = FieldMetadata(
    name='NV_R_EPIENTQD_F_DGJQCFOB',
    msb=14,
    lsb=14,
    register=NV_R_EPIENTQD
)

NV_R_EPIENTQD_F_NFFILEOR = FieldMetadata(
    name='NV_R_EPIENTQD_F_NFFILEOR',
    msb=22,
    lsb=22,
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

NV_R_ANIRSEGB_F_WJREHQDC = FieldMetadata(
    name='NV_R_ANIRSEGB_F_WJREHQDC',
    msb=6,
    lsb=6,
    register=NV_R_ANIRSEGB
)

NV_R_ANIRSEGB_F_PONHSIDC = FieldMetadata(
    name='NV_R_ANIRSEGB_F_PONHSIDC',
    msb=14,
    lsb=14,
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

NV_R_ETZZJOQG_F_SSEPMUCC = FieldMetadata(
    name='NV_R_ETZZJOQG_F_SSEPMUCC',
    msb=14,
    lsb=14,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_JQUYSLAG = FieldMetadata(
    name='NV_R_ETZZJOQG_F_JQUYSLAG',
    msb=6,
    lsb=6,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_USGKTZKL = FieldMetadata(
    name='NV_R_ETZZJOQG_F_USGKTZKL',
    msb=30,
    lsb=30,
    register=NV_R_ETZZJOQG
)

NV_R_ETZZJOQG_F_GEDGITBD = FieldMetadata(
    name='NV_R_ETZZJOQG_F_GEDGITBD',
    msb=22,
    lsb=22,
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

NV_R_VHJZHKLK_F_CKVFCUZC = FieldMetadata(
    name='NV_R_VHJZHKLK_F_CKVFCUZC',
    msb=22,
    lsb=22,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_RHPFJQXO = FieldMetadata(
    name='NV_R_VHJZHKLK_F_RHPFJQXO',
    msb=6,
    lsb=6,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_BOLDFQEJ = FieldMetadata(
    name='NV_R_VHJZHKLK_F_BOLDFQEJ',
    msb=14,
    lsb=14,
    register=NV_R_VHJZHKLK
)

NV_R_VHJZHKLK_F_SABNWEWF = FieldMetadata(
    name='NV_R_VHJZHKLK_F_SABNWEWF',
    msb=30,
    lsb=30,
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

NV_R_KTLFSNMO_F_HKAVMBTC = FieldMetadata(
    name='NV_R_KTLFSNMO_F_HKAVMBTC',
    msb=14,
    lsb=14,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_IEGPLVJY = FieldMetadata(
    name='NV_R_KTLFSNMO_F_IEGPLVJY',
    msb=6,
    lsb=6,
    register=NV_R_KTLFSNMO
)

NV_R_KTLFSNMO_F_EIHHFQOJ = FieldMetadata(
    name='NV_R_KTLFSNMO_F_EIHHFQOJ',
    msb=30,
    lsb=30,
    register=NV_R_KTLFSNMO
)

NV_R_JYWTIGEE = RegisterMetadata(
    name='NV_R_JYWTIGEE',
    address=0x1d0,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_JYWTIGEE_F_EFZGTKCD = FieldMetadata(
    name='NV_R_JYWTIGEE_F_EFZGTKCD',
    msb=0,
    lsb=0,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_INCZVMMT = FieldMetadata(
    name='NV_R_JYWTIGEE_F_INCZVMMT',
    msb=24,
    lsb=24,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_WOZGKVLP = FieldMetadata(
    name='NV_R_JYWTIGEE_F_WOZGKVLP',
    msb=8,
    lsb=8,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_MJKVDMSV = FieldMetadata(
    name='NV_R_JYWTIGEE_F_MJKVDMSV',
    msb=16,
    lsb=16,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_HZCBTQHO = FieldMetadata(
    name='NV_R_JYWTIGEE_F_HZCBTQHO',
    msb=1,
    lsb=1,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_JXIORVOJ = FieldMetadata(
    name='NV_R_JYWTIGEE_F_JXIORVOJ',
    msb=25,
    lsb=25,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_GWKZPXPP = FieldMetadata(
    name='NV_R_JYWTIGEE_F_GWKZPXPP',
    msb=9,
    lsb=9,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_YZVTIBII = FieldMetadata(
    name='NV_R_JYWTIGEE_F_YZVTIBII',
    msb=17,
    lsb=17,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_AOUPFMWM = FieldMetadata(
    name='NV_R_JYWTIGEE_F_AOUPFMWM',
    msb=2,
    lsb=2,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_IWMKHWMQ = FieldMetadata(
    name='NV_R_JYWTIGEE_F_IWMKHWMQ',
    msb=26,
    lsb=26,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_DOSCKVIB = FieldMetadata(
    name='NV_R_JYWTIGEE_F_DOSCKVIB',
    msb=10,
    lsb=10,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_JKDMHRYW = FieldMetadata(
    name='NV_R_JYWTIGEE_F_JKDMHRYW',
    msb=18,
    lsb=18,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_BUKDWYIR = FieldMetadata(
    name='NV_R_JYWTIGEE_F_BUKDWYIR',
    msb=3,
    lsb=3,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_EWRMNVPA = FieldMetadata(
    name='NV_R_JYWTIGEE_F_EWRMNVPA',
    msb=27,
    lsb=27,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_CFHHNEHI = FieldMetadata(
    name='NV_R_JYWTIGEE_F_CFHHNEHI',
    msb=11,
    lsb=11,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_LFGUEAJB = FieldMetadata(
    name='NV_R_JYWTIGEE_F_LFGUEAJB',
    msb=19,
    lsb=19,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_JYCDHYQA = FieldMetadata(
    name='NV_R_JYWTIGEE_F_JYCDHYQA',
    msb=4,
    lsb=4,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_OLBLVJJG = FieldMetadata(
    name='NV_R_JYWTIGEE_F_OLBLVJJG',
    msb=28,
    lsb=28,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_MIAARQHW = FieldMetadata(
    name='NV_R_JYWTIGEE_F_MIAARQHW',
    msb=12,
    lsb=12,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_PNEJKZVG = FieldMetadata(
    name='NV_R_JYWTIGEE_F_PNEJKZVG',
    msb=20,
    lsb=20,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_MFJVZYXT = FieldMetadata(
    name='NV_R_JYWTIGEE_F_MFJVZYXT',
    msb=5,
    lsb=5,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_EIHLNEJD = FieldMetadata(
    name='NV_R_JYWTIGEE_F_EIHLNEJD',
    msb=29,
    lsb=29,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_LKDOUWBR = FieldMetadata(
    name='NV_R_JYWTIGEE_F_LKDOUWBR',
    msb=13,
    lsb=13,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_BMRNQEQR = FieldMetadata(
    name='NV_R_JYWTIGEE_F_BMRNQEQR',
    msb=21,
    lsb=21,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_QLHBESBL = FieldMetadata(
    name='NV_R_JYWTIGEE_F_QLHBESBL',
    msb=6,
    lsb=6,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_AZVYWRPI = FieldMetadata(
    name='NV_R_JYWTIGEE_F_AZVYWRPI',
    msb=30,
    lsb=30,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_DNDXEAUR = FieldMetadata(
    name='NV_R_JYWTIGEE_F_DNDXEAUR',
    msb=14,
    lsb=14,
    register=NV_R_JYWTIGEE
)

NV_R_JYWTIGEE_F_HMLGCTJH = FieldMetadata(
    name='NV_R_JYWTIGEE_F_HMLGCTJH',
    msb=22,
    lsb=22,
    register=NV_R_JYWTIGEE
)

NV_R_VHQAUYTU = RegisterMetadata(
    name='NV_R_VHQAUYTU',
    address=0x1d4,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_VHQAUYTU_F_DMHHIYYG = FieldMetadata(
    name='NV_R_VHQAUYTU_F_DMHHIYYG',
    msb=0,
    lsb=0,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_WLFUQEOX = FieldMetadata(
    name='NV_R_VHQAUYTU_F_WLFUQEOX',
    msb=1,
    lsb=1,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_DOQWJUMZ = FieldMetadata(
    name='NV_R_VHQAUYTU_F_DOQWJUMZ',
    msb=2,
    lsb=2,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_VWYOVUOW = FieldMetadata(
    name='NV_R_VHQAUYTU_F_VWYOVUOW',
    msb=3,
    lsb=3,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_MKJMODNE = FieldMetadata(
    name='NV_R_VHQAUYTU_F_MKJMODNE',
    msb=4,
    lsb=4,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_GHCXEZOU = FieldMetadata(
    name='NV_R_VHQAUYTU_F_GHCXEZOU',
    msb=5,
    lsb=5,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_JFTRHJLD = FieldMetadata(
    name='NV_R_VHQAUYTU_F_JFTRHJLD',
    msb=6,
    lsb=6,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_IUZCHAAK = FieldMetadata(
    name='NV_R_VHQAUYTU_F_IUZCHAAK',
    msb=16,
    lsb=16,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_UBVBHMBT = FieldMetadata(
    name='NV_R_VHQAUYTU_F_UBVBHMBT',
    msb=17,
    lsb=17,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_ZWQUBJCF = FieldMetadata(
    name='NV_R_VHQAUYTU_F_ZWQUBJCF',
    msb=18,
    lsb=18,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_MZELCHNP = FieldMetadata(
    name='NV_R_VHQAUYTU_F_MZELCHNP',
    msb=19,
    lsb=19,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_MUDBMRHR = FieldMetadata(
    name='NV_R_VHQAUYTU_F_MUDBMRHR',
    msb=20,
    lsb=20,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_SRTWZZCV = FieldMetadata(
    name='NV_R_VHQAUYTU_F_SRTWZZCV',
    msb=21,
    lsb=21,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_GGEBTEBS = FieldMetadata(
    name='NV_R_VHQAUYTU_F_GGEBTEBS',
    msb=22,
    lsb=22,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_OQHSVLUK = FieldMetadata(
    name='NV_R_VHQAUYTU_F_OQHSVLUK',
    msb=8,
    lsb=8,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_YLVQTBHQ = FieldMetadata(
    name='NV_R_VHQAUYTU_F_YLVQTBHQ',
    msb=9,
    lsb=9,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_TDDHCDLK = FieldMetadata(
    name='NV_R_VHQAUYTU_F_TDDHCDLK',
    msb=10,
    lsb=10,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_OKSUOQOO = FieldMetadata(
    name='NV_R_VHQAUYTU_F_OKSUOQOO',
    msb=11,
    lsb=11,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_UTIGWPCC = FieldMetadata(
    name='NV_R_VHQAUYTU_F_UTIGWPCC',
    msb=12,
    lsb=12,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_CIPJELPF = FieldMetadata(
    name='NV_R_VHQAUYTU_F_CIPJELPF',
    msb=13,
    lsb=13,
    register=NV_R_VHQAUYTU
)

NV_R_VHQAUYTU_F_XUUXMBQQ = FieldMetadata(
    name='NV_R_VHQAUYTU_F_XUUXMBQQ',
    msb=14,
    lsb=14,
    register=NV_R_VHQAUYTU
)

NV_R_UXVPSJYM = RegisterMetadata(
    name='NV_R_UXVPSJYM',
    address=0x1d8,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_UXVPSJYM_F_TFBBCCTL = FieldMetadata(
    name='NV_R_UXVPSJYM_F_TFBBCCTL',
    msb=24,
    lsb=24,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_KLRSVZEK = FieldMetadata(
    name='NV_R_UXVPSJYM_F_KLRSVZEK',
    msb=0,
    lsb=0,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_DVTIKISL = FieldMetadata(
    name='NV_R_UXVPSJYM_F_DVTIKISL',
    msb=16,
    lsb=16,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_MGCARYJK = FieldMetadata(
    name='NV_R_UXVPSJYM_F_MGCARYJK',
    msb=8,
    lsb=8,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_ENOXTIYF = FieldMetadata(
    name='NV_R_UXVPSJYM_F_ENOXTIYF',
    msb=25,
    lsb=25,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_MWZPHSWX = FieldMetadata(
    name='NV_R_UXVPSJYM_F_MWZPHSWX',
    msb=1,
    lsb=1,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_YEHHPPVR = FieldMetadata(
    name='NV_R_UXVPSJYM_F_YEHHPPVR',
    msb=17,
    lsb=17,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_GTAGSKOO = FieldMetadata(
    name='NV_R_UXVPSJYM_F_GTAGSKOO',
    msb=9,
    lsb=9,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_QNEGJJDP = FieldMetadata(
    name='NV_R_UXVPSJYM_F_QNEGJJDP',
    msb=26,
    lsb=26,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_ENKEXEGX = FieldMetadata(
    name='NV_R_UXVPSJYM_F_ENKEXEGX',
    msb=2,
    lsb=2,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_UCPMYUDX = FieldMetadata(
    name='NV_R_UXVPSJYM_F_UCPMYUDX',
    msb=18,
    lsb=18,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_UFDGAMUY = FieldMetadata(
    name='NV_R_UXVPSJYM_F_UFDGAMUY',
    msb=10,
    lsb=10,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_DSAQIRCI = FieldMetadata(
    name='NV_R_UXVPSJYM_F_DSAQIRCI',
    msb=27,
    lsb=27,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_CWSUDMIW = FieldMetadata(
    name='NV_R_UXVPSJYM_F_CWSUDMIW',
    msb=3,
    lsb=3,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_YJBVRKQS = FieldMetadata(
    name='NV_R_UXVPSJYM_F_YJBVRKQS',
    msb=19,
    lsb=19,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_UZYJFHDW = FieldMetadata(
    name='NV_R_UXVPSJYM_F_UZYJFHDW',
    msb=11,
    lsb=11,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_VIRLSRDR = FieldMetadata(
    name='NV_R_UXVPSJYM_F_VIRLSRDR',
    msb=28,
    lsb=28,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_RNTSYJIP = FieldMetadata(
    name='NV_R_UXVPSJYM_F_RNTSYJIP',
    msb=4,
    lsb=4,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_BWDPQJJL = FieldMetadata(
    name='NV_R_UXVPSJYM_F_BWDPQJJL',
    msb=20,
    lsb=20,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_DMPOGMYC = FieldMetadata(
    name='NV_R_UXVPSJYM_F_DMPOGMYC',
    msb=12,
    lsb=12,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_FZGPXQHM = FieldMetadata(
    name='NV_R_UXVPSJYM_F_FZGPXQHM',
    msb=29,
    lsb=29,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_UJZEGWXJ = FieldMetadata(
    name='NV_R_UXVPSJYM_F_UJZEGWXJ',
    msb=5,
    lsb=5,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_IIUKBYQV = FieldMetadata(
    name='NV_R_UXVPSJYM_F_IIUKBYQV',
    msb=21,
    lsb=21,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_CRVYWQOI = FieldMetadata(
    name='NV_R_UXVPSJYM_F_CRVYWQOI',
    msb=13,
    lsb=13,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_HSDLEIRV = FieldMetadata(
    name='NV_R_UXVPSJYM_F_HSDLEIRV',
    msb=30,
    lsb=30,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_SWROQSFL = FieldMetadata(
    name='NV_R_UXVPSJYM_F_SWROQSFL',
    msb=6,
    lsb=6,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_FRUKUWBG = FieldMetadata(
    name='NV_R_UXVPSJYM_F_FRUKUWBG',
    msb=22,
    lsb=22,
    register=NV_R_UXVPSJYM
)

NV_R_UXVPSJYM_F_DRUFIGIT = FieldMetadata(
    name='NV_R_UXVPSJYM_F_DRUFIGIT',
    msb=14,
    lsb=14,
    register=NV_R_UXVPSJYM
)

NV_R_INVYUCZA = RegisterMetadata(
    name='NV_R_INVYUCZA',
    address=0x1dc,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_INVYUCZA_F_YGYCDTHC = FieldMetadata(
    name='NV_R_INVYUCZA_F_YGYCDTHC',
    msb=8,
    lsb=8,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_WYWNQOHS = FieldMetadata(
    name='NV_R_INVYUCZA_F_WYWNQOHS',
    msb=9,
    lsb=9,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_AFUDWTZU = FieldMetadata(
    name='NV_R_INVYUCZA_F_AFUDWTZU',
    msb=10,
    lsb=10,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_PDWRKOOU = FieldMetadata(
    name='NV_R_INVYUCZA_F_PDWRKOOU',
    msb=11,
    lsb=11,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_DGIDUDHK = FieldMetadata(
    name='NV_R_INVYUCZA_F_DGIDUDHK',
    msb=12,
    lsb=12,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_CENKKGXO = FieldMetadata(
    name='NV_R_INVYUCZA_F_CENKKGXO',
    msb=13,
    lsb=13,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_FQOMJFOZ = FieldMetadata(
    name='NV_R_INVYUCZA_F_FQOMJFOZ',
    msb=14,
    lsb=14,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_BVJJZQWJ = FieldMetadata(
    name='NV_R_INVYUCZA_F_BVJJZQWJ',
    msb=16,
    lsb=16,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_FSVXMCDF = FieldMetadata(
    name='NV_R_INVYUCZA_F_FSVXMCDF',
    msb=17,
    lsb=17,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_YQUMTZXS = FieldMetadata(
    name='NV_R_INVYUCZA_F_YQUMTZXS',
    msb=18,
    lsb=18,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_YTZUELKI = FieldMetadata(
    name='NV_R_INVYUCZA_F_YTZUELKI',
    msb=19,
    lsb=19,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_GWSOZKVQ = FieldMetadata(
    name='NV_R_INVYUCZA_F_GWSOZKVQ',
    msb=20,
    lsb=20,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_CPOKDGOD = FieldMetadata(
    name='NV_R_INVYUCZA_F_CPOKDGOD',
    msb=21,
    lsb=21,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_YVIZOHFL = FieldMetadata(
    name='NV_R_INVYUCZA_F_YVIZOHFL',
    msb=22,
    lsb=22,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_GYLCXLKN = FieldMetadata(
    name='NV_R_INVYUCZA_F_GYLCXLKN',
    msb=0,
    lsb=0,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_JISVTTJX = FieldMetadata(
    name='NV_R_INVYUCZA_F_JISVTTJX',
    msb=1,
    lsb=1,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_NDFWVRQC = FieldMetadata(
    name='NV_R_INVYUCZA_F_NDFWVRQC',
    msb=2,
    lsb=2,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_VHQSXFWL = FieldMetadata(
    name='NV_R_INVYUCZA_F_VHQSXFWL',
    msb=3,
    lsb=3,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_RKBVUCOO = FieldMetadata(
    name='NV_R_INVYUCZA_F_RKBVUCOO',
    msb=4,
    lsb=4,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_BWNLHSTU = FieldMetadata(
    name='NV_R_INVYUCZA_F_BWNLHSTU',
    msb=5,
    lsb=5,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_WAUWUSZS = FieldMetadata(
    name='NV_R_INVYUCZA_F_WAUWUSZS',
    msb=6,
    lsb=6,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_MQCWCLPV = FieldMetadata(
    name='NV_R_INVYUCZA_F_MQCWCLPV',
    msb=25,
    lsb=25,
    register=NV_R_INVYUCZA
)

NV_R_INVYUCZA_F_BIVRZSIR = FieldMetadata(
    name='NV_R_INVYUCZA_F_BIVRZSIR',
    msb=24,
    lsb=24,
    register=NV_R_INVYUCZA
)

NV_R_DIFKGJWN = RegisterMetadata(
    name='NV_R_DIFKGJWN',
    address=0x1e0,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_DIFKGJWN_F_NULALQEB = FieldMetadata(
    name='NV_R_DIFKGJWN_F_NULALQEB',
    msb=16,
    lsb=16,
    register=NV_R_DIFKGJWN
)

NV_R_DIFKGJWN_F_FKCQGEXL = FieldMetadata(
    name='NV_R_DIFKGJWN_F_FKCQGEXL',
    msb=0,
    lsb=0,
    register=NV_R_DIFKGJWN
)

NV_R_DIFKGJWN_F_ZEBTRXQU = FieldMetadata(
    name='NV_R_DIFKGJWN_F_ZEBTRXQU',
    msb=17,
    lsb=17,
    register=NV_R_DIFKGJWN
)

NV_R_DIFKGJWN_F_YLQFUGYI = FieldMetadata(
    name='NV_R_DIFKGJWN_F_YLQFUGYI',
    msb=1,
    lsb=1,
    register=NV_R_DIFKGJWN
)

NV_R_DIFKGJWN_F_IAHMLKVO = FieldMetadata(
    name='NV_R_DIFKGJWN_F_IAHMLKVO',
    msb=18,
    lsb=18,
    register=NV_R_DIFKGJWN
)

NV_R_DIFKGJWN_F_HWPKKPOS = FieldMetadata(
    name='NV_R_DIFKGJWN_F_HWPKKPOS',
    msb=2,
    lsb=2,
    register=NV_R_DIFKGJWN
)

NV_R_DIFKGJWN_F_FEPNOJNG = FieldMetadata(
    name='NV_R_DIFKGJWN_F_FEPNOJNG',
    msb=19,
    lsb=19,
    register=NV_R_DIFKGJWN
)

NV_R_DIFKGJWN_F_YONASMHB = FieldMetadata(
    name='NV_R_DIFKGJWN_F_YONASMHB',
    msb=3,
    lsb=3,
    register=NV_R_DIFKGJWN
)

NV_R_DIFKGJWN_F_UNAHFNSJ = FieldMetadata(
    name='NV_R_DIFKGJWN_F_UNAHFNSJ',
    msb=20,
    lsb=20,
    register=NV_R_DIFKGJWN
)

NV_R_DIFKGJWN_F_GTNHTLPC = FieldMetadata(
    name='NV_R_DIFKGJWN_F_GTNHTLPC',
    msb=4,
    lsb=4,
    register=NV_R_DIFKGJWN
)

NV_R_DIFKGJWN_F_UKSHBIPB = FieldMetadata(
    name='NV_R_DIFKGJWN_F_UKSHBIPB',
    msb=21,
    lsb=21,
    register=NV_R_DIFKGJWN
)

NV_R_DIFKGJWN_F_WKNXUYEN = FieldMetadata(
    name='NV_R_DIFKGJWN_F_WKNXUYEN',
    msb=5,
    lsb=5,
    register=NV_R_DIFKGJWN
)

NV_R_DIFKGJWN_F_HBQYKTFR = FieldMetadata(
    name='NV_R_DIFKGJWN_F_HBQYKTFR',
    msb=22,
    lsb=22,
    register=NV_R_DIFKGJWN
)

NV_R_DIFKGJWN_F_PVTUWAZI = FieldMetadata(
    name='NV_R_DIFKGJWN_F_PVTUWAZI',
    msb=6,
    lsb=6,
    register=NV_R_DIFKGJWN
)

NV_R_DIFKGJWN_F_IIJOINOH = FieldMetadata(
    name='NV_R_DIFKGJWN_F_IIJOINOH',
    msb=23,
    lsb=23,
    register=NV_R_DIFKGJWN
)

NV_R_DIFKGJWN_F_XGPRZMOL = FieldMetadata(
    name='NV_R_DIFKGJWN_F_XGPRZMOL',
    msb=7,
    lsb=7,
    register=NV_R_DIFKGJWN
)

NV_R_OHXNDKZO = RegisterMetadata(
    name='NV_R_OHXNDKZO',
    address=0x1e4,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_OHXNDKZO_F_KBHOOAYF = FieldMetadata(
    name='NV_R_OHXNDKZO_F_KBHOOAYF',
    msb=0,
    lsb=0,
    register=NV_R_OHXNDKZO
)

NV_R_OHXNDKZO_F_QQNSDYPH = FieldMetadata(
    name='NV_R_OHXNDKZO_F_QQNSDYPH',
    msb=1,
    lsb=1,
    register=NV_R_OHXNDKZO
)

NV_R_OHXNDKZO_F_PQYKPSKJ = FieldMetadata(
    name='NV_R_OHXNDKZO_F_PQYKPSKJ',
    msb=2,
    lsb=2,
    register=NV_R_OHXNDKZO
)

NV_R_OHXNDKZO_F_FRKPLMUI = FieldMetadata(
    name='NV_R_OHXNDKZO_F_FRKPLMUI',
    msb=3,
    lsb=3,
    register=NV_R_OHXNDKZO
)

NV_R_OHXNDKZO_F_DZRYQXUL = FieldMetadata(
    name='NV_R_OHXNDKZO_F_DZRYQXUL',
    msb=4,
    lsb=4,
    register=NV_R_OHXNDKZO
)

NV_R_OHXNDKZO_F_NECPJIYP = FieldMetadata(
    name='NV_R_OHXNDKZO_F_NECPJIYP',
    msb=5,
    lsb=5,
    register=NV_R_OHXNDKZO
)

NV_R_OHXNDKZO_F_KZUAUTMT = FieldMetadata(
    name='NV_R_OHXNDKZO_F_KZUAUTMT',
    msb=6,
    lsb=6,
    register=NV_R_OHXNDKZO
)

NV_R_OHXNDKZO_F_WZOVEWYN = FieldMetadata(
    name='NV_R_OHXNDKZO_F_WZOVEWYN',
    msb=7,
    lsb=7,
    register=NV_R_OHXNDKZO
)

NV_R_OHXNDKZO_F_LXOBZGHC = FieldMetadata(
    name='NV_R_OHXNDKZO_F_LXOBZGHC',
    msb=16,
    lsb=16,
    register=NV_R_OHXNDKZO
)

NV_R_OHXNDKZO_F_JMTEDEAN = FieldMetadata(
    name='NV_R_OHXNDKZO_F_JMTEDEAN',
    msb=17,
    lsb=17,
    register=NV_R_OHXNDKZO
)

NV_R_OHXNDKZO_F_IRWGTEAR = FieldMetadata(
    name='NV_R_OHXNDKZO_F_IRWGTEAR',
    msb=18,
    lsb=18,
    register=NV_R_OHXNDKZO
)

NV_R_OHXNDKZO_F_TMEABNQC = FieldMetadata(
    name='NV_R_OHXNDKZO_F_TMEABNQC',
    msb=19,
    lsb=19,
    register=NV_R_OHXNDKZO
)

NV_R_OHXNDKZO_F_LSGZLVHX = FieldMetadata(
    name='NV_R_OHXNDKZO_F_LSGZLVHX',
    msb=20,
    lsb=20,
    register=NV_R_OHXNDKZO
)

NV_R_OHXNDKZO_F_SUNVVQKE = FieldMetadata(
    name='NV_R_OHXNDKZO_F_SUNVVQKE',
    msb=21,
    lsb=21,
    register=NV_R_OHXNDKZO
)

NV_R_OHXNDKZO_F_MBAGPZXG = FieldMetadata(
    name='NV_R_OHXNDKZO_F_MBAGPZXG',
    msb=22,
    lsb=22,
    register=NV_R_OHXNDKZO
)

NV_R_OHXNDKZO_F_ODQHEEUA = FieldMetadata(
    name='NV_R_OHXNDKZO_F_ODQHEEUA',
    msb=23,
    lsb=23,
    register=NV_R_OHXNDKZO
)

NV_R_YOXQNARR = RegisterMetadata(
    name='NV_R_YOXQNARR',
    address=0x824,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_YOXQNARR_F_LXTYFTPY = FieldMetadata(
    name='NV_R_YOXQNARR_F_LXTYFTPY',
    msb=0,
    lsb=0,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_YODTZACI = FieldMetadata(
    name='NV_R_YOXQNARR_F_YODTZACI',
    msb=8,
    lsb=8,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_IPBTLEBX = FieldMetadata(
    name='NV_R_YOXQNARR_F_IPBTLEBX',
    msb=16,
    lsb=16,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_YWUVNVTK = FieldMetadata(
    name='NV_R_YOXQNARR_F_YWUVNVTK',
    msb=1,
    lsb=1,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_NCACRWHK = FieldMetadata(
    name='NV_R_YOXQNARR_F_NCACRWHK',
    msb=9,
    lsb=9,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_ULLZUFTO = FieldMetadata(
    name='NV_R_YOXQNARR_F_ULLZUFTO',
    msb=17,
    lsb=17,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_HSWZNWQR = FieldMetadata(
    name='NV_R_YOXQNARR_F_HSWZNWQR',
    msb=2,
    lsb=2,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_FFRPXOQA = FieldMetadata(
    name='NV_R_YOXQNARR_F_FFRPXOQA',
    msb=10,
    lsb=10,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_RVNRIEFE = FieldMetadata(
    name='NV_R_YOXQNARR_F_RVNRIEFE',
    msb=18,
    lsb=18,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_TNPNUXBZ = FieldMetadata(
    name='NV_R_YOXQNARR_F_TNPNUXBZ',
    msb=3,
    lsb=3,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_IDQORZEN = FieldMetadata(
    name='NV_R_YOXQNARR_F_IDQORZEN',
    msb=11,
    lsb=11,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_OTNGPIEK = FieldMetadata(
    name='NV_R_YOXQNARR_F_OTNGPIEK',
    msb=19,
    lsb=19,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_AJKKMIVL = FieldMetadata(
    name='NV_R_YOXQNARR_F_AJKKMIVL',
    msb=4,
    lsb=4,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_ZSIEXSHQ = FieldMetadata(
    name='NV_R_YOXQNARR_F_ZSIEXSHQ',
    msb=12,
    lsb=12,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_ZXIDKDTW = FieldMetadata(
    name='NV_R_YOXQNARR_F_ZXIDKDTW',
    msb=20,
    lsb=20,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_OZYLDQNS = FieldMetadata(
    name='NV_R_YOXQNARR_F_OZYLDQNS',
    msb=5,
    lsb=5,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_SIXDIVOL = FieldMetadata(
    name='NV_R_YOXQNARR_F_SIXDIVOL',
    msb=13,
    lsb=13,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_PGICNWKE = FieldMetadata(
    name='NV_R_YOXQNARR_F_PGICNWKE',
    msb=21,
    lsb=21,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_HHFLQEOY = FieldMetadata(
    name='NV_R_YOXQNARR_F_HHFLQEOY',
    msb=6,
    lsb=6,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_HBAAERYM = FieldMetadata(
    name='NV_R_YOXQNARR_F_HBAAERYM',
    msb=14,
    lsb=14,
    register=NV_R_YOXQNARR
)

NV_R_YOXQNARR_F_ZFHZLSEB = FieldMetadata(
    name='NV_R_YOXQNARR_F_ZFHZLSEB',
    msb=22,
    lsb=22,
    register=NV_R_YOXQNARR
)

NV_R_PUIWZUDC = RegisterMetadata(
    name='NV_R_PUIWZUDC',
    address=0x828,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_PUIWZUDC_F_ABDPNLJW = FieldMetadata(
    name='NV_R_PUIWZUDC_F_ABDPNLJW',
    msb=0,
    lsb=0,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_PUNKZFGK = FieldMetadata(
    name='NV_R_PUIWZUDC_F_PUNKZFGK',
    msb=16,
    lsb=16,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_JVZQFAQN = FieldMetadata(
    name='NV_R_PUIWZUDC_F_JVZQFAQN',
    msb=8,
    lsb=8,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_QHOZOTBM = FieldMetadata(
    name='NV_R_PUIWZUDC_F_QHOZOTBM',
    msb=24,
    lsb=24,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_BNZSQIFR = FieldMetadata(
    name='NV_R_PUIWZUDC_F_BNZSQIFR',
    msb=1,
    lsb=1,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_IIXBSFLJ = FieldMetadata(
    name='NV_R_PUIWZUDC_F_IIXBSFLJ',
    msb=17,
    lsb=17,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_JECPESUS = FieldMetadata(
    name='NV_R_PUIWZUDC_F_JECPESUS',
    msb=9,
    lsb=9,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_NQKWXYLK = FieldMetadata(
    name='NV_R_PUIWZUDC_F_NQKWXYLK',
    msb=25,
    lsb=25,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_IQXEFCSO = FieldMetadata(
    name='NV_R_PUIWZUDC_F_IQXEFCSO',
    msb=2,
    lsb=2,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_DOBXZEQA = FieldMetadata(
    name='NV_R_PUIWZUDC_F_DOBXZEQA',
    msb=18,
    lsb=18,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_QHGRCESR = FieldMetadata(
    name='NV_R_PUIWZUDC_F_QHGRCESR',
    msb=10,
    lsb=10,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_GATVWNSQ = FieldMetadata(
    name='NV_R_PUIWZUDC_F_GATVWNSQ',
    msb=26,
    lsb=26,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_RGSZQKFJ = FieldMetadata(
    name='NV_R_PUIWZUDC_F_RGSZQKFJ',
    msb=3,
    lsb=3,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_NTMNQEWN = FieldMetadata(
    name='NV_R_PUIWZUDC_F_NTMNQEWN',
    msb=19,
    lsb=19,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_YETFCUNI = FieldMetadata(
    name='NV_R_PUIWZUDC_F_YETFCUNI',
    msb=11,
    lsb=11,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_OBIEANLE = FieldMetadata(
    name='NV_R_PUIWZUDC_F_OBIEANLE',
    msb=27,
    lsb=27,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_CUDOINML = FieldMetadata(
    name='NV_R_PUIWZUDC_F_CUDOINML',
    msb=4,
    lsb=4,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_OROZORRB = FieldMetadata(
    name='NV_R_PUIWZUDC_F_OROZORRB',
    msb=20,
    lsb=20,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_OKKHJKPW = FieldMetadata(
    name='NV_R_PUIWZUDC_F_OKKHJKPW',
    msb=12,
    lsb=12,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_GVXZSRPL = FieldMetadata(
    name='NV_R_PUIWZUDC_F_GVXZSRPL',
    msb=28,
    lsb=28,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_MMQKCXTR = FieldMetadata(
    name='NV_R_PUIWZUDC_F_MMQKCXTR',
    msb=5,
    lsb=5,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_WRYSRWEJ = FieldMetadata(
    name='NV_R_PUIWZUDC_F_WRYSRWEJ',
    msb=21,
    lsb=21,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_KYYBTZAH = FieldMetadata(
    name='NV_R_PUIWZUDC_F_KYYBTZAH',
    msb=13,
    lsb=13,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_XPCNXEAU = FieldMetadata(
    name='NV_R_PUIWZUDC_F_XPCNXEAU',
    msb=29,
    lsb=29,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_ZWFITLVA = FieldMetadata(
    name='NV_R_PUIWZUDC_F_ZWFITLVA',
    msb=6,
    lsb=6,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_IFNOAULU = FieldMetadata(
    name='NV_R_PUIWZUDC_F_IFNOAULU',
    msb=22,
    lsb=22,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_PIGONQTN = FieldMetadata(
    name='NV_R_PUIWZUDC_F_PIGONQTN',
    msb=14,
    lsb=14,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_OOTEXLKV = FieldMetadata(
    name='NV_R_PUIWZUDC_F_OOTEXLKV',
    msb=30,
    lsb=30,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_ZUHLCYRB = FieldMetadata(
    name='NV_R_PUIWZUDC_F_ZUHLCYRB',
    msb=7,
    lsb=7,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_CYWIKMPS = FieldMetadata(
    name='NV_R_PUIWZUDC_F_CYWIKMPS',
    msb=23,
    lsb=23,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_ELFXKQIN = FieldMetadata(
    name='NV_R_PUIWZUDC_F_ELFXKQIN',
    msb=15,
    lsb=15,
    register=NV_R_PUIWZUDC
)

NV_R_PUIWZUDC_F_FOYABZPO = FieldMetadata(
    name='NV_R_PUIWZUDC_F_FOYABZPO',
    msb=31,
    lsb=31,
    register=NV_R_PUIWZUDC
)

NV_R_ZDZLMUKR = RegisterMetadata(
    name='NV_R_ZDZLMUKR',
    address=0x82c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_ZDZLMUKR_F_DCVZTBVP = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_DCVZTBVP',
    msb=0,
    lsb=0,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_AUJEDMYV = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_AUJEDMYV',
    msb=8,
    lsb=8,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_UUEWJYVI = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_UUEWJYVI',
    msb=16,
    lsb=16,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_KWJRHKNK = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_KWJRHKNK',
    msb=24,
    lsb=24,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_OVGJUXAP = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_OVGJUXAP',
    msb=1,
    lsb=1,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_QEFSVZVO = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_QEFSVZVO',
    msb=9,
    lsb=9,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_PNETFVEN = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_PNETFVEN',
    msb=17,
    lsb=17,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_LWZJDYQO = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_LWZJDYQO',
    msb=25,
    lsb=25,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_RDZVAOLK = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_RDZVAOLK',
    msb=2,
    lsb=2,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_IPPPPZYB = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_IPPPPZYB',
    msb=10,
    lsb=10,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_JFPYOBDJ = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_JFPYOBDJ',
    msb=18,
    lsb=18,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_XDLVFQJA = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_XDLVFQJA',
    msb=26,
    lsb=26,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_WWHYFDME = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_WWHYFDME',
    msb=3,
    lsb=3,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_POGKDWNN = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_POGKDWNN',
    msb=11,
    lsb=11,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_DOMWSXLD = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_DOMWSXLD',
    msb=19,
    lsb=19,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_YMKBYMLD = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_YMKBYMLD',
    msb=27,
    lsb=27,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_ZBOPRKFF = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_ZBOPRKFF',
    msb=4,
    lsb=4,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_BBBUJFFX = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_BBBUJFFX',
    msb=12,
    lsb=12,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_AMLTQEPJ = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_AMLTQEPJ',
    msb=20,
    lsb=20,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_LVBXXZUC = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_LVBXXZUC',
    msb=28,
    lsb=28,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_SXSKGARG = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_SXSKGARG',
    msb=5,
    lsb=5,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_IAIWHGLE = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_IAIWHGLE',
    msb=13,
    lsb=13,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_YZQMJXWQ = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_YZQMJXWQ',
    msb=21,
    lsb=21,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_MQOMFOMP = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_MQOMFOMP',
    msb=29,
    lsb=29,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_KFQNQGDQ = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_KFQNQGDQ',
    msb=6,
    lsb=6,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_HHCVCIBF = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_HHCVCIBF',
    msb=14,
    lsb=14,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_SWFAINTV = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_SWFAINTV',
    msb=22,
    lsb=22,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_UMLOLDUM = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_UMLOLDUM',
    msb=30,
    lsb=30,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_NIQKJZPF = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_NIQKJZPF',
    msb=7,
    lsb=7,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_NVIYZGCJ = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_NVIYZGCJ',
    msb=15,
    lsb=15,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_JBZKRMEW = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_JBZKRMEW',
    msb=23,
    lsb=23,
    register=NV_R_ZDZLMUKR
)

NV_R_ZDZLMUKR_F_PYTALWFF = FieldMetadata(
    name='NV_R_ZDZLMUKR_F_PYTALWFF',
    msb=31,
    lsb=31,
    register=NV_R_ZDZLMUKR
)

NV_R_ORJFNXKK = RegisterMetadata(
    name='NV_R_ORJFNXKK',
    address=0x830,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_ORJFNXKK_F_EWUEIUFQ = FieldMetadata(
    name='NV_R_ORJFNXKK_F_EWUEIUFQ',
    msb=0,
    lsb=0,
    register=NV_R_ORJFNXKK
)

NV_R_ORJFNXKK_F_WGRGQHMN = FieldMetadata(
    name='NV_R_ORJFNXKK_F_WGRGQHMN',
    msb=16,
    lsb=16,
    register=NV_R_ORJFNXKK
)

NV_R_ORJFNXKK_F_SVVREGFA = FieldMetadata(
    name='NV_R_ORJFNXKK_F_SVVREGFA',
    msb=1,
    lsb=1,
    register=NV_R_ORJFNXKK
)

NV_R_ORJFNXKK_F_TVLGAQTZ = FieldMetadata(
    name='NV_R_ORJFNXKK_F_TVLGAQTZ',
    msb=17,
    lsb=17,
    register=NV_R_ORJFNXKK
)

NV_R_ORJFNXKK_F_LIUBHIXK = FieldMetadata(
    name='NV_R_ORJFNXKK_F_LIUBHIXK',
    msb=2,
    lsb=2,
    register=NV_R_ORJFNXKK
)

NV_R_ORJFNXKK_F_QESCWVQY = FieldMetadata(
    name='NV_R_ORJFNXKK_F_QESCWVQY',
    msb=18,
    lsb=18,
    register=NV_R_ORJFNXKK
)

NV_R_ORJFNXKK_F_MPOJZJWM = FieldMetadata(
    name='NV_R_ORJFNXKK_F_MPOJZJWM',
    msb=3,
    lsb=3,
    register=NV_R_ORJFNXKK
)

NV_R_ORJFNXKK_F_BAHRFDET = FieldMetadata(
    name='NV_R_ORJFNXKK_F_BAHRFDET',
    msb=19,
    lsb=19,
    register=NV_R_ORJFNXKK
)

NV_R_ORJFNXKK_F_FLWYLUNH = FieldMetadata(
    name='NV_R_ORJFNXKK_F_FLWYLUNH',
    msb=4,
    lsb=4,
    register=NV_R_ORJFNXKK
)

NV_R_ORJFNXKK_F_JCZNWMND = FieldMetadata(
    name='NV_R_ORJFNXKK_F_JCZNWMND',
    msb=20,
    lsb=20,
    register=NV_R_ORJFNXKK
)

NV_R_ORJFNXKK_F_XGWKBGKW = FieldMetadata(
    name='NV_R_ORJFNXKK_F_XGWKBGKW',
    msb=5,
    lsb=5,
    register=NV_R_ORJFNXKK
)

NV_R_ORJFNXKK_F_WTKQRSYR = FieldMetadata(
    name='NV_R_ORJFNXKK_F_WTKQRSYR',
    msb=21,
    lsb=21,
    register=NV_R_ORJFNXKK
)

NV_R_ORJFNXKK_F_PXWKJBYG = FieldMetadata(
    name='NV_R_ORJFNXKK_F_PXWKJBYG',
    msb=6,
    lsb=6,
    register=NV_R_ORJFNXKK
)

NV_R_ORJFNXKK_F_JSTLQMMV = FieldMetadata(
    name='NV_R_ORJFNXKK_F_JSTLQMMV',
    msb=22,
    lsb=22,
    register=NV_R_ORJFNXKK
)

NV_R_ORJFNXKK_F_GUSLARCD = FieldMetadata(
    name='NV_R_ORJFNXKK_F_GUSLARCD',
    msb=7,
    lsb=7,
    register=NV_R_ORJFNXKK
)

NV_R_ORJFNXKK_F_DFIFSJHJ = FieldMetadata(
    name='NV_R_ORJFNXKK_F_DFIFSJHJ',
    msb=23,
    lsb=23,
    register=NV_R_ORJFNXKK
)

NV_R_TMFXURCX = RegisterMetadata(
    name='NV_R_TMFXURCX',
    address=0x834,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_TMFXURCX_F_JQQOSHHN = FieldMetadata(
    name='NV_R_TMFXURCX_F_JQQOSHHN',
    msb=0,
    lsb=0,
    register=NV_R_TMFXURCX
)

NV_R_TMFXURCX_F_BWRFYPXT = FieldMetadata(
    name='NV_R_TMFXURCX_F_BWRFYPXT',
    msb=16,
    lsb=16,
    register=NV_R_TMFXURCX
)

NV_R_TMFXURCX_F_PNMVPTRO = FieldMetadata(
    name='NV_R_TMFXURCX_F_PNMVPTRO',
    msb=1,
    lsb=1,
    register=NV_R_TMFXURCX
)

NV_R_TMFXURCX_F_MOSIXIQI = FieldMetadata(
    name='NV_R_TMFXURCX_F_MOSIXIQI',
    msb=17,
    lsb=17,
    register=NV_R_TMFXURCX
)

NV_R_TMFXURCX_F_CZRWNPRU = FieldMetadata(
    name='NV_R_TMFXURCX_F_CZRWNPRU',
    msb=2,
    lsb=2,
    register=NV_R_TMFXURCX
)

NV_R_TMFXURCX_F_HSTKOYXZ = FieldMetadata(
    name='NV_R_TMFXURCX_F_HSTKOYXZ',
    msb=18,
    lsb=18,
    register=NV_R_TMFXURCX
)

NV_R_TMFXURCX_F_BVAZDSRF = FieldMetadata(
    name='NV_R_TMFXURCX_F_BVAZDSRF',
    msb=3,
    lsb=3,
    register=NV_R_TMFXURCX
)

NV_R_TMFXURCX_F_UJUJKZMP = FieldMetadata(
    name='NV_R_TMFXURCX_F_UJUJKZMP',
    msb=19,
    lsb=19,
    register=NV_R_TMFXURCX
)

NV_R_TMFXURCX_F_YCXMAITD = FieldMetadata(
    name='NV_R_TMFXURCX_F_YCXMAITD',
    msb=4,
    lsb=4,
    register=NV_R_TMFXURCX
)

NV_R_TMFXURCX_F_HCKCTMKZ = FieldMetadata(
    name='NV_R_TMFXURCX_F_HCKCTMKZ',
    msb=20,
    lsb=20,
    register=NV_R_TMFXURCX
)

NV_R_TMFXURCX_F_PXKZCZYZ = FieldMetadata(
    name='NV_R_TMFXURCX_F_PXKZCZYZ',
    msb=5,
    lsb=5,
    register=NV_R_TMFXURCX
)

NV_R_TMFXURCX_F_HUXEYOBI = FieldMetadata(
    name='NV_R_TMFXURCX_F_HUXEYOBI',
    msb=21,
    lsb=21,
    register=NV_R_TMFXURCX
)

NV_R_TMFXURCX_F_DHAGQGIE = FieldMetadata(
    name='NV_R_TMFXURCX_F_DHAGQGIE',
    msb=6,
    lsb=6,
    register=NV_R_TMFXURCX
)

NV_R_TMFXURCX_F_TDNJIDOA = FieldMetadata(
    name='NV_R_TMFXURCX_F_TDNJIDOA',
    msb=22,
    lsb=22,
    register=NV_R_TMFXURCX
)

NV_R_TMFXURCX_F_SDLGRUVO = FieldMetadata(
    name='NV_R_TMFXURCX_F_SDLGRUVO',
    msb=7,
    lsb=7,
    register=NV_R_TMFXURCX
)

NV_R_TMFXURCX_F_GLQUMCOS = FieldMetadata(
    name='NV_R_TMFXURCX_F_GLQUMCOS',
    msb=23,
    lsb=23,
    register=NV_R_TMFXURCX
)

NV_R_HEOFRDQP = RegisterMetadata(
    name='NV_R_HEOFRDQP',
    address=0x838,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_HEOFRDQP_F_KOMXCILD = FieldMetadata(
    name='NV_R_HEOFRDQP_F_KOMXCILD',
    msb=16,
    lsb=16,
    register=NV_R_HEOFRDQP
)

NV_R_HEOFRDQP_F_SICZXNMB = FieldMetadata(
    name='NV_R_HEOFRDQP_F_SICZXNMB',
    msb=0,
    lsb=0,
    register=NV_R_HEOFRDQP
)

NV_R_HEOFRDQP_F_ZVDUDLOB = FieldMetadata(
    name='NV_R_HEOFRDQP_F_ZVDUDLOB',
    msb=17,
    lsb=17,
    register=NV_R_HEOFRDQP
)

NV_R_HEOFRDQP_F_WTFFCJMP = FieldMetadata(
    name='NV_R_HEOFRDQP_F_WTFFCJMP',
    msb=1,
    lsb=1,
    register=NV_R_HEOFRDQP
)

NV_R_HEOFRDQP_F_STVAKCIY = FieldMetadata(
    name='NV_R_HEOFRDQP_F_STVAKCIY',
    msb=18,
    lsb=18,
    register=NV_R_HEOFRDQP
)

NV_R_HEOFRDQP_F_XTBNWBHI = FieldMetadata(
    name='NV_R_HEOFRDQP_F_XTBNWBHI',
    msb=2,
    lsb=2,
    register=NV_R_HEOFRDQP
)

NV_R_HEOFRDQP_F_XWGFFAMR = FieldMetadata(
    name='NV_R_HEOFRDQP_F_XWGFFAMR',
    msb=19,
    lsb=19,
    register=NV_R_HEOFRDQP
)

NV_R_HEOFRDQP_F_WGOBSKUW = FieldMetadata(
    name='NV_R_HEOFRDQP_F_WGOBSKUW',
    msb=3,
    lsb=3,
    register=NV_R_HEOFRDQP
)

NV_R_HEOFRDQP_F_LUBIHGTB = FieldMetadata(
    name='NV_R_HEOFRDQP_F_LUBIHGTB',
    msb=20,
    lsb=20,
    register=NV_R_HEOFRDQP
)

NV_R_HEOFRDQP_F_WOJMNMHM = FieldMetadata(
    name='NV_R_HEOFRDQP_F_WOJMNMHM',
    msb=4,
    lsb=4,
    register=NV_R_HEOFRDQP
)

NV_R_HEOFRDQP_F_SNCVJRWG = FieldMetadata(
    name='NV_R_HEOFRDQP_F_SNCVJRWG',
    msb=21,
    lsb=21,
    register=NV_R_HEOFRDQP
)

NV_R_HEOFRDQP_F_HSUAVBKV = FieldMetadata(
    name='NV_R_HEOFRDQP_F_HSUAVBKV',
    msb=5,
    lsb=5,
    register=NV_R_HEOFRDQP
)

NV_R_HEOFRDQP_F_XJBOMWMI = FieldMetadata(
    name='NV_R_HEOFRDQP_F_XJBOMWMI',
    msb=22,
    lsb=22,
    register=NV_R_HEOFRDQP
)

NV_R_HEOFRDQP_F_OUYHIHRM = FieldMetadata(
    name='NV_R_HEOFRDQP_F_OUYHIHRM',
    msb=6,
    lsb=6,
    register=NV_R_HEOFRDQP
)

NV_R_HEOFRDQP_F_UBOLPNMU = FieldMetadata(
    name='NV_R_HEOFRDQP_F_UBOLPNMU',
    msb=23,
    lsb=23,
    register=NV_R_HEOFRDQP
)

NV_R_HEOFRDQP_F_DZPQHVKZ = FieldMetadata(
    name='NV_R_HEOFRDQP_F_DZPQHVKZ',
    msb=7,
    lsb=7,
    register=NV_R_HEOFRDQP
)

NV_R_XVCZKKIN = RegisterMetadata(
    name='NV_R_XVCZKKIN',
    address=0x83c,
    zero_based=True,
    debug_dump={'tags': ['error'], 'interesting': {'rule': 'nonzero', 'reason': 'error status nonzero'}}
)

NV_R_XVCZKKIN_F_NXEYVFLY = FieldMetadata(
    name='NV_R_XVCZKKIN_F_NXEYVFLY',
    msb=0,
    lsb=0,
    register=NV_R_XVCZKKIN
)

NV_R_XVCZKKIN_F_MXNEXENG = FieldMetadata(
    name='NV_R_XVCZKKIN_F_MXNEXENG',
    msb=1,
    lsb=1,
    register=NV_R_XVCZKKIN
)

NV_R_XVCZKKIN_F_CTKLZGNT = FieldMetadata(
    name='NV_R_XVCZKKIN_F_CTKLZGNT',
    msb=2,
    lsb=2,
    register=NV_R_XVCZKKIN
)

NV_R_XVCZKKIN_F_WRVGEORD = FieldMetadata(
    name='NV_R_XVCZKKIN_F_WRVGEORD',
    msb=3,
    lsb=3,
    register=NV_R_XVCZKKIN
)

NV_R_XVCZKKIN_F_XCTMQXJT = FieldMetadata(
    name='NV_R_XVCZKKIN_F_XCTMQXJT',
    msb=4,
    lsb=4,
    register=NV_R_XVCZKKIN
)

NV_R_XVCZKKIN_F_RQZEYYFI = FieldMetadata(
    name='NV_R_XVCZKKIN_F_RQZEYYFI',
    msb=5,
    lsb=5,
    register=NV_R_XVCZKKIN
)

NV_R_XVCZKKIN_F_UODNCNNR = FieldMetadata(
    name='NV_R_XVCZKKIN_F_UODNCNNR',
    msb=6,
    lsb=6,
    register=NV_R_XVCZKKIN
)

NV_R_XVCZKKIN_F_DSLRASHZ = FieldMetadata(
    name='NV_R_XVCZKKIN_F_DSLRASHZ',
    msb=7,
    lsb=7,
    register=NV_R_XVCZKKIN
)

