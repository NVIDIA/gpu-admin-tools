#
# SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

GPU_BAR0_SIZE = 16 * 1024 * 1024
NVSWITCH_BAR0_SIZE = 32 * 1024 * 1024

NV_PMC_ENABLE = 0x200
NV_PMC_ENABLE_PRIV_RING = (0x1 << 5)
NV_PMC_ENABLE_HOST = (0x1 << 8)
NV_PMC_ENABLE_PWR = (0x1 << 13)
NV_PMC_ENABLE_PGRAPH = (0x1 << 12)
NV_PMC_ENABLE_MSVLD = (0x1 << 15)
NV_PMC_ENABLE_MSPDEC = (0x1 << 17)
NV_PMC_ENABLE_MSENC = (0x1 << 18)
NV_PMC_ENABLE_MSPPP = (0x1 << 1)
NV_PMC_ENABLE_NVLINK = (0x1 << 25)
NV_PMC_ENABLE_SEC = (0x1 << 14)
NV_PMC_ENABLE_PDISP = (0x1 << 30)
NV_PMC_ENABLE_PERFMON = (0x1 << 28)

NV_PMC_DEVICE_ENABLE = 0x600

def NV_PMC_ENABLE_NVDEC(nvdec):
    if nvdec == 0:
        return (0x1 << 15)
    elif nvdec == 1:
        return (0x1 << 16)
    elif nvdec == 2:
        return (0x1 << 20)
    else:
        assert 0, "Unhandled nvdec %d" % nvdec

def NV_PMC_ENABLE_NVENC(nvenc):
    if nvenc == 0:
        return (0x1 << 18)
    elif nvenc == 1:
        return (0x1 << 19)
    elif nvenc == 2:
        return (0x1 << 4)
    else:
        assert 0, "Unhandled nvenc %d" % nvenc

NV_PMC_BOOT_0 = 0x0
NV_PROM_DATA = 0x300000
NV_PROM_DATA_SIZE_K520 = 262144
NV_PROM_DATA_SIZE_KEPLER = 524288
NV_PROM_DATA_SIZE_VOLTA = 1048576

NV_BAR0_WINDOW_CFG = 0x1700
NV_BAR0_WINDOW_CFG_TARGET_SYSMEM_COHERENT = 0x2000000
NV_BAR0_WINDOW = 0x700000

NV_PPWR_FALCON_DMACTL = 0x10a10c

def NV_PPWR_FALCON_IMEMD(i):
    return 0x10a184 + i * 16

def NV_PPWR_FALCON_IMEMC(i):
    return 0x10a180 + i * 16

NV_PPWR_FALCON_IMEMC_AINCW_TRUE = 1 << 24
NV_PPWR_FALCON_IMEMC_AINCR_TRUE = 1 << 25
NV_PPWR_FALCON_IMEMC_SECURE_ENABLED = 1 << 28

def NV_PPWR_FALCON_IMEMT(i):
    return 0x10a188 + i * 16

def NV_PPWR_FALCON_DMEMC(i):
    return 0x0010a1c0 + i * 8

NV_PPWR_FALCON_BOOTVEC = 0x10a104
NV_PPWR_FALCON_CPUCTL = 0x10a100
NV_PPWR_FALCON_HWCFG = 0x10a108
NV_PPWR_FALCON_HWCFG1 = 0x10a12c
NV_PPWR_FALCON_ENGINE_RESET = 0x10a3c0

NV_PMSVLD_FALCON_CPUCTL = 0x84100
NV_PMSPDEC_FALCON_CPUCTL = 0x85100
NV_PMSPPP_FALCON_CPUCTL = 0x86100
NV_PMSENC_FALCON_CPUCTL = 0x1c2100
NV_PHDAFALCON_FALCON_CPUCTL = 0x1c3100
NV_PMINION_FALCON_CPUCTL = 0xa06100
NV_PDISP_FALCON_CPUCTL = 0x627100

# NVDECs
def NV_PNVDEC_FALCON_CPUCTL_AMPERE(nvdec):
    return 0x848100 + nvdec * 0x4000

def NV_PNVDEC_FALCON_CPUCTL_TURING(nvdec):
    return 0x830100 + nvdec * 0x4000

def NV_PNVDEC_FALCON_CPUCTL_MAXWELL(nvdec):
    return 0x84100 + nvdec * 0x4000

# NVENCs
def NV_PNVENC_FALCON_CPUCTL(nvenc):
    return 0x1c8100 + nvenc * 0x4000

# GSP
NV_PGSP_FALCON_CPUCTL = 0x110100

# SEC
NV_PSEC_FALCON_CPUCTL_MAXWELL = 0x87100
NV_PSEC_FALCON_CPUCTL_TURING = 0x840100

# FB FALCON
NV_PFBFALCON_FALCON_CPUCTL = 0x9a4100

