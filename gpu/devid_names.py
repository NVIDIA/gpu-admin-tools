#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


GPU_NAME_BY_DEVID = {
    0x102d: 'K80',
    0x118a: 'K520',
    0x13f2: 'M60',
    0x15f7: 'P100',
    0x15f8: 'P100',
    0x15f9: 'P100',
    0x15fc: 'P100',
    0x1613: 'M60',
    0x1b38: 'P40',
    0x1bb3: 'P4',
    0x1db1: 'V100',
    0x1db2: 'V100',
    0x1db3: 'V100',
    0x1db4: 'V100',
    0x1db5: 'V100',
    0x1db6: 'V100',
    0x1db7: 'V100',
    0x1db8: 'V100',
    0x1df4: 'V100',
    0x1df5: 'V100',
    0x1df6: 'V100S',
    0x1eb4: 'T4G',
    0x1eb8: 'T4',
    0x1eb9: 'T4',
    0x20b0: 'A100',
    0x20b1: 'A100',
    0x20b3: 'A100',
    0x20b5: 'A100',
    0x20b7: 'A30',
    0x20f0: 'A100',
    0x20f1: 'A100',
    0x20f2: 'A100',
    0x2236: 'A10',
    0x2309: 'H20',
    0x230c: 'H20',
    0x230e: 'H20',
    0x2313: 'H100',
    0x2321: 'H100',
    0x2322: 'H800',
    0x2324: 'H800',
    0x2328: 'H20',
    0x2329: 'H20',
    0x232c: 'H20',
    0x2330: 'H100',
    0x2331: 'H100',
    0x2335: 'H200',
    0x2336: 'H100',
    0x2337: 'H100',
    0x2338: 'H100',
    0x2339: 'H100',
    0x233a: 'H800',
    0x233b: 'H200',
    0x233d: 'H100',
    0x2342: 'GH200',
    0x2343: 'GH200',
    0x2345: 'GH200',
    0x2348: 'GH200',
    0x26b7: 'L20',
    0x27b6: 'L2',
    0x27b8: 'L4',
    0x2901: 'B200',
    0x2909: 'B200',
    0x290a: 'B200',
    0x2941: 'GB200',
    0x2bb1: 'RTX-PRO-6000',
    0x2bb4: 'RTX-PRO-6000',
    0x2c31: 'RTX-PRO-4500',
    0x2c3a: 'RTX-PRO-4500',
    0x3182: 'B300',
    0x31a1: 'GB300',
    0x31c2: 'GB300',
    0x31c3: 'GB300',
}

# Products sharing a device ID require a subsystem device ID.
# None suppresses a generic default for a known different product.
GPU_NAME_BY_DEVID_SSID = {
    (0x1b38, 0x180e): None,
    (0x1db8, 0x13db): None,
    (0x20b0, 0x1469): None,
    (0x20b0, 0x1583): None,
    (0x20b2, 0x1463): 'A100',
    (0x20b2, 0x147f): 'A100',
    (0x20b2, 0x1484): 'A100',
    (0x20b2, 0x1622): 'A100',
    (0x20b2, 0x1623): 'A100',
    (0x2237, 0x152f): 'A10G',
    (0x25b6, 0x14a9): 'A16',
    (0x25b6, 0x157e): 'A2',
    (0x26b5, 0x169d): 'L40',
    (0x26b5, 0x17da): 'L40',
    (0x26b8, 0x169e): 'L40G',
    (0x26b9, 0x1851): 'L40S',
    (0x26b9, 0x18cf): 'L40S',
    (0x26ba, 0x1957): 'L20',
    (0x26ba, 0x1990): 'L20',
    (0x2bb5, 0x204e): 'RTX-PRO-6000',
    (0x2bb5, 0x220b): 'RTX-PRO-6000',
}
