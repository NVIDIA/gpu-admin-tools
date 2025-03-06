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

GPU_PROPS_BY_DEVID = {
  (0x2321,0x1839): ['is_pcie'],
  (0x2322,0x17A4): ['is_pcie'],
  (0x2331,0x1626): ['is_pcie'],
  (0x233A,0x183A): ['is_pcie'],
  (0x233B,0x1996): ['is_pcie'],
  (0x233D,0x1626): ['is_pcie'],
  (0x2324,0x17A6): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2324,0x17A8): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2328,0x1905): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2328,0x1906): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2329,0x198B): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2329,0x198C): ['is_sxm', 'has_module_id_bit_flip'],
  (0x232C,0x2063): ['is_sxm', 'has_module_id_bit_flip'],
  (0x232C,0x2064): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2330,0x16C0): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2330,0x16C1): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2330,0x2044): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2330,0x20C1): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2335,0x18BE): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2335,0x18BF): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2336,0x16C2): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2336,0x16C7): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2337,0x16E5): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2337,0x16EF): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2338,0x16F6): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2338,0x16F7): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2339,0x17D9): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2339,0x17FC): ['is_sxm', 'has_module_id_bit_flip'],
  (0x2342,0x16EB): ['has_c2c', 'is_sxm'],
  (0x2342,0x16EC): ['has_c2c', 'is_sxm'],
  (0x2342,0x16ED): ['has_c2c', 'is_sxm'],
  (0x2342,0x1805): ['has_c2c', 'is_sxm'],
  (0x2342,0x1809): ['has_c2c', 'is_sxm'],
  (0x2342,0x1935): ['has_c2c', 'is_sxm'],
  (0x2342,0x1937): ['has_c2c', 'is_sxm'],
  (0x2343,0x16EC): ['has_c2c', 'is_sxm'],
  (0x2345,0x16ED): ['has_c2c', 'is_sxm'],
  (0x2348,0x18D2): ['has_c2c', 'is_sxm'],
  (0x2924,0x18B6): ['is_pcie'],
  (0x2924,0x20D4): ['is_pcie'],
  (0x2925,0x18B7): ['is_pcie'],
  (0x293D,0x18B6): ['is_pcie'],
  (0x29BC,0x1997): ['is_pcie'],
  (0x29BC,0x1998): ['is_pcie'],
  (0x2901,0x1999): ['is_sxm'],
  (0x2901,0x199B): ['is_sxm'],
  (0x2901,0x199D): ['is_sxm'],
  (0x2901,0x20DA): ['is_sxm'],
  (0x2920,0x197F): ['is_sxm'],
  (0x2920,0x20DE): ['is_sxm'],
  (0x293D,0x197F): ['is_sxm'],
  (0x293D,0x1999): ['is_sxm'],
  (0x29BC,0x1985): ['is_sxm'],
  (0x29F1,0x20DC): ['is_sxm'],
  (0x2941,0x0000): ['has_c2c', 'is_sxm'],
  (0x2941,0x2045): ['has_c2c', 'is_sxm'],
  (0x2941,0x2046): ['has_c2c', 'is_sxm'],
  (0x2941,0x20CA): ['has_c2c', 'is_sxm'],
  (0x297E,0x2046): ['has_c2c', 'is_sxm'],
  (0x29BC,0x2045): ['has_c2c', 'is_sxm'],
}
