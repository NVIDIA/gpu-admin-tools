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
NV_R_ZXCXJFYF = RegisterMetadata(
    name='NV_R_ZXCXJFYF',
    address=0x50dc
)

NV_R_ZXCXJFYF_F_JKKXLPAF = FieldMetadata(
    name='NV_R_ZXCXJFYF_F_JKKXLPAF',
    msb=3,
    lsb=0,
    register=NV_R_ZXCXJFYF
)

NV_R_ZXCXJFYF_F_JKKXLPAF_V_ZRRJKDVX = ValueMetadata(
    name='NV_R_ZXCXJFYF_F_JKKXLPAF_V_ZRRJKDVX',
    value=3,
    field=NV_R_ZXCXJFYF_F_JKKXLPAF
)

