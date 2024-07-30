#
# SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from utils import NiceStruct

class MctpHeader(NiceStruct):
    _fields_ = [
            ("version", "I", 4),
            ("rsvd0", "I", 4),
            ("deid", "I", 8),
            ("seid", "I", 8),
            ("tag", "I", 3),
            ("to", "I", 1),
            ("seq", "I", 2),
            ("eom", "I", 1),
            ("som", "I", 1),
        ]

    def __init__(self):
        super().__init__()

        self.som = 1
        self.eom = 1

class MctpMessageHeader(NiceStruct):
    _fields_ = [
            ("type", "I", 7),
            ("ic", "I", 1),
            ("vendor_id", "I", 16),
            ("nvdm_type", "I", 8),
    ]

    def __init__(self):
        super().__init__()

        self.type = 0x7e
        self.vendor_id = 0x10de
