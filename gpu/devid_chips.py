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


GPU_DEVID_CHIPS = [
    (0x1180, 0x11bf, 'kepler', 'gk104'),

    (0x11c0, 0x11ff, 'kepler', 'gk106'),

    (0x0fc0, 0x0fff, 'kepler', 'gk107'),

    (0x1000, 0x103f, 'kepler', 'gk110'),

    (0x1280, 0x12bf, 'kepler', 'gk208'),

    (0x1380, 0x13bf, 'maxwell', 'gm107'),
    (0x1780, 0x17bf, 'maxwell', 'gm107'),

    (0x1340, 0x137f, 'maxwell', 'gm108'),
    (0x1740, 0x177f, 'maxwell', 'gm108'),

    (0x17c0, 0x183f, 'maxwell', 'gm200'),

    (0x13c0, 0x13ff, 'maxwell', 'gm204'),
    (0x1600, 0x163f, 'maxwell', 'gm204'),

    (0x1400, 0x143f, 'maxwell', 'gm206'),
    (0x1640, 0x167f, 'maxwell', 'gm206'),

    (0x15c0, 0x15ff, 'pascal', 'gp100'),
    (0x1700, 0x173f, 'pascal', 'gp100'),

    (0x1b00, 0x1b7f, 'pascal', 'gp102'),

    (0x1b80, 0x1bff, 'pascal', 'gp104'),

    (0x1c00, 0x1c7f, 'pascal', 'gp106'),

    (0x1c80, 0x1cff, 'pascal', 'gp107'),

    (0x1d00, 0x1d7f, 'pascal', 'gp108'),

    (0x1d80, 0x1dff, 'volta', 'gv100'),

    (0x1e00, 0x1e7f, 'turing', 'tu102'),

    (0x1e80, 0x1eff, 'turing', 'tu104'),

    (0x1f00, 0x1f7f, 'turing', 'tu106'),

    (0x2180, 0x21ff, 'turing', 'tu116'),

    (0x1f80, 0x1fff, 'turing', 'tu117'),

    (0x2080, 0x20ff, 'ampere', 'ga100'),

    (0x2200, 0x227f, 'ampere', 'ga102'),

    (0x2400, 0x247f, 'ampere', 'ga103'),

    (0x2480, 0x24ff, 'ampere', 'ga104'),

    (0x2500, 0x257f, 'ampere', 'ga106'),

    (0x2580, 0x25ff, 'ampere', 'ga107'),

    (0x2680, 0x26ff, 'ada', 'ad102'),

    (0x2700, 0x277f, 'ada', 'ad103'),

    (0x2780, 0x27ff, 'ada', 'ad104'),

    (0x2800, 0x287f, 'ada', 'ad106'),

    (0x2880, 0x28ff, 'ada', 'ad107'),

    (0x2300, 0x237f, 'hopper', 'gh100'),

    (0x2900, 0x297f, 'blackwell', 'gb100'),

    (0x2980, 0x29ff, 'blackwell', 'gb102'),

    (0x3180, 0x31ff, 'blackwell', 'gb110'),

    (0x3200, 0x327f, 'blackwell', 'gb112'),

    (0x2b80, 0x2bff, 'blackwell', 'gb202'),

    (0x2c00, 0x2c7f, 'blackwell', 'gb203'),

    (0x2f00, 0x2f7f, 'blackwell', 'gb205'),

    (0x2d00, 0x2d7f, 'blackwell', 'gb206'),

    (0x2d80, 0x2dff, 'blackwell', 'gb207'),



]
