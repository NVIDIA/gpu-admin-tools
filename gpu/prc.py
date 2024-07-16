#
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from enum import Enum

class PrcKnob(Enum):
    PRC_KNOB_ID_1                                       = 1

    PRC_KNOB_ID_2                                       = 2

    PRC_KNOB_ID_3                                       = 3

    PRC_KNOB_ID_4                                       = 4

    PRC_KNOB_ID_CCD_ALLOW_INB                           = 5
    PRC_KNOB_ID_CCD                                     = 6
    PRC_KNOB_ID_CCM_ALLOW_INB                           = 7
    PRC_KNOB_ID_CCM                                     = 8
    PRC_KNOB_ID_BAR0_DECOUPLER_ALLOW_INB                = 9
    PRC_KNOB_ID_BAR0_DECOUPLER                          = 10

    PRC_KNOB_ID_11                                      = 11

    PRC_KNOB_ID_12                                      = 12

    PRC_KNOB_ID_13                                      = 13

    PRC_KNOB_ID_14                                      = 14

    PRC_KNOB_ID_15                                      = 15

    PRC_KNOB_ID_16                                      = 16

    PRC_KNOB_ID_17                                      = 17

    PRC_KNOB_ID_18                                      = 18

    PRC_KNOB_ID_19                                      = 19

    PRC_KNOB_ID_20                                      = 20

    PRC_KNOB_ID_21                                      = 21

    PRC_KNOB_ID_22                                      = 22

    PRC_KNOB_ID_23                                      = 23

    PRC_KNOB_ID_24                                      = 24

    PRC_KNOB_ID_25                                      = 25

    PRC_KNOB_ID_26                                      = 26

    PRC_KNOB_ID_27                                      = 27

    PRC_KNOB_ID_28                                      = 28

    PRC_KNOB_ID_29                                      = 29

    PRC_KNOB_ID_30                                      = 30

    PRC_KNOB_ID_31                                      = 31

    PRC_KNOB_ID_32                                      = 32

    PRC_KNOB_ID_33                                      = 33

    PRC_KNOB_ID_34                                      = 34

    PRC_KNOB_ID_35                                      = 35

    PRC_KNOB_ID_36                                      = 36

    PRC_KNOB_ID_37                                      = 37

    PRC_KNOB_ID_38                                      = 38

    PRC_KNOB_ID_39                                      = 39

    PRC_KNOB_ID_40                                      = 40

    PRC_KNOB_ID_41                                      = 41

    PRC_KNOB_ID_42                                      = 42

    PRC_KNOB_ID_43                                      = 43

    PRC_KNOB_ID_PPCIE_ALLOW_INB                         = 44
    PRC_KNOB_ID_PPCIE                                   = 45

    PRC_KNOB_ID_46                                      = 46

    @classmethod
    def str_from_knob_id(cls, knob_id):
        try:
            prc_knob = PrcKnob(knob_id)
            knob_name = f"{prc_knob.name} "
        except ValueError:
            knob_name = ""

        knob_name += f"{knob_id} ({knob_id:#x})"
        return knob_name
