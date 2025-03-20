#
# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
NV_PTOP_DEVICE_INFO_CFG = RegisterMetadata(
    name='NV_PTOP_DEVICE_INFO_CFG',
    address=0x224fc
)

NV_PTOP_DEVICE_INFO_CFG_MAX_DEVICES = FieldMetadata(
    name='NV_PTOP_DEVICE_INFO_CFG_MAX_DEVICES',
    msb=15,
    lsb=4,
    register=NV_PTOP_DEVICE_INFO_CFG
)

NV_PTOP_DEVICE_INFO_CFG_MAX_DEVICES_INIT = ValueMetadata(
    name='NV_PTOP_DEVICE_INFO_CFG_MAX_DEVICES_INIT',
    value=153,
    field=NV_PTOP_DEVICE_INFO_CFG_MAX_DEVICES
)

NV_PTOP_DEVICE_INFO_CFG_MAX_ROWS_PER_DEVICE = FieldMetadata(
    name='NV_PTOP_DEVICE_INFO_CFG_MAX_ROWS_PER_DEVICE',
    msb=19,
    lsb=16,
    register=NV_PTOP_DEVICE_INFO_CFG
)

NV_PTOP_DEVICE_INFO_CFG_MAX_ROWS_PER_DEVICE_INIT = ValueMetadata(
    name='NV_PTOP_DEVICE_INFO_CFG_MAX_ROWS_PER_DEVICE_INIT',
    value=3,
    field=NV_PTOP_DEVICE_INFO_CFG_MAX_ROWS_PER_DEVICE
)

NV_PTOP_DEVICE_INFO_CFG_NUM_ROWS = FieldMetadata(
    name='NV_PTOP_DEVICE_INFO_CFG_NUM_ROWS',
    msb=31,
    lsb=20,
    register=NV_PTOP_DEVICE_INFO_CFG
)

NV_PTOP_DEVICE_INFO_CFG_NUM_ROWS_INIT = ValueMetadata(
    name='NV_PTOP_DEVICE_INFO_CFG_NUM_ROWS_INIT',
    value=353,
    field=NV_PTOP_DEVICE_INFO_CFG_NUM_ROWS
)

NV_PTOP_DEVICE_INFO_CFG_VERSION = FieldMetadata(
    name='NV_PTOP_DEVICE_INFO_CFG_VERSION',
    msb=3,
    lsb=0,
    register=NV_PTOP_DEVICE_INFO_CFG
)

NV_PTOP_DEVICE_INFO_CFG_VERSION_INIT = ValueMetadata(
    name='NV_PTOP_DEVICE_INFO_CFG_VERSION_INIT',
    value=2,
    field=NV_PTOP_DEVICE_INFO_CFG_VERSION
)

# Array definitions
NV_PTOP_DEVICE_INFO2 = ArrayMetadata(
    name='NV_PTOP_DEVICE_INFO2',
    base_address=0x22800,
    stride=4,
    size=353
)

NV_PTOP_DEVICE_INFO2_DEV_DEVICE_PRI_BASE = FieldMetadata(
    name='NV_PTOP_DEVICE_INFO2_DEV_DEVICE_PRI_BASE',
    msb=57,
    lsb=40,
    register=NV_PTOP_DEVICE_INFO2
)

NV_PTOP_DEVICE_INFO2_DEV_FAULT_ID = FieldMetadata(
    name='NV_PTOP_DEVICE_INFO2_DEV_FAULT_ID',
    msb=10,
    lsb=0,
    register=NV_PTOP_DEVICE_INFO2
)

NV_PTOP_DEVICE_INFO2_DEV_FAULT_ID_INVALID = ValueMetadata(
    name='NV_PTOP_DEVICE_INFO2_DEV_FAULT_ID_INVALID',
    value=0,
    field=NV_PTOP_DEVICE_INFO2_DEV_FAULT_ID
)

NV_PTOP_DEVICE_INFO2_DEV_GROUP_ID = FieldMetadata(
    name='NV_PTOP_DEVICE_INFO2_DEV_GROUP_ID',
    msb=15,
    lsb=11,
    register=NV_PTOP_DEVICE_INFO2
)

NV_PTOP_DEVICE_INFO2_DEV_INSTANCE_ID = FieldMetadata(
    name='NV_PTOP_DEVICE_INFO2_DEV_INSTANCE_ID',
    msb=23,
    lsb=16,
    register=NV_PTOP_DEVICE_INFO2
)

NV_PTOP_DEVICE_INFO2_DEV_IS_ENGINE = FieldMetadata(
    name='NV_PTOP_DEVICE_INFO2_DEV_IS_ENGINE',
    msb=62,
    lsb=62,
    register=NV_PTOP_DEVICE_INFO2
)

NV_PTOP_DEVICE_INFO2_DEV_IS_ENGINE_FALSE = ValueMetadata(
    name='NV_PTOP_DEVICE_INFO2_DEV_IS_ENGINE_FALSE',
    value=0,
    field=NV_PTOP_DEVICE_INFO2_DEV_IS_ENGINE
)
NV_PTOP_DEVICE_INFO2_DEV_IS_ENGINE_TRUE = ValueMetadata(
    name='NV_PTOP_DEVICE_INFO2_DEV_IS_ENGINE_TRUE',
    value=1,
    field=NV_PTOP_DEVICE_INFO2_DEV_IS_ENGINE
)

NV_PTOP_DEVICE_INFO2_DEV_RESET_ID = FieldMetadata(
    name='NV_PTOP_DEVICE_INFO2_DEV_RESET_ID',
    msb=39,
    lsb=32,
    register=NV_PTOP_DEVICE_INFO2
)

NV_PTOP_DEVICE_INFO2_DEV_RESET_ID_INVALID = ValueMetadata(
    name='NV_PTOP_DEVICE_INFO2_DEV_RESET_ID_INVALID',
    value=0,
    field=NV_PTOP_DEVICE_INFO2_DEV_RESET_ID
)

NV_PTOP_DEVICE_INFO2_DEV_RLENG_ID = FieldMetadata(
    name='NV_PTOP_DEVICE_INFO2_DEV_RLENG_ID',
    msb=65,
    lsb=64,
    register=NV_PTOP_DEVICE_INFO2
)

NV_PTOP_DEVICE_INFO2_DEV_RUNLIST_PRI_BASE = FieldMetadata(
    name='NV_PTOP_DEVICE_INFO2_DEV_RUNLIST_PRI_BASE',
    msb=89,
    lsb=74,
    register=NV_PTOP_DEVICE_INFO2
)

NV_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM = FieldMetadata(
    name='NV_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM',
    msb=30,
    lsb=24,
    register=NV_PTOP_DEVICE_INFO2
)

NV_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_HSHUB = ValueMetadata(
    name='NV_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_HSHUB',
    value=24,
    field=NV_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM
)
NV_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE = ValueMetadata(
    name='NV_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM_LCE',
    value=19,
    field=NV_PTOP_DEVICE_INFO2_DEV_TYPE_ENUM
)

NV_PTOP_DEVICE_INFO2_ROW_CHAIN = FieldMetadata(
    name='NV_PTOP_DEVICE_INFO2_ROW_CHAIN',
    msb=31,
    lsb=31,
    register=NV_PTOP_DEVICE_INFO2
)

NV_PTOP_DEVICE_INFO2_ROW_CHAIN_LAST = ValueMetadata(
    name='NV_PTOP_DEVICE_INFO2_ROW_CHAIN_LAST',
    value=0,
    field=NV_PTOP_DEVICE_INFO2_ROW_CHAIN
)
NV_PTOP_DEVICE_INFO2_ROW_CHAIN_MORE = ValueMetadata(
    name='NV_PTOP_DEVICE_INFO2_ROW_CHAIN_MORE',
    value=1,
    field=NV_PTOP_DEVICE_INFO2_ROW_CHAIN
)

NV_PTOP_DEVICE_INFO2_ROW_VALUE = FieldMetadata(
    name='NV_PTOP_DEVICE_INFO2_ROW_VALUE',
    msb=31,
    lsb=0,
    register=NV_PTOP_DEVICE_INFO2
)

NV_PTOP_DEVICE_INFO2_ROW_VALUE_INVALID = ValueMetadata(
    name='NV_PTOP_DEVICE_INFO2_ROW_VALUE_INVALID',
    value=0,
    field=NV_PTOP_DEVICE_INFO2_ROW_VALUE
)

