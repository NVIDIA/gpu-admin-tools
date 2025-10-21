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

# Constants
IANA_NVIDIA = 0x1647

# Enum classes
class VendorDefinedIanaCommand:
    DownloadLog = 0x06

class MctpMessageType:
    VendorDefinedIana = 0x7F

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

class MctpVdmIanaReqHeader(NiceStruct):
    _fields_ = [
            ("messageType", "B", 7),
            ("ic", "B", 1),
            ("iana", "I"),  # 32-bit integer
            ("instanceId", "B", 5),
            ("rsvd", "B", 1),
            ("d", "B", 1),
            ("rq", "B", 1),
            ("vendorMessageType", "B"),
            ("commandCode", "B"),
            ("messageVersion", "B"),
    ]

    def __init__(self):
        super().__init__()

        # Initialize with default values
        self.messageType = MctpMessageType.VendorDefinedIana & 0x7F
        self.ic = 0
        self.iana = IANA_NVIDIA
        self.instanceId = 0
        self.rsvd = 0
        self.d = 0
        self.rq = 1
        self.vendorMessageType = 1
        self.commandCode = 0  # Will be set per request
        self.messageVersion = 0  # Will be set per request

    def set_commandCode(self, commandCode):
        self.commandCode = commandCode & 0xFF

    def set_messageVersion(self, messageVersion):
        self.messageVersion = messageVersion & 0xFF

class MctpVdmIanaRspHdr(NiceStruct):
    _fields_ = [
            ("messageType", "B", 7),
            ("ic", "B", 1),
            ("iana", "I"),
            ("instanceId", "B", 5),
            ("rsvd", "B", 1),
            ("d", "B", 1),
            ("rq", "B", 1),
            ("vendorMessageType", "B"),
            ("commandCode", "B"),
            ("messageVersion", "B"),
            ("completionCode", "B"),
    ]

    def get_messageType(self):
        return self.messageType

    def get_commandCode(self):
        return self.commandCode

    def get_completionCode(self):
        return self.completionCode

    def __init__(self):
        super().__init__()

class MctpVdmIanaDownloadLog(NiceStruct):
    _fields_ = [
            ("sessionId", "B"),  # Single byte matching NvU8 sessionId
    ]

    def __init__(self):
        super().__init__()
        self.sessionId = 0xFF

    def set_sessionId(self, sessionId):
        self.sessionId = sessionId & 0xFF

class MctpVdmIanaDownloadLogResponse(NiceStruct):
    _fields_ = [
            ("sessionId", "B"),  # NvU8 sessionId
            ("length", "B"),     # NvU8 length
            ("data", "52s"),     # NvU8 data[52] - 52 bytes
    ]

    def __init__(self):
        super().__init__()
        self.sessionId = 0
        self.length = 0
        self.data = b'\x00' * 52  # Initialize 52 bytes of zeros

