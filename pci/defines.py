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

from utils import Bitfield

PCI_CFG_SPACE_SIZE = 256
PCI_CFG_SPACE_EXP_SIZE = 4096

PCI_CAPABILITY_LIST = 0x34
# PCI Express
PCI_CAP_ID_EXP = 0x10
# Power management
PCI_CAP_ID_PM  = 0x01

CAP_ID_MASK = 0xff

# Advanced Error Reporting
PCI_EXT_CAP_ID_ERR = 0x01

# SRIOV
PCI_EXT_CAP_ID_SRIOV = 0x10

# Uncorrectable Error Status
PCI_ERR_UNCOR_STATUS = 4
# Uncorrectable Error Mask
PCI_ERR_UNCOR_MASK = 8
# Uncorrectable Error Severity
PCI_ERR_UNCOR_SEVER = 12

class PciUncorrectableErrors(Bitfield):
    size = 4
    fields = {
    # Undefined
    "UND": 0x00000001,

    # Data Link Protocol
    "DLP": 0x00000010,

    # Surprise Down
    "SURPDN":  0x00000020,

    # Poisoned TLP
    "POISON_TLP": 0x00001000,

    # Flow Control Protocol
    "FCP": 0x00002000,

    # Completion Timeout
    "COMP_TIME": 0x00004000,

    # Completer Abort
    "COMP_ABORT": 0x00008000,

    # Unexpected Completion
    "UNX_COMP": 0x00010000,

    # Receiver Overflow
    "RX_OVER": 0x00020000,

    # Malformed TLP
    "MALF_TLP": 0x00040000,

    # ECRC Error Status
    "ECRC": 0x00080000,

    # Unsupported Request
    "UNSUP": 0x00100000,

    # ACS Violation
    "ACSV": 0x00200000,

    # internal error
    "INTN": 0x00400000,

    # MC blocked TLP
    "MCBTLP": 0x00800000,

    # Atomic egress blocked
    "ATOMEG": 0x01000000,

    # TLP prefix blocked
    "TLPPRE": 0x02000000,
    }

    def __str__(self):
        # Print only the non zero bits
        return "%s %s" % (self.name, str(self.non_zero_fields()))

PCI_ERR_COR_STATUS = 0x10
PCI_ERR_COR_MASK = 0x14
class PciCorrectableErrors(Bitfield):
    size = 4
    fields = {
        "RECEIVER":         0x00000001,
        "BAD_TLP":          0x00000040,
        "BAD_DDLP":         0x00000080,
        "REPLAY_ROLLOVER":  0x00000100,
        "REPLAY_TIMER":     0x00001000,
        "ADV_NON_FATAL":    0x00002000,
        "INTERNAL":         0x00004000,
        "LOG_OVERFLOW":     0x00008000,
    }

    def __str__(self):
        # Print only the non zero bits
        return f"{self.name} {self.non_zero_fields()}"


PCI_EXP_DEVCAP2 = 36
PCI_EXP_DEVCTL2 = 40
class PciDevCtl2(Bitfield):
    size = 2
    fields = {
        # Completion Timeout Value
        "COMP_TIMEOUT":         0x000f,

        # Completion Timeout Disable
        "COMP_TMOUT_DIS":       0x0010,

        # Alternative Routing-ID
        "ARI":                  0x0020,

        # Set Atomic requests
        "ATOMIC_REQ":           0x0040,

        # Block atomic egress
        "ATOMIC_EGRESS_BLOCK":  0x0080,

        # Allow IDO for requests
        "IDO_REQ_EN":           0x0100,

        # Allow IDO for completions
        "IDO_CMP_EN":           0x0200,

        # Enable LTR mechanism
        "LTR_EN":               0x0400,

        # Enable OBFF Message type A
        "OBFF_MSGA_EN":         0x2000,

        # Enable OBFF Message type B
        "OBFF_MSGB_EN":         0x4000,

        # OBFF using WAKE# signaling
        "OBFF_WAKE_EN":         0x6000,
    }

# Access Control Services
PCI_EXT_CAP_ID_ACS = 0x0D

# ACS control
PCI_EXT_ACS_CTL = 6
class AcsCtl(Bitfield):
    size = 2
    fields = {
    "SOURCE_VALIDATION":    0x0001,
    "TRANSLATION_BLOCKING": 0x0002,
    "P2P_REQUEST_REDIRECT": 0x0004,
    "P2P_COMPLETION_REDIRECT": 0x0008,
    "UPSTREAM_FORWARDING": 0x0010,
    "P2P_EGRESS_CONTROL": 0x0020,
    "DIRECT_TRANSLATED_P2P": 0x0040,
    }

# Downstream Port Containment
PCI_EXT_CAP_ID_DPC = 0x1D

# DPC control
PCI_EXP_DPC_CTL = 6
class DpcCtl(Bitfield):
    size = 2
    fields = {

    # Enable trigger on ERR_FATAL message
    "EN_FATAL": 0x0001,

    # Enable trigger on ERR_NONFATAL message
    "EN_NONFATAL": 0x0002,

    # DPC Interrupt Enable
    "INT_EN": 0x0008,
    }

# DPC Status
PCI_EXP_DPC_STATUS = 8
class DpcStatus(Bitfield):
    size = 2
    fields = {
    # Trigger Status
    "STATUS_TRIGGER":         0x0001,
    # Trigger Reason
    "STATUS_TRIGGER_RSN":     0x0006,
    # Interrupt Status
    "STATUS_INTERRUPT":       0x0008,
    # Root Port Busy
    "RP_BUSY":                0x0010,
    # Trig Reason Extension
    "STATUS_TRIGGER_RSN_EXT": 0x0060,
    }

PCI_COMMAND = 0x04
class PciCommand(Bitfield):
    size = 2
    fields = {
        "MEMORY": 0x0002,
        "MASTER": 0x0004,
        "PARITY": 0x0040,
        "SERR":   0x0100,
    }

PCI_EXP_FLAGS = 2
class PciExpFlags(Bitfield):
    size = 2
    fields = {
        # Capability version
        "VERS": 0x000f,

        # Device/Port type
        "TYPE": 0x00f0,
#define   PCI_EXP_TYPE_ENDPOINT	   0x0	/* Express Endpoint */
#define   PCI_EXP_TYPE_LEG_END	   0x1	/* Legacy Endpoint */
#define   PCI_EXP_TYPE_ROOT_PORT   0x4	/* Root Port */
#define   PCI_EXP_TYPE_UPSTREAM	   0x5	/* Upstream Port */
#define   PCI_EXP_TYPE_DOWNSTREAM  0x6	/* Downstream Port */
#define   PCI_EXP_TYPE_PCI_BRIDGE  0x7	/* PCIe to PCI/PCI-X Bridge */
#define   PCI_EXP_TYPE_PCIE_BRIDGE 0x8	/* PCI/PCI-X to PCIe Bridge */
#define   PCI_EXP_TYPE_RC_END	   0x9	/* Root Complex Integrated Endpoint */
#define   PCI_EXP_TYPE_RC_EC	   0xa	/* Root Complex Event Collector */

        # Slot implemented
        "SLOT": 0x0100,

        # Interrupt message number
        "IRQ": 0x3e00,
    }

PCI_EXP_RTCTL = 28
class PciRootControl(Bitfield):
    size = 2
    fields = {
        # System Error on Correctable Error
        "SECEE": 0x0001,

        # System Error on Non-Fatal Error
        "SENFEE": 0x0002,

        # System Error on Fatal Error
        "SEFEE": 0x0004,

        # PME Interrupt Enable
        "PMEIE": 0x0008,

        # CRS Software Visibility Enable
        "CRSSVE": 0x0010,
    }

PCI_EXP_DEVCAP = 4
class PciDevCap(Bitfield):
    size = 4
    fields = {
        # Max payload
        "PAYLOAD":  0x00000007,

        # Phantom functions
        "PHANTOM":  0x00000018,

        # Extended tags
        "EXT_TAG":  0x00000020,

        # L0s acceptable latency
        "L0S":      0x000001c0,

        # L1 acceptable latency
        "L1":       0x00000e00,

        # Attention Button Present
        "ATN_BUT":  0x00001000,

        # Attention indicator present
        "ATN_IND":  0x00002000,

        # Power indicator present
        "PWR_IND":  0x00004000,

        # Role-based error reporting
        "RBER":     0x00008000,

        # Slot power limit value
        "PWR_VAL":  0x03fc0000,

        # Slot Power Limit Scale
        "PWR_SCL":  0x0c000000,

        # Function level reset
        "FLR":      0x10000000,
    }

PCI_EXP_DEVCTL = 8
class PciDevCtl(Bitfield):
    size = 4
    fields = {
        # /* Correctable Error Reporting En. */
        "CERE": 0x0001,

        # /* Non-Fatal Error Reporting Enable */
        "NFERE": 0x0002,

        # /* Fatal Error Reporting Enable */
        "FERE": 0x0004,

        # /* Unsupported Request Reporting En. */
        "URRE": 0x0008,

        # /* Enable relaxed ordering */
        "RELAX_EN": 0x0010,
        # /* Max_Payload_Size */
        "PAYLOAD": 0x00e0,

        # /* Extended Tag Field Enable */
        "EXT_TAG": 0x0100,

        # /* Phantom Functions Enable */
        "PHANTOM": 0x0200,

        # /* Auxiliary Power PM Enable */
        "AUX_PME": 0x0400,

        # /* Enable No Snoop */
        "NOSNOOP_EN": 0x0800,

        # /* Max_Read_Request_Size */
        #"READRQ_128B  0x0000 /* 128 Bytes */
        #"READRQ_256B  0x1000 /* 256 Bytes */
        #"READRQ_512B  0x2000 /* 512 Bytes */
        #"READRQ_1024B 0x3000 /* 1024 Bytes */
        "READRQ": 0x7000,

        # /* Bridge Configuration Retry / FLR */
        "BCR_FLR": 0x8000,
    }

PCI_EXP_LNKCAP = 12
class PciLinkCap(Bitfield):
    size = 4
    fields = {
        # Maximum Link Width
        "MLW":   0x000003f0,

        # Surprise Down Error Reporting Capable
        "SDERC": 0x00080000,

        # Port Number
        "PN":    0xff000000,
    }

    def __str__(self):
        return "{ Link cap " + str(self.values()) + " raw " + hex(self.raw) + " }"



# Link Control
PCI_EXP_LNKCTL = 16
class PciLinkControl(Bitfield):
    size = 2
    fields = {
        # ASPM Control
        "ASPMC": 0x0003,

        # Read Completion Boundary
        "RCB": 0x0008,

        # Link Disable
        "LD": 0x0010,

        # Retrain Link
        "RL": 0x0020,

        # Common Clock Configuration
        "CCC": 0x0040,

        # Extended Synch
        "ES": 0x0080,

        # Hardware Autonomous Width Disable
        "HAWD": 0x0200,

        # Enable clkreq
        "CLKREQ_EN": 0x100,

        # Link Bandwidth Management Interrupt Enable
        "LBMIE": 0x0400,

        # Lnk Autonomous Bandwidth Interrupt Enable
        "LABIE": 0x0800,
    }

    def __str__(self):
        return "{ Link control " + str(self.values()) + " raw " + hex(self.raw) + " }"

# Link Status
PCI_EXP_LNKSTA = 18
class PciLinkStatus(Bitfield):
    size = 2
    fields = {
        # Current Link Speed
        # CLS_2_5GB 0x01 Current Link Speed 2.5GT/s
        # CLS_5_0GB 0x02 Current Link Speed 5.0GT/s
        "CLS": 0x000f,

        # Nogotiated Link Width
        "NLW": 0x03f0,

        # Link Training
        "LT": 0x0800,

        # Slot Clock Configuration
        "SLC": 0x1000,

        # Data Link Layer Link Active
        "DLLLA": 0x2000,

        # Link Bandwidth Management Status
        "LBMS": 0x4000,

        # Link Autonomous Bandwidth Status */
        "LABS": 0x8000,
    }

    def __str__(self):
        return "{ Link status " + str(self.values()) + " raw " + hex(self.raw) + " }"

# Link Status 2
PCI_EXP_LNKSTA2 = 0x32
class PciLinkStatus2(Bitfield):
    size = 2
    fields = {
        "DEEMPHASIS_LEVEL": 0x1,
        "EQ_COMPLETE": 0x2,
        "EQ_1": 0x4,
        "EQ_2": 0x8,
        "EQ_3": 0x10,
        "LINK_EQ_REQUEST": 0x20,
        "RETIMER_PRESENT": 0x40,
        "2_RETIMERS_PRESENT": 0x80,
        "CROSSLINK_RESOLUTION": 0x300,
        "FLIT_MODE_STATUS": 0x400,
        "DOWNSTREAM_COMPONENT": 0x7000,
        "DRS_MSG_RECEIVED": 0x8000,
    }

    def __str__(self):
        return f"{{ Link status2 {self.values()} raw {self.raw:#x} }}"

PCI_EXT_CAP_GEN4 = 0x26
PCI_GEN4_STATUS = 0xc
class PciGen4Status(Bitfield):
    size = 2
    fields = {
        "EQ_COMPLETE": 0x1,
        "EQ_1": 0x2,
        "EQ_2": 0x4,
        "EQ_3": 0x8,
        "EQ_REQUEST": 0x10,
    }

    def __str__(self):
        return f"{{ Gen4 status {self.values()} raw {self.raw:#x} }}"


PCI_EXT_CAP_GEN5 = 0x2a

PCI_GEN5_CAPS = 0x4
class PciGen5Caps(Bitfield):
    size = 2
    fields = {
        "EQ_BYPASS": 0x1,
        "NO_EQ_NEEDED": 0x2,
    }

    def __str__(self):
        return f"{{ Gen5 caps {self.values()} raw {self.raw:#x} }}"

PCI_GEN5_CONTROL = 0x8
class PciGen5Control(Bitfield):
    size = 2
    fields = {
        "EQ_BYPASS_DISABLE": 0x1,
        "NO_EQ_NEEDED_DISABLE": 0x2,
    }

    def __str__(self):
        return f"{{ Gen5 control {self.values()} raw {self.raw:#x} }}"

PCI_GEN5_STATUS = 0xc
class PciGen5Status(Bitfield):
    size = 2
    fields = {
        "EQ_COMPLETE": 0x1,
        "EQ_1": 0x2,
        "EQ_2": 0x4,
        "EQ_3": 0x8,
        "EQ_REQUEST": 0x10,
    }

    def __str__(self):
        return f"{{ Gen5 status {self.values()} raw {self.raw:#x} }}"

PCI_EXP_SLTCAP = 20
PCI_EXP_SLTCTL = 24
class PciSlotControl(Bitfield):
    size = 2
    fields = {
        # Attention Button Pressed Enable
        "ABPE": 0x0001,

        # Power Fault Detected Enable
        "PFDE": 0x0002,

        # MRL Sensor Changed Enable
        "MRLSCE": 0x0004,

        # Presence Detect Changed Enable
        "PDCE": 0x0008,

        # Command Completed Interrupt Enable
        "CCIE": 0x0010,

        # Hot-Plug Interrupt Enable
        "HPIE": 0x0020,

        # Attention Indicator Control
        "AIC": 0x00c0,

        # Power Indicator Control
        "PIC": 0x0300,

        # Power Controller Control
        "PCC": 0x0400,

        # Electromechanical Interlock Control
        "EIC": 0x0800,

        # Data Link Layer State Changed Enable
        "DLLSCE": 0x1000,
    }

PCI_EXP_SLTSTA = 26
class PciSlotStatus(Bitfield):
    size = 2
    fields = {
        # Attention Button Pressed
        "ABP": 0x0001,

        # Power Fault Detected
        "PFD": 0x0002,

        # MRL Sensor Changed
        "MRLSC": 0x0004,

        # Presence Detect Changed
        "PDC": 0x0008,

        # Command Completed
        "CC": 0x0010,

        # MRL Sensor State
        "MRLSS": 0x0020,

        # Presense Detect State
        "PDS": 0x0040,

        # Electromechanical Interlock Status
        "EIS": 0x0080,

        # Data Link Layer State Changed
        "DLLSC": 0x0100,
    }

PCI_EXP_LNKCTL2 = 48
class PciLinkControl2(Bitfield):
    size = 2
    fields = {
        # Target link speed
        "TLS": 0x000f,
    }

PCI_PM_CTRL = 4
class PciPmControl(Bitfield):
    size = 2
    fields = {
        "STATE": 0x0003,
        "NO_SOFT_RESET": 0x0008,
    }
