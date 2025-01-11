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

import os
import platform
from logging import debug, warning, error

from utils import DeviceField, FileRaw, FileMap
from .defines import *

is_linux = platform.system() == "Linux"
is_sysfs_available = is_linux

NV_XVE_DEV_CTRL = 0x4
NV_XVE_BAR0 = 0x10
NV_XVE_BAR1_LO = 0x14
NV_XVE_BAR1_HI = 0x18
NV_XVE_BAR2_LO = 0x1c
NV_XVE_BAR2_HI = 0x20
NV_XVE_BAR3 = 0x24
NV_XVE_VCCAP_CTRL0 = 0x114

GPU_CFG_SPACE_OFFSETS = [
    NV_XVE_DEV_CTRL,
    NV_XVE_BAR0,
    NV_XVE_BAR1_LO,
    NV_XVE_BAR1_HI,
    NV_XVE_BAR2_LO,
    NV_XVE_BAR2_HI,
    NV_XVE_BAR3,
    NV_XVE_VCCAP_CTRL0,
]

class Device:
    def __init__(self):
        self.parent = None
        self.children = []
        self.is_cx7 = False

    def is_hidden(self):
        return True

    def has_aer(self):
        return False

    def is_bridge(self):
        return False

    def is_root(self):
        return self.parent == None

    def is_gpu(self):
        return False

    def is_nvswitch(self):
        return False

    def is_plx(self):
        return False

    def is_intel(self):
        return False

    def has_dpc(self):
        return False

    def has_acs(self):
        return False

    def has_exp(self):
        return False

class PciDevice(Device):
    mmio_access_type = "sysfs"

    @staticmethod
    def _open_config(dev_path):
        dev_path_config = os.path.join(dev_path, "config")
        return FileRaw(dev_path_config, 0, os.path.getsize(dev_path_config))

    def _map_cfg_space(self):
        return self._open_config(self.dev_path)

    def __init__(self, dev_path):
        super().__init__()

        self.parent = None
        self.children = []
        self.dev_path = dev_path
        self.bdf = os.path.basename(dev_path)
        self.config = self._map_cfg_space()

        self.vendor = self.config.read16(0)
        self.device = self.config.read16(2)
        self.svid = self.config.read16(0x2c)
        self.ssid = self.config.read16(0x2e)
        self.header_type = self.config.read8(0xe)
        self.cfg_space_broken = False
        self._init_caps()
        self._init_bars()
        if not self.cfg_space_broken:
            self.command = DeviceField(PciCommand, self.config, PCI_COMMAND)
            if self.has_exp():
                self.pciflags = DeviceField(PciExpFlags, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_FLAGS)
                self.devcap = DeviceField(PciDevCap, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_DEVCAP)
                self.devctl = DeviceField(PciDevCtl, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_DEVCTL)
                self.devctl2 = DeviceField(PciDevCtl2, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_DEVCTL2)
                self.link_cap = DeviceField(PciLinkCap, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_LNKCAP)
                self.link_ctl = DeviceField(PciLinkControl, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_LNKCTL)
                self.link_status = DeviceField(PciLinkStatus, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_LNKSTA)
                self.link_status2 = DeviceField(PciLinkStatus2, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_LNKSTA2)
                # Root port or downstream port
                if self.pciflags["TYPE"] == 0x4 or self.pciflags["TYPE"] == 0x6:
                    self.link_ctl_2 = DeviceField(PciLinkControl2, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_LNKCTL2)
                if self.pciflags["TYPE"] == 4:
                    self.rtctl = DeviceField(PciRootControl, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_RTCTL)
                if self.pciflags["SLOT"] == 1:
                    self.slot_ctl = DeviceField(PciSlotControl, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_SLTCTL)
                    self.slot_status = DeviceField(PciSlotStatus, self.config, self.caps[PCI_CAP_ID_EXP] + PCI_EXP_SLTSTA)
            if self.has_aer():
                self.uncorr_status = DeviceField(PciUncorrectableErrors, self.config, self.ext_caps[PCI_EXT_CAP_ID_ERR] + PCI_ERR_UNCOR_STATUS, name="UNCOR_STATUS")
                self.uncorr_mask   = DeviceField(PciUncorrectableErrors, self.config, self.ext_caps[PCI_EXT_CAP_ID_ERR] + PCI_ERR_UNCOR_MASK, name="UNCOR_MASK")
                self.uncorr_sever  = DeviceField(PciUncorrectableErrors, self.config, self.ext_caps[PCI_EXT_CAP_ID_ERR] + PCI_ERR_UNCOR_SEVER, name="UNCOR_SEVER")
                self.corr_status   = DeviceField(PciCorrectableErrors, self.config, self.ext_caps[PCI_EXT_CAP_ID_ERR] + PCI_ERR_COR_STATUS, name="COR_STATUS")
                self.corr_mask   = DeviceField(PciCorrectableErrors, self.config, self.ext_caps[PCI_EXT_CAP_ID_ERR] + PCI_ERR_COR_MASK, name="COR_MASK")
            if self.has_pm():
                self.pmctrl = DeviceField(PciPmControl, self.config, self.caps[PCI_CAP_ID_PM] + PCI_PM_CTRL)
            if self.has_acs():
                self.acs_ctl = DeviceField(AcsCtl, self.config, self.ext_caps[PCI_EXT_CAP_ID_ACS] + PCI_EXT_ACS_CTL)
            if self.has_dpc():
                self.dpc_ctrl   = DeviceField(DpcCtl, self.config, self.ext_caps[PCI_EXT_CAP_ID_DPC] + PCI_EXP_DPC_CTL)
                self.dpc_status = DeviceField(DpcStatus, self.config, self.ext_caps[PCI_EXT_CAP_ID_DPC] + PCI_EXP_DPC_STATUS)

            if self.has_pcie_gen4():
                self.pci_gen4_status = DeviceField(PciGen4Status, self.config, self.ext_caps[PCI_EXT_CAP_GEN4] + PCI_GEN4_STATUS)

            if self.has_pcie_gen5():
                self.pci_gen5_status = DeviceField(PciGen5Status, self.config, self.ext_caps[PCI_EXT_CAP_GEN5] + PCI_GEN5_STATUS)
                self.pci_gen5_caps = DeviceField(PciGen5Caps, self.config, self.ext_caps[PCI_EXT_CAP_GEN5] + PCI_GEN5_CAPS)
                self.pci_gen5_control = DeviceField(PciGen5Control, self.config, self.ext_caps[PCI_EXT_CAP_GEN5] + PCI_GEN5_CONTROL)

        if is_sysfs_available:
            from .devices import PciDevices
            from utils.sysfs import sysfs_find_parent
            self.parent = PciDevices.find_or_init(sysfs_find_parent(dev_path))
        else:
            # Create a dummy device as the parent if sysfs is not available
            self.parent = Device()

    def _save_cfg_space(self):
        self.saved_cfg_space = {}
        for offset in GPU_CFG_SPACE_OFFSETS:
            if offset >= self.config.size:
                continue
            self.saved_cfg_space[offset] = self.config.read32(offset)
            #debug("%s saving cfg space %s = %s", self, hex(offset), hex(self.saved_cfg_space[offset]))

    def _restore_cfg_space(self):
        assert self.saved_cfg_space
        for offset in sorted(self.saved_cfg_space):
            old = self.config.read32(offset)
            new = self.saved_cfg_space[offset]
            #debug("%s restoring cfg space %s = %s to %s", self, hex(offset), hex(old), hex(new))
            self.config.write32(offset, new)

    def is_hidden(self):
        return False

    def has_aer(self):
        return PCI_EXT_CAP_ID_ERR in self.ext_caps

    def has_sriov(self):
        return PCI_EXT_CAP_ID_SRIOV in self.ext_caps

    def has_dpc(self):
        return PCI_EXT_CAP_ID_DPC in self.ext_caps

    def has_acs(self):
        return PCI_EXT_CAP_ID_ACS in self.ext_caps

    def has_exp(self):
        return PCI_CAP_ID_EXP in self.caps

    def has_pm(self):
        return PCI_CAP_ID_PM in self.caps

    def has_pcie_gen4(self):
        return PCI_EXT_CAP_GEN4 in self.ext_caps

    def has_pcie_gen5(self):
        return PCI_EXT_CAP_GEN5 in self.ext_caps

    def reinit(self):
        from .devices import PciDevices
        if self.bdf in PciDevices.DEVICES:
            del PciDevices.DEVICES[self.bdf]
        return PciDevices.find_or_init(self.dev_path)

    def get_root_port(self):
        dev = self.parent
        while dev.parent != None and not dev.parent.is_hidden():
            dev = dev.parent
        return dev

    def get_first_plx_parent(self):
        dev = self.parent
        while dev != None:
            if dev.is_plx():
                return dev
            dev = dev.parent
        return None

    def _bar_num_to_sysfs_resource(self, barnum):
        sysfs_num = barnum
        # sysfs has gaps in case of 64-bit BARs
        for b in range(barnum):
            if self.bars[b][2]:
                sysfs_num += 1
        return sysfs_num

    def _init_bars_sysfs(self):
        self.bars = []
        resources = open(os.path.join(self.dev_path, "resource")).readlines()

        # Consider only first 6 resources
        for bar_line in resources[:6]:
            bar_line = bar_line.split(" ")
            addr = int(bar_line[0], base=16)
            end = int(bar_line[1], base=16)
            flags = int(bar_line[2], base=16)
            # Skip non-MMIO regions
            if flags & 0x1 != 0:
                continue
            if addr != 0:
                size = end - addr + 1
                is_64bit = False
                if (flags >> 1) & 0x3 == 0x2:
                    is_64bit = True
                self.bars.append((addr, size, is_64bit))

    def _bar_reg_mask(self, offset, high):
        all_1 = 0xffffffff
        org = self.config.read32(offset)
        self.config.write32(offset, all_1)
        value = self.config.read32(offset)
        self.config.write32(offset, org)
        if not high:
            value &= ~0xf
        return value

    def _bar_size_32(self, offset):
        return (~self._bar_reg_mask(offset, high=False) & (2**32 - 1)) + 1

    def _bar_size_64(self, offset):
        mask = self._bar_reg_mask(offset, high=False) | (self._bar_reg_mask(offset + 4, high=True) << 32)
        return (~mask & (2**64 - 1)) + 1

    def _init_bars_config_space(self):
        self.bars = []
        if self.header_type == 0x0:
            max_bars = 6
        else:
            max_bars = 2

        bar_num = 0
        while bar_num < max_bars:
            bar_reg = self.config.read32(0x10 + bar_num * 4)
            is_mmio = bar_reg & 0x1 == 0
            if not is_mmio:
                bar_num += 1
                continue
            is_64bit = (bar_reg >> 1) & 0x3 == 0x2
            bar_addr = bar_reg & ~0xf
            if is_64bit:
                bar_addr |= self.config.read32(0x10 + (bar_num + 1) * 4) << 32
                bar_size = self._bar_size_64(0x10 + bar_num * 4)
                bar_num += 2
            else:
                bar_size = self._bar_size_32(0x10 + bar_num * 4)
                bar_num += 1
            if bar_addr != 0:
                self.bars.append((bar_addr, bar_size, is_64bit))

    def _init_bars(self):
        if is_sysfs_available:
            self._init_bars_sysfs()
        else:
            self._init_bars_config_space()

    def _map_bar(self, bar_num, bar_size=None):
        bar_addr = self.bars[bar_num][0]
        if not bar_size:
            bar_size = self.bars[bar_num][1]

        if self.mmio_access_type == "sysfs":
            return FileMap(os.path.join(self.dev_path, f"resource{self._bar_num_to_sysfs_resource(bar_num)}"), 0, bar_size)
        else:
            return FileMap("/dev/mem", bar_addr, bar_size)

    def _init_caps(self):
        import collections
        self.caps = {}
        self.ext_caps = {}
        self.ext_caps_all = collections.defaultdict(list)

        # DVSEC capability map by vendor and id
        self.dvsec_caps = {}

        cap_offset = self.config.read8(PCI_CAPABILITY_LIST)
        data = 0
        if cap_offset == 0xff:
            self.cfg_space_broken = True
            error("Broken device %s", self.dev_path)
            return
        while cap_offset != 0:
            data = self.config.read32(cap_offset)
            cap_id = data & CAP_ID_MASK
            self.caps[cap_id] = cap_offset
            cap_offset = (data >> 8) & 0xff

        self._init_ext_caps()


    def _init_ext_caps(self):
        if self.config.size <= PCI_CFG_SPACE_SIZE:
            return

        offset = PCI_CFG_SPACE_SIZE
        header = self.config.read32(PCI_CFG_SPACE_SIZE)
        offsets = set()
        while offset != 0:
            if offset in offsets:
                warning(f"{self} extended cap loop at {offset:#x}")
                return
            offsets.add(offset)
            cap = header & 0xffff
            self.ext_caps[cap] = offset
            self.ext_caps_all[cap].append(offset)

            offset = (header >> 20) & 0xffc
            header = self.config.read32(offset)

        for dvsec_offset in self.ext_caps_all[0x23]:
            header_1 = self.config.read32(dvsec_offset + 0x4)
            vendor = header_1 & 0xffff
            header_2 = self.config.read32(dvsec_offset + 0x8)
            dvsec_id = header_2 & 0xffff
            self.dvsec_caps[vendor, dvsec_id] = dvsec_offset

    def config_read_dvsec_cap(self, vendor, dvsec_id, offset_in_cap):
        offset = self.dvsec_caps.get((vendor, dvsec_id), None)
        if offset is None:
            return None

        return self.config.read32(offset + offset_in_cap)

    def __str__(self):
        return "PCI %s %s:%s" % (self.bdf, hex(self.vendor), hex(self.device))

    def __hash__(self):
        return hash((self.bdf, self.vendor, self.device))

    def set_command_memory(self, enable):
        self.command["MEMORY"] = 1 if enable else 0

    def set_bus_master(self, enable):
        self.command["MASTER"] = 1 if enable else 0

    def cfg_read8(self, offset):
        return self.config.read8(offset)

    def cfg_read32(self, offset):
        return self.config.read32(offset)

    def cfg_write32(self, offset, data):
        self.config.write32(offset, data)

    def sanity_check(self):
        if not self.sanity_check_cfg_space():
            debug("%s sanity check of config space failed", self)
            return False

        return True

    def sanity_check_cfg_space(self):
        # Use an offset unlikely to be intercepted in case of virtualization
        vendor = self.config.read16(0xf0)
        return vendor != 0xffff

    def sanity_check_cfg_space_bars(self):
        """Check whether BAR0 is configured"""
        bar0 = self.config.read32(0x10)
        if bar0 == 0:
            return False
        if bar0 == 0xffffffff:
            return False
        return True

    def sysfs_power_control_get(self):
        path = os.path.join(self.dev_path, "power", "control")
        if not os.path.exists(path):
            debug(f"{self} path not present: '{path}'")
            return "not_present"
        return open(path, "r").readlines()[0].strip()

    def sysfs_power_control_set(self, mode):
        path = os.path.join(self.dev_path, "power", "control")
        if not os.path.exists(path):
            debug("%s path not present: '%s'", self, path)
            return
        with open(path, "w") as f:
            f.write(mode)

    def sysfs_remove(self):
        remove_path = os.path.join(self.dev_path, "remove")
        if not os.path.exists(remove_path):
            debug("%s remove not present: '%s'", self, remove_path)
        with open(remove_path, "w") as f:
            f.write("1")

    def sysfs_rescan(self):
        path = os.path.join(self.dev_path, "rescan")
        if not os.path.exists(path):
            debug("%s path not present: '%s'", self, path)
        with open(path, "w") as f:
            f.write("1")

    def sysfs_unbind(self):
        unbind_path = os.path.join(self.dev_path, "driver", "unbind")
        if not os.path.exists(unbind_path):
            debug("%s unbind not present: '%s', already unbound?", self, unbind_path)
            return
        with open(unbind_path, "w") as f:
            f.write(self.bdf)
        debug("%s unbind done", self)

    def sysfs_bind(self, driver):
        bind_path = os.path.join("/sys/bus/pci/drivers/", driver, "bind")
        if not os.path.exists(bind_path):
            debug("%s bind not present: '%s'", self, bind_path)
            return
        with open(bind_path, "w") as f:
            f.write(self.bdf)
        debug("%s bind to %s done", self, driver)

    def sysfs_get_driver(self):
        driver_path = os.path.join(self.dev_path, "driver")
        if not os.path.exists(driver_path):
            return None
        return os.path.basename(os.readlink(driver_path))

    def sysfs_get_module(self):
        module_path = os.path.join(self.dev_path, "driver", "module")
        if not os.path.exists(module_path):
            return None
        return os.path.basename(os.readlink(module_path))

    def sysfs_reset(self):
        reset_path = os.path.join(self.dev_path, "reset")
        if not os.path.exists(reset_path):
            error("%s reset not present: '%s'", self, reset_path)
        with open(reset_path, "w") as rf:
            self.reset_pre()
            rf.write("1")
        debug(f"{self} reset via sysfs")

        self.reset_post()

    def reset_with_os(self):
        if is_linux:
            return self.sysfs_reset()

        # For now fallback to a custom implementation on Windows
        if self.is_flr_supported():
            return self.reset_with_flr()
        return self.reset_with_sbr()

    def is_flr_supported(self):
        if not self.has_exp():
            return False

        return self.devcap["FLR"] == 1

    def reset_pre(self):
        pass

    def reset_post(self):
        pass

    def read(self, reg):
        return self.bar0.read32(reg)
