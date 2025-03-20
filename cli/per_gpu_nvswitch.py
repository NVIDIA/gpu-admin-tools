#
# SPDX-FileCopyrightText: Copyright (c) 2018-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import traceback
from logging import debug, warning, error, info

from utils import sysfs
from utils.binaries import read_bin_ucode
from gpu import GpuError, FspRpcError



def main_per_gpu_or_nvswitch(device, opts):
    if device.is_broken_gpu():
        # If the GPU is broken, try to recover it if requested,
        # otherwise just exit immediately
        if opts.recover_broken_gpu:
            if device.parent.is_hidden():
                error("Cannot recover the GPU as the upstream port is hidden")
                return False

            # Reset the GPU with SBR and if successful,
            # remove and rescan it to recover BARs
            if device.reset_with_sbr():
                device.sysfs_remove()
                sysfs.pci_rescan()
                device.reinit()
                if device.is_broken_gpu():
                    error("Failed to recover %s", device)
                    return False
                else:
                    info("Recovered %s", device)
            else:
                error("Failed to recover %s %s", device, device.parent.link_status)
                return False
        else:
            error("%s is broken and --recover-broken-gpu was not specified, returning failure.", device)
            return False

    if not device.is_broken_gpu():

        if device.is_in_recovery():
            warning(f"{device} is in recovery")
        else:
            if device.is_gpu() and device.is_cc_query_supported:
                if device.is_boot_done():
                    cc_mode = device.query_cc_mode()
                    if cc_mode != "off":
                        warning(f"{device} has CC mode {cc_mode}, some functionality may not work")
            if device.is_ppcie_query_supported:
                if device.is_boot_done():
                    ppcie_mode = device.query_ppcie_mode()
                    if ppcie_mode != "off":
                        warning(f"{device} has PPCIe mode {ppcie_mode}, some functionality may not work")

    if opts.read_bar0 is not None:
        addr = opts.read_bar0
        data = device.read(addr)
        info("BAR0 read 0x%x = 0x%x", addr, data)

    if opts.write_bar0 is not None:
        addr = opts.write_bar0[0]
        data = opts.write_bar0[1]
        device.write_verbose(addr, data)

    if opts.debug_dump:
        info(f"{device} debug dump:")
        device.debug_dump()

    if opts.query_prc_knobs:
        if not device.has_fsp:
            error(f"Querying PRC knobs is not supported on {device}")
            return False

        prc_knobs = device.query_prc_knobs()
        info(f"{device} PRC knobs:")
        for name, value in prc_knobs:
            info(f"  {name} = {value}")


    if opts.set_ppcie_mode:
        if not device.is_ppcie_query_supported:
            error(f"Configuring PPCIe not supported on {device}")
            return False

        try:
            device.set_ppcie_mode(opts.set_ppcie_mode)
        except GpuError as err:
            if isinstance(err, FspRpcError) and err.is_invalid_knob_error:
                error(f"{device} does not support PPCIe on current FW. A FW update is required.")
                return False
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            device.debug_dump()
            prc_knobs = device.query_prc_knobs()
            debug(f"{device} PRC knobs:")
            for name, value in prc_knobs:
                debug(f"  {name} = {value}")
            raise

        info(f"{device} PPCIe mode set to {opts.set_ppcie_mode}. It will be active after GPU or switch reset.")
        if opts.reset_after_ppcie_mode_switch:
            device.reset_with_os()
            new_mode = device.query_ppcie_mode()
            if new_mode != opts.set_ppcie_mode:
                raise GpuError(f"{device} failed to switch to PPCIe mode {opts.set_ppcie_mode}, current mode is {new_mode}.")
            info(f"{device} was reset to apply the new PPCIe mode.")

    if opts.query_ppcie_mode:
        if not device.is_ppcie_query_supported:
            error(f"Querying PPCIe mode is not supported on {device}")
            return False

        ppcie_mode = device.query_ppcie_mode()
        info(f"{device} PPCIe mode is {ppcie_mode}")

    if opts.query_ppcie_settings:
        if not device.is_ppcie_query_supported:
            error(f"Querying PPCIe settings is not supported on {device}")
            return False
        try:
            ppcie_settings = device.query_ppcie_settings()
        except GpuError as err:
            if isinstance(err, FspRpcError) and err.is_invalid_knob_error:
                error(f"{device} does not support PPCIe on current FW. A FW update is required.")
                return False
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            device.debug_dump()
            prc_knobs = device.query_prc_knobs()
            debug(f"{device} PRC knobs:")
            for name, value in prc_knobs:
                debug(f"  {name} = {value}")
            raise

        info(f"{device} PPCIe settings:")
        for name, value in ppcie_settings:
            info(f"  {name} = {value}")


    if opts.test_ppcie_mode_switch:
        if not device.is_ppcie_query_supported:
            error(f"Configuring PPCIE not supported on {device}")
            return False
        try:
            device.test_ppcie_mode_switch()
        except GpuError as err:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            device.debug_dump()
            prc_knobs = device.query_prc_knobs()
            debug(f"{device} PRC knobs:")
            for name, value in prc_knobs:
                debug(f"  {name} = {value}")
            raise



    if opts.query_module_name:
        if not device.is_module_name_supported:
            error(f"{device} does not support module name query")
            return False
        info(f"{device} module name {device.module_name}")

    if opts.block_all_nvlinks or opts.block_nvlink:
        if not device.is_nvlink_supported:
            error(f"{device} does not support NVLink blocking")
            return False

        num_nvlinks = device.nvlink["number"]

        if opts.block_all_nvlinks:
            links_to_block = range(num_nvlinks)
        else:
            links_to_block = opts.block_nvlink

        for link in links_to_block:
            if link < 0 or link >= num_nvlinks:
                error(f"Invalid link {link}, num nvlinks {num_nvlinks}")
                return False

            device.block_nvlinks(links_to_block)
            info(f"{device} blocked NVLinks {links_to_block}")

    if opts.knobs_reset_to_defaults_list:
        if len(device.knob_defaults) == 0:
            raise ValueError(f"{device} does not support knob reset")
        info(f"{device} {device.knob_defaults}")

    if opts.knobs_reset_to_defaults:
        if len(device.knob_defaults) == 0:
            raise ValueError(f"{device} does not support knob reset")
        info(f"{device} resetting knobs {opts.knobs_reset_to_defaults} to defaults")
        device.knobs_reset_to_defaults(opts.knobs_reset_to_defaults, opts.knobs_reset_to_defaults_assume_no_pending_changes)

    if opts.knobs_reset_to_defaults_test:
        if len(device.knob_defaults) == 0:
            raise ValueError(f"{device} does not support knob reset")
        device.knobs_reset_to_defaults_test()

    if opts.nvlink_debug_dump:
        device.nvlink_debug()


    return True

def main_gpu_or_nvswitch_optional(device, opts):


    return True
