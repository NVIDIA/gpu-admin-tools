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

import os
import sys
import time
import mmap
import logging
import argparse
import traceback
import random

from logging import info, error, warning, debug

from utils import platform_config, FileMap, int_from_data, read_ints_from_path
from utils.binaries import read_bin_ucode
from pci import PciDevice, PciDevices
from pci import devices
from gpu.defines import *
from gpu import GpuError, FspRpcError


from pci.devices import find_gpus

if hasattr(time, "perf_counter"):
    perf_counter = time.perf_counter
else:
    perf_counter = time.time

if platform_config.is_linux:
    import ctypes

VERSION = "v2025.02.21o"

# Check that modules needed to access devices on the system are available
def check_device_module_deps():
    pass

class PageInfo(object):
    def __init__(self, vaddr, num_pages=1, pid="self"):
        pagemap_path = "/proc/{0}/pagemap".format(pid)
        self.page_size = os.sysconf("SC_PAGE_SIZE")
        offset  = (vaddr // self.page_size) * 8
        self._pagemap_entries = read_ints_from_path(pagemap_path, offset, int_size=8, int_num=num_pages)

    def physical_address(self, page):
        pfn = self._pagemap_entries[page] & 0x7FFFFFFFFFFFFF
        return pfn * self.page_size


def print_topo_indent(root, indent):
    if root.is_hidden():
        indent = indent - 1
    else:
        print(" " * indent, root)
    for c in root.children:
        print_topo_indent(c, indent + 1)


def gpu_dma_test(gpu, verify_reads=True, verify_writes=True):
    if platform_config.is_windows:
        error("%s DMA test on Windows is not supported currently", gpu)
        return

    #verify_reads=False
    #verify_writes=False

    use_falcon_dma = False
    if gpu.is_ampere_plus:
        # Ampere+ cannot use the bar0 window for accessing sysmem any more. Use
        # the falcon DMA path that's functional but much slower.
        use_falcon_dma = True

    if use_falcon_dma:
        sysmem_write = lambda pa, data: gpu.dma_sys_write32(pa, data)
        sysmem_read = lambda pa: gpu.dma_sys_read32(pa)
    else:
        sysmem_write = lambda pa, data: gpu.bar0_window_sys_write32(pa, data)
        sysmem_read = lambda pa: gpu.bar0_window_sys_read32(pa)

    page_size = os.sysconf("SC_PAGE_SIZE")
    total_size = 0

    # Approximate total
    total_mem_size = page_size * os.sysconf('SC_PHYS_PAGES')
    info("Total memory ~%d GBs, verifying reads %s verifying writes %s", total_mem_size // 2**30, verify_reads, verify_writes)

    start = perf_counter()
    last_debug = start
    buffers = []
    chunk_size = 8 * 1024 * 1024

    min_pa = 2**128
    max_pa = 0

    # Can't really pin pages without a kernel driver, but mlockall() should be
    # good enough for our purpose.
    # 1 is MCL_CURRENT
    # 2 is MCL_FUTURE
    flags = 2
    libc = ctypes.cdll.LoadLibrary('libc.so.6')
    libc.mlockall(ctypes.c_int(flags))

    while True:
        try:
            buf = mmap.mmap(-1, chunk_size)
        except Exception as err:
            if chunk_size == 4096:
                info("Cannot allocate any more memory")
                return
            info("Failed to allocate %d bytes, retrying with half the size", chunk_size)
            chunk_size /= 2
            continue

        # Hold onto the memory so that it cannot be allocated again
        buffers.append(buf)

        base_addr = ctypes.addressof(ctypes.c_int.from_buffer(buf))
        assert base_addr % page_size == 0

        ctypes.memset(base_addr, 0xab, chunk_size)

        num_pages = chunk_size // page_size
        page_info = PageInfo(base_addr, num_pages)

        stride = 1

        for va in range(base_addr, base_addr + chunk_size, page_size * stride):
            offset = va - base_addr
            page_index = offset // page_size
            pa = page_info.physical_address(page_index)
            min_pa = min(min_pa, pa)
            max_pa = max(max_pa, pa)

            if verify_reads:
                gpu_data = sysmem_read(pa)
                if gpu_data != 0xabababab:
                    error("VA 0x{:x} PA 0x{:x} GPU didn't read expected data after CPU write, saw 0x{:x}".format(va, pa, gpu_data))

            if verify_writes:
                sysmem_write(pa, 0xbcbcbcbc)
                gpu_data_2 = sysmem_read(pa)
                cpu_data = int_from_data(buf[offset : offset + 4], 4)
                if cpu_data != 0xbcbcbcbc:
                    error("PA 0x{:x} CPU didn't read expected data after GPU write, saw 0x{:x}".format(pa, cpu_data))
                if gpu_data_2 != 0xbcbcbcbc:
                    error("PA 0x{:x} GPU didn't read expected data after GPU write, saw 0x{:x}".format(pa, gpu_data_2))

            if perf_counter() - last_debug > 1:
                last_debug = perf_counter()
                mbs = max(total_size, 1) / (1024 * 1024.)
                t = last_debug - start
                time_left = (total_mem_size - total_size) / (mbs / t) / (1024 * 1024.)

                pa_diff_gb = (max_pa - min_pa) // 2**30
                info("So far verified %.1f MB, %.1f MB/s, time %.1f s, time left ~%.1f s, min PA 0x%x max PA 0x%x max-min %d GB", mbs, mbs/t, t, time_left, min_pa, max_pa, pa_diff_gb)

            total_size += page_size * stride

def pcie_p2p_test(gpus):
    for g1 in gpus:
        for g2 in gpus:
            if g1 == g2:
                continue
            fail = False

            g2_bar0 = g2.bar0_addr
            g2_boot_p2p = g1.dma_sys_read32(g2_bar0)
            g2_boot = g2.read(0)
            if g2_boot != g2_boot_p2p:
                error(" {0} cannot read BAR0 of {1}, read 0x{2:x} != 0x{3:x}".format(g1, g2, g2_boot_p2p, g2_boot))
                fail = True

            scratch_offset = g2.flr_resettable_scratch()
            g2.write(scratch_offset, 0)
            g1.dma_sys_write32(g2_bar0 + scratch_offset, 0xcafe)
            g2_scratch = g2.read(scratch_offset)
            g2.write(scratch_offset, 0)
            if g2_scratch != 0xcafe:
                error(" {0} cannot write BAR0 of {1}, read 0x{2:x} != 0x{3:x}".format(g1, g2, g2_scratch, 0xcafe))
                fail = True

            g2_bar1 = g2.bar1_addr
            bar1_offset = 0
            g2_bar1_misc_p2p = g1.dma_sys_read32(g2_bar1 + bar1_offset)
            g2_bar1_misc = g2.bar1.read32(bar1_offset)
            if g2_bar1_misc_p2p != g2_bar1_misc:
                error(" {0} cannot read BAR1 of {1}, read 0x{2:x} != 0x{3:x}".format(g1, g2, g2_bar1_misc_p2p, g2_bar1_misc))
                fail = True

            g2.bar1.write32(bar1_offset, 0x0)
            g1.dma_sys_write32(g2_bar1 + bar1_offset, 0xcafe)
            g2_bar1_scratch = g2.bar1.read32(bar1_offset)
            g2.bar1.write32(bar1_offset, 0x0)

            if g2_bar1_scratch != 0xcafe:
                error(" {0} cannot write BAR1 of {1}, read 0x{2:x} != 0x{3:x}".format(g1, g2, g2_bar1_scratch, 0xcafe))
                fail = True

            if not fail:
                info(" {0} can access {1} p2p boot 0x{2:x} bar1 0x{3:x}".format(g1, g2, g2_boot_p2p, g2_bar1_misc))


def print_topo():
    print("Topo:")
    for c in PciDevices.DEVICES:
        dev = PciDevices.DEVICES[c]
        if dev.is_root():
            print_topo_indent(dev, 1)
    sys.stdout.flush()

def sysfs_pci_rescan():
    with open("/sys/bus/pci/rescan", "w") as rf:
        rf.write("1")

def auto_int(x):
    return int(x, 0)

class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)

def create_args():
    argp = argparse.ArgumentParser(prog="nvidia_gpu_tools.py", formatter_class=SmartFormatter)
    if platform_config.is_sysfs_available:
        argp.add_argument("--devices",
                           help="R|Generic device selector supporting multiple comma-separated specifiers:\n"
                           "- 'gpus' - Find all NVIDIA GPUs\n"
                           "- 'gpus[n]' - Find nth NVIDIA GPU\n"
                           "- 'gpus[n:m]' - Find NVIDIA GPUs from index n to m\n"
                           "- 'nvswitches' - Find all NVIDIA NVSwitches\n"
                           "- 'nvswitches[n]' - Find nth NVIDIA NVSwitch\n"
                           "- 'vendor:device' - Find devices matching 4-digit hex vendor:device ID\n"
                           "- 'domain:bus:device.function' - Find device at specific BDF address")
    argp.add_argument("--gpu", type=auto_int, default=-1)
    argp.add_argument("--gpu-bdf", help="Select a single GPU by providing a substring of the BDF, e.g. '01:00'.")
    argp.add_argument("--gpu-name", help="Select a single GPU by providing a substring of the GPU name, e.g. 'T4'. If multiple GPUs match, the first one will be used.")
    argp.add_argument("--no-gpu", action='store_true', help="Do not use any of the GPUs; commands requiring one will not work.")
    argp.add_argument("--log", choices=['debug', 'info', 'warning', 'error', 'critical'], default='info')
    if platform_config.is_linux:
        argp.add_argument("--mmio-access-type", choices=['devmem', 'sysfs'], default='sysfs',
                          help="On Linux, specify whether to do MMIO through /dev/mem or /sys/bus/pci/devices/.../resourceN")

    argp.add_argument("--recover-broken-gpu", action='store_true', default=False,
                      help="""Attempt recovering a broken GPU (unresponsive config space or MMIO) by performing an SBR. If the GPU is
broken from the beginning and hence correct config space wasn't saved then
reenumarate it in the OS by sysfs remove/rescan to restore BARs etc.""")
    argp.add_argument("--set-next-sbr-to-fundamental-reset", action='store_true', default=False,
                      help="Configure the GPU to make the next SBR same as fundamental reset. After the SBR this setting resets back to False. Supported on H100 only.")
    argp.add_argument("--reset-with-sbr", action='store_true', default=False,
                      help="Reset the GPU with SBR and restore its config space settings, before any other actions")
    argp.add_argument("--reset-with-flr", action='store_true', default=False,
                      help="Reset the GPU with FLR and restore its config space settings, before any other actions")
    argp.add_argument("--reset-with-os", action='store_true', default=False,
                      help="Reset with OS through /sys/.../reset")
    argp.add_argument("--remove-from-os", action='store_true', default=False,
                      help="Remove from OS through /sys/.../remove")
    argp.add_argument("--unbind-gpu", action='store_true', default=False, help="Unbind GPU")
    argp.add_argument("--unbind-gpus", action='store_true', default=False, help="Unbind GPUs")
    argp.add_argument("--bind-gpu", help="Bind GPUs to the specified driver")
    argp.add_argument("--bind-gpus", help="Bind GPUs to the specified driver")
    argp.add_argument("--query-ecc-state", action='store_true', default=False,
                      help="Query the ECC state of the GPU")
    argp.add_argument("--query-cc-mode", action='store_true', default=False,
                      help="Query the current Confidential Computing (CC) mode of the GPU.")
    argp.add_argument("--query-cc-settings", action='store_true', default=False,
                      help="Query the Confidential Computing (CC) settings of the GPU."
                      "This prints the lower level setting knobs that will take effect upon GPU reset.")
    argp.add_argument("--query-ppcie-mode", action='store_true', default=False,
                      help="Query the current Protected PCIe (PPCIe) mode of the GPU or switch.")
    argp.add_argument("--query-ppcie-settings", action='store_true', default=False,
                      help="Query the Protected PPCIe (PPCIe) settings of the GPU or switch."
                      "This prints the lower level setting knobs that will take effect upon GPU or switch reset.")
    argp.add_argument("--query-prc-knobs", action='store_true', default=False,
                      help="Query all the Product Reconfiguration (PRC) knobs.")
    argp.add_argument("--set-cc-mode", choices=["off", "on", "devtools"],
                      help="Configure Confidentail Computing (CC) mode. The choices are off (disabled), on (enabled) or devtools (enabled in DevTools mode)."
                      "The GPU needs to be reset to make the selected mode active. See --reset-after-cc-mode-switch for one way of doing it.")
    argp.add_argument("--reset-after-cc-mode-switch", action='store_true', default=False,
                    help="Reset the GPU after switching CC mode such that it is activated immediately.")
    argp.add_argument("--test-cc-mode-switch", action='store_true', default=False,
                    help="Test switching CC modes.")
    argp.add_argument("--reset-after-ppcie-mode-switch", action='store_true', default=False,
                    help="Reset the GPU or switch after switching PPCIe mode such that it is activated immediately.")
    argp.add_argument("--set-ppcie-mode", choices=["off", "on"],
                      help="Configure Protected PCIe (PPCIe) mode. The choices are off (disabled) or on (enabled)."
                      "The GPU or switch needs to be reset to make the selected mode active. See --reset-after-ppcie-mode-switch for one way of doing it.")
    argp.add_argument("--test-ppcie-mode-switch", action='store_true', default=False,
                    help="Test switching PPCIE mode.")
    argp.add_argument("--set-bar0-firewall-mode", choices=["off", "on"],
                    help="Configure BAR0 firewall mode. The choices are off (disabled) or on (enabled).")
    argp.add_argument("--query-bar0-firewall-mode", action='store_true', default=False,
                    help="Query the current BAR0 firewall mode of the GPU. Blackwell+ only.")
    argp.add_argument("--query-l4-serial-number", action='store_true', default=False,
                    help="Query the L4 certificate serial number without the MSB. The MSB could be either 0x41 or 0x40 based on the RoT returning the certificate chain.")
    argp.add_argument("--query-module-name", action='store_true', help="Query the module name (aka physical ID and module ID). Supported only on H100 SXM and NVSwitch_gen3")
    argp.add_argument("--clear-memory", action='store_true', default=False,
                      help="Clear the contents of the GPU memory. Supported on Pascal+ GPUs. Assumes the GPU has been reset with SBR prior to this operation and can be comined with --reset-with-sbr if not.")

    argp.add_argument("--debug-dump", action='store_true', default=False, help="Dump various state from the device for debug")
    argp.add_argument("--nvlink-debug-dump", action="store_true", help="Dump NVLINK debug state.")
    argp.add_argument("--knobs-reset-to-defaults-list", action='store_true', help="""Show the supported knobs and their default state""")
    argp.add_argument("--knobs-reset-to-defaults", action='append', help="""Set various device configuration knobs to defaults. Supported on Turing+ GPUs and NvSwitch_gen3. See --reset-knobs-to-defaults-query for the list of supported knobs and their defaults on a specific device.
The option can be specified multiple times to list specific knobs or 'all' can be used to indicate all supported ones should be reset.""")
    argp.add_argument("--knobs-reset-to-defaults-assume-no-pending-changes", action='store_true', help="Indicate that the device was reset after last time any knobs were modified. This allows the reset to defaults to be slightly optimized by querying the current state")
    argp.add_argument("--knobs-reset-to-defaults-test", action='store_true', help="Test knob setting and resetting")
    argp.add_argument("--force-ecc-on-after-reset", action='store_true', default=False,
                    help="Force ECC to be enabled after a subsequent GPU reset")
    argp.add_argument("--test-ecc-toggle", action='store_true', default=False,
                    help="Test toggling ECC mode.")
    argp.add_argument("--query-mig-mode", action='store_true', default=False,
                    help="Query whether MIG mode is enabled.")
    argp.add_argument("--force-mig-off-after-reset", action='store_true', default=False,
                    help="Force MIG mode to be disabled after a subsequent GPU reset")
    argp.add_argument("--test-mig-toggle", action='store_true', default=False,
                    help="Test toggling MIG mode.")
    argp.add_argument("--block-nvlink", type=auto_int, action='append',
                    help="Block the specified NVLink. Can be specified multiple times to block more NVLinks. NVLinks will be blocked until an SBR. Supported on A100 only.")
    argp.add_argument("--block-all-nvlinks", action='store_true', default=False,
                    help="Block all NVLinks. NVLinks will be blocked until a subsequent SBR. Supported on A100 only.")
    argp.add_argument("--dma-test", action='store_true', default=False,
                    help="Check that GPUs are able to perform DMA to all/most of available system memory.")
    argp.add_argument("--test-pcie-p2p", action='store_true', default=False,
                    help="Check that all GPUs are able to perform DMA to each other.")
    argp.add_argument("--read-sysmem-pa", type=auto_int, help="""Use GPU's DMA to read 32-bits from the specified sysmem physical address""")
    argp.add_argument("--write-sysmem-pa", type=auto_int, nargs=2, help="""Use GPU's DMA to write specified 32-bits to the specified sysmem physical address""")
    argp.add_argument("--read-config-space", type=auto_int, help="""Read 32-bits from device's config space at specified offset""")
    argp.add_argument("--write-config-space", type=auto_int, nargs=2, help="""Write 32-bit to device's config space at specified offset""")
    argp.add_argument("--read-bar0", type=auto_int, help="""Read 32-bits from GPU BAR0 at specified offset""")
    argp.add_argument("--write-bar0", type=auto_int, nargs=2, help="""Write 32-bit to GPU BAR0 at specified offset""")
    argp.add_argument("--read-bar1", type=auto_int, help="""Read 32-bits from GPU BAR1 at specified offset""")
    argp.add_argument("--write-bar1", type=auto_int, nargs=2, help="""Write 32-bit to GPU BAR1 at specified offset""")
    argp.add_argument("--ignore-nvidia-driver", action='store_true', default=False, help="Do not treat nvidia driver apearing to be loaded as an error")

    return argp

# Called instead of main() when imported as a library rather than run as a
# command.
def init():
    global opts

    argp = create_args()
    opts = argp.parse_args([])

def main():
    print(f"NVIDIA GPU Tools version {VERSION}")
    print(f"Command line arguments: {sys.argv}")
    sys.stdout.flush()

    global opts

    argp = create_args()
    opts = argp.parse_args()

    logging.basicConfig(level=getattr(logging, opts.log.upper()),
                        format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')

    if platform_config.is_linux:
        PciDevice.mmio_access_type = opts.mmio_access_type

    if not opts.no_gpu:
        check_device_module_deps()



    if opts.gpu_bdf is not None:
        gpus, other = find_gpus(opts.gpu_bdf)
        if len(gpus) == 0:
            error("Matching for {0} found nothing".format(opts.gpu_bdf))
            sys.exit(1)
        elif len(gpus) > 1:
            error("Matching for {0} found more than one GPU {1}".format(opts.gpu_bdf, ", ".join([str(g) for g in gpus])))
            sys.exit(1)
        else:
            gpu = gpus[0]
        device = gpu
        devices = [gpu]
    elif opts.gpu_name is not None:
        gpus, other = find_gpus()
        gpus = [g for g in gpus if opts.gpu_name in g.name]
        if len(gpus) == 0:
            error("Matching for {0} found nothing".format(opts.gpu_name))
            sys.exit(1)
        gpu = gpus[0]
        device = gpu
        devices = [gpu]
    elif platform_config.is_sysfs_available and opts.devices:
        import pci
        devices = pci.devices.find_devices_from_string(opts.devices)
        if len(devices) == 0:
            error(f"No devices found matching: {opts.devices}")
            sys.exit(1)
        device = devices[0]
    elif opts.no_gpu:
        gpu = None
        device = None
        devices = []
        info("Using no GPU")
    else:
        gpus, other = find_gpus()
        print("GPUs:")
        for i, g in enumerate(gpus):
            print(" ", i, g)
        print("Other:")
        for i, o in enumerate(other):
            print(" ", i, o)
        sys.stdout.flush()

        if opts.gpu == -1:
            info("No GPU specified, select GPU with --gpu, --gpu-bdf, or --gpu-name")
            return 0

        if opts.gpu >= len(gpus):
            raise ValueError("GPU index out of bounds")
        gpu = gpus[opts.gpu]
        device = gpu
        devices = [gpu]


    if device and device.is_gpu():
        gpu = device

    if len(devices) != 0:
        print_topo()
        for d in devices:
            info(f"Selected {d}")

    if gpu:
        if gpu.is_driver_loaded():
            if not opts.ignore_nvidia_driver:
                error("The nvidia driver appears to be using %s, aborting. Specify --ignore-nvidia-driver to ignore this check.", gpu)
                sys.exit(1)
            else:
                warning("The nvidia driver appears to be using %s, but --ignore-nvidia-driver was specified, continuing.", gpu)

        if not gpu.is_broken_gpu():

            if gpu.is_in_recovery():
                warning(f"{gpu} is in recovery")
            else:
                if gpu.is_gpu() and gpu.is_hopper_plus:
                    if gpu.is_boot_done():
                        cc_mode = gpu.query_cc_mode()
                        if cc_mode != "off":
                            warning(f"{gpu} has CC mode {cc_mode}, some functionality may not work")
                if gpu.is_ppcie_query_supported:
                    if gpu.is_boot_done():
                        ppcie_mode = gpu.query_ppcie_mode()
                        if ppcie_mode != "off":
                            warning(f"{gpu} has PPCIe mode {ppcie_mode}, some functionality may not work")

    if gpu:

        if opts.unbind_gpu:
            gpu.sysfs_unbind()

        if opts.unbind_gpus:
            for dev in gpus:
                if dev.is_gpu():
                    dev.sysfs_unbind()

        if opts.bind_gpu:
            gpu.sysfs_bind(opts.bind_gpu)

        if opts.bind_gpus:
            for dev in gpus:
                if dev.is_gpu():
                    dev.sysfs_bind(opts.bind_gpus)


        if gpu.is_broken_gpu():
            # If the GPU is broken, try to recover it if requested,
            # otherwise just exit immediately
            if opts.recover_broken_gpu:
                if gpu.parent.is_hidden():
                    error("Cannot recover the GPU as the upstream port is hidden")
                    sys.exit(1)
                    return

                # Reset the GPU with SBR and if successful,
                # remove and rescan it to recover BARs
                if gpu.reset_with_sbr():
                    gpu.sysfs_remove()
                    sysfs_pci_rescan()
                    gpu.reinit()
                    if gpu.is_broken_gpu():
                        error("Failed to recover %s", gpu)
                        sys.exit(1)
                    else:
                        info("Recovered %s", gpu)
                else:
                    error("Failed to recover %s %s", gpu, gpu.parent.link_status)
                    sys.exit(1)
            else:
                error("%s is broken and --recover-broken-gpu was not specified, returning failure.", gpu)
                sys.exit(1)
            return


    if opts.set_next_sbr_to_fundamental_reset:
        if not gpu.is_reset_coupling_supported:
            error(f"{gpu} does not support reset coupling")
            sys.exit(1)
        if not gpu.set_next_sbr_to_fundamental_reset():
            error(f"{gpu} failed to configure next SBR to fundamental reset")
            sys.exit(1)
        info(f"{gpu} configured next SBR to fundamental reset")

    # Reset the GPU with SBR, if requested
    if opts.reset_with_sbr:
        if not gpu.parent.is_bridge():
            error("Cannot reset the GPU with SBR as the upstream bridge is not accessible")
        else:
            gpu.reset_with_sbr()

    # Reset the GPU with FLR, if requested
    if opts.reset_with_flr:
        if gpu.is_flr_supported():
            gpu.reset_with_flr()
        else:
            error("Cannot reset the GPU with FLR as it is not supported")

    if opts.reset_with_os:
        gpu.sysfs_reset()

    if opts.remove_from_os:
        gpu.sysfs_remove()

    if opts.query_ecc_state:
        if not gpu.is_gpu() or not gpu.is_ecc_query_supported:
            error("Querying ECC state not supported on %s", gpu)
            sys.exit(1)

        ecc_state = gpu.query_final_ecc_state()
        info("%s ECC is %s", gpu, "enabled" if ecc_state else "disabled")

    if opts.query_cc_settings:
        if not gpu.is_gpu() or not gpu.is_cc_query_supported:
            error(f"Querying CC settings is not supported on {gpu}")
            sys.exit(1)

        cc_settings = gpu.query_cc_settings()
        info(f"{gpu} CC settings:")
        for name, value in cc_settings:
            info(f"  {name} = {value}")

    if opts.query_ppcie_settings:
        if not gpu.is_ppcie_query_supported:
            error(f"Querying PPCIe settings is not supported on {gpu}")
            sys.exit(1)

        try:
            ppcie_settings = gpu.query_ppcie_settings()
        except GpuError as err:
            if isinstance(err, FspRpcError) and err.is_invalid_knob_error:
                error(f"{gpu} does not support PPCIe on current FW. A FW update is required.")
                sys.exit(1)
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            gpu.debug_dump()
            prc_knobs = gpu.query_prc_knobs()
            debug(f"{gpu} PRC knobs:")
            for name, value in prc_knobs:
                debug(f"  {name} = {value}")
            raise

        info(f"{gpu} PPCIe settings:")
        for name, value in ppcie_settings:
            info(f"  {name} = {value}")

    if opts.query_prc_knobs:
        if not gpu.has_fsp:
            error(f"Querying PRC knobs is not supported on {gpu}")
            sys.exit(1)

        prc_knobs = gpu.query_prc_knobs()
        info(f"{gpu} PRC knobs:")
        for name, value in prc_knobs:
            info(f"  {name} = {value}")

    if opts.set_cc_mode:
        if not gpu.is_gpu() or not gpu.is_cc_query_supported:
            error(f"Configuring CC not supported on {gpu}")
            sys.exit(1)

        try:
            gpu.set_cc_mode(opts.set_cc_mode)
        except GpuError as err:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            gpu.debug_dump()
            prc_knobs = gpu.query_prc_knobs()
            debug(f"{gpu} PRC knobs:")
            for name, value in prc_knobs:
                debug(f"  {name} = {value}")
            raise

        info(f"{gpu} CC mode set to {opts.set_cc_mode}. It will be active after GPU reset.")
        if opts.reset_after_cc_mode_switch:
            gpu.reset_with_os()
            new_mode = gpu.query_cc_mode()
            if new_mode != opts.set_cc_mode:
                raise GpuError(f"{gpu} failed to switch to CC mode {opts.set_cc_mode}, current mode is {new_mode}.")
            info(f"{gpu} was reset to apply the new CC mode.")

    if opts.query_cc_mode:
        if not gpu.is_gpu() or not gpu.is_cc_query_supported:
            error(f"Querying CC mode is not supported on {gpu}")
            sys.exit(1)

        cc_mode = gpu.query_cc_mode()
        info(f"{gpu} CC mode is {cc_mode}")

    if opts.set_ppcie_mode:
        if not gpu.is_ppcie_query_supported:
            error(f"Configuring PPCIe not supported on {gpu}")
            sys.exit(1)

        try:
            gpu.set_ppcie_mode(opts.set_ppcie_mode)
        except GpuError as err:
            if isinstance(err, FspRpcError) and err.is_invalid_knob_error:
                error(f"{gpu} does not support PPCIe on current FW. A FW update is required.")
                sys.exit(1)
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            gpu.debug_dump()
            prc_knobs = gpu.query_prc_knobs()
            debug(f"{gpu} PRC knobs:")
            for name, value in prc_knobs:
                debug(f"  {name} = {value}")
            raise

        info(f"{gpu} PPCIe mode set to {opts.set_ppcie_mode}. It will be active after GPU or switch reset.")
        if opts.reset_after_ppcie_mode_switch:
            gpu.reset_with_os()
            new_mode = gpu.query_ppcie_mode()
            if new_mode != opts.set_ppcie_mode:
                raise GpuError(f"{gpu} failed to switch to PPCIe mode {opts.set_ppcie_mode}, current mode is {new_mode}.")
            info(f"{gpu} was reset to apply the new PPCIe mode.")

    if opts.query_ppcie_mode:
        if not gpu.is_ppcie_query_supported:
            error(f"Querying PPCIe mode is not supported on {gpu}")
            sys.exit(1)

        ppcie_mode = gpu.query_ppcie_mode()
        info(f"{gpu} PPCIe mode is {ppcie_mode}")

    if opts.query_bar0_firewall_mode:
        if not gpu.is_bar0_firewall_supported:
            error(f"Querying BAR0 firewall mode is not supported on {gpu}")
            sys.exit(1)

        bar0_firewall_mode = gpu.query_bar0_firewall_mode()
        info(f"{gpu} BAR0 firewall mode is {bar0_firewall_mode}")

    if opts.set_bar0_firewall_mode:
        if not gpu.is_bar0_firewall_supported:
            error(f"Configuring BAR0 firewall is not supported on {gpu}")
            sys.exit(1)
        try:
            gpu.set_bar0_firewall_mode(opts.set_bar0_firewall_mode)
        except GpuError as err:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            gpu.debug_dump()
            prc_knobs = gpu.query_prc_knobs()
            debug(f"{gpu} PRC knobs:")
            for name, value in prc_knobs:
                debug(f"  {name} = {value}")
            raise

        info(f"{gpu} BAR0 firewall mode set to {opts.set_bar0_firewall_mode}. It will be active after device reset.")

    if opts.test_cc_mode_switch:
        if not gpu.is_gpu() or not gpu.is_cc_query_supported:
            error(f"Configuring CC not supported on {gpu}")
            sys.exit(1)
        try:
            gpu.test_cc_mode_switch()
        except GpuError as err:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            gpu.debug_dump()
            prc_knobs = gpu.query_prc_knobs()
            debug(f"{gpu} PRC knobs:")
            for name, value in prc_knobs:
                debug(f"  {name} = {value}")
            raise

    if opts.test_ppcie_mode_switch:
        if not gpu.is_ppcie_query_supported:
            error(f"Configuring PPCIE not supported on {gpu}")
            sys.exit(1)
        try:
            gpu.test_ppcie_mode_switch()
        except GpuError as err:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)
            gpu.debug_dump()
            prc_knobs = gpu.query_prc_knobs()
            debug(f"{gpu} PRC knobs:")
            for name, value in prc_knobs:
                debug(f"  {name} = {value}")
            raise

    if opts.query_l4_serial_number:
        if not gpu.has_pdi:
            error(f"Querying L4 serial number not supported on {gpu}")
            sys.exit(1)
        print(f"L4 serial number: {gpu.get_pdi():#x}")

    if opts.clear_memory:
        if gpu.is_memory_clear_supported:
            gpu.clear_memory()

        else:
            error("Clearing memory not supported on %s", gpu)


    if opts.debug_dump:
        info(f"{gpu} debug dump:")
        gpu.debug_dump()



    if opts.force_ecc_on_after_reset:
        if gpu.is_forcing_ecc_on_after_reset_supported:
            gpu.force_ecc_on_after_reset()
        else:
            error("Forcing ECC on after reset not supported on %s", gpu)

    if opts.test_ecc_toggle:
        if gpu.is_forcing_ecc_on_after_reset_supported:
            try:
                gpu.test_ecc_toggle()
            except Exception as err:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)
                error("%s", str(err))
                error("%s testing ECC toggle failed", gpu)
                sys.exit(1)
        else:
            error("Toggling ECC not supported on %s", gpu)
            sys.exit(1)

    if opts.query_mig_mode:
        mig_state = gpu.query_mig_mode()
        info("%s MIG mode is %s", gpu, "enabled" if mig_state else "disabled")

    if opts.force_mig_off_after_reset:
        if gpu.is_mig_mode_supported:
            gpu.set_mig_mode_after_reset(enabled=False)
        else:
            error("Forcing MIG off after reset not supported on %s", gpu)

    if opts.test_mig_toggle:
        if gpu.is_mig_mode_supported:
            try:
                gpu.test_mig_toggle()
            except Exception as err:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)
                error("%s", str(err))
                error("%s testing MIG toggle failed", gpu)
                sys.exit(1)
        else:
            error("Toggling MIG not supported on %s", gpu)
            sys.exit(1)



    if opts.block_all_nvlinks or opts.block_nvlink:
        if not gpu.is_nvlink_supported:
            error(f"{gpu} does not support NVLink blocking")
            sys.exit(1)

        num_nvlinks = gpu.nvlink["number"]

        if opts.block_all_nvlinks:
            links_to_block = range(num_nvlinks)
        else:
            links_to_block = opts.block_nvlink

        for link in links_to_block:
            if link < 0 or link >= num_nvlinks:
                error(f"Invalid link {link}, num nvlinks {num_nvlinks}")
                sys.exit(1)

        gpu.block_nvlinks(links_to_block)
        info(f"{gpu} blocked NVLinks {links_to_block}")

    if opts.dma_test:
        gpu_dma_test(gpu)

    if opts.test_pcie_p2p:
        pcie_p2p_test([gpu for gpu in gpus if gpu.is_gpu()])

    if opts.read_sysmem_pa is not None:
        addr = opts.read_sysmem_pa

        data = gpu.dma_sys_read32(addr)
        info("%s read PA 0x%x = 0x%x", gpu, addr, data)

    if opts.write_sysmem_pa:
        addr = opts.write_sysmem_pa[0]
        data = opts.write_sysmem_pa[1]

        gpu.dma_sys_write32(addr, data)
        info("%s wrote PA 0x%x = 0x%x", gpu, addr, data)

    if opts.read_config_space is not None:
        addr = opts.read_config_space
        data = gpu.config.read32(addr)
        info("Config space read 0x%x = 0x%x", addr, data)

    if opts.write_config_space is not None:
        addr = opts.write_config_space[0]
        data = opts.write_config_space[1]
        gpu.config.write32(addr, data)

    if opts.read_bar0 is not None:
        addr = opts.read_bar0
        data = gpu.read(addr)
        info("BAR0 read 0x%x = 0x%x", addr, data)

    if opts.write_bar0 is not None:
        addr = opts.write_bar0[0]
        data = opts.write_bar0[1]
        gpu.write_verbose(addr, data)

    if opts.read_bar1 is not None:
        addr = opts.read_bar1
        data = gpu.read_bar1(addr)
        info("BAR1 read 0x%x = 0x%x", addr, data)

    if opts.write_bar1:
        addr = opts.write_bar1[0]
        data = opts.write_bar1[1]

        gpu.write_bar1(addr, data)

    if opts.knobs_reset_to_defaults_list:
        if len(gpu.knob_defaults) == 0:
            raise ValueError(f"{gpu} does not support knob reset")
        info(f"{gpu} {gpu.knob_defaults}")

    if opts.knobs_reset_to_defaults:
        if len(gpu.knob_defaults) == 0:
            raise ValueError(f"{gpu} does not support knob reset")
        gpu.knobs_reset_to_defaults(opts.knobs_reset_to_defaults, opts.knobs_reset_to_defaults_assume_no_pending_changes)

    if opts.knobs_reset_to_defaults_test:
        if len(gpu.knob_defaults) == 0:
            raise ValueError(f"{gpu} does not support knob reset")
        gpu.knobs_reset_to_defaults_test()




    if opts.nvlink_debug_dump:
        gpu.nvlink_debug()
