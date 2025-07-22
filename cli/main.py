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
import time
import logging
import argparse

from logging import info, error, warning, debug

from cli.no_device import main_no_device
from cli.per_device import main_per_device
from cli.per_gpu_nvswitch import main_per_gpu_or_nvswitch
from utils import platform_config, FileMap, int_from_data, read_ints_from_path
from pci import PciDevice, PciDevices
from gpu.defines import *
from gpu import GpuError, FspRpcError


from pci.devices import find_gpus

VERSION = "v2025.07.21o"

# Check that modules needed to access devices on the system are available
def check_device_module_deps():
    pass

def print_topo_indent(root, indent):
    if root.is_hidden():
        indent = indent - 1
    else:
        print(" " * indent, root)
    for c in root.children:
        print_topo_indent(c, indent + 1)


def print_topo():
    print("Topo:")
    for c in PciDevices.DEVICES:
        dev = PciDevices.DEVICES[c]
        if dev.is_root():
            print_topo_indent(dev, 1)
    sys.stdout.flush()

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
    argp.add_argument("--sysfs-bind", help="Bind devices to the specified driver")
    argp.add_argument("--sysfs-unbind", action='store_true', help="Unbind devices from the current driver")
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
    argp.add_argument("--knobs-reset-to-defaults", nargs='+', help="""Set various device configuration knobs to defaults. Supported on Turing+ GPUs and NvSwitch_gen3. See --knobs-reset-to-defaults-list for the list of supported knobs and their defaults on a specific device.
The option can be specified multiple times to list specific knobs or 'all' can be used to indicate all supported ones should be reset.""")
    argp.add_argument("--knobs-reset-to-defaults-assume-no-pending-changes", action='store_true', help="Indicate that the device was reset after last time any knobs were modified. This allows the reset to defaults to be slightly optimized by querying the current state")
    argp.add_argument("--knobs-reset-to-defaults-test", action='store_true', help="Test knob setting and resetting")
    argp.add_argument("--noop", action='store_true', help="An empty option that can be used to separate nargs=+ options from positional arguments")
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
    argp.add_argument("--block-nvlink", type=auto_int, nargs='+',
                    help="Block the specified NVLinks. NVLinks will be blocked until a subsequent GPU reset (SBR on A100, FLR or SBR on Hopper GPUs [based on OOB configuration], FLR or SBR on Blackwell and later). Supported on A100 and later GPUs that have NVLinks.")
    argp.add_argument("--block-all-nvlinks", action='store_true', default=False,
                    help="Block all NVLinks. See --block-nvlink for more details.")
    argp.add_argument("--test-nvlink-blocking", action='store_true', default=False, help="Test blocking NVLinks.")
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

    subparsers = argp.add_subparsers(dest="command", required=False)
    from cli.plugins import load_plugins
    plugins = load_plugins()
    for name, plugin in plugins.items():
        plugin_parser = subparsers.add_parser(name)
        plugin.register_options(plugin_parser)

    return argp, plugins

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



# Called instead of main() when imported as a library rather than run as a
# command.
def init():
    global opts

    argp, plugins = create_args()
    opts = argp.parse_args([])

def main():
    print(f"NVIDIA GPU Tools version {VERSION}")
    print(f"Command line arguments: {sys.argv}")
    sys.stdout.flush()

    global opts

    argp, plugins = create_args()
    opts = argp.parse_args()

    logging.basicConfig(level=getattr(logging, opts.log.upper()),
                        format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d,%H:%M:%S')

    plugin = None
    if opts.command is not None:
        plugin = plugins[opts.command]
    if plugin:
        if not plugin.execute_early(opts):
            sys.exit(1)

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

    if plugin:
        if not plugin.execute_before_main(opts, devices):
            sys.exit(1)

    if len(devices) != 0:
        print_topo()
        for d in devices:
            info(f"Selected {d}")

    from .no_device import main_no_device
    from .per_gpu import main_per_gpu
    from .per_gpu_nvswitch import main_per_gpu_or_nvswitch, main_gpu_or_nvswitch_optional
    from .per_device import main_per_device

    if not main_no_device(opts):
        sys.exit(1)

    if not main_gpu_or_nvswitch_optional(device, opts):
        sys.exit(1)

    for d in devices:
        if not main_per_device(d, opts):
            sys.exit(1)
        if d.is_gpu():
            if not main_per_gpu(d, opts):
                sys.exit(1)
        if d.is_gpu() or d.is_nvswitch():
            if not main_per_gpu_or_nvswitch(d, opts):
                sys.exit(1)

    if opts.test_pcie_p2p:
        pcie_p2p_test([gpu for gpu in devices if gpu.is_gpu()])

    if plugin:
        if not plugin.execute_after_main(opts, devices):
            sys.exit(1)
