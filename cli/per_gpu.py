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
import traceback
from logging import debug, warning, error, info

from utils import platform_config, int_from_data, read_ints_from_path
from gpu import GpuError


import time
if hasattr(time, "perf_counter"):
    perf_counter = time.perf_counter
else:
    perf_counter = time.time


class PageInfo(object):
    def __init__(self, vaddr, num_pages=1, pid="self"):
        pagemap_path = "/proc/{0}/pagemap".format(pid)
        self.page_size = os.sysconf("SC_PAGE_SIZE")
        offset  = (vaddr // self.page_size) * 8
        self._pagemap_entries = read_ints_from_path(pagemap_path, offset, int_size=8, int_num=num_pages)

    def physical_address(self, page):
        pfn = self._pagemap_entries[page] & 0x7FFFFFFFFFFFFF
        return pfn * self.page_size

def gpu_dma_test(gpu, verify_reads=True, verify_writes=True):
    if platform_config.is_windows:
        error("%s DMA test on Windows is not supported currently", gpu)
        return

    import ctypes
    import mmap

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

def main_per_gpu(gpu, opts):
    if gpu.is_driver_loaded():
        if not opts.ignore_nvidia_driver:
            error("The nvidia driver appears to be using %s, aborting. Specify --ignore-nvidia-driver to ignore this check.", gpu)
            return False
        else:
            warning("The nvidia driver appears to be using %s, but --ignore-nvidia-driver was specified, continuing.", gpu)



    if opts.set_next_sbr_to_fundamental_reset:
        if not gpu.is_reset_coupling_supported:
            error(f"{gpu} does not support reset coupling")
            return False
        if not gpu.set_next_sbr_to_fundamental_reset():
            error(f"{gpu} failed to configure next SBR to fundamental reset")
            return False
        info(f"{gpu} configured next SBR to fundamental reset")

    if opts.query_ecc_state:
        if not gpu.is_gpu() or not gpu.is_ecc_query_supported:
            error("Querying ECC state not supported on %s", gpu)
            return False

        ecc_state = gpu.query_final_ecc_state()
        info("%s ECC is %s", gpu, "enabled" if ecc_state else "disabled")

    if opts.query_cc_settings:
        if not gpu.is_gpu() or not gpu.is_cc_query_supported:
            error(f"Querying CC settings is not supported on {gpu}")
            return False

        cc_settings = gpu.query_cc_settings()
        info(f"{gpu} CC settings:")
        for name, value in cc_settings:
            info(f"  {name} = {value}")

    if opts.set_cc_mode:
        if not gpu.is_gpu() or not gpu.is_cc_query_supported:
            error(f"Configuring CC not supported on {gpu}")
            return False

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
            return False

        cc_mode = gpu.query_cc_mode()
        info(f"{gpu} CC mode is {cc_mode}")

    if opts.query_bar0_firewall_mode:
        if not gpu.is_bar0_firewall_supported:
            error(f"Querying BAR0 firewall mode is not supported on {gpu}")
            return False

        bar0_firewall_mode = gpu.query_bar0_firewall_mode()
        info(f"{gpu} BAR0 firewall mode is {bar0_firewall_mode}")

    if opts.set_bar0_firewall_mode:
        if not gpu.is_bar0_firewall_supported:
            error(f"Configuring BAR0 firewall is not supported on {gpu}")
            return False
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
            return False
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

    if opts.query_l4_serial_number:
        if not gpu.has_pdi:
            error(f"Querying L4 serial number not supported on {gpu}")
            return False
        print(f"L4 serial number: {gpu.get_pdi():#x}")

    if opts.clear_memory:
        if gpu.is_memory_clear_supported:
            gpu.clear_memory()

        else:
            error("Clearing memory not supported on %s", gpu)


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
                return False
        else:
            error("Toggling ECC not supported on %s", gpu)
            return False

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
                return False
        else:
            error("Toggling MIG not supported on %s", gpu)
            return False


    if opts.dma_test:
        gpu_dma_test(gpu)

    if opts.read_sysmem_pa is not None:
        addr = opts.read_sysmem_pa

        data = gpu.dma_sys_read32(addr)
        info("%s read PA 0x%x = 0x%x", gpu, addr, data)

    if opts.write_sysmem_pa:
        addr = opts.write_sysmem_pa[0]
        data = opts.write_sysmem_pa[1]

        gpu.dma_sys_write32(addr, data)
        info("%s wrote PA 0x%x = 0x%x", gpu, addr, data)

    if opts.read_bar1 is not None:
        addr = opts.read_bar1
        data = gpu.read_bar1(addr)
        info("BAR1 read 0x%x = 0x%x", addr, data)

    if opts.write_bar1:
        addr = opts.write_bar1[0]
        data = opts.write_bar1[1]

        gpu.write_bar1(addr, data)




    return True
