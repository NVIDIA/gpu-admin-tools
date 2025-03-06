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

from logging import error, info

def main_per_device(device, opts):
    if opts.sysfs_unbind:
        device.sysfs_unbind()

    if opts.sysfs_bind:
        device.sysfs_bind(opts.sysfs_bind)

    if opts.reset_with_sbr:
        if not device.parent.is_bridge():
            error("Cannot reset the GPU with SBR as the upstream bridge is not accessible")
        else:
            device.reset_with_sbr()

    if opts.reset_with_flr:
        if device.is_flr_supported():
            device.reset_with_flr()
        else:
            error(f"{device} cannot reset with FLR as it is not supported")
            return False

    if opts.reset_with_os:
        device.reset_with_os()

    if opts.remove_from_os:
        device.sysfs_remove()

    if opts.read_config_space is not None:
        addr = opts.read_config_space
        data = device.config.read32(addr)
        info("Config space read 0x%x = 0x%x", addr, data)

    if opts.write_config_space is not None:
        addr = opts.write_config_space[0]
        data = opts.write_config_space[1]
        device.config.write32(addr, data)



    return True
