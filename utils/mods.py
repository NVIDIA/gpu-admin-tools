#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes
import errno
import fcntl
import mmap
import os
import platform
import threading


# Minimal /dev/mods interface needed to map PCI BARs.  The structure layouts
# and ioctl numbers mirror mods.h in the open-source MODS kernel driver:
# https://github.com/NVIDIA/mods-kernel-driver/blob/v4.31/mods.h

MODS_IOC_MAGIC = ord('x')
_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS
_IOC_WRITE = 1
_IOC_READ = 2

MODS_ACCESS_TOKEN_NONE = 0xffffffff
MODS_MEMORY_UNCACHED = 1
_SUPPORTED_IOCTL_ARCHITECTURES = frozenset(("aarch64", "x86_64"))


def _verify_ioctl_architecture():
    architecture = platform.machine().lower()
    if architecture not in _SUPPORTED_IOCTL_ARCHITECTURES:
        supported = ", ".join(sorted(_SUPPORTED_IOCTL_ARCHITECTURES))
        raise OSError(
            errno.ENOTSUP,
            f"/dev/mods MMIO access is unsupported on architecture "
            f"{architecture!r}; supported architectures: {supported}",
        )


def _IOC(direction, ioctl_type, number, size):
    return ((direction << _IOC_DIRSHIFT) |
            (ioctl_type << _IOC_TYPESHIFT) |
            (number << _IOC_NRSHIFT) |
            (size << _IOC_SIZESHIFT))


def _IOW(number, struct_type):
    return _IOC(_IOC_WRITE, MODS_IOC_MAGIC, number, ctypes.sizeof(struct_type))


def _IOWR(number, struct_type):
    return _IOC(_IOC_READ | _IOC_WRITE, MODS_IOC_MAGIC, number, ctypes.sizeof(struct_type))


class _ModsPackedStructure(ctypes.Structure):
    # ctypes implements non-default packing through its "ms" layout.  For
    # these integer-only structures, pack(1) matches MODS_PACKED exactly.
    _pack_ = 1
    _layout_ = "ms"


class ModsPciDev2(_ModsPackedStructure):
    _fields_ = [
        ("domain", ctypes.c_uint16),
        ("bus", ctypes.c_uint16),
        ("device", ctypes.c_uint16),
        ("function", ctypes.c_uint16),
    ]


class ModsGetVersion(_ModsPackedStructure):
    _fields_ = [("version", ctypes.c_uint64)]


class ModsAccessToken(_ModsPackedStructure):
    _fields_ = [("token", ctypes.c_uint32)]


class ModsPciGetBarInfo2(_ModsPackedStructure):
    _fields_ = [
        ("pci_device", ModsPciDev2),
        ("bar_index", ctypes.c_uint32),
        ("base_address", ctypes.c_uint64),
        ("bar_size", ctypes.c_uint64),
    ]


class ModsMemoryType(_ModsPackedStructure):
    _fields_ = [
        ("physical_address", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("type", ctypes.c_uint32),
    ]


MODS_ESC_GET_API_VERSION = _IOWR(17, ModsGetVersion)
MODS_ESC_SET_MEMORY_TYPE = _IOW(22, ModsMemoryType)
MODS_ESC_PCI_GET_BAR_INFO_2 = _IOWR(60, ModsPciGetBarInfo2)
MODS_ESC_VERIFY_ACCESS_TOKEN = _IOW(109, ModsAccessToken)


def _parse_bdf(bdf):
    parts = bdf.split(':')
    if len(parts) != 3 or parts[2].count('.') != 1:
        raise ValueError(f"BDF must be in DDDD:BB:DD.F format, got {bdf!r}")
    domain, bus, dev_func = parts
    device, function = dev_func.split('.')
    return ModsPciDev2(int(domain, 16), int(bus, 16),
                       int(device, 16), int(function, 16))


class _ModsSession:
    """Keep one MODS client open for the lifetime of all BAR mappings.

    Unless the driver is configured for multiple instances or an access token
    has been acquired, it accepts only one open client.  The driver also expects
    that a client's mappings are removed before that client is closed.
    """

    _lock = threading.Lock()
    _map_lock = threading.Lock()
    _fd = None
    _path = None
    # Number of live ModsBar objects using the shared client.
    _users = 0

    @classmethod
    def acquire(cls, path):
        _verify_ioctl_architecture()
        with cls._lock:
            if cls._fd is None:
                fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
                try:
                    fcntl.ioctl(fd, MODS_ESC_GET_API_VERSION, ModsGetVersion(), True)
                    # This tool does not acquire a token.  Verifying NONE makes
                    # initialization fail if a token-owning session is active.
                    fcntl.ioctl(fd, MODS_ESC_VERIFY_ACCESS_TOKEN,
                                ModsAccessToken(MODS_ACCESS_TOKEN_NONE), True)
                except Exception:
                    os.close(fd)
                    raise
                cls._fd = fd
                cls._path = path
            elif cls._path != path:
                raise OSError(f"MODS session is already open on {cls._path}")
            cls._users += 1
            return cls._fd

    @classmethod
    def mapping(cls):
        return cls._map_lock

    @classmethod
    def release(cls):
        with cls._lock:
            cls._users -= 1
            if cls._users == 0:
                os.close(cls._fd)
                cls._fd = None
                cls._path = None


class ModsBar:
    def __init__(self, bdf, bar_num, size=None, path="/dev/mods"):
        self.mapped = None
        self.view = None
        self._has_session = False

        self.fd = _ModsSession.acquire(path)
        self._has_session = True
        try:
            self._map_bar(bdf, bar_num, size)
        except Exception:
            self.close()
            raise

    def _map_bar(self, bdf, bar_num, size):
        # SET_MEMORY_TYPE updates state on the shared client which mmap then
        # reads for the requested physical range.  Keep the complete mapping
        # sequence serialized so another ModsBar cannot replace that state.
        with _ModsSession.mapping():
            bar_info = self._get_bar_info(bdf, bar_num)
            size = self._validate_size(bdf, bar_num, size, bar_info)
            self._set_memory_type(bar_info)
            self._mmap(bar_info.base_address, size)

    def _get_bar_info(self, bdf, bar_num):
        bar_info = ModsPciGetBarInfo2()
        bar_info.pci_device = _parse_bdf(bdf)
        bar_info.bar_index = bar_num
        self._ioctl(MODS_ESC_PCI_GET_BAR_INFO_2, bar_info,
                    "MODS_ESC_PCI_GET_BAR_INFO_2")

        if bar_info.base_address == 0 or bar_info.bar_size == 0:
            raise OSError(f"/dev/mods returned invalid BAR{bar_num} for {bdf}: "
                          f"base {bar_info.base_address:#x} "
                          f"size {bar_info.bar_size:#x}")
        return bar_info

    def _validate_size(self, bdf, bar_num, size, bar_info):
        if size is None:
            return bar_info.bar_size
        if size > bar_info.bar_size:
            raise ValueError(f"Requested BAR{bar_num} mapping size {size:#x} "
                             f"exceeds /dev/mods BAR size "
                             f"{bar_info.bar_size:#x} for {bdf}")
        return size

    def _set_memory_type(self, bar_info):
        # Mark the MMIO range as uncached before mapping it.
        mem_type = ModsMemoryType(bar_info.base_address,
                                  bar_info.bar_size,
                                  MODS_MEMORY_UNCACHED)
        self._ioctl(MODS_ESC_SET_MEMORY_TYPE, mem_type,
                    "MODS_ESC_SET_MEMORY_TYPE")

    def _mmap(self, base_address, size):
        page_size = mmap.ALLOCATIONGRANULARITY
        map_offset = base_address & ~(page_size - 1)
        self.map_delta = base_address - map_offset
        self.size = size
        self.map_size = ((self.map_delta + size + page_size - 1) //
                         page_size) * page_size

        self.mapped = mmap.mmap(self.fd, self.map_size,
                                flags=mmap.MAP_SHARED,
                                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                                offset=map_offset)
        self.view = memoryview(self.mapped)[self.map_delta:
                                            self.map_delta + size]
        self.map_8 = self.view.cast("B")
        self.map_16 = self.view.cast("H")
        self.map_32 = self.view.cast("I")

    def _ioctl(self, request, data, name):
        try:
            return fcntl.ioctl(self.fd, request, data, True)
        except OSError as err:
            raise OSError(err.errno, f"/dev/mods ioctl {name} failed: {err.strerror}") from err

    def close(self):
        for attr in ("map_8", "map_16", "map_32", "view"):
            if hasattr(self, attr):
                view = getattr(self, attr)
                if view is not None:
                    view.release()
                    setattr(self, attr, None)
        if self.mapped is not None:
            self.mapped.close()
            self.mapped = None
        if self._has_session:
            _ModsSession.release()
            self._has_session = False

    def __del__(self):
        self.close()

    def write8(self, offset, data):
        self.map_8[offset // 1] = data

    def write16(self, offset, data):
        self.map_16[offset // 2] = data

    def write32(self, offset, data):
        self.map_32[offset // 4] = data

    def read8(self, offset):
        return self.map_8[offset // 1]

    def read16(self, offset):
        return self.map_16[offset // 2]

    def read32(self, offset):
        return self.map_32[offset // 4]

    def read(self, offset, size):
        if size == 1:
            return self.read8(offset)
        elif size == 2:
            return self.read16(offset)
        elif size == 4:
            return self.read32(offset)
        else:
            raise ValueError(f"Unhandled read size {size}")
