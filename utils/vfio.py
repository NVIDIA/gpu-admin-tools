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

"""PCI register access through vfio-pci.

The legacy VFIO group/container interface uses a real TYPE1 IOMMU, with no
DMA mappings.  Sysfs provides discovery and per-device driver binding.
The ABI is defined in Linux include/uapi/linux/vfio.h:
https://docs.kernel.org/driver-api/vfio.html
"""

import ctypes
import errno
import fcntl
import mmap
import operator
import os
import platform
import re
import struct
import subprocess
import threading

from utils.vfio_abi import (
    VFIO_API_VERSION,
    VFIO_TYPE1_IOMMU,
    VFIO_TYPE1v2_IOMMU,
    VFIO_GET_API_VERSION,
    VFIO_CHECK_EXTENSION,
    VFIO_SET_IOMMU,
    VFIO_GROUP_GET_STATUS,
    VFIO_GROUP_SET_CONTAINER,
    VFIO_GROUP_UNSET_CONTAINER,
    VFIO_GROUP_GET_DEVICE_FD,
    VFIO_DEVICE_GET_INFO,
    VFIO_DEVICE_GET_REGION_INFO,
    VFIO_GROUP_FLAGS_VIABLE,
    VFIO_GROUP_FLAGS_CONTAINER_SET,
    VFIO_DEVICE_FLAGS_PCI,
    VFIO_REGION_INFO_FLAG_READ,
    VFIO_REGION_INFO_FLAG_WRITE,
    VFIO_REGION_INFO_FLAG_MMAP,
    VFIO_REGION_INFO_FLAG_CAPS,
    VFIO_REGION_INFO_CAP_SPARSE_MMAP,
    VFIO_PCI_CONFIG_REGION_INDEX,
    VfioGroupStatus,
    VfioDeviceInfo,
    VfioRegionInfo,
)


SYSFS_PCI_DEVICES = "/sys/bus/pci/devices"
SYSFS_PCI_DRIVERS = "/sys/bus/pci/drivers"
SYSFS_PCI_DRIVERS_PROBE = "/sys/bus/pci/drivers_probe"
VFIO_DEV_PATH = "/dev/vfio"


def _ioctl(fd, request, data=0):
    try:
        return fcntl.ioctl(fd, request, data, True)
    except OSError as err:
        raise OSError(err.errno,
                      f"VFIO ioctl {request:#x} failed: {err.strerror}") from err


def _validate_bdf(bdf):
    architecture = platform.machine().lower()
    if architecture not in ("aarch64", "x86_64"):
        raise OSError(errno.ENOTSUP,
                      f"VFIO MMIO is unsupported on architecture {architecture!r}")
    if not re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-1][0-9a-fA-F]\.[0-7]", bdf):
        raise ValueError(f"BDF must be in DDDD:BB:DD.F format, got {bdf!r}")


def _get_driver(bdf):
    try:
        return os.path.basename(os.readlink(os.path.join(SYSFS_PCI_DEVICES, bdf, "driver")))
    except FileNotFoundError:
        return None


def _iommu_group_path(bdf):
    device_path = os.path.join(SYSFS_PCI_DEVICES, bdf)
    try:
        group = os.path.basename(os.readlink(os.path.join(device_path, "iommu_group")))
    except FileNotFoundError as err:
        raise OSError(errno.ENODEV,
                      f"{bdf} has no IOMMU group; VFIO access requires an enabled "
                      "IOMMU with device isolation") from err
    if not group.isascii() or not group.isdecimal():
        raise OSError(errno.ENODEV, f"{bdf} has an invalid IOMMU group {group!r}")
    return os.path.join(VFIO_DEV_PATH, group)


def _group_path(bdf):
    _validate_bdf(bdf)
    driver = _get_driver(bdf)
    if driver != "vfio-pci":
        raise OSError(errno.ENODEV,
                      f"{bdf} is bound to {driver or 'unbound'}; VFIO access requires "
                      "it to be bound to vfio-pci before running this tool")
    return _iommu_group_path(bdf)


def _select_iommu(fd):
    if _ioctl(fd, VFIO_GET_API_VERSION) != VFIO_API_VERSION:
        raise OSError(errno.ENOTSUP, "Unsupported VFIO API version")
    for kind in (VFIO_TYPE1v2_IOMMU, VFIO_TYPE1_IOMMU):
        if _ioctl(fd, VFIO_CHECK_EXTENSION, kind) > 0:
            return kind
    raise OSError(errno.ENOTSUP, "VFIO requires a TYPE1 or TYPE1v2 IOMMU backend")


def _write_sysfs(path, value):
    with open(path, "w") as stream:
        stream.write(value + "\n")


def _restore_driver(bdf, original_driver, override_path):
    driver = _get_driver(bdf)
    if driver == original_driver:
        return
    if driver == "vfio-pci":
        _write_sysfs(os.path.join(SYSFS_PCI_DRIVERS, driver, "unbind"), bdf)
    elif driver is not None:
        raise OSError(errno.EBUSY,
                      f"Cannot restore {bdf}: another driver ({driver}) has bound it")
    if original_driver is not None:
        _write_sysfs(override_path, original_driver)
        _write_sysfs(SYSFS_PCI_DRIVERS_PROBE, bdf)
    if _get_driver(bdf) != original_driver:
        raise OSError(errno.EIO, f"Could not restore {bdf} to {original_driver or 'unbound'}")


def bind_vfio_pci(bdf, force=False):
    """Bind one PCI endpoint to vfio-pci, preserving its driver_override.

    Existing drivers require force=True.  A successful binding is retained
    after the tool exits.  On failure, attempt to restore the original driver.
    Return whether this call established a new binding.
    """
    bdf = bdf.lower()
    _validate_bdf(bdf)
    original_driver = _get_driver(bdf)
    if original_driver == "vfio-pci":
        return False
    if original_driver is not None and not force:
        raise OSError(errno.EBUSY,
                      f"{bdf} is bound to {original_driver}; use --vfio-force-bind "
                      "to unbind its current driver and bind vfio-pci")

    device_path = os.path.join(SYSFS_PCI_DEVICES, bdf)
    # vfio-pci supports normal endpoints, not bridges.  Check before unbinding
    # a host driver such as pcieport, even when force=True was requested.
    with open(os.path.join(device_path, "config"), "rb") as config:
        config.seek(0x0e)
        header = config.read(1)
    if len(header) != 1 or header[0] & 0x7f:
        raise OSError(errno.ENOTSUP,
                      f"{bdf} is not a type-0 PCI endpoint supported by vfio-pci")
    group_path = _iommu_group_path(bdf)
    # Old no-IOMMU VFIO drivers can create synthetic, numerically named groups.
    # Their optional sysfs name identifies them as lacking real isolation.
    try:
        with open(os.path.join(device_path, "iommu_group", "name")) as stream:
            group_name = stream.read().strip()
    except FileNotFoundError:
        group_name = ""
    if group_name == "vfio-noiommu" or os.path.exists(
            os.path.join(VFIO_DEV_PATH, "noiommu-" + os.path.basename(group_path))):
        raise OSError(errno.ENOTSUP, f"{bdf} requires a real IOMMU; no-IOMMU is unsupported")

    if not os.path.isdir(os.path.join(SYSFS_PCI_DRIVERS, "vfio-pci")):
        try:
            subprocess.run(["modprobe", "vfio-pci"], check=True,
                           capture_output=True, text=True)
        except subprocess.CalledProcessError as err:
            raise OSError(errno.ENODEV,
                          f"Cannot load vfio-pci: {(err.stderr or str(err)).strip()}") from err
    # Check container permissions and IOMMU support before releasing a driver.
    fd = os.open(os.path.join(VFIO_DEV_PATH, "vfio"), os.O_RDWR | os.O_CLOEXEC)
    try:
        _select_iommu(fd)
    finally:
        os.close(fd)

    override_path = os.path.join(device_path, "driver_override")
    with open(override_path) as stream:
        original_override = stream.read().rstrip("\n")
    if original_override == "(null)":
        original_override = ""
    if _get_driver(bdf) != original_driver:
        raise OSError(errno.EBUSY, f"{bdf} driver changed while preparing VFIO binding; retry")

    bind_error = rollback_error = override_error = None
    try:
        _write_sysfs(override_path, "vfio-pci")
        if original_driver is not None:
            _write_sysfs(os.path.join(SYSFS_PCI_DRIVERS, original_driver, "unbind"), bdf)
        _write_sysfs(SYSFS_PCI_DRIVERS_PROBE, bdf)
        if _get_driver(bdf) != "vfio-pci":
            raise OSError(errno.ENODEV, f"{bdf} did not bind to vfio-pci after probing")
    except BaseException as err:
        bind_error = err
        try:
            _restore_driver(bdf, original_driver, override_path)
        except Exception as err:
            rollback_error = err
    finally:
        try:
            _write_sysfs(override_path, original_override)
        except OSError as err:
            override_error = err

    if bind_error is not None:
        details = f"Failed to bind {bdf} to vfio-pci: {bind_error}"
        if rollback_error is not None:
            details += f"; driver restoration failed: {rollback_error}"
        if override_error is not None:
            details += f"; driver_override restoration failed: {override_error}"
        if not isinstance(bind_error, Exception):
            if rollback_error is not None or override_error is not None:
                raise OSError(errno.EIO, details) from bind_error
            raise bind_error
        raise OSError(getattr(bind_error, "errno", None) or errno.EIO, details) from bind_error
    if override_error is not None:
        raise OSError(override_error.errno,
                      f"{bdf} is bound to vfio-pci, but its original driver_override "
                      f"could not be restored: {override_error}") from override_error
    return True


class _VfioGroup:
    """One container per group, one device fd per BDF, shared by all regions."""

    _lock = threading.Lock()
    _groups = {}

    def __init__(self, path):
        self.path = path
        self.container_fd = None
        self.group_fd = None
        self.attached = False
        # bdf -> [device fd, live region count, number of device regions]
        self.devices = {}
        try:
            self.container_fd = os.open(os.path.join(VFIO_DEV_PATH, "vfio"),
                                        os.O_RDWR | os.O_CLOEXEC)
            iommu = _select_iommu(self.container_fd)
            try:
                self.group_fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
            except OSError as err:
                raise OSError(err.errno,
                              f"Cannot open VFIO IOMMU group {path}: {err.strerror}; "
                              "check vfio-pci binding and group permissions") from err
            status = VfioGroupStatus(ctypes.sizeof(VfioGroupStatus), 0)
            _ioctl(self.group_fd, VFIO_GROUP_GET_STATUS, status)
            if not status.flags & VFIO_GROUP_FLAGS_VIABLE:
                raise OSError(errno.EBUSY,
                              f"VFIO group {path} is not viable; all devices in the "
                              "IOMMU group must be released from their host drivers")
            if status.flags & VFIO_GROUP_FLAGS_CONTAINER_SET:
                raise OSError(errno.EBUSY, f"VFIO group {path} is already in use")
            _ioctl(self.group_fd, VFIO_GROUP_SET_CONTAINER,
                   ctypes.c_int(self.container_fd))
            self.attached = True
            _ioctl(self.container_fd, VFIO_SET_IOMMU, iommu)
        except Exception:
            self.close()
            raise

    @classmethod
    def acquire(cls, bdf):
        path = _group_path(bdf)
        with cls._lock:
            group = cls._groups.get(path)
            if group is None:
                group = cls(path)
                cls._groups[path] = group
            try:
                if bdf not in group.devices:
                    # A mutable buffer makes Python return the ioctl's integer
                    # result (the fd), rather than a copy of the device name.
                    fd = _ioctl(group.group_fd, VFIO_GROUP_GET_DEVICE_FD,
                                bytearray(bdf.encode("ascii") + b"\0"))
                    try:
                        os.set_inheritable(fd, False)
                        info = VfioDeviceInfo(ctypes.sizeof(VfioDeviceInfo), 0, 0, 0)
                        _ioctl(fd, VFIO_DEVICE_GET_INFO, info)
                        if not info.flags & VFIO_DEVICE_FLAGS_PCI:
                            raise OSError(errno.ENODEV, f"{bdf} is not a VFIO PCI device")
                    except Exception:
                        os.close(fd)
                        raise
                    group.devices[bdf] = [fd, 0, info.num_regions]
                device = group.devices[bdf]
                device[1] += 1
                return group, device[0]
            except Exception:
                if not group.devices:
                    del cls._groups[path]
                    group.close()
                raise

    def release(self, bdf):
        with self._lock:
            device = self.devices[bdf]
            device[1] -= 1
            if device[1] == 0:
                del self.devices[bdf]
                try:
                    os.close(device[0])
                finally:
                    if not self.devices:
                        del self._groups[self.path]
                        self.close()

    def close(self):
        try:
            if self.attached:
                self.attached = False
                _ioctl(self.group_fd, VFIO_GROUP_UNSET_CONTAINER)
        finally:
            try:
                if self.group_fd is not None:
                    fd, self.group_fd = self.group_fd, None
                    os.close(fd)
            finally:
                if self.container_fd is not None:
                    fd, self.container_fd = self.container_fd, None
                    os.close(fd)


def _get_region_info(fd, index):
    length = ctypes.sizeof(VfioRegionInfo)
    for _ in range(4):
        data = bytearray(length)
        struct.pack_into("=IIIIQQ", data, 0, length, 0, index, 0, 0, 0)
        _ioctl(fd, VFIO_DEVICE_GET_REGION_INFO, data)
        info = VfioRegionInfo.from_buffer_copy(data)
        if info.argsz < ctypes.sizeof(VfioRegionInfo) or info.argsz > 1024 * 1024:
            raise OSError(errno.EIO, "Invalid VFIO region info size")
        if info.argsz <= length:
            break
        length = info.argsz
    else:
        raise OSError(errno.EIO, "VFIO region info size kept changing")

    areas = None
    offset = info.cap_offset if info.flags & VFIO_REGION_INFO_FLAG_CAPS else 0
    seen = set()
    while offset:
        if offset in seen or offset < ctypes.sizeof(VfioRegionInfo) or offset + 8 > info.argsz:
            raise OSError(errno.EIO, "Invalid VFIO region capability chain")
        seen.add(offset)
        cap_id, version, next_offset = struct.unpack_from("=HHI", data, offset)
        if cap_id == VFIO_REGION_INFO_CAP_SPARSE_MMAP:
            if version != 1 or areas is not None or offset + 16 > info.argsz:
                raise OSError(errno.ENOTSUP, "Unsupported VFIO sparse mmap capability")
            count, = struct.unpack_from("=I", data, offset + 8)
            if offset + 16 + count * 16 > info.argsz:
                raise OSError(errno.EIO, "Truncated VFIO sparse mmap capability")
            areas = [struct.unpack_from("=QQ", data, offset + 16 + i * 16)
                     for i in range(count)]
        offset = next_offset
    return info, areas


class _VfioRegion:
    def __init__(self, bdf, index):
        self._group = None
        self.fd = None
        self.bdf = bdf.lower()
        self._group, self.fd = _VfioGroup.acquire(self.bdf)
        try:
            if index >= self._group.devices[self.bdf][2]:
                raise OSError(errno.ENODEV, f"{bdf} has no VFIO region {index}")
            self.info, self._areas = _get_region_info(self.fd, index)
            self.size = self.info.size
            if not self.size:
                raise OSError(errno.ENODEV, f"{bdf} VFIO region {index} is empty")
            required = VFIO_REGION_INFO_FLAG_READ | VFIO_REGION_INFO_FLAG_WRITE
            if self.info.flags & required != required:
                raise OSError(errno.EACCES,
                              f"{bdf} VFIO region {index} does not support read/write")
        except Exception:
            self.close()
            raise

    def close(self):
        if self._group is not None:
            group, self._group = self._group, None
            self.fd = None
            group.release(self.bdf)

    def __del__(self):
        self.close()

    def _check_access(self, offset, size):
        if self._group is None:
            raise ValueError("VFIO region is closed")
        offset, size = operator.index(offset), operator.index(size)
        if offset < 0 or size <= 0 or offset + size > self.size:
            raise ValueError(f"Access at {offset:#x} of size {size} exceeds "
                             f"VFIO region size {self.size:#x}")
        return offset, size

    def read8(self, offset):
        return self.read(offset, 1)

    def read16(self, offset):
        return self.read(offset, 2)

    def read32(self, offset):
        return self.read(offset, 4)

    def write8(self, offset, data):
        self.write(offset, data, 1)

    def write16(self, offset, data):
        self.write(offset, data, 2)

    def write32(self, offset, data):
        self.write(offset, data, 4)


class VfioConfig(_VfioRegion):
    """PCI configuration reads/writes mediated by vfio-pci."""

    def __init__(self, bdf):
        super().__init__(bdf, VFIO_PCI_CONFIG_REGION_INDEX)

    def _read_bytes(self, offset, size):
        offset, size = self._check_access(offset, size)
        data = os.pread(self.fd, size, self.info.offset + offset)
        if len(data) != size:
            raise OSError(errno.EIO, f"Short VFIO configuration read for {self.bdf}")
        return data

    def read(self, offset, size):
        return int.from_bytes(self._read_bytes(offset, size), "little")

    def read_format(self, fmt, offset):
        return struct.unpack(fmt, self._read_bytes(offset, struct.calcsize(fmt)))

    def write(self, offset, data, size):
        offset, size = self._check_access(offset, size)
        value = operator.index(data).to_bytes(size, "little")
        if os.pwrite(self.fd, value, self.info.offset + offset) != size:
            raise OSError(errno.EIO, f"Short VFIO configuration write for {self.bdf}")


class VfioBar(_VfioRegion):
    """Map physical PCI BAR index 0..5, respecting VFIO sparse mmap areas."""

    def __init__(self, bdf, bar_num, size=None):
        self._group = None
        self._mappings = []
        bar_num = operator.index(bar_num)
        if not 0 <= bar_num <= 5:
            raise ValueError(f"Invalid PCI BAR index {bar_num}; expected 0..5")
        if size is not None:
            size = operator.index(size)
            if size <= 0:
                raise ValueError("VFIO BAR mapping size must be positive")
        super().__init__(bdf, bar_num)
        try:
            if size is not None:
                if size > self.size:
                    raise ValueError(f"Requested BAR{bar_num} mapping size {size:#x} "
                                     f"exceeds VFIO BAR size {self.size:#x} for {bdf}")
                self.size = size
            if not self.info.flags & VFIO_REGION_INFO_FLAG_MMAP:
                raise OSError(errno.ENOTSUP, f"{bdf} BAR{bar_num} does not support VFIO mmap")
            self._map_areas()
        except Exception:
            self.close()
            raise

    def _map_areas(self):
        page = mmap.PAGESIZE
        if self.info.offset % page:
            raise OSError(errno.ENOTSUP, "VFIO BAR region offset is not page aligned")
        areas = [(0, self.info.size)] if self._areas is None else sorted(self._areas)
        end = 0
        for offset, length in areas:
            if not length:
                continue
            if offset < end or offset + length > self.info.size:
                raise OSError(errno.EIO, "Invalid VFIO sparse mmap area bounds")
            end = offset + length
            if offset % page or (self._areas is not None and length % page):
                raise OSError(errno.ENOTSUP, "VFIO sparse mmap area is not page aligned")
            if offset >= self.size:
                continue
            length = min(length, self.size - offset)
            mapped = mmap.mmap(self.fd, length, flags=mmap.MAP_SHARED,
                               prot=mmap.PROT_READ | mmap.PROT_WRITE,
                               offset=self.info.offset + offset)
            views = {}
            self._mappings.append((offset, offset + length, mapped, views))
            # Typed views retain the MMIO access width, as in FileMap.  Trim
            # their ends so arbitrary positive mapping sizes remain valid.
            for width, fmt in ((1, "B"), (2, "H"), (4, "I")):
                views[width] = memoryview(mapped)[:length // width * width].cast(fmt)
        if not self._mappings:
            raise OSError(errno.ENOTSUP, "Requested VFIO BAR range has no mmap areas")

    def close(self):
        while self._mappings:
            _, _, mapped, views = self._mappings[-1]
            for view in views.values():
                view.release()
            mapped.close()
            self._mappings.pop()
        super().close()

    def _access_view(self, offset, size):
        offset, size = self._check_access(offset, size)
        if size not in (1, 2, 4):
            raise ValueError(f"Unhandled VFIO MMIO access size {size}")
        if offset % size:
            raise ValueError(f"Unaligned {size}-byte VFIO MMIO access at {offset:#x}")
        for start, end, _, views in self._mappings:
            if start <= offset and offset + size <= end:
                return views[size], (offset - start) // size
        raise OSError(errno.ENXIO, f"VFIO BAR offset {offset:#x} of size {size} "
                      "is outside the mmap areas advertised by the driver")

    def read(self, offset, size):
        view, index = self._access_view(offset, size)
        return view[index]

    def write(self, offset, data, size):
        view, index = self._access_view(offset, size)
        view[index] = data
