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

from logging import warning
from pathlib import Path
import re

from gpu.properties import GpuProperties


_BDF_RE = re.compile(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]")


def _bdf_key(device):
    return int(device.bdf.replace(":", "").replace(".", ""), 16)


class DeviceInfo:
    """PCI identity only: constructing this object never accesses hardware."""

    def __init__(self, bdf, vendor, device, class_id, svid=None, ssid=None,
                 dev_path=None, description=None):
        self.bdf = bdf.lower()
        self.vendor = vendor
        self.device = device
        self.class_id = class_id
        self.svid = svid
        self.ssid = ssid
        self.dev_path = dev_path
        self.description = description
        metadata = {}
        if self.is_gpu():
            metadata = GpuProperties(None, device, ssid).get_metadata()
        elif self.is_nvswitch():
            metadata = GpuProperties.get_nvswitch_metadata(device)
        self.name = metadata.get("name")
        self.arch = metadata.get("arch")
        self.chip = metadata.get("chip")

    @property
    def devid(self):
        return self.device

    def is_gpu(self):
        return self.vendor == 0x10DE and self.class_id in (0x030000, 0x030200)

    def is_nvswitch(self):
        return self.vendor == 0x10DE and self.class_id == 0x068000

    def __str__(self):
        kind = "GPU" if self.is_gpu() else "NvSwitch" if self.is_nvswitch() else "PCI device"
        name = self.name or self.description or "unknown"
        return (f"{kind} {self.bdf} {name} {self.vendor:04x}:{self.device:04x} "
                f"arch={self.arch or 'unknown'} chip={self.chip or 'unknown'}")


def _bdf_selector_patterns(devices, gpu_bdf):
    """Return BDF-only selectors that can safely narrow sysfs discovery."""
    if gpu_bdf is not None and devices is None:
        patterns = [gpu_bdf.strip().lower()]
    elif devices is not None and gpu_bdf is None:
        patterns = [item.strip().lower() for item in devices.split(",")]
        if any(":" not in item or re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{4}", item)
               for item in patterns):
            return None
    else:
        return None
    if not all(re.fullmatch(r"[0-9a-f:.]+", item) for item in patterns):
        return None
    return list(dict.fromkeys(patterns))


def _bdf_matches(bdf, pattern):
    return bdf == pattern if _BDF_RE.fullmatch(pattern) else pattern in bdf


def discover_sysfs_devices(sysfs_root="/sys", devices=None, gpu_bdf=None):
    """Read selected PCI identities, using direct paths for complete BDFs."""
    device_root = Path(sysfs_root) / "bus" / "pci" / "devices"
    patterns = _bdf_selector_patterns(devices, gpu_bdf)
    if patterns and all(_BDF_RE.fullmatch(pattern) for pattern in patterns):
        paths = [device_root / pattern for pattern in patterns]
    else:
        try:
            paths = list(device_root.iterdir())
        except OSError as err:
            raise ValueError(f"Cannot enumerate PCI devices at {device_root}: {err}") from err
    inventory = []
    for path in paths:
        if not _BDF_RE.fullmatch(path.name):
            continue
        if patterns and not any(_bdf_matches(path.name.lower(), pattern) for pattern in patterns):
            continue
        try:
            values = {
                attribute: int((path / attribute).read_text().strip(), 0)
                for attribute in ("vendor", "device", "class")
            }
            subsystem = {}
            for field, attribute in (("svid", "subsystem_vendor"), ("ssid", "subsystem_device")):
                try:
                    subsystem[field] = int((path / attribute).read_text().strip(), 0)
                except FileNotFoundError:
                    subsystem[field] = None
        except (OSError, ValueError) as err:
            warning("Cannot read PCI identity for %s: %s", path.name, err)
            continue
        inventory.append(DeviceInfo(path.name, values["vendor"], values["device"],
                                    values["class"], dev_path=str(path), **subsystem))
    return sorted(inventory, key=_bdf_key)


def _index_devices(matches, index, selector):
    try:
        if ":" in index:
            parts = index.split(":")
            if len(parts) > 3:
                raise ValueError("a slice has at most three components")
            values = [int(part) if part else None for part in parts]
            return matches[slice(*values)]
        value = int(index)
        if not -len(matches) <= value < len(matches):
            raise ValueError(f"index {value} out of range for {len(matches)} device(s)")
        return [matches[value]]
    except ValueError as err:
        raise ValueError(f"Invalid device index or slice {selector}: {err}") from err


def _match_bdf(inventory, pattern):
    pattern = pattern.strip().lower()
    if not pattern or not re.fullmatch(r"[0-9a-fA-F:.]+", pattern):
        raise ValueError(f"Invalid BDF selector: {pattern}")
    matches = [device for device in inventory if _bdf_matches(device.bdf, pattern)]
    if len(matches) != 1:
        detail = "nothing" if not matches else "more than one device: " + ", ".join(d.bdf for d in matches)
        raise ValueError(f"BDF selector {pattern} matched {detail}")
    return matches


def _select_device_string(inventory, selector):
    selected = []
    for item in selector.split(","):
        item = item.strip()
        match = re.fullmatch(r"([^\[\]]+)(?:\[([^\[\]]+)\])?", item)
        if match is None:
            raise ValueError(f"Invalid device selector: {item}")
        base, index = match.groups()
        if base == "gpus":
            matches = [device for device in inventory if device.is_gpu()]
        elif base == "nvswitches":
            matches = [device for device in inventory if device.is_nvswitch()]
        elif re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{4}", base):
            vendor, devid = [int(value, 16) for value in base.split(":")]
            matches = [device for device in inventory
                       if device.vendor == vendor and device.device == devid]
        elif ":" in base:
            matches = _match_bdf(inventory, base)
        else:
            raise ValueError(f"Unknown device selector: {base}")
        if index is not None:
            matches = _index_devices(matches, index, item)
        for device in matches:
            if device.bdf not in {existing.bdf for existing in selected}:
                selected.append(device)
    if not selected:
        raise ValueError(f"No devices found matching: {selector}")
    return selected


def select_devices(inventory, devices=None, gpu=-1, gpu_bdf=None, gpu_name=None, no_gpu=False):
    """Apply a selector in discovery order without initializing any device."""
    selectors = [devices is not None, gpu not in (None, -1), gpu_bdf is not None,
                 gpu_name is not None, bool(no_gpu)]
    if sum(selectors) > 1:
        raise ValueError("Select devices with exactly one of --devices, --gpu, --gpu-bdf, "
                         "--gpu-name, or --no-gpu")
    if no_gpu or not any(selectors):
        return []
    inventory = list(inventory)
    nvidia = [device for device in inventory if device.is_gpu() or device.is_nvswitch()]
    if devices is not None:
        return _select_device_string(inventory, devices)
    if gpu not in (None, -1):
        if gpu < 0 or gpu >= len(nvidia):
            raise ValueError(f"GPU index {gpu} out of range for {len(nvidia)} NVIDIA device(s)")
        return [nvidia[gpu]]
    if gpu_bdf is not None:
        return _match_bdf(nvidia, gpu_bdf)
    if not gpu_name:
        raise ValueError("GPU name selector cannot be empty")
    for device in nvidia:
        if any(gpu_name in name for name in (device.name, device.description) if name):
            return [device]
    raise ValueError(f"No GPU found matching name: {gpu_name}")
