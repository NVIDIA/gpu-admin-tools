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

import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
from logging import info, warning

try:
    import fcntl
except ImportError:
    fcntl = None


NVIDIA_VENDOR_ID = 0x10DE
VFIO_PCI_DRIVER = "vfio-pci"
STATE_VERSION = 1
DEFAULT_STATE_DIR = "/run/nvidia-gpu-tools"

_BDF_RE = re.compile(
    r"^(?:(?P<domain>[0-9a-fA-F]{4}):)?"
    r"(?P<bus>[0-9a-fA-F]{2}):(?P<device>[0-9a-fA-F]{2})\."
    r"(?P<function>[0-7])$"
)

_PCI_CLASS_NAMES = {
    0x01: "storage",
    0x02: "network",
    0x03: "display",
    0x04: "multimedia",
    0x05: "memory",
    0x06: "bridge",
    0x07: "communication",
    0x08: "system peripheral",
    0x0C: "serial bus",
    0x12: "processing accelerator",
}

_HIGH_IMPACT_CLASSES = {
    0x01: "storage",
    0x02: "network",
    0x08: "system or management",
}


class VfioError(RuntimeError):
    pass


def _bdf_sort_key(bdf):
    match = _BDF_RE.match(bdf)
    if not match:
        return (bdf,)
    return tuple(
        int(match.group(name) or "0", 16)
        for name in ("domain", "bus", "device", "function")
    )


def _ioctl_io(type_char, number):
    return (ord(type_char) << 8) | number


VFIO_GROUP_GET_STATUS = _ioctl_io(";", 100 + 3)
VFIO_GROUP_FLAGS_VIABLE = 1 << 0


class PciDeviceInfo:
    def __init__(self, bdf, vendor, device, class_code, driver, driver_override=None):
        self.bdf = bdf
        self.vendor = vendor
        self.device = device
        self.class_code = class_code
        self.driver = driver
        self.driver_override = driver_override

    @property
    def base_class(self):
        return (self.class_code >> 16) & 0xFF

    @property
    def class_name(self):
        return _PCI_CLASS_NAMES.get(self.base_class, "class 0x%02x" % self.base_class)

    @property
    def is_bridge(self):
        return self.base_class == 0x06

    @property
    def is_gpu(self):
        return self.vendor == NVIDIA_VENDOR_ID and self.base_class == 0x03

    @property
    def high_impact_name(self):
        return _HIGH_IMPACT_CLASSES.get(self.base_class)


class LinuxVfioAccess:
    """Small boundary around the Linux sysfs and VFIO interfaces."""

    def __init__(self, sysfs_root="/sys", dev_root="/dev", command_runner=None):
        self.sysfs_root = Path(sysfs_root)
        self.dev_root = Path(dev_root)
        self.command_runner = command_runner or subprocess.run

    @property
    def pci_devices_path(self):
        return self.sysfs_root / "bus" / "pci" / "devices"

    def device_path(self, bdf):
        return self.pci_devices_path / bdf

    def list_pci_bdfs(self):
        if not self.pci_devices_path.is_dir():
            raise VfioError("PCI sysfs is not available at %s" % self.pci_devices_path)
        return sorted(
            (entry.name for entry in self.pci_devices_path.iterdir() if _BDF_RE.match(entry.name)),
            key=_bdf_sort_key,
        )

    def device_exists(self, bdf):
        return self.device_path(bdf).exists()

    def read_hex(self, bdf, attribute):
        path = self.device_path(bdf) / attribute
        try:
            return int(path.read_text().strip(), 0)
        except (OSError, ValueError) as err:
            raise VfioError("Cannot read %s for %s: %s" % (attribute, bdf, err))

    def current_driver(self, bdf):
        path = self.device_path(bdf) / "driver"
        if not os.path.lexists(str(path)):
            return None
        try:
            return os.path.basename(os.path.realpath(str(path)))
        except OSError as err:
            raise VfioError("Cannot read driver for %s: %s" % (bdf, err))

    def driver_override(self, bdf):
        path = self.device_path(bdf) / "driver_override"
        try:
            value = path.read_text().strip()
        except OSError as err:
            raise VfioError("Cannot read driver_override for %s: %s" % (bdf, err))
        if not value or value == "(null)":
            return None
        return value

    def iommu_group(self, bdf):
        path = self.device_path(bdf) / "iommu_group"
        if not os.path.lexists(str(path)):
            raise VfioError("PCI device %s has no IOMMU group" % bdf)
        try:
            return os.path.basename(os.path.realpath(str(path)))
        except OSError as err:
            raise VfioError("Cannot resolve IOMMU group for %s: %s" % (bdf, err))

    def group_members(self, group):
        path = self.sysfs_root / "kernel" / "iommu_groups" / str(group) / "devices"
        if not path.is_dir():
            raise VfioError("IOMMU group %s has no devices directory" % group)
        return sorted((entry.name for entry in path.iterdir()), key=_bdf_sort_key)

    def set_driver_override(self, bdf, driver):
        path = self.device_path(bdf) / "driver_override"
        try:
            path.write_text((driver or "") + "\n")
        except OSError as err:
            raise VfioError("Cannot set driver_override for %s: %s" % (bdf, err))

    def unbind(self, bdf):
        driver = self.current_driver(bdf)
        if driver is None:
            return
        path = self.device_path(bdf) / "driver" / "unbind"
        try:
            path.write_text(bdf + "\n")
        except OSError as err:
            raise VfioError("Cannot unbind %s from %s: %s" % (bdf, driver, err))

    def probe(self, bdf):
        path = self.sysfs_root / "bus" / "pci" / "drivers_probe"
        try:
            path.write_text(bdf + "\n")
        except OSError as err:
            raise VfioError("Cannot probe a driver for %s: %s" % (bdf, err))

    def load_module(self, module):
        try:
            self.command_runner(
                ["modprobe", module],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
        except (OSError, subprocess.CalledProcessError) as err:
            detail = getattr(err, "stderr", None) or str(err)
            raise VfioError("Cannot load kernel module %s: %s" % (module, detail.strip()))

    def group_viable(self, group):
        if fcntl is None:
            return False, "VFIO group status is unavailable on this platform"
        path = self.dev_root / "vfio" / str(group)
        if not path.exists():
            return False, "%s does not exist" % path

        status = bytearray(struct.pack("=II", 8, 0))
        try:
            fd = os.open(str(path), os.O_RDWR)
            try:
                fcntl.ioctl(fd, VFIO_GROUP_GET_STATUS, status, True)
            finally:
                os.close(fd)
        except OSError as err:
            return False, "VFIO_GROUP_GET_STATUS failed: %s" % err

        _, flags = struct.unpack("=II", status)
        if flags & VFIO_GROUP_FLAGS_VIABLE:
            return True, "VFIO group reports viable"
        return False, "VFIO group reports not viable"


class VfioManager:
    def __init__(self, access=None, state_dir=DEFAULT_STATE_DIR):
        self.access = access or LinuxVfioAccess()
        self.state_dir = Path(state_dir)

    def device_info(self, bdf):
        return PciDeviceInfo(
            bdf=bdf,
            vendor=self.access.read_hex(bdf, "vendor"),
            device=self.access.read_hex(bdf, "device"),
            class_code=self.access.read_hex(bdf, "class"),
            driver=self.access.current_driver(bdf),
            driver_override=self.access.driver_override(bdf),
        )

    def gpu_bdfs(self):
        gpus = []
        for bdf in self.access.list_pci_bdfs():
            device = self.device_info(bdf)
            if device.is_gpu:
                gpus.append(bdf)
        return gpus

    def resolve_gpu(self, gpu=-1, gpu_bdf=None, devices=None, gpu_name=None):
        selectors = int(gpu is not None and gpu >= 0) + int(bool(gpu_bdf)) + int(bool(devices))
        if gpu_name:
            raise VfioError("VFIO actions do not support --gpu-name; use --gpu, --gpu-bdf, or --devices")
        if selectors != 1:
            raise VfioError("Select exactly one GPU with --gpu, --gpu-bdf, or --devices")

        gpus = self.gpu_bdfs()
        if gpu is not None and gpu >= 0:
            if gpu >= len(gpus):
                raise VfioError("GPU index %d is out of range; found %d GPU(s)" % (gpu, len(gpus)))
            return gpus[gpu]

        if gpu_bdf:
            return self._match_one_gpu(gpu_bdf, gpus)

        return self._resolve_devices_selector(devices, gpus)

    def _match_one_gpu(self, pattern, gpus):
        pattern = pattern.lower()
        matches = [bdf for bdf in gpus if pattern in bdf.lower()]
        if not matches:
            raise VfioError("No NVIDIA GPU matches %s" % pattern)
        if len(matches) > 1:
            raise VfioError("GPU selector %s matches more than one GPU: %s" % (pattern, ", ".join(matches)))
        return matches[0]

    def _resolve_devices_selector(self, selector, gpus):
        if "," in selector:
            raise VfioError("VFIO actions require exactly one GPU selector")

        match = re.match(r"^gpus(?:\[(.*)\])?$", selector)
        if match:
            index = match.group(1)
            if index is None:
                selected = gpus
            elif ":" in index:
                parts = index.split(":")
                if len(parts) > 3:
                    raise VfioError("Invalid GPU slice: %s" % selector)
                values = [int(part) if part else None for part in parts]
                values += [None] * (3 - len(values))
                selected = gpus[slice(*values)]
            else:
                try:
                    selected = [gpus[int(index)]]
                except (ValueError, IndexError):
                    raise VfioError("Invalid GPU index: %s" % selector)

            if len(selected) != 1:
                raise VfioError("VFIO actions require exactly one GPU; %s selected %d" % (selector, len(selected)))
            return selected[0]

        return self._match_one_gpu(selector, gpus)

    def inspect_group(self, target_bdf):
        if not self.access.device_exists(target_bdf):
            raise VfioError("PCI device %s does not exist" % target_bdf)
        target = self.device_info(target_bdf)
        if not target.is_gpu:
            raise VfioError("PCI device %s is not an NVIDIA GPU" % target_bdf)

        group = self.access.iommu_group(target_bdf)
        members = [self.device_info(bdf) for bdf in self.access.group_members(group)]
        if target_bdf not in [member.bdf for member in members]:
            raise VfioError("GPU %s is not listed in IOMMU group %s" % (target_bdf, group))
        return group, members

    def state_path(self, target_bdf, override=None):
        if override:
            return Path(override)
        safe_bdf = target_bdf.replace(":", "_").replace(".", "_")
        return self.state_dir / ("vfio-%s.json" % safe_bdf)

    def query(self, target_bdf, state_path=None):
        group, members = self.inspect_group(target_bdf)
        lines = ["IOMMU group %s for GPU %s:" % (group, target_bdf)]
        for member in members:
            role = "target GPU" if member.bdf == target_bdf else ("bridge" if member.is_bridge else "peer endpoint")
            lines.append(
                "  %s  %s  vendor=%04x device=%04x class=%06x driver=%s"
                % (
                    member.bdf,
                    role,
                    member.vendor,
                    member.device,
                    member.class_code,
                    member.driver or "unbound",
                )
            )

        blockers = self._driver_blockers(members)
        if blockers:
            lines.append("Driver readiness: blocked by %s" % ", ".join(
                "%s (%s)" % (member.bdf, member.driver) for member in blockers
            ))
        else:
            viable, detail = self.access.group_viable(group)
            lines.append("VFIO viability: %s (%s)" % ("ready" if viable else "not ready", detail))

        state_path = self.state_path(target_bdf, state_path)
        lines.append("Restore state: %s" % (state_path if state_path.exists() else "not recorded"))
        return "\n".join(lines)

    def bind(self, target_bdf, allowed_members=None, state_path=None, dry_run=False):
        allowed_members = allowed_members or []
        group, members = self.inspect_group(target_bdf)
        members_by_bdf = {member.bdf: member for member in members}
        allowed = self._resolve_allowed_members(allowed_members, members)

        unauthorized = []
        bridge_blockers = []
        for member in members:
            if member.bdf == target_bdf or member.driver in (None, VFIO_PCI_DRIVER):
                continue
            if member.is_bridge:
                bridge_blockers.append(member)
            elif member.bdf not in allowed:
                unauthorized.append(member)

        if bridge_blockers:
            raise VfioError(
                "IOMMU group %s contains bridge device(s) bound to host drivers; "
                "this tool will not detach bridges: %s"
                % (group, ", ".join("%s (%s)" % (m.bdf, m.driver) for m in bridge_blockers))
            )
        if unauthorized:
            detail = ", ".join(
                "%s (%s, driver=%s)" % (member.bdf, member.class_name, member.driver)
                for member in unauthorized
            )
            raise VfioError(
                "IOMMU group %s has additional host-bound endpoint(s): %s. "
                "Authorize each device with --allow-group-member BDF"
                % (group, detail)
            )

        selected_bdfs = []
        for bdf in [target_bdf] + sorted(allowed, key=_bdf_sort_key):
            if bdf not in selected_bdfs:
                selected_bdfs.append(bdf)
        for bdf in selected_bdfs:
            member = members_by_bdf[bdf]
            if member.is_bridge:
                raise VfioError("Cannot bind bridge %s to vfio-pci" % bdf)
            if member.high_impact_name:
                warning(
                    "%s is a %s device; binding it to vfio-pci can disrupt host services",
                    bdf,
                    member.high_impact_name,
                )

        state_path = self.state_path(target_bdf, state_path)
        state = self._load_state(state_path, required=False)
        if state:
            self._validate_state(
                state,
                target_bdf,
                group,
                [member.bdf for member in members],
            )
        else:
            state = {
                "version": STATE_VERSION,
                "target_bdf": target_bdf,
                "iommu_group": str(group),
                "devices": {},
            }

        to_bind = []
        for bdf in selected_bdfs:
            member = members_by_bdf[bdf]
            if member.driver == VFIO_PCI_DRIVER:
                info("%s is already bound to vfio-pci", bdf)
                continue
            if bdf not in state["devices"]:
                state["devices"][bdf] = {
                    "driver": member.driver,
                    "driver_override": member.driver_override,
                }
            to_bind.append(bdf)

        if dry_run:
            if not to_bind:
                info("Dry run: all selected devices are already bound to vfio-pci")
            for bdf in to_bind:
                info("Dry run: would bind %s to vfio-pci", bdf)
            return

        if os.geteuid() != 0:
            raise VfioError("Binding devices to vfio-pci requires root privileges")

        self.access.load_module("vfio")
        self.access.load_module(VFIO_PCI_DRIVER)
        for bdf in to_bind:
            expected = state["devices"][bdf]["driver"]
            current = self.access.current_driver(bdf)
            if current != expected:
                raise VfioError(
                    "%s driver changed during VFIO preflight: expected %s, found %s"
                    % (bdf, expected or "unbound", current or "unbound")
                )
        if to_bind:
            self._write_state(state_path, state)

        attempted = []
        try:
            for bdf in to_bind:
                attempted.append(bdf)
                self._bind_one(bdf)

            post_group, post_members = self.inspect_group(target_bdf)
            blockers = self._driver_blockers(post_members)
            if blockers:
                raise VfioError("IOMMU group %s still has host-bound devices: %s" % (
                    post_group,
                    ", ".join("%s (%s)" % (m.bdf, m.driver) for m in blockers),
                ))
            viable, detail = self.access.group_viable(post_group)
            if not viable:
                raise VfioError("IOMMU group %s is not viable: %s" % (post_group, detail))
        except Exception as err:
            rollback_errors = self._rollback(attempted, state)
            if not rollback_errors and self._state_is_restored(state):
                self._remove_state(state_path)
            detail = str(err)
            if rollback_errors:
                detail += "; rollback errors: " + "; ".join(rollback_errors)
            raise VfioError(detail)

        info("IOMMU group %s is viable for VFIO", group)
        if to_bind:
            info("Original driver state recorded in %s", state_path)

    def restore(self, target_bdf, state_path=None, dry_run=False):
        state_path = self.state_path(target_bdf, state_path)
        state = self._load_state(state_path, required=False)
        if not state:
            current = self.device_info(target_bdf).driver
            if current == VFIO_PCI_DRIVER:
                raise VfioError(
                    "%s is bound to vfio-pci but no restore state exists at %s"
                    % (target_bdf, state_path)
                )
            info("No restore state at %s; nothing to do", state_path)
            return

        group = self.access.iommu_group(target_bdf)
        self._validate_state(
            state,
            target_bdf,
            group,
            self.access.group_members(group),
        )

        if not dry_run and os.geteuid() != 0:
            raise VfioError("Restoring host drivers requires root privileges")

        failures = []
        for bdf in reversed(list(state["devices"])):
            original = state["devices"][bdf]
            if dry_run:
                info("Dry run: would restore %s to %s", bdf, original["driver"] or "unbound")
                continue
            try:
                self._restore_one(bdf, original)
            except VfioError as err:
                failures.append(str(err))

        if failures:
            raise VfioError("Restore incomplete; state retained at %s: %s" % (state_path, "; ".join(failures)))
        if not dry_run:
            self._remove_state(state_path)
            info("Original host driver state restored; removed %s", state_path)

    def _resolve_allowed_members(self, patterns, members):
        allowed = set()
        bdfs = [member.bdf for member in members]
        for pattern in patterns:
            pattern = pattern.lower()
            matches = [bdf for bdf in bdfs if pattern in bdf.lower()]
            if not matches:
                raise VfioError("Allowed group member %s is not in the target IOMMU group" % pattern)
            if len(matches) > 1:
                raise VfioError("Allowed group member %s is ambiguous: %s" % (pattern, ", ".join(matches)))
            allowed.add(matches[0])
        return allowed

    @staticmethod
    def _driver_blockers(members):
        return [member for member in members if member.driver not in (None, VFIO_PCI_DRIVER)]

    def _bind_one(self, bdf):
        current = self.access.current_driver(bdf)
        if current == VFIO_PCI_DRIVER:
            return
        self.access.set_driver_override(bdf, VFIO_PCI_DRIVER)
        if current is not None:
            self.access.unbind(bdf)
        self.access.probe(bdf)
        actual = self.access.current_driver(bdf)
        if actual != VFIO_PCI_DRIVER:
            raise VfioError("%s driver is %s after probe; expected vfio-pci" % (bdf, actual or "unbound"))
        info("Bound %s to vfio-pci", bdf)

    def _restore_one(self, bdf, original):
        original_driver = original.get("driver")
        original_override = original.get("driver_override")
        current = self.access.current_driver(bdf)

        if current != original_driver:
            if original_driver is not None:
                self.access.set_driver_override(bdf, original_driver)
            if current is not None:
                self.access.unbind(bdf)
            if original_driver is not None:
                self.access.probe(bdf)

        self.access.set_driver_override(bdf, original_override)
        actual = self.access.current_driver(bdf)
        if actual != original_driver:
            raise VfioError(
                "%s driver is %s after restore; expected %s"
                % (bdf, actual or "unbound", original_driver or "unbound")
            )
        info("Restored %s to %s", bdf, original_driver or "unbound")

    def _rollback(self, attempted, state):
        errors = []
        for bdf in reversed(attempted):
            try:
                self._restore_one(bdf, state["devices"][bdf])
            except Exception as err:
                errors.append("%s: %s" % (bdf, err))
        return errors

    def _state_is_restored(self, state):
        for bdf, original in state["devices"].items():
            if self.access.current_driver(bdf) != original.get("driver"):
                return False
        return True

    @staticmethod
    def _validate_state(state, target_bdf, group, group_members):
        if state.get("version") != STATE_VERSION:
            raise VfioError("Unsupported VFIO state version in restore file")
        if state.get("target_bdf") != target_bdf:
            raise VfioError("Restore state belongs to GPU %s, not %s" % (state.get("target_bdf"), target_bdf))
        if str(state.get("iommu_group")) != str(group):
            raise VfioError("Restore state records IOMMU group %s, current group is %s" % (
                state.get("iommu_group"), group
            ))
        if not isinstance(state.get("devices"), dict):
            raise VfioError("Restore state has no device map")
        for bdf, original in state["devices"].items():
            if not _BDF_RE.match(bdf) or not isinstance(original, dict):
                raise VfioError("Restore state contains an invalid device entry")
            if bdf not in group_members:
                raise VfioError(
                    "Restore state contains %s, which is not in IOMMU group %s"
                    % (bdf, group)
                )
            for key in ("driver", "driver_override"):
                value = original.get(key)
                if value is not None and not isinstance(value, str):
                    raise VfioError("Restore state contains an invalid %s for %s" % (key, bdf))

    @staticmethod
    def _load_state(path, required):
        if not path.exists():
            if required:
                raise VfioError("Restore state does not exist: %s" % path)
            return None
        try:
            with path.open("r") as stream:
                return json.load(stream)
        except (OSError, ValueError) as err:
            raise VfioError("Cannot read restore state %s: %s" % (path, err))

    @staticmethod
    def _write_state(path, state):
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "w") as stream:
                    json.dump(state, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, str(path))
            except Exception:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
        except OSError as err:
            raise VfioError("Cannot write restore state %s: %s" % (path, err))

    @staticmethod
    def _remove_state(path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as err:
            raise VfioError("Cannot remove restore state %s: %s" % (path, err))


def vfio_action_requested(opts):
    return any(
        getattr(opts, name, False)
        for name in ("query_vfio_state", "bind_vfio", "restore_vfio_drivers")
    )


def vfio_options_requested(opts):
    return vfio_action_requested(opts) or any(
        (
            getattr(opts, "allow_group_member", []),
            getattr(opts, "vfio_state_file", None),
            getattr(opts, "dry_run", False),
        )
    )


def _active_non_vfio_options(opts):
    allowed = {
        "allow_group_member",
        "bind_vfio",
        "devices",
        "dry_run",
        "gpu",
        "gpu_bdf",
        "gpu_name",
        "log",
        "mmio_access_type",
        "query_vfio_state",
        "restore_vfio_drivers",
        "vfio_state_file",
    }
    active = []
    for name, value in vars(opts).items():
        if name in allowed or value in (None, False, -1, [], ()):
            continue
        active.append("--" + name.replace("_", "-"))
    return active


def execute_vfio_action(opts, manager=None):
    if not vfio_action_requested(opts):
        raise VfioError("A VFIO action is required with VFIO-specific options")
    incompatible = _active_non_vfio_options(opts)
    if incompatible:
        raise VfioError("VFIO actions cannot be combined with %s" % ", ".join(incompatible))

    manager = manager or VfioManager()
    target_bdf = manager.resolve_gpu(
        gpu=opts.gpu,
        gpu_bdf=opts.gpu_bdf,
        devices=getattr(opts, "devices", None),
        gpu_name=opts.gpu_name,
    )
    state_path = getattr(opts, "vfio_state_file", None)

    if opts.query_vfio_state:
        if opts.allow_group_member:
            raise VfioError("--allow-group-member is only valid with --bind-vfio")
        if opts.dry_run:
            raise VfioError("--dry-run is only valid with --bind-vfio or --restore-vfio-drivers")
        print(manager.query(target_bdf, state_path=state_path))
    elif opts.bind_vfio:
        manager.bind(
            target_bdf,
            allowed_members=opts.allow_group_member,
            state_path=state_path,
            dry_run=opts.dry_run,
        )
        print(manager.query(target_bdf, state_path=state_path))
    else:
        if opts.allow_group_member:
            raise VfioError("--allow-group-member is only valid with --bind-vfio")
        manager.restore(target_bdf, state_path=state_path, dry_run=opts.dry_run)
        print(manager.query(target_bdf, state_path=state_path))
