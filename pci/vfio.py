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

import copy
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import textwrap
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


class PciDeviceInfo:
    def __init__(self, bdf, vendor, device, class_code, driver, driver_override=None,
                 driver_module=None, header_type=0):
        self.bdf = bdf
        self.vendor = vendor
        self.device = device
        self.class_code = class_code
        self.driver = driver
        self.driver_override = driver_override
        self.driver_module = driver_module
        self.header_type = header_type

    @property
    def base_class(self):
        return (self.class_code >> 16) & 0xFF

    @property
    def class_name(self):
        if self.vendor == NVIDIA_VENDOR_ID and self.class_code == 0x068000:
            return "NVSwitch"
        return _PCI_CLASS_NAMES.get(self.base_class, "class 0x%02x" % self.base_class)

    @property
    def is_bridge(self):
        return self.header_type in (1, 2)

    @property
    def is_gpu(self):
        return self.vendor == NVIDIA_VENDOR_ID and self.class_code in (0x030000, 0x030200)

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

    def driver_registered(self, driver):
        return (self.sysfs_root / "bus" / "pci" / "drivers" / driver).is_dir()

    def header_type(self, bdf):
        try:
            with (self.device_path(bdf) / "config").open("rb") as stream:
                stream.seek(0x0E)
                value = stream.read(1)
            if len(value) != 1:
                raise ValueError("short PCI configuration read")
            return value[0] & 0x7F
        except (OSError, ValueError) as err:
            raise VfioError("Cannot read PCI header type for %s: %s" % (bdf, err))

    def driver_module(self, bdf):
        path = self.device_path(bdf) / "driver" / "module"
        return path.resolve().name if path.exists() else None

    def iommu_group(self, bdf):
        path = self.device_path(bdf) / "iommu_group"
        if not os.path.lexists(str(path)):
            raise VfioError("PCI device %s has no IOMMU group" % bdf)
        try:
            group = os.path.basename(os.path.realpath(str(path)))
            if not re.fullmatch(r"[0-9]+", group):
                raise VfioError("PCI device %s requires a real IOMMU group" % bdf)
            try:
                name = (path / "name").read_text().strip()
            except FileNotFoundError:
                name = ""
            if name in ("vfio-noiommu", "noiommu") or (
                    self.dev_root / "vfio" / ("noiommu-" + group)).exists():
                raise VfioError("PCI device %s requires a real IOMMU; no-IOMMU is unsupported" % bdf)
            return group
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
        from utils.vfio_abi import VFIO_GROUP_FLAGS_VIABLE, VFIO_GROUP_GET_STATUS

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
            driver_module=self.access.driver_module(bdf),
            header_type=self.access.header_type(bdf),
        )

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
        target = next(member for member in members if member.bdf == target_bdf)
        state_path = self.state_path(target_bdf, state_path)
        state = self._load_state(state_path, required=False)
        has_state = state is not None
        if has_state:
            self._validate_state(state, target_bdf, group, [m.bdf for m in members])
        peer_blockers = [
            member
            for member in self._driver_blockers(members)
            if member.bdf != target_bdf
        ]

        status = "READY TO BIND"
        action = "Run vfio bind for this GPU; use --dry-run first to preview the change."
        viability = None
        if has_state and target.driver != VFIO_PCI_DRIVER:
            status = "RESTORE REQUIRED"
            action = "Run vfio restore to finish returning the recorded devices to the host."
        elif target.driver == VFIO_PCI_DRIVER and not has_state:
            status = "EXTERNALLY MANAGED"
            action = (
                "No restore record exists. Use the tool that performed the bind "
                "to return this GPU to its host driver."
            )
        elif peer_blockers:
            status = "NOT READY" if target.driver == VFIO_PCI_DRIVER else "REVIEW REQUIRED"
            flags = " ".join(
                "--allow-group-member %s" % member.bdf
                for member in peer_blockers
            )
            action = "Review the host impact, then add %s to vfio bind." % flags
        elif target.driver == VFIO_PCI_DRIVER:
            viability = self.access.group_viable(group)
            if viability[0]:
                status = "READY FOR QEMU"
                action = "Launch the guest; after it stops, run vfio restore."
            else:
                status = "NOT READY"
                action = "Inspect /dev/vfio/%s and the kernel log before launching QEMU." % group

        lines = [
            "VFIO status for NVIDIA GPU %s" % target_bdf,
            "Status: %s" % status,
            "IOMMU group: %s" % group,
            "",
            "Devices in this group:",
        ]
        for member in members:
            if member.bdf == target_bdf:
                role = "selected NVIDIA GPU"
            elif member.is_bridge:
                role = "PCI bridge (not assigned)"
            else:
                role = "%s peer" % member.class_name
            lines.append(
                "  %-12s  %-27s  driver=%s"
                % (
                    member.bdf,
                    role,
                    member.driver or "unbound",
                )
            )

        if viability is not None:
            lines.extend(["", "VFIO check: %s" % viability[1]])
        lines.extend([
            "",
            "Restore record: %s" % (state_path if has_state else "none"),
            textwrap.fill(
                action,
                width=88,
                initial_indent="Next action: ",
                subsequent_indent="             ",
            ),
        ])
        return "\n".join(lines)

    def query_many(self, target_bdfs):
        return "\n\n".join(self.query(target_bdf) for target_bdf in target_bdfs)

    def bind_many(self, target_bdfs, allowed_members=None, dry_run=False):
        contexts = self._batch_contexts(target_bdfs, allowed_members or [])
        if len(contexts) == 1:
            context = contexts[0]
            return self.bind(
                context["target"],
                allowed_members=context["allowed"],
                dry_run=dry_run,
            )

        preflight_errors = []
        for context in contexts:
            try:
                self.bind(
                    context["target"],
                    allowed_members=context["allowed"],
                    dry_run=True,
                    report=dry_run,
                )
            except VfioError as err:
                preflight_errors.append("%s: %s" % (context["target"], err))
        if preflight_errors:
            raise VfioError("Multi-GPU preflight failed: " + "; ".join(preflight_errors))
        if dry_run:
            return

        attempted = []
        try:
            for context in contexts:
                target_bdf = context["target"]
                path = self.state_path(target_bdf)
                previous_state = self._load_state(path, required=False)
                snapshot = {
                    bdf: self._device_state(self.device_info(bdf))
                    for bdf in [target_bdf] + context["allowed"]
                }
                self.bind(target_bdf, allowed_members=context["allowed"])
                attempted.append((path, previous_state, snapshot))
        except BaseException as err:
            rollback_errors = []
            for path, previous_state, snapshot in reversed(attempted):
                errors = self._rollback(list(snapshot), {"devices": snapshot})
                if not errors:
                    try:
                        self._restore_record(path, previous_state)
                    except VfioError as restore_error:
                        errors.append(str(restore_error))
                rollback_errors.extend(errors)
            detail = str(err)
            if rollback_errors:
                detail += "; multi-GPU rollback errors: " + "; ".join(rollback_errors)
            if not isinstance(err, Exception):
                if rollback_errors:
                    warning("Interrupted VFIO operation: %s", detail)
                raise
            raise VfioError(detail) from err

    def restore_many(self, target_bdfs, dry_run=False):
        contexts = self._batch_contexts(target_bdfs, [])
        if len(contexts) == 1:
            return self.restore(contexts[0]["target"], dry_run=dry_run)

        preflight_errors = []
        for context in contexts:
            try:
                self.restore(
                    context["target"],
                    dry_run=True,
                    report_dry_run=dry_run,
                )
            except VfioError as err:
                preflight_errors.append("%s: %s" % (context["target"], err))
        if preflight_errors:
            raise VfioError("Multi-GPU restore preflight failed: " + "; ".join(preflight_errors))
        if dry_run:
            return

        restore_errors = []
        for context in reversed(contexts):
            try:
                self.restore(context["target"])
            except VfioError as err:
                restore_errors.append("%s: %s" % (context["target"], err))
        if restore_errors:
            raise VfioError("Multi-GPU restore incomplete: " + "; ".join(restore_errors))

    def bind(
        self,
        target_bdf,
        allowed_members=None,
        state_path=None,
        dry_run=False,
        report=True,
    ):
        allowed_members = allowed_members or []
        group, members = self.inspect_group(target_bdf)
        members_by_bdf = {member.bdf: member for member in members}
        allowed = self._resolve_allowed_members(allowed_members, members)

        unauthorized = []
        for member in members:
            if member.bdf == target_bdf or member.driver in (None, VFIO_PCI_DRIVER):
                continue
            if member.is_bridge:
                continue
            if member.bdf not in allowed:
                unauthorized.append(member)

        if unauthorized:
            detail = ", ".join(
                "%s (%s, driver=%s)" % (member.bdf, member.class_name, member.driver)
                for member in unauthorized
            )
            flags = " ".join(
                "--allow-group-member %s" % member.bdf
                for member in unauthorized
            )
            raise VfioError(
                "GPU %s shares IOMMU group %s with device(s) in use by the host: %s. "
                "Binding them can disrupt host services. Review the impact, then "
                "repeat vfio bind with %s"
                % (target_bdf, group, detail, flags)
            )

        selected_bdfs = []
        for bdf in [target_bdf] + sorted(allowed, key=_bdf_sort_key):
            if bdf not in selected_bdfs:
                selected_bdfs.append(bdf)
        for bdf in selected_bdfs:
            member = members_by_bdf[bdf]
            if member.header_type != 0:
                raise VfioError("Cannot bind bridge or unsupported PCI header for %s to vfio-pci" % bdf)
            if member.high_impact_name and report:
                warning(
                    "%s is a %s device; binding it to vfio-pci can disrupt host services",
                    bdf,
                    member.high_impact_name,
                )

        state_path = self.state_path(target_bdf, state_path)
        previous_state = self._load_state(state_path, required=False)
        state = copy.deepcopy(previous_state)
        if state is not None:
            self._validate_state(
                state,
                target_bdf,
                group,
                [member.bdf for member in members],
            )
        else:
            unmanaged = [
                bdf
                for bdf in selected_bdfs
                if members_by_bdf[bdf].driver == VFIO_PCI_DRIVER
            ]
            if unmanaged:
                raise VfioError(
                    "%s already bound to vfio-pci, but no restore state exists at %s; "
                    "refusing to adopt externally managed devices"
                    % (", ".join(unmanaged), state_path)
                )
            state = {
                "version": STATE_VERSION,
                "target_bdf": target_bdf,
                "iommu_group": str(group),
                "devices": {},
            }

        to_bind = []
        snapshot = {}
        for bdf in selected_bdfs:
            member = members_by_bdf[bdf]
            if member.driver == VFIO_PCI_DRIVER:
                info("%s is already bound to vfio-pci", bdf)
                continue
            if bdf not in state["devices"]:
                state["devices"][bdf] = self._device_state(member)
            else:
                self._check_restore_driver(bdf, member.driver, state["devices"][bdf])
            snapshot[bdf] = self._device_state(member)
            to_bind.append(bdf)

        if dry_run:
            if report:
                if not to_bind:
                    info("Dry run: all selected devices are already bound to vfio-pci")
                for bdf in to_bind:
                    info("Dry run: would bind %s to vfio-pci", bdf)
            return

        if os.geteuid() != 0:
            raise VfioError("Binding devices to vfio-pci requires root privileges")

        if to_bind and not self.access.driver_registered(VFIO_PCI_DRIVER):
            self.access.load_module(VFIO_PCI_DRIVER)
        for bdf in to_bind:
            expected = snapshot[bdf]["driver"]
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
                self._bind_one(bdf, snapshot[bdf], attempted)

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
        except BaseException as err:
            rollback_errors = self._rollback(attempted, {"devices": snapshot})
            if not rollback_errors:
                try:
                    self._restore_record(state_path, previous_state)
                except VfioError as restore_error:
                    rollback_errors.append(str(restore_error))
            detail = str(err)
            if rollback_errors:
                detail += "; rollback errors: " + "; ".join(rollback_errors)
            if not isinstance(err, Exception):
                if rollback_errors:
                    warning("Interrupted VFIO operation: %s", detail)
                raise
            raise VfioError(detail) from err

        info("IOMMU group %s is viable for VFIO", group)
        if to_bind:
            info("Original driver state recorded in %s", state_path)

    def restore(self, target_bdf, state_path=None, dry_run=False, report_dry_run=True):
        state_path = self.state_path(target_bdf, state_path)
        state = self._load_state(state_path, required=False)
        if state is None:
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

        # Reject a changed owner before restoring any device in this group.
        for bdf, original in state["devices"].items():
            self._check_restore_driver(bdf, self.access.current_driver(bdf), original)

        failures = []
        for bdf in reversed(list(state["devices"])):
            original = state["devices"][bdf]
            if dry_run:
                if report_dry_run:
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

    def _batch_contexts(self, target_bdfs, allowed_patterns):
        targets = []
        for target_bdf in target_bdfs:
            if target_bdf not in targets:
                targets.append(target_bdf)
        if not targets:
            raise VfioError("Select at least one GPU for VFIO")

        contexts = []
        groups = {}
        for target_bdf in targets:
            group, members = self.inspect_group(target_bdf)
            if group in groups:
                raise VfioError(
                    "Selected GPUs %s and %s share IOMMU group %s; "
                    "multi-GPU VFIO currently requires one selected GPU per group"
                    % (groups[group], target_bdf, group)
                )
            groups[group] = target_bdf
            contexts.append({
                "target": target_bdf,
                "group": group,
                "members": members,
                "allowed": [],
            })

        member_contexts = []
        for context in contexts:
            for member in context["members"]:
                member_contexts.append((member.bdf, context))

        for pattern in allowed_patterns:
            normalized = pattern.lower()
            matches = [
                (bdf, context)
                for bdf, context in member_contexts
                if normalized in bdf.lower()
            ]
            if not matches:
                raise VfioError(
                    "Allowed group member %s is not in any selected GPU's IOMMU group"
                    % pattern
                )
            if len(matches) > 1:
                raise VfioError(
                    "Allowed group member %s is ambiguous: %s"
                    % (pattern, ", ".join(bdf for bdf, _ in matches))
                )
            bdf, context = matches[0]
            if bdf == context["target"]:
                raise VfioError("Selected GPU %s does not need --allow-group-member" % bdf)
            if bdf not in context["allowed"]:
                context["allowed"].append(bdf)
        return contexts

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
        return [
            member
            for member in members
            if not member.is_bridge
            and member.driver not in (None, VFIO_PCI_DRIVER)
        ]

    @staticmethod
    def _device_state(device):
        return {
            "driver": device.driver,
            "driver_override": device.driver_override,
            "module": device.driver_module,
        }

    def _restore_record(self, path, previous_state):
        if previous_state is None:
            self._remove_state(path)
        else:
            self._write_state(path, previous_state)

    @staticmethod
    def _check_restore_driver(bdf, current, original):
        expected = original["driver"]
        if current not in (None, VFIO_PCI_DRIVER, expected):
            raise VfioError(
                "Cannot restore %s: another driver (%s) has bound it; "
                "expected %s, vfio-pci, or unbound"
                % (bdf, current, expected or "unbound")
            )

    def _bind_one(self, bdf, expected, attempted):
        current = self.access.current_driver(bdf)
        if (current != expected["driver"] or
                self.access.driver_override(bdf) != expected["driver_override"]):
            raise VfioError("%s driver state changed during VFIO preflight" % bdf)
        attempted.append(bdf)
        self.access.set_driver_override(bdf, VFIO_PCI_DRIVER)
        if current is not None:
            self.access.unbind(bdf)
        self.access.probe(bdf)
        actual = self.access.current_driver(bdf)
        if actual != VFIO_PCI_DRIVER:
            raise VfioError("%s driver is %s after probe; expected vfio-pci" % (bdf, actual or "unbound"))
        info("Bound %s to vfio-pci", bdf)

    def _restore_one(self, bdf, original):
        original_driver = original["driver"]
        original_override = original["driver_override"]
        current = self.access.current_driver(bdf)
        self._check_restore_driver(bdf, current, original)

        if current != original_driver:
            if (original_driver is not None and
                    not self.access.driver_registered(original_driver)):
                self.access.load_module(original.get("module") or original_driver)
                if not self.access.driver_registered(original_driver):
                    raise VfioError("Cannot restore %s: driver %s is not registered after module load"
                                    % (bdf, original_driver))
            try:
                if original_driver is not None:
                    self.access.set_driver_override(bdf, original_driver)
                if current is not None:
                    self.access.unbind(bdf)
                if original_driver is not None:
                    self.access.probe(bdf)
            finally:
                self.access.set_driver_override(bdf, original_override)
        else:
            self.access.set_driver_override(bdf, original_override)
        actual = self.access.current_driver(bdf)
        if actual != original_driver:
            raise VfioError(
                "%s driver is %s after restore; expected %s"
                % (bdf, actual or "unbound", original_driver or "unbound")
            )
        if self.access.driver_override(bdf) != original_override:
            raise VfioError("%s driver_override was not restored" % bdf)
        info("Restored %s to %s", bdf, original_driver or "unbound")

    def _rollback(self, attempted, state):
        errors = []
        for bdf in reversed(attempted):
            try:
                self._restore_one(bdf, state["devices"][bdf])
            except Exception as err:
                errors.append("%s: %s" % (bdf, err))
        return errors

    @staticmethod
    def _validate_state(state, target_bdf, group, group_members):
        if not isinstance(state, dict):
            raise VfioError("Restore state must be a JSON object")
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
        if target_bdf not in state["devices"]:
            raise VfioError("Restore state does not include target GPU %s" % target_bdf)
        for bdf, original in state["devices"].items():
            if (not isinstance(bdf, str) or not _BDF_RE.fullmatch(bdf) or
                    not isinstance(original, dict)):
                raise VfioError("Restore state contains an invalid device entry")
            if bdf not in group_members:
                raise VfioError(
                    "Restore state contains %s, which is not in IOMMU group %s"
                    % (bdf, group)
                )
            for key in ("driver", "driver_override"):
                if key not in original:
                    raise VfioError("Restore state is missing %s for %s" % (key, bdf))
            for key in ("driver", "driver_override", "module"):
                value = original.get(key)
                if value is not None and (not isinstance(value, str) or
                                           not value or "\n" in value or "\0" in value):
                    raise VfioError("Restore state contains an invalid %s for %s" % (key, bdf))
                if key in ("driver", "module") and value is not None and not re.fullmatch(
                        r"[a-zA-Z0-9_][a-zA-Z0-9_.-]*", value):
                    raise VfioError("Restore state contains an invalid %s for %s" % (key, bdf))

    @staticmethod
    def _load_state(path, required):
        if not path.exists():
            if required:
                raise VfioError("Restore state does not exist: %s" % path)
            return None
        try:
            with path.open("r") as stream:
                state = json.load(stream)
        except (OSError, ValueError) as err:
            raise VfioError("Cannot read restore state %s: %s" % (path, err))
        if not isinstance(state, dict):
            raise VfioError("Restore state must be a JSON object: %s" % path)
        return state

    @staticmethod
    def _write_state(path, state):
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
            fd_open = True
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "w") as stream:
                    fd_open = False
                    json.dump(state, stream, indent=2, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, str(path))
            except BaseException:
                if fd_open:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
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
