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
import stat
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from pci.vfio import LinuxVfioAccess, VfioError, VfioManager, execute_vfio_action


GPU0 = "0000:1b:00.0"
GPU1 = "0000:2b:00.0"
NIC0 = "0000:18:00.0"
BRIDGE0 = "0000:00:01.0"


class FakeAccess:
    def __init__(self):
        self.devices = {}
        self.groups = {}
        self.loaded_modules = []
        self.fail_probe = set()
        self.fail_module = None
        self.force_not_viable = False

    def add_device(self, bdf, group, vendor, device, class_code, driver, driver_override=None):
        self.devices[bdf] = {
            "group": str(group),
            "vendor": vendor,
            "device": device,
            "class": class_code,
            "driver": driver,
            "driver_override": driver_override,
        }
        self.groups.setdefault(str(group), []).append(bdf)

    def list_pci_bdfs(self):
        return sorted(self.devices)

    def device_exists(self, bdf):
        return bdf in self.devices

    def read_hex(self, bdf, attribute):
        return self.devices[bdf][attribute]

    def current_driver(self, bdf):
        return self.devices[bdf]["driver"]

    def driver_override(self, bdf):
        return self.devices[bdf]["driver_override"]

    def iommu_group(self, bdf):
        return self.devices[bdf]["group"]

    def group_members(self, group):
        return sorted(self.groups[str(group)])

    def set_driver_override(self, bdf, driver):
        self.devices[bdf]["driver_override"] = driver

    def unbind(self, bdf):
        self.devices[bdf]["driver"] = None

    def probe(self, bdf):
        if bdf in self.fail_probe and self.devices[bdf]["driver_override"] == "vfio-pci":
            raise VfioError("injected probe failure")
        self.devices[bdf]["driver"] = self.devices[bdf]["driver_override"]

    def load_module(self, module):
        if module == self.fail_module:
            raise VfioError("injected module load failure")
        self.loaded_modules.append(module)

    def group_viable(self, group):
        if self.force_not_viable:
            return False, "injected non-viable group"
        viable = all(
            self.devices[bdf]["driver"] in (None, "vfio-pci")
            for bdf in self.groups[str(group)]
        )
        return viable, "fake group status"


def populated_access(shared=True):
    access = FakeAccess()
    access.add_device(GPU0, 14, 0x10DE, 0x2901, 0x030200, "nvidia")
    access.add_device(BRIDGE0, 14, 0x8086, 0x1234, 0x060400, None)
    if shared:
        access.add_device(NIC0, 14, 0x15B3, 0x101D, 0x020000, "mlx5_core")
    access.add_device(GPU1, 15, 0x10DE, 0x2901, 0x030200, "nvidia")
    return access


class VfioManagerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.state_dir = Path(self.temporary.name)

    def manager(self, access):
        return VfioManager(access=access, state_dir=self.state_dir)

    def test_resolves_existing_gpu_selectors(self):
        manager = self.manager(populated_access())
        self.assertEqual(manager.resolve_gpu(gpu=0), GPU0)
        self.assertEqual(manager.resolve_gpu(gpu_bdf="1b:00"), GPU0)
        self.assertEqual(manager.resolve_gpu(devices="gpus[1]"), GPU1)
        self.assertEqual(manager.resolve_gpu(devices=GPU1), GPU1)

    def test_query_lists_group_members_and_host_driver_blocker(self):
        report = self.manager(populated_access()).query(GPU0)
        self.assertIn("IOMMU group 14", report)
        self.assertIn("target GPU", report)
        self.assertIn("bridge", report)
        self.assertIn("peer endpoint", report)
        self.assertIn("blocked by %s (mlx5_core)" % NIC0, report)

    def test_bind_refuses_unauthorized_peer_without_changes(self):
        access = populated_access()
        manager = self.manager(access)

        with self.assertRaisesRegex(VfioError, "--allow-group-member"):
            with mock.patch("os.geteuid", return_value=0):
                manager.bind(GPU0)

        self.assertEqual(access.current_driver(GPU0), "nvidia")
        self.assertEqual(access.current_driver(NIC0), "mlx5_core")
        self.assertFalse(manager.state_path(GPU0).exists())

    def test_bind_and_restore_shared_group(self):
        access = populated_access()
        manager = self.manager(access)

        with mock.patch("os.geteuid", return_value=0):
            manager.bind(GPU0, allowed_members=[NIC0])

        self.assertEqual(access.current_driver(GPU0), "vfio-pci")
        self.assertEqual(access.current_driver(NIC0), "vfio-pci")
        self.assertEqual(access.current_driver(GPU1), "nvidia")
        self.assertEqual(access.loaded_modules, ["vfio", "vfio-pci"])

        state_path = manager.state_path(GPU0)
        state = json.loads(state_path.read_text())
        self.assertEqual(state["devices"][GPU0]["driver"], "nvidia")
        self.assertEqual(state["devices"][NIC0]["driver"], "mlx5_core")
        self.assertEqual(stat.S_IMODE(state_path.stat().st_mode), 0o600)

        with mock.patch("os.geteuid", return_value=0):
            manager.restore(GPU0)

        self.assertEqual(access.current_driver(GPU0), "nvidia")
        self.assertEqual(access.current_driver(NIC0), "mlx5_core")
        self.assertIsNone(access.driver_override(GPU0))
        self.assertIsNone(access.driver_override(NIC0))
        self.assertFalse(state_path.exists())

    def test_partial_bind_failure_rolls_back_all_devices(self):
        access = populated_access()
        access.fail_probe.add(NIC0)
        manager = self.manager(access)

        with self.assertRaisesRegex(VfioError, "injected probe failure"):
            with mock.patch("os.geteuid", return_value=0):
                manager.bind(GPU0, allowed_members=[NIC0])

        self.assertEqual(access.current_driver(GPU0), "nvidia")
        self.assertEqual(access.current_driver(NIC0), "mlx5_core")
        self.assertFalse(manager.state_path(GPU0).exists())

    def test_non_viable_group_rolls_back_all_devices(self):
        access = populated_access(shared=False)
        access.force_not_viable = True
        manager = self.manager(access)

        with self.assertRaisesRegex(VfioError, "injected non-viable group"):
            with mock.patch("os.geteuid", return_value=0):
                manager.bind(GPU0)

        self.assertEqual(access.current_driver(GPU0), "nvidia")
        self.assertFalse(manager.state_path(GPU0).exists())

    def test_module_load_failure_leaves_devices_and_state_unchanged(self):
        access = populated_access(shared=False)
        access.fail_module = "vfio-pci"
        manager = self.manager(access)

        with self.assertRaisesRegex(VfioError, "module load failure"):
            with mock.patch("os.geteuid", return_value=0):
                manager.bind(GPU0)

        self.assertEqual(access.current_driver(GPU0), "nvidia")
        self.assertFalse(manager.state_path(GPU0).exists())

    def test_bind_refuses_host_bound_bridge(self):
        access = populated_access(shared=False)
        access.devices[BRIDGE0]["driver"] = "pcieport"
        manager = self.manager(access)

        with self.assertRaisesRegex(VfioError, "will not detach bridges"):
            with mock.patch("os.geteuid", return_value=0):
                manager.bind(GPU0)

        self.assertEqual(access.current_driver(GPU0), "nvidia")
        self.assertEqual(access.current_driver(BRIDGE0), "pcieport")

    def test_restore_preserves_preexisting_driver_override(self):
        access = populated_access(shared=False)
        access.devices[GPU0]["driver_override"] = "nvidia"
        manager = self.manager(access)

        with mock.patch("os.geteuid", return_value=0):
            manager.bind(GPU0)
            manager.restore(GPU0)

        self.assertEqual(access.current_driver(GPU0), "nvidia")
        self.assertEqual(access.driver_override(GPU0), "nvidia")

    def test_restore_rejects_state_for_device_outside_group(self):
        access = populated_access(shared=False)
        manager = self.manager(access)
        state_path = manager.state_path(GPU0)
        state_path.write_text(json.dumps({
            "version": 1,
            "target_bdf": GPU0,
            "iommu_group": "14",
            "devices": {
                GPU0: {"driver": "nvidia", "driver_override": None},
                GPU1: {"driver": "nvidia", "driver_override": None},
            },
        }))

        with self.assertRaisesRegex(VfioError, "not in IOMMU group 14"):
            with mock.patch("os.geteuid", return_value=0):
                manager.restore(GPU0)

        self.assertEqual(access.current_driver(GPU0), "nvidia")
        self.assertEqual(access.current_driver(GPU1), "nvidia")

    def test_dry_run_does_not_modify_devices_or_state(self):
        access = populated_access()
        manager = self.manager(access)

        manager.bind(GPU0, allowed_members=[NIC0], dry_run=True)

        self.assertEqual(access.current_driver(GPU0), "nvidia")
        self.assertEqual(access.current_driver(NIC0), "mlx5_core")
        self.assertEqual(access.loaded_modules, [])
        self.assertFalse(manager.state_path(GPU0).exists())

    def test_bind_is_idempotent(self):
        access = populated_access(shared=False)
        manager = self.manager(access)

        with mock.patch("os.geteuid", return_value=0):
            manager.bind(GPU0)
            state_before = manager.state_path(GPU0).read_text()
            manager.bind(GPU0)

        self.assertEqual(access.current_driver(GPU0), "vfio-pci")
        self.assertEqual(manager.state_path(GPU0).read_text(), state_before)

    def test_restore_without_state_is_idempotent_when_not_bound(self):
        access = populated_access(shared=False)
        manager = self.manager(access)

        with mock.patch("os.geteuid", return_value=0):
            manager.restore(GPU0)

        self.assertEqual(access.current_driver(GPU0), "nvidia")

    def test_restore_without_state_refuses_unknown_vfio_origin(self):
        access = populated_access(shared=False)
        access.devices[GPU0]["driver"] = "vfio-pci"
        manager = self.manager(access)

        with self.assertRaisesRegex(VfioError, "no restore state"):
            manager.restore(GPU0)

    def test_vfio_action_rejects_other_active_operations(self):
        opts = SimpleNamespace(
            allow_group_member=[],
            bind_vfio=True,
            devices=None,
            dry_run=False,
            gpu=0,
            gpu_bdf=None,
            gpu_name=None,
            log="info",
            mmio_access_type="sysfs",
            query_vfio_state=False,
            restore_vfio_drivers=False,
            set_cc_mode="on",
            vfio_state_file=None,
        )

        with self.assertRaisesRegex(VfioError, "--set-cc-mode"):
            execute_vfio_action(opts, manager=self.manager(populated_access()))


class LinuxVfioAccessFixtureTest(unittest.TestCase):
    def test_reads_group_topology_from_fixture_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            devices = root / "bus" / "pci" / "devices"
            group_devices = root / "kernel" / "iommu_groups" / "14" / "devices"
            driver = root / "bus" / "pci" / "drivers" / "nvidia"
            devices.mkdir(parents=True)
            group_devices.mkdir(parents=True)
            driver.mkdir(parents=True)

            device = devices / GPU0
            device.mkdir()
            (device / "vendor").write_text("0x10de\n")
            (device / "device").write_text("0x2901\n")
            (device / "class").write_text("0x030200\n")
            (device / "driver_override").write_text("(null)\n")
            (device / "driver").symlink_to(driver)
            (device / "iommu_group").symlink_to(root / "kernel" / "iommu_groups" / "14")
            (group_devices / GPU0).symlink_to(device)

            access = LinuxVfioAccess(sysfs_root=root, dev_root=root / "dev")
            manager = VfioManager(access=access, state_dir=root / "run")
            group, members = manager.inspect_group(GPU0)

            self.assertEqual(group, "14")
            self.assertEqual([member.bdf for member in members], [GPU0])
            self.assertEqual(members[0].driver, "nvidia")
            self.assertTrue(members[0].is_gpu)


if __name__ == "__main__":
    unittest.main()
