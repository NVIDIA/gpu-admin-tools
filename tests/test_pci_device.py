#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#

import unittest
from unittest.mock import mock_open, patch

from pci.device import PciDevice


class PciDeviceSysfsTest(unittest.TestCase):
    def setUp(self):
        self.device = PciDevice.__new__(PciDevice)
        self.device.dev_path = "/sys/bus/pci/devices/0000:00:00.0"
        self.device.bdf = "0000:00:00.0"
        self.device.vendor = 0x10DE
        self.device.device = 0x0000

    @patch("pci.device.os.path.exists", return_value=False)
    @patch("builtins.open", new_callable=mock_open)
    def test_sysfs_remove_skips_missing_path(self, open_mock, _exists_mock):
        self.device.sysfs_remove()
        open_mock.assert_not_called()

    @patch("pci.device.os.path.exists", return_value=False)
    @patch("builtins.open", new_callable=mock_open)
    def test_sysfs_rescan_skips_missing_path(self, open_mock, _exists_mock):
        self.device.sysfs_rescan()
        open_mock.assert_not_called()

    @patch("pci.device.os.path.exists", return_value=False)
    @patch("builtins.open", new_callable=mock_open)
    def test_sysfs_reset_skips_missing_path(self, open_mock, _exists_mock):
        self.device.sysfs_reset()
        open_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
