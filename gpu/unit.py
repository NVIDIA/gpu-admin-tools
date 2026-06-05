#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

class GpuUnit:
    name = None

    def __init__(self, device):
        self.device = device
        self.device.units[self.name] = self

    def read(self, bar0_offset):
        return self.device.read(bar0_offset)

    def write(self, bar0_offset, data):
        return self.device.write(bar0_offset, data)

    def debug_print(self):
        pass

    def debug_dump_capture(self, capture, options):
        del capture, options

    def __str__(self):
        return f"{self.device} {self.name}"


class GpuRegisterModuleUnit(GpuUnit):
    def __init__(self, device, register_module):
        super().__init__(device)
        setattr(device, self.name, self)
        self.register_module = register_module

    def debug_dump_capture(self, capture, _options):
        capture.module(self.register_module)


class GpuFixedBaseRegisterModuleUnit(GpuRegisterModuleUnit):
    def __init__(self, device, register_module, base):
        super().__init__(device, register_module)
        self.base = base

    def debug_dump_capture(self, capture, _options):
        capture.module(self.register_module, base=self.base)


class GpuReplicatedRegisterModuleUnit(GpuRegisterModuleUnit):
    def debug_dump_capture(self, capture, _options):
        for instance_name, pri_base in self._instances():
            capture.module(self.register_module, base=pri_base, instance=instance_name)

    def _instances(self):
        raise NotImplementedError


class GpuTopReplicatedRegisterModuleUnit(GpuReplicatedRegisterModuleUnit):
    def __init__(self, device, register_module, top_device_type):
        super().__init__(device, register_module)
        self.top_device_type = top_device_type

    def _instances(self):
        return [
            (f"{self.name}{instance.instance}", instance.pri_base)
            for instance in self._top_instances()
        ]

    def _top_instances(self):
        top_instances = self.device.top.device_info_instances
        instances = top_instances.get(self.top_device_type, [])
        return sorted(instances, key=lambda instance: (instance.instance, instance.pri_base))


DEBUG_DUMP_CAPABILITY = "debug_dump"


def ensure_unit(device, unit_name):
    unit = device.units.get(unit_name)
    if unit is not None:
        return unit

    unit_auto = device.device_units.get(unit_name)
    if unit_auto is None:
        return None

    return unit_auto.create_instance(device)


def ensure_units_with_capability(device, capability):
    ensured = []
    for unit_auto in device.device_units.values():
        if capability not in unit_auto.capabilities:
            continue

        unit = ensure_unit(device, unit_auto.name)
        if unit is not None:
            ensured.append(unit)
    return ensured


def init_autoload_units(device):
    for unit_auto in device.device_units.values():
        if unit_auto.autoload:
            ensure_unit(device, unit_auto.name)


# Base class for GPU units that are automatically picked up from gpu/units/
# See gpu/units/__init__.py for the discovery mechanism
class GpuUnitAutoBase:
    # Initialization order - lower values are initialized first
    order = 10
    autoload = True
    capabilities = frozenset()

    # Create an instance of the unit for the given device, if applicable
    @classmethod
    def create_instance(cls, device):
        return None
