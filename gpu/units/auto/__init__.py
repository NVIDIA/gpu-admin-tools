#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# GPU unit auto-discovery. Only modules in this directory are scanned —
# concrete unit implementations in gpu/units/ are imported lazily by
# each auto class's create_instance().

import pkgutil
import importlib
from gpu.unit import GpuUnitAutoBase

def _load_gpu_units():
    """Dynamically discover GPU units, sorted by order."""
    gpu_units = {}
    for _, module_name, _ in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{module_name}")
        for item_name in dir(module):
            item = getattr(module, item_name)
            if isinstance(item, type) and issubclass(item, GpuUnitAutoBase) and item is not GpuUnitAutoBase:
                gpu_units[item.name] = item()
    # Sort by order (lower values first) - dicts preserve insertion order in Python 3.7+
    return dict(sorted(gpu_units.items(), key=lambda x: x[1].order))

_gpu_units = None
def gpu_units_cached():
    global _gpu_units
    if _gpu_units is None:
        _gpu_units = _load_gpu_units()
    return _gpu_units
