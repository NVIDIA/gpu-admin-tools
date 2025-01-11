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

from time import perf_counter
import os

from .bridge import PciBridge
from logging import debug

class Cx7(PciBridge):
    _cx7_nics = None

    @classmethod
    def find_cx7_nics(cls):
        if cls._cx7_nics != None:
            return cls._cx7_nics

        # Example CX7 + GPU topology below
        # We need to be able to find the CX7 NIC that shares common topology with CX7 PCIe switches

        # 0000:00:0d.0/05:00.0 Infiniband controller [0207]: Mellanox Technologies MT2910 Family [ConnectX-7] [15b3:1021]
        # 0000:00:0d.0/05:00.1 Infiniband controller [0207]: Mellanox Technologies MT2910 Family [ConnectX-7] [15b3:1021]
        # 0000:00:0d.0/05:00.2 Infiniband controller [0207]: Mellanox Technologies MT2910 Family [ConnectX-7] [15b3:1021]
        # 0000:00:0d.0/05:00.3 Infiniband controller [0207]: Mellanox Technologies MT2910 Family [ConnectX-7] [15b3:1021]
        # 0000:15:01.0/16:00.0/17:00.0/18:00.0 Infiniband controller [0207]: Mellanox Technologies MT2910 Family [ConnectX-7] [15b3:1021]
        # 0000:3d:01.0/3e:00.0/3f:00.0/40:00.0 Infiniband controller [0207]: Mellanox Technologies MT2910 Family [ConnectX-7] [15b3:1021]
        # 0000:4c:01.0/4d:00.0/4e:00.0/4f:00.0 Infiniband controller [0207]: Mellanox Technologies MT2910 Family [ConnectX-7] [15b3:1021]
        # 0000:5b:01.0/5c:00.0/5d:00.0/5e:00.0 Infiniband controller [0207]: Mellanox Technologies MT2910 Family [ConnectX-7] [15b3:1021]
        # 0000:97:01.0/98:00.0/99:00.0/9a:00.0 Infiniband controller [0207]: Mellanox Technologies MT2910 Family [ConnectX-7] [15b3:1021]
        # 0000:bd:01.0/be:00.0/bf:00.0/c0:00.0 Infiniband controller [0207]: Mellanox Technologies MT2910 Family [ConnectX-7] [15b3:1021]
        # 0000:cb:01.0/cc:00.0/cd:00.0/ce:00.0 Infiniband controller [0207]: Mellanox Technologies MT2910 Family [ConnectX-7] [15b3:1021]
        # 0000:d9:01.0/da:00.0/db:00.0/dc:00.0 Infiniband controller [0207]: Mellanox Technologies MT2910 Family [ConnectX-7] [15b3:1021]

        # 0000:15:01.0/16:00.0/17:02.0/19:00.0/1a:00.0/1b:00.0 3D controller [0302]: NVIDIA Corporation Device [10de:2901] (rev a1)
        # 0000:3d:01.0/3e:00.0/3f:02.0/41:00.0/42:00.0/43:00.0 3D controller [0302]: NVIDIA Corporation Device [10de:2901] (rev a1)
        # 0000:4c:01.0/4d:00.0/4e:02.0/50:00.0/51:00.0/52:00.0 3D controller [0302]: NVIDIA Corporation Device [10de:2901] (rev a1)
        # 0000:5b:01.0/5c:00.0/5d:02.0/5f:00.0/60:00.0/61:00.0 3D controller [0302]: NVIDIA Corporation Device [10de:2901] (rev a1)
        # 0000:97:01.0/98:00.0/99:02.0/9b:00.0/9c:00.0/9d:00.0 3D controller [0302]: NVIDIA Corporation Device [10de:2901] (rev a1)
        # 0000:bd:01.0/be:00.0/bf:02.0/c1:00.0/c2:00.0/c3:00.0 3D controller [0302]: NVIDIA Corporation Device [10de:2901] (rev a1)
        # 0000:cb:01.0/cc:00.0/cd:02.0/cf:00.0/d0:00.0/d1:00.0 3D controller [0302]: NVIDIA Corporation Device [10de:2901] (rev a1)
        # 0000:d9:01.0/da:00.0/db:02.0/dd:00.0/de:00.0/df:00.0 3D controller [0302]: NVIDIA Corporation Device [10de:2901] (rev a1)

        from utils.sysfs import sysfs_find_devices, sysfs_find_topo_bdfs

        cls._cx7_nics = {}
        for sysfs_dev_path in sysfs_find_devices(0x15b3, 0x1021):
            cx7_bdf = os.path.basename(sysfs_dev_path)
            # Skip the CX7 NIC itself from the topo bdf list
            parent_bdfs = sysfs_find_topo_bdfs(sysfs_dev_path)[1:]
            cls._cx7_nics[cx7_bdf] = parent_bdfs

        debug(f"Founds CX7 NICs {cls._cx7_nics}")
        return cls._cx7_nics

    def devpath_to_id(dev_path):
        bdf = os.path.basename(dev_path)
        return int(bdf.replace(":","").replace(".",""), base=16)

    def __init__(self, dev_path):
        super().__init__(dev_path)

        self.is_cx7 = True

        self._cx7_nic_bdf = None
        self._cx7_nic_bdf_found = False

    def __str__(self):
        return f"CX7 {self.bdf}"

    @property
    def cx7_nic_bdf(self):
        if self._cx7_nic_bdf_found:
            return self._cx7_nic_bdf

        from utils.sysfs import sysfs_find_topo_bdfs
        my_topo_bdfs = sysfs_find_topo_bdfs(self.dev_path)
        cx7_nics = self.find_cx7_nics()

        best_distance = float("inf")
        best_nic = None

        for nic, nic_topo_bdfs in cx7_nics.items():
            for nic_topo_bdf in nic_topo_bdfs:
                try:
                    distance = my_topo_bdfs.index(nic_topo_bdf)
                    if distance < best_distance:
                        best_nic = nic
                        best_distance = distance
                        break
                except:
                    continue

        self._cx7_nic_bdf = best_nic
        self._cx7_nic_bdf_found = True

        return self._cx7_nic_bdf

    def run_mlxlink(self):
        import json
        mlx_link_dump = os.popen(f"mlxlink -d {self.cx7_nic_bdf} --port_type PCIE --depth 3 --pcie_index 0 --node 0 -c -e --json").read()
        mlx_link_dump = json.loads(mlx_link_dump)
        last_fom = mlx_link_dump['result']['output']["EYE Opening Info (PCIe)"]["Last FOM"]["values"]
        last_fom = [int(f) for f in last_fom]
        initial_fom = mlx_link_dump['result']['output']["EYE Opening Info (PCIe)"]["Initial FOM"]["values"]
        initial_fom = [int(f) for f in initial_fom]
        counters = mlx_link_dump['result']['output']["Management PCIe Performance Counters Info"]
        debug(f"full mlxlink dump {mlx_link_dump}")
        return {"last_fom": last_fom, "initial_fom": initial_fom, "counters": counters}

    def run_mcra_query(self, first_addr, last_addr):
        regs = {}
        offset = first_addr
        size = last_addr - first_addr + 4
        time_diff = perf_counter()
        lines =  list(os.popen(f"mcra {self.cx7_nic_bdf} {offset:#x},{size}").readlines())
        time_diff = perf_counter() - time_diff
        debug(f"MCRA query {offset:#x} size {size} took {time_diff*1000:.1f} ms")

        for l in lines:
            offset, value = l.split(" ")
            offset = int(offset, base=16)
            value = int(value, base=16)
            regs[offset] = value

        return regs

