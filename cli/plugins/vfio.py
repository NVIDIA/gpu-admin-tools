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

from logging import error

from cli.core import PluginBase
from pci.vfio import VfioError, VfioManager


class VfioPlugin(PluginBase):
    command_name = "vfio"
    requires_device_init = False

    def register_options(self, parser):
        parser.description = (
            "Prepare one or more NVIDIA GPUs for assignment to QEMU with VFIO, "
            "and restore their host drivers afterward. Every selected GPU's "
            "complete IOMMU group is checked before any driver changes."
        )
        actions = parser.add_subparsers(dest="vfio_action", required=True)

        status = actions.add_parser(
            "status",
            aliases=["query"],
            help="Show the selected GPU, related devices, readiness, and next action",
        )
        self._add_state_file_option(status)

        bind = actions.add_parser(
            "bind",
            help="Temporarily move selected GPUs and authorized peers to vfio-pci",
        )
        bind.add_argument(
            "--allow-group-member",
            action="append",
            default=[],
            metavar="BDF",
            help=(
                "Allow this additional device to leave its host driver; "
                "repeat for each device reported by status"
            ),
        )
        self._add_state_file_option(bind)
        self._add_dry_run_option(bind)

        restore = actions.add_parser(
            "restore",
            help="Return devices to the host drivers recorded by bind",
        )
        self._add_state_file_option(restore)
        self._add_dry_run_option(restore)

    @staticmethod
    def _add_state_file_option(parser):
        parser.add_argument(
            "--state-file",
            help=(
                "Use a different restore-record path instead of the per-GPU "
                "default under /run/nvidia-gpu-tools; single-GPU actions only"
            ),
        )

    @staticmethod
    def _add_dry_run_option(parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show planned driver changes without applying them",
        )

    def execute_after_main(self, args, devices):
        try:
            if not devices:
                raise VfioError(
                    "Select at least one NVIDIA GPU with --devices, --gpu, "
                    "--gpu-bdf, or --gpu-name"
                )
            for device in devices:
                if not device.is_gpu():
                    raise VfioError(
                        "VFIO passthrough requires NVIDIA GPUs; %s is not a GPU "
                        "(use --devices gpus[N] for GPU-only indexing)" % device.bdf
                    )
            target_bdfs = [device.bdf for device in devices]
            if len(target_bdfs) > 1 and args.state_file:
                raise VfioError(
                    "--state-file is only supported for a single GPU; "
                    "multi-GPU actions use per-GPU restore records"
                )
            manager = VfioManager()
            if args.vfio_action in ("status", "query"):
                if len(target_bdfs) == 1:
                    print(manager.query(target_bdfs[0], state_path=args.state_file))
                else:
                    print(manager.query_many(target_bdfs))
            elif args.vfio_action == "bind":
                if len(target_bdfs) == 1:
                    manager.bind(
                        target_bdfs[0],
                        allowed_members=args.allow_group_member,
                        state_path=args.state_file,
                        dry_run=args.dry_run,
                    )
                else:
                    manager.bind_many(
                        target_bdfs,
                        allowed_members=args.allow_group_member,
                        dry_run=args.dry_run,
                    )
                if args.dry_run:
                    print("VFIO dry run complete: planned driver changes were not applied.\n")
                if len(target_bdfs) == 1:
                    print(manager.query(target_bdfs[0], state_path=args.state_file))
                else:
                    print(manager.query_many(target_bdfs))
            else:
                if len(target_bdfs) == 1:
                    manager.restore(
                        target_bdfs[0],
                        state_path=args.state_file,
                        dry_run=args.dry_run,
                    )
                else:
                    manager.restore_many(target_bdfs, dry_run=args.dry_run)
                if args.dry_run:
                    print("VFIO dry run complete: planned driver changes were not applied.\n")
                if len(target_bdfs) == 1:
                    print(manager.query(target_bdfs[0], state_path=args.state_file))
                else:
                    print(manager.query_many(target_bdfs))
        except VfioError as err:
            error("%s", err)
            return False
        return True
