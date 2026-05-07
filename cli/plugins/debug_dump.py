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
import tempfile
from logging import error, info, warning

from cli.core import PluginBase
from gpu.debug_dump import CaptureOptions, capture_gpu_bundle, default_output_prefix


class DebugDumpPlugin(PluginBase):
    command_name = "debug-dump"

    def register_options(self, parser):
        parser.add_argument(
            "--output-file",
            help="Write the debug-dump ZIP bundle to this file. If the path does not end in .zip, .zip will be appended.",
        )

    def execute_after_main(self, args, devices):

        gpus = [device for device in devices if device.is_gpu()]
        skipped_devices = [device for device in devices if not device.is_gpu()]
        for device in skipped_devices:
            warning(f"{device} skipped by debug-dump (GPU support only in v1)")

        if not gpus:
            error("debug-dump capture requires at least one GPU selected by the main CLI")
            return False

        options = CaptureOptions()
        output_file = args.output_file if args.output_file else default_output_prefix()
        capture_output_path = output_file
        output_path = capture_gpu_bundle(gpus, capture_output_path, options)
        info(f"Wrote debug dump bundle to {output_path}")
        return True


