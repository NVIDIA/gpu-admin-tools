#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

class PluginBase:
    def register_options(self, parser):
        """Register CLI options for this plugin."""
        raise NotImplementedError

    def execute_early(self, args):
        return True

    def execute_before_main(self, args, devices):
        return True

    def execute_after_main(self, args, devices):
        if not self.execute_after_main_no_device(args):
            return False

        for d in devices:
            if not self.execute_after_main_per_device(args, d):
                return False
            if d.is_gpu():
                if not self.execute_after_main_per_gpu(args, d):
                    return False
        return True

    def execute_after_main_no_device(self, args):
        return True

    def execute_after_main_per_device(self, args, device):
        return True

    def execute_after_main_per_gpu(self, args, gpu):
        return True
