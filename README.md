# NVIDIA GPU Admin Tools

This utility is used for various configuration including the Confidential Computing modes of supported GPUs as well as some debug/test tasks. It is designed to be run as a privileged python3 command.

Supported CC modes are:

- on
  - All supported GPU security features are enabled (e.g., bus encryption, performance counters off)
- devtools
  - All supported GPU security features are enabled, however blocks preventing DevTools profiling/debugging are lifted
- off
  - The GPU operates in its default mode; no supplementary confidential computing features are enabled

## Most Commonly Used Examples
##### Query the CC mode of all GPUs the system
` sudo python3 ./nvidia_gpu_tools.py --devices gpus --query-cc-mode`
##### Query the CC mode of first 4 GPUs the system
` sudo python3 ./nvidia_gpu_tools.py --devices gpus[0:4] --query-cc-mode`
##### Enable CC mode on all GPUs
` sudo python3 ./nvidia_gpu_tools.py --devices gpus --set-cc-mode=on --reset-after-cc-mode-switch `
##### Disable CC mode on a specific GPU in the system
` sudo python3 ./nvidia_gpu_tools.py --devices 45:00.0 --set-cc-mode=off --reset-after-cc-mode-switch`


##### Generic debug dump from GPU
` sudo python3 ./nvidia_gpu_tools.py --gpu-bdf=45:00.0 --debug-dump --log debug`
##### Debug dump of NVLINK state
` sudo python3 ./nvidia_gpu_tools.py --gpu-bdf=45:00.0 --nvlink-debug-dump --log debug`

## Usage
  ```bash
sudo python3 nvidia_gpu_tools.py --help

NVIDIA GPU Tools version v2025.03.26o
Command line arguments: ['nvidia_gpu_tools.py', '--help']
usage: nvidia_gpu_tools.py [-h] [--devices DEVICES] [--gpu GPU]
                           [--gpu-bdf GPU_BDF] [--gpu-name GPU_NAME]
                           [--no-gpu]
                           [--log {debug,info,warning,error,critical}]
                           [--mmio-access-type {devmem,sysfs}]
                           [--recover-broken-gpu]
                           [--set-next-sbr-to-fundamental-reset]
                           [--reset-with-sbr] [--reset-with-flr]
                           [--reset-with-os] [--remove-from-os]
                           [--sysfs-bind SYSFS_BIND] [--sysfs-unbind]
                           [--query-ecc-state] [--query-cc-mode]
                           [--query-cc-settings] [--query-ppcie-mode]
                           [--query-ppcie-settings] [--query-prc-knobs]
                           [--set-cc-mode {off,on,devtools}]
                           [--reset-after-cc-mode-switch]
                           [--test-cc-mode-switch]
                           [--reset-after-ppcie-mode-switch]
                           [--set-ppcie-mode {off,on}]
                           [--test-ppcie-mode-switch]
                           [--set-bar0-firewall-mode {off,on}]
                           [--query-bar0-firewall-mode]
                           [--query-l4-serial-number] [--query-module-name]
                           [--clear-memory] [--debug-dump]
                           [--nvlink-debug-dump]
                           [--knobs-reset-to-defaults-list]
                           [--knobs-reset-to-defaults KNOBS_RESET_TO_DEFAULTS [KNOBS_RESET_TO_DEFAULTS ...]]
                           [--knobs-reset-to-defaults-assume-no-pending-changes]
                           [--knobs-reset-to-defaults-test] [--noop]
                           [--force-ecc-on-after-reset] [--test-ecc-toggle]
                           [--query-mig-mode] [--force-mig-off-after-reset]
                           [--test-mig-toggle]
                           [--block-nvlink BLOCK_NVLINK [BLOCK_NVLINK ...]]
                           [--block-all-nvlinks] [--test-nvlink-blocking]
                           [--dma-test] [--test-pcie-p2p]
                           [--read-sysmem-pa READ_SYSMEM_PA]
                           [--write-sysmem-pa WRITE_SYSMEM_PA WRITE_SYSMEM_PA]
                           [--read-config-space READ_CONFIG_SPACE]
                           [--write-config-space WRITE_CONFIG_SPACE WRITE_CONFIG_SPACE]
                           [--read-bar0 READ_BAR0]
                           [--write-bar0 WRITE_BAR0 WRITE_BAR0]
                           [--read-bar1 READ_BAR1]
                           [--write-bar1 WRITE_BAR1 WRITE_BAR1]
                           [--ignore-nvidia-driver]
                           {} ...

positional arguments:
  {}

options:
  -h, --help            show this help message and exit
  --devices DEVICES     Generic device selector supporting multiple comma-separated specifiers:
                        - 'gpus' - Find all NVIDIA GPUs
                        - 'gpus[n]' - Find nth NVIDIA GPU
                        - 'gpus[n:m]' - Find NVIDIA GPUs from index n to m
                        - 'nvswitches' - Find all NVIDIA NVSwitches
                        - 'nvswitches[n]' - Find nth NVIDIA NVSwitch
                        - 'vendor:device' - Find devices matching 4-digit hex vendor:device ID
                        - 'domain:bus:device.function' - Find device at specific BDF address
  --gpu GPU
  --gpu-bdf GPU_BDF     Select a single GPU by providing a substring of the
                        BDF, e.g. '01:00'.
  --gpu-name GPU_NAME   Select a single GPU by providing a substring of the
                        GPU name, e.g. 'T4'. If multiple GPUs match, the first
                        one will be used.
  --no-gpu              Do not use any of the GPUs; commands requiring one
                        will not work.
  --log {debug,info,warning,error,critical}
  --mmio-access-type {devmem,sysfs}
                        On Linux, specify whether to do MMIO through /dev/mem
                        or /sys/bus/pci/devices/.../resourceN
  --recover-broken-gpu  Attempt recovering a broken GPU (unresponsive config
                        space or MMIO) by performing an SBR. If the GPU is
                        broken from the beginning and hence correct config
                        space wasn't saved then reenumarate it in the OS by
                        sysfs remove/rescan to restore BARs etc.
  --set-next-sbr-to-fundamental-reset
                        Configure the GPU to make the next SBR same as
                        fundamental reset. After the SBR this setting resets
                        back to False. Supported on H100 only.
  --reset-with-sbr      Reset the GPU with SBR and restore its config space
                        settings, before any other actions
  --reset-with-flr      Reset the GPU with FLR and restore its config space
                        settings, before any other actions
  --reset-with-os       Reset with OS through /sys/.../reset
  --remove-from-os      Remove from OS through /sys/.../remove
  --sysfs-bind SYSFS_BIND
                        Bind devices to the specified driver
  --sysfs-unbind        Unbind devices from the current driver
  --query-ecc-state     Query the ECC state of the GPU
  --query-cc-mode       Query the current Confidential Computing (CC) mode of
                        the GPU.
  --query-cc-settings   Query the Confidential Computing (CC) settings of the
                        GPU.This prints the lower level setting knobs that
                        will take effect upon GPU reset.
  --query-ppcie-mode    Query the current Protected PCIe (PPCIe) mode of the
                        GPU or switch.
  --query-ppcie-settings
                        Query the Protected PPCIe (PPCIe) settings of the GPU
                        or switch.This prints the lower level setting knobs
                        that will take effect upon GPU or switch reset.
  --query-prc-knobs     Query all the Product Reconfiguration (PRC) knobs.
  --set-cc-mode {off,on,devtools}
                        Configure Confidentail Computing (CC) mode. The
                        choices are off (disabled), on (enabled) or devtools
                        (enabled in DevTools mode).The GPU needs to be reset
                        to make the selected mode active. See --reset-after-
                        cc-mode-switch for one way of doing it.
  --reset-after-cc-mode-switch
                        Reset the GPU after switching CC mode such that it is
                        activated immediately.
  --test-cc-mode-switch
                        Test switching CC modes.
  --reset-after-ppcie-mode-switch
                        Reset the GPU or switch after switching PPCIe mode
                        such that it is activated immediately.
  --set-ppcie-mode {off,on}
                        Configure Protected PCIe (PPCIe) mode. The choices are
                        off (disabled) or on (enabled).The GPU or switch needs
                        to be reset to make the selected mode active. See
                        --reset-after-ppcie-mode-switch for one way of doing
                        it.
  --test-ppcie-mode-switch
                        Test switching PPCIE mode.
  --set-bar0-firewall-mode {off,on}
                        Configure BAR0 firewall mode. The choices are off
                        (disabled) or on (enabled).
  --query-bar0-firewall-mode
                        Query the current BAR0 firewall mode of the GPU.
                        Blackwell+ only.
  --query-l4-serial-number
                        Query the L4 certificate serial number without the
                        MSB. The MSB could be either 0x41 or 0x40 based on the
                        RoT returning the certificate chain.
  --query-module-name   Query the module name (aka physical ID and module ID).
                        Supported only on H100 SXM and NVSwitch_gen3
  --clear-memory        Clear the contents of the GPU memory. Supported on
                        Pascal+ GPUs. Assumes the GPU has been reset with SBR
                        prior to this operation and can be comined with
                        --reset-with-sbr if not.
  --debug-dump          Dump various state from the device for debug
  --nvlink-debug-dump   Dump NVLINK debug state.
  --knobs-reset-to-defaults-list
                        Show the supported knobs and their default state
  --knobs-reset-to-defaults KNOBS_RESET_TO_DEFAULTS [KNOBS_RESET_TO_DEFAULTS ...]
                        Set various device configuration knobs to defaults.
                        Supported on Turing+ GPUs and NvSwitch_gen3. See
                        --knobs-reset-to-defaults-list for the list of
                        supported knobs and their defaults on a specific
                        device. The option can be specified multiple times to
                        list specific knobs or 'all' can be used to indicate
                        all supported ones should be reset.
  --knobs-reset-to-defaults-assume-no-pending-changes
                        Indicate that the device was reset after last time any
                        knobs were modified. This allows the reset to defaults
                        to be slightly optimized by querying the current state
  --knobs-reset-to-defaults-test
                        Test knob setting and resetting
  --noop                An empty option that can be used to separate nargs=+
                        options from positional arguments
  --force-ecc-on-after-reset
                        Force ECC to be enabled after a subsequent GPU reset
  --test-ecc-toggle     Test toggling ECC mode.
  --query-mig-mode      Query whether MIG mode is enabled.
  --force-mig-off-after-reset
                        Force MIG mode to be disabled after a subsequent GPU
                        reset
  --test-mig-toggle     Test toggling MIG mode.
  --block-nvlink BLOCK_NVLINK [BLOCK_NVLINK ...]
                        Block the specified NVLinks. NVLinks will be blocked
                        until a subsequent GPU reset (SBR on A100, FLR or SBR
                        on Hopper GPUs [based on OOB configuration], FLR or
                        SBR on Blackwell and later). Supported on A100 and
                        later GPUs that have NVLinks.
  --block-all-nvlinks   Block all NVLinks. See --block-nvlink for more
                        details.
  --test-nvlink-blocking
                        Test blocking NVLinks.
  --dma-test            Check that GPUs are able to perform DMA to all/most of
                        available system memory.
  --test-pcie-p2p       Check that all GPUs are able to perform DMA to each
                        other.
  --read-sysmem-pa READ_SYSMEM_PA
                        Use GPU's DMA to read 32-bits from the specified
                        sysmem physical address
  --write-sysmem-pa WRITE_SYSMEM_PA WRITE_SYSMEM_PA
                        Use GPU's DMA to write specified 32-bits to the
                        specified sysmem physical address
  --read-config-space READ_CONFIG_SPACE
                        Read 32-bits from device's config space at specified
                        offset
  --write-config-space WRITE_CONFIG_SPACE WRITE_CONFIG_SPACE
                        Write 32-bit to device's config space at specified
                        offset
  --read-bar0 READ_BAR0
                        Read 32-bits from GPU BAR0 at specified offset
  --write-bar0 WRITE_BAR0 WRITE_BAR0
                        Write 32-bit to GPU BAR0 at specified offset
  --read-bar1 READ_BAR1
                        Read 32-bits from GPU BAR1 at specified offset
  --write-bar1 WRITE_BAR1 WRITE_BAR1
                        Write 32-bit to GPU BAR1 at specified offset
  --ignore-nvidia-driver
                        Do not treat nvidia driver apearing to be loaded as an
                        error