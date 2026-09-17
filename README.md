# NVIDIA GPU Admin Tools

This utility is used for various configuration including the Confidential Computing (CC) and Bare Metal Secure AI (BMSAI) modes of supported GPUs as well as some debug/test tasks. It is designed to be run as a privileged python3 command.

Supported CC and BMSAI modes are:

- on
  - All supported GPU security features are enabled (e.g., bus encryption, performance counters off)
- devtools
  - All supported GPU security features are enabled, however blocks preventing DevTools profiling/debugging are lifted
- off
  - The GPU operates in its default mode; no supplementary security features are enabled

## Most Commonly Used Examples
##### Query the CC mode of all GPUs the system
` sudo python3 ./nvidia_gpu_tools.py --devices gpus --query-cc-mode`
##### Query the CC mode of first 4 GPUs the system
` sudo python3 ./nvidia_gpu_tools.py --devices gpus[0:4] --query-cc-mode`
##### Enable CC mode on all GPUs
` sudo python3 ./nvidia_gpu_tools.py --devices gpus --set-cc-mode=on --reset-after-cc-mode-switch `
##### Disable CC mode on a specific GPU in the system
` sudo python3 ./nvidia_gpu_tools.py --devices 45:00.0 --set-cc-mode=off --reset-after-cc-mode-switch`
##### Query the BMSAI mode of all GPUs the system
` sudo python3 ./nvidia_gpu_tools.py --devices gpus --query-bmsai-mode`
##### Enable BMSAI mode on all GPUs
` sudo python3 ./nvidia_gpu_tools.py --devices gpus --set-bmsai-mode=on --reset-after-mode-switch `
##### Disable BMSAI mode on a specific GPU in the system
` sudo python3 ./nvidia_gpu_tools.py --devices 45:00.0 --set-bmsai-mode=off --reset-after-mode-switch`


##### Generic debug dump from GPU
` sudo python3 ./nvidia_gpu_tools.py --gpu-bdf=45:00.0 --debug-dump --log debug`
##### Debug dump of NVLINK state
` sudo python3 ./nvidia_gpu_tools.py --gpu-bdf=45:00.0 --nvlink-debug-dump --log debug`

## Device discovery and selection

Running the tool without a device operation lists PCI identity information. On
Linux, discovery reads the sysfs identity attributes without opening PCI
configuration space or mapping BARs. Product names and chip families come from
the generated PCI ID tables; shared product IDs may also require a known
subsystem ID. Unknown products remain selectable by BDF or index.
Full BDF selectors read only the named sysfs devices. Partial BDF selectors
filter directory names before reading device identities; names, indices, and
mixed selectors use the full inventory.

`--devices 'gpus[0]'`, `--gpu`, `--gpu-bdf`, and `--gpu-name` select from this
inventory before hardware initialization. `--gpu` indexes GPUs and NVSwitches
together; `gpus[...]` indexes only GPUs. Only selected devices are initialized,
and only when a requested operation needs hardware access. On Windows, only one
selected device can be initialized per invocation.
`--no-gpu` skips discovery entirely for operations on files.

For scripts, add `--list-device-bdfs` to print the selected PCI addresses, one
per stdout line. Other output goes to stderr. The query itself does not require
GPU initialization; any accompanying hardware operations still run normally.
For example, `python3 nvidia_gpu_tools.py --gpu-name H100 --list-device-bdfs`
prints the first matching GPU's BDF, including when it is bound to `vfio-pci`.

## MMIO through vfio-pci

On Linux, `--mmio-access-type vfio` accesses PCI BARs through the standard
`vfio-pci` driver. When an operation initializes a selected device, it loads
`vfio-pci` and binds that device automatically if it is unbound. An enabled IOMMU and the distribution's signed VFIO modules allow
this access when kernel lockdown blocks sysfs BAR mappings and `/dev/mem`.
Run as root for automatic binding, or use an already bound device with permission
to access `/dev/vfio/vfio` and its `/dev/vfio/GROUP`.
Select devices from the shared metadata inventory with `--devices`, `--gpu`,
a unique `--gpu-bdf`, or `--gpu-name`. Listing and metadata-only commands do not
bind devices.

```bash
sudo python3 nvidia_gpu_tools.py --mmio-access-type vfio --devices 0009:01:00.0 --query-cc-mode
```

An existing driver is left in place by default and the tool reports an error.
After stopping device workloads, add `--vfio-force-bind` to unbind that driver and
bind the selected device to `vfio-pci`:

```bash
sudo python3 nvidia_gpu_tools.py --mmio-access-type vfio --vfio-force-bind --devices 0009:01:00.0 --query-cc-mode
```

Binding changes apply to the selected devices, not their parent bridges or other
members of their IOMMU groups. Every group member must be available to VFIO;
release any remaining host drivers separately. No-IOMMU mode is not supported.
Opening or closing a VFIO device can reset it, so use an idle device under your
control. Devices remain bound to `vfio-pci` after the command exits; any previous
`driver_override` value is preserved. To return a device to the NVIDIA driver:

```bash
bdf=0009:01:00.0  # Replace with the intended device's full PCI address.
printf '%s\n' "$bdf" | sudo tee /sys/bus/pci/drivers/vfio-pci/unbind
sudo modprobe nvidia
printf '%s\n' "$bdf" | sudo tee /sys/bus/pci/drivers_probe
```

VFIO also handles PCI configuration access for bound devices, including the
memory-enable command used during initialization. Readable sysfs metadata is
still required for device discovery, BAR layout, IOMMU groups, and parent bridge
configuration reads. Operations that write parent bridge configuration (such as
secondary bus reset) remain subject to lockdown. This backend does not establish
DMA mappings; commands requiring device DMA to host memory need separate support.
See the [Linux VFIO documentation](https://www.kernel.org/doc/html/latest/driver-api/vfio.html)
for IOMMU group ownership and host setup details.

## Prepare GPUs for QEMU with VFIO

The `vfio` plugin temporarily moves one or more selected NVIDIA GPUs from their
host drivers to `vfio-pci`, making them available for assignment to a QEMU
guest. It checks every selected GPU's IOMMU isolation group, records the
original host drivers, and restores them after the guest stops.

Before binding, stop workloads and services that use the selected GPUs and
confirm that the host can tolerate their temporary removal. `vfio status`
checks device-assignment prerequisites, but it does not determine whether an
application is actively using a GPU.

The shortest workflow is status, preview, bind, and restore:

```bash
# 1. Show the selected GPU, related devices, readiness, and next action.
sudo python3 ./nvidia_gpu_tools.py --gpu 0 vfio status

# 2. Preview the driver changes. This does not modify the host.
sudo python3 ./nvidia_gpu_tools.py --gpu 0 vfio bind --dry-run

# 3. Prepare the GPU for QEMU, then launch the guest.
sudo python3 ./nvidia_gpu_tools.py --gpu 0 vfio bind

# 4. After the guest has stopped, return the GPU to its host driver.
sudo python3 ./nvidia_gpu_tools.py --gpu 0 vfio restore
```

For a fixed deployment, `--gpu-bdf 45:00.0` selects the GPU by PCI address and
avoids relying on an index. `vfio query` remains an alias for `vfio status`.
VFIO actions use the shared PCI metadata discovery and selectors, including
`--gpu-name`. Selection does not open GPU BARs or initialize the GPU, so it
also works after a device belongs to `vfio-pci`.
The legacy `--gpu` index includes NVSwitch devices in its ordering; the plugin
rejects an index pointing to an NVSwitch. Use `--devices 'gpus[0]'` for an index
that counts only GPUs.

### Relationship to VFIO MMIO access

`vfio bind` records the original drivers and keeps a `vfio-pci` override until
`vfio restore`. The `--mmio-access-type vfio` backend opens BARs and PCI
configuration space for normal GPU-tools commands. Its automatic binding keeps
the prior override and does not create a restore record.

These workflows can be combined in separate commands: use `vfio bind`, run an
MMIO command with `--mmio-access-type vfio` and the same BDF, then use
`vfio restore`. Stop QEMU before accessing the GPU with GPU tools or restoring
its driver. Global GPU options, including their modifiers, request initialization
before the VFIO action. A plain `vfio` action
uses the selected PCI metadata without initializing GPUs. `vfio bind --dry-run` or `vfio restore --dry-run`
previews only the VFIO action; any explicitly requested global operations still run.
If MMIO automatic binding ran first, `vfio status` reports the GPU as externally
managed; it cannot infer the original driver for a later restore.

The passthrough workflow checks group viability without configuring a VFIO
container, so QEMU can use either its group/container interface or IOMMUFD.
The readiness check currently requires the `/dev/vfio/GROUP` interface even
when the guest uses IOMMUFD.
Our MMIO backend currently requires a TYPE1 or TYPE1v2 IOMMU container.

### Multi-GPU passthrough

Use the existing `--devices` selector for Multi-GPU Passthrough (MPT). A fixed
BDF list is preferable in deployment automation; a `gpus` slice is convenient
for interactive use:

```bash
# Inspect and preview every selected GPU and IOMMU group before changing drivers.
sudo python3 ./nvidia_gpu_tools.py --devices '0000:45:00.0,0000:85:00.0' vfio status
sudo python3 ./nvidia_gpu_tools.py --devices '0000:45:00.0,0000:85:00.0' \
  vfio bind --dry-run

# Bind the complete set, then restore the same set after the guest stops.
sudo python3 ./nvidia_gpu_tools.py --devices '0000:45:00.0,0000:85:00.0' vfio bind
sudo python3 ./nvidia_gpu_tools.py --devices '0000:45:00.0,0000:85:00.0' vfio restore

# Equivalent selector for the first two GPUs discovered through sysfs.
sudo python3 ./nvidia_gpu_tools.py --devices 'gpus[0:2]' vfio status
```

Multi-GPU bind preflights every group before making changes. If a later group
fails, changes made by that command are undone in reverse order, including any
new changes to a previously managed group. Each
GPU retains its own recovery record so an interrupted operation can be inspected
and restored by BDF. Selected GPUs must belong to distinct IOMMU groups.

### When the GPU shares an IOMMU group

An IOMMU group is the smallest set of PCI devices the host can securely assign
to a guest. Some systems place a GPU in the same group as another endpoint, such
as a network adapter. The plugin will not remove that device from the host
without explicit authorization. `vfio status` identifies it and prints the flag
to use after its host impact has been reviewed:

```bash
sudo python3 ./nvidia_gpu_tools.py --gpu-bdf 45:00.0 vfio bind --dry-run \
  --allow-group-member 44:00.0
sudo python3 ./nvidia_gpu_tools.py --gpu-bdf 45:00.0 vfio bind \
  --allow-group-member 44:00.0
```

Repeat `--allow-group-member BDF` for each additional endpoint reported by
`status`. PCI bridges are shown for context but are not assigned to the guest.
Warnings identify network, storage, and management-class devices because
removing one can interrupt host services. Before authorizing a network device,
confirm that it is not carrying host management, storage, or fabric traffic.

The bind operation stores a per-GPU restore record under
`/run/nvidia-gpu-tools`. Use `--state-file PATH` only when another location is
required for a single-GPU operation. The plugin refuses to take ownership of a
GPU already bound by a different tool when no matching restore record exists.
Coordinate driver changes with other users and tools; these commands do not
serialize concurrent invocations. Status and dry-run do not create state
directories or records. Restore refuses to detach an unexpected driver and
retains the record if recovery fails.
The default records are volatile and disappear on reboot; they are intended for
bind/restore operations within one host boot.

This plugin prepares device drivers only. It does not enable the platform IOMMU,
configure GPU Confidential Computing mode, create a VM definition, or launch
QEMU. Complete those platform and guest steps before or after VFIO preparation
as required by the deployment guide.

## Usage
  ```bash
sudo python3 nvidia_gpu_tools.py --help

NVIDIA GPU Tools version v2025.03.26o
Command line arguments: ['nvidia_gpu_tools.py', '--help']
usage: nvidia_gpu_tools.py [-h] [--devices DEVICES] [--gpu GPU]
                           [--gpu-bdf GPU_BDF] [--gpu-name GPU_NAME]
                           [--no-gpu]
                           [--log {debug,info,warning,error,critical}]
                           [--mmio-access-type {devmem,sysfs,mods,vfio}]
                           [--vfio-force-bind]
                           [--recover-broken-gpu]
                           [--set-next-sbr-to-fundamental-reset]
                           [--reset-with-sbr] [--reset-with-flr]
                           [--reset-with-os] [--remove-from-os]
                           [--sysfs-bind SYSFS_BIND] [--sysfs-unbind]
                           [--query-ecc-state] [--query-cc-mode]
                           [--query-bmsai-mode] [--query-cc-settings]
                           [--query-ppcie-mode] [--query-ppcie-settings]
                           [--query-prc-knobs]
                           [--set-cc-mode {off,on,devtools}]
                           [--set-bmsai-mode {off,on,devtools}]
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
                           {debug-dump,vfio} ...

positional arguments:
  {debug-dump,vfio}

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
  --mmio-access-type {devmem,sysfs,mods,vfio}
                        On Linux, specify whether to do MMIO through /dev/mem,
                        /sys/bus/pci/devices/.../resourceN, /dev/mods, or vfio-pci
                        (automatically binds unbound devices; requires an IOMMU).
                        mods also uses /dev/mods for PCI config access;
                        vfio uses VFIO for bound devices; devmem and sysfs use
                        sysfs for PCI config access.
  --vfio-force-bind     With --mmio-access-type vfio, unbind selected devices
                        from their current driver and bind to vfio-pci; stop
                        device workloads first
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
  --query-bmsai-mode    Query the current Bare Metal Secure AI (BMSAI) mode of
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
                        mode-switch for one way of doing it.
  --set-bmsai-mode {off,on,devtools}
                        Configure Bare Metal Secure AI (BMSAI) mode. The
                        choices are off (disabled), on (enabled) or devtools
                        (enabled in DevTools mode).The GPU needs to be reset
                        to make the selected mode active. See --reset-after-
                        mode-switch for one way of doing it.
  --reset-after-cc-mode-switch, --reset-after-mode-switch
                        Reset the GPU after switching CC or BMSAI mode such
                        that it is activated immediately.
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
