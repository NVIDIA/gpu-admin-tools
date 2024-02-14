# NVIDIA GPU Admin Tools

This utility is used for various configuration including the Confidential Computing modes of supported GPUs as well as some debug/test tasks. It is designed to be run as a privileged python3 command.

Supported CC modes are:

- on
  - All supported GPU security features are enabled (e.g., bus encryption, performance counters off)
- devtools
  - All supported GPU security features are enabled, however blocks preventing DevTools profiling/debugging are lifted
- off
  - The GPU operates in its default mode; no supplementary confidential computing features are enabled

## Prerequesites:
  ```bash
  sudo apt install patchelf python3-pip
  ```
## Most Commonly Used Examples
##### Query the CC mode of the first H100 in the system
` sudo python3 ./nvidia_gpu_tools.py --gpu-name=H100 --query-cc-mode`
##### Enable CC-On mode on the first H100 in the system
` sudo python3 ./nvidia_gpu_tools.py --gpu-name=H100 --set-cc-mode=on --reset-after-cc-mode-switch `
##### Disable CC mode on a specific H100 in the system
` sudo python3 ./nvidia_gpu_tools.py --gpu-bdf=45:00.0 --set-cc-mode=off --reset-after-cc-mode-switch`


##### Generic debug dump from GPU
` sudo python3 ./nvidia_gpu_tools.py --gpu-bdf=45:00.0 --debug-dump --log debug`
##### Debug dump of NVLINK state
` sudo python3 ./nvidia_gpu_tools.py --gpu-bdf=45:00.0 --nvlink-debug-dump --log debug`

## Usage
  ```bash
  sudo python3 nvidia_gpu_tools.py --help

NVIDIA GPU Tools version v2024.01.19o
Command line arguments: ['./nvidia_gpu_tools.py', '--help']
Usage: nvidia_gpu_tools.py [options]

Options:
  -h, --help            show this help message and exit
  --gpu=GPU
  --gpu-bdf=GPU_BDF     Select a single GPU by providing a substring of the
                        BDF, e.g. '01:00'.
  --gpu-name=GPU_NAME   Select a single GPU by providing a substring of the
                        GPU name, e.g. 'T4'. If multiple GPUs match, the first
                        one will be used.
  --no-gpu              Do not use any of the GPUs; commands requiring one
                        will not work.
  --log=LOG
  --mmio-access-type=MMIO_ACCESS_TYPE
                        On Linux, specify whether to do MMIO through /dev/mem
                        or /sys/bus/pci/devices/.../resourceN
  --recover-broken-gpu  Attempt recovering a broken GPU (unresponsive config
                        space or MMIO) by performing an SBR. If the GPU is
                        broken from the beginning and hence correct config
                        space wasn't saved then reenumarate it in the OS by
                        sysfs remove/rescan to restore BARs etc.
  --reset-with-sbr      Reset the GPU with SBR and restore its config space
                        settings, before any other actions
  --reset-with-flr      Reset the GPU with FLR and restore its config space
                        settings, before any other actions
  --reset-with-os       Reset with OS through /sys/.../reset
  --remove-from-os      Remove from OS through /sys/.../remove
  --unbind-gpu          Unbind GPU
  --unbind-gpus         Unbind GPUs
  --bind-gpu=BIND_GPU   Bind GPUs to the specified driver
  --bind-gpus=BIND_GPUS
                        Bind GPUs to the specified driver
  --query-ecc-state     Query the ECC state of the GPU
  --query-cc-mode       Query the current Confidential Computing (CC) mode of
                        the GPU.
  --query-cc-settings   Query the Confidential Computing (CC) settings of the
                        GPU.This prints the lower level setting knobs that
                        will take effect upon GPU reset.
  --query-prc-knobs     Query all the Product Reconfiguration (PRC) knobs.
  --set-cc-mode=SET_CC_MODE
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
  --force-ecc-on-after-reset
                        Force ECC to be enabled after a subsequent GPU reset
  --test-ecc-toggle     Test toggling ECC mode.
  --query-mig-mode      Query whether MIG mode is enabled.
  --force-mig-off-after-reset
                        Force MIG mode to be disabled after a subsequent GPU
                        reset
  --test-mig-toggle     Test toggling MIG mode.
  --block-nvlink=BLOCK_NVLINK
                        Block the specified NVLink. Can be specified multiple
                        times to block more NVLinks. NVLinks will be blocked
                        until an SBR. Supported on A100 only.
  --block-all-nvlinks   Block all NVLinks. NVLinks will be blocked until a
                        subsequent SBR. Supported on A100 only.
  --dma-test            Check that GPUs are able to perform DMA to all/most of
                        available system memory.
  --test-pcie-p2p       Check that all GPUs are able to perform DMA to each
                        other.
  --read-sysmem-pa=READ_SYSMEM_PA
                        Use GPU's DMA to read 32-bits from the specified
                        sysmem physical address
  --write-sysmem-pa=WRITE_SYSMEM_PA
                        Use GPU's DMA to write specified 32-bits to the
                        specified sysmem physical address
  --read-config-space=READ_CONFIG_SPACE
                        Read 32-bits from device's config space at specified
                        offset
  --write-config-space=WRITE_CONFIG_SPACE
                        Write 32-bit to device's config space at specified
                        offset
  --read-bar0=READ_BAR0
                        Read 32-bits from GPU BAR0 at specified offset
  --write-bar0=WRITE_BAR0
                        Write 32-bit to GPU BAR0 at specified offset
  --read-bar1=READ_BAR1
                        Read 32-bits from GPU BAR1 at specified offset
  --write-bar1=WRITE_BAR1
                        Write 32-bit to GPU BAR1 at specified offset
  --ignore-nvidia-driver
                        Do not treat nvidia driver apearing to be loaded as an
                        error
