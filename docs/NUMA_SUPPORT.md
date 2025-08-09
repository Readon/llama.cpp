# NUMA Support for CPU Column-wise Tensor Parallelism

This document describes the NUMA (Non-Uniform Memory Access) support for CPU column-wise tensor parallelism in llama.cpp.

## Overview

NUMA support enables efficient CPU column-wise tensor parallelism on multi-socket systems by:
- Allocating memory on specific NUMA nodes for optimal locality
- Binding computation threads to corresponding NUMA nodes
- Minimizing cross-node memory access latency

## Build Configuration

### Enabling NUMA Support

To enable NUMA support, use the `GGML_NUMA` CMake option:

```bash
# Enable NUMA support
cmake -B build -DGGML_NUMA=ON

# Or disable NUMA support (default)
cmake -B build -DGGML_NUMA=OFF
```

### Requirements

When `GGML_NUMA=ON`:
- **Linux system** (NUMA support is Linux-only)
- **libnuma-dev** package installed
- **Multi-socket system** with multiple NUMA nodes

#### Installing libnuma-dev

```bash
# Ubuntu/Debian
sudo apt install libnuma-dev

# CentOS/RHEL/Fedora
sudo yum install numactl-devel
# or
sudo dnf install numactl-devel
```

### Build Examples

```bash
# Build with NUMA support
cmake -B build -DGGML_NUMA=ON
cd build && make -j$(nproc)

# Build without NUMA support (default)
cmake -B build
cd build && make -j$(nproc)
```

## Usage

### System Requirements Check

Before using CPU column TP, verify your system has multiple NUMA nodes:

```bash
# Check NUMA topology
lscpu | grep -i numa
numactl --hardware

# Expected output for multi-socket system:
# NUMA node(s):          2
# NUMA node0 CPU(s):     0-19,40-59
# NUMA node1 CPU(s):     20-39,60-79
```

### Using CPU Column TP

```bash
# Enable CPU column-wise tensor parallelism
# (Requires NUMA system when GGML_NUMA=ON)
./llama-bench --split-mode col --numa distribute

# Force CPU-only execution
CUDA_VISIBLE_DEVICES="" ./llama-bench --split-mode col --numa distribute
```

### Behavior by Configuration

| GGML_NUMA | System Type | Behavior |
|-----------|-------------|----------|
| `OFF` | Any | Uses regular memory allocation, no NUMA awareness |
| `ON` | Single NUMA node | Shows warning, falls back to regular buffer |
| `ON` | Multi NUMA nodes | Enables NUMA-aware column TP |
| `ON` | Non-Linux | Build warning, NUMA disabled |

## Performance Considerations

### When to Use NUMA Support

✅ **Recommended for:**
- Multi-socket servers (2+ CPU sockets)
- Systems with multiple NUMA nodes
- Large model inference workloads
- Memory-bandwidth intensive operations

❌ **Not recommended for:**
- Single-socket systems
- Desktop/laptop systems
- Systems with single NUMA node

### Performance Tips

1. **Disable NUMA balancing** for better performance:
   ```bash
   echo 0 | sudo tee /proc/sys/kernel/numa_balancing
   ```

2. **Use appropriate NUMA strategy**:
   ```bash
   --numa distribute  # Spread threads across NUMA nodes
   --numa isolate     # Keep threads on current NUMA node
   ```

3. **Monitor NUMA usage**:
   ```bash
   numastat -p $(pgrep llama)
   ```

## Troubleshooting

### Build Issues

**Error: "NUMA library not found"**
```bash
# Install libnuma development package
sudo apt install libnuma-dev
```

**Error: "NUMA support is only available on Linux"**
- NUMA support is Linux-only
- Use `GGML_NUMA=OFF` on other platforms

### Runtime Issues

**Warning: "CPU column-wise tensor parallelism requires NUMA"**
- System has only one NUMA node
- Column TP will fall back to regular CPU buffer
- This is expected behavior on single-socket systems

**Warning: "numa_balancing is enabled"**
- NUMA balancing can impact performance
- Disable with: `echo 0 | sudo tee /proc/sys/kernel/numa_balancing`

## Implementation Details

### Memory Allocation Strategy

- **With NUMA**: Uses `numa_alloc_onnode()` for node-specific allocation
- **Without NUMA**: Uses `ggml_aligned_malloc()` for regular allocation

### Thread Affinity

- **With NUMA**: Binds threads to specific NUMA nodes using `numa_sched_setaffinity()`
- **Without NUMA**: No thread affinity management

### Data Distribution

- Tensor columns are distributed across NUMA nodes
- Each split is allocated on its corresponding NUMA node
- Computation threads access local memory for optimal performance

## Example Output

### NUMA Enabled System
```
NUMA support enabled: 2 nodes detected
CPU Column TP: Allocating buffers across 2 NUMA nodes
  Split 0: 1048576 bytes on NUMA node 0 (ptr=0x7f8b40000000)
  Split 1: 1048576 bytes on NUMA node 1 (ptr=0x7f8b50000000)
Thread bound to NUMA node 0
Thread bound to NUMA node 1
```

### NUMA Disabled System
```
NUMA support not compiled - using single node allocation
  Split 0: 1048576 bytes using regular allocation (ptr=0x7f8b40000000)
  Split 1: 1048576 bytes using regular allocation (ptr=0x7f8b40100000)
```
