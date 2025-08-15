# Tensor Parallelism in llama.cpp

This document describes the tensor parallelism feature in llama.cpp, which enables accelerated model inference across multiple GPUs within tensor parallel groups.

## Overview

Tensor parallelism allows you to split model tensors across multiple GPUs within each group, enabling faster inference for large models. This is different from layer-wise model parallelism, where entire layers are distributed across GPUs.

## Key Features

- **`--gpus-tp` parameter**: Specifies the number of GPUs per tensor parallel group
- **NCCL-based communication**: Uses NCCL for efficient collective operations between GPUs
- **Automatic tensor splitting**: Intelligently determines how to split tensors based on their names and properties
- **Backward compatibility**: Preserves all existing functionality when `--gpus-tp=1` (default)

## Usage

### Basic Usage

```bash
# Single GPU (default behavior)
./llama-cli -m model.gguf --prompt "Hello world"

# Tensor parallelism with 2 GPUs per group
./llama-cli -m model.gguf --gpus-tp 2 --prompt "Hello world"

# Tensor parallelism with 4 GPUs per group
./llama-cli -m model.gguf --gpus-tp 4 --prompt "Hello world"
```

### Advanced Usage

```bash
# Combine tensor parallelism with layer distribution
./llama-cli -m model.gguf --gpus-tp 2 --tensor-split 8,8,8 -ngl 32 --prompt "Hello world"

# This creates 3 GPU groups, each with 2 GPUs for tensor parallelism
# Total of 6 GPUs used: Groups [0,1], [2,3], [4,5]
```

### Environment Variable

You can also set the tensor parallel size using an environment variable:

```bash
export LLAMA_ARG_GPUS_TP=2
./llama-cli -m model.gguf --prompt "Hello world"
```

## Requirements

### Hardware Requirements

- Multiple CUDA-compatible GPUs
- Sufficient GPU memory for model shards
- High-bandwidth GPU interconnect (NVLink recommended for best performance)

### Software Requirements

- CUDA toolkit
- NCCL library (for multi-GPU communication)
- llama.cpp compiled with CUDA support (`-DGGML_USE_CUDA=ON`)

## How It Works

### Tensor Splitting Strategies

The implementation automatically determines how to split tensors based on their names:

1. **Column-wise splitting**: Applied to output projection layers
   - `*.attn_output.weight`
   - `*.ffn_down.weight`
   - `*.ffn_gate.weight`
   - `*.ffn_up.weight`

2. **Row-wise splitting**: Applied to input projection layers
   - `*.attn_q.weight`
   - `*.attn_k.weight`
   - `*.attn_v.weight`

3. **Replication**: Applied to normalization layers and embeddings
   - `*.norm.weight`
   - `*.tok_embd.weight`

### Communication Patterns

- **All-reduce**: Used after column-wise split operations to sum partial results
- **All-gather**: Used after row-wise split operations to concatenate results
- **Replication**: No communication needed, tensors are identical across GPUs

## Configuration

### Valid Parameter Ranges

- `--gpus-tp`: 1-8 (default: 1)
- Must be compatible with available GPU count
- Total GPUs must be divisible by `--gpus-tp`

### Compatibility

- **Compatible with**: `--tensor-split`, `-ngl`, `--main-gpu`
- **Requires**: `--split-mode layer` (default)
- **Not compatible with**: `--split-mode row` or `--split-mode none`

## Performance Considerations

### When to Use Tensor Parallelism

- **Large models**: When model size exceeds single GPU memory
- **High throughput**: When you need to maximize inference speed
- **Multiple small requests**: Better GPU utilization across parallel groups

### When NOT to Use Tensor Parallelism

- **Small models**: Overhead may outweigh benefits
- **Limited GPU interconnect**: Poor performance without high-bandwidth connections
- **Single large request**: Layer parallelism might be more efficient

### Optimization Tips

1. **GPU Placement**: Use GPUs with high-bandwidth interconnects (NVLink)
2. **Memory Balance**: Ensure all GPUs in a group have similar memory capacity
3. **Batch Size**: Larger batch sizes can better amortize communication overhead
4. **Model Size**: Larger models benefit more from tensor parallelism

## Troubleshooting

### Common Issues

1. **"number of available GPUs must be divisible by gpus_tp"**
   - Solution: Adjust `--gpus-tp` to divide evenly into your GPU count

2. **"tensor parallelism is only compatible with layer split mode"**
   - Solution: Remove `--split-mode` parameter or set it to `layer`

3. **NCCL initialization failures**
   - Solution: Check NCCL installation and GPU connectivity

4. **Poor performance**
   - Check GPU interconnect bandwidth
   - Verify all GPUs are being utilized
   - Consider reducing `--gpus-tp` for smaller models

### Debugging

Enable verbose logging to see tensor parallelism initialization:

```bash
./llama-cli -m model.gguf --gpus-tp 2 --verbose
```

Look for log messages like:
- "tensor parallelism enabled with X GPUs per group"
- "NCCL initialized for tensor parallelism"
- GPU memory allocation messages

## Examples

### Example 1: Small Model with 2 GPUs
```bash
./llama-cli -m 7b-model.gguf --gpus-tp 2 -ngl 32 --prompt "Explain quantum computing"
```

### Example 2: Large Model with Multiple GPU Groups
```bash
./llama-cli -m 70b-model.gguf --gpus-tp 4 --tensor-split 16,16 -ngl 64 --prompt "Write a story"
```

### Example 3: Maximum Performance Setup
```bash
./llama-cli -m 405b-model.gguf --gpus-tp 8 -ngl 128 --prompt "Solve this problem" -n 100
```

## Testing

Run the tensor parallelism tests:

```bash
# Unit tests
make test-tensor-parallel

# Integration tests
./tests/test-tensor-parallel-integration.sh
```

## Future Improvements

- Support for pipeline parallelism
- Dynamic load balancing
- Automatic GPU topology detection
- Integration with distributed inference frameworks
