# Column-wise Tensor Parallelism for llama.cpp / ggml (Design Draft)

Status: Draft (experimental)
Target scope: CUDA backend, single-host, NCCL collectives. Fallback CPU path: none. Future: SYCL/oneCCL.

## Motivation
Row-split TP is implemented today via split buffers and mul_mat row partitioning. Column-wise TP enables sharding large linears along the K dimension, commonly used in Megatron-LM (ColumnParallelLinear/RowParallelLinear) to reduce VRAM per GPU and optimize comm/compute overlap.

## High-level approach
- Add multi-GPU collectives (all-reduce, all-gather, reduce-scatter) to ggml backends.
- Extend split buffer to support a COLUMN axis in addition to ROW.
- Teach CUDA mul_mat to handle column-sharded weights and to perform the required output reductions.
- Integrate into llama graph: introduce column-parallel and row-parallel patterns for attention/MLP.

## API additions
1) ggml collective ops (new ggml_op values):
- GGML_OP_ALL_REDUCE_SUM (float, bf16; scope: last dim contiguous, simple tensor view support)
- GGML_OP_ALL_GATHER (concat along a chosen axis)
- GGML_OP_REDUCE_SCATTER_SUM (optional for later optimization)

Each op carries attrs:
- axis (int8): which dim to gather/concat; for all-reduce axis is ignored (sum over devices)
- group_id (int32): communicator/group
- dtype: driven by tensor dtype

2) Backend registry extensions:
- ggml_backend_reg_get_proc_address:
  - "ggml_backend_collectives_context" -> returns an opaque handle per backend for communicator init/join.
  - "ggml_backend_allreduce"/"ggml_backend_allgather" (function pointers) or implement via compute_forward switch in backend.

3) Split buffer axis support:
- New C++ enum ggml_split_axis { ROW=0, COL=1 }
- New function signature (backend side):
  - ggml_backend_split_buffer_type(int main_device, const float * tensor_split, int axis)
- llama_model: if split_mode == COL, request axis=COL when selecting GPU buft list.

Note: keep existing signature and add an alternate proc name, e.g. "ggml_backend_split_buffer_type_v2" that includes axis to avoid breaking other backends initially.

## CUDA backend design
- Column partitioning: For weight W [K,M] (stored transposed in ggml as needed), define per-device col_low/col_high slices along K. Compute local y_partial = W_i x_i on each GPU.
- Reduction: All-reduce y_partial across devices to produce final y on each participating device; or reduce-scatter if subsequent op can consume sharded output (RowParallel pattern).
- Implementation specifics:
  - Extend split buffer type context with axis and compute col ranges analogous to get_row_split.
  - Extend ggml_cuda_mul_mat to branch on axis==COL path:
    - Compute per-device slice offsets using nb0/nb1 strides.
    - Launch GEMM with sliced W and x (x slice corresponds to K cols).
    - Accumulate into local dst buffer; then perform NCCL all-reduce on dst across devices.
  - Introduce a small NCCL wrapper initialized once per process; rank==device-id mapping.

## Graph integration
- Add two composite helpers in llama graph builder:
  - column_parallel_linear(W_col_sharded, x): returns y_full with implicit all-reduce. Optionally return sharded y via reduce-scatter (future).
  - row_parallel_linear(W_row_sharded, x_sharded): consumes sharded input, produces y_full or sharded depending on model wiring.
- Apply to:
  - Attention q,k,v projections: column-parallel on input emb dim; out projection as row-parallel.
  - MLP up/gate as column-parallel; down as row-parallel.

## Weight sharding
- At model load (llama_model::load_tensors), for split_mode=COL and n_devices>1, place linear weights on column-split buffers.
- Ensure quantization block alignment across device boundaries; initial CUDA path supports fp16/bf16 for COL TP. Quantized column kernels can come later.

## Milestones
- M1 (this PR series):
  - Add split-mode col flag and plumb axis=COL through buffer selection (without enabling compute yet).
  - Add design doc (this file).
- M2:
  - Implement CUDA NCCL collectives and a minimal GGML_OP_ALL_REDUCE_SUM.
  - Add split_buffer_type_v2 with axis and CUDA implementation for axis==COL (fp16/bf16 only), teach mul_mat column path with all-reduce.
  - Gate behind env flag LLAMA_COL_TP=1.
- M3:
  - Graph wiring for ColumnParallel/RowParallel on attention/MLP.
  - Basic tests: numerical parity vs single-GPU, multi-GPU smoke.
- M4:
  - Quantized kernels for column path, perf tuning, overlap comm/compute.

## Testing strategy
- Unit: small GEMM with column splits across 2 GPUs, compare to reference GEMM.
- Integration: run llama-bench with --split-mode col and compare outputs/latency vs 1 GPU.
- Error paths: single GPU + col split, device with zero-range shard, NCCL absence -> fallback or error.

## Risks
- Complexity in memory layout for quantized formats.
- Communicator lifecycle and stream synchronization bugs.
- Cross-backend portability; keep CUDA-only until abstractions settle.

