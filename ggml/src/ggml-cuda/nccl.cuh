#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Define symbol visibility attribute for shared libraries
#if defined(_WIN32) || defined(__CYGWIN__)
    #define GGML_CUDA_API __declspec(dllexport)
#else
    #define GGML_CUDA_API __attribute__((visibility("default")))
#endif

// Function declarations - always available regardless of NCCL support
// Initialize NCCL for tensor parallelism
bool ggml_cuda_nccl_init(const std::vector<int>& device_ids);

// Cleanup NCCL resources
void ggml_cuda_nccl_cleanup();

// Check if NCCL is available
bool ggml_cuda_nccl_available();

// All-reduce operations
void ggml_cuda_nccl_all_reduce_f32(float* data, size_t count, int device_id, cudaStream_t stream);
void ggml_cuda_nccl_all_reduce_f16(half* data, size_t count, int device_id, cudaStream_t stream);

// All-gather operations
void ggml_cuda_nccl_all_gather_f32(const float* sendbuff, float* recvbuff, size_t sendcount, int device_id, cudaStream_t stream);
void ggml_cuda_nccl_all_gather_f16(const half* sendbuff, half* recvbuff, size_t sendcount, int device_id, cudaStream_t stream);
