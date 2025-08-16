#pragma once

#include "common.cuh"
#include "ggml-cuda.h"
#include <string>

// Use the types defined in ggml-cuda.h to avoid redefinition

// Determine tensor parallelism strategy based on tensor name and properties
ggml_tp_strategy ggml_get_tensor_parallel_strategy(const std::string& tensor_name, 
                                                   const struct ggml_tensor* tensor,
                                                   const ggml_tp_config& tp_config);

// Check if a tensor should use tensor parallelism
bool ggml_tensor_supports_tp(const std::string& tensor_name, const struct ggml_tensor* tensor);

// Calculate the split dimensions for a tensor given the TP strategy
struct ggml_tp_split_info {
    int64_t split_dim;      // Which dimension to split (-1 for no split)
    int64_t split_size;     // Size of split for this rank
    int64_t split_offset;   // Offset for this rank
    bool needs_all_reduce;  // Whether all-reduce is needed after computation
    bool needs_all_gather;  // Whether all-gather is needed after computation
    ggml_tp_strategy strategy; // The tensor parallelism strategy used
    int tp_rank;            // Tensor parallel rank
    int tp_size;            // Tensor parallel size
    int group_id;           // Which TP group this tensor belongs to
    int64_t original_ne[GGML_MAX_DIMS]; // Original tensor dimensions
};

ggml_tp_split_info ggml_calculate_tp_split(const struct ggml_tensor* tensor,
                                          ggml_tp_strategy strategy,
                                          const ggml_tp_config& tp_config);

// Apply tensor parallelism to a tensor during model loading
bool ggml_apply_tensor_parallel_split(struct ggml_tensor* tensor,
                                     const ggml_tp_config& tp_config,
                                     ggml_tp_strategy strategy);

// Tensor name patterns for different TP strategies
namespace ggml_tp_patterns {
    // Patterns that should use column-wise splitting
    extern const char* column_split_patterns[];
    
    // Patterns that should use row-wise splitting  
    extern const char* row_split_patterns[];
    
    // Patterns that should be replicated
    extern const char* replicate_patterns[];
    
    // Check if tensor name matches any pattern in the list
    bool matches_pattern(const std::string& tensor_name, const char* patterns[]);
}

// Utility functions for tensor parallel operations
namespace ggml_tp_utils {
    // Calculate the number of elements for a given rank in a split
    int64_t get_split_elements(int64_t total_elements, int tp_size, int tp_rank);
    
    // Calculate the offset for a given rank in a split
    int64_t get_split_offset(int64_t total_elements, int tp_size, int tp_rank);
    
    // Check if dimensions are compatible with tensor parallelism
    bool check_tp_compatibility(const struct ggml_tensor* tensor, int tp_size, int split_dim);
}

// Forward declarations for NCCL types
#ifdef GGML_USE_NCCL
#include <nccl.h>
#else
typedef void* ncclComm_t;
#endif

// Integration with CUDA backend
struct ggml_backend_cuda_tp_context {
    ggml_tp_config config;
    std::vector<int> device_ids;
    bool nccl_initialized;
    int group_id;  // ID of this TP group

    // NCCL communication resources
    ncclComm_t nccl_comm;
    cudaStream_t cuda_stream;

    ggml_backend_cuda_tp_context(int tp_size, const std::vector<int>& devices, int group_id = 0);
    ~ggml_backend_cuda_tp_context();

    bool init();
    void cleanup();
};

// Multi-group tensor parallelism manager
struct ggml_backend_cuda_multi_tp_context {
    std::vector<std::unique_ptr<ggml_backend_cuda_tp_context>> tp_groups;
    int num_groups;
    int gpus_per_group;

    ggml_backend_cuda_multi_tp_context(int num_groups, int gpus_per_group);
    ~ggml_backend_cuda_multi_tp_context();

    bool init_all_groups();
    void cleanup_all_groups();
    ggml_backend_cuda_tp_context* get_group(int group_id);
    const ggml_tp_config& get_config(int group_id);
};

// Global multi-group tensor parallelism context
extern std::unique_ptr<ggml_backend_cuda_multi_tp_context> g_cuda_multi_tp_ctx;

// Initialize CUDA tensor parallelism (legacy single-group interface)
bool ggml_cuda_tp_init(int tp_size, const int* device_ids, int num_devices);

// Initialize multi-group CUDA tensor parallelism
bool ggml_cuda_multi_tp_init(int num_groups, int gpus_per_group);

// Cleanup CUDA tensor parallelism
void ggml_cuda_tp_cleanup();

// Check if CUDA tensor parallelism is available
bool ggml_cuda_tp_available();

// Check if multi-group tensor parallelism is available
bool ggml_cuda_multi_tp_available();

// Get the current TP configuration (for single-group compatibility)
const ggml_tp_config& ggml_cuda_tp_get_config();

// Get TP configuration for a specific group
const ggml_tp_config& ggml_cuda_tp_get_config(int group_id);

// Get the number of TP groups
int ggml_cuda_tp_get_num_groups();

// Get GPU ID for a specific group and rank
int ggml_cuda_tp_get_device_id(int group_id, int rank);

// NCCL communication functions
#ifdef GGML_USE_NCCL
bool ggml_cuda_tp_allreduce(void* data, size_t count, ncclDataType_t datatype, int group_id = 0);
bool ggml_cuda_tp_allgather(void* sendbuf, void* recvbuf, size_t count, ncclDataType_t datatype, int group_id = 0);
bool ggml_cuda_tp_reduce_scatter(void* sendbuf, void* recvbuf, size_t count, ncclDataType_t datatype, int group_id = 0);
#else
bool ggml_cuda_tp_allreduce(void* data, size_t count, int datatype, int group_id = 0);
bool ggml_cuda_tp_allgather(void* sendbuf, void* recvbuf, size_t count, int datatype, int group_id = 0);
bool ggml_cuda_tp_reduce_scatter(void* sendbuf, void* recvbuf, size_t count, int datatype, int group_id = 0);
#endif

// C interface functions
extern "C" {
    ggml_tp_strategy ggml_get_tensor_parallel_strategy_c(const char* tensor_name, const struct ggml_tensor* tensor, const ggml_tp_config* tp_config);
    bool ggml_apply_tensor_parallel_split_c(struct ggml_tensor* tensor, const ggml_tp_config* tp_config, ggml_tp_strategy strategy);

    // NCCL communication C interface
    bool ggml_cuda_tp_allreduce_c(void* data, size_t count, int datatype, int group_id);
    bool ggml_cuda_tp_allgather_c(void* sendbuf, void* recvbuf, size_t count, int datatype, int group_id);
    bool ggml_cuda_tp_reduce_scatter_c(void* sendbuf, void* recvbuf, size_t count, int datatype, int group_id);

    // Get TP configuration pointer
    const ggml_tp_config* ggml_cuda_tp_get_config_ptr();
}
