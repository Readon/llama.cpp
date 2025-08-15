#pragma once

#include "common.cuh"
#include <string>

// Tensor parallelism strategies
enum ggml_tp_strategy {
    GGML_TP_STRATEGY_REPLICATE,  // Replicate tensor across all GPUs
    GGML_TP_STRATEGY_COLUMN,     // Split tensor column-wise (along last dimension)
    GGML_TP_STRATEGY_ROW,        // Split tensor row-wise (along first dimension)
    GGML_TP_STRATEGY_AUTO        // Automatically determine strategy
};

// Tensor parallelism configuration
struct ggml_tp_config {
    int tp_size;        // Number of GPUs in tensor parallel group
    int tp_rank;        // Rank of current GPU in the group
    bool enabled;       // Whether tensor parallelism is enabled
    
    ggml_tp_config() : tp_size(1), tp_rank(0), enabled(false) {}
    ggml_tp_config(int size, int rank) : tp_size(size), tp_rank(rank), enabled(size > 1) {}
};

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

// Integration with CUDA backend
struct ggml_backend_cuda_tp_context {
    ggml_tp_config config;
    std::vector<int> device_ids;
    bool nccl_initialized;
    
    ggml_backend_cuda_tp_context(int tp_size, const std::vector<int>& devices);
    ~ggml_backend_cuda_tp_context();
    
    bool init();
    void cleanup();
};

// Global tensor parallelism context
extern std::unique_ptr<ggml_backend_cuda_tp_context> g_cuda_tp_ctx;

// Initialize CUDA tensor parallelism
bool ggml_cuda_tp_init(int tp_size, const int* device_ids, int num_devices);

// Cleanup CUDA tensor parallelism
void ggml_cuda_tp_cleanup();

// Check if CUDA tensor parallelism is available
bool ggml_cuda_tp_available();

// Get the current TP configuration
const ggml_tp_config& ggml_cuda_tp_get_config();
