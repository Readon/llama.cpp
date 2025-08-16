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

// Tensor name patterns for different TP strategies
namespace ggml_tp_patterns {
    // Patterns that should use column-wise splitting
    extern const char* column_split_patterns[];

    // Patterns that should use row-wise splitting
    extern const char* row_split_patterns[];

    // Patterns that should be replicated
    extern const char* replicate_patterns[];

    // Check if tensor name matches any pattern in the list
    inline bool matches_pattern(const std::string& tensor_name, const char* patterns[]) {
        for (int i = 0; patterns[i] != nullptr; i++) {
            if (tensor_name.find(patterns[i]) != std::string::npos) {
                return true;
            }
        }
        return false;
    }
}

// Determine tensor parallelism strategy based on tensor name and properties
inline ggml_tp_strategy ggml_get_tensor_parallel_strategy(const std::string& tensor_name,
                                                   const struct ggml_tensor* tensor,
                                                   const ggml_tp_config& tp_config) {
    if (!tp_config.enabled) {
        return GGML_TP_STRATEGY_REPLICATE;
    }

    // Check for explicit patterns first
    if (ggml_tp_patterns::matches_pattern(tensor_name, ggml_tp_patterns::column_split_patterns)) {
        return GGML_TP_STRATEGY_COLUMN;
    }

    if (ggml_tp_patterns::matches_pattern(tensor_name, ggml_tp_patterns::row_split_patterns)) {
        return GGML_TP_STRATEGY_ROW;
    }

    if (ggml_tp_patterns::matches_pattern(tensor_name, ggml_tp_patterns::replicate_patterns)) {
        return GGML_TP_STRATEGY_REPLICATE;
    }

    // Auto-determine strategy based on tensor properties
    if (tensor->ne[0] % tp_config.tp_size == 0 && tensor->ne[0] >= tp_config.tp_size) {
        // Can split along first dimension
        return GGML_TP_STRATEGY_ROW;
    } else if (tensor->ne[1] % tp_config.tp_size == 0 && tensor->ne[1] >= tp_config.tp_size) {
        // Can split along second dimension
        return GGML_TP_STRATEGY_COLUMN;
    }

    // Default to replication
    return GGML_TP_STRATEGY_REPLICATE;
}

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

inline ggml_tp_split_info ggml_calculate_tp_split(const struct ggml_tensor* tensor,
                                          ggml_tp_strategy strategy,
                                          const ggml_tp_config& tp_config) {
    ggml_tp_split_info info = {};
    info.split_dim = -1;
    info.split_size = 0;
    info.split_offset = 0;
    info.needs_all_reduce = false;
    info.needs_all_gather = false;

    if (!tp_config.enabled || strategy == GGML_TP_STRATEGY_REPLICATE) {
        return info;
    }

    switch (strategy) {
        case GGML_TP_STRATEGY_COLUMN:
            if (tensor->ne[1] % tp_config.tp_size == 0) {
                info.split_dim = 1;
                info.split_size = tensor->ne[1] / tp_config.tp_size;
                info.split_offset = tp_config.tp_rank * info.split_size;
                info.needs_all_reduce = true;
            }
            break;

        case GGML_TP_STRATEGY_ROW:
            if (tensor->ne[0] % tp_config.tp_size == 0) {
                info.split_dim = 0;
                info.split_size = tensor->ne[0] / tp_config.tp_size;
                info.split_offset = tp_config.tp_rank * info.split_size;
                info.needs_all_gather = true;
            }
            break;

        default:
            break;
    }

    return info;
}

// Apply tensor parallelism to a tensor during model loading
inline bool ggml_apply_tensor_parallel_split(struct ggml_tensor* tensor,
                                     const ggml_tp_config& tp_config,
                                     ggml_tp_strategy strategy) {
    if (!tp_config.enabled || strategy == GGML_TP_STRATEGY_REPLICATE) {
        return true;
    }

    ggml_tp_split_info split_info = ggml_calculate_tp_split(tensor, strategy, tp_config);

    if (split_info.split_dim == -1) {
        return false; // Cannot split this tensor
    }

    // Modify tensor dimensions to reflect the split
    if (split_info.split_dim == 0) {
        tensor->ne[0] = split_info.split_size;
    } else if (split_info.split_dim == 1) {
        tensor->ne[1] = split_info.split_size;
    }

    // Recalculate strides
    tensor->nb[0] = ggml_type_size(tensor->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i-1] * tensor->ne[i-1];
    }

    return true;
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
    int group_id;  // ID of this TP group

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
