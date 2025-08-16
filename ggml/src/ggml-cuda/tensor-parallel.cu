#include "tensor-parallel.cuh"
#include "nccl.cuh"
#include <algorithm>
#include <cstring>

// Global tensor parallelism context
std::unique_ptr<ggml_backend_cuda_tp_context> g_cuda_tp_ctx = nullptr;

// Tensor name patterns for different TP strategies
namespace ggml_tp_patterns {
    // Column-wise split patterns (output projections, feed-forward layers)
    const char* column_split_patterns[] = {
        ".attn_output.weight",
        ".ffn_down.weight", 
        ".ffn_gate.weight",
        ".ffn_up.weight",
        ".output.weight",
        nullptr
    };
    
    // Row-wise split patterns (input projections)
    const char* row_split_patterns[] = {
        ".attn_q.weight",
        ".attn_k.weight", 
        ".attn_v.weight",
        ".attn_qkv.weight",
        nullptr
    };
    
    // Replicate patterns (embeddings, layer norms, biases)
    const char* replicate_patterns[] = {
        ".tok_embd.weight",
        ".norm.weight",
        ".norm.bias",
        ".attn_norm.weight",
        ".ffn_norm.weight",
        ".output_norm.weight",
        nullptr
    };
    
    bool matches_pattern(const std::string& tensor_name, const char* patterns[]) {
        for (int i = 0; patterns[i] != nullptr; i++) {
            if (tensor_name.find(patterns[i]) != std::string::npos) {
                return true;
            }
        }
        return false;
    }
}

ggml_tp_strategy ggml_get_tensor_parallel_strategy(const std::string& tensor_name,
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

bool ggml_tensor_supports_tp(const std::string& tensor_name, const struct ggml_tensor* tensor) {
    // Only support tensor parallelism for 2D weight matrices
    if (ggml_n_dims(tensor) != 2) {
        return false;
    }

    // Skip very small tensors
    if (ggml_nelements(tensor) < 1024) {
        return false;
    }

    // Check if it's a weight tensor (not bias or other parameters)
    return tensor_name.find(".weight") != std::string::npos;
}

ggml_tp_split_info ggml_calculate_tp_split(const struct ggml_tensor* tensor,
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

bool ggml_apply_tensor_parallel_split(struct ggml_tensor* tensor,
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

namespace ggml_tp_utils {
    int64_t get_split_elements(int64_t total_elements, int tp_size, int tp_rank) {
        int64_t base_size = total_elements / tp_size;
        int64_t remainder = total_elements % tp_size;
        
        if (tp_rank < remainder) {
            return base_size + 1;
        } else {
            return base_size;
        }
    }
    
    int64_t get_split_offset(int64_t total_elements, int tp_size, int tp_rank) {
        int64_t base_size = total_elements / tp_size;
        int64_t remainder = total_elements % tp_size;
        
        int64_t offset = tp_rank * base_size;
        if (tp_rank < remainder) {
            offset += tp_rank;
        } else {
            offset += remainder;
        }
        
        return offset;
    }
    
    bool check_tp_compatibility(const struct ggml_tensor* tensor, int tp_size, int split_dim) {
        if (split_dim < 0 || split_dim >= ggml_n_dims(tensor)) {
            return false;
        }

        return tensor->ne[split_dim] % tp_size == 0;
    }
}

ggml_backend_cuda_tp_context::ggml_backend_cuda_tp_context(int tp_size, const std::vector<int>& devices)
    : config(tp_size, 0), device_ids(devices), nccl_initialized(false) {
}

ggml_backend_cuda_tp_context::~ggml_backend_cuda_tp_context() {
    cleanup();
}

bool ggml_backend_cuda_tp_context::init() {
    if (config.tp_size <= 1) {
        return true;
    }

    // Try to initialize NCCL for collective operations
    nccl_initialized = ggml_cuda_nccl_init(device_ids);

    if (!nccl_initialized) {
        GGML_LOG_INFO("NCCL not available, using basic tensor parallelism mode\n");
        GGML_LOG_INFO("Note: Install NCCL for optimized collective operations\n");
        // Continue without NCCL - basic tensor parallelism can still work
    } else {
        GGML_LOG_INFO("NCCL initialized for optimized tensor parallelism\n");
    }

    GGML_LOG_INFO("Tensor parallelism initialized: %d-way TP using GPUs ", config.tp_size);
    for (size_t i = 0; i < device_ids.size(); i++) {
        GGML_LOG_INFO("%d%s", device_ids[i], (i < device_ids.size() - 1) ? "," : "");
    }
    GGML_LOG_INFO("\n");
    return true;
}

void ggml_backend_cuda_tp_context::cleanup() {
    if (nccl_initialized) {
        ggml_cuda_nccl_cleanup();
        nccl_initialized = false;
    }
}



bool ggml_cuda_tp_available() {
    return g_cuda_tp_ctx != nullptr && g_cuda_tp_ctx->config.enabled;
}

const ggml_tp_config& ggml_cuda_tp_get_config() {
    static ggml_tp_config default_config;
    if (g_cuda_tp_ctx) {
        return g_cuda_tp_ctx->config;
    }
    return default_config;
}

// C interface functions for external linkage
extern "C" {
bool ggml_cuda_tp_init(int tp_size, const int* device_ids, int num_devices) {
    if (tp_size <= 1) {
        return true;
    }

    std::vector<int> device_vec(device_ids, device_ids + num_devices);
    g_cuda_tp_ctx = std::make_unique<ggml_backend_cuda_tp_context>(tp_size, device_vec);
    return g_cuda_tp_ctx->init();
}

void ggml_cuda_tp_cleanup() {
    g_cuda_tp_ctx.reset();
}
}
