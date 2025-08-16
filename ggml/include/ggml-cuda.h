#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

// tensor parallelism functions
GGML_BACKEND_API bool ggml_cuda_tp_init(int tp_size, const int* device_ids, int num_devices);
GGML_BACKEND_API bool ggml_cuda_multi_tp_init(int num_groups, int gpus_per_group);
GGML_BACKEND_API void ggml_cuda_tp_cleanup(void);
GGML_BACKEND_API bool ggml_cuda_multi_tp_available(void);
GGML_BACKEND_API int ggml_cuda_tp_get_num_groups(void);
GGML_BACKEND_API int ggml_cuda_tp_get_device_id(int group_id, int rank);

// tensor parallelism strategy and splitting functions
typedef enum {
    GGML_TP_STRATEGY_REPLICATE = 0,
    GGML_TP_STRATEGY_COLUMN = 1,
    GGML_TP_STRATEGY_ROW = 2,
    GGML_TP_STRATEGY_AUTO = 3
} ggml_tp_strategy;

typedef struct {
    int tp_size;
    int tp_rank;
    bool enabled;
} ggml_tp_config;

GGML_BACKEND_API const ggml_tp_config* ggml_cuda_tp_get_config_ptr(void);
GGML_BACKEND_API ggml_tp_strategy ggml_get_tensor_parallel_strategy_c(const char* tensor_name, const struct ggml_tensor* tensor, const ggml_tp_config* tp_config);
GGML_BACKEND_API bool ggml_apply_tensor_parallel_split_c(struct ggml_tensor* tensor, const ggml_tp_config* tp_config, ggml_tp_strategy strategy);

// Placeholder NCCL communication functions (not implemented yet)
GGML_BACKEND_API bool ggml_cuda_tp_allreduce_c(void* data, size_t count, int datatype, int group_id);
GGML_BACKEND_API bool ggml_cuda_tp_allgather_c(void* sendbuf, void* recvbuf, size_t count, int datatype, int group_id);
GGML_BACKEND_API bool ggml_cuda_tp_reduce_scatter_c(void* sendbuf, void* recvbuf, size_t count, int datatype, int group_id);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
