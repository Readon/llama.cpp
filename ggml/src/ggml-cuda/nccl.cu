#include "nccl.cuh"  // Include own header for declarations

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "../ggml-impl.h"  // For GGML_LOG_ERROR

#ifdef GGML_USE_NCCL
#include <nccl.h>

#define NCCL_CHECK(call) do { \
    ncclResult_t result = call; \
    if (result != ncclSuccess) { \
        GGML_LOG_ERROR("NCCL error: %s at %s:%d\n", ncclGetErrorString(result), __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

// NCCL context implementation
struct ggml_cuda_nccl_context {
    std::vector<ncclComm_t> comms;
    std::vector<int> device_ids;
    int tp_size;
    int rank;
    
    ggml_cuda_nccl_context(const std::vector<int>& devices) 
        : device_ids(devices), tp_size(devices.size()), rank(0) {
        comms.resize(tp_size);
    }
    
    ~ggml_cuda_nccl_context() {
        for (auto& comm : comms) {
            if (comm != nullptr) {
                ncclCommDestroy(comm);
            }
        }
    }
    
    bool init_tp_group() {
        // For single-process multi-GPU tensor parallelism, use ncclCommInitAll
        // This is the recommended approach for single-process scenarios
        std::vector<ncclComm_t> temp_comms(tp_size);
        std::vector<int> temp_devices(device_ids.begin(), device_ids.end());

        ncclResult_t result = ncclCommInitAll(temp_comms.data(), tp_size, temp_devices.data());
        if (result != ncclSuccess) {
            GGML_LOG_ERROR("Failed to initialize NCCL communicators: %s\n", ncclGetErrorString(result));
            return false;
        }

        // Copy the communicators to our member variable
        for (int i = 0; i < tp_size; i++) {
            comms[i] = temp_comms[i];
        }

        return true;
    }
    
    int get_rank(int device_id) {
        for (int i = 0; i < tp_size; i++) {
            if (device_ids[i] == device_id) {
                return i;
            }
        }
        return -1;
    }
    
    ncclResult_t all_reduce(const void* sendbuff, void* recvbuff, size_t count, 
                           ncclDataType_t datatype, ncclRedOp_t op, int device_id, cudaStream_t stream) {
        int rank = get_rank(device_id);
        if (rank < 0) {
            return ncclInvalidArgument;
        }
        
        return ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comms[rank], stream);
    }
    
    ncclResult_t all_gather(const void* sendbuff, void* recvbuff, size_t sendcount,
                           ncclDataType_t datatype, int device_id, cudaStream_t stream) {
        int rank = get_rank(device_id);
        if (rank < 0) {
            return ncclInvalidArgument;
        }
        
        return ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comms[rank], stream);
    }
};

// Global NCCL context
static std::unique_ptr<ggml_cuda_nccl_context> g_nccl_ctx;

#endif // GGML_USE_NCCL

// Function implementations - always available, with conditional NCCL support
bool ggml_cuda_nccl_init(const std::vector<int>& device_ids) {
#ifdef GGML_USE_NCCL
    if (device_ids.size() <= 1) {
        return true; // No NCCL needed for single GPU
    }

    g_nccl_ctx = std::make_unique<ggml_cuda_nccl_context>(device_ids);
    return g_nccl_ctx->init_tp_group();
#else
    (void)device_ids;
    return false; // NCCL not available
#endif
}

void ggml_cuda_nccl_cleanup() {
#ifdef GGML_USE_NCCL
    g_nccl_ctx.reset();
#endif
}

bool ggml_cuda_nccl_available() {
#ifdef GGML_USE_NCCL
    return g_nccl_ctx != nullptr && g_nccl_ctx->tp_size > 1;
#else
    return false;
#endif
}

void ggml_cuda_nccl_all_reduce_f32(float* data, size_t count, int device_id, cudaStream_t stream) {
#ifdef GGML_USE_NCCL
    if (ggml_cuda_nccl_available()) {
        g_nccl_ctx->all_reduce(data, data, count, ncclFloat32, ncclSum, device_id, stream);
    }
#else
    (void)data; (void)count; (void)device_id; (void)stream;
#endif
}

void ggml_cuda_nccl_all_reduce_f16(half* data, size_t count, int device_id, cudaStream_t stream) {
#ifdef GGML_USE_NCCL
    if (ggml_cuda_nccl_available()) {
        g_nccl_ctx->all_reduce(data, data, count, ncclFloat16, ncclSum, device_id, stream);
    }
#else
    (void)data; (void)count; (void)device_id; (void)stream;
#endif
}

void ggml_cuda_nccl_all_gather_f32(const float* sendbuff, float* recvbuff, size_t sendcount,
                                   int device_id, cudaStream_t stream) {
#ifdef GGML_USE_NCCL
    if (ggml_cuda_nccl_available()) {
        g_nccl_ctx->all_gather(sendbuff, recvbuff, sendcount, ncclFloat32, device_id, stream);
    }
#else
    (void)sendbuff; (void)recvbuff; (void)sendcount; (void)device_id; (void)stream;
#endif
}

void ggml_cuda_nccl_all_gather_f16(const half* sendbuff, half* recvbuff, size_t sendcount,
                                   int device_id, cudaStream_t stream) {
#ifdef GGML_USE_NCCL
    if (ggml_cuda_nccl_available()) {
        g_nccl_ctx->all_gather(sendbuff, recvbuff, sendcount, ncclFloat16, device_id, stream);
    }
#else
    (void)sendbuff; (void)recvbuff; (void)sendcount; (void)device_id; (void)stream;
#endif
}
