// STRATA_HELPER_FAST: the helper GPU's transport kernels (include/strata/core/remote_helper_fast.hpp).  Compiled for
// the helper architectures too (CMakeLists.txt, STRATA_HIP_HELPER_OFFLOAD_OPTIONS), as the helper runs them.
// Plain grid-stride loops over 16-byte words with 256-thread blocks: no warp-size assumption, wave32 and wave64 alike.
#include "strata/core/remote_helper_fast.hpp"

#include <algorithm>

namespace strata::core::helper_fast {
namespace {
// The same shape as DeepSeek's helper_fetch_kernel (8 x 256 threads of uint4 loads): enough loads in flight to cover a
// PCIe read's latency for the few hundred KiB a layer's window moves, without a launch sized by the host.
constexpr int kThreads = 256;
constexpr int kMaxBlocks = 8;

__global__ void fetch_kernel(const uint4* __restrict__ a_src, uint4* __restrict__ a_dst, int64_t a_n,
                             const uint4* __restrict__ b_src, uint4* __restrict__ b_dst, int64_t b_n) {
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < a_n + b_n; i += stride) {
        if (i < a_n) a_dst[i] = a_src[i];
        else b_dst[i - a_n] = b_src[i - a_n];
    }
}

__global__ void push_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, int64_t n) {
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) dst[i] = src[i];
    __threadfence_system();   // (the flag kernel after it fences again; this keeps a fused caller correct too)
}

__global__ void flag_kernel(volatile uint32_t* flag, uint32_t value) {
    // every store of the earlier launches on this stream to coherent host memory is visible before the word is
    __threadfence_system();
    *flag = value;
    __threadfence_system();
}

__global__ void flag_next_kernel(volatile uint32_t* flag, uint32_t* counter) {
    const uint32_t v = *counter + 1;
    *counter = v;
    __threadfence_system();
    *flag = v;
    __threadfence_system();
}

int blocks_for(int64_t words) {
    return (int) std::max<int64_t>(1, std::min<int64_t>(kMaxBlocks, (words + kThreads - 1) / kThreads));
}
}  // namespace

cudaError_t fetch(const void* a_src, void* a_dst, size_t a_bytes, const void* b_src, void* b_dst, size_t b_bytes,
                  cudaStream_t s) {
    const int64_t a = (int64_t) (a_bytes / 16), b = (int64_t) (b_bytes / 16);
    if (a + b <= 0) return cudaSuccess;
    fetch_kernel<<<blocks_for(a + b), kThreads, 0, s>>>((const uint4*) a_src, (uint4*) a_dst, a, (const uint4*) b_src,
                                                        (uint4*) b_dst, b);
    return cudaGetLastError();
}

cudaError_t push(const void* src, void* dst, size_t bytes, cudaStream_t s) {
    const int64_t n = (int64_t) (bytes / 16);
    if (n <= 0) return cudaSuccess;
    push_kernel<<<blocks_for(n), kThreads, 0, s>>>((const uint4*) src, (uint4*) dst, n);
    return cudaGetLastError();
}

cudaError_t flag(uint32_t* f, uint32_t value, cudaStream_t s) {
    flag_kernel<<<1, 1, 0, s>>>(f, value);
    return cudaGetLastError();
}

cudaError_t flag_next(uint32_t* f, uint32_t* counter, cudaStream_t s) {
    flag_next_kernel<<<1, 1, 0, s>>>(f, counter);
    return cudaGetLastError();
}
}  // namespace strata::core::helper_fast
