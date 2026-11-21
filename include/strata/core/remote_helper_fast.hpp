#pragma once
// STRATA_HELPER_FAST (RemoteExperts, HIP builds only): the helper GPU's transport kernels.  The DeepSeek engine's
// 11-07 helper round trip (src/dsv41/ds4_helper.cu: helper_fetch_kernel / helper_flag_kernel, measured 104.6 -> 68.9
// us a round trip on an MI50) applied to the general engine's helper caches: the helper reads its inputs from
// coherent pinned host memory with one kernel instead of two host-to-device copies, returns its rows by a kernel's
// stores into coherent host rows instead of a device-to-host copy, and raises a host word the CPU polls instead of a
// stream query.  No copy engine (SDMA) is involved, and no read crosses to a peer GPU (push-only: the MI50 reads and
// writes host memory only).  Byte copies: the expert kernels see exactly the bytes the copy path gives them.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strata::core::helper_fast {
/// One launch on `s`: `a_bytes` from `a_src` to `a_dst` and `b_bytes` from `b_src` to `b_dst` (16-byte multiples,
/// 16-byte aligned).  The sources are the device views of coherent host memory, the destinations VRAM.
cudaError_t fetch(const void* a_src, void* a_dst, size_t a_bytes, const void* b_src, void* b_dst, size_t b_bytes,
                  cudaStream_t s);
/// One launch on `s`: `bytes` (a 16-byte multiple) from VRAM `src` to coherent host memory `dst` (its device view).
cudaError_t push(const void* src, void* dst, size_t bytes, cudaStream_t s);
/// One single-thread launch on `s`, after everything before it on `s`: the system-scope fence, then `*flag = value`
/// (the device view of a coherent host word the CPU polls).
cudaError_t flag(uint32_t* flag, uint32_t value, cudaStream_t s);
/// The same with the value taken from the device: `*counter += 1`, then `*flag = *counter` (a graph replays it as it is:
/// no host scalar in its arguments).  The host keeps its own count of the launches in step with `*counter`.
cudaError_t flag_next(uint32_t* flag, uint32_t* counter, cudaStream_t s);
}  // namespace strata::core::helper_fast
