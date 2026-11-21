// include/strata/kernels/tp_allreduce.hpp - the tensor-parallel all-reduce of the verify window (--tensor-parallel 2).
//
// The host-staged all-reduce of the DeepSeek engine (src/dsv41/ds4_kernels.cu: xfer_put into
// host-mapped rows + a flag, ar_sum_kernel summing in member order), as tp_proof_bench (src/kernels/tp_proof_bench.cpp)
// ported it to run INSIDE a captured graph: the sequence number lives in device memory (a graph replays fixed kernel
// arguments), every wait is bounded (timeout -> the shared error word is raised and the sums are skipped), and the sum
// is taken in member order, so every member holds the same bits.  Measured on 2x RX 7900 XTX: 11.8 us at 20 KiB.
//
// Every member must issue the same calls in the same order (one per verify-window layer half); a member that stops
// early makes the others' waits time out (`tp_ar_failed`).  No peer reads: each member writes only its own host rows.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels {

struct TpArGroup;

/// One group over `devs` (member i = devs[i]; the HIP / CUDA ordinals), rows of up to `max_floats` floats.
/// Allocates host-mapped rows / flags (portable) and a small device counter block per member.  nullptr + err on failure.
TpArGroup* tp_ar_open(const std::vector<int>& devs, size_t max_floats, std::string& err);
/// Member `mi`'s all-reduce of x[0, n) (n % 4 == 0, n <= max_floats) on `stream` (that member's device current);
/// capturable.  Afterwards x holds the members' sum, in member order.  `stamps` (device, 4 words; null: none) gets GPU
/// times in ns: [0] the put starts, [1] this card's partial is out and it starts waiting, [2] every other card's partial
/// is in, [3] the sum is done (STRATA_TP_TIMING: wait for the peer = [2]-[1], transfer = [1]-[0], sum = [3]-[2]).
void tp_ar_call(TpArGroup* g, int mi, float* x, int n, void* stream, unsigned long long* stamps = nullptr);
/// A member's bounded wait gave up (the window's sums are not valid; the group stays failed).
bool tp_ar_failed(const TpArGroup* g);
/// The calls member `mi` has completed (its progress word, host-mapped): for diagnostics.
unsigned tp_ar_progress(const TpArGroup* g, int mi);
int tp_ar_members(const TpArGroup* g);

// ---- cycle 3: a card plans its own experts on the device (no host flag, no mapped-row copy when it has no CPU expert)
/// One group's plan (the host pool's layout: counts | start | dst | tok | pad | ptr | ptr2 | start2, `capx` entries) of
/// the entries THIS card computes: an expert in its VRAM (res_layer >= 0) is planned; one that is not is dropped (its
/// row stays zero: the card holding it adds it in the all-reduce) when `peer_res_layer` is null (card 1, which owns
/// only its VRAM experts) or the peer holds it (peer_res_layer >= 0); an expert in neither card's VRAM is the CPU's:
/// the plan is not made and *skip = 0 (the host plans the step), or without `skip`, an empty plan and *plan_err = 1.
/// Planned: *skip = ring.  1..256 entries.
void tp_resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, const int32_t* peer_res_layer,
                      int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
                      int32_t* plan, long long capx, uint32_t* skip, uint32_t ring, void* stream, uint32_t* plan_err = nullptr);
/// The doorbell for a layer that may be device-planned: planned (*skip == ring): *d_mark = ring, then the ring, and no
/// rows cross; else doorbell_publish_res exactly (ids, and the activations when an id is not in this card's VRAM).
void tp_doorbell(const float* x, const int32_t* ids, const int32_t* d_res, int n_expert, int64_t n, int64_t k, float* x_out,
                 int32_t* ids_out, uint32_t* d_seq, const uint32_t* skip, uint32_t ring, uint32_t* d_mark, void* stream);
/// wait_copy_i32_from_mapped (wait for *flag >= value, then copy n ints) unless *skip == value.
void tp_wait_copy_plan(const uint32_t* flag, uint32_t value, int32_t* dst, const int32_t* src, int64_t n, const uint32_t* skip,
                       void* stream);
/// copy_rows_from_mapped_after (wait for the flag; the hit rows zero, the others from the mapped rows), or every row
/// zero when *skip == value.
void tp_copy_rows(const uint32_t* flag, uint32_t value, float* dst, const float* src, int64_t rows, int64_t width,
                  const int32_t* hit_rows, const int32_t* count, const uint32_t* skip, void* stream);

}  // namespace strata::kernels
