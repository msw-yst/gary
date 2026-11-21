#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::core {
class RemoteExpertOpt;

/// A static, profile-filled expert tier on another CUDA device. CUDA0 keeps all
/// dense weights and state; results return through the existing pinned CPU rows.
class RemoteExperts {
public:
    RemoteExperts() = default;
    ~RemoteExperts();
    RemoteExperts(const RemoteExperts&) = delete;
    RemoteExperts& operator=(const RemoteExperts&) = delete;

    /// Initialise the device before the host expert arena registers
    /// tens of GiB of portable mapped memory with CUDA.
    static bool preflight(int device, double& free_gib, std::string& err);
    bool open(int device, int slots, int64_t layers, int64_t experts,
              const std::vector<std::pair<int32_t, int32_t>>& ranked,
              const ExpertCache& primary, ExpertSource& source,
              std::vector<uint8_t>& claimed, std::string& err, bool auto_size = false);
    void close();
    /// dp pool (shared MI50 pool, opt-in): share this helper's cache with other engine processes.  export_handle() gives
    /// the HIP IPC handle (64 bytes) of the cache arena and its size; pairs() is the cache's (layer, expert) list in
    /// slot order.  open_import() is open() over another process's exported cache: the same pairs in the same slots, no
    /// allocation and no fill (STRATA_POOL_VERIFY slots, default 16, are byte-compared to `source` first); everything
    /// else - staging, stream, flag, graphs, issue thread - is this process's own, as with open().  HIP builds only.
    bool export_handle(void* handle64, uint64_t& bytes, std::string& err) const;
    bool open_import(int device, const void* handle64, uint64_t bytes, const std::vector<std::pair<int32_t, int32_t>>& pairs,
                     int64_t layers, int64_t experts, const ExpertCache& primary, ExpertSource& source,
                     std::vector<uint8_t>& claimed, std::string& err);
    const std::vector<std::pair<int32_t, int32_t>>& pairs() const { return selected_; }
    /// dp pool: a device's PCI bus id ("0000:c3:00.0", lower case) - ordinals differ between processes, bus ids do not
    static std::string bus_id(int device);
    bool imported() const { return imported_ != nullptr; }

    /// `kind` is the primary verifier's classification (-1 = CPU candidate),
    /// or null on the one-token path. Entries already served on CUDA0 are excluded.
    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
               int64_t k, const int32_t* kind, const int32_t* primary_res,
               std::string& err);
    bool owns(int64_t index) const { return owned_[(size_t) index] != 0; }
    /// begin() in two halves (cycle 6): claim() classifies the rows (host only, no device call: owns() is valid after
    /// it), launch() stages them and queues the device work.  begin() = claim() + launch().
    bool claim(int64_t layer, const int32_t* ids, int64_t n_tok, int64_t k, const int32_t* kind,
               const int32_t* primary_res, std::string& err);
    bool launch(int64_t layer, const float* x, int64_t n_tok, std::string& err);
    /// claim() gave this helper rows of the current request
    bool has_work() const { return !group_id_.empty(); }
    /// STRATA_HELPER_THREADS=1 (cycle 6; HIP builds, not with --remote-expert-opt): launch() + finish() on this helper's
    /// own thread, so the helpers are issued and collected concurrently.  post() after claim() hands the request to the
    /// thread (x and out must stay valid until join()); join() waits for it.  The thread writes only the rows this
    /// helper owns (finish), which the caller must neither write nor zero meanwhile.
    static bool threads_requested();
    bool threaded() const { return threaded_; }
    void post(int64_t layer, const float* x, int64_t n_tok, float* out);
    bool join(std::string& err);
    /// How this helper's requests move (the startup line): "copies", "zero-copy", or STRATA_HELPER_FAST's
    /// "kernel fetch + push + flag" (1) / "kernel fetch + rows to host + flag" (2), "+ graph" with STRATA_HELPER_GRAPH.
    std::string transport() const;
    /// STRATA_HELPER_GRAPH: requests replayed from a captured graph / captured / issued directly (cumulative)
    int64_t graph_replays() const { return graph_replays_; }
    int64_t graph_captures() const { return graph_captures_; }
    int64_t graph_direct() const { return graph_direct_; }
    /// This helper's cache holds (layer, expert): begin() will take its rows unless the plan gave them away.
    bool holds(int64_t layer, int32_t expert) const { return cache_.slot_of(layer, expert) >= 0; }
    bool optimized_decode() const { return remote_opt_ != nullptr; }
    bool finish(float* out, std::string& err);
    int64_t resident() const { return cache_.resident(); }
    int64_t computed() const { return computed_; }
    int64_t launched_layers() const { return launched_layers_; }
    double gib() const { return cache_.gib(); }
    /// open(): ranked candidates passed over because CUDA0, a split stage or another helper already held them
    int64_t skipped_held() const { return skipped_held_; }
    uint64_t returned_bytes() const { return returned_bytes_; }
    uint64_t full_row_bytes() const { return full_row_bytes_; }
    /// host time spent in begin() (staging + launches) and in finish() (waiting for this GPU), cumulative
    double ms_begin() const { return ms_begin_; }
    double ms_wait() const { return ms_wait_; }
    /// STRATA_HELPER_TIMING=1 (opt-in; cycle 5): this GPU's own clock per launched layer, cumulative - the input and group
    /// copies in, the kernels, the result copy out (stream events, read in finish() after the stream has drained) - and
    /// the host's whole round trip per launched layer (begin() entry -> finish() return).  Zero when off.
    double ms_gpu_in() const { return ms_gpu_in_; }
    double ms_gpu_kernels() const { return ms_gpu_kernels_; }
    double ms_gpu_out() const { return ms_gpu_out_; }
    double ms_round_trip() const { return ms_round_trip_; }
    int64_t groups_launched() const { return groups_launched_; }

    /// STRATA_PREFILL_HELPER_SOURCE (opt-in): copy up to `n` cached blobs into the host buffers `dst[i]`
    /// on this helper's own stream and wait.  `hit[i]` is set where this helper held (layer, expert); the
    /// bytes are the ones open() filled from the same source, so the prompt path gets identical bytes.
    /// Misses are left to the caller.  False with `err` on a CUDA failure; the hits copied so far are
    /// complete (the stream is drained), so the caller falls back for the rest.  Safe from several threads
    /// (one stream, one lock): the prompt path's stager is threaded.
    bool copy_cached_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n,
                           uint8_t* hit, std::string& err);

private:
    friend class RemoteExpertOpt;
    RemoteExpertOpt* remote_opt_ = nullptr;
    int device_ = -1;
    int64_t n_expert_ = 0;
    int32_t groups_ = 0;
    int64_t computed_ = 0;
    int64_t launched_layers_ = 0;
    uint64_t returned_bytes_ = 0;
    uint64_t full_row_bytes_ = 0;
    double ms_begin_ = 0, ms_wait_ = 0;
    double ms_gpu_in_ = 0, ms_gpu_kernels_ = 0, ms_gpu_out_ = 0, ms_round_trip_ = 0;
    int64_t groups_launched_ = 0;
    bool timing_ = false;                   ///< STRATA_HELPER_TIMING=1: the events below are recorded
    cudaEvent_t ev_[4] = {nullptr, nullptr, nullptr, nullptr};   ///< start, inputs in, kernels done, results out
    std::chrono::steady_clock::time_point rt0_{};                 ///< begin() entry of the launched request
    int64_t skipped_held_ = 0;
    /// begin() launched work that finish() has not waited for yet.  Every caller pairs them inside one pool call; a
    /// failure between the two (the pool's early returns) leaves it set, and the next begin() drains the stream before
    /// it rewrites the staging that work may still read (the pipelined windows rely on one request at a time).
    bool pending_ = false;
    /// The device sequence of one request on `s` (inputs in, kernels, results out, the flag); also what a graph captures
    /// (`capturing`: no timing events).
    bool enqueue(int64_t layer, int64_t n_tok, bool reduce, bool capturing, std::string& err);
    /// STRATA_HELPER_FAST (HIP only; 0 = off, the copies above): 1 = one fetch kernel reads x and the metadata from
    /// coherent host staging into VRAM (no host-to-device copies), the expert kernels write VRAM, a push kernel stores
    /// the compact rows into the coherent host rows, a one-thread kernel raises the host flag word, and the CPU polls
    /// that word (no device-to-host copy, no stream query, no device switch in the wait).  2 = the same, but the expert
    /// kernels store their rows straight into the coherent host rows (no push kernel).  Push-only: the helper reads and
    /// writes host memory, never a peer GPU's.
    int fast_ = 0;
    uint32_t* h_flag_ = nullptr;   ///< one coherent mapped host word (64 bytes)
    uint32_t* z_flag_ = nullptr;   ///< h_flag_ as the helper GPU sees it
    uint32_t* d_seq_ = nullptr;    ///< the device's count of raised flags (flag_next)
    uint32_t seq_ = 0;             ///< the host's count of queued flags: the request is done once *h_flag_ == seq_
    void* z_meta_ = nullptr;       ///< h_meta_ as the helper GPU sees it
    size_t meta_copy_ = 0;         ///< bytes the fetch kernel moves of the metadata (16-byte multiple)
    bool flag_seen() const;
    bool wait_flag(std::string& err);
    /// STRATA_HELPER_GRAPH=1 (with STRATA_HELPER_FAST, not with STRATA_HELPER_TIMING): each (expert format, rows, groups,
    /// entries) sequence is issued directly the first time, captured into a graph the second, replayed after that.  The
    /// graph's launches are the direct ones (same kernels, grids and buffers; the flag's value comes from d_seq_).
    bool graph_ = false;
    std::unordered_map<uint64_t, std::pair<int, cudaGraphExec_t>> graphs_;   ///< key -> (times seen, exec or null)
    std::vector<uint64_t> fmt_keys_;   ///< the distinct expert formats seen (their index is the key's format part)
    int64_t graph_replays_ = 0, graph_captures_ = 0, graph_direct_ = 0;
    /// STRATA_HELPER_THREADS: the helper's thread and its doorbell
    bool threaded_ = false;
    std::thread worker_;
    std::atomic<uint64_t> posted_{0}, done_{0};
    std::atomic<bool> stop_{false}, sleeping_{false};
    std::mutex wake_mu_;
    std::condition_variable wake_cv_;
    int64_t job_layer_ = 0, job_ntok_ = 0;
    const float* job_x_ = nullptr;
    float* job_out_ = nullptr;
    bool job_ok_ = true;
    std::string job_err_;
    void worker_loop(int cpu);
    void stop_worker();
    ExpertCache cache_;
    std::vector<std::pair<int32_t, int32_t>> selected_;   ///< the cache's pairs in slot order (dp pool export)
    const void* import_handle_ = nullptr;   ///< open_import() in progress: the handle and size to map instead of filling
    uint64_t import_bytes_ = 0;
    void* imported_ = nullptr;              ///< the mapped IPC arena (closed in close())
    cudaStream_t stream_ = nullptr;
    cudaStream_t read_stream_ = nullptr;   ///< STRATA_PREFILL_HELPER_SOURCE: cache -> host copies
    std::mutex read_mu_;                   ///< one cache reader at a time (the stager is threaded)
    float* h_x_ = nullptr;
    float* h_out_ = nullptr;
    void* h_meta_ = nullptr;
    float* d_x_ = nullptr;
    float* z_x_ = nullptr;     ///< h_x_ as the helper GPU sees it (zero-copy: no input copy per layer)
    float* z_out_ = nullptr;   ///< h_out_ as the helper GPU sees it (zero-copy: no result copy)
    bool zero_copy_ = false;
    float* d_out_ = nullptr;
    uint8_t* d_q8_ = nullptr;
    float* d_scales_ = nullptr;
    float* d_xr_ = nullptr;                 // APR: the Hadamard-rotated expert input (CAP x H)
    const float* d_sx_ = nullptr;           // APR: device sign vectors (x: H, h: FF); nullptr = identity
    const float* d_sh_ = nullptr;
    void* d_scratch_ = nullptr;
    void* d_meta_ = nullptr;  ///< one contiguous upload of grouped indices, instead of five small copies
    int32_t* d_start_ = nullptr;
    int32_t* d_dst_ = nullptr;
    int32_t* d_tok_ = nullptr;
    int32_t* d_count_ = nullptr;
    unsigned long long* d_ptr_ = nullptr;
    std::vector<uint8_t> owned_;
    std::vector<uint8_t> layers_present_;
    std::vector<int32_t> group_of_, group_id_;
    std::vector<int32_t> start_, dst_, tok_, original_row_;
    std::vector<unsigned long long> ptr_;
};

/// STRATA_PREFILL_HELPER_SOURCE (opt-in): the prompt path's view of the expert tiers, where a blob a helper
/// GPU already holds in its VRAM cache is copied from that cache instead of being read from the files again.
/// Everything else is forwarded to `fallback` unchanged, so its batched unbuffered reads, its pinned RAM copy
/// and the prompt path's own decisions keep their meaning.  Nothing is cached here: every call asks the
/// helpers, so an entry an adaptive tier moves between requests is a miss (a file read), never a wrong blob.
///
/// The serve loop builds this only for a serial request path (no batch slots, no layer split), while the
/// helpers are idle: during a prompt read the helper GPUs compute nothing, so their caches can be read
/// without ordering the copies against their kernel streams.
class RemotePrefillSource final : public ExpertSource {
public:
    RemotePrefillSource(ExpertSource& fallback, const std::vector<RemoteExperts*>& helpers)
        : fallback_(fallback), helpers_(helpers) {}

    /// Whether any helper's VRAM cache holds (layer, expert) right now.
    bool from_helper(int64_t layer, int64_t expert) const {
        for (const RemoteExperts* h : helpers_) if (h->holds(layer, (int32_t) expert)) return true;
        return false;
    }
    /// Blobs / bytes this source has taken from the helpers so far (the per-request evidence line).
    uint64_t helper_blobs() const { return helper_blobs_.load(std::memory_order_relaxed); }
    uint64_t helper_bytes() const { return helper_bytes_.load(std::memory_order_relaxed); }

    const uint8_t* blob(int64_t layer, int64_t expert) override { return fallback_.blob(layer, expert); }
    const uint8_t* blob_stable(int64_t layer, int64_t expert) override {
        return fallback_.blob_stable(layer, expert);
    }
    int blob_parts(int64_t layer, int64_t expert, BlobPart* out) override {
        return from_helper(layer, expert) ? 0 : fallback_.blob_parts(layer, expert, out);
    }
    int64_t reads() const override { return fallback_.reads(); }
    bool pinned(int64_t layer, int64_t expert) const override {
        return from_helper(layer, expert) ? false : fallback_.pinned(layer, expert);
    }
    void begin_layer(int64_t layer, const int32_t* ids, int64_t k) override {
        fallback_.begin_layer(layer, ids, k);
    }
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override {
        return from_helper(layer, expert) ? nullptr : fallback_.device_alias(layer, expert);
    }
    void prefetch(int64_t layer, int64_t expert) override { fallback_.prefetch(layer, expert); }
    uint64_t release(int64_t layer, int64_t expert) override {
        return from_helper(layer, expert) ? 0 : fallback_.release(layer, expert);
    }
    bool pcie_layer(int64_t layer) const override { return fallback_.pcie_layer(layer); }
    /// True where a helper holds the blob: the prompt path's stager must then take the copy_blob(s) route,
    /// which is where the cache-to-host copy happens.  Everywhere else the fallback answers as it always did.
    bool transient(int64_t layer, int64_t expert) const override {
        return from_helper(layer, expert) ? true : fallback_.transient(layer, expert);
    }
    double cached_share(int64_t samples) const override { return fallback_.cached_share(samples); }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    bool copy_blobs(const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n) override;
    void prefetch(int64_t layer, const int64_t* experts, int64_t n) override {
        fallback_.prefetch(layer, experts, n);
    }
    void warm(int64_t layer, const int64_t* experts, int64_t n) override {
        fallback_.warm(layer, experts, n);
    }
    bool warms() const override { return fallback_.warms(); }
    bool advise_pairs(const std::pair<int32_t, int32_t>* pairs, int64_t n) const override {
        return fallback_.advise_pairs(pairs, n);
    }

private:
    void warn(const std::string& err);
    ExpertSource& fallback_;
    std::vector<RemoteExperts*> helpers_;
    std::atomic<uint64_t> helper_blobs_{0}, helper_bytes_{0};
    std::atomic<bool> warned_{false};
};
} // namespace strata::core
