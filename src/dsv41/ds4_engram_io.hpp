// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_engram_io.hpp - DeepSeek's Engram (n-gram memory) rows read off the decode's critical path.
//
// A token's Engram rows depend only on the token ids, so they are asked for as soon as the window's tokens are known
// and collected only right before the layer that adds them (layer 1's hide behind the embedding and layer 0, layer
// 14's behind layers 0-13).  The reads run on the reader's own threads, each with its own handles to the model's files
// (on Windows the ReadFile calls on one synchronous handle run one at a time), FIFO: the first Engram layer's rows
// first.  A bounded row cache (clock eviction) keeps rows already fetched: a row that recurs (the same n-gram again,
// in the prompt or later) is a copy, not a read.  The rows are the file's bytes either way, so the output is the same.
#pragma once

#include "ds4_os.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::ds4 {

inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// Rows already fetched: key -> `bytes` of payload, at most `cap` rows, clock eviction.  Not thread-safe (the caller locks).
/// every structure is sized at init - the payload one block (its pages touched only as rows land), the keys and
/// marks, and an open-addressing index (linear probing, deletion by backward shift) - so filling the cache during the
/// decode allocates nothing.  (It was an unordered_map node a new row, payload blocks of 4096 rows and doubling key
/// vectors: the main arena growing by brk, and blocks above the mmap threshold mmapped, while the decode ran -
/// docs/dsv41/cards/EB-05.md, mechanism 2.)
class EgRowCache {
public:
    void init(size_t cap, size_t bytes) {
        cap_ = std::min<size_t>(cap, (size_t) 0x7fffffff);
        bytes_ = bytes;
        n_ = 0;
        hand_ = 0;
        if (cap_ == 0 || bytes_ == 0) return;
        data_.reset(new uint8_t[cap_ * bytes_]);   // (no pages until written: rows land into it as they are read)
        keys_.assign(cap_, 0);
        ref_.assign(cap_, 0);
        size_t t = 64;
        while (t < 2 * cap_) t <<= 1;
        ix_key_.assign(t, kEmpty);
        ix_slot_.assign(t, 0);
        mask_ = t - 1;
    }
    size_t capacity() const { return cap_; }
    size_t size() const { return n_; }
    /// The row's payload, or nullptr (a hit marks it recently used).
    const uint8_t* find(uint64_t key) {
        const int64_t s = lookup(key);
        if (s < 0) return nullptr;
        ref_[(size_t) s] = 1;
        return at((uint32_t) s);
    }
    /// Room for key's payload (the caller fills it): a new slot while below capacity, else the clock's victim.
    uint8_t* put(uint64_t key) {
        if (cap_ == 0 || bytes_ == 0 || key == kEmpty) return nullptr;
        const int64_t have = lookup(key);
        if (have >= 0) return at((uint32_t) have);
        uint32_t s;
        if (n_ < cap_) {
            s = (uint32_t) n_++;
        } else {
            while (ref_[hand_]) {   // (a second chance for rows used since the hand last passed)
                ref_[hand_] = 0;
                hand_ = (hand_ + 1) % (uint32_t) cap_;
            }
            s = hand_;
            hand_ = (hand_ + 1) % (uint32_t) cap_;
            erase(keys_[s]);
        }
        keys_[s] = key;
        ref_[s] = 0;
        insert(key, s);
        return at(s);
    }

private:
    static constexpr uint64_t kEmpty = ~uint64_t{0};   // (keys are id * 8 + layer: never this)
    uint8_t* at(uint32_t s) { return data_.get() + (size_t) s * bytes_; }
    size_t home(uint64_t k) const { return (size_t) ((k * 0x9E3779B97F4A7C15ull) >> 17) & mask_; }
    int64_t lookup(uint64_t k) const {
        if (ix_key_.empty()) return -1;
        for (size_t i = home(k);; i = (i + 1) & mask_) {
            if (ix_key_[i] == kEmpty) return -1;
            if (ix_key_[i] == k) return (int64_t) ix_slot_[i];
        }
    }
    void insert(uint64_t k, uint32_t s) {
        size_t i = home(k);
        while (ix_key_[i] != kEmpty) i = (i + 1) & mask_;
        ix_key_[i] = k;
        ix_slot_[i] = s;
    }
    void erase(uint64_t k) {   // (backward shift: the probe chains stay whole, no tombstones)
        size_t i = home(k);
        while (ix_key_[i] != kEmpty && ix_key_[i] != k) i = (i + 1) & mask_;
        if (ix_key_[i] == kEmpty) return;
        for (size_t j = i;;) {
            j = (j + 1) & mask_;
            if (ix_key_[j] == kEmpty) break;
            const size_t h = home(ix_key_[j]);
            // entry j may move into the hole at i unless its home lies cyclically in (i, j]
            const bool stays = i <= j ? (i < h && h <= j) : (i < h || h <= j);
            if (stays) continue;
            ix_key_[i] = ix_key_[j];
            ix_slot_[i] = ix_slot_[j];
            i = j;
        }
        ix_key_[i] = kEmpty;
    }
    size_t cap_ = 0, bytes_ = 0, n_ = 0, mask_ = 0;
    uint32_t hand_ = 0;
    std::unique_ptr<uint8_t[]> data_;
    std::vector<uint64_t> keys_;
    std::vector<uint8_t> ref_;
    std::vector<uint64_t> ix_key_;
    std::vector<uint32_t> ix_slot_;
};

/// One row to fetch: its bytes to dst, its scales to scales.  mx8: n bytes at off of a file, [scale, 32 codes] blocks
/// (a GGUF's MXFP8 row); else n row bytes at off and n2 scale bytes at off2 of file2 (two reads; n2 0: none - a GGUF's
/// Q6_K or IQ4_NL row, their scales inside their n bytes: 210 a 256 values, 18 a 32).
struct EgRead {
    int file = -1;
    uint64_t off = 0;
    uint32_t n = 0;
    uint8_t* dst = nullptr;
    uint8_t* scales = nullptr;
    bool mx8 = false;
    int file2 = -1;
    uint64_t off2 = 0;
    uint32_t n2 = 0;
    uint64_t key = 0;
};

/// A batch of rows (one Engram layer of a window): reads, the other places a row read once goes, and its state.
struct EgBatch {
    std::vector<EgRead> reads;
    std::vector<std::pair<int, std::pair<uint8_t*, uint8_t*>>> dups;   // read i -> another (codes, scales)
    std::vector<uint8_t> ok;                                            // per read: 1 read whole
    std::atomic<int> left{0};                                           // reads not yet done
    std::atomic<int> post{0};                                           // 0 pending, 1 finishing, 2 finished
    int64_t t0 = 0;                                                     // submitted (now_ns())
    std::atomic<int64_t> t1{0};                                         // the last read done
    bool live = false;                                                  // submitted and not yet finished
    void clear() {
        reads.clear();
        dups.clear();
        ok.clear();
        left.store(0);
        post.store(0);
        live = false;
    }
};

/// The reader's threads: each read on whichever is free, in submission order, with that thread's own file handles.
class EgReader {
public:
    ~EgReader() { stop(); }
    /// direct[i]: file i read unbuffered (Windows, a file nothing maps: os::open_unbuffered), else buffered
    void start(const std::vector<std::string>& paths, int threads, std::vector<char> direct = {}) {
        paths_ = paths;
        direct_ = std::move(direct);
        for (int i = 0; i < threads; ++i) th_.emplace_back([this] { loop(); });
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
        th_.clear();
        if (check_n_.load() > 0)
            std::fprintf(stderr, "ds4 Engram check (DS4_EG_CHECK): %lld rows read unbuffered, %lld differed from a buffered read\n",
                         (long long) check_n_.load(), (long long) check_bad_.load());
    }
    int threads() const { return (int) th_.size(); }
    void submit(EgBatch& b) {
        b.ok.assign(b.reads.size(), 0);
        b.post.store(0);
        b.t0 = now_ns();
        b.t1.store(b.t0);
        b.live = true;
        b.left.store((int) b.reads.size(), std::memory_order_release);
        const int n = (int) b.reads.size();
        if (n == 0) return;
        // a token's rows one job each; a prompt chunk's in runs (fewer trips through the queue's lock)
        const int step = std::max(1, std::min(32, n / (int) (th_.size() * 4)));
        int jobs = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (int i = 0; i < n; i += step, ++jobs) q_.push_back({&b, i, std::min(n, i + step)});
        }
        if (jobs >= (int) th_.size()) cv_.notify_all();
        else
            for (int i = 0; i < jobs; ++i) cv_.notify_one();
    }
    /// Until every read of b is done.
    void wait(EgBatch& b) {
        if (b.left.load(std::memory_order_acquire) == 0) return;
        std::unique_lock<std::mutex> lk(done_mu_);
        done_cv_.wait(lk, [&] { return b.left.load(std::memory_order_acquire) == 0; });
    }

private:
    void loop() {
        std::vector<os::fd_t> fds(paths_.size(), os::kBadFd);   // this thread's own handles, opened on first use
        std::vector<os::fd_t> ufds(paths_.size(), os::kBadFd);  // (and its unbuffered ones)
        std::vector<uint8_t> buf, abuf(3 * os::kDirectAlign);
        uint8_t* al = abuf.data() + (os::kDirectAlign - (uintptr_t) abuf.data() % os::kDirectAlign) % os::kDirectAlign;
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return quit_ || !q_.empty(); });
                if (quit_ && q_.empty()) break;
                job = q_.front();
                q_.pop_front();
            }
            EgBatch& b = *job.b;
            for (int i = job.i0; i < job.i1; ++i) b.ok[(size_t) i] = read(fds, ufds, al, b.reads[(size_t) i], buf) ? 1 : 0;
            const int64_t t = now_ns();
            for (int64_t prev = b.t1.load(); t > prev && !b.t1.compare_exchange_weak(prev, t);) {
            }
            if (b.left.fetch_sub(job.i1 - job.i0, std::memory_order_acq_rel) == job.i1 - job.i0) {   // (b is the waiter's again)
                std::lock_guard<std::mutex> lk(done_mu_);
                done_cv_.notify_all();
            }
        }
        for (os::fd_t f : fds) os::close_fd(f);
        for (os::fd_t f : ufds) os::close_fd(f);
    }
    bool read(std::vector<os::fd_t>& fds, std::vector<os::fd_t>& ufds, uint8_t* al, const EgRead& r,
              std::vector<uint8_t>& buf) {
        if ((r.mx8 || r.n2 == 0) && r.file >= 0 && (size_t) r.file < direct_.size() && direct_[(size_t) r.file] &&
            os::direct_span(r.off, r.n) <= 2 * os::kDirectAlign) {
            os::fd_t& u = ufds[(size_t) r.file];
            if (u == os::kBadFd) u = os::open_unbuffered(paths_[(size_t) r.file]);
            if (u != os::kBadFd && os::pread_unbuffered(u, al, r.n, r.off)) {
                const uint8_t* src = al + r.off % os::kDirectAlign;
                static const bool check = std::getenv("DS4_EG_CHECK") != nullptr;   // (each row read buffered too)
                if (check) {
                    std::vector<uint8_t> ref(r.n);
                    os::fd_t& f2 = fds[(size_t) r.file];
                    if (f2 == os::kBadFd) f2 = os::open_ro(paths_[(size_t) r.file]);
                    const bool ok = f2 != os::kBadFd && os::pread_full(f2, ref.data(), r.n, r.off);
                    if (!ok || std::memcmp(ref.data(), src, r.n) != 0) ++check_bad_;
                    ++check_n_;
                }
                if (!r.mx8) {   // (a GGUF's Q6_K / IQ4_NL row: its bytes as they are, the scales inside)
                    std::memcpy(r.dst, src, r.n);
                    return true;
                }
                for (uint32_t k = 0; k < r.n / 33; ++k) {   // [scale, 32 codes] -> codes | scales
                    r.scales[k] = src[(size_t) k * 33];
                    std::memcpy(r.dst + (size_t) k * 32, src + (size_t) k * 33 + 1, 32);
                }
                return true;
            }   // (else buffered, below)
        }
        auto fd = [&](int file) {
            if (file < 0 || file >= (int) fds.size()) return os::kBadFd;
            if (fds[(size_t) file] == os::kBadFd) fds[(size_t) file] = os::open_ro(paths_[(size_t) file]);
            return fds[(size_t) file];
        };
        const os::fd_t f = fd(r.file);
        if (f == os::kBadFd) return false;
        if (!r.mx8) {
            if (!os::pread_full(f, r.dst, r.n, r.off)) return false;
            if (r.n2 == 0) return true;
            const os::fd_t f2 = fd(r.file2);
            return f2 != os::kBadFd && os::pread_full(f2, r.scales, r.n2, r.off2);
        }
        buf.resize(r.n);
        if (!os::pread_full(f, buf.data(), r.n, r.off)) return false;
        for (uint32_t k = 0; k < r.n / 33; ++k) {   // [scale, 32 codes] -> codes | scales
            r.scales[k] = buf[(size_t) k * 33];
            std::memcpy(r.dst + (size_t) k * 32, buf.data() + (size_t) k * 33 + 1, 32);
        }
        return true;
    }

    std::vector<std::string> paths_;
    std::vector<char> direct_;

public:
    std::atomic<int64_t> check_n_{0}, check_bad_{0};   // (DS4_EG_CHECK: rows compared, rows that differed)
    std::vector<std::thread> th_;
    std::mutex mu_;
    std::condition_variable cv_;
    struct Job {
        EgBatch* b = nullptr;
        int i0 = 0, i1 = 0;   // reads [i0, i1)
    };
    std::deque<Job> q_;
    bool quit_ = false;
    std::mutex done_mu_;
    std::condition_variable done_cv_;
};

}  // namespace strata::ds4
