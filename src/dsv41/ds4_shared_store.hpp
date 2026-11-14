// one RAM copy of a GGUF's routed experts shared by several dsv41 engines on one box (STRATA_DS4_SHARED_STORE).
// A file in a tmpfs (or hugetlbfs) directory, mapped MAP_SHARED by every engine; the first engine to find it missing fills
// it (the experts in the layout of the engine's own RAM copy, copy_experts), the others map it read-only.  Linux only: on
// other systems the class only reports "not supported" and the engine fails the option loudly.  Design: docs/dsv41/cards/HX-03.md.
//
// Protocol (flock on the file's descriptor, kept for the engine's life):
//   ready file (header says so, identity matches)  -> LOCK_SH (blocks only while a filler converts its lock), map read-only
//   not ready, LOCK_EX|LOCK_NB succeeds           -> this engine fills (the others see EWOULDBLOCK and wait), publish() writes
//                                                     the header's ready word last, then converts its lock to LOCK_SH
//   not ready, lock busy                           -> a filler is at work: wait (a crashed filler's lock is released by the
//                                                     kernel; the next engine finds ready == 0 and fills again from scratch)
// The segment outlives the engines (the next start attaches at once) unless STRATA_DS4_SHARED_STORE_DROP=1, when the last
// engine out unlinks it.  Remove it by hand otherwise (rm the file; its pages go when the last mapping goes).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__linux__)
#include <cerrno>
#include <chrono>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#endif

namespace strata::ds4 {

class SharedStore {
public:
    enum class Role { kNone, kAttach, kFill };
    struct Hdr {
        char magic[8];
        uint64_t version, ident, total, data_off, data_bytes;
        uint64_t ready;   // written last, by publish()
        uint64_t filler_pid;
        double fill_s;
    };
    static constexpr uint64_t kVersion = 1;
    static constexpr uint64_t kHdrBytes = 2ull << 20;   // the data starts on a 2 MiB boundary (hugetlbfs maps need it)
    static bool supported() {
#if defined(__linux__)
        return true;
#else
        return false;
#endif
    }

    SharedStore() = default;
    SharedStore(const SharedStore&) = delete;
    SharedStore& operator=(const SharedStore&) = delete;
    ~SharedStore() { close_all(); }

    const std::string& path() const { return path_; }
    uint64_t data_bytes() const { return data_bytes_; }
    double filler_seconds() const { return hdr_.fill_s; }
    uint64_t filler_pid() const { return hdr_.filler_pid; }
    // the data area: writable for a filler before publish(), read-only (PROT_READ: a stray store faults) afterwards
    uint8_t* data() const { return map_ == nullptr ? nullptr : map_ + kHdrBytes; }

#if defined(__linux__)
    // the file for `spec` (a directory: <dir>/strata-ds4-experts.<ident>; a path: <spec>.<ident>; a bare name: /dev/shm/<name>.<ident>;
    // "1" / "on": the default name in /dev/shm)
    static std::string resolve(const std::string& spec, uint64_t ident) {
        char hex[24];
        std::snprintf(hex, sizeof hex, "%016llx", (unsigned long long) ident);
        std::string base;
        struct stat sb;
        if (spec == "1" || spec == "on" || spec == "true") base = "/dev/shm/strata-ds4-experts";
        else if (::stat(spec.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) base = spec + (spec.back() == '/' ? "" : "/") + "strata-ds4-experts";
        else if (spec.find('/') != std::string::npos) base = spec;
        else base = "/dev/shm/" + spec;
        return base + "." + hex;
    }

    // Opens (creates) the file and takes a role.  data_bytes: the data area's size (rounded up to 2 MiB).  Waits up to
    // wait_s while another engine fills.  kNone: err says why (nothing is held).  kFill: the caller writes data() and then
    // calls publish() - or abort().
    Role open(const std::string& path, uint64_t ident, uint64_t data_bytes, double wait_s, std::string& err) {
        path_ = path;
        ident_ = ident;
        data_bytes_ = (data_bytes + kHdrBytes - 1) / kHdrBytes * kHdrBytes;
        total_ = kHdrBytes + data_bytes_;
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd_ < 0) {
            err = path + ": cannot open (" + std::strerror(errno) + ")";
            return Role::kNone;
        }
        const auto t0 = std::chrono::steady_clock::now();
        bool said = false;
        for (;;) {
            Hdr h;
            if (peek(h)) {   // ready: share it (blocks only while a filler converts its lock)
                if (::flock(fd_, LOCK_SH) == 0) {
                    if (peek(h)) return attach(h, err) ? Role::kAttach : fail();
                    ::flock(fd_, LOCK_UN);
                }
            } else if (::flock(fd_, LOCK_EX | LOCK_NB) == 0) {
                if (peek(h)) {   // another engine finished between the peek and the lock
                    ::flock(fd_, LOCK_SH);
                    return attach(h, err) ? Role::kAttach : fail();
                }
                return begin_fill(err) ? Role::kFill : fail();
            } else if (errno != EWOULDBLOCK) {
                err = path + ": flock failed (" + std::strerror(errno) + ")";
                return fail();
            }
            const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (!said) {
                std::fprintf(stderr, "ds4: shared store %s: another engine is filling it, waiting\n", path.c_str());
                said = true;
            }
            if (waited > wait_s) {
                err = path + ": still being filled by another engine after " + std::to_string((int) waited) +
                      " s (STRATA_DS4_SHARED_STORE_WAIT sets the limit), or held by an engine of an incompatible build";
                return fail();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    // The fill is complete: the header's ready word is written (last), the mapping becomes read-only, the lock shared.
    bool publish(double fill_seconds, std::string& err) {
        if (role_ != Role::kFill || fd_ < 0) return false;
        hdr_.ready = 1;
        hdr_.fill_s = fill_seconds;
        ::munmap(map_, (size_t) total_);
        map_ = nullptr;
        if (::pwrite(fd_, &hdr_, sizeof hdr_, 0) != (ssize_t) sizeof hdr_) {
            err = path_ + ": cannot write the header (" + std::strerror(errno) + ")";
            return false;
        }
        if (::flock(fd_, LOCK_SH) != 0) {   // (EX -> SH: not atomic, but waiters only ever test ready and then ask for SH)
            err = path_ + ": cannot share the lock (" + std::strerror(errno) + ")";
            return false;
        }
        role_ = Role::kAttach;
        return map_ro(err);
    }

    // The fill failed: free the pages and take the file away.
    void abort() {
        if (fd_ < 0) return;
        if (map_ != nullptr) ::munmap(map_, (size_t) total_);
        map_ = nullptr;
        if (::ftruncate(fd_, 0) != 0) { /* (nothing more to do) */ }
        ::unlink(path_.c_str());
        ::close(fd_);
        fd_ = -1;
        role_ = Role::kNone;
    }

    void set_drop(bool d) { drop_ = d; }

private:
    bool peek(Hdr& h) const {
        if (::pread(fd_, &h, sizeof h, 0) != (ssize_t) sizeof h) return false;
        return std::memcmp(h.magic, "YSD4SHM1", 8) == 0 && h.version == kVersion && h.ident == ident_ && h.total == total_ &&
               h.data_off == kHdrBytes && h.data_bytes == data_bytes_ && h.ready == 1;
    }
    bool attach(const Hdr& h, std::string& err) {
        hdr_ = h;
        role_ = Role::kAttach;
        return map_ro(err);
    }
    bool map_ro(std::string& err) {
        int flags = MAP_SHARED;
        const char* pop = std::getenv("STRATA_DS4_SHARED_STORE_POPULATE");   // page tables built now (default), not on the first token
        if (pop == nullptr || std::atoi(pop) != 0) flags |= MAP_POPULATE;
        void* p = ::mmap(nullptr, (size_t) total_, PROT_READ, flags, fd_, 0);
        if (p == MAP_FAILED) {
            err = path_ + ": cannot map (" + std::strerror(errno) + ")";
            return false;
        }
        map_ = (uint8_t*) p;
        return true;
    }
    bool begin_fill(std::string& err) {
        // stale or half-written contents go; fallocate reserves every page now, so a tmpfs that is too small says ENOSPC here
        // instead of a SIGBUS in the middle of the fill
        if (::ftruncate(fd_, 0) != 0) {
            err = path_ + ": cannot truncate (" + std::strerror(errno) + ")";
            return false;
        }
        if (::fallocate(fd_, 0, 0, (off_t) total_) != 0) {
            if (errno == EOPNOTSUPP || errno == EINVAL) {
                if (::ftruncate(fd_, (off_t) total_) != 0) {
                    err = path_ + ": cannot size (" + std::strerror(errno) + ")";
                    return false;
                }
            } else {
                err = path_ + ": no room for " + std::to_string((unsigned long long) (total_ >> 20)) + " MiB (" + std::strerror(errno) +
                      "); /dev/shm is half the RAM by default: mount -o remount,size=<n>G /dev/shm";
                if (::ftruncate(fd_, 0) != 0) { /* (nothing more to do) */ }
                return false;
            }
        }
        std::memset(&hdr_, 0, sizeof hdr_);
        std::memcpy(hdr_.magic, "YSD4SHM1", 8);
        hdr_.version = kVersion;
        hdr_.ident = ident_;
        hdr_.total = total_;
        hdr_.data_off = kHdrBytes;
        hdr_.data_bytes = data_bytes_;
        hdr_.ready = 0;
        hdr_.filler_pid = (uint64_t) ::getpid();
        if (::pwrite(fd_, &hdr_, sizeof hdr_, 0) != (ssize_t) sizeof hdr_) {
            err = path_ + ": cannot write the header (" + std::strerror(errno) + ")";
            return false;
        }
        void* p = ::mmap(nullptr, (size_t) total_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) {
            err = path_ + ": cannot map for the fill (" + std::strerror(errno) + ")";
            return false;
        }
        map_ = (uint8_t*) p;
        role_ = Role::kFill;
        return true;
    }
    Role fail() {   // (nothing held; a half-made file stays for the next engine to redo)
        if (map_ != nullptr) ::munmap(map_, (size_t) total_);
        map_ = nullptr;
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        role_ = Role::kNone;
        return Role::kNone;
    }
    void close_all() {
        if (map_ != nullptr) ::munmap(map_, (size_t) total_);
        map_ = nullptr;
        if (fd_ >= 0) {
            if (drop_ && role_ == Role::kAttach) {   // the last engine out takes the segment away
                ::flock(fd_, LOCK_UN);
                if (::flock(fd_, LOCK_EX | LOCK_NB) == 0) ::unlink(path_.c_str());
            }
            ::close(fd_);   // (the lock goes with the descriptor)
        }
        fd_ = -1;
    }
#else
public:
    static std::string resolve(const std::string&, uint64_t) { return std::string(); }
    Role open(const std::string&, uint64_t, uint64_t, double, std::string& err) {
        err = "the shared expert store is Linux only";
        return Role::kNone;
    }
    bool publish(double, std::string&) { return false; }
    void abort() {}
    void set_drop(bool) {}

private:
    void close_all() {}
#endif

    std::string path_;
    uint64_t ident_ = 0, data_bytes_ = 0, total_ = 0;
    int fd_ = -1;
    uint8_t* map_ = nullptr;
    Hdr hdr_{};
    Role role_ = Role::kNone;
    bool drop_ = false;
};

}   // namespace strata::ds4
