// Strata prototype (docs/dsv41/cards/HX-03.md). src/dsv41/ds4_cpuslots.hpp - STRATA_CPU_SLOTS=N for strata-dsv41, the same
// semantics and the same slot files as the main engine's block in src/program/generate.cpp (a Qwen engine and a DeepSeek
// engine on one box share the slot set).
//
// Several engines on one host each put their host thread on the first CPU they may use - CPU 0 for all of them - and
// halve each other (docs/dsv41/cards/HX-03.md).  STRATA_CPU_SLOTS=N splits the physical cores N ways: this
// engine claims the lowest free slot (a flock on /dev/shm/strata-cpu-slot-k, held for life; STRATA_CPU_SLOT=i picks one),
// narrows its affinity to that slice's cores + their SMT siblings (so the host thread lands on the slice's first core), and,
// when the engine serves its helpers (STRATA_POOL_SERVE=dir[,dir]), puts one pool-server core per directory at the end of
// the slice: STRATA_POOL_CORE, unless set already.  The CPU lane's workers (ds4_model.cu node_cores) then take the slice's
// cores minus the first (the lane's own thread) and minus the server cores.  Call before any thread is created.
#pragma once

#if defined(__linux__)
#include <sched.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::ds4 {

inline void apply_cpu_slots() {
    const char* ns = std::getenv("STRATA_CPU_SLOTS");
    if (!ns || std::atoi(ns) <= 1) return;
    const int n_slots = std::atoi(ns);
    int slot = -1;
    if (const char* s = std::getenv("STRATA_CPU_SLOT"); s && s[0]) slot = std::atoi(s);
    for (int k = 0; slot < 0 && k < n_slots; ++k) {
        const std::string p = "/dev/shm/strata-cpu-slot-" + std::to_string(k);
        const int fd = ::open(p.c_str(), O_RDWR | O_CREAT, 0666);
        if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0) slot = k;   // the fd stays open: the lock lives as long as we do
        else if (fd >= 0) ::close(fd);
    }
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    sched_getaffinity(0, sizeof allowed, &allowed);
    auto sibling_list = [](int cpu) {
        std::vector<int> out;
        const std::string p = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list";
        if (std::FILE* f = std::fopen(p.c_str(), "r")) {
            char buf[128] = {0};
            if (std::fgets(buf, sizeof buf, f)) {
                std::string s = buf;
                size_t i = 0;
                while (i < s.size()) {   // "a,b" or "a-b"
                    size_t j = i;
                    while (j < s.size() && std::isdigit((unsigned char) s[j])) ++j;
                    if (j == i) { ++i; continue; }
                    const int a = std::atoi(s.substr(i, j - i).c_str());
                    if (j < s.size() && s[j] == '-') {
                        size_t k = j + 1;
                        while (k < s.size() && std::isdigit((unsigned char) s[k])) ++k;
                        for (int c = a; c <= std::atoi(s.substr(j + 1, k - j - 1).c_str()); ++c) out.push_back(c);
                        i = k;
                    } else {
                        out.push_back(a);
                        i = j;
                    }
                }
            }
            std::fclose(f);
        }
        return out;
    };
    std::vector<int> phys;   // one CPU per physical core: the lowest-numbered hardware thread, in CPU order
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &allowed)) continue;
        const auto sib = sibling_list(c);
        if (sib.empty() || *std::min_element(sib.begin(), sib.end()) == c) phys.push_back(c);
    }
    int servers = 0;
    if (const char* sv = std::getenv("STRATA_POOL_SERVE"); sv && sv[0]) {
        servers = 1;
        for (const char* q = sv; *q; ++q) servers += *q == ',';
    }
    const int per = (int) phys.size() / n_slots;
    if (slot < 0 || slot >= n_slots || per < servers + 2) {
        std::fprintf(stderr, "strata-dsv41: STRATA_CPU_SLOTS=%d: %s; the default placement stays\n", n_slots,
                     slot < 0 || slot >= n_slots ? "no free slot" : "too few physical cores per slot");
        return;
    }
    const int lo = slot * per, hi = slot == n_slots - 1 ? (int) phys.size() : lo + per;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    std::string pool, srv, all;
    for (int i = lo; i < hi; ++i) {
        for (int c : sibling_list(phys[(size_t) i]))
            if (CPU_ISSET(c, &allowed)) CPU_SET(c, &mask);
        if (sibling_list(phys[(size_t) i]).empty()) CPU_SET(phys[(size_t) i], &mask);
        const std::string c = std::to_string(phys[(size_t) i]);
        all += (all.empty() ? "" : ",") + c;
        if (i == lo) continue;                                  // the host (lane) thread's core
        if (i >= hi - servers) srv += (srv.empty() ? "" : ",") + c;
        else pool += (pool.empty() ? "" : ",") + c;
    }
    if (sched_setaffinity(0, sizeof mask, &mask) != 0) {
        std::fprintf(stderr, "strata-dsv41: STRATA_CPU_SLOTS: sched_setaffinity failed; the default placement stays\n");
        return;
    }
    if (servers > 0 && !std::getenv("STRATA_POOL_CORE")) setenv("STRATA_POOL_CORE", srv.c_str(), 1);
    std::fprintf(stderr, "strata-dsv41: CPU slot %d of %d: physical cores %s (+ SMT siblings); host %d, lane workers %s%s%s\n", slot,
                 n_slots, all.c_str(), phys[(size_t) lo], pool.c_str(), servers > 0 ? ", pool servers " : "",
                 servers > 0 ? std::getenv("STRATA_POOL_CORE") : "");
}

}  // namespace strata::ds4
#else
namespace strata::ds4 {
inline void apply_cpu_slots() {}
}  // namespace strata::ds4
#endif
