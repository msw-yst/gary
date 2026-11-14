// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// src/ds4/ds4_main.cpp - maya-ds4: DeepSeek-V4.1-Flash on Maya.
//
//   maya-ds4 --pack <model>/maya-ds4 --ids 0,128803,... --n 16          greedy tokens after the ids
//   maya-ds4 --pack ... --ids ... --dump-logits out.f32                every position's logits (the oracle's check)
//   maya-ds4 --pack ... --serve                                       Maya's engine protocol on stdin / stdout
//
// --serve speaks what serve/server.py's StrataEngine expects: INFO lines, then `READY <context> stop`; per request
// `GEN <max_new> [key=value ...] <id,id,...>` answered by `T <id>` lines and `DONE <generated> <prompt> <prompt ms>
// <decode ms> <stop|length|cancel> 0 0 <reused> 0 0`; `STOP` ends a request early, `QUIT` the engine.  A prompt that
// continues the state (the conversation so far) reuses it; one that only shares the last prompt's start goes back to
// that prompt's end; anything else starts over.  `GENI <max_new> [keys] <file> <ids>` carries the prompt's pictures
// (the file: see read_pics); the state is shared only where the same picture sits at the same place.  With
// STRATA_DS4_SLOT_KEEP=1 the conversations are kept when another one takes the model (a slot each: the state, the
// compressed KV included), and across runs in STRATA_DS4_SLOT_DIR - so a request that does not continue the state but
// does continue a kept conversation reads only its new tokens.
// `pin=N` on a GEN marks the first N prompt tokens as a start other requests share (a system prompt and its tools): the
// read stops there for a checkpoint, so a request behind the same start - another agent - goes back to it instead of
// reading it again (also without slots).  STRATA_DS4_REUSE=0 turns every reuse off (each request reads its whole prompt).
#if defined(__APPLE__)
#include <execinfo.h>
#include <csignal>
#include <unistd.h>
#include <libproc.h>   // proc_pid_rusage: the bytes this process had the storage read (io_read_bytes)
#endif
#include "ds4_launch.h"   // ds4_dbg
#include "ds4_model.hpp"
#include "ds4_cpuslots.hpp"
#include "ds4_experts_rule.hpp"
#include "ds4_os.hpp"
#include "ds4_packinfo.hpp"
#include "ds4_slotmeta.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <atomic>
#include <chrono>
#include <cctype>
#include <climits>
#include <condition_variable>
#include <csignal>
#include <filesystem>
#include <memory>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// the bytes this process had the storage read for it (Linux's /proc/self/io, macOS's rusage; -1 where there is none): the
// Monitor's disk reads a token, the experts read in place from the model's files included
long long io_read_bytes() {
#ifdef __linux__
    std::FILE* f = std::fopen("/proc/self/io", "r");
    if (f == nullptr) return -1;
    char line[128];
    long long v = -1;
    while (std::fgets(line, sizeof line, f) != nullptr)
        if (std::sscanf(line, "read_bytes: %lld", &v) == 1) break;
    std::fclose(f);
    return v;
#elif defined(__APPLE__)
    // (the same number `iotop`/Activity Monitor's "Bytes read" shows: every read the storage did for this process, the
    // page cache's misses of the experts read in place included; a page-cache hit counts no storage read, which is what
    // the 60 tps go/no-go needs)
    struct rusage_info_v4 ri;
    std::memset(&ri, 0, sizeof ri);
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t*) &ri) != 0) return -1;
    return (long long) ri.ri_diskio_bytesread;
#else
    return -1;
#endif
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    const char* p = s.c_str();
    while (*p) {
        char* e = nullptr;
        const long x = std::strtol(p, &e, 10);
        if (e == p) break;
        v.push_back((int) x);
        p = e;
        while (*p == ',' || *p == ' ') ++p;
    }
    return v;
}

void usage() {
    std::fprintf(stderr,
                 "maya-ds4 --pack <model>/maya-ds4 [--gpus 0,1,...] [--max-context N] [--reserve-mib N]\n"
                 "           [--vram-experts F] [--cpu-threads N] [--expert-profile FILE|none] [--profile-readonly] [--ssd-cache GB] [--ssd-policy lru|lfu|lfulru] [--no-expert-cache] [--verbose]\n"
                 "           [--tp N] [--threads on|off] [--prefill-chunk N] [--dspark on|off|none|auto] [--dspark-block N] [--dspark-conf F] [--experts N|auto]\n"
                 "           [--vision on|off] [--mmap-experts on|off|auto] [--helper-device N[,N...]] [--helper-slots N]\n"
                 "           (--ids I,J,... [--images FILE] [--n N] [--dump-logits FILE] | --serve\n"
                 "           | --encode-image FILE.rgb --size WxH --dump-image OUT\n"
                 "           | --batch-prompts F1,F2,... [--batch-sizes 1,2,4] [--batch-rows 1,2,3,4 | --batch-spec] --n N   (DS4_BATCH_DECODE=N; --batch-spec: DS4_BATCH_DSPARK=1)\n"
                 "           | --pack-info)\n");
}

// stdin lines on a thread, so a request can see STOP while it decodes
struct Lines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;
    std::atomic<bool> stop{false};
    // srvbatch: `BSTOP <bank>` ends that bank at its next batch step - flagged here, so it is seen while a prompt is
    // read between steps too (the serve loop clears a bank's flag when it admits a new request into it)
    std::atomic<bool> bstop[16] = {};
    void start() {
        std::thread([this] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                std::lock_guard<std::mutex> lk(mu);
                if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                    stop.store(true);
                    continue;
                }
                if (l.rfind("BSTOP ", 0) == 0) {
                    const int b = std::atoi(l.c_str() + 6);
                    if (b >= 0 && b < 16) bstop[b].store(true);
                    continue;
                }
                if (l == "QUIT") stop.store(true);
                q.push_back(l);
                cv.notify_all();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            stop.store(true);
            cv.notify_all();
        }).detach();
    }
    bool next(std::string& l) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        l = q.front();
        q.pop_front();
        return true;
    }
    // srvbatch: the next line if one is waiting (the serve loop between batch steps); closed() then says why not
    bool try_next(std::string& l) {
        std::lock_guard<std::mutex> lk(mu);
        if (q.empty()) return false;
        l = q.front();
        q.pop_front();
        return true;
    }
    bool closed() {
        std::lock_guard<std::mutex> lk(mu);
        return eof && q.empty();
    }
};

// The drafts between two stats: each draft position (1st, 2nd, ..) - how often it was verified with every draft
// before it kept, and how often it was kept then
std::string per_position(const strata::ds4::Stats& s0, const strata::ds4::Stats& s1) {
    std::string pp = "per position (kept / reached):";
    for (int i = 0; i < 8; ++i) {
        const long long r = s1.draft_reached[i] - s0.draft_reached[i], k = s1.draft_kept[i] - s0.draft_kept[i];
        if (r <= 0) break;
        char b[64];
        std::snprintf(b, sizeof b, " %d: %lld/%lld %.0f%%", i + 1, k, r, 100.0 * (double) k / (double) r);
        pp += b;
    }
    return pp;
}

static std::FILE* g_round_log = nullptr;
static std::string g_round_tag;
static int g_spec_depth = 64;   // --dspark-depth: at most this many drafts a round
static std::vector<float> g_round_conf;
static int g_round_nd = 0;

// The drafts' cut on the product of their confidences (DeepInfer's cumulative rule): a draft is verified while the chance
// that it and every draft before it are kept stays at least this - a long window whose later drafts are likely lost is
// not verified whole.  STRATA_DS4_DSPARK_CUM=F, a request's dspark_cum key; 0: off, only the floor on each draft's own
// confidence (dspark_conf).  0.5 is the default on gfx1200 / gfx1201 (RDNA4: R9700, RX 9000) only: measured on 4x R9700, DeepSeek 2.25 bpw,
// six 128-token decodes an arm in one load - 71.7 -> 76.9 tok/s (medians), the same ids.  Project Maya ran 0.5 on a Radeon 8065S (Strix Halo,
// gfx1151: +6.5% decode, the same tokens), but that is a single datapoint on a box the 0.2.7 defaults have not been tested on, so
// Strix keeps 0 until it is; the CUDA measurements in 0.2.7 had it flat, so CUDA keeps 0.  STRATA_DS4_DSPARK_CUM overrides everywhere.
float dspark_cum_default(const strata::ds4::Model& m) {
    static const char* const e = std::getenv("STRATA_DS4_DSPARK_CUM");
    return strata::ds4::dspark_cum_value(e != nullptr, e != nullptr ? (float) std::atof(e) : 0.0f, m.gpus_are_gfx12(),
                                         strata::ds4::defaults_026());
}

// One round of decoding after `tok` (the token at pos(), not fed yet): with the drafter, its drafts while their
// confidence holds (at most `room` - 1 of them), verified together; else a plain step.  `out` gets the tokens that follow
// tok (the accepted drafts, then the trunk's next), the last of them the next round's tok.
struct RoundTimes {   // DS4_ROUND_TIMES: where a decode round's time goes (ms, summed), printed at exit
    double draft = 0, verify = 0, step = 0;
    unsigned long long draft_d = 0, verify_d = 0, step_d = 0;
    long rounds = 0, spec_rounds = 0, steps = 0;
    ~RoundTimes() {
        if (ds4_dbg("DS4_ROUND_TIMES") && rounds > 0)
            std::fprintf(stderr, "rounds: %ld (%ld speculative, %ld plain steps): draft %.1f ms (%.2f a round), verify %.1f ms (%.2f), plain step %.1f ms (%.2f)\n",
                         rounds, spec_rounds, steps, draft, spec_rounds ? draft / (double) spec_rounds : 0.0, verify, spec_rounds ? verify / (double) spec_rounds : 0.0,
                         step, steps ? step / (double) steps : 0.0);
        if (ds4_dbg("DS4_ROUND_TIMES") && rounds > 0)
            std::fprintf(stderr, "dispatches a round: draft %.0f, verify %.0f, plain step %.0f\n", spec_rounds ? (double) draft_d / (double) spec_rounds : 0.0, spec_rounds ? (double) verify_d / (double) spec_rounds : 0.0, steps ? (double) step_d / (double) steps : 0.0);
    }
} g_rt;
#if defined(STRATA_METAL)
extern "C" unsigned long long ds4_dispatch_seq();
#endif
#if defined(STRATA_METAL)
extern "C" void ds4_timeline_gate(int);
#endif
// DS4_DSPARK_ADAPT=1 (drafter/sweep.sh; default off: round() is as before, byte for byte): the drafts a
// round are chosen by their expected worth instead of the fixed floor + block alone.  Draft i is verified while the chance
// that it and every draft before it are kept, cum_i = prod_{j<=i} p(j, conf_j), stays at least theta = row_ms / tok_ms: an
// extra verify row costs row_ms (MEASURED 3.1 ms on tp4h4) and pays off only if it is kept with a chance worth that time
// at the current ms a token.  p(j, c) is the kept rate of position j's drafts with confidence c (10 buckets), learned
// online from the verify verdicts (each cell an exponential memory, DS4_DSPARK_ADAPT_DECAY), with the confidence itself
// as the prior (DS4_DSPARK_ADAPT_PRIOR pseudo-counts).  tok_ms is an EWMA of the rounds' own ms a token.  The conf floor
// (--dspark-conf / dspark_conf), room and --dspark-depth still apply; --dspark-block caps the drafts the drafter hands
// over (run it at 5 to let the rule reach the whole block).  DS4_DSPARK_ADAPT_THETA=F fixes theta;
// DS4_DSPARK_ADAPT_ROW_MS=F sets row_ms.  Exactness: only how many drafts a round are verified changes, never which
// tokens are kept - greedy verify keeps a draft only when it is the trunk's argmax for its row (as at any --dspark-block).
struct DraftAdapt {
    static constexpr int P = 8, NB = 10;
    bool on = false;
    float theta_fixed = -1.0f, row_ms = 3.1f, prior = 16.0f, decay = 0.998f;
    double r[P][NB] = {}, k[P][NB] = {};
    double tok_ms = 0.0;   // EWMA of a round's ms per emitted token (0: none yet)
    long long rounds = 0, offered = 0, kept = 0, tokens = 0;
    double ms = 0.0;
    DraftAdapt() {
        const char* e = std::getenv("DS4_DSPARK_ADAPT");
        on = e != nullptr && std::atoi(e) != 0;
        if (const char* v = std::getenv("DS4_DSPARK_ADAPT_THETA")) theta_fixed = (float) std::atof(v);
        if (const char* v = std::getenv("DS4_DSPARK_ADAPT_ROW_MS")) row_ms = std::max(0.0f, (float) std::atof(v));
        if (const char* v = std::getenv("DS4_DSPARK_ADAPT_PRIOR")) prior = std::max(0.5f, (float) std::atof(v));
        if (const char* v = std::getenv("DS4_DSPARK_ADAPT_DECAY")) decay = std::min(1.0f, std::max(0.5f, (float) std::atof(v)));
        if (on)
            std::fprintf(stderr, "ds4: DS4_DSPARK_ADAPT: drafts a round by expected worth (theta %s, row %.2f ms, prior %.1f, decay %.4f)\n",
                         theta_fixed >= 0 ? std::to_string(theta_fixed).c_str() : "auto = row / ms a token", row_ms, prior, decay);
    }
    static int bucket(float c) { return std::min(NB - 1, std::max(0, (int) (c * (float) NB))); }
    double p(int i, float c) const {
        const int j = std::min(i, P - 1), b = bucket(c);
        const double c0 = std::min(0.98, std::max(0.02, (double) c));
        return (k[j][b] + prior * c0) / (r[j][b] + prior);
    }
    double theta() const {
        if (theta_fixed >= 0) return theta_fixed;
        return tok_ms > 0 ? (double) row_ms / tok_ms : (double) row_ms / 11.4;   // (11.4: MEASURED live ms a token, 2026-11-10)
    }
    // how many of the drafts to verify (the conf floor, room and depth as round()'s own loop)
    int choose(const std::vector<float>& conf, int nd, int room, int depth, float conf_min) const {
        const double th = theta();
        double cum = 1.0;
        int keep = 0;
        while (keep < nd && keep < room - 1 && keep < depth && conf[(size_t) keep] >= conf_min) {
            cum *= p(keep, conf[(size_t) keep]);
            if (cum < th) break;
            ++keep;
        }
        return keep;
    }
    void learn(const std::vector<float>& conf, int n, int acc) {
        for (int i = 0; i < n && i <= acc; ++i) {   // (reached: every draft before it kept)
            const int j = std::min(i, P - 1), b = bucket(conf[(size_t) i]);
            r[j][b] = r[j][b] * decay + 1.0;
            k[j][b] = k[j][b] * decay + (i < acc ? 1.0 : 0.0);
        }
        offered += n;
        kept += acc;
    }
    void time(double round_ms, int emitted) {
        if (emitted <= 0) return;
        const double t = round_ms / (double) emitted;
        tok_ms = tok_ms > 0 ? 0.97 * tok_ms + 0.03 * t : t;
        ++rounds;
        tokens += emitted;
        ms += round_ms;
        if ((rounds & 4095) == 0) report("every 4096 rounds");
    }
    void report(const char* when) const {
        std::fprintf(stderr, "ds4: DS4_DSPARK_ADAPT (%s): %lld rounds, %lld tokens in %.0f ms (%.2f ms a token, EWMA %.2f), %lld drafts verified, "
                             "%lld kept; theta %.3f\n", when, rounds, tokens, ms, tokens ? ms / (double) tokens : 0.0, tok_ms, offered, kept, theta());
        for (int i = 0; i < 5; ++i) {
            std::string row;
            for (int b = 0; b < NB; ++b) {
                char c[48];
                std::snprintf(c, sizeof c, " %.1f:%.0f/%.0f", (double) b / NB, k[i][b], r[i][b]);
                row += c;
            }
            std::fprintf(stderr, "ds4: DS4_DSPARK_ADAPT position %d kept/reached by confidence bucket:%s\n", i + 1, row.c_str());
        }
    }
    ~DraftAdapt() {
        if (on && rounds > 0) report("at exit");
    }
};
static DraftAdapt& draft_adapt() {
    static DraftAdapt a;
    return a;
}
// DS4_ROUND_LOG_MS=1 (with DS4_ROUND_LOG; default off: the log's lines as before): each line also gets
// "\tms\t<draft ms>\t<verify ms>\tnd\t<drafts handed over>\t<every one's confidence>" - the round's cost and the
// confidences of the drafts it did not verify, for drafter/replay.py
static const bool g_round_log_ms = std::getenv("DS4_ROUND_LOG_MS") != nullptr && std::atoi(std::getenv("DS4_ROUND_LOG_MS")) != 0;

static void tl_gate(bool open) {
#if defined(STRATA_METAL)
    static const bool only_draft = std::getenv("DS4_TL_DRAFT") != nullptr;
    if (only_draft) ds4_timeline_gate(open ? 1 : 0);
#endif
}
static unsigned long long rt_seq() {
#if defined(STRATA_METAL)
    return ds4_dispatch_seq();
#else
    return 0;
#endif
}
static double rt_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
bool round(strata::ds4::Model& m, int tok, int room, bool spec, float conf_min, float temperature, float top_p, int top_k,
           float min_p, unsigned long long seed, unsigned long long& counter, std::vector<int>& out, int& offered,
           int& accepted, std::string& err, float cum_min = -1.0f) {
    out.clear();
    offered = accepted = 0;
    const double rt_in = rt_now();   // (DS4_DSPARK_ADAPT / DS4_ROUND_LOG_MS: the round's start)
    bool drafted = false;
    if (spec && room > 1 && m.draft_block() > 0) {
        // the drafts' buffers kept from round to round (a round is a decode step: no allocation in it)
        thread_local std::vector<int> drafts;
        thread_local std::vector<float> conf;
        if (drafts.capacity() < 64) {
            drafts.reserve(64);
            conf.reserve(64);
            g_round_conf.reserve(64);
        }
        ++g_rt.rounds;
        static const bool tl_init = (tl_gate(false), true);
        (void) tl_init;
        tl_gate(true);
        const double rt0 = rt_now();
        const unsigned long long rs0 = rt_seq();
        if (!m.draft(tok, drafts, conf, err, temperature, top_p, top_k, min_p, seed, counter)) return false;
        g_rt.draft += rt_now() - rt0;
        drafted = true;
        tl_gate(false);
        g_rt.draft_d += rt_seq() - rs0;
        g_round_conf = conf;
        g_round_nd = (int) drafts.size();
        int keep = 0;
        if (cum_min < 0) cum_min = dspark_cum_default(m);
        DraftAdapt& ad = draft_adapt();
        if (ad.on)   // DS4_DSPARK_ADAPT
            keep = ad.choose(conf, (int) drafts.size(), room, g_spec_depth, conf_min);
        else {
            float cum = 1.0f;
            while (keep < (int) drafts.size() && keep < room - 1 && keep < g_spec_depth && conf[(size_t) keep] >= conf_min &&
                   (cum *= conf[(size_t) keep]) >= cum_min)
                ++keep;
        }
        if (keep > 0) {
            drafts.resize((size_t) keep);
            int acc = 0, next = -1;
            const double rt1 = rt_now();
            const unsigned long long rs1 = rt_seq();
            if (!m.verify(tok, drafts, temperature, top_p, top_k, min_p, seed, counter, acc, next, err)) return false;
            const double rt_v = rt_now() - rt1;
            g_rt.verify += rt_v;
            g_rt.verify_d += rt_seq() - rs1;
            ++g_rt.spec_rounds;
            if (ad.on) {
                ad.learn(g_round_conf, keep, acc);
                ad.time(rt_now() - rt0, acc + 1);
            }
            if (g_round_log != nullptr) {   // DS4_ROUND_LOG: tag, drafts offered, accepted, the confidences
                std::fprintf(g_round_log, "%s\t%d\t%d", g_round_tag.c_str(), keep, acc);
                for (int i = 0; i < keep; ++i) std::fprintf(g_round_log, "\t%.4f", g_round_conf[(size_t) i]);
                if (g_round_log_ms) {   // DS4_ROUND_LOG_MS
                    std::fprintf(g_round_log, "\tms\t%.3f\t%.3f\tnd\t%d", rt1 - rt0, rt_v, g_round_nd);
                    for (int i = 0; i < g_round_nd; ++i) std::fprintf(g_round_log, "\t%.4f", g_round_conf[(size_t) i]);
                }
                std::fprintf(g_round_log, "\n");
            }
            counter += (unsigned long long) acc + 1;
            for (int i = 0; i < acc; ++i) out.push_back(drafts[(size_t) i]);
            out.push_back(next);
            offered = keep;
            accepted = acc;
            return true;
        }
    }
    const double rt2 = rt_now();
    const unsigned long long rs2 = rt_seq();
    if (!m.step(tok, true, err)) return false;
    out.push_back(m.sample(temperature, top_p, top_k, min_p, seed, counter++));
    const double rt_s = rt_now() - rt2;
    g_rt.step += rt_s;
    g_rt.step_d += rt_seq() - rs2;
    ++g_rt.steps;
    if (draft_adapt().on) draft_adapt().time(rt_now() - rt_in, 1);   // (DS4_DSPARK_ADAPT: plain steps count in the ms a token)
    if (drafted && g_round_log != nullptr && g_round_log_ms) {   // DS4_ROUND_LOG_MS: a round whose drafts were all cut
        std::fprintf(g_round_log, "%s\t0\t0\tms\t%.3f\t%.3f\tnd\t%d", g_round_tag.c_str(), rt2 - rt_in, rt_s, g_round_nd);
        for (int i = 0; i < g_round_nd; ++i) std::fprintf(g_round_log, "\t%.4f", g_round_conf[(size_t) i]);
        std::fprintf(g_round_log, "\n");
    }
    return true;
}

struct Req {
    long long max_new = 0;
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 64;
    unsigned long long seed = 0;
    int spec = -1;          // spec=0|1: this request drafts or not (-1: as the engine was started)
    float conf = -1.0f;     // dspark_conf=F: the drafts' confidence floor for this request (-1: the engine's)
    long long pin = 0;      // pin=N: the first N prompt tokens are a start other requests share (a system prompt and its
                            // tools): the state at N is kept as a checkpoint, so another agent behind the same start
                            // goes back to it instead of reading it again
    float cum = -1.0f;      // dspark_cum=F: the cut on the product of the confidences (-1: the engine's, dspark_cum_default)
    std::string file;   // GENI: the pictures
    std::vector<int> ids;
};

// A prompt's picture: where its span starts, its pixels (the server's resize: RGB rows, both sides multiples of the
// patch size), a hash of them; once encoded, its span's rows
struct Pic {
    int64_t start = 0;
    int width = 0, height = 0;
    uint64_t hash = 0;
    size_t len = 0;   // the span (0: not known yet)
    std::vector<uint8_t> rgb;
};

// upstream fix C3 (upstream e1b27d54, #4350): after a fatal GPU error (an illegal memory access, a launch failure, a hardware
// exception) the device context is gone and every later HIP call of this process fails, so an engine that keeps serving
// only turns each request into an error.  It says so on stdout ("ERR fatal GPU error: ...", no DONE after it) and exits
// with code 3; the server fails the request in flight as an engine death and starts the engine again.  Only error paths
// call these.
namespace {
bool gpu_fatal_text(const std::string& e) {
    static const char* const k[] = {"illegal memory access", "unspecified launch failure", "hardware exception", "illegal instruction",
                                    "misaligned address", "device-side assert", "uncorrectable ECC", "launch timed out",
                                    "launch failure", "device lost", "ERROR_DEVICE_LOST"};
    for (const char* w : k)
        if (e.find(w) != std::string::npos) return true;
    return false;
}
[[noreturn]] void die_gpu_fatal(const std::string& e) {
    std::fprintf(stderr, "ds4: fatal GPU error (%s): the GPU context cannot be used again, so the engine ends and the server starts it again\n",
                 e.c_str());
    std::fflush(stderr);
    std::printf("ERR fatal GPU error: %s\n", e.c_str());
    std::fflush(stdout);
    std::_Exit(3);   // (not a clean unwind: freeing device memory on a dead context can hang)
}
}  // namespace

// GENI's file: int32 'DSI1', int32 count, then per picture int64 start, int32 width, int32 height and its
// height x width x 3 RGB bytes, in prompt order
bool read_pics(const std::string& path, std::vector<Pic>& v, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open the pictures' file " + path;
        return false;
    }
    char magic[4] = {};
    int32_t n = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "DSI1", 4) == 0 && std::fread(&n, 4, 1, f) == 1 &&
              n >= 0 && n <= 4096;
    for (int i = 0; ok && i < n; ++i) {
        Pic p;
        int32_t wh[2] = {};
        ok = std::fread(&p.start, 8, 1, f) == 1 && std::fread(wh, 4, 2, f) == 2 && wh[0] > 0 && wh[1] > 0 &&
             (int64_t) wh[0] * wh[1] <= (64 << 20);
        if (!ok) break;
        p.width = wh[0], p.height = wh[1];
        p.rgb.resize((size_t) p.width * p.height * 3);
        ok = std::fread(p.rgb.data(), 1, p.rgb.size(), f) == p.rgb.size();
        uint64_t h = 1469598103934665603ull;   // FNV-1a over the size and the pixels
        for (int32_t x : wh)
            for (int b = 0; b < 4; ++b) h = (h ^ (uint8_t) (x >> (8 * b))) * 1099511628211ull;
        for (uint8_t b : p.rgb) h = (h ^ b) * 1099511628211ull;
        p.hash = h;
        v.push_back(std::move(p));
    }
    std::fclose(f);
    if (!ok) err = path + ": not a pictures' file (DSI1)";
    return ok;
}

// the encoded spans, the latest few (a conversation sends its pictures with every turn)
struct SpanCache {
    std::deque<std::pair<uint64_t, std::shared_ptr<const std::vector<uint16_t>>>> q;
    std::shared_ptr<const std::vector<uint16_t>> get(strata::ds4::Model& m, const Pic& p, std::string& err) {
        for (size_t i = 0; i < q.size(); ++i)
            if (q[i].first == p.hash) {
                auto r = q[i].second;
                q.erase(q.begin() + (long) i);
                q.emplace_front(p.hash, r);
                return r;
            }
        auto rows = std::make_shared<std::vector<uint16_t>>();
        if (!m.encode_image(p.rgb.data(), p.width, p.height, *rows, err)) return nullptr;
        q.emplace_front(p.hash, rows);
        if (q.size() > 8) q.pop_back();
        return rows;
    }
};

bool parse_gen(const std::string& line, Req& r, bool geni) {
    const char* p = line.c_str() + (geni ? 5 : 4);
    char* e = nullptr;
    r.max_new = std::strtoll(p, &e, 10);
    if (e == p) return false;
    p = e;
    for (;;) {
        while (*p == ' ') ++p;
        const char* s = p;
        while (*p && *p != ' ') ++p;
        if (p == s) break;
        const std::string tok(s, (size_t) (p - s));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos && geni && r.file.empty()) {
            r.file = tok;
            continue;
        }
        if (eq == std::string::npos) {
            r.ids = parse_ids(tok);
            break;
        }
        const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        if (k == "temperature") r.temperature = std::strtof(v.c_str(), nullptr);
        else if (k == "top_p") r.top_p = std::strtof(v.c_str(), nullptr);
        else if (k == "top_k") r.top_k = std::atoi(v.c_str());
        else if (k == "min_p") r.min_p = std::strtof(v.c_str(), nullptr);
        else if (k == "seed") r.seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "spec") r.spec = std::atoi(v.c_str()) != 0;
        else if (k == "dspark_conf") r.conf = std::strtof(v.c_str(), nullptr);
        else if (k == "pin") r.pin = std::atoll(v.c_str());
        else if (k == "dspark_cum" || k == "spec_cum") r.cum = std::strtof(v.c_str(), nullptr);
    }
    return !r.ids.empty() && (!geni || !r.file.empty());
}

int serve(strata::ds4::Model& m, const std::string& eos_list, bool spec, float conf_min, const std::string& pack) {
    std::vector<int> eos = parse_ids(eos_list);
    for (const std::string& l : m.info_lines()) std::printf("%s\n", l.c_str());
    // srvbatch: the server's batch protocol (StrataEngine.generate_batched) on DS4_BATCH_DECODE's banks.  slot_cache=1:
    // a request that leaves its bank (BSTOP, then GEN with its prompt + output) finds its state there (see the bank match)
    const int bt_n = m.batch_slots();
    // (batch_pictures=0: the banks hold text only - the server runs a request with pictures alone, never as BGENI)
    if (bt_n > 0) std::printf("INFO batch_slots=%d\nINFO slot_cache=1\nINFO batch_pictures=0\n", bt_n);
    std::printf("READY %lld stop\n", (long long) m.max_context());
    std::fflush(stdout);
    std::fprintf(stderr, "maya ds4 serve: GEN <max_new> [keys] <ids>; GENI <max_new> [keys] <pictures> <ids>; STOP, QUIT\n");
    Lines in;
    in.start();
    std::string line;
    unsigned long long counter = 0;
    std::vector<Pic> held;   // the pictures the state holds (start, hash, span)
    SpanCache spans;
    const int img_tok = m.image_token();
    // CONVERSATION SLOTS: a request that does not continue the conversation in the model first sets that one aside
    // (Model::slot_save: its window rings, the compressors' partial groups and every source's compressed KV and index
    // keys), then takes back the kept conversation this prompt does continue (slot_load) - chats, and an agent's side
    // requests, that interleave no longer re-read each other's prompts (70k and 185k tokens alternating: 75-200 s a
    // switch at ~930 tok/s).  STRATA_DS4_SLOT_KEEP=1 turns them on and keeps them across runs, in STRATA_DS4_SLOT_DIR
    // (default <pack>/slots): a slot is a .bin (the state) and a .meta (this model's identity, the conversation's
    // tokens and its size), both written under a temporary name first; a slot whose .meta is missing, foreign or does
    // not match its file, or unused for STRATA_DS4_SLOT_DAYS (7) days, is removed at start.  STRATA_DS4_SLOT_CAP (4) is
    // the conversations kept in RAM; past it the least recently used is written to the folder and read back from there
    // (RAM first, the folder past the cap), and the engine's end writes the ones in RAM to the folder, so the next run
    // reads on from them.  STRATA_DS4_SLOT_GB (16) caps the folder's .bin files: at start, and after every write, the
    // folder's least recently used conversation is removed while they are over it (never the one just written, so a
    // conversation bigger than the cap stays).  STRATA_DS4_SLOT_MIN (1024) is the shortest conversation set aside (a
    // shorter one is read again in about the time its slot would take).  Text only: a conversation that holds pictures
    // is not set aside, and a prompt with pictures takes none back.
    const auto env_num = [](const char* k, double d) { const char* v = std::getenv(k); return v ? std::atof(v) : d; };
    const bool slot_keep =
        std::getenv("STRATA_DS4_SLOT_KEEP") != nullptr && std::atoi(std::getenv("STRATA_DS4_SLOT_KEEP")) != 0;
    const bool reuse = std::getenv("STRATA_DS4_REUSE") == nullptr || std::atoi(std::getenv("STRATA_DS4_REUSE")) != 0;
    int slot_max = slot_keep ? std::max(0, (int) env_num("STRATA_DS4_SLOT_CAP", 4)) : 0;
    const size_t slot_min = (size_t) std::max(1.0, env_num("STRATA_DS4_SLOT_MIN", 1024));
    const double slot_days = env_num("STRATA_DS4_SLOT_DAYS", 7);
    const uint64_t slot_budget = (uint64_t) (env_num("STRATA_DS4_SLOT_GB", 16) * 1073741824.0);
    namespace fs = std::filesystem;
    std::filesystem::path slot_dir;
    struct Slot {
        std::vector<int32_t> tokens;
        std::string path;                            // empty: the state is in data (RAM)
        uint64_t bytes = 0, used = 0;
        std::shared_ptr<std::vector<uint8_t>> data;   // the RAM slot's bytes (null: a file slot)
        std::vector<int64_t> cks;                     // the positions short of its end it can be taken back at
    };
    std::vector<Slot> slots;
    uint64_t slot_clock = 0, slot_seq = 0;
    const std::string slot_identity = m.slot_identity();
    const auto note = [&](const std::string& msg) {
        std::fprintf(stderr, "ds4 slots: %s\n", msg.c_str());
        std::printf("NOTE sessions: %s\n", msg.c_str());
        std::fflush(stdout);
    };
    const auto gb = [](uint64_t b) {
        char s[32];
        std::snprintf(s, sizeof s, "%.2f GB", (double) b / 1e9);
        return std::string(s);
    };
    const auto meta_of = [](const std::string& bin) { return bin.substr(0, bin.size() - 4) + ".meta"; };
    const auto meta_write = [&](const std::string& path, const std::vector<int32_t>& toks, uint64_t bytes,
                                const std::vector<int64_t>& cks) -> bool {
        return strata::ds4::slotmeta::write(path, slot_identity, toks, bytes, cks);
    };
    // a .meta of this model -> its tokens and the slot's size (false: foreign, damaged or not this model's)
    const auto meta_read = [&](const std::string& path, std::vector<int32_t>& toks, uint64_t& bytes,
                               std::vector<int64_t>& cks) -> bool {
        return strata::ds4::slotmeta::read(path, slot_identity, m.max_context(), toks, bytes, cks);
    };
    const auto slot_age_days = [](const std::string& p) {
        std::error_code ec;
        const auto t = fs::last_write_time(p, ec);
        if (ec) return 1e9;
        return std::chrono::duration<double>(fs::file_time_type::clock::now() - t).count() / 86400.0;
    };
    const auto slot_drop = [&](size_t i) {
        if (slots[i].data)   // a RAM slot: its buffer simply dies
            slots[i].data.reset();
        else {
            std::error_code ec;
            std::filesystem::remove(slots[i].path, ec);
            std::filesystem::remove(meta_of(slots[i].path), ec);
        }
        slots.erase(slots.begin() + (long) i);
    };
    const auto slot_count = [&](int& ram, int& disk) {
        ram = disk = 0;
        for (const Slot& s : slots) (s.data ? ram : disk) += 1;
    };
    // the folder's conversations (their .bin; a .meta beside each is a few KB: identity and tokens)
    const auto slot_folder_bytes = [&]() {
        uint64_t total = 0;
        std::error_code ec;
        for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {
            const std::string fn = de.path().filename().string();
            if (fn.rfind("slot-", 0) == 0 && fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0) {
                const uint64_t bytes = (uint64_t) std::filesystem::file_size(de.path(), ec);
                if (!ec) total += bytes;
            }
        }
        return total;
    };
    // the folder over STRATA_DS4_SLOT_GB: its least recently used conversation goes first, and never `keep` (the one just
    // written), so a conversation bigger than the cap stays
    const auto slot_trim = [&](const std::string& keep) {
        const uint64_t before = slot_folder_bytes();
        size_t dropped = 0;
        while (slot_folder_bytes() > slot_budget) {
            size_t lru = SIZE_MAX;
            for (size_t i = 0; i < slots.size(); ++i)
                if (!slots[i].data && slots[i].path != keep && (lru == SIZE_MAX || slots[i].used < slots[lru].used))
                    lru = i;
            if (lru == SIZE_MAX) break;
            slot_drop(lru);
            ++dropped;
        }
        if (dropped > 0) {
            char msg[256];
            std::snprintf(msg, sizeof msg, "the folder held %s (over STRATA_DS4_SLOT_GB, %s): %zu least recently used "
                          "conversation(s) removed, %s left", gb(before).c_str(), gb(slot_budget).c_str(), dropped,
                          gb(slot_folder_bytes()).c_str());
            note(msg);
        }
    };
    // a RAM slot into the folder (its bytes, then its .meta); false: not written (its bytes go, then)
    const auto slot_write_file = [&](Slot& s) -> bool {
        std::error_code ec;
        const auto space = std::filesystem::space(slot_dir, ec);
        if (!ec && space.available < ((uint64_t) 8 << 30)) {   // (8 GB free, the GLM engine's guard)
            std::fprintf(stderr, "ds4 slots: under 8 GB free in %s - the conversation is not written\n",
                         slot_dir.string().c_str());
            return false;
        }
        const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string path =
            (slot_dir / ("slot-" + std::to_string(stamp) + "-" + std::to_string(++slot_seq) + ".bin")).string();
        {
            std::ofstream f(path + ".tmp", std::ios::binary | std::ios::trunc);
            f.write((const char*) s.data->data(), (std::streamsize) s.data->size());
            f.flush();
            if (!f) {
                std::filesystem::remove(path + ".tmp", ec);
                std::fprintf(stderr, "ds4 slots: writing %s.tmp failed (disk full?)\n", path.c_str());
                return false;
            }
        }
        fs::rename(path + ".tmp", path, ec);
        if (ec || !meta_write(meta_of(path), s.tokens, s.data->size(), s.cks)) {
            std::filesystem::remove(path + ".tmp", ec);
            std::filesystem::remove(path, ec);
            std::fprintf(stderr, "ds4 slots: writing %s failed\n", path.c_str());
            return false;
        }
        s.path = path;
        s.bytes = s.data->size();
        s.data.reset();
        slot_trim(path);   // (the folder over its cap: the least recently used conversation goes, not this one)
        return true;
    };
    // the conversation in the model into a slot (the state is lost otherwise: another one takes the model)
    const auto slot_put = [&](const char* why) {
        if (slot_max <= 0) return;
        const std::vector<int32_t>& t = m.tokens();
        if (t.size() < slot_min) return;   // (a fresh engine's 0-token state is no conversation: nothing to set aside)
        const auto t0 = std::chrono::steady_clock::now();
        auto buf = std::make_shared<std::vector<uint8_t>>();
        std::string e;
        const uint64_t bytes = m.slot_save(*buf, e);   // (a state unchanged since it was taken refuses here: nothing
        if (bytes == 0) {                             // changes, its slot stays)
            if (e != "it is already set aside")
                std::fprintf(stderr, "ds4 slots: setting the conversation aside failed (%s) - it is read from token 0 "
                                     "when it comes back\n", e.c_str());
            return;
        }
        for (size_t i = slots.size(); i-- > 0;)   // an earlier state of this conversation is superseded
            if (slots[i].tokens.size() <= t.size() && std::equal(slots[i].tokens.begin(), slots[i].tokens.end(), t.begin()))
                slot_drop(i);
        slots.push_back({t, "", bytes, ++slot_clock, nullptr, m.check_positions()});
        slots.back().data = buf;
        while (true) {   // past the cap: the least recently used conversation in RAM to the folder, or gone
            int ram = 0, disk = 0;
            slot_count(ram, disk);
            if (ram <= slot_max) break;
            size_t lru = SIZE_MAX;
            for (size_t i = 0; i < slots.size(); ++i)
                if (slots[i].data && (lru == SIZE_MAX || slots[i].used < slots[lru].used)) lru = i;
            if (lru == SIZE_MAX) break;
            if (!slot_write_file(slots[lru])) slot_drop(lru);
        }
        int ram = 0, disk = 0;
        slot_count(ram, disk);
        char msg[256];
        std::snprintf(msg, sizeof msg, "saved a %zu-token conversation%s (%s in RAM in %.1f s); %d in RAM, %d in %s",
                      t.size(), why, gb(bytes).c_str(),
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), ram, disk,
                      slot_dir.string().c_str());
        note(msg);
    };
    // the kept conversation this prompt shares the most of - past `floor` tokens, what the model itself can go back
    // to - back in the model: the tokens it covers (0: none).  A conversation the prompt continues whole comes back
    // whole (`whole`); one it leaves part way - a retried prompt, a history the client renders again without the
    // earlier reasoning - comes back at its latest checkpoint the prompt still agrees with (the end of a prompt, the
    // position before it, the anchor where its run of continued prompts began).
    const auto slot_take = [&](const std::vector<int>& prompt, size_t floor, bool& whole) -> size_t {
        whole = false;
        if (slot_max <= 0 || prompt.size() < 2) return 0;
        std::vector<strata::ds4::slotmeta::Cand> cands;
        for (const Slot& sl : slots) cands.push_back({&sl.tokens, &sl.cks});
        size_t len = 0;
        const size_t best = strata::ds4::slotmeta::pick(cands, prompt, floor, &len);
        if (best == SIZE_MAX) return 0;
        const size_t whole_len = slots[best].tokens.size();
        const auto t0 = std::chrono::steady_clock::now();
        const bool ram = slots[best].data != nullptr;
        std::string e;
        bool back = false;
        if (ram) back = m.slot_load(slots[best].data->data(), slots[best].data->size(), (int64_t) whole_len, e);
        else {
            std::ifstream f(slots[best].path, std::ios::binary);
            std::vector<uint8_t> file;
            if (f) {
                f.seekg(0, std::ios::end);
                const std::streamoff n = f.tellg();
                f.seekg(0, std::ios::beg);
                if (n > 0) {
                    file.resize((size_t) n);
                    if (f.read((char*) file.data(), n))
                        back = m.slot_load(file.data(), file.size(), (int64_t) whole_len, e);
                    else e = "reading " + slots[best].path + " failed";
                } else e = slots[best].path + " is empty";
            } else e = "cannot read " + slots[best].path;
        }
        if (!back) {
            note("taking a conversation back failed (" + (e.empty() ? std::string("it did not load") : e) +
                 ") - reading its prompt again");
            slot_drop(best);
            return 0;
        }
        if (len < whole_len && !(m.restore((int64_t) len) && (size_t) m.pos() == len)) {
            note("going back to token " + std::to_string(len) + " of a kept conversation failed - reading its prompt again");
            m.reset();
            return 0;
        }
        whole = len == whole_len;
        slots[best].used = ++slot_clock;
        if (!ram) {   // its age counts from its last use (a file slot's only)
            std::error_code ec;
            fs::last_write_time(meta_of(slots[best].path), fs::file_time_type::clock::now(), ec);
        }
        char msg[256];
        std::string part;
        if (!whole) part = " at token " + std::to_string(len) + " of " + std::to_string(whole_len);
        std::snprintf(msg, sizeof msg, "took back a %zu-token conversation%s from %s (%.1f s); %zu new tokens to read",
                      len, part.c_str(), ram ? "RAM" : "the folder",
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                      prompt.size() - len);
        note(msg);
        return len;
    };
    if (slot_max > 0) {
        slot_dir = std::getenv("STRATA_DS4_SLOT_DIR") ? std::filesystem::path(std::getenv("STRATA_DS4_SLOT_DIR"))
                                                      : std::filesystem::path(pack) / "slots";
        std::error_code ec;
        std::filesystem::create_directories(slot_dir, ec);
        if (!std::filesystem::is_directory(slot_dir, ec)) {
            std::fprintf(stderr, "ds4 slots: %s is not writable - conversations are not set aside\n",
                         slot_dir.string().c_str());
            slot_max = 0;
        } else {
            struct Found {
                std::string bin;
                std::vector<int32_t> toks;
                uint64_t bytes;
                std::vector<int64_t> cks;
                fs::file_time_type t;
            };
            std::vector<Found> found;
            size_t refused = 0;
            for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {   // a previous run's slots
                const std::string fn = de.path().filename().string(), p = de.path().string();
                if (fn.rfind("slot-", 0) != 0) continue;
                const bool is_bin = fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0;
                const bool is_meta = fn.size() > 10 && fn.compare(fn.size() - 5, 5, ".meta") == 0;
                if (!is_bin && !is_meta) {   // a .tmp an interrupted write left
                    std::filesystem::remove(de.path(), ec);
                    continue;
                }
                if (!is_meta) continue;      // a .bin is judged by its .meta
                const std::string bin = p.substr(0, p.size() - 5) + ".bin";
                Found fd{bin, {}, 0, {}, fs::last_write_time(de.path(), ec)};
                const uint64_t have = fs::exists(bin, ec) ? (uint64_t) fs::file_size(bin, ec) : 0;
                if (meta_read(p, fd.toks, fd.bytes, fd.cks) && have == fd.bytes && fd.bytes > 0 && slot_age_days(p) <= slot_days)
                    found.push_back(std::move(fd));
                else {
                    ++refused;
                    std::filesystem::remove(p, ec);
                    std::filesystem::remove(bin, ec);
                }
            }
            for (const auto& de : std::filesystem::directory_iterator(slot_dir, ec)) {   // a .bin without its .meta
                const std::string p = de.path().string(), fn = de.path().filename().string();
                if (fn.rfind("slot-", 0) == 0 && fn.size() > 9 && fn.compare(fn.size() - 4, 4, ".bin") == 0 &&
                    !fs::exists(meta_of(p), ec))
                    std::filesystem::remove(de.path(), ec);
            }
            std::sort(found.begin(), found.end(), [](const Found& a, const Found& b) { return a.t < b.t; });
            uint64_t total = 0;
            for (auto& fd : found) {
                total += fd.bytes;
                slots.push_back({std::move(fd.toks), fd.bin, fd.bytes, ++slot_clock, nullptr, std::move(fd.cks)});
            }
            std::string msg = "folder " + slot_dir.string() + ": " + std::to_string(slots.size()) +
                              " conversation(s) kept from earlier runs (" + gb(total) + ")";
            if (refused) msg += ", " + std::to_string(refused) + " not this model's or incomplete - removed";
            note(msg);
            slot_trim("");   // (the folder over STRATA_DS4_SLOT_GB: the least recently used conversation goes first)
        }
    }
    if (slot_max > 0) {
#if !defined(_WIN32)
        // (the server ends the engine with QUIT and waits for it; a SIGTERM to the whole process group - systemd
        // stopping a service, a supervisor killing the server's group - must not end it mid-write, and a server that
        // is already gone must not end it with SIGPIPE when it writes its last lines)
        std::signal(SIGTERM, SIG_IGN);
        std::signal(SIGINT, SIG_IGN);
        std::signal(SIGPIPE, SIG_IGN);
#endif
    }
    // WP-8's option (1): the post-prompt cache settle runs in the gaps between requests and gives way the moment one arrives
    // (DS4_SETTLE_IDLE=0 turns it off).  An agent's turn ends, its tool runs, and the swaps the next answer wants go in during
    // that idle - which is what the 800 ms prompt-side budget cannot do for a run that has no next request.
    std::atomic<bool> settle_cancel{false};
    std::thread settle_th;
    const bool settle_idle_on = [] {
        const char* v = std::getenv("DS4_SETTLE_IDLE");
        return v == nullptr || std::atoi(v) != 0;
    }();
    const auto settle_stop = [&] {
        if (!settle_th.joinable()) return;
        settle_cancel.store(true, std::memory_order_release);
        settle_th.join();
    };
    const auto settle_start = [&] {
        if (!settle_idle_on || settle_th.joinable()) return;
        settle_cancel.store(false, std::memory_order_release);
        settle_th = std::thread([&m, &settle_cancel] { m.settle_idle(settle_cancel); });
    };
    // ---- srvbatch: continuous batching over the banks (DS4_BATCH_DECODE=N; the server's "parallel": N / --batch N) ----
    // `BGEN <bank> <max_new> [keys] <ids>` is read as `GEN 1 [keys] <ids>` - the solo prompt path with all its reuse (the
    // conversation in the model, its checkpoints, pin=, the kept slots, and the banks themselves: the bank match below) -
    // and its one token answered on the control lines (T, DONE), then the state is swapped into the bank (bank_swap, no
    // copy: what the bank held becomes the conversation in the model, kept like any other) and `BADM <bank> 1` says it
    // goes on there: `BT <bank> <id>` a batch step, `BDONE <bank> <n> <stop|length|cancel> <ms>` at its end.
    // Between two lines the banks step; while a prompt is read they step between its chunks (DS4_BATCH_PREFILL_SHARE).
    // The drafter rule is the engine's (DS4_BATCH_DSPARK_MAX, decided each step): a step drafts while at most that many
    // banks are active and all of them are greedy; else a plain step, where a temperature > 0 row is sampled on its own.
    struct BankRun {
        bool active = false, stop = false, cool = false;
        int cur = -1;                                // the token the next step feeds (the last one sent)
        long long produced = 0, max_new = 0;
        float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f, conf = -1.0f, cum = -1.0f;
        int top_k = 64, spec = -1;
        unsigned long long seed = 0, counter = 0;
        double t0 = 0.0;
    };
    std::vector<BankRun> bk((size_t) bt_n);
    std::vector<std::vector<Pic>> bheld((size_t) bt_n);   // the pictures each bank's state holds (it traded places with `held`)
    const int bt_smax = m.batch_spec_max();
    // DS4_BATCH_DSPARK_SAMPLED=1 (srvbatch12, retest b5-ii; default off: the rule below, byte for byte): sampled rows
    // draft too - a step drafts while at most DS4_BATCH_DSPARK_MAX banks are active, greedy or not, and a sampled row's
    // drafts are kept by the speculative-sampling rule over its own chain (Model::batch_verify_sampled); greedy rows
    // keep the argmax rule.  Off, a step with a sampled row is a plain step (each sampled row drawn on its own)
    const bool bt_samp = [] {
        const char* v = std::getenv("DS4_BATCH_DSPARK_SAMPLED");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (bt_samp && bt_n > 0)
        std::fprintf(stderr, "ds4 batch: DS4_BATCH_DSPARK_SAMPLED=1: sampled rows draft too while at most %d banks are active "
                             "(the speculative-sampling rule a row)%s\n", bt_smax, bt_smax > 0 ? "" : " - but batch drafts are off");
    std::vector<strata::ds4::Model::BatchSampling> bt_sv;
    // the share of the time the banks get while a prompt is read (0: the whole prompt first; 0.5: a chunk, then as long in steps)
    const double bt_share = [&] {
        const double v = env_num("DS4_BATCH_PREFILL_SHARE", 0.5);
        return v < 0.0 ? 0.0 : v > 0.9 ? 0.9 : v;
    }();
    bool bt_fatal = false;
    // srvbatch12: the batch loop's counters (stderr at each BDONE): steps, the drafted ones, rows, drafts kept
    // (greedy rows / sampled rows) and sampled rows in plain steps
    struct BtStats {
        long long steps = 0, drafted = 0, rows = 0, off_g = 0, acc_g = 0, off_s = 0, acc_s = 0, plain_s = 0;
    } bts;
    const auto bank_live = [&] {
        for (const BankRun& x : bk)
            if (x.active) return true;
        return false;
    };
    const auto bank_trade = [&](int b) -> bool {   // bank b <-> the conversation in the model (and their pictures)
        std::string e;
        if (!m.bank_swap(b, e)) {
            std::fprintf(stderr, "ds4 batch: bank %d: %s\n", b, e.c_str());
            return false;
        }
        std::swap(held, bheld[(size_t) b]);
        if (m.pos() == 0) m.reset();   // (a bank never used: a clean state, as a fresh engine's)
        return true;
    };
    const auto bank_done = [&](int b, const char* fin) {
        BankRun& x = bk[(size_t) b];
        std::printf("BDONE %d %lld %s %.1f\n", b, x.produced, fin, now_ms() - x.t0);
        x.active = false;
        std::fprintf(stderr, "ds4 batch: bank %d %s after %lld tokens in %.0f ms; so far %lld steps (%lld drafted), %lld rows, "
                             "drafts kept greedy %lld/%lld sampled %lld/%lld, sampled rows in plain steps %lld\n",
                     b, fin, x.produced, now_ms() - x.t0, bts.steps, bts.drafted, bts.rows, bts.acc_g, bts.off_g, bts.acc_s,
                     bts.off_s, bts.plain_s);
    };
    const auto bank_emit = [&](int b, int y) -> bool {   // true: the bank ended on this token
        BankRun& x = bk[(size_t) b];
        std::printf("BT %d %d\n", b, y);
        ++x.produced;
        x.cur = y;
        bool end = false;
        for (int e : eos) end = end || y == e;
        const char* fin = end ? "stop" : x.stop ? "cancel" : x.produced >= x.max_new ? "length"
                        : m.batch_pos(b) + 2 > m.max_context() ? "length" : nullptr;
        if (fin == nullptr) return false;
        bank_done(b, fin);
        return true;
    };
    std::vector<int> bt_bk, bt_nr, bt_rows, bt_acc, bt_nx, bt_tried, bt_drafts;
    std::vector<float> bt_conf;
    // one batch step over the active banks; false: the step failed (the engine ends: its banks' states are unknown)
    const auto batch_tick = [&](std::string& e) -> bool {
        bt_bk.clear();
        bt_nr.clear();
        bt_rows.clear();
        bt_tried.clear();
        int A = 0;
        bool greedy = true;
        for (int b = 0; b < bt_n; ++b) {
            BankRun& x = bk[(size_t) b];
            if (!x.active) continue;
            if (in.bstop[b].exchange(false)) x.stop = true;
            if (x.stop) {
                bank_done(b, "cancel");
                continue;
            }
            ++A;
            greedy = greedy && x.temperature <= 0.0f;
        }
        if (A == 0) {
            std::fflush(stdout);
            return true;
        }
        const bool draft_now = bt_smax > 0 && (greedy || bt_samp) && A <= bt_smax;
        const bool samp_now = draft_now && !greedy;   // (DS4_BATCH_DSPARK_SAMPLED only: else draft_now needs all greedy)
        int R = 0, left = A;
        for (int b = 0; b < bt_n; ++b) {
            BankRun& x = bk[(size_t) b];
            if (!x.active) continue;
            --left;
            const long long room = std::min<long long>(x.max_new - x.produced, m.max_context() - 1 - m.batch_pos(b));
            int keep = 0;
            const bool try_spec = draft_now && (x.spec < 0 ? spec : x.spec != 0) && !x.cool && room > 1;
            if (try_spec) {
                if (!m.batch_draft(b, x.cur, bt_drafts, bt_conf, e)) return false;
                const float cmin = x.conf >= 0 ? x.conf : conf_min, cum_min = x.cum >= 0 ? x.cum : dspark_cum_default(m);
                float cum = 1.0f;
                while (keep < (int) bt_drafts.size() && keep < room - 1 && keep < g_spec_depth &&
                       bt_conf[(size_t) keep] >= cmin && (cum *= bt_conf[(size_t) keep]) >= cum_min)
                    ++keep;
            }
            keep = std::max(0, std::min(keep, 16 - R - 1 - left));   // (the window's 16 rows: one kept for each later bank)
            bt_bk.push_back(b);
            bt_nr.push_back(1 + keep);
            bt_rows.push_back(x.cur);
            for (int i = 0; i < keep; ++i) bt_rows.push_back(bt_drafts[(size_t) i]);
            bt_tried.push_back(try_spec ? 1 : 0);
            R += 1 + keep;
        }
        const int S = (int) bt_bk.size();
        ++bts.steps;
        bts.drafted += draft_now ? 1 : 0;
        bts.rows += R;
        bt_acc.assign((size_t) S, 0);
        bt_nx.assign((size_t) S, 0);
        if (samp_now) {   // (each sequence's chain, at its counter now: row i of its run draws at counter + i)
            bt_sv.assign((size_t) S, strata::ds4::Model::BatchSampling{});
            for (int s = 0; s < S; ++s) {
                const BankRun& x = bk[(size_t) bt_bk[(size_t) s]];
                strata::ds4::Model::BatchSampling& q = bt_sv[(size_t) s];
                q.temperature = x.temperature;
                q.top_p = x.top_p;
                q.top_k = x.top_k;
                q.min_p = x.min_p;
                q.seed = x.seed;
                q.counter = x.counter;
            }
        }
        if (samp_now ? !m.batch_verify_sampled(bt_bk.data(), bt_nr.data(), bt_rows.data(), S, bt_sv.data(), bt_acc.data(),
                                               bt_nx.data(), e)
            : draft_now ? !m.batch_verify(bt_bk.data(), bt_nr.data(), bt_rows.data(), S, bt_acc.data(), bt_nx.data(), e)
                        : !m.batch_step(bt_bk.data(), bt_rows.data(), S, bt_nx.data(), e))
            return false;
        if (!draft_now)   // (a plain step: one row a bank, row s = bank bt_bk[s]; greedy rows keep the argmax)
            for (int s = 0; s < S; ++s) {
                BankRun& x = bk[(size_t) bt_bk[(size_t) s]];
                if (x.temperature > 0.0f) {
                    bt_nx[(size_t) s] = m.batch_sample(s, x.temperature, x.top_p, x.top_k, x.min_p, x.seed, x.counter++);
                    ++bts.plain_s;
                }
                if (bt_nx[(size_t) s] < 0) {   // (srv10: a sample's CUDA error, as the solo path's Maya #619 check - no BT -1)
                    e = "the sample of bank " + std::to_string(bt_bk[(size_t) s]) + " failed";
                    return false;
                }
            }
        size_t ri = 0;
        for (int s = 0; s < S; ++s) {
            const int b = bt_bk[(size_t) s];
            BankRun& x = bk[(size_t) b];
            bool ended = false;
            // (DS4_BATCH_DSPARK_SAMPLED: a sampled row's verify drew rows 0 .. acc - the solo round's counter step)
            if (samp_now && x.temperature > 0.0f) x.counter += (unsigned long long) bt_acc[(size_t) s] + 1;
            if (bt_tried[(size_t) s]) {
                (x.temperature > 0.0f ? bts.off_s : bts.off_g) += bt_nr[(size_t) s] - 1;
                (x.temperature > 0.0f ? bts.acc_s : bts.acc_g) += bt_acc[(size_t) s];
            }
            for (int a = 0; a < bt_acc[(size_t) s] && !ended; ++a) ended = bank_emit(b, bt_rows[ri + 1 + (size_t) a]);
            if (!ended) bank_emit(b, bt_nx[(size_t) s]);
            x.cool = bt_tried[(size_t) s] && bt_nr[(size_t) s] == 1;
            ri += (size_t) bt_nr[(size_t) s];
        }
        std::fflush(stdout);
        return true;
    };
    for (;;) {
        if (bt_fatal) return 1;
        bool have_line;
        if (bank_live()) {   // (banks decoding: a line if one waits, else a batch step)
            have_line = in.try_next(line);
            if (!have_line) {
                if (in.closed()) break;
                std::string e;
                if (!batch_tick(e)) {
                    if (gpu_fatal_text(e)) die_gpu_fatal("a batch step failed: " + e);   // (upstream fix C3)
                    std::fprintf(stderr, "ds4 batch: a batch step failed: %s; the engine ends\n", e.c_str());
                    return 1;
                }
                continue;
            }
        } else {
            settle_start();   // (nothing in flight: this is the idle)
            have_line = in.next(line);
            settle_stop();    // (a line arrived, or the input closed)
            if (!have_line) break;
        }
        if (line == "QUIT") break;
        if (line.rfind("BYIELD", 0) == 0 || line.rfind("BCO ", 0) == 0) continue;   // (no-ops: decoding interleaves with reads)
        // BGEN <bank> <max_new> [keys] <ids>: an admission into a free bank, read as GEN 1 (above)
        int admit = -1;
        long long admit_max_new = 0;
        if (line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0) {
            char* e1 = nullptr;
            char* e2 = nullptr;
            const bool bimg = line.rfind("BGENI ", 0) == 0;
            const long b = std::strtol(line.c_str() + (bimg ? 6 : 5), &e1, 10);
            const long long mn = std::strtoll(e1, &e2, 10);
            if (bimg || bt_n <= 0 || b < 0 || b >= bt_n || bk[(size_t) b].active || mn < 1 || e2 == e1) {
                std::printf("ERR BGEN: no such free bank (--batch %d), a bad max_new, or pictures (never batched)\n", bt_n);
                std::fflush(stdout);
                continue;
            }
            admit = (int) b;
            admit_max_new = mn;
            line = std::string("GEN 1") + e2;
        }
        // an admission that ends before its bank takes it answers BADM <bank> 0 - except after an ERR (#1059: no line follows)
        struct AdmitEnd {
            int b = -1;
            ~AdmitEnd() {
                if (b < 0) return;
                std::printf("BADM %d 0\n", b);
                std::fflush(stdout);
            }
        } admit_end;
        admit_end.b = admit;
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (!geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR unsupported command '%s' (GEN, GENI, STOP and QUIT)\n", line.substr(0, line.find(' ')).c_str());
            std::fflush(stdout);
            continue;
        }
        in.stop.store(false);
        Req r;
        if (!parse_gen(line, r, geni)) {
            std::printf("ERR expected: GEN <max_new> [keys] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
            admit_end.b = -1;
            std::fflush(stdout);
            continue;
        }
        // srvbatch12: an admission reads its prompt as GEN 1, so the solo path's context check further down sees
        // max_new 1: the request's own max_new is checked here, with the solo path's words (PAR 1 and PAR N refuse alike;
        // a bank still ends "length" at the context's edge, bank_emit)
        if (admit >= 0 && (int64_t) r.ids.size() + admit_max_new > m.max_context()) {
            admit_end.b = -1;
            std::printf("ERR the prompt (%zu tokens) and max_new %lld are more than the context %lld\n", r.ids.size(),
                        admit_max_new, (long long) m.max_context());
            std::fflush(stdout);
            continue;
        }
        const double t0 = now_ms();
        std::vector<Pic> pics;
        {
            std::string e;
            if (geni && img_tok < 0) e = "this engine has no vision tower, so it cannot read pictures";
            else if (geni && read_pics(r.file, pics, e))
                for (const Pic& q : pics)
                    if (q.start < 0 || (size_t) q.start >= r.ids.size() || r.ids[(size_t) q.start] != img_tok)
                        e = "a picture's span does not start at an image token of the prompt";
            if (!e.empty()) {
                std::printf("ERR %s\n", e.c_str());
                std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
                admit_end.b = -1;
                std::fflush(stdout);
                continue;
            }
        }
        if (!reuse) {   // (STRATA_DS4_REUSE=0, for A/Bs and tests: every request is a fresh read of its whole prompt)
            m.reset();
            held.clear();
        }
        // srvbatch: the bank match.  A conversation that went on in a bank - its next turn (BGEN or GEN), or the solo
        // path taking it back after BSTOP - is in that bank, not in the model: an idle bank that shares more of this prompt
        // than the conversation in the model trades places with it first (bank_swap, no copy, nothing lost: the one in the
        // model waits in the bank).  The checks below then reuse it exactly as they reuse the conversation in the model.
        if (bt_n > 0 && reuse && pics.empty()) {
            const auto lcp = [&](const std::vector<int32_t>& t) {
                size_t k = 0;
                while (k < t.size() && k < r.ids.size() && t[k] == r.ids[k]) ++k;
                return k;
            };
            size_t best_c = lcp(m.tokens());
            int best = -1;
            for (int b = 0; b < bt_n; ++b) {
                const std::vector<int32_t>* t = m.bank_tokens(b);
                if (bk[(size_t) b].active || t == nullptr || !bheld[(size_t) b].empty()) continue;
                const size_t c2 = lcp(*t);
                if (c2 > best_c) {
                    best = b;
                    best_c = c2;
                }
            }
            if (best >= 0 && held.empty() && bank_trade(best))
                std::fprintf(stderr, "ds4 batch: bank %d holds %zu tokens of this prompt: taken into the model\n", best, best_c);
        }
        const auto& st = m.tokens();
        const size_t n_state = st.size();
        size_t c = 0;
        while (c < n_state && c < r.ids.size() && st[c] == r.ids[c]) ++c;
        // the same tokens with another picture (or none) do not share the state from that picture's span on
        auto same = [](const std::vector<Pic>& v, const Pic& q) {
            for (const Pic& x : v)
                if (x.start == q.start && x.hash == q.hash) return true;
            return false;
        };
        for (const Pic& q : pics)
            if ((size_t) q.start < c && !same(held, q)) c = (size_t) q.start;
        for (const Pic& q : held)
            if ((size_t) q.start < c && !same(pics, q)) c = (size_t) q.start;
        size_t from = 0;
        bool took = false, continued = false;   // (continued: the conversation goes on - its anchor stays)
        // the conversation continues, reused in place - a conversation has to be in the model first: a fresh engine
        // (the start of a run, so every restart) holds 0 tokens, and every prompt "continues" that to token 0; taken
        // as a continuation, the take below would never run for the folder's conversations, the case they are for
        if ((n_state > 0 || slot_max <= 0) && c == n_state && n_state < r.ids.size()) {   // (slots off: as before)
            from = n_state;
            continued = true;
        }
        else {
            // CONVERSATION SLOTS: this prompt does not continue the conversation in the model - that one is set aside
            // first (another one takes the model, and its state is lost otherwise), then the kept conversation this
            // prompt does continue is taken back - if it goes further back than the checkpoints the state itself holds
            if (held.empty()) slot_put("");
            const int64_t limit = (int64_t) std::min(c, r.ids.size() - 1);
            const int64_t rs = m.restorable(limit);
            bool whole = false;
            const size_t back = pics.empty() ? slot_take(r.ids, rs > 0 ? (size_t) rs : 0, whole) : 0;
            if (back > 0) {
                from = back;
                took = true;
                continued = whole;
            } else if (rs >= 0 && m.restore(limit))
                from = (size_t) m.pos();
            else m.reset();
        }
        if (took || from == 0) held.clear();   // (a slot is text only: the state that comes back holds no pictures)
        if ((int64_t) r.ids.size() + std::max(0LL, r.max_new) > m.max_context()) {
            admit_end.b = -1;
            std::printf("ERR the prompt (%zu tokens) and max_new %lld are more than the context %lld\n", r.ids.size(),
                        r.max_new, (long long) m.max_context());
            std::fflush(stdout);
            continue;
        }
        std::string err;
        bool ok = true;
        // the pictures whose spans the prompt reads (the rest the state holds): encoded (or the cache's), checked
        if (!pics.empty()) {
            std::vector<strata::ds4::Image> imgs;
            const double te = now_ms();
            int encoded = 0;
            for (Pic& q : pics) {
                for (const Pic& x : held)
                    if (x.start == q.start && x.hash == q.hash) q.len = x.len;
                if (q.len > 0 && (size_t) q.start + q.len <= from) continue;
                std::shared_ptr<const std::vector<uint16_t>> rows = spans.get(m, q, err);
                if (!rows) {
                    ok = false;
                    break;
                }
                ++encoded;
                q.len = rows->size() / (size_t) m.dim();
                bool fits = (size_t) q.start + q.len <= r.ids.size() - 1;
                for (size_t i = 0; fits && i < q.len; ++i) fits = r.ids[(size_t) q.start + i] == img_tok;
                if (!fits) {
                    err = "a picture's span (" + std::to_string(q.len) + " tokens at " + std::to_string(q.start) +
                          ") is not a run of image tokens in the prompt";
                    ok = false;
                    break;
                }
                imgs.push_back({q.start, rows});
            }
            if (ok) {
                m.set_images(std::move(imgs));
                std::fprintf(stderr, "maya ds4 serve: %zu pictures, %d read (%.0f ms)\n", pics.size(), encoded,
                             now_ms() - te);
            }
            for (Pic& q : pics) q.rgb = {};   // (only the start, hash and span stay)
        }
        // the prompt in windows (PP <read> <total> <ms> <tok/s> <reuse> every 64 tokens), a checkpoint before its last
        // token and one after it: the next turn re-renders this one, and shares its start up to one of them
        const size_t total = r.ids.size();
        size_t done = from, next_pp = from + 64;
        m.set_progress([&](int64_t d) {   // (a run of chunks: each one through every layer)
            const double ms = now_ms() - t0;
            std::printf("PP %zu %zu %.1f %.1f %zu\n", done + (size_t) d, r.ids.size(), ms,
                        ms > 0 ? 1000.0 * (double) (done + (size_t) d - from) / ms : 0.0, from);
            std::fflush(stdout);
        });
        // (DS4_LAST_WINDOW=1: the last token in a window of its own after the checkpoint, as before - for A/B runs)
        static const bool last_window = std::getenv("DS4_LAST_WINDOW") != nullptr;
        // (pin=N: the read stops at N for a checkpoint of the shared start; only when this read crosses N, and N leaves
        //  the prompt's last token to the usual end)
        const size_t pin_at = (r.pin >= 256 && (size_t) r.pin + 1 < total) ? (size_t) r.pin : 0;
        // a STOP while the prompt is read ends it there (the engine's run of chunks at a layer or chunk, windows between
        // feeds) and goes back to the state before this prompt - kept here first, in checkpoint 1, which the prompt's
        // own end overwrites as before - so the next request still shares the conversation up to it
        if (from > 0 && from < total) m.checkpoint(1);
        m.set_cancel([&in] { return in.stop.load(); });
        m.set_prompt_total(total > from ? (int64_t) (total - from) : 0);   // (DS4_PF_HELPER_MAXP: this read's new tokens)
        bool stopped = false;
        while (ok && done < total) {
            if (in.stop.load()) {
                stopped = true;
                break;
            }
            size_t n = std::min<size_t>((size_t) m.feed_unit(), total - done);
            const double tc = now_ms();   // (srvbatch: the chunk's time, for the banks' share below)
            if (pin_at > done && pin_at < done + n) n = pin_at - done;
            // (the prompt's end: the last token in the same run, the checkpoint before it kept on the way)
            const bool fin = done + n == total;
            if (fin && last_window) {
                ok = (n < 2 || m.feed(r.ids.data() + done, (int) n - 1, 0, err));
                if (ok) m.checkpoint(1);
                ok = ok && m.feed(r.ids.data() + done + n - 1, 1, 1, err);
            } else
                ok = fin ? m.feed_mark(r.ids.data() + done, (int) n, 1, 1, err)
                         : m.feed(r.ids.data() + done, (int) n, 0, err);
            done += n;
            if (ok && pin_at != 0 && done == pin_at) m.checkpoint(3);
            // srvbatch: the banks decode while this prompt is read.  After each chunk but the last, batch steps for
            // bt_share / (1 - bt_share) of that chunk's time (one at least): a long read slows the decoders down, it never
            // stalls them, and its own speed drops by the same share.  The steps touch only the banks' states, never the
            // conversation being read here (each feed() is whole, so its state is complete between two of them).
            if (ok && !fin && bt_share > 0.0 && pics.empty() && bank_live()) {
                const double until = now_ms() + (now_ms() - tc) * bt_share / (1.0 - bt_share);
                do {
                    std::string e;
                    if (!batch_tick(e)) {
                        err = "a batch step failed: " + e;
                        bt_fatal = true;   // (the engine ends at the top of the loop, after this request's ERR)
                        ok = false;
                        break;
                    }
                } while (bank_live() && now_ms() < until);
            }
            if (ok && (done >= next_pp || done == total)) {
                const double ms = now_ms() - t0;
                std::printf("PP %zu %zu %.1f %.1f %zu\n", done, r.ids.size(), ms,
                            ms > 0 ? 1000.0 * (double) (done - from) / ms : 0.0, from);
                std::fflush(stdout);
                next_pp = done + 64;
            }
        }
        m.set_cancel(nullptr);
        m.set_prompt_total(0);
        m.set_images({});
        if (!ok && gpu_fatal_text(err)) die_gpu_fatal(err);   // (upstream fix C3)
        if (stopped || (!ok && m.cancelled())) {
            const bool back = from > 0 && m.restore((int64_t) from) && (size_t) m.pos() == from;
            if (!back) {
                m.reset();
                held.clear();
            }
            std::fprintf(stderr, "ds4 serve: stopped while reading the prompt of %zu tokens, after %.0f ms%s\n",
                         total, now_ms() - t0, back ? "; the conversation before it is kept" : "");
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0\n", r.ids.size(), now_ms() - t0, back ? from : (size_t) 0);
            std::fflush(stdout);
            continue;
        }
        if (!ok) {
            std::fprintf(stderr, "ds4: %s\n", err.c_str());
            std::printf("ERR %s\n", err.c_str());
            std::printf("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", r.ids.size());
            admit_end.b = -1;
            std::fflush(stdout);
            m.reset();
            held.clear();
            continue;
        }
        held = std::move(pics);
        m.checkpoint(0);
        if (!continued && slot_max > 0) m.pin();   // (a new run of prompts: where it may be rendered again from)
        const double t1 = now_ms();
        long long produced = 0;
        const char* finish = "length";
        const long long limit = r.max_new > 0 ? r.max_new : m.max_context() - (long long) r.ids.size();
        long long d_off = 0, d_acc = 0;
        bool cool = false;
        int tok = m.sample(r.temperature, r.top_p, r.top_k, r.min_p, r.seed, counter++);
        std::vector<int> got;
        // STAT for the Monitor every 8 tokens and at the end, over this answer so far: the GLM engine's keys (speed, the
        // VRAM hit, promotions a token) and DeepSeek's own - experts a token on the CPU lane, MB a token the storage
        // read (Linux), the n-gram memory's (Engram) read time a token, the drafts' acceptance
        const strata::ds4::Stats s_dec = m.stats();
        const long long io_dec = io_read_bytes();
        auto stat_line = [&]() {
            const strata::ds4::Stats& s = m.stats();
            const double ms = now_ms() - t1;
            const double fed = (double) std::max<int64_t>(1, s.tokens - s_dec.tokens);
            const double ex = (double) (s.experts - s_dec.experts), cx = (double) (s.cpu_experts - s_dec.cpu_experts);
            std::string out = "STAT";
            char b[160];
            std::snprintf(b, sizeof b, " tok_s=%.2f ms_tok=%.2f vram_hit=%.4f cpu=%.3f ram=%.3f promo=%.3f engram_ms=%.3f",
                          ms > 0 ? 1000.0 * (double) produced / ms : 0.0, produced > 0 ? ms / (double) produced : 0.0,
                          ex > 0 ? 1.0 - cx / ex : 1.0, cx / fed, (double) (s.ram_experts - s_dec.ram_experts) / fed,
                          (double) (s.swaps - s_dec.swaps) / fed, (s.engram_ms - s_dec.engram_ms) / fed);
            out += b;
            const long long io = io_read_bytes();
            if (io >= 0 && io_dec >= 0) {
                std::snprintf(b, sizeof b, " disk_mb=%.3f", (double) (io - io_dec) / 1e6 / fed);
                out += b;
            }
            if (d_off > 0) {
                std::snprintf(b, sizeof b, " drafts=%.3f", (double) d_acc / (double) d_off);
                out += b;
            }
            if (s.skipped > s_dec.skipped) {   // (STRATA_DS4_DISK_SKIP_W: experts left out a token)
                std::snprintf(b, sizeof b, " skipped=%.3f", (double) (s.skipped - s_dec.skipped) / fed);
                out += b;
            }
            std::printf("%s\n", out.c_str());
        };
        for (;;) {
            if (in.stop.load()) {
                finish = "cancel";
                break;
            }
            if (tok < 0) {   // (a sample's CUDA error, printed by the sampler: no T -1 to the client; Maya #619)
                std::fprintf(stderr, "ds4: the sample failed\n");
                finish = "cancel";
                break;
            }
            std::printf("T %d\n", tok);
            ++produced;
            if (produced % 8 == 0) stat_line();
            std::fflush(stdout);
            bool end = false;
            for (int e : eos) end = end || tok == e;
            if (end) {
                finish = "stop";
                break;
            }
            if (produced >= limit || m.pos() >= m.max_context()) break;
            if (got.empty()) {   // the next round: the drafts (if any) and the token after them
                const int room = (int) std::min<long long>(limit - produced, m.max_context() - m.pos());
                int off = 0, acc = 0;
                // a round whose drafts were all below the confidence floor: the next one does without the drafter
                // (a request's spec / dspark_conf keys over the engine's: the speed calibration measures both ways)
                const bool try_spec = (r.spec < 0 ? spec : r.spec != 0) && !cool;
                if (!round(m, tok, room, try_spec, r.conf >= 0 ? r.conf : conf_min, r.temperature, r.top_p, r.top_k,
                           r.min_p, r.seed, counter, got,
                           off, acc, err, r.cum)) {
                    if (gpu_fatal_text(err)) die_gpu_fatal(err);   // (upstream fix C3)
                    std::fprintf(stderr, "ds4: %s\n", err.c_str());
                    finish = "cancel";
                    break;
                }
                d_off += off;
                d_acc += acc;
                cool = try_spec && off == 0;
                std::reverse(got.begin(), got.end());
            }
            tok = got.back();
            got.pop_back();
        }
        // (a stop within accepted drafts leaves the state past it: the next prompt shares less with it, and goes back to
        //  the checkpoint at this prompt's end)
        stat_line();
        const double t2 = now_ms();
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0\n", produced, r.ids.size(), t1 - t0, t2 - t1, finish,
                    d_acc, d_off, from);
        std::fflush(stdout);
        // srvbatch: an admission whose one token did not end it goes on in its bank.  The state (the prompt read, the
        // checkpoints this read kept, the token not yet fed) trades places with the bank's; the bank's first step feeds
        // that token, as the solo path's next round would.  Anything else: BADM 0 (admit_end), the request is done.
        if (admit >= 0 && produced == 1 && std::strcmp(finish, "length") == 0 && admit_max_new > 1 && !in.stop.load() &&
            m.pos() + 1 < m.max_context() && held.empty() && bank_trade(admit)) {
            BankRun& x = bk[(size_t) admit];
            x = BankRun{};
            x.active = true;
            x.cur = tok;
            x.produced = 1;
            x.max_new = admit_max_new;
            x.temperature = r.temperature;
            x.top_p = r.top_p;
            x.top_k = r.top_k;
            x.min_p = r.min_p;
            x.seed = r.seed;
            x.counter = counter;
            // srvbatch12 fix 3: the bank's counter range [counter, counter + max_new + 16) is its own - the next admission
            // (and the solo path) draw past it.  Without this every seed-0 request's stream overlapped the previous bank's
            counter += (unsigned long long) admit_max_new + 16ull;
            x.spec = r.spec;
            x.conf = r.conf;
            x.cum = r.cum;
            x.t0 = now_ms();
            in.bstop[admit].store(false);   // (a BSTOP for the bank's last request, late: not this one's)
            std::printf("BADM %d 1\n", admit);
            std::fflush(stdout);
            admit_end.b = -1;
        }
        const auto& s = m.stats();
        std::fprintf(stderr,
                     "maya ds4 serve: prompt %zu tokens = %zu reused + %zu read in %.0f ms (%.1f tok/s), %lld generated "
                     "in %.0f ms (%.2f tok/s); CPU lane %.1f%% of the experts\n",
                     r.ids.size(), from, r.ids.size() - from, t1 - t0,
                     t1 > t0 ? 1000.0 * (double) (r.ids.size() - from) / (t1 - t0) : 0.0, produced, t2 - t1,
                     t2 > t1 ? 1000.0 * (double) produced / (t2 - t1) : 0.0,
                     s.experts ? 100.0 * (double) s.cpu_experts / (double) s.experts : 0.0);
        if (d_off > 0) std::fprintf(stderr, "maya ds4 serve: drafts %lld of %lld kept; %s\n", d_acc, d_off,
                                    per_position(s_dec, s).c_str());
        m.save_profile();
    }
    // the engine ends (QUIT or the end of its input): the conversation in the model is set aside, and every one in
    // RAM is written to the folder, so the next run reads on from them
    if (slot_max > 0) {
        if (held.empty()) slot_put(" as the engine ended");
        size_t written = 0;
        while (true) {   // (a write can take the folder's least recently used conversation away: the list is asked again)
            size_t i = SIZE_MAX;
            for (size_t k = 0; k < slots.size(); ++k)
                if (slots[k].data) {
                    i = k;
                    break;
                }
            if (i == SIZE_MAX) break;
            if (slot_write_file(slots[i])) ++written;
            else slot_drop(i);   // (its state could not be written: there is nothing more to do at the engine's end)
        }
        if (written > 0) {
            int ram = 0, disk = 0;
            slot_count(ram, disk);
            char msg[256];
            std::snprintf(msg, sizeof msg, "wrote %zu conversation(s) to %s; %d there in all", written,
                          slot_dir.string().c_str(), disk);
            note(msg);
        }
    }
    settle_stop();   // (the idle thread must be off the cache before the profile is saved and the engine ends)
    m.save_profile();
    return 0;
}


// ---- Strata addition: teacher-forced scoring (the quality gate) -------------------------------------------------
// --score IDS.json --score-out RUN.tsv: every segment of a tools/quant_quality.py ids file (name + ids) is read through
// the model from a fresh state; per position the log-probability of the text's next token and the 64 most likely
// tokens go out in the STRATA_LOGPOS row format that `quant_quality.py compare` reads (natural logs, full-vocabulary
// log-softmax in double).  RUN.tsv.seg.json lists each segment's first line and size.
struct ScoreSeg {
    std::string name;
    std::vector<int> ids;
};

bool read_score_ids(const std::string& path, std::vector<ScoreSeg>& segs, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = path + ": cannot open";
        return false;
    }
    std::string s;
    char buf[1 << 16];
    size_t k;
    while ((k = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, k);
    std::fclose(f);
    size_t p = 0;
    for (;;) {
        const size_t n0 = s.find("\"name\"", p);
        if (n0 == std::string::npos) break;
        const size_t q0 = s.find('"', s.find(':', n0) + 1);
        const size_t q1 = s.find('"', q0 + 1);
        const size_t i0 = s.find("\"ids\"", q1);
        if (q0 == std::string::npos || q1 == std::string::npos || i0 == std::string::npos) break;
        const size_t b0 = s.find('[', i0), b1 = s.find(']', b0);
        if (b0 == std::string::npos || b1 == std::string::npos) break;
        ScoreSeg g;
        g.name = s.substr(q0 + 1, q1 - q0 - 1);
        g.ids = parse_ids(s.substr(b0 + 1, b1 - b0 - 1));
        segs.push_back(std::move(g));
        p = b1;
    }
    if (segs.empty()) {
        err = path + ": no segments (expected quant_quality.py's ids file)";
        return false;
    }
    return true;
}

int score(strata::ds4::Model& m, const std::string& ids_path, const std::string& out_path, int window, int eos) {
    std::vector<ScoreSeg> segs;
    std::string err;
    if (!read_score_ids(ids_path, segs, err)) {
        std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
        return 1;
    }
    // DS4_SCORE_PREFIX=N (a test of the long-prompt layer skip): the segments are joined into one stream, its first N
    // tokens are read as one prompt (logits of the last row only: the path a long prompt takes, where the skip applies)
    // and only the positions after them are scored, in windows
    const long long prefix_n = std::getenv("DS4_SCORE_PREFIX") ? std::atoll(std::getenv("DS4_SCORE_PREFIX")) : 0;
    if (prefix_n > 0 && segs.size() > 1) {
        ScoreSeg all;
        all.name = "concat";
        for (const ScoreSeg& sg : segs) all.ids.insert(all.ids.end(), sg.ids.begin(), sg.ids.end());
        segs.assign(1, all);
    }
    std::FILE* out = std::fopen(out_path.c_str(), "wb");
    std::FILE* sj = std::fopen((out_path + ".seg.json").c_str(), "wb");
    if (out == nullptr || sj == nullptr) {
        std::fprintf(stderr, "strata-dsv41: cannot write %s\n", out_path.c_str());
        return 1;
    }
    std::fprintf(sj, "{\"segments\": [");
    const int V = m.vocab();
    long long line = 0;
    std::vector<float> lg;
    std::vector<int> order;
    const double t0 = now_ms();
    for (size_t si = 0; si < segs.size(); ++si) {
        const std::vector<int>& ids = segs[si].ids;
        m.reset();
        const long long first = line;
        double sum_nll = 0.0;
        long long top1 = 0, cnt = 0;
        size_t a0 = 0;
        if (prefix_n > 0 && (size_t) prefix_n + 1 < ids.size()) {
            a0 = (size_t) prefix_n;
            if (!m.feed(ids.data(), (int) a0, 1, err)) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
                return 1;
            }
        }
        for (size_t a = a0; a + 1 < ids.size();) {
            const int T = (int) std::min<size_t>((size_t) window, ids.size() - 1 - a);   // (positions a .. a+T-1 are scored)
            if (!m.feed(ids.data() + a, T, T, err) || !m.logits(lg) || lg.size() < (size_t) T * (size_t) V) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.empty() ? "no logits" : err.c_str());
                return 1;
            }
            for (int t = 0; t < T; ++t) {
                const float* r = lg.data() + (size_t) t * (size_t) V;
                float mx = r[0];
                for (int v = 1; v < V; ++v) mx = std::max(mx, r[v]);
                double z = 0.0;
                for (int v = 0; v < V; ++v) z += std::exp((double) (r[v] - mx));
                const double lz = std::log(z) + (double) mx;
                order.resize((size_t) V);
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin() + 64, order.end(), [&](int x, int y) {
                    return r[x] > r[y] || (r[x] == r[y] && x < y);
                });
                const int tgt = ids[a + (size_t) t + 1];
                const double lt = (double) r[tgt] - lz;
                const double la = (double) r[order[0]] - lz;
                const double le = eos >= 0 && eos < V ? (double) r[eos] - lz : -1e30;
                // logp(target) with the end token left out of the distribution, as the engine's row has it
                const double lt_ne = tgt == eos ? lt : std::log(std::max(1e-300, std::exp(lt) / std::max(1e-300, 1.0 - std::exp(le))));
                std::fprintf(out, "%zu\t%d\t%.6f\t%d\t%.6f\t%d\t%.6f\t%.6f", a + (size_t) t, tgt, lt, order[0], la,
                             order[0] == tgt ? 1 : 0, le, lt_ne);
                for (int j = 0; j < 64; ++j) std::fprintf(out, "\t%d:%.6f", order[(size_t) j], (double) r[order[(size_t) j]] - lz);
                std::fprintf(out, "\n");
                ++line;
                ++cnt;
                top1 += order[0] == tgt;
                sum_nll -= lt;
            }
            a += (size_t) T;
        }
        std::fprintf(sj, "%s{\"name\": \"%s\", \"first_line\": %lld, \"end_line\": %lld, \"n\": %lld, \"ppl\": %.6f, \"top1\": %.6f}", si ? ", " : "",
                     segs[si].name.c_str(), first, first + cnt, cnt, std::exp(sum_nll / (double) std::max(1LL, cnt)),
                     (double) top1 / (double) std::max(1LL, cnt));
        std::fprintf(stderr, "score %s: %lld positions, ppl %.4f, top-1 %.4f (%.0f s)\n", segs[si].name.c_str(), cnt,
                     std::exp(sum_nll / (double) std::max(1LL, cnt)), (double) top1 / (double) std::max(1LL, cnt),
                     (now_ms() - t0) / 1000.0);
        std::fflush(out);
    }
    {   // who computed the routed experts over the whole score (so a helper arm can be seen to have used the helper)
        const auto& s = m.stats();
        std::fprintf(stderr, "score experts: %lld routed, %lld on the CPU lane, %lld on the helper; the cache moved %lld\n",
                     (long long) s.experts, (long long) s.cpu_experts, (long long) s.helper_experts, (long long) s.swaps);
    }
    std::fprintf(sj, "]}\n");
    std::fclose(sj);
    std::fclose(out);
    return 0;
}
// ---- Strata H100 addition: profile run -------------------------------------------------------------------------
// --profile-run IDS.json: every segment is fed through the model from a fresh state (no logits); the engine's expert
// profile (--expert-profile FILE, counts a layer and expert) and DS4_TRACE collect the routing.  The profile is saved
// every 10 segments.  --window sets the chunk (a chunked prompt path above 16).
int profile_run(strata::ds4::Model& m, const std::string& ids_path, int window) {
    std::vector<ScoreSeg> segs;
    std::string err;
    if (!read_score_ids(ids_path, segs, err)) {
        std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
        return 1;
    }
    const double t0 = now_ms();
    long long tot = 0;
    for (size_t si = 0; si < segs.size(); ++si) {
        const std::vector<int>& ids = segs[si].ids;
        m.reset();
        for (size_t a = 0; a < ids.size();) {
            const int T = (int) std::min<size_t>((size_t) window, ids.size() - a);
            if (!m.feed(ids.data() + a, T, 0, err)) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
                return 1;
            }
            a += (size_t) T;
        }
        tot += (long long) ids.size();
        std::fprintf(stderr, "profile-run %zu/%zu %s: %zu tokens, %lld in all, %.0f s, %.1f tok/s\n", si + 1, segs.size(),
                     segs[si].name.c_str(), ids.size(), tot, (now_ms() - t0) / 1000.0, 1000.0 * (double) tot / (now_ms() - t0));
        if ((si + 1) % 10 == 0) m.save_profile();
    }
    m.save_profile();
    return 0;
}
// ---- Strata H100 addition: DSpark acceptance log ---------------------------------------------------------------
// --spec-eval IDS.json --n N: each segment is a prompt: fed from reset, then N greedy tokens decoded with the drafter, every
// draft verified (--dspark-conf 0); DS4_ROUND_LOG=file gets a line a round (segment name, drafts, accepted, confidences).
int spec_eval(strata::ds4::Model& m, const std::string& ids_path, int n, int window, float conf_min, bool spec) {
    std::vector<ScoreSeg> segs;
    std::string err;
    if (!read_score_ids(ids_path, segs, err)) {
        std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
        return 1;
    }
    if (const char* f = std::getenv("DS4_ROUND_LOG")) g_round_log = std::fopen(f, "wb");
    const double t0 = now_ms();
    for (size_t si = 0; si < segs.size(); ++si) {
        const std::vector<int>& ids = segs[si].ids;
        g_round_tag = segs[si].name;
        m.reset();
        for (size_t a = 0; a < ids.size();) {
            const int T = (int) std::min<size_t>((size_t) window, ids.size() - a);
            const bool last = a + (size_t) T == ids.size();
            if (!m.feed(ids.data() + a, T, last ? 1 : 0, err)) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
                return 1;
            }
            a += (size_t) T;
        }
        unsigned long long counter = 0;
        int tok = m.sample(0.0f, 1.0f, 1, 0.0f, 1, counter++);
        const double td0 = now_ms();   // (DS4_ROUND_LOG_MS: the segment's decode time, after its prompt)
        std::vector<int> got;
        int emitted = 1;
        long long rounds = 0, acc_all = 0;
        bool stop = false;
        while (emitted < n && !stop) {
            int off = 0, acc = 0;
            if (!round(m, tok, n - emitted, spec, conf_min, 0.0f, 1.0f, 1, 0.0f, 1, counter, got, off, acc, err)) {
                std::fprintf(stderr, "strata-dsv41: %s\n", err.c_str());
                return 1;
            }
            ++rounds;
            acc_all += acc;
            for (int t : got) {
                ++emitted;
                if (t == 1) stop = true;
            }
            tok = got.back();
        }
        std::fprintf(stderr, "spec-eval %zu/%zu %s: prompt %zu, %d tokens, %lld rounds, %.2f tokens a round, accepted %lld (%.0f s)\n", si + 1,
                     segs.size(), segs[si].name.c_str(), ids.size(), emitted, rounds, (double) (emitted - 1) / (double) std::max(1LL, rounds),
                     acc_all, (now_ms() - t0) / 1000.0);
        if (g_round_log_ms)   // DS4_ROUND_LOG_MS: decode only (the rounds after the prompt's first token)
            std::fprintf(stderr, "spec-eval decode %s: %d tokens in %.1f ms (%.3f ms a token), %lld rounds\n", segs[si].name.c_str(),
                         emitted - 1, now_ms() - td0, emitted > 1 ? (now_ms() - td0) / (double) (emitted - 1) : 0.0, rounds);
        if (g_round_log != nullptr) std::fflush(g_round_log);
    }
    return 0;
}
}  // namespace

#ifndef DS4_BUILD_ID
#define DS4_BUILD_ID "unknown"
#endif

#if defined(STRATA_METAL)
#include "sampler_stat.inc"
#endif
// "1" or "1,4": the helper expert cache's devices (a negative or unreadable entry is dropped: helper off)
static std::vector<int> parse_device_list(const std::string& s) {
    std::vector<int> v;
    size_t i = 0;
    while (i <= s.size()) {
        const size_t j = std::min(s.find(',', i), s.size());
        const std::string t = s.substr(i, j - i);
        if (!t.empty() && (std::isdigit((unsigned char) t[0]))) v.push_back(std::atoi(t.c_str()));
        i = j + 1;
    }
    return v;
}

// ---- DS4_BATCH_DECODE: the batched decode's test (docs/dsv41/cards/SV-03.md) ------------------------------------------
// --batch-prompts A,B,C,D [--batch-sizes 1,2,4] --n N: each prompt (a file of comma-separated ids, as --ids @file) is first
// decoded alone - the normal prompt path, then N greedy windows of one on the conversation in the model (the reference) -
// and then, for each size B, prompts 0 .. B-1 are read into banks 0 .. B-1 and decoded together, N tokens each (the first
// from the prompt, then N - 1 batch steps).  Each sequence's tokens are compared with its own reference: the same process,
// so the same expert placement (run it with --no-expert-cache so the placement cannot move between them).  stdout: a line
// a sequence and run ("solo 2: ids..." / "B4 seq 2: ids..."); stderr: speeds, the comparison, the expert counts a step.
// rc 0: every sequence identical to its reference; 3: one differs; 1: an error.
static uint64_t fnv_ids(const std::vector<int>& v) {
    uint64_t h = 1469598103934665603ull;
    for (int t : v) {
        const uint32_t u = (uint32_t) t;
        for (int i = 0; i < 4; ++i) h = (h ^ ((u >> (8 * i)) & 0xff)) * 1099511628211ull;
    }
    return h;
}
// DS4_BATCH_DSPARK: --batch-rows 1,2,3,4 (teacher-forced: each step feeds every sequence T of its reference tokens as
// one batch window - the verify-shaped cost against T = 1, each row's argmax checked against the reference) and
// --batch-spec (drafted batch decoding: each sequence drafts from its own bank, one window verifies all; compared with
// its reference and with its own solo drafted run, which goes through round() as the CLI's decode does).
int batch_test(strata::ds4::Model& m, const std::string& files, const std::string& sizes_s, int n,
               const std::string& rows_s = "", bool spec = false, float conf_min = 0.5f) {
    std::string err;
    std::vector<std::vector<int>> prompts;
    {
        size_t i = 0;
        while (i <= files.size()) {
            const size_t j = std::min(files.find(',', i), files.size());
            const std::string f = files.substr(i, j - i);
            i = j + 1;
            if (f.empty()) continue;
            std::ifstream idf(f);
            std::stringstream ss;
            ss << idf.rdbuf();
            prompts.push_back(parse_ids(ss.str()));
            if (prompts.back().empty()) {
                std::fprintf(stderr, "batch-test: %s: no ids\n", f.c_str());
                return 1;
            }
        }
    }
    std::vector<int> sizes = parse_ids(sizes_s);
    if (sizes.empty())
        for (int b = 1; b <= (int) prompts.size(); ++b) sizes.push_back(b);
    const int P = (int) prompts.size();
    int bmax = 0;
    for (int b : sizes) bmax = std::max(bmax, b);
    if (n < 1 || P < 1 || bmax > P || bmax > m.batch_slots() || sizes.front() < 1) {
        std::fprintf(stderr, "batch-test: %d prompts, sizes up to %d, %d banks (DS4_BATCH_DECODE), --n %d: needs sizes <= the prompts "
                             "and the banks, --n >= 1\n", P, bmax, m.batch_slots(), n);
        return 1;
    }
    const int unit = std::max(16, std::min(m.feed_unit(), 1 << 30));
    const auto line = [](const char* tag, int i, const std::vector<int>& v) {
        std::printf("%s %d:", tag, i);
        for (size_t k = 0; k < v.size(); ++k) std::printf("%s%d", k ? "," : " ", v[k]);
        std::printf("\n");
        std::fflush(stdout);
    };
    // the references: each prompt alone, windows of one
    std::vector<std::vector<int>> ref((size_t) P);
    double solo_tps = 0;
    for (int i = 0; i < P; ++i) {
        m.reset();
        const std::vector<int>& ids = prompts[(size_t) i];
        for (size_t a = 0; a < ids.size();) {
            const int T = (int) std::min<size_t>((size_t) unit, ids.size() - a);
            if (!m.feed(ids.data() + a, T, a + (size_t) T == ids.size() ? 1 : 0, err)) {
                std::fprintf(stderr, "batch-test: %s\n", err.c_str());
                return 1;
            }
            a += (size_t) T;
        }
        std::vector<int>& out = ref[(size_t) i];
        out.reserve((size_t) n);
        int tok = m.sample(0.0f, 1.0f, 1, 0.0f, 1, 0);
        out.push_back(tok);
        const double t0 = now_ms();
        for (int k = 1; k < n; ++k) {
            if (!m.step(tok, true, err)) {
                std::fprintf(stderr, "batch-test: %s\n", err.c_str());
                return 1;
            }
            tok = m.sample(0.0f, 1.0f, 1, 0.0f, 1, (unsigned long long) k);
            out.push_back(tok);
        }
        const double ms = now_ms() - t0, tps = n > 1 ? 1000.0 * (n - 1) / ms : 0.0;
        solo_tps += tps / P;
        std::fprintf(stderr, "batch-test solo %d: prompt %zu tokens, %d decoded in %.0f ms (%.2f tok/s), ids %016llx\n", i,
                     ids.size(), n, ms, tps, (unsigned long long) fnv_ids(out));
        line("solo", i, out);
    }
    m.reset();
    int rc = 0;
    const auto prefill = [&](int B, std::vector<int>& cur, std::vector<std::vector<int>>& out) -> bool {
        cur.assign((size_t) B, 0);
        out.assign((size_t) B, {});
        for (int b = 0; b < B; ++b) {
            const std::vector<int>& ids = prompts[(size_t) b];
            if (!m.batch_prefill(b, ids.data(), (int) ids.size(), cur[(size_t) b], err)) {
                std::fprintf(stderr, "batch-test: bank %d: %s\n", b, err.c_str());
                return false;
            }
            out[(size_t) b].reserve((size_t) n + 16);
            out[(size_t) b].push_back(cur[(size_t) b]);
        }
        return true;
    };
    const auto compare = [&](int B, const std::vector<std::vector<int>>& out, const char* tag, std::string& cmp,
                             const std::vector<int>* lens = nullptr) -> int {   // (lens: each sequence's own length)
        int same = 0;
        for (int b = 0; b < B; ++b) {
            const int len = lens != nullptr ? std::min(n, (*lens)[(size_t) b]) : n;
            std::vector<int> r = ref[(size_t) b];
            if ((int) r.size() > len) r.resize((size_t) len);
            std::vector<int> v = out[(size_t) b];
            if ((int) v.size() > len) v.resize((size_t) len);
            size_t at = 0;
            while (at < r.size() && at < v.size() && r[at] == v[at]) ++at;
            char c[96];
            if (at == r.size() && at == v.size()) {
                ++same;
                std::snprintf(c, sizeof c, " seq %d identical;", b);
            } else
                std::snprintf(c, sizeof c, " seq %d DIFFERS at token %zu;", b, at);
            cmp += c;
            char t[24];
            std::snprintf(t, sizeof t, "%s%d seq", tag, B);
            line(t, b, v);
        }
        return same;
    };
    if (!rows_s.empty()) {   // the verify-shaped cost: B sequences x T rows a step, teacher-forced from the references
        for (int B : sizes)
            for (int T : parse_ids(rows_s)) {
                if (T < 1 || B * T > 16) {
                    std::fprintf(stderr, "batch-test rows: B=%d x T=%d is more than 16 rows: skipped\n", B, T);
                    continue;
                }
                std::vector<int> cur;
                std::vector<std::vector<int>> out;
                if (!prefill(B, cur, out)) return 1;
                std::vector<int> bk, tk, nx;
                int steps = 0, rows_n = 0, rows_ok = 0;
                const strata::ds4::Stats s0 = m.stats();
                const double t0 = now_ms();
                for (;;) {
                    const int k = (int) out[0].size() - 1;   // (every sequence is at the same index)
                    const int te = std::min(T, n - 1 - k);
                    if (te <= 0) break;
                    bk.clear();
                    tk.clear();
                    for (int b = 0; b < B; ++b)
                        for (int t = 0; t < te; ++t) {
                            bk.push_back(b);
                            tk.push_back(ref[(size_t) b][(size_t) (k + t)]);
                        }
                    nx.assign(bk.size(), 0);
                    if (!m.batch_step(bk.data(), tk.data(), (int) bk.size(), nx.data(), err)) {
                        std::fprintf(stderr, "batch-test rows B=%d T=%d: %s\n", B, T, err.c_str());
                        return 1;
                    }
                    for (int b = 0; b < B; ++b)
                        for (int t = 0; t < te; ++t) {
                            const int got = nx[(size_t) (b * te + t)], want = ref[(size_t) b][(size_t) (k + t + 1)];
                            ++rows_n;
                            rows_ok += got == want;
                            out[(size_t) b].push_back(want);
                        }
                    ++steps;
                }
                const double ms = now_ms() - t0;
                const strata::ds4::Stats& s1 = m.stats();
                const double toks = (double) B * (n - 1);
                if (rows_ok != rows_n) rc = 3;
                std::fprintf(stderr, "batch-test rows B=%d T=%d: %d steps of %d rows in %.0f ms (%.1f ms a step, %.2f ms a row): "
                                     "%.2f tok/s in all; row argmax = reference: %d of %d; helpers %.1f experts / %.2f ms a step\n",
                             B, T, steps, B * T, ms, ms / std::max(1, steps), ms / std::max(1.0, toks), 1000.0 * toks / ms,
                             rows_ok, rows_n, (double) (s1.helper_experts - s0.helper_experts) / std::max(1, steps),
                             (s1.helper_ms - s0.helper_ms) / std::max(1, steps));
                const long long ma_sh = (long long) (s1.mattn_shared - s0.mattn_shared), ma_pr = (long long) (s1.mattn_per_run - s0.mattn_per_run);
                if (ma_sh + ma_pr > 0)   // (DS4_MULTI_ATTN set)
                    std::fprintf(stderr, "batch-test rows B=%d T=%d attention (DS4_MULTI_ATTN): %lld member-layer passes in shared "
                                         "launches, %lld per run\n", B, T, ma_sh, ma_pr);
            }
        return rc;
    }
    if (spec) {   // drafted batch decoding
        if (!m.batch_spec_ok()) {
            std::fprintf(stderr, "batch-test: --batch-spec needs DS4_BATCH_DSPARK=1 and the drafter loaded (--dspark on)\n");
            return 1;
        }
        // each prompt's solo drafted run (the CLI's round(), on the conversation in the model): its tokens and tok/s
        double dsolo = 0;
        for (int i = 0; i < P; ++i) {
            m.reset();
            const std::vector<int>& ids = prompts[(size_t) i];
            for (size_t a = 0; a < ids.size();) {
                const int T = (int) std::min<size_t>((size_t) unit, ids.size() - a);
                if (!m.feed(ids.data() + a, T, a + (size_t) T == ids.size() ? 1 : 0, err)) {
                    std::fprintf(stderr, "batch-test: %s\n", err.c_str());
                    return 1;
                }
                a += (size_t) T;
            }
            std::vector<int> out, got;
            out.reserve((size_t) n + 16);
            unsigned long long counter = 0;
            int tok = m.sample(0.0f, 1.0f, 1, 0.0f, 1, counter++);
            bool cool = false;
            long long rounds = 0, off_n = 0, acc_n = 0;
            const double t0 = now_ms();
            for (;;) {
                out.push_back(tok);
                if ((int) out.size() >= n) break;
                if (got.empty()) {
                    int off = 0, acc = 0;
                    const bool try_spec = !cool;
                    if (!round(m, tok, n - (int) out.size(), try_spec, conf_min, 0.0f, 1.0f, 1, 0.0f, 1, counter, got, off, acc,
                               err)) {
                        std::fprintf(stderr, "batch-test: %s\n", err.c_str());
                        return 1;
                    }
                    ++rounds;
                    off_n += off;
                    acc_n += acc;
                    cool = try_spec && off == 0;
                    std::reverse(got.begin(), got.end());
                }
                tok = got.back();
                got.pop_back();
            }
            const double ms = now_ms() - t0, tps = 1000.0 * (n - 1) / ms;
            dsolo += tps / P;
            size_t at = 0;
            while (at < out.size() && at < ref[(size_t) i].size() && out[at] == ref[(size_t) i][at]) ++at;
            std::fprintf(stderr, "batch-test drafted solo %d: %d decoded in %.0f ms (%.2f tok/s), %lld rounds, %lld offered / %lld "
                                 "accepted; vs the plain solo: %s\n", i, n, ms, tps, rounds, off_n, acc_n,
                         at == out.size() && at == ref[(size_t) i].size() ? "identical" : ("DIFFERS at token " + std::to_string(at)).c_str());
            line("dsolo", i, out);
        }
        m.reset();
        const float cum_min = dspark_cum_default(m);
        // DS4_BATCH_DSPARK_MAX (the scheduler's rule): a step drafts while at most smax sequences are active, else it is
        // a plain batch step; decided step by step.  Equal lengths keep the count fixed, so when smax < the largest B, a
        // last run at that B gives the sequences staggered lengths (n, n - n/B, ..): sequences leave, and the steps
        // switch from plain to drafted on the way down (exactness against each one's reference prefix)
        const int smax = m.batch_spec_max();
        std::vector<std::pair<int, bool>> jobs;
        for (int B : sizes) jobs.push_back({B, false});
        if (smax < bmax && bmax > 1) jobs.push_back({bmax, true});
        for (const auto& job : jobs) {
            const int B = job.first;
            const bool stag = job.second;
            std::vector<int> want((size_t) B, n);
            if (stag)
                for (int b = 0; b < B; ++b) want[(size_t) b] = std::max(2, n - b * (n / B));
            long long want_all = 0;
            for (int b = 0; b < B; ++b) want_all += want[(size_t) b] - 1;
            const std::string name = std::to_string(B) + (stag ? " staggered" : "");
            std::vector<int> cur;
            std::vector<std::vector<int>> out;
            if (!prefill(B, cur, out)) return 1;
            std::vector<char> cool((size_t) B, 0);
            std::vector<double> done_ms((size_t) B, 0.0);
            std::vector<int> bk, nr, rows, acc, nx;
            std::vector<std::vector<int>> dr((size_t) B);
            std::vector<int> drafts;
            std::vector<float> conf;
            long long rounds = 0, off_n = 0, acc_n = 0, steps = 0, rows_n = 0, plain_steps = 0;
            const strata::ds4::Stats s0 = m.stats();
            const double t0 = now_ms();
            for (;;) {
                bk.clear();
                nr.clear();
                rows.clear();
                int R = 0, active = 0;
                for (int b = 0; b < B; ++b) active += (int) out[(size_t) b].size() < want[(size_t) b];
                const bool draft_now = active <= smax;   // (else a plain batch step: no drafts, taps or seeding)
                std::vector<int> tried;
                for (int b = 0; b < B; ++b) {
                    if ((int) out[(size_t) b].size() >= want[(size_t) b]) continue;
                    const int room = want[(size_t) b] - (int) out[(size_t) b].size();
                    int keep = 0;
                    const bool try_spec = draft_now && !cool[(size_t) b] && room > 1;
                    if (try_spec) {
                        if (!m.batch_draft(b, cur[(size_t) b], drafts, conf, err)) {
                            std::fprintf(stderr, "batch-test spec B=%s: %s\n", name.c_str(), err.c_str());
                            return 1;
                        }
                        float cum = 1.0f;
                        while (keep < (int) drafts.size() && keep < room - 1 && keep < g_spec_depth &&
                               conf[(size_t) keep] >= conf_min && (cum *= conf[(size_t) keep]) >= cum_min)
                            ++keep;
                    }
                    keep = std::max(0, std::min(keep, 16 - R - 1 - (B - 1 - b)));   // (the window's 16 rows: a row for each later one)
                    bk.push_back(b);
                    nr.push_back(1 + keep);
                    rows.push_back(cur[(size_t) b]);
                    for (int i = 0; i < keep; ++i) rows.push_back(drafts[(size_t) i]);
                    R += 1 + keep;
                    tried.push_back(try_spec ? 1 : 0);
                }
                if (bk.empty()) break;
                acc.assign(bk.size(), 0);
                nx.assign(bk.size(), 0);
                if (draft_now ? !m.batch_verify(bk.data(), nr.data(), rows.data(), (int) bk.size(), acc.data(), nx.data(), err)
                              : !m.batch_step(bk.data(), rows.data(), (int) bk.size(), nx.data(), err)) {
                    std::fprintf(stderr, "batch-test spec B=%s: %s\n", name.c_str(), err.c_str());
                    return 1;
                }
                plain_steps += !draft_now;
                ++steps;
                rows_n += R;
                size_t ri = 0;
                for (size_t i = 0; i < bk.size(); ++i) {
                    const int b = bk[i];
                    const int off = nr[i] - 1;
                    for (int a = 0; a < acc[i]; ++a) out[(size_t) b].push_back(rows[ri + 1 + (size_t) a]);
                    out[(size_t) b].push_back(nx[i]);
                    cur[(size_t) b] = nx[i];
                    ri += (size_t) nr[i];
                    ++rounds;
                    off_n += off;
                    acc_n += acc[i];
                    cool[(size_t) b] = tried[i] && off == 0;
                    if ((int) out[(size_t) b].size() >= want[(size_t) b] && done_ms[(size_t) b] == 0.0) done_ms[(size_t) b] = now_ms() - t0;
                }
            }
            const double ms = now_ms() - t0;
            const strata::ds4::Stats& s1 = m.stats();
            std::vector<double> per;
            for (int b = 0; b < B; ++b) per.push_back(1000.0 * (want[(size_t) b] - 1) / std::max(1.0, done_ms[(size_t) b]));
            std::sort(per.begin(), per.end());
            std::string cmp;
            const int same = compare(B, out, stag ? "St" : "S", cmp, &want);
            if (same != B) rc = 3;
            std::fprintf(stderr, "batch-test spec B=%s: %lld steps in %.0f ms (%.1f ms a step, %.1f rows a step): %.2f tok/s in all, "
                                 "a sequence %.2f median [%.2f-%.2f] (drafted solo %.2f, plain solo %.2f); %lld rounds, %.2f tokens a "
                                 "round, %lld offered / %lld accepted; %lld of the steps plain (DS4_BATCH_DSPARK_MAX=%d); greedy vs "
                                 "solo: %d of %d identical (%s )\n",
                         name.c_str(), steps, ms, ms / std::max(1LL, steps), (double) rows_n / std::max(1LL, steps),
                         1000.0 * want_all / ms, per[per.size() / 2], per.front(), per.back(), dsolo, solo_tps, rounds,
                         (double) want_all / std::max(1LL, rounds) + 0.0, off_n, acc_n, plain_steps, smax, same, B, cmp.c_str());
            std::fprintf(stderr, "batch-test spec B=%s a step: %.1f helper experts (%.2f ms issue + wait), %.2f on the CPU lane; "
                                 "drafting %.2f ms a step; the cache moved %lld\n", name.c_str(),
                         (double) (s1.helper_experts - s0.helper_experts) / std::max(1LL, steps),
                         (s1.helper_ms - s0.helper_ms) / std::max(1LL, steps),
                         (double) (s1.cpu_experts - s0.cpu_experts) / std::max(1LL, steps),
                         (s1.draft_ms - s0.draft_ms) / std::max(1LL, steps), (long long) (s1.swaps - s0.swaps));
            const long long ma_sh = (long long) (s1.mattn_shared - s0.mattn_shared), ma_pr = (long long) (s1.mattn_per_run - s0.mattn_per_run);
            if (ma_sh + ma_pr > 0)   // (DS4_MULTI_ATTN set)
                std::fprintf(stderr, "batch-test spec B=%s attention (DS4_MULTI_ATTN): %lld member-layer passes in shared launches, "
                                     "%lld per run\n", name.c_str(), ma_sh, ma_pr);
        }
        return rc;
    }
    for (int B : sizes) {
        std::vector<int> bk((size_t) B), cur((size_t) B), nx((size_t) B);
        std::vector<std::vector<int>> out((size_t) B);
        for (int b = 0; b < B; ++b) {
            bk[(size_t) b] = b;
            const std::vector<int>& ids = prompts[(size_t) b];
            if (!m.batch_prefill(b, ids.data(), (int) ids.size(), cur[(size_t) b], err)) {
                std::fprintf(stderr, "batch-test: bank %d: %s\n", b, err.c_str());
                return 1;
            }
            out[(size_t) b].reserve((size_t) n);
            out[(size_t) b].push_back(cur[(size_t) b]);
        }
        const strata::ds4::Stats s0 = m.stats();
        const double t0 = now_ms();
        for (int k = 1; k < n; ++k) {
            if (!m.batch_step(bk.data(), cur.data(), B, nx.data(), err)) {
                std::fprintf(stderr, "batch-test: B=%d step %d: %s\n", B, k, err.c_str());
                return 1;
            }
            for (int b = 0; b < B; ++b) {
                cur[(size_t) b] = nx[(size_t) b];
                out[(size_t) b].push_back(nx[(size_t) b]);
            }
        }
        const double ms = now_ms() - t0;
        const strata::ds4::Stats& s1 = m.stats();
        const double steps = std::max(1, n - 1), seq = n > 1 ? 1000.0 * (n - 1) / ms : 0.0;
        int same = 0;
        std::string cmp;
        for (int b = 0; b < B; ++b) {
            const std::vector<int>& r = ref[(size_t) b];
            const std::vector<int>& v = out[(size_t) b];
            size_t at = 0;
            while (at < r.size() && at < v.size() && r[at] == v[at]) ++at;
            char c[96];
            if (at == r.size() && at == v.size()) {
                ++same;
                std::snprintf(c, sizeof c, " seq %d identical;", b);
            } else
                std::snprintf(c, sizeof c, " seq %d DIFFERS at token %zu;", b, at);
            cmp += c;
            char tag[16];
            std::snprintf(tag, sizeof tag, "B%d seq", B);
            line(tag, b, v);
        }
        if (same != B) rc = 3;
        std::fprintf(stderr, "batch-test B=%d: %d steps in %.0f ms (%.1f ms a step): %.2f tok/s a sequence, %.2f tok/s in all "
                             "(solo %.2f a sequence); greedy vs solo: %d of %d identical (%s )\n",
                     B, n - 1, ms, ms / steps, seq, seq * B, solo_tps, same, B, cmp.c_str());
        std::fprintf(stderr, "batch-test B=%d a step: %.1f routed experts, %.1f on the helpers (%.2f ms issue + wait), %.2f on the "
                             "CPU lane; the cache moved %lld\n", B, (double) (s1.experts - s0.experts) / steps,
                     (double) (s1.helper_experts - s0.helper_experts) / steps, (s1.helper_ms - s0.helper_ms) / steps,
                     (double) (s1.cpu_experts - s0.cpu_experts) / steps, (long long) (s1.swaps - s0.swaps));
        // DS4_MULTI_ATTN (set): which attention path the batch windows' member layers took
        const long long ma_sh = (long long) (s1.mattn_shared - s0.mattn_shared), ma_pr = (long long) (s1.mattn_per_run - s0.mattn_per_run);
        if (ma_sh + ma_pr > 0)
            std::fprintf(stderr, "batch-test B=%d attention (DS4_MULTI_ATTN): %lld member-layer passes in shared launches, %lld per "
                                 "run\n", B, ma_sh, ma_pr);
    }
    // each helper's run counters (with DS4_HELPER_ARMED, its armed windows' used / cancelled / timed-out counts)
    if (m.stats().helper_experts > 0) std::fprintf(stderr, "%s", m.helper_report().c_str());
    return rc;
}

int main(int argc, char** argv) {
    strata::ds4::os::fine_timer();
#if defined(STRATA_METAL)
    if (argc > 1 && std::string(argv[1]) == "--sampler-test") return sampler_stat::run(argc - 1, argv + 1);   // (the sampler's statistics harness)
#endif
#if defined(__APPLE__)
    if (std::getenv("DS4_BT")) {   // a crash prints where it was (no debugger on the Mac box)
        signal(SIGSEGV, [](int) { void* fr[40]; const int n = backtrace(fr, 40); backtrace_symbols_fd(fr, n, 2); _exit(139); });
        signal(SIGBUS, [](int) { void* fr[40]; const int n = backtrace(fr, 40); backtrace_symbols_fd(fr, n, 2); _exit(138); });
        signal(SIGABRT, [](int) { void* fr[40]; const int n = backtrace(fr, 40); backtrace_symbols_fd(fr, n, 2); _exit(134); });
    }
#endif
    strata::ds4::Options o;
    std::string ids_s, dump, eos = "1";
    int n = 0, window = 0;   // (0: the model's feed unit - a whole prompt on the chunked path; 16 for the scoring runs)
    bool spec = true;
    float temp = 0.0f;   // the CLI's sampling (0: greedy)
    unsigned long long cli_seed = 1;
    bool serve_mode = false;
    bool pack_info_mode = false;
    std::string images, enc_rgb, enc_out;
    std::string score_ids, score_out, prof_ids, spec_ids;
    std::string batch_prompts, batch_sizes, batch_rows;   // DS4_BATCH_DECODE's test (batch_test)
    bool batch_spec = false;
    int enc_w = 0, enc_h = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--pack") o.pack = val("--pack");
        else if (a == "--gpus") {
            for (int g : parse_ids(val("--gpus"))) o.gpus.push_back(g);
        } else if (a == "--max-context") o.max_context = std::atoll(val("--max-context").c_str());
        else if (a == "--reserve-mib") o.reserve_mib = std::atoll(val("--reserve-mib").c_str());
        else if (a == "--vram-experts") { o.vram_experts = std::atof(val("--vram-experts").c_str()); o.vram_experts_auto = false; }
        else if (a == "--cpu-threads") o.cpu_threads = std::atoi(val("--cpu-threads").c_str());
        else if (a == "--expert-profile") o.profile = val("--expert-profile");
        else if (a == "--profile-readonly") o.profile_readonly = true;   // (read the placement, never write the counts back)
        else if (a == "--ssd-cache") o.ssd_cache_gb = std::atof(val("--ssd-cache").c_str());
        else if (a == "--ssd-policy") {
            const std::string v = val("--ssd-policy");
            o.ssd_policy = v == "lfu" ? 1 : v == "lfulru" ? 2 : 0;
        }
        else if (a == "--eos") eos = val("--eos");
        else if (a == "--no-expert-cache") o.expert_cache = false;
        else if (a == "--dspark") {
            const std::string v = val("--dspark");
            spec = v != "off";
            o.dspark = v != "none";   // none: not even loaded
            o.dspark_auto = v == "auto";
        } else if (a == "--dspark-conf") o.dspark_conf = (float) std::atof(val("--dspark-conf").c_str());
        else if (a == "--experts") {
            const std::string v = val("--experts");
            o.experts = v == "auto" ? -1 : std::max(1, std::atoi(v.c_str()));
        }
        else if (a == "--dspark-block") o.dspark_block = std::max(0, std::atoi(val("--dspark-block").c_str()));
        else if (a == "--dspark-depth") g_spec_depth = std::max(1, std::atoi(val("--dspark-depth").c_str()));
        else if (a == "--tp") o.tp = std::max(1, std::atoi(val("--tp").c_str()));
        else if (a == "--helper-device") {   // the helper expert cache's GPU(s), "1" or "1,4" (ds4_helper.hpp); repeated flags add up
            const std::vector<int> hv = parse_device_list(val("--helper-device"));   // (--helper-device 4 --helper-device 5 = 4,5, not 5)
            o.helper_devices.insert(o.helper_devices.end(), hv.begin(), hv.end());
        }
        else if (a == "--helper-slots") o.helper_slots = std::max<int64_t>(0, std::atoll(val("--helper-slots").c_str()));
        else if (a == "--prefill-chunk") o.prefill_chunk = std::max(0, std::atoi(val("--prefill-chunk").c_str()));
        else if (a == "--threads") o.threads = val("--threads") != "off";
        else if (a == "--vision") o.vision = val("--vision") != "off";
        else if (a == "--mmap-experts") {
            const std::string v = val("--mmap-experts");
            o.mmap_experts = v == "on" ? 1 : v == "off" ? 0 : -1;
        } else if (a == "--images") images = val("--images");
        else if (a == "--encode-image") enc_rgb = val("--encode-image");
        else if (a == "--dump-image") enc_out = val("--dump-image");
        else if (a == "--size") {
            const std::string v = val("--size");
            enc_w = std::atoi(v.c_str());
            enc_h = v.find('x') != std::string::npos ? std::atoi(v.c_str() + v.find('x') + 1) : 0;
        }
        else if (a == "--temperature") temp = (float) std::atof(val("--temperature").c_str());
        else if (a == "--seed") cli_seed = std::strtoull(val("--seed").c_str(), nullptr, 10);
        else if (a == "--verbose" || a == "-v") {
            o.verbose = true;
            std::fprintf(stderr, "ds4: build %s (%s)\n", DS4_BUILD_ID, STRATA_VERSION);
        } else if (a == "--version") {
            std::printf("strata-dsv41 %s build %s\n", STRATA_VERSION, DS4_BUILD_ID);
            return 0;
        }
        else if (a == "--ids") ids_s = val("--ids");
        else if (a == "--n") n = std::atoi(val("--n").c_str());
        else if (a == "--window") window = std::max(1, std::atoi(val("--window").c_str()));   // (> 16: the chunked prompt path)
        else if (a == "--dump-logits") dump = val("--dump-logits");
        else if (a == "--score") score_ids = val("--score");
        else if (a == "--score-out") score_out = val("--score-out");
        else if (a == "--profile-run") prof_ids = val("--profile-run");
        else if (a == "--spec-eval") spec_ids = val("--spec-eval");
        else if (a == "--batch-prompts") batch_prompts = val("--batch-prompts");
        else if (a == "--batch-sizes") batch_sizes = val("--batch-sizes");
        else if (a == "--batch-rows") batch_rows = val("--batch-rows");
        else if (a == "--batch-spec") batch_spec = true;
        else if (a == "--batch" || a == "--slots") {   // srvbatch: the server's "parallel": N -> the banks (read at load)
            const std::string v = val(a.c_str());
            if (std::getenv("DS4_BATCH_DECODE") == nullptr) {   // (an explicit DS4_BATCH_DECODE wins)
#if defined(_WIN32)
                _putenv_s("DS4_BATCH_DECODE", v.c_str());
#else
                setenv("DS4_BATCH_DECODE", v.c_str(), 1);
#endif
            }
        } else if (a == "--batch-groups") (void) val("--batch-groups");   // (the server's; one tensor-parallel group here)
        else if (a == "--serve") serve_mode = true;
        else if (a == "--pack-info") pack_info_mode = true;
        else if (a == "--layer-split") (void) val("--layer-split");   // the server's (this engine balances its own split)
        else if (a == "--help" || a == "-h") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (pack_info_mode) {   // the pack's description only: no device, no weights, no lock
        if (o.pack.empty()) {
            usage();
            return 2;
        }
        return strata::ds4::pack_info(o.pack, stdout);
    }
    if (o.pack.empty() || (!serve_mode && ids_s.empty() && enc_rgb.empty() && score_ids.empty() && prof_ids.empty() && spec_ids.empty() && batch_prompts.empty())) {
        usage();
        return 2;
    }
    // STRATA_DS4_HELPER=N (the helper expert cache's device) where the flag was not given; STRATA_DS4_HELPER_SLOTS=N
    if (o.helper_devices.empty())
        if (const char* hv = std::getenv("STRATA_DS4_HELPER")) o.helper_devices = parse_device_list(hv);
    if (o.helper_slots == 0)
        if (const char* hs = std::getenv("STRATA_DS4_HELPER_SLOTS")) o.helper_slots = std::max<int64_t>(0, std::atoll(hs));
    if (o.helper_devices.empty() && (std::getenv("STRATA_POOL_CLIENT") || std::getenv("STRATA_POOL_SERVE")))
        std::fprintf(stderr, "strata-dsv41: STRATA_POOL_CLIENT / STRATA_POOL_SERVE need --helper-device (one entry a helper); the pool is off\n");
    strata::ds4::apply_cpu_slots();   // STRATA_CPU_SLOTS=N (ds4_cpuslots.hpp), before the lane's threads exist
    // the runtime's host allocations as GTT memory, never migrated (ds4_os.hpp host_alloc_gtt), before the first HIP call
    if (const char* hg = strata::ds4::os::host_alloc_gtt(); hg[0] != '\0')
        std::fprintf(stderr, "strata-dsv41: host-pinned memory: %s\n", hg);
    if (o.profile.empty()) o.profile = o.pack + "/expert-profile.bin";   // "none": no profile
    if (o.profile == "none") o.profile.clear();
    o.vision_only = !enc_rgb.empty();
    // DS4_DSPARK_NOSEED=1 (SV-09; see Options::dspark_noseed): a --dspark off run that can never draft
    // skips the drafter's per-window feeding.  Ignored (with a line) where this process could draft.
    if (const char* ns = std::getenv("DS4_DSPARK_NOSEED"); ns != nullptr && std::atoi(ns) != 0) {
        if (!spec && !batch_spec && spec_ids.empty()) {
            o.dspark_noseed = true;
            std::fprintf(stderr, "strata-dsv41: DS4_DSPARK_NOSEED=1: --dspark off - the drafter is never seeded or tapped, "
                                 "and offers no drafts (a serve request's spec=1 gets plain steps)\n");
        } else
            std::fprintf(stderr, "strata-dsv41: DS4_DSPARK_NOSEED=1 ignored: this run can draft (--dspark on / auto, "
                                 "--batch-spec or --spec-eval)\n");
    }
    strata::ds4::Model m;
    std::string err;
    if (!m.load(o, err)) {
        std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
        return 1;
    }
    if (!o.vision_only) std::fprintf(stderr, "%s", m.describe().c_str());
    // malloc kept out of brk / mprotect from here on (ds4_os.hpp malloc_pin; DS4_MALLOC_PIN=0: off), before any token
    if (const std::string mp = strata::ds4::os::malloc_pin(); !mp.empty()) std::fprintf(stderr, "strata-dsv41: malloc pinned: %s\n", mp.c_str());
    if (const char* idle = std::getenv("DS4_TEST_IDLE_MS")) std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(idle)));   // (tests: a person typing)
    if (!batch_prompts.empty())   // DS4_BATCH_DECODE (--batch-rows: the verify-shaped cost; --batch-spec: DS4_BATCH_DSPARK)
        return batch_test(m, batch_prompts, batch_sizes, n, batch_rows, batch_spec, o.dspark_conf);
    if (serve_mode) return serve(m, eos, spec, o.dspark_conf, o.pack);
    const int window_given = window;
    if (window == 0) window = 16;   // (the scoring / profiling runs; the plain run below takes the model's feed unit)
    if (!prof_ids.empty()) return profile_run(m, prof_ids, window);
    if (!spec_ids.empty()) return spec_eval(m, spec_ids, n, window, o.dspark_conf, spec);
    if (!score_ids.empty()) return score(m, score_ids, score_out.empty() ? score_ids + ".tsv" : score_out, window, std::atoi(eos.c_str()));

    if (!enc_rgb.empty()) {   // the vision tower alone: a picture's span rows (BF16 [rows][dim]) into --dump-image
        std::vector<uint8_t> px((size_t) std::max(0, enc_w) * std::max(0, enc_h) * 3);
        std::FILE* f = std::fopen(enc_rgb.c_str(), "rb");
        const bool got = f != nullptr && !px.empty() && std::fread(px.data(), 1, px.size(), f) == px.size();
        if (f != nullptr) std::fclose(f);
        if (!got) {
            std::fprintf(stderr, "maya-ds4: %s: expected %d x %d RGB bytes (--size WxH)\n", enc_rgb.c_str(), enc_w, enc_h);
            return 1;
        }
        std::vector<uint16_t> span;
        for (int k = 0; k < 2; ++k) {   // (the second: warm)
            const double te = now_ms();
            if (!m.encode_image(px.data(), enc_w, enc_h, span, err)) {
                std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "vision: %d x %d pixels -> a span of %zu rows in %.0f ms\n", enc_w, enc_h,
                         span.size() / (size_t) m.dim(), now_ms() - te);
        }
        std::FILE* o2 = enc_out.empty() ? nullptr : std::fopen(enc_out.c_str(), "wb");
        if (o2 != nullptr) {
            std::fwrite(span.data(), 2, span.size(), o2);
            std::fclose(o2);
        }
        return 0;
    }
    if (!ids_s.empty() && ids_s[0] == '@') {   // --ids @file: the ids in a file (a command line is 32 KB on Windows)
        std::ifstream idf(ids_s.substr(1));
        std::stringstream ss;
        ss << idf.rdbuf();
        ids_s = ss.str();
    }
    const std::vector<int> ids = parse_ids(ids_s);
    if (!images.empty()) {   // --images: a GENI file's pictures, for the prompt's spans
        std::vector<Pic> pics;
        if (!read_pics(images, pics, err)) {
            std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
            return 1;
        }
        std::vector<strata::ds4::Image> imgs;
        for (const Pic& q : pics) {
            auto rows = std::make_shared<std::vector<uint16_t>>();
            const double te = now_ms();
            if (!m.encode_image(q.rgb.data(), q.width, q.height, *rows, err)) {
                std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "picture at %lld: %d x %d pixels, %zu tokens, %.0f ms\n", (long long) q.start, q.width,
                         q.height, rows->size() / (size_t) m.dim(), now_ms() - te);
            imgs.push_back({q.start, rows});
        }
        m.set_images(std::move(imgs));
    }
    std::FILE* df = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
    std::vector<float> lg;
    if (window_given == 0) window = df != nullptr ? 16 : std::max(16, std::min(m.feed_unit(), 1 << 30));
    const double t0 = now_ms();
    for (size_t a = 0; a < ids.size();) {
        const int T = (int) std::min<size_t>((size_t) window, ids.size() - a);
        const bool last = a + (size_t) T == ids.size();
        const double tf0 = now_ms();
        if (!m.feed(ids.data() + a, T, df != nullptr ? T : (last ? 1 : 0), err)) {
            std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
            return 1;
        }
        if (std::getenv("DS4_FEEDLOG") != nullptr)   // each feed's time: the incremental turn (--window W: a long prefix, then a short one)
            std::fprintf(stderr, "feed at %zu: %d tokens in %.0f ms (%.1f tok/s)\n", a, T, now_ms() - tf0, 1000.0 * T / (now_ms() - tf0));
        if (df != nullptr && m.logits(lg)) std::fwrite(lg.data(), 4, lg.size(), df);
        a += (size_t) T;
    }
    m.set_images({});
    const double t1 = now_ms();
    std::fprintf(stderr, "prompt: %zu tokens in %.0f ms (%.2f tok/s, windows of %d)\n", ids.size(), t1 - t0,
                 1000.0 * (double) ids.size() / (t1 - t0), window);
#if defined(STRATA_METAL)
    if (ds4_dbg("DS4_SEQ_LOG")) std::fprintf(stderr, "dispatch seq after the prompt: %llu\n", ds4_dispatch_seq());
#endif
    const strata::ds4::Stats s0 = m.stats();
    if (s0.eg_rows > 0)
        std::fprintf(stderr, "prompt's Engram rows: %lld, %.1f%% without a read (repeats and the row cache), %lld read\n",
                     (long long) s0.eg_rows, 100.0 * (double) s0.eg_hits / (double) s0.eg_rows, (long long) s0.eg_reads);
    const long long io0 = io_read_bytes();
    std::vector<int> out;
    std::vector<double> tok_ms;   // (DS4_TOKLOG: each token's time since the one before - the latency ledger)
    out.reserve((size_t) std::max(0, n) + 1);   // sized now - no growth (a heap grown by brk) inside the decode
    tok_ms.reserve((size_t) std::max(0, n) + 1);
    double t_prev = t1;
    unsigned long long counter = 0;
    long long d_off = 0, d_acc = 0, rounds = 0;
    bool cool = false;
    if (n > 0) {
        int tok = m.sample(temp, 1.0f, temp > 0 ? 64 : 1, 0.0f, cli_seed, counter++);
        std::vector<int> got;
        got.reserve(64);   // (a round's tokens: kWin at most)
        for (;;) {
            if (tok < 0) {   // (a sample's CUDA error, printed by the sampler; Maya #619)
                std::fprintf(stderr, "maya-ds4: the sample failed\n");
                return 1;
            }
            out.push_back(tok);
            {
                const double tn = now_ms();
                tok_ms.push_back(tn - t_prev);
                t_prev = tn;
            }
            std::printf("%d,", tok);
            std::fflush(stdout);
            if (ds4_dbg("DS4_PROGRESS") && out.size() % 200 == 0) {   // (sustained-run time series)
                static double tp = now_ms();
                static size_t np = 0;
                const double tn = now_ms();
                std::fprintf(stderr, "progress %zu tokens: %.2f tok/s over the last %zu\n", out.size(), 1000.0 * (double) (out.size() - np) / (tn - tp), out.size() - np);
                tp = tn;
                np = out.size();
            }
            if ((tok == 1 && std::getenv("DS4_NO_EOS") == nullptr) || (int) out.size() >= n) break;   // (DS4_NO_EOS: decode n tokens whatever)
            if (got.empty()) {
                int off = 0, acc = 0;
                const bool try_spec = spec && !cool;
                if (!round(m, tok, n - (int) out.size(), try_spec, o.dspark_conf, temp, 1.0f, temp > 0 ? 64 : 1, 0.0f,
                           cli_seed, counter, got, off, acc, err)) {
                    std::fprintf(stderr, "maya-ds4: %s\n", err.c_str());
                    return 1;
                }
                d_off += off;
                d_acc += acc;
                ++rounds;
                cool = try_spec && off == 0;
                std::reverse(got.begin(), got.end());
            }
            tok = got.back();
            got.pop_back();
        }
        std::printf("\n");
        if (df != nullptr && m.logits(lg)) std::fwrite(lg.data(), 4, lg.size(), df);
    }
    if (spec && rounds > 0) {
        const auto& s1 = m.stats();
        const double nv = (double) std::max<int64_t>(1, s1.drafts_rounds - s0.drafts_rounds);
        std::fprintf(stderr, "dspark: %lld rounds, %lld drafts offered, %lld accepted (%.2f tokens a round); drafting "
                             "%.2f ms a round, verifying %.1f ms in all (%lld windows)\n",
                     rounds, d_off, d_acc, (double) out.size() / (double) rounds, (s1.draft_ms - s0.draft_ms) / nv,
                     s1.verify_ms - s0.verify_ms, (long long) (s1.drafts_offered > 0 ? 0 : 0));
        std::fprintf(stderr, "dspark %s\n", per_position(s0, s1).c_str());
    }
    if (std::getenv("DS4_TOKLOG") != nullptr && tok_ms.size() > 8) {   // the decode's tokens after the first (which carries the prompt's tail)
        std::vector<double> v(tok_ms.begin() + 1, tok_ms.end());
        std::sort(v.begin(), v.end());
        std::fprintf(stderr, "token latency ms: p50 %.0f p95 %.0f p99 %.0f max %.0f mean %.0f (%zu tokens)\n", v[v.size() / 2],
                     v[(size_t) ((double) v.size() * 0.95)], v[(size_t) ((double) v.size() * 0.99)], v.back(),
                     std::accumulate(v.begin(), v.end(), 0.0) / (double) v.size(), v.size());
        // DS4_TOKLOG=2: every token slower than DS4_TOKLOG_X (5) x the median, by index, its ms since the decode began and the wall
        // clock (epoch ms) it ended at, to line a stall up with outside samplers (/proc/vmstat, sysfs)
        if (std::atoi(std::getenv("DS4_TOKLOG")) >= 2) {
            const double med = v[v.size() / 2];
            const double slow_x = std::getenv("DS4_TOKLOG_X") ? std::atof(std::getenv("DS4_TOKLOG_X")) : 5.0;   // (the cut, x the median)
            const double wall_end = (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::system_clock::now().time_since_epoch()).count();
            double at = 0.0;
            for (size_t i = 0; i < tok_ms.size(); ++i) {
                at += tok_ms[i];
                if (i > 0 && tok_ms[i] > slow_x * med)
                    std::fprintf(stderr, "slow token %zu: %.0f ms, ending %.0f ms into the decode (epoch ms %.0f)\n", i, tok_ms[i], at,
                                 wall_end - (t_prev - t1 - at));
            }
        }
    }
    const double t2 = now_ms();
    const auto& s = m.stats();
    const double nt = std::max<double>(1.0, (double) (s.tokens - s0.tokens));
    std::fprintf(stderr, "decode: %zu tokens in %.0f ms (%.2f tok/s); CPU lane: %lld of %lld experts; the cache moved %lld "
                         "(%lld in all)\n", out.size(), t2 - t1,
                 out.size() > 1 ? 1000.0 * (double) (out.size() - 1) / (t2 - t1) : 0.0,
                 (long long) (s.cpu_experts - s0.cpu_experts), (long long) (s.experts - s0.experts),
                 (long long) (s.swaps - s0.swaps), (long long) s.swaps);
    std::fprintf(stderr, "decode per token: layers %.2f ms (of it: routing waits %.2f, CPU lane %.2f, the rest host launches), "
                         "head wait %.2f, Engram reads %.2f\n",
                 (s.layer_ms - s0.layer_ms) / nt, (s.gpu_wait_ms - s0.gpu_wait_ms) / nt, (s.cpu_ms - s0.cpu_ms) / nt,
                 (s.head_ms - s0.head_ms) / nt, (s.engram_ms - s0.engram_ms) / nt);
    if (s.helper_experts > s0.helper_experts)   // the helper's share and the lane's time on it (issue + wait, on the lane's critical path)
        std::fprintf(stderr, "helper per token: %.1f experts, the lane's issue + wait %.2f ms\n",
                     (double) (s.helper_experts - s0.helper_experts) / nt, (s.helper_ms - s0.helper_ms) / nt);
    if (s.helper_arm_ms > s0.helper_arm_ms)   // DS4_HELPER_ARMED: off the critical path (after the lane's answer)
        std::fprintf(stderr, "helper arming per token: %.2f ms of the lane's time after its answers\n", (s.helper_arm_ms - s0.helper_arm_ms) / nt);
    if (s.helper_experts > 0) std::fprintf(stderr, "%s", m.helper_report().c_str());   // each helper's own counters (host ms issuing + waiting)
    if (s.eg_rows > s0.eg_rows) {   // the Engram rows: how many the row cache served, the reads' time behind the layers
        const double rows = (double) (s.eg_rows - s0.eg_rows);
        std::fprintf(stderr, "Engram per token: %.0f rows, %.1f%% without a read (the row cache holds %lld of %lld), "
                             "%.1f read, the reads %.2f ms from asking to the last one (left to wait: %.2f)\n",
                     rows / nt, 100.0 * (double) (s.eg_hits - s0.eg_hits) / rows, (long long) s.eg_cached,
                     (long long) s.eg_cache_cap, (double) (s.eg_reads - s0.eg_reads) / nt, (s.eg_io_ms - s0.eg_io_ms) / nt,
                     (s.engram_ms - s0.engram_ms) / nt);
    }
    const long long io1 = io_read_bytes();
    if (io0 >= 0 && io1 >= 0)
        std::fprintf(stderr, "decode reads from the disk: %.1f MB a token\n", (double) (io1 - io0) / 1e6 / nt);
    if (s.skipped > s0.skipped)   // STRATA_DS4_DISK_SKIP_W
        std::fprintf(stderr, "decode left out %.1f experts a token not in RAM (%.1f%% of the routed)\n",
                     (double) (s.skipped - s0.skipped) / nt,
                     100.0 * (double) (s.skipped - s0.skipped) / std::max<double>(1.0, (double) (s.experts - s0.experts)));
    if (s.store_slots_max > 0) {   // --ssd-cache
        const double h = (double) (s.store_hits - s0.store_hits), m = (double) (s.store_misses - s0.store_misses);
        std::fprintf(stderr,
                     "store: %.1f%% of the CPU lane's experts were in RAM (%.1f hits, %.1f misses a token); read %.1f MB a token "
                     "(demand and guesses), the CPU lane waited %.1f ms a token for them; %lld prefetched, %lld used, %lld given up "
                     "unused; slots %lld of %lld, %lld evictions, %lld failed reads; the GPU computed %.1f experts a token from the store "
                     "(%lld slots visible)\n",
                     100.0 * h / std::max(1.0, h + m), h / nt, m / nt, (double) (s.store_bytes - s0.store_bytes) / 1e6 / nt,
                     (s.store_wait_ms - s0.store_wait_ms) / nt, (long long) (s.store_prefetched - s0.store_prefetched),
                     (long long) (s.store_pf_used - s0.store_pf_used), (long long) (s.store_pf_wasted - s0.store_pf_wasted),
                     (long long) s.store_slots, (long long) s.store_slots_max, (long long) (s.store_evictions - s0.store_evictions),
                     (long long) (s.store_fails - s0.store_fails), (double) (s.store_zc - s0.store_zc) / nt,
                     (long long) s.store_zc_visible);
    }
    if (s.ahead_picks > s0.ahead_picks) {   // lookahead
        const double p = (double) (s.ahead_picks - s0.ahead_picks), c = (double) (s.ahead_cold - s0.ahead_cold);
        std::fprintf(stderr, "lookahead: the guesses a layer before held %.1f%% of the picks (their top k %.1f%%, two layers "
                             "before %.1f%%); file-read picks asked for early %.1f%% (%lld of %lld); %.1f experts asked "
                             "for early a token\n",
                     100.0 * (double) (s.ahead_hit - s0.ahead_hit) / p, 100.0 * (double) (s.ahead_hit_k - s0.ahead_hit_k) / p,
                     100.0 * (double) (s.ahead_hit2 - s0.ahead_hit2) / p,
                     c > 0 ? 100.0 * (double) (s.ahead_cold_hit - s0.ahead_cold_hit) / c : 0.0,
                     (long long) (s.ahead_cold_hit - s0.ahead_cold_hit), (long long) (s.ahead_cold - s0.ahead_cold),
                     (double) (s.ahead_pf - s0.ahead_pf) / nt);
        std::string rk = "lookahead: picked by the guess's rank (1 ..):";
        for (int i = 0; i < 16; ++i) {
            const long long n = (long long) (s.ahead_rank_n[i] - s0.ahead_rank_n[i]);
            if (n <= 0) break;
            char b[24];
            std::snprintf(b, sizeof b, " %.0f%%", 100.0 * (double) (s.ahead_rank_hit[i] - s0.ahead_rank_hit[i]) / (double) n);
            rk += b;
        }
        std::fprintf(stderr, "%s\n", rk.c_str());
    }
    if (s.prof_tokens > s0.prof_tokens) {   // DS4_PROF
        const double pt = (double) (s.prof_tokens - s0.prof_tokens);
        double sum = 0;
        for (int i = 0; i < 6; ++i) sum += (s.prof_ms[i] - s0.prof_ms[i]) / pt;
        std::fprintf(stderr, "GPU per token (DS4_PROF): %.2f ms - between layers %.2f, attention %.2f, routing + shared "
                             "expert %.2f, experts gate/up %.2f, waiting for the CPU lane %.2f, down + post %.2f\n", sum,
                     (s.prof_ms[0] - s0.prof_ms[0]) / pt, (s.prof_ms[1] - s0.prof_ms[1]) / pt,
                     (s.prof_ms[2] - s0.prof_ms[2]) / pt, (s.prof_ms[3] - s0.prof_ms[3]) / pt,
                     (s.prof_ms[4] - s0.prof_ms[4]) / pt, (s.prof_ms[5] - s0.prof_ms[5]) / pt);
    }
    if (df != nullptr) std::fclose(df);
    m.save_profile();
    return 0;
}
