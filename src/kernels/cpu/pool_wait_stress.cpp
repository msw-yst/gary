// src/kernels/cpu/pool_wait_stress.cpp - STRATA_POOL_WAIT (futex / nap / pred, 2026-11-21) under the load decode gives it.
//
// pool_stress drives run() / run_split() and needs AVX-512; decode runs run_split_multi_native, two phases a layer.  This
// drives that path with 1-12 experts a batch and random gaps of 0-1500 us between batches (shorter and longer than the spin,
// the nap's prediction and its margin), so workers sleep, nap, time out and wake late all the time, and checks every
// batch's output bitwise against a reference computed once.  A watchdog turns a hang (a lost wake-up) into a failure.
// The wait mode comes from the environment, as in the engine:
// Every batch is noted as a layer (0-47, as the expert source does), and ~15 % of layers have no CPU work (noted, no run),
// so pred's per-layer predictions and its re-post on an empty layer run against gaps that make them wrong in both directions.
//   STRATA_POOL_WAIT=futex|nap|pred [STRATA_POOL_SPIN_US=n] [STRATA_POOL_WAKE_FANOUT=k] pool_wait_stress [seconds] [workers]
//   (default 10 s, 8 workers)
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "ggml.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;

int main(int argc, char** argv) {
    const double seconds = argc > 1 ? std::atof(argv[1]) : 10.0;
    const int workers = argc > 2 ? std::atoi(argv[2]) : 8;
    if (!cpu::cpu_avx1_ok()) { std::printf("pool_wait_stress: no AVX on this CPU, skipped\n"); return 0; }
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, cpu::H, cpu::FF, f, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    constexpr int NE = 12;
    std::vector<std::vector<uint8_t>> blob(NE, std::vector<uint8_t>(f.bytes));
    std::vector<float> w((size_t) cpu::H * cpu::FF);
    for (int e = 0; e < NE; ++e) {
        for (size_t i = 0; i < w.size(); ++i) w[i] = 0.02f * std::sin((float) i * 0.17f + (float) e);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, w.data(), blob[e].data(), 0, cpu::FF, cpu::H, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, w.data(), blob[e].data() + f.up_off, 0, cpu::FF, cpu::H, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, w.data(), blob[e].data() + f.down_off, 0, cpu::H, cpu::FF, nullptr);
    }
    constexpr int NT = 2;
    std::vector<std::vector<uint8_t>> act(NT, std::vector<uint8_t>(cpu::kNativeActBytes));
    std::vector<cpu::ActQ> qact(NT);
    std::vector<float> x(cpu::H);
    for (int t = 0; t < NT; ++t) {
        for (int i = 0; i < cpu::H; ++i) x[i] = std::sin((float) (i + 13 * t) * 0.031f);
        cpu::native_quant_act(f, x.data(), act[t].data());
        cpu::act_quant_any(x.data(), cpu::H, qact[t]);
    }
    std::vector<cpu::ExpertJobMulti> jobs(NE);
    std::vector<float> out((size_t) NE * NT * cpu::H);
    for (int e = 0; e < NE; ++e) {
        jobs[e].blob = blob[e].data();
        jobs[e].nt = 1 + e % NT;
        for (int t = 0; t < jobs[e].nt; ++t) {
            jobs[e].nact[t] = act[t].data();
            jobs[e].act[t] = &qact[t];
            jobs[e].out[t] = out.data() + ((size_t) e * NT + t) * cpu::H;
        }
    }
    { cpu::ExpertPool ref(3, false); ref.run_split_multi_native(f, jobs.data(), NE); }   // the reference (the default wait)
    const std::vector<float> ref = out;

    cpu::ExpertPool pool(workers, false);
    std::atomic<long long> batches{0};
    std::atomic<bool> done{false};
    std::thread dog([&] {
        long long last = -1;
        auto t_last = std::chrono::steady_clock::now();
        while (!done.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            const long long b = batches.load();
            const auto now = std::chrono::steady_clock::now();
            if (b != last) { last = b; t_last = now; }
            else if (now - t_last > std::chrono::seconds(10)) {
                std::fprintf(stderr, "pool_wait_stress: FAIL - no batch finished for 10 s after %lld (a lost wake-up)\n", b);
                pool.diag(stderr);
                std::_Exit(1);
            }
        }
    });
    std::mt19937 rng(1234);
    const auto t0 = std::chrono::steady_clock::now();
    long long mism = 0;
    int layer = -1;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        layer = (layer + 1) % 48;
        const int n = 1 + (int) (rng() % NE);
        if (rng() % 100 < 15) {   // a layer with no CPU work: noted, nothing published
            cpu::ExpertPool::note_layer(layer, 0);
            std::this_thread::sleep_for(std::chrono::microseconds(rng() % 400));
            continue;
        }
        for (int e = 0; e < n; ++e)
            for (int t = 0; t < jobs[e].nt; ++t) std::fill(jobs[e].out[t], jobs[e].out[t] + cpu::H, std::nanf(""));
        cpu::ExpertPool::note_layer(layer, n);
        pool.run_split_multi_native(f, jobs.data(), n);
        for (int e = 0; e < n; ++e)
            for (int t = 0; t < jobs[e].nt; ++t)
                if (std::memcmp(jobs[e].out[t], ref.data() + ((size_t) e * NT + t) * cpu::H, cpu::H * sizeof(float)) != 0) ++mism;
        batches.fetch_add(1);
        // the gap: mostly short (a layer's two phases, the next layer), sometimes long enough for any spin to end
        const uint32_t r = rng() % 100;
        const int gap_us = r < 30 ? 0 : r < 70 ? (int) (rng() % 300) : r < 95 ? 300 + (int) (rng() % 700) : 1000 + (int) (rng() % 500);
        const auto g0 = std::chrono::steady_clock::now();
        if (r % 2 == 0) std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
        else while (std::chrono::steady_clock::now() - g0 < std::chrono::microseconds(gap_us)) {}
    }
    done.store(true);
    dog.join();
    std::printf("pool_wait_stress: %s - %lld batches (%lld phases), %lld mismatching rows, %d workers\n",
                mism == 0 ? "PASS" : "FAIL", batches.load(), 2 * batches.load(), mism, workers);
    return mism == 0 ? 0 : 1;
}
