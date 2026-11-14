// pool_test: Pool::start waking a worker that is going to sleep (a batch no caller helps).  CPU only.
// The case is Project Maya's tests/ds4_pool_test.cpp pool_start_wakes_sleeper (a7f3c24, #618; MIT,
// third_party/project-maya/NOTICE.md), standalone here: our tree has no ds4_pool_test.
// Build: g++ -std=c++17 -O2 -pthread -I src/dsv41 tests/dsv41/pool_test.cpp -o pool_test && ./pool_test
#include "ds4_pool.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>

using strata::ds4::Pool;

// Pool::start stores the generation, then loads the sleeper count; a worker going to sleep adds itself to that count,
// then loads the generation.  Without a fence between the two (store -> load) both can read the old value: the worker
// sleeps through the batch and the starter skips the notify.  A batch nobody help()s then never runs, so the test is a
// batch no caller helps, many times over, on a worker that sleeps as soon as it is idle (spin 0).  Its first batch also
// checks the worker's start: one that read gen_ only once its thread ran took a batch start()ed before that as already
// seen.  A missed wakeup hangs wait() for good, so a watchdog reports it and exits 1.
int main() {
    Pool p(1, {}, 0);
    const int batches = 200000;
    std::atomic<int> ran{0}, done{0};
    std::atomic<bool> over{false};
    const std::function<void(int)> fn = [&](int) { ran.fetch_add(1, std::memory_order_relaxed); };
    std::thread dog([&] {
        int last = -1;
        for (int still = 0; !over.load(); std::this_thread::sleep_for(std::chrono::milliseconds(100))) {
            const int d = done.load();
            still = d == last ? still + 1 : 0;
            last = d;
            if (still == 50) {
                std::fprintf(stderr, "pool_test FAILED: batch %d of a pool nobody helps never ran (a missed wakeup)\n", d);
                std::_Exit(1);
            }
        }
    });
    for (int i = 0; i < batches; ++i) {
        p.start(1, fn);
        p.wait();
        done.store(i + 1);
    }
    over.store(true);
    dog.join();
    if (ran.load() != batches) {
        std::fprintf(stderr, "pool_test FAILED: %d of %d jobs ran\n", ran.load(), batches);
        return 1;
    }
    std::printf("pool_test: %d batches, all ran - ok\n", batches);
    return 0;
}
