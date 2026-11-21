// src/qwen35/q36_main.cpp - strata-q36: the Qwen3.6 engine path on its own (the same entry point `strata` switches
// to for a qwen35moe model).
#include "strata/kernels/iq_kernels.hpp"
#include "strata/qwen35/engine.hpp"

int main(int argc, char** argv) {
#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
    strata::kernels::xtx_fast_perm_startup();   // STRATA_XTX_FAST_PERM, before any graph capture (as generate.cpp)
#endif
    return strata::qwen35::run(argc, argv);
}
