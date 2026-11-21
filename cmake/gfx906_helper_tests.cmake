# prototype (docs/dsv41/cards/EB-01.md, EB-01): tests of the gfx906 helper expert caches.  Included from
# CMakeLists.txt next to native_expert_parity, only when STRATA_ENABLE_HIP is on and STRATA_HIP_HELPER_ARCHS is
# non-empty; strata_engine, strata_kernels, strata_kernels_cpu, strata_artifact and _strata_gpu_runtime_target exist
# at that point.

# cross_device_grouped_parity: native_expert_grouped (the entry point the helper expert caches call) on two devices of
# the fat binary - the wave32 primary and a wave64 helper - on identical synthetic q4_K/q5_1 and iq3_s/iq4_nl experts,
# phase by phase (q8_1 of x, gate/up, SwiGLU + q8_1, down on identical hq, the whole pipeline), in STRATA_EXP_MODE 0 on
# both and in each device's own default policy, against each other and against native_expert_parity's CPU float
# reference.  Registered as native_expert_parity is (strata_kernels for the GPU path, strata_kernels_cpu for the CPU
# formats and ggml, strata_artifact for the headers); no strata_engine: nothing of the engine's start-up runs here.
add_executable(cross_device_grouped_parity src/kernels/cross_device_grouped_parity.cpp)
target_include_directories(cross_device_grouped_parity PRIVATE ${_strata_gpu_include_directories})
target_link_libraries(cross_device_grouped_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
if(STRATA_BUILD_TESTS)
  # devices 0 and 1 (HIP_VISIBLE_DEVICES orders them); exit 77 = skipped on a machine with one device.  The exit code
  # follows the CPU-reference tolerance only; the cross-device differences are printed (--strict would count them).
  add_test(NAME cross_device_grouped_parity COMMAND cross_device_grouped_parity 0 1 q4_K/q5_1 iq3_s/iq4_nl)
  set_tests_properties(cross_device_grouped_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
endif()

# pool_ipc_smoke (shared MI50 pool): a holder process exports an expert cache on an MI50 through a HIP IPC
# memory handle; client processes compute from it with native_expert_grouped, bit-compared to a private copy, and time
# one-launch-one-sync iterations alone and side by side.  Driven by hand (several processes), not a ctest.
add_executable(pool_ipc_smoke src/kernels/pool_ipc_smoke.cpp)
target_include_directories(pool_ipc_smoke PRIVATE ${_strata_gpu_include_directories})
target_link_libraries(pool_ipc_smoke PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
