#include <benchmark/benchmark.h>

#include "plastix/plastix.hpp"

static void BM_Version(benchmark::State &state) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(plastix::version());
  }
}
BENCHMARK(BM_Version);
