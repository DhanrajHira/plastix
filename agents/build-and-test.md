# Build and test

## When to use

Starting work in a fresh clone, after pulling new dependencies, or when a build looks stale. Also as a reference for the canonical invocations to paste into PR descriptions.

## Steps

### Configure (once per fresh clone or after top-level CMakeLists change)

```bash
cmake -S . -B build
```

Optional flags (all default ON except CUDA):

- `-DPLASTIX_BUILD_TESTS=OFF`
- `-DPLASTIX_BUILD_EXAMPLES=OFF`
- `-DPLASTIX_BUILD_BENCHMARKS=OFF`
- `-DPLASTIX_ENABLE_CUDA=ON` — only meaningful when `src/kernels/*.cu` exists.

Debug / sanitizer configs:

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake -S . -B build-asan  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -g -O1"
```

### Build

```bash
cmake --build build -j                # everything
cmake --build build -j --target plastix            # library only
cmake --build build -j --target plastix_tests      # test binary only
cmake --build build -j --target mlp_xor            # one specific example
```

### Test

```bash
ctest --test-dir build --output-on-failure           # CTest wrapper
build/tests/plastix_tests                            # direct run
build/tests/plastix_tests --gtest_filter=PassTest.*  # filter by group
build/tests/plastix_tests --gtest_list_tests         # enumerate
```

### Benchmarks

```bash
build/benchmarks/bench_plastix
build/benchmarks/bench_plastix --benchmark_filter=BM_Version
```

### Examples

Every example is its own executable under `build/examples/<dir>/<name>`:

```bash
build/examples/mlp-xor/mlp_xor
build/examples/swifttd/swifttd
build/examples/ipc-linear/ipc_linear
```

## Pitfalls

- The first configure downloads GoogleTest and Google Benchmark via FetchContent. No network → configure fails. Subsequent builds cache into `build/_deps/`.
- `compile_commands.json` is generated under `build/` — point your editor / clangd at it. The `.gitignore` already excludes it.
- Passing the wrong C++ standard (or disabling extensions) through the environment will break `std::span`, concepts, and `requires` clauses. The top-level `CMakeLists.txt` sets C++20 with extensions off; do not override unless you're replacing the toolchain entirely.
- Example subdirectories must be listed in `examples/CMakeLists.txt` to be built. Adding a directory without updating that file silently produces nothing.

## Verify

A clean `cmake --build build -j` followed by `ctest --test-dir build --output-on-failure` should exit 0 with all tests passing. The test binary should report all four suites (`PlastixTest`, `NetworkTest`, `PipelineNetworkTest`, `SOAMacroTest`, plus allocator tests) as green.
