# Add a new example

## When to use

You need to demonstrate a new algorithm, a new policy combination, or a new traits configuration as a runnable binary. Examples double as smoke tests and are where the paper points readers.

## Steps

1. Create a new subdirectory under `examples/`, named with hyphens (e.g. `examples/my-algo/`).

2. Add the source file using the subdirectory name with underscores (e.g. `my_algo.cpp`). Match the existing examples in shape:

   - `#include <plastix/plastix.hpp>` (angle-bracket, library include).
   - Put algorithm-local tags, traits, and policies inside an anonymous namespace at file scope to avoid leaking symbols.
   - Derive traits from `plastix::DefaultNetworkTraits<MyGlobals>` and override only the policies you need; this keeps the example focused on what's novel.
   - `static_assert(plastix::NetworkTraits<MyTraits>);` right after the traits definition — catches policy signature mistakes at the right location.
   - In `main`, print a banner, run the algorithm, print a PASS/FAIL summary, and return the appropriate exit code. Examples are expected to self-check whenever the algorithm has a known answer (see `mlp-xor`, `linear-regression`, `ipc-linear`).

3. Add a CMakeLists.txt in the new directory mirroring `examples/mlp-xor/CMakeLists.txt`:

   ```cmake
   add_executable(my_algo my_algo.cpp)
   target_link_libraries(my_algo PRIVATE plastix)
   target_compile_options(my_algo PRIVATE -Wall -Wextra -Wpedantic)
   ```

4. Register the directory in `examples/CMakeLists.txt`:

   ```cmake
   add_subdirectory(my-algo)
   ```

5. Run `clang-format -i examples/my-algo/my_algo.cpp`, then build and run the example to confirm it compiles and prints something sensible:

   ```bash
   cmake --build build -j --target my_algo
   build/examples/my-algo/my_algo
   ```

## Pitfalls

- Forgetting to add the directory to `examples/CMakeLists.txt` → CMake silently builds nothing and the binary is missing. Symptom: running the binary fails with "No such file or directory".
- `FullyConnected` always puts its outputs at `level(prev) + 1`. If you want a flat 1-layer regression, pass exactly one `FullyConnected` builder — no hidden layer.
- Don't put hyperparameters in `GlobalState` expecting to tune them from `main`: the `Network` owns `GlobalState` by value and exposes no setter. Use a `constexpr` in a namespace or default member initializers (`MetaStepSize = 1e-3f` in the struct body, as SwiftTD does).
- When extending per-connection state, add `SOAField<WeightTag, float>` explicitly as the first entry of `ExtraConnFields` if you want the weight — `DefaultNetworkTraits`' default list is replaced, not extended, when you override `ExtraConnFields`.

## Verify

- `cmake --build build -j` succeeds with no warnings.
- `build/examples/my-algo/my_algo` runs to completion and (when self-checking is possible) prints `PASS`.
- `ctest --test-dir build --output-on-failure` still passes — adding an example must not regress the framework.
