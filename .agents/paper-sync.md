# Keep the paper in sync with framework changes

## When to use

You're changing the public interface, stage ordering, policy concepts, propagation semantics, or a headline algorithm example. The paper under `paper/` is the authoritative prose description of Plastix's design and must not drift from the code.

## Cross-references to check

| You changed | Also update |
|---|---|
| A policy signature in `include/plastix/traits.hpp` | `paper/interface_overview.tex` — the per-stage signatures and $s(u)$/$s(u,v)$ descriptions. |
| Stage ordering in `DoStep` (`plastix.hpp`) | `paper/interface_overview.tex` figure `fig:pipeline` and `paper/implementation_details.tex`. |
| Propagation model semantics (`Propagation::Topological` / `Pipeline`) | `paper/implementation_details.tex` (pipelined vs. layered section) and `paper/implementation_elaboration.tex`. |
| SOA allocator layout, capacity strategy, or mmap approach | `paper/implementation_details.tex` and `paper/implementation_elaboration.tex` (memory layout, duplicate weight buffers). |
| The imprinting learner example (`examples/imprinting-learner/`) | `paper/examples.tex` — the walkthrough must still describe the live behavior. |
| The iPC examples (`examples/ipc-linear/`, `examples/ipc-multilayer/`) | `paper/examples.tex`. |
| A new example with narrative value | Add a paragraph to `paper/examples.tex` if it exercises a distinct feature; otherwise leave the paper alone. |
| `AGENTS.md` sections describing policies, propagation, or the pipeline | Cross-check against the same sections of `main.tex` / `interface_overview.tex`. |

## Process

1. Identify the change class from the table above.
2. Read the relevant `.tex` section before and after the change. Update prose, equations, and any `fig:` references.
3. If a LaTeX figure is out of date, edit the TikZ in place — figures live inline in the `.tex` files (see `fig:pipeline` in `interface_overview.tex`) or under `paper/tikz/` for externalized figures.
4. When the paper and code conflict, prefer what the code does and edit the paper. The code is ground truth; the paper is documentation.
5. Do not build the paper as part of CI. Compiling LaTeX is out of scope for this repository's build system. If you touch the paper, spot-check the diff renders by running whatever LaTeX toolchain is locally available.

## Pitfalls

- The paper uses a neuron-first framing that maps onto the code but uses different names (`Forward` in the paper, `ForwardPass` policy in code; `UpdateUnit` in both). Keep the names consistent within each medium — don't rename the policy to match the paper or vice versa without touching both.
- The paper's pipeline figure lists eight stages in a fixed order. The code's order in `DoStep` is authoritative; update the figure to match, not the other way around.
- `paper/main.tex` contains an **outline** block near the top that was used during drafting. It may not reflect the final structure of the paper — verify against the actual section order when editing.
- The `sample.bib` includes a wide range of citations. When adding a new citation, prefer existing keys where they exist rather than duplicating entries.

## Verify

- A diff review: open the changed code and the changed paper side-by-side and confirm that every public-facing name, stage, and equation matches.
- If you have `latexmk` or similar available, build `paper/main.tex` once to catch compile-level errors. This is a local check, not a CI gate.
