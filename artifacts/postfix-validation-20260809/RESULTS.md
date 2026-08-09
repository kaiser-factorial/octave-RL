# Post-repair validation — 2026-08-09

Validation of the 0.2.0 taskset repair described in `PIPELINE_LOG.md`
("Prompts now state the shape the grader compares", and the two entries below
it).

## Runtime

Scoring ran on the pinned **GNU Octave 10.2.0** inside
`gnuoctave/octave:10.2.0`, under `linux/amd64` emulation on an arm64 Mac. The
container is the isolation boundary, so `OCTAVE_RL_ALLOW_UNISOLATED_LOCAL=1`
was set and records report `network_isolated = false`. This is the same
interpreter build as every prior measurement; only the isolation mechanism
differs, and isolation does not affect arithmetic.

Zero Prime Sandboxes, zero GPU, no network for the scorer.

## The check that did not exist before

`scripts/validate_natural_solutions.py` runs a deliberately naive solution for
each family and level — written from the prompt alone, with no `(:)` coercion
and no transposes the prompt does not ask for — and requires every hidden case
to pass. The two existing validators both score the generator's *own reference
solution*, which cannot fail when a family is solvable only through an
undisclosed convention: the reference passes precisely because it contains the
convention.

### Result

```
uv run python scripts/validate_natural_solutions.py --num-tasks 500 --seed 314159
```

| level | hidden cases | failures |
|---|---:|---:|
| 1 | 3000 / 3000 | 0 |
| 2 | 3000 / 3000 | 0 |
| 3 | 3000 / 3000 | 0 |
| **total** | **9000 / 9000** | **0** |

Elapsed 3,730 s (500 tasks per level, six cases each, emulated).
Report: `natural_solutions.json`.

### The same check against the pre-fix generator

Run before the fix, with `generators.py` restored from `HEAD`, the naive
solution fails exactly where the diagnosis predicted:

| level | failing families | hidden cases |
|---|---|---|
| 1 | `broadcast_arith`, `linsolve_tolerance` | 0/6 each |
| 2 | `broadcast_arith`, `linsolve_tolerance` | 0/6 each |
| 3 | `broadcast_arith`, `linsolve_tolerance` | 0/6 each |

with `operator +: nonconformant arguments (op1 is 1x5, op2 is 1x4)` and
`operator \: nonconformant arguments (op1 is 6x6, op2 is 1x6)`. Both existing
validators were green at the time.

This counterfactual is the point of the artifact: it establishes that the new
check is non-vacuous, which is the property the two older validators lacked.

## Repository checks

- `pytest tests/` — **78 passed, 6 skipped** (was 74 passed, 6 skipped; the
  four new tests cover the prompt/grader shape relation, level-3 restatement,
  argument orientation, and truncated-fence extraction). Skips need a local
  `octave` binary.
- `ruff check` on the package, the new script and the changed test — clean.
- `python -m compileall` — clean.
- Wheel build — 9 files, `README.md` included, no `.venv` or `__pycache__`,
  `METADATA` carries the long description.

## Both validators, re-run natively with full isolation

The runs above were emulated with the container as the isolation boundary. Both
were then re-run on the Prime CPU Sandbox used for the re-measurement — native
x86_64, with the real `unshare --net` network namespace — which is the
configuration the environment actually ships:

| validator | isolation | result | elapsed |
|---|---|---|---|
| `validate_natural_solutions.py` | `/usr/bin/unshare --net --` | **9,000 / 9,000, PASS** | 587 s |
| `validate_local_runtime.py` | `/usr/bin/unshare --net --` | **9,000 / 9,000, PASS** | 698 s |

Reports: `natural_solutions_native.json`, `reference_pool_native.json`. Both on
GNU Octave 10.2.0, 500 tasks per level, seed `314159`.

That covers both host-side paths: the reference solutions still pass (their
`b=b(:)` and `a(:)` coercions are now no-ops on the corrected inputs rather
than load-bearing), and the naive solutions pass too.

## Consumer smoke from the Environments Hub

Everything above tests the repository. This tests the *published artifact*, on
the path a stranger would take, in a container that had never seen this project:

1. fresh Prime CPU Sandbox on `gnuoctave/octave:10.2.0`;
2. `pip install prime`, authenticate;
3. `prime env pull kaiser-factorial/octave-rl` — resolves and downloads;
4. `pip install -e .` — installs as the package `octave-rl` **0.2.2**;
5. `import octave_rl` from outside the source directory — exports
   `OctaveTaskset` and `load_environment`;
6. `eval @ eval.toml` with `[taskset] id = "octave-rl"` — verifiers resolves the
   taskset **by name from package metadata**, not from `PYTHONPATH`.

| result | |
|---|---|
| rollouts | 16 (8 tasks x 2, Level 2, Nemotron, T=1.0) |
| mean reward | 0.500 |
| mean execution fraction | 0.667 |
| infrastructure errors | **0** |
| families exercised | 8 of 10 |

The by-name resolution in step 6 is the part worth calling out: every earlier
run in this project reached the taskset through a repository checkout or an
explicit `PYTHONPATH`. This is the first evidence that the *published wheel*
resolves and scores on its own.

Locally, `prime env pull` also confirmed all five published files are
byte-identical to the repository copies (`generators.py`, `harness.py`,
`executors.py`, `octave_rl.py`, `README.md`).

## Not re-run

`validate_reference_pool.py` — the Prime Sandbox path exercising `build_harness`
and its in-Octave comparison — was not re-run. This change does not touch that
code path, and it incurs per-candidate Sandbox provisioning. Run it before the
next training run.
