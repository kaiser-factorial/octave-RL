# Pipeline log

A running record of defects found in the training pipeline, what each one
actually broke, and how it was caught. `OCTAVE_HANDOFF.md` says where the work
*is*; this file says what went wrong on the way and what that implies about
what to trust.

Entries are newest-first. Every entry answers "why did this survive until now",
because on this project the recurring failure has not been the bug itself — it
has been a green check that measured something adjacent to the thing that
mattered. A defect fixed without that field recorded will be reintroduced.

## How to add an entry

```markdown
## YYYY-MM-DD — one-line symptom

**Symptom.** What was observed, with the concrete numbers.
**Root cause.** The mechanism, precisely enough to re-derive it.
**Blast radius.** What data, runs, and published claims are affected.
**Why it survived.** Which check should have caught it and what it measured instead.
**Fix.** What changed, and what deliberately did not.
**Verification.** The command and its result.
**Residual risk.** What remains unproven.
```

Record the entry when the diagnosis lands, not at the end of the session. An
entry written from memory loses the numbers, and the numbers are the reason the
entry is worth having.

---

## 2026-08-08 — Reward path scored correct matrix answers as zero

**Symptom.** Reference solutions — code known to be correct, since the
generators produce it — scored `0/6` on every hidden case of
`octave-l1-broadcast_arith-00003`. The candidate transport was well-formed
(`structured_result = 1.0`) and reported the right *shape* (`[2, 4]`) and the
right *values*, in the wrong order:

```text
expected (row-major)  : [-4, -3, -4, -8,  3,  4,  3, -1]
reported (column-major): [-4,  3, -3,  4, -4,  3, -8, -1]
```

**Root cause.** `build_candidate_runner` reports `actual(:)'`, which Octave
flattens column by column. `candidate_record_matches` flattened the expected
JSON value row by row via `_flatten`, then compared the two sequences
element-wise. Scalars, row vectors, and column vectors flatten identically
under both orders, so the two agreed everywhere except on results with more
than one row *and* more than one column.

**Blast radius.** 250 of 1,500 tasks in the default pool — 1,500 of 9,000
hidden cases (16.7%). All six cases of every affected task are matrix-valued,
so an affected task is worth exactly `0.0` no matter what the model writes:

| Level | Affected families | Unsolvable tasks |
| ---: | --- | ---: |
| 1 | `broadcast_arith` | 50 / 500 (10%) |
| 2 | `broadcast_arith`, `struct_cell_wrangle` | 100 / 500 (20%) |
| 3 | `broadcast_arith`, `struct_cell_wrangle` | 100 / 500 (20%) |

This is worse than lost signal. Under GRPO, a correct matrix solution receives
the same zero advantage as a broken one, so the objective actively pushes the
policy away from correct matrix-producing code.

Affected runs: the 2026-08-06 two-step RTX 6000 Ada continuation (effective
rewards 0.8000 and 0.5000, trainable 4/8 and 2/8) was scored through this path.
Treat its `weights/step_2` policy as trained against a corrupted reward, not as
a clean continuation base.

Unaffected: the original 20-step run and its `0.905` held-out Level 1 result
predate the 2026-08-05 reward hardening and were scored through `build_harness`,
which compares inside Octave where both sides are Octave arrays.

**Why it survived.** `validate_reference_pool.py` — the source of the
"9,000/9,000 hidden cases passed" line repeated in `README.md` and
`OCTAVE_HANDOFF.md` — exercises `build_harness` + `parse_harness_result`. That
path is correct, and it is **not the path that computes rewards**. The reward
path (`build_candidate_runner` + `candidate_record_matches`) arrived with the
2026-08-05 reward hardening and was never validated against the reference pool;
the pre-existing 9,000/9,000 claim was carried across the rewrite unchanged.
Same shape as the WS2 "8/8 STABLE" false green: the check verified the
intention, not the mechanism.

**Fix.** `harness._octave_flatten` reorders a two-dimensional expected value
into Octave's column-major order before comparison, and
`candidate_record_matches` uses it. The candidate protocol is deliberately
unchanged — the trusted side holds the ground truth and is the correct side to
adapt. A transposed answer still fails, which
`test_matrix_results_are_compared_in_octave_column_major_order` pins.

**Verification.** `scripts/validate_local_runtime.py --num-tasks 500 --seed 0`
runs the full default pool through the *reward* path: **9,000/9,000 hidden
cases across 1,500 tasks, 138.9 s**, zero Prime calls. Pre-fix, the same command
fails 250 tasks.

**Residual risk.** None outstanding for comparison order. The first validation
ran on GNU Octave 8.4.0; it was subsequently rerun against the pinned 10.2.0
interpreter (see the rootfs entry below), which independently confirms the
column-major behaviour on the exact build the reference pool was validated with:

```text
c = [11 21 31; 12 22 32];  c(:)'  ->  [11 12 21 22 31 32]
```

---

## 2026-08-08 — Prime Sandbox is no longer required to score a candidate

**Change, not a defect.** Prime's managed Sandbox service has blocked this
project three times in ten days: `Payment required` mid-run on 2026-07-30,
`PROVISIONING` stalls on 2026-08-05, and four probes that never reached
`RUNNING` on 2026-08-07 — a plain-Python control reproducing the last one, which
ruled out the Octave image and this repository as causes. GPU compute was
healthy through all three.

`environments/octave_rl/executors.py` adds a `runtime = "local"` backend that
runs the same generated `run_candidate.m` on the calling host. `runtime`
defaults to `"prime"`; nothing changes unless it is set.

What is identical across backends: the input-only runner, so hidden expected
values and pass counters never enter the interpreter running model output; and
`harness.score_candidate_output`, so one scorer serves both. The property that
protects the reward signal does not depend on where the interpreter runs.

What is weaker locally: containment. A Sandbox is a separate container on
separate hardware. The local backend defends with a from-scratch child
environment (no `PRIME_API_KEY`, no Hugging Face or cloud credentials — built
rather than filtered, so a newly added variable cannot leak), a private 0700
working directory, `ulimit` bounds on CPU/address space/file size/process count,
a wall-clock kill of the whole process group, and a network namespace via
`unshare` when the host grants one. If a namespace cannot be obtained the
backend **raises** rather than quietly running with host network access;
`OCTAVE_RL_ALLOW_UNISOLATED_LOCAL=1` overrides that, and every record then
reports `network_isolated = false`. Note that the Prime CPU Sandbox also cannot
claim egress denial — its request model does not serialize a network policy — so
`runtime = "prime"` reports `network_isolated = false` as well.

Speed is the incidental result that matters for iteration: 1,500 tasks in 139 s
locally, against roughly five minutes of provisioning *per candidate* through a
cold Sandbox.

**Interpreter fidelity — resolved.** The first version of this backend used
whatever Octave the host had, which is a silent correctness risk: `apt install
octave` gives 8.4.0 on Ubuntu 24.04 while the reference pool was validated
against 10.2.0, and these tasks deliberately include tolerance and orientation
edge cases where interpreter versions can disagree.

`scripts/fetch_pinned_octave.py` removes the risk instead of documenting it. It
pulls the `gnuoctave/octave:10.2.0` layers straight from the registry over
HTTPS and extracts them into a directory — no Docker daemon, which matters
because Prime pods are themselves containers and cannot run one. Setting
`OCTAVE_RL_OCTAVE_ROOTFS` makes the backend `chroot` into that tree, so the
interpreter is the pinned build rather than a lookalike. Cost: one 1.6 GB pull,
about 4.3 GB unpacked.

This also strengthens containment rather than weakening it. The full stack is
`unshare --net --map-root-user` → `chroot <rootfs>` → `sh` applying the
`ulimit` envelope → `octave-cli`. The user namespace means it needs no
privileges, and candidate code sees neither the host filesystem nor a network.

**Verification.** Against the pinned 10.2.0 rootfs: 1,800/1,800 hidden cases at
60 tasks/level in 22.9 s, and the full 9,000-case pool clean. `runtime_description()`
records interpreter, source, and both isolation flags into every validation
report, so a number can always be traced to the interpreter that produced it.

**Residual risk.** `chroot` needs either root or a user namespace. Hosts with
unprivileged user namespaces disabled and no root will fall back to the host
interpreter — the report's `filesystem_isolated: false` and the `octave` version
string are the tell, so check them rather than assuming the rootfs took effect.

---

## 2026-08-08 — Prime API key rejected

**Symptom.** `prime pods list` and `prime sandbox list` both returned
`API key unauthorized` (CLI 0.6.21, key from `~/.prime/config.json`, no
`PRIME_API_KEY` shadowing it). Every Prime operation was blocked.

**Root cause.** Not established. Resolved by re-running `prime login`; the
account then showed zero pods and zero sandboxes.

**Blast radius.** Session-blocking only. No run was in flight.

**Why it survived.** Nothing to catch — but worth noting that a dead key and
the 2026-08-07 Sandbox stalls present very differently, and conflating them
would waste a debugging session. Those four probes *created* successfully,
returning IDs and two scheduler-audit UUIDs, then failed to reach `RUNNING`. A
revoked key fails at creation. Check which stage failed before assuming a
common cause.

**Residual risk.** Unknown whether the key expired, was rotated, or was
disabled by a billing state change. If it recurs, capture the exact failing
stage and the account's billing state before re-authenticating, since
re-authenticating destroys the evidence.
