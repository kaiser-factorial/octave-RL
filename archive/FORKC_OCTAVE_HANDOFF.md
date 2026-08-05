# Fork C: adding an Octave track to AiderPolyglot

**For:** Codex (or whichever agent picks this up)
**Depends on:** Forks A and B landing first. This document describes the work; don't start it until
the security fixes are submitted and the backend abstraction exists.
**Related docs in this folder:** `AIDERPOLYGLOT_FORK_HANDOFF.md` (overall plan),
`BUG_REPORT_FEEDBACK.md` (Fork A/B review).

---

## 0. The job, stated precisely

Add a seventh language to `primeintellect/aiderpolyglot`: **Octave**. That means producing a
`octave/exercises/practice/<slug>/` tree that is structurally identical to the six existing
language trees, so the environment's existing machinery picks it up with minimal code change.

**MATLAB is not the target.** MATLAB is licensed and cannot go into a public container image.
Octave is the free, open-source, largely-compatible substitute, and `.m` files written for Octave
will *mostly* run under MATLAB — but the deliverable is Octave, tested against Octave.

**The bar for "valid" is:** an exercise is valid if a competent Octave programmer could solve it
from the instructions and stub alone, the hidden tests would pass for a correct solution and fail
for an incorrect one, and the difficulty is comparable to the existing 225 exercises.

---

## 1. Read this before searching: there is no Octave track to port

**Exercism has 83 language tracks and neither Octave nor MATLAB is among them.** Verified against
`exercism.org/tracks` — the list runs 8th, ABAP, ARM64 Assembly … Wren, x86-64 Assembly, YAMLScript,
Zig, with no MATLAB and no Octave anywhere in it.

So there is no `exercism/octave` repo to fork and no MATLAB track to adapt. Don't spend time
looking. The exercises must be authored.

**But they should not be invented from scratch either**, and this is the key move:

---

## 2. The right source: `exercism/problem-specifications`

Exercism keeps its exercise definitions in a **language-agnostic** repo:

```
https://github.com/exercism/problem-specifications
  exercises/<slug>/description.md        # the prose, before track-specific additions
  exercises/<slug>/canonical-data.json   # every test case, with inputs and expected outputs
  exercises/<slug>/metadata.toml
```

Every existing track's test file is generated from `canonical-data.json`. The Python tests in
polyglot-benchmark say so in their own header:

> These tests are auto-generated with test data from:
> `https://github.com/exercism/problem-specifications/tree/main/exercises/bowling/canonical-data.json`

`canonical-data.json` shape (real excerpt from `bowling`):

```json
{
  "exercise": "bowling",
  "cases": [
    { "uuid": "656ae006-...", "description": "should be able to score a game with all zeros",
      "property": "score",
      "input": { "previousRolls": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] },
      "expected": 0 },
    { "uuid": "1245216b-...", "description": "rolls cannot score negative points",
      "property": "roll",
      "input": { "previousRolls": [], "roll": -1 },
      "expected": { "error": "Negative roll is invalid" } }
  ]
}
```

**Why this matters more than convenience:** generating from canonical data guarantees the Octave
exercises are *the same problems, with the same test cases, at the same difficulty* as the other
six languages. Hand-writing tests would produce a seventh language whose numbers aren't comparable
to the other six — which defeats the point of a polyglot benchmark.

Note the two case shapes: an `expected` scalar/array, and an `expected: {"error": "..."}` object.
Both need handling. Error cases are a meaningful fraction of most exercises.

---

## 3. Which exercises to author

The 225 exercises in polyglot-benchmark are **not** an arbitrary selection. Aider chose them from
697 total by keeping only those solved by **3 or fewer of 7 strong models** — deliberately
calibrated so top models land roughly between 5% and 50%. Picking easy exercises would make the
Octave column meaningless.

**Start from the 34 slugs already in the Python track**, since those are known to be in the
difficulty-calibrated set and all have canonical data:

```
affine-cipher  beer-song  book-store  bottle-song  bowling  connect  dominoes  dot-dsl
food-chain  forth  go-counting  grade-school  grep  hangman  list-ops  paasio
phone-number  pig-latin  poker  pov  proverb  react  rest-api  robot-name
scale-generator  sgf-parsing  simple-linked-list  transpose  tree-building  two-bucket
variable-length-quantity  wordy  zebra-puzzle  zipper
```

**Target 25–30 exercises**, matching the other languages' counts (C++ 26, Rust 30, Python 34, Go 39,
Java 47, JavaScript 49).

**Screen out the ones that don't fit Octave** before starting:

- **Anything requiring `classdef`.** Octave's OOP support is incomplete and differs from MATLAB's.
  This rules out or forces redesign of exercises that other tracks model as classes —
  `simple-linked-list`, `robot-name`, `zipper`, `react`, `paasio`, `dot-dsl` are the obvious
  candidates. Some can be redesigned functionally (see §5); some should just be dropped.
- **Anything I/O- or network-shaped** — `rest-api`, `paasio` lean on language-specific stream and
  interface idioms that don't translate cleanly.
- **Anything needing a parser combinator or heavy string machinery** is *fine* but expensive to
  write well in Octave — `forth`, `sgf-parsing`, `wordy` are good hard exercises but budget more
  time.

Good early candidates, roughly ordered by ease of translation: `bowling` (as a function over a
rolls vector), `transpose`, `pig-latin`, `phone-number`, `grep`, `go-counting`, `two-bucket`,
`variable-length-quantity`, `scale-generator`, `poker`, `dominoes`, `connect`.

---

## 4. Exact directory layout to produce

This is what the six existing trees look like. Match it. Real listing from
`python/exercises/practice/bowling`:

```
bowling/
├── .docs/instructions.md            # REQUIRED — the environment crashes without it (pre-Fork-A)
├── .docs/instructions.append.md     # optional, track-specific addendum
├── .meta/config.json                # authorship + which files are which
├── .meta/example.py                 # reference solution, never shown to the model
├── .meta/tests.toml                 # which canonical cases are included
├── bowling.py                       # the stub — THIS is what the model sees
└── bowling_test.py                  # the hidden tests — NEVER shown to the model
```

The Octave equivalent:

```
octave/exercises/practice/bowling/
├── .docs/instructions.md
├── .docs/instructions.append.md     # document the Octave function signature here — see §5
├── .meta/config.json
├── .meta/example.m                  # reference solution
├── .meta/tests.toml
├── bowling.m                        # stub
└── bowling_test.m                   # hidden tests
```

**Naming rule that matters:** the test file must be `<slug>_test.m`, with underscores where the
slug has hyphens if the function name requires it. See §6 for why this exact naming makes the code
change nearly free.

### `.meta/config.json`

Real example, from `python/bowling`:

```json
{
  "authors": ["..."],
  "contributors": ["..."],
  "files": {
    "solution": ["bowling.py"],
    "test": ["bowling_test.py"],
    "example": [".meta/example.py"]
  },
  "blurb": "Score a bowling game.",
  "source": "The Bowling Game Kata from UncleBob",
  "source_url": "https://..."
}
```

Reproduce this faithfully with `.m` extensions. Copy `blurb`, `source` and `source_url` from the
Python exercise's config so attribution stays intact.

**Worth flagging upward:** this `files` block is a *declarative* statement of which files are
solution vs. test vs. example — strictly better than the glob heuristics `_get_template_files`
currently uses. Reading `config.json` instead of globbing would be a genuine robustness improvement
for **Fork B**. Note it; don't implement it here.

### `.docs/instructions.md`

Take `description.md` from problem-specifications, or copy the existing Python exercise's
`instructions.md` verbatim — they're the same content, and copying keeps the wording identical
across languages, which is what you want for comparability.

### `.docs/instructions.append.md`

**This is where the real design work is documented.** Other tracks use it for language-specific
guidance — the Python `bowling` one explains how to raise exceptions with messages. Yours must
specify the **Octave function signature and error convention** for the exercise, because
canonical-data gives inputs and expected outputs but *not* an interface.

---

## 5. The actual creative work: designing an Octave interface per exercise

`canonical-data.json` says "given `previousRolls: [...]`, `score` should be 0." It does not say
whether that's a class with a `roll` method, a function taking a vector, or something else. Each
track decides, idiomatically, and documents it.

For Octave, prefer **plain functions over vectors, cell arrays and structs**. Concretely for
`bowling`, rather than Python's `BowlingGame` class:

```matlab
function total = bowling(rolls)
  % BOWLING  Score a completed game of ten-pin bowling.
  %   rolls: a row vector of pin counts, in order
  %   Raises an error with identifier 'bowling:invalidRoll' or
  %   'bowling:gameIncomplete' as appropriate.
```

Rules for signature design:

- **One entry-point function per exercise, named exactly like the file.** Octave requires the
  function name to match the filename for function files. This is non-negotiable and constrains
  slugs containing hyphens — `affine-cipher` must become `affine_cipher.m` with
  `function ... = affine_cipher(...)`.
- **Errors via `error('exercise:reason', 'message')`.** Tests then check with `try/catch` and
  inspect `err.identifier` (stable) rather than `err.message` (brittle). Document the exact
  identifiers in `instructions.append.md` — the model cannot pass error cases without knowing them,
  and no other track expects a model to guess them either.
- **Return types:** numeric scalar/matrix where natural, cell array of char for string lists,
  struct for named fields. Avoid `containers.Map` — Octave supports it, but behaviour has diverged
  from MATLAB's across versions.
- **Document the signature in `instructions.append.md`, not in the stub's comments alone.** The
  model sees both, but the append file is where every other track puts contract information.

---

## 6. The code changes in `AiderPolyglot.py`

Three dictionaries plus possibly nothing else.

### `_get_template_files` — likely a one-token change

The existing match arm:

```python
case "cpp" | "go" | "python":
    files_to_read = [
        (item.name, item)
        for item in exercise_dir.glob("*")
        if (not item.is_dir()
            and not item.name.startswith(".")
            and not any(item.match(p) for p in ["*_test.*", "CMakeLists.txt", "*.mod"]))
    ]
```

If Octave exercises are flat (stub and test in the exercise root) and the test is named
`<slug>_test.m`, then `*_test.*` already excludes it and this arm works unmodified. **Add `"octave"`
to that case arm and nothing else changes.**

Verify this rather than assuming it: after authoring one exercise, load the dataset and assert that
`info["template_files"] == ["bowling.m"]` and that `bowling_test.m` is absent.

### `IMAGES`

```python
"octave": modal.Image.from_registry("debian:bookworm-slim", add_python="3.11")
    .apt_install("octave"),
```

Pin the base image tag rather than using `:latest`, consistent with the other entries (and with
Fork A's rationale — reproducible benchmarks need pinned images). Consider whether
`apt_install("octave")` gives a new enough Octave; if the Debian version is too old for the
language features you use, either pin a newer base or install from a backport. **Record the Octave
version the image actually produces** and state it in the environment README — the other languages
all pin theirs.

### `LANG_CMDS` — the part that needs real testing

This is the piece I could not verify: **no Octave available in the environment where this document
was written, and no root to install one.** Everything below is a starting point to be tested, not a
known-good command.

The constraint: `LANG_CMDS` values are *static per-language strings* — they cannot interpolate the
exercise slug. So the command must discover the test file itself. Candidate:

```python
"octave": (
    "octave --no-gui --quiet --eval \""
    "fs = glob('*_test.m'); ok = true; "
    "for i = 1:numel(fs); "
    "  try; source(fs{i}); "
    "  catch e; disp(e.message); ok = false; end; "
    "end; "
    "exit(!ok);\""
),
```

Things to actually verify when you can run it:

1. **Exit codes.** Octave does not reliably exit non-zero on an uncaught error by default. The
   `exit(!ok)` must produce 0 on pass and non-zero on fail — confirm both directions explicitly,
   because `proc.returncode == 0` is the entire reward signal.
2. **`assert` failures raise catchable errors.** Octave's `assert()` should throw, and the
   `try/catch` should catch it. Confirm a deliberately failing assert produces a non-zero exit.
3. **Output reaches stdout.** `_test_solution` reads `proc.stdout` and feeds it back to the model
   as the "Tests failed:" message. If Octave writes diagnostics to stderr, the model gets an empty
   failure message and cannot debug on its second turn. The existing command wrapper appends
   `2>&1`, which should cover it — verify.
4. **Quoting.** The command goes through `sh -c "{test_cmd} 2>&1"`. Nested quotes are a real hazard;
   consider writing a tiny `run_tests.m` into the image and calling `octave run_tests.m` instead, if
   the inline `--eval` proves fragile.
5. **Startup time.** Octave's startup is slower than Python's. The sandbox timeout is 60s; confirm
   a typical exercise finishes well inside it.

---

## 7. Non-negotiable: the tests stay hidden

The environment's correctness depends on the model never seeing the tests it will be graded
against. `_get_template_files` strips them for all six existing languages — `*_test.*` for
Python/Go/C++, `.spec.js` for JavaScript, `src/main/**` only for Java, `src/*.rs` only for Rust.

**Do not use Octave's built-in `%!test` / `%!assert` block syntax.** Those live *inside the source
file*, which means putting them in the stub would hand the model the entire test suite, and putting
them in the reference solution wouldn't run against the model's code. Separate `<slug>_test.m` file,
always.

After authoring the first exercise, verify empirically: load the dataset, print the generated
prompt, and confirm the test contents appear nowhere in it.

---

## 8. Octave gotchas that will bite

- **Function name must match filename.** Slugs with hyphens become underscores. Be consistent
  between the directory name (keep the hyphenated slug, matching other tracks) and the `.m` filename
  (underscored).
- **1-based indexing.** Canonical data is written language-neutrally but examples in other tracks
  are 0-based. Index arithmetic must be re-derived, not translated.
- **`end` is required** to close functions in script-context files, and its absence fails
  confusingly.
- **Strings:** single-quoted are char arrays; double-quoted support escapes. Cell arrays of char are
  the idiom for string lists — `strjoin`, `strsplit`, `regexprep` all expect them. There is no
  native string-array type as in modern MATLAB.
- **Errors:** `error('id:sub', 'msg')`. Catch with `try ... catch err`, inspect `err.identifier`.
  Identifiers need a colon and must not start with a digit.
- **Integer division and type promotion** differ from MATLAB in edge cases; prefer explicit
  `floor`/`idivide`.
- **No `classdef` in practice.** Design around it (§5).

---

## 9. Validation before this is considered done

1. **One exercise end to end first.** Author `bowling` (or `transpose`, which is simpler), get the
   image building, the test command exiting correctly in both directions, and the dataset loading
   with the right `template_files`. Do not author 25 exercises against an unverified runner.
2. **Reference solutions pass.** Every `.meta/example.m` must make its own `<slug>_test.m` exit 0.
   Script this as a loop; it's the single most valuable check in this document.
3. **A wrong solution fails.** For at least a few exercises, confirm a deliberately incorrect
   implementation produces a non-zero exit. A test suite that always passes is worse than none.
4. **Difficulty check.** Run `prime eval run` on the Octave subset with a strong model, `-n 10`.
   You want something in the 5–50% band. If it's near 100%, the exercises are too easy for the set
   they're joining; if 0%, either the exercises are unreasonably hard, the interface documentation
   is insufficient, or the runner is broken — distinguish these before adjusting difficulty.
5. **Compare against a known language.** Same model, same `-n`, Octave vs Python. A large gap means
   something about the Octave track is harder or easier in a way unrelated to the problems, and
   that's worth understanding before publishing.

---

## 10. Licensing and attribution — settled, one action required

**Decision already made:** the Octave exercises live in their own public repo, cloned at runtime
alongside polyglot-benchmark. Do not vendor them into the environment package.

### What's whose

| Content | Origin | Status |
|---|---|---|
| `<slug>.m` stubs | newly written | yours |
| `.meta/example.m` reference solutions | newly written | yours |
| Test harness / runner code | newly written | yours |
| `<slug>_test.m` structure and assertions | newly written | yours |
| **The input/expected values inside those tests** | Exercism `canonical-data.json` | **Exercism's** |
| **`.docs/instructions.md`** | Exercism `description.md` | **Exercism's** |
| `blurb`, `source`, `source_url` in `.meta/config.json` | copied from the existing track | **Exercism's** |

### The obligation, and why runtime-cloning doesn't remove it

`exercism/problem-specifications` is **MIT licensed** — verified: "The MIT License (MIT),
Copyright (c) 2014, 2019, 2021 Exercism." MIT explicitly grants the right to copy, modify, publish
and distribute. The only condition:

> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.

Note carefully: **cloning at runtime does not avoid this.** That arrangement keeps *the environment
wheel* free of Exercism content, which is why the current environment is clear. But the Octave repo
itself will contain Exercism-derived prose and test data, and publishing that repo is
redistribution whenever anything clones it. The clone timing protects the package, not the repo.

### What to actually do

Two files in the Octave exercises repo:

1. **`LICENSE-exercism`** (or a clearly-named equivalent) containing Exercism's MIT license text
   verbatim, including the copyright line. Your own license for your own code sits alongside it.
2. **A README attribution section**, following the pattern polyglot-benchmark already uses:

   > Exercise instructions and canonical test data are derived from
   > [Exercism's problem-specifications](https://github.com/exercism/problem-specifications),
   > copyright © Exercism, used under the MIT license (see `LICENSE-exercism`). Octave stubs,
   > reference solutions, and test harness code are original work.

Keep `source` and `source_url` in each `.meta/config.json` intact — that's per-exercise attribution
and costs nothing.

That's the whole obligation. Five minutes, and it's handled.

---

## 11. Deliverables

- `octave/exercises/practice/<slug>/` for 25–30 exercises, structurally identical to the existing six
- A reference solution per exercise that provably passes its own tests
- `IMAGES["octave"]`, `LANG_CMDS["octave"]`, and `"octave"` added to the `_get_template_files` match arm
- A **verified** test-runner command, with the Octave version it was verified against recorded
- README updated: language table, exercise counts, supported-languages list
- The validation results from §9, especially the difficulty comparison against Python
- `LICENSE-exercism` and the README attribution block in the exercises repo (§10)
- A note back on anything in §6 that turned out wrong — that section is reasoned, not run
