"""``string_parse``: text in, numbers out, under a stated separator convention.

Written in the 0.5.0 variant form described in ``specs.py`` and modelled on
``families/reduce_along_dim.py``, the worked exemplar; the separator prose
follows ``families/sliding_window.py``, which states its windowing convention
precisely enough that a reader cannot land off-by-one. The module exposes
exactly two names: ``VARIANT_KEYS`` and ``build(rng, level, key)``.

## The spec

Three dimensions, per ``PARAMETERIZATION_DESIGN.md``: separator, number format,
output orientation. That is 4 x 2 x 2 = 16 combinations and eight are wanted, so
the eight shipped are the full cross of separator x number format, with the
orientation alternating: every separator appears once with integers and once
with decimals, and once with a row output and once with a column output. No
separator and no number format is a minority dialect, and orientation is
balanced 4/4 rather than riding on one axis.

## Every convention this family could hide, and where it is stated

String parsing is a convention minefield: leading and trailing separators,
repeated separators, surrounding whitespace, whether an empty field is an error
or a value, a lone ``-``, a leading ``+``, decimal formatting, and what an empty
input returns are **all** undisclosed conventions unless the prompt states them.
Each one is disposed of exactly once, and the same way each time -- the
description states it *and* the draw never produces it, so the prose is a
promise rather than a rule the solver must apply:

- **leading/trailing separators, adjacent separators** -- stated absent
  ("neither begins nor ends with a separator, no two separators are adjacent"),
  never drawn. This is why an empty field never arises, so whether it would be
  an error or a value never has to be decided. It also makes ``strsplit``'s
  ``collapsedelimiters`` default (true in Octave, so ``strsplit('1,,2', ',')``
  yields two fields, not three) unobservable rather than load-bearing.
- **surrounding whitespace** -- stated absent, never drawn. The space-separated
  variants say the only spaces are the separating ones. ``str2double`` happens
  to trim, but nothing here rests on that.
- **a lone ``-``, a leading ``+``, exponents** -- the number grammar is stated
  as "an optional minus sign followed by ..." and only that is ever drawn.
- **decimal formatting** -- stated as exactly two digits after the point, and
  every decimal is drawn as a multiple of 0.25 (see the tolerance note below).
- **the empty input** -- deliberately **not** described, and never drawn: level
  1 always holds at least three numbers and level 2 at least two records of at
  least two numbers each. An empty numeric answer would serialise as
  ``zeros(1, 0)`` rather than ``[]`` (see ``harness.octave_literal``), which is
  a graded 1x0-versus-0x0 distinction no prompt here makes, so no case is
  allowed to reach it.

Output orientation is the one convention **not** stated in this file's prose:
the generator appends a shape sentence derived from the expected values
themselves, so the prompt's claim and the grader's ``size(actual)`` comparison
cannot drift apart. That is ``sliding_window``'s arrangement and the reason is
the same.

## The level ladder

Level 1: one char row vector holding N numbers; return the numbers.
Level 2: a **cell array of char row vectors**, each holding two or more numbers;
return one value per record, the sum of that record's numbers.
Level 3: level 2 restated in full, plus "Do not use for/while loops".

**Level 3 adds the constraint only; it does not change the task.** The 0.4.x
level 3 for this family switched the task to decimals, which is why
``tests/test_generation.py`` excludes ``string_parse`` from its
``same_task_at_level_three`` set. Decimals are a *variant* dimension here, not a
level, so this family now behaves like every other converted one and
``"string_parse"`` should be **added to that set** in the change that registers
this module in ``generators.VARIANT_MODULES``.

### Why level 2 cannot collapse onto level 1

The trap this conversion keeps falling into is a level 2 whose answer equals its
level 1 -- a distinct prompt that is not a distinct problem. ``reduce_along_dim``
shipped a symmetric trim that preserves the median exactly; ``sliding_window``
rejected the same ladder for the same reason. No validator catches it, because
the reference and the naive solution both compute what the description asks for.

Here the collapse is impossible by construction, twice over:

1. The **input type changes**. Level 1 takes a char row vector; level 2 takes a
   cell array of char row vectors. A level-1 solution does not return a wrong
   answer at level 2, it raises -- ``strsplit`` and ``sscanf`` both reject a cell
   argument. Measured on the pinned interpreter: the level-1 ``natural``
   solution run against level-2 and level-3 tasks scored **0 of 576 hidden
   cases, all 576 as execution errors** (8 variants x 2 levels x 6 seeds x 6
   cases).
2. The **answer length changes**. Level 2 returns one value per record (2 to 4
   of them) while the numbers those records hold number 4 to 16, since every
   record holds at least two. So the level-2 answer is never the vector of
   parsed numbers, whatever the values are, and a solution that parses without
   summing is not a partial-credit case but a shape failure. Measured over 2,000
   tasks per variant (12,000 cases each, 96,000 in total): the level-2 answer
   equals the numbers its input contains in **0 of 12,000** cases for every one
   of the eight variants, and equals the level-1 answer of the same seed in
   **0 of 12,000**.

Both near-misses were also run through the grader rather than reasoned about: a
transposed answer and a parse-without-summing answer score 0.000 on every one of
the eight variants at every level.

### Rejected level-2 steps

- **A running total of the parsed numbers.** The step ``reduce_along_dim`` and
  ``broadcast_arith`` both use, and it works here -- ``cumsum(v) == v`` only when
  every value is zero. It was passed over because three families in a ten-family
  pool already train ``cumsum``, and this branch exists to measure problem
  *diversity*. A parsing family's second level should ask for more parsing.
- **Records separated by a second character inside one string**, e.g.
  ``'1,2|3,4'`` returning a matrix. Rejected: the loop-free solution needs
  ``reshape`` plus a transpose to get records into rows, so the ``natural``
  solution could not be written without the reshaping the natural-solution rule
  forbids -- and a ragged-record convention would have to be described on top.
- **Records holding a single number.** Allowed by an earlier draft ("one or more
  numbers"), and it re-opens the collapse: with every record holding one number,
  the per-record sums *are* the parsed numbers, which is the level-1 answer in
  cell-array clothing. Records now hold two or more, stated and drawn.

### Rejected variants

- **Optional spaces around separators** (``'1, 2, 3'``), which is what the 0.4.x
  level 2 did. Shippable -- ``str2double`` trims and ``strsplit`` on ``','``
  leaves the spaces inside the fields -- but it is a ninth variant, and as a
  *level* it is degenerate: it does not change the answer at all, which is
  precisely how the 0.4.x ladder wasted a level.
- **Empty fields** (``'1,,3'``): whether that is an error, a skipped field, or a
  ``NaN`` is a pure convention with no defensible default. Excluded from the
  draws instead of described.
- **Exponent notation** (``'1e3'``) and **leading ``+``**: both parse under
  ``str2double`` and ``sscanf``, so the risk is not the interpreter -- it is that
  the prompt would have to spell out a grammar longer than the task.

## Tolerance

The default ``1e-9`` is inherited and is **never exercised**. Integer variants
draw integers in [-99, 99]. Decimal variants draw multiples of 0.25 in
[-24.75, 24.75] and render them with two digits, so every literal in the input
string is exactly representable as a double; ``strtod`` (behind both
``str2double`` and ``sscanf``) is correctly rounded, and the level-2 sum of at
most four such values is exact. Both sides of every comparison are therefore
bit-identical, and no case is decided by a last-ulp disagreement between NumPy
and Octave. Exactly representable values were preferred over justifying a
tolerance, which is the cheaper of the two arguments to make.
"""

from __future__ import annotations

import numpy as np
from specs import Variant

# (separator, number format, output orientation). Order is the round-robin order
# and is part of the split contract -- appending is safe, reordering silently
# changes which task gets which problem.
VARIANT_KEYS: list[str] = [
    "comma-integers-row",
    "comma-decimals-column",
    "semicolon-integers-column",
    "semicolon-decimals-row",
    "space-integers-row",
    "space-decimals-column",
    "mixed-integers-column",
    "mixed-decimals-row",
]

_SEPARATORS = {
    # name -> (the characters a case may use between two numbers, how the
    # description names them, the clause that rules out every other whitespace,
    # the Octave delimiter argument a `strsplit` reading needs).
    #
    # `mixed` draws per gap from both characters, so one string can carry both.
    # Octave's `strsplit` accepts a cell array of delimiters, which is what the
    # naive reading of "either a comma or a semicolon" writes.
    "comma": (
        (",",),
        "commas (',')",
        "contains no whitespace",
        "','",
    ),
    "semicolon": (
        (";",),
        "semicolons (';')",
        "contains no whitespace",
        "';'",
    ),
    "space": (
        (" ",),
        "single spaces (' ')",
        "contains no whitespace other than those separating spaces",
        "' '",
    ),
    "mixed": (
        (",", ";"),
        (
            "single-character separators, each of which is either a comma (',') "
            "or a semicolon (';'), so one string may contain both"
        ),
        "contains no whitespace",
        "{',', ';'}",
    ),
}

_NUMBERS = {
    # name -> (the grammar the description states, an example number).
    #
    # The integer draw is [-99, 99] and the decimal draw is that same integer
    # divided by 4, so both formats consume the rng identically and every
    # decimal is a multiple of 0.25 -- exactly representable, and exactly
    # summable at level 2. See the tolerance note in the module docstring.
    "integers": (
        "an optional minus sign followed by one or more digits",
        "-17",
    ),
    "decimals": (
        (
            "an optional minus sign followed by one or more digits, a decimal "
            "point, and exactly two more digits"
        ),
        "-17.25",
    ),
}


def _parse(key: str) -> tuple[str, str, str]:
    parts = key.split("-")
    if len(parts) != 3:
        raise ValueError(f"unknown string_parse variant {key!r}")
    separator, numbers, orientation = parts
    if (
        separator not in _SEPARATORS
        or numbers not in _NUMBERS
        or orientation not in ("row", "column")
    ):
        raise ValueError(f"unknown string_parse variant {key!r}")
    return separator, numbers, orientation


def _render(value: float, numbers: str) -> str:
    """One number as it appears in the input string, per the stated grammar."""
    return str(int(value)) if numbers == "integers" else f"{value:.2f}"


def _example(separator: str, numbers: str) -> str:
    """A three-number example string, built from the same rules as the draws.

    Generated rather than written out, so the example in the prompt cannot
    disagree with the separator or the number format the prompt just described
    -- the drift this whole form exists to prevent, in miniature.
    """
    characters = _SEPARATORS[separator][0]
    values = [4, -17, 25] if numbers == "integers" else [4.0, -17.25, 0.5]
    pieces = [_render(value, numbers) for value in values]
    # For `mixed`, alternate the two characters so the example shows both.
    gaps = [characters[index % len(characters)] for index in range(len(pieces) - 1)]
    text = pieces[0]
    for gap, piece in zip(gaps, pieces[1:], strict=True):
        text += gap + piece
    return text


def _describe(key: str, level: int) -> str:
    """The prompt text. Every level restates its own task in full.

    Level 3 saying only "...without loops" is what let `struct_cell_wrangle`
    level 3 fall from 0.792 to 0.000 while models guessed at the task from the
    family name alone. Guarded by
    `test_level_three_restates_its_own_task_for_every_problem` -- once
    `"string_parse"` joins that test's `same_task_at_level_three` set, which is
    now correct for this family and was not for the 0.4.x one.

    The output orientation is deliberately absent: the generator appends the
    shape sentence it derives from the expected values, so the prompt's claim
    and the grader's comparison cannot drift apart.
    """
    separator, numbers, _ = _parse(key)
    _, separator_english, whitespace_clause, _ = _SEPARATORS[separator]
    grammar = _NUMBERS[numbers][0]
    example = _example(separator, numbers)

    # One sentence for the separator convention, stated the same way at every
    # level: it is what keeps "leading separator", "repeated separator" and
    # "stray whitespace" from being three undisclosed conventions.
    def convention(subject: str) -> str:
        point = ", decimal points" if numbers == "decimals" else ""
        return (
            f"Every separator lies between two numbers: {subject} neither begins "
            f"nor ends with a separator and no two separators are adjacent. "
            f"{subject[0].upper()}{subject[1:]} {whitespace_clause}, and no "
            f"characters other than digits, minus signs{point} and separators."
        )

    if level == 1:
        return (
            f"s is a char row vector holding numbers separated by "
            f"{separator_english}, for example '{example}'. Each number is "
            f"written as {grammar}. {convention('s')} Return the numbers of s, "
            f"in the order they appear."
        )
    task = (
        f"s is a cell array of char row vectors. Each element of s is one "
        f"record: a char row vector holding two or more numbers separated by "
        f"{separator_english}, for example '{example}'. Each number is written "
        f"as {grammar}. {convention('a record')} Different records may hold "
        f"different counts of numbers. Return one value per record, in the "
        f"order the records appear in s: the sum of the numbers of s{{1}}, then "
        f"the sum of the numbers of s{{2}}, and so on."
    )
    if level == 2:
        return task
    return task + " Do not use for/while loops."


def _oriented(values: list, orientation: str) -> list:
    """The expected value in the orientation the shape sentence will promise.

    A JSON list of scalars is a row; a column is a list of one-element lists.
    """
    return [[value] for value in values] if orientation == "column" else list(values)


def build(rng: np.random.Generator, level: int, key: str) -> Variant:
    separator, numbers, orientation = _parse(key)
    characters, _, _, delimiter = _SEPARATORS[separator]

    cases: list[dict] = []
    for _ in range(6):
        # One draw pattern for every variant and every key, so a variant
        # selection cannot shift the shared rng stream: the sizes below depend
        # on the level and on earlier draws, never on `key`. `picks` is drawn
        # at full width even by the six single-separator variants, which collapse
        # it to a constant -- the same discipline `sliding_window` applies to the
        # stride its stride-1 variants never use.
        if level == 1:
            counts = [int(rng.integers(3, 8))]
        else:
            records = int(rng.integers(2, 5))
            counts = rng.integers(2, 5, records).tolist()
        total = sum(counts)
        raw = rng.integers(-99, 100, total)
        # Always two-valued, and reduced modulo the number of separator
        # characters afterwards. `rng.integers(0, 1, n)` would be the obvious
        # spelling for a single-separator variant and returns zeros *without
        # consuming the stream*, which would make the shared rng advance
        # differently for `mixed` than for the other six -- measured, not
        # assumed: it shifted the level-1 draws of every later family.
        picks = rng.integers(0, 2, total).tolist()

        # Integers, or the same draw as multiples of 0.25. Both are exactly
        # representable, so no expected value depends on the tolerance.
        values = raw.tolist() if numbers == "integers" else (raw / 4).tolist()

        pieces: list[str] = []
        offset = 0
        for count in counts:
            chunk = values[offset:offset + count]
            gaps = [
                characters[pick % len(characters)]
                for pick in picks[offset:offset + count - 1]
            ]
            text = _render(chunk[0], numbers)
            for gap, value in zip(gaps, chunk[1:], strict=True):
                text += gap + _render(value, numbers)
            pieces.append(text)
            offset += count

        if level == 1:
            # One char row vector; the answer is the numbers themselves.
            args: list = [pieces[0]]
            out = values
        else:
            # A cell array of char row vectors -- `octave_literal` renders a
            # JSON list of strings as `{'...', '...'}`. The answer is one sum
            # per record, and with every record holding at least two numbers it
            # is strictly shorter than the numbers parsed, so it can never be
            # the level-1 answer.
            args = [pieces]
            out = []
            offset = 0
            for count in counts:
                out.append(sum(values[offset:offset + count]))
                offset += count
        cases.append({"args": args, "expected": _oriented(out, orientation)})

    signature = "function out = string_parse(s)"
    # Octave's `sscanf` returns a column, so a row output is the transposed one.
    transpose = "'" if orientation == "row" else ""
    # ... and `cellfun`/`strsplit` produce a row, so a column output is the
    # transposed one. The transpose is asked for either way: the generated shape
    # sentence states which orientation is graded.
    cell_transpose = "" if orientation == "row" else "'"

    # The reference may coerce defensively, and does: it normalises both
    # candidate separators to spaces and lets `sscanf` do the parsing, so it is
    # an implementation independent of the `strsplit`/`str2double` reading below
    # rather than the same code twice. It is loop-free at every level, so it
    # satisfies the level-3 constraint as well.
    normalise = "strrep(strrep({t}, ',', ' '), ';', ' ')"
    if level == 1:
        body = (
            f" v = sscanf({normalise.format(t='s')}, '%f');\n"
            f" out = v{transpose};"
        )
    else:
        body = (
            " v = cellfun(@(t) sum(sscanf("
            f"{normalise.format(t='t')}, '%f')), s(:)');\n"
            f" out = v{cell_transpose};"
        )

    # What a competent Octave programmer writes from the description alone:
    # split on the separator the prompt names, convert, and -- at level 2 --
    # sum each record. No `(:)`, no reshape, and no transpose beyond the one the
    # shape sentence asks for. If this cannot pass, the variant is not shippable.
    # Concatenated rather than `str.format`ed: the `mixed` delimiter is the
    # Octave cell literal `{',', ';'}`, whose braces a format string would read
    # as fields of its own.
    def field(text: str) -> str:
        return f"str2double(strsplit({text}, {delimiter}))"

    if level == 1:
        natural = f" out = {field('s')}{cell_transpose};"
    elif level == 2:
        # Loops are still allowed at level 2, so the direct transcription is a
        # loop over the records. Preallocating in the graded orientation is what
        # a reader of the shape sentence writes, and it needs no transpose.
        shape = "1, numel(s)" if orientation == "row" else "numel(s), 1"
        natural = (
            f" out = zeros({shape});\n"
            " for i = 1:numel(s)\n"
            f"   out(i) = sum({field('s{i}')});\n"
            " endfor"
        )
    else:
        natural = f" out = cellfun(@(t) sum({field('t')}), s){cell_transpose};"

    return Variant(
        key=key,
        description=_describe(key, level),
        signature=signature,
        cases=cases,
        reference=f"{signature}\n{body}\nendfunction",
        natural=f"{signature}\n{natural}\nendfunction",
        vectorized=level == 3,
    )
