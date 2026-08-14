---
name: decision-record
description: Write a decision record into docs/decisions/ when a design choice in Shadowcast is made, reversed, or discovered to be load-bearing. Use when a choice has a non-obvious rationale, when an approach was tried and rejected, when a dataset assumption turns out to be false, or when the user says "why did we..." and the answer is not written down anywhere.
---

# Decision records

`docs/decisions/` exists and is empty. It is for the choices whose *rationale* is the valuable
part — the ones where the code shows what was done and nothing shows why the obvious alternative
is wrong.

This repo already writes rationale inline, at length, in module docstrings and comments. A
decision record is for reasoning that does not belong to any single file: a property the whole
design rests on, a dataset fact that invalidates an approach, an alternative that was implemented
and then removed.

## When to write one

Write one when any of these is true:

- An approach was **tried and rejected** for a reason that is not visible in the surviving code.
  (The removed subsampling in `simplify_path` is the archetype: the reason truncation is correct
  is that subsampling creates unverified chords.)
- A choice is **load-bearing across modules** — break it somewhere and something unrelated fails.
- A **dataset claim turned out to be false** and the measurement is worth keeping. These are the
  rows in the README's reality table; the record holds the method behind the number.
- A constraint is **external and non-obvious** (Riot's see-through cells, numba's numpy pin).

Do not write one for: a routine refactor, anything a docstring already covers, or a decision that
is still open. An open question is a `[pending]` row in `docs/validation.md`, not a record.

## Format

File: `docs/decisions/NNNN-kebab-case-title.md`, numbered sequentially from `0001`.

```markdown
# NNNN — Title stated as the decision, not the topic

Date: YYYY-MM-DD
Status: accepted | superseded by [NNNN](NNNN-....md)

## Context

What forced the choice. Include the measurement if there is one — a count, a timing, a
mismatch rate. State the constraint that makes this non-trivial.

## Decision

What was chosen, in one or two sentences, in the present tense.

## Alternatives rejected

The ones a competent person would reach for first, and the specific reason each fails. This is
the section the record exists for. If it is empty, the decision did not need a record.

## Consequences

What this costs, what it now forbids, and what test or assertion keeps it true.
```

## House rules

- **Numbers must be measured.** Same rule as `docs/validation.md` and the README: a figure in a
  record is one that came out of a command. Say which command.
- **Name the guard.** If an invariant is enforced by a test, name the test. A record describing a
  property with nothing keeping it true is describing a property that will stop being true.
- Write in the repo's register: plain, specific, willing to say an earlier attempt was wrong.
- Records are append-only. Superseding one means writing a new record and setting the old one's
  status — never editing history to look correct.
- Add a link to the new record from `README.md` only if it changes what a reader of the README
  would otherwise believe.
