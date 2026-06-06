# Second-annotator kit — inter-annotator agreement for the 28-case gold

This kit lets a **second, independent annotator** label the 28 documented cases so
we can report inter-annotator agreement (pairwise F1 + Cohen's κ) and settle an
adjudicated gold. It upgrades the current "curator-plus-independent-review" gold
toward a measured two-annotator benchmark — the one gap the paper cannot close
automatically (it needs a second human, ideally with a forensics/legal background).

## What the annotator does

1. Work through `cases/RC-*.json`. Each file has the case `title`, a `source` link,
   the factual `narrative`, and an **empty** `annotation` block.
2. **Label independently from the `narrative`.** Consult the `source` link only to
   disambiguate. **Do _not_ open `data/real_cases/`** — that is the first
   annotator's gold, and looking at it destroys independence (and the κ).
3. Fill each list in `annotation`:
   - **persons** — `[["Full Name", "role"], ...]`; role ∈
     `{victim, accused, witness, officer, forensic_analyst}`. Use the role the case
     asserts (an "accused" stays "accused" even if later acquitted).
   - **locations** — the **single principal scene** (one entry, the place the crime
     happened), e.g. `"Bagiya restaurant, New Delhi"`.
   - **weapons** — the weapon or method **as a noun**: `revolver`, `knife`,
     `cyanide`, `strangulation`, `hit-and-run`. Leave empty if none is stated.
   - **crime_types** — canonical type(s): `murder`, `rape`, `gang rape`,
     `kidnapping`, `robbery`, `hit-and-run`, `poisoning`, …
   - **time_events** — the **documented dates** as stated (full date or bare year),
     e.g. `["2 July 1995", "2003"]`. A bare event with no date is not scored.
4. Record only facts the case **asserts**, not inferences.

## Scoring (run after the kit is filled)

```bash
cd paper/annotation_kit
python3 compute_agreement.py          # annotator-2 (cases/annotated_cases) vs annotator-1
python3 compute_agreement.py --selftest   # sanity: annotator-1 vs itself
```

**Forensic-expert third pass.** When a domain expert is available, copy the blank
templates into `cases/expert/`, have the expert fill them the same way, then run
`python3 compute_agreement.py expert` to get expert-vs-curator κ/F1. (Any
subdirectory name works: `compute_agreement.py <dirname>`.)

`agreement_report.json` reports:

- **pairwise F1** for the open-set categories (Person/Location/Weapon mentions,
  CrimeType set, TimeEvent dates). For open-ended entity extraction this is the
  standard agreement metric; Cohen's κ is *degenerate* here because there is no
  closed "true-negative" universe of mentions.
- **Cohen's κ** for the genuinely **categorical** decisions where a closed label
  set exists: the **role** of each shared person, and each case's **primary crime
  type**.
- an **adjudication list** — every disagreement (a mention in only one annotation,
  or a role conflict) — so the two annotators can reconcile a final gold.

### Self-test note

`--selftest` scores the gold against itself and returns `1.0` everywhere **except
TimeEvent ≈ 0.97**: two gold entries are bare events with no year (`"conviction"`),
which the year-based date matcher — the same one used throughout the paper — cannot
match. That is expected matcher behaviour, not a scorer bug.

## Regenerating the blank kit

`python3 make_annotation_kit.py` rewrites `cases/` from `data/real_cases/` with the
gold stripped. (Do not run this after annotating — it would erase the annotations.)
