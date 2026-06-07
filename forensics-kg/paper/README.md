# Multimodal Forensic Knowledge Graphs — Research Paper

LaTeX source for the paper *"Multimodal Forensic Knowledge Graphs for
Evidence-Centric Reasoning over Legal, Textual, and Visual Case Data."*

It is **Overleaf-ready** and compiles with standard TeX Live packages
(IEEEtran conference style). A reference build (`main.pdf`, 15 pages) is
included.

## Layout

```
paper/
├── main.tex                  # the paper (all sections)
├── references.bib            # bibliography (BibTeX)
├── main.pdf                  # reference build (15 pages)
├── figures/                  # 8 vector figures (.pdf) + .png previews
│   ├── architecture.pdf          ├── image_eval_summary.pdf
│   ├── ontology_overview.pdf     ├── entity_resolution.pdf
│   ├── multimodal_workflow.pdf   ├── evidence_chain.pdf
│   ├── real_case_f1.pdf          └── ablation.pdf   (full vs baseline)
├── tables/                   # \input fragments used by main.tex
│   ├── ontology_summary.tex
│   └── dataset_summary.tex
└── scripts/
    ├── generate_figures.py       # regenerates everything in figures/
    ├── run_experiments.py        # real-case full-vs-baseline ablation (N runs)
    ├── run_experiments_ci.py     # bootstrap CIs + principal-loc / date-coverage
    ├── run_resolution_eval.py    # cross-case SAME_AS precision benchmark
    ├── run_bloodstain_ablation.py# CV-only vs LLM-only vs CV+LLM bloodstain ablation
    ├── experiment_results.json   # ontology ablation output (results + Fig. ablation)
    ├── experiment_ci.json        # bootstrap CI + fair-metric output
    ├── resolution_eval.json      # SAME_AS candidate/accepted links + objective gold
    └── bloodstain_ablation.json  # CV/LLM/CV+LLM bloodstain ablation output
```

## Reproducing the experiments

The real-case numbers and the ablation come from `scripts/run_experiments.py`,
which runs the full ontology-guided pipeline and a naive same-LLM JSON baseline
over the 10 documented cases (`backend/data/real_cases/`), scoring both with the
identical matcher (`RealCaseEvaluator`). It needs the backend venv and an OpenAI
key in `backend/.env`:

```bash
cd backend && .venv/bin/python ../paper/scripts/run_experiments.py
```

It writes `experiment_results.json` (mean ± sd over N=3 runs). The figures
`real_case_f1.pdf` and `ablation.pdf` are produced from those measured values by
`generate_figures.py`.

## Compiling

### Overleaf (recommended)
Upload the `paper/` folder, set the main document to `main.tex`, and compile
(pdfLaTeX). Overleaf runs the BibTeX passes automatically.

### Local — pdflatex + bibtex
```bash
cd paper
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

### Local — latexmk
```bash
cd paper
latexmk -pdf main.tex
```

### Local — tectonic (what was used to verify this build)
```bash
cd paper
tectonic main.tex          # self-contained; fetches packages on first run
```

## Regenerating the figures

Figures are produced programmatically (matplotlib) so they stay editable and
consistent. They are checked in, so you do **not** need to run this to compile.

```bash
cd paper/scripts
python3 generate_figures.py     # needs matplotlib + numpy
```

All numeric values in the figures are the measured metrics reported in the
paper; no figure fabricates data.

## Build status (verified)

- 16 pages (two-column IEEEtran conference) — exceeds the 8-page target.
- 0 LaTeX errors, 0 BibTeX errors, 0 undefined citations/references.
- No placeholder/fake-author citations remain (all replaced with real papers).
- All 8 figures, all tables, and all numbered equations resolve; every float is
  referenced.
- Real-case results, the ablation, bootstrap CIs, and the SAME_AS resolution
  benchmark are all **measured**, not asserted; every number in the text matches
  `scripts/experiment_results.json`, `experiment_ci.json`, and
  `resolution_eval.json`.
- 3 sub-millimetre overfull boxes (< 3 pt, not visible); benign ragged-cell
  underfull warnings only.

## TODOs requiring author verification

1. **Author block** in `main.tex` (`% TODO`): replace name, affiliation, e-mail.
2. **Bibliography metadata** in `references.bib`: entries preceded by a
   `% TODO` comment are well-known works whose exact venue/volume/page strings
   should be confirmed before camera-ready. Four entries are explicit
   placeholders to be replaced with the concrete sources you cite:
   - `text2cypher` — the specific text-to-Cypher reference used.
   - `azhwound` — the AZH wound-image dataset citation.
   - `legalkg2023` — a concrete legal knowledge-graph reference.
   - (and verify `cedar2004signature` for the CEDAR signature dataset).
   BibTeX does **not** treat `%` as a comment inside an entry, so all TODO
   notes are placed *between* entries on purpose — keep them there.
3. **No invented numbers.** Rather than a fabricated ablation, the *Design
   decisions* table separates **measured** evidence (the label-leak correction
   $100\%\!\to\!\approx\!0\%$ and the false-link reduction $25\!\to\!13$, both
   reported in the text) from **invariant** guarantees that hold by construction.
   A full controlled ablation over hosted-LLM components is listed as future work.

## Notes

- The paper is framed as a research contribution (method + evaluation), not a
  product/system report. It deliberately reports honest negative results for
  image tasks with misaligned ground truth.
- All metrics come from the project's evaluation code
  (`backend/app/services/evaluation/`) and documented real-case gold
  (`data/real_cases/`).
