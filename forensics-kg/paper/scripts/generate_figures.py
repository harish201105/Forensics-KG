#!/usr/bin/env python3
"""Generate all figures for the Multimodal Forensic Knowledge Graph paper.

Produces seven vector PDFs (plus PNG previews) in ../figures:
  1. architecture.pdf        - end-to-end system architecture
  2. ontology_overview.pdf    - grouped ontology node categories + relation families
  3. multimodal_workflow.pdf  - text branch + image branch + fusion + KG insertion
  4. real_case_f1.pdf         - per-entity-type P/R/F1 on 10 documented real cases
  5. image_eval_summary.pdf   - per-image-task evaluation with honest negative results
  6. entity_resolution.pdf    - embedding-blocked, verified, non-destructive SAME_AS
  7. evidence_chain.pdf       - example evidence chain with supporting/contradicting edges

All numeric values are the measured metrics reported in the paper; no figure
fabricates data. Run:  python3 generate_figures.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import numpy as np

# ----------------------------------------------------------------------------
# Global style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.edgecolor": "#3a3a3a",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,   # embed TrueType so text stays selectable/editable
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# Consistent palette across all figures
C = {
    "ink":     "#1f2933",
    "muted":   "#52606d",
    "blue":    "#2c5f8a",
    "blue_l":  "#dbe7f1",
    "teal":    "#2a9d8f",
    "teal_l":  "#d6efeb",
    "amber":   "#e9a23b",
    "amber_l": "#faedd4",
    "red":     "#c0566b",
    "red_l":   "#f4dde2",
    "violet":  "#7d6ca3",
    "violet_l":"#e7e1f0",
    "slate":   "#5b6b7b",
    "slate_l": "#e3e8ed",
    "green":   "#3f9d5a",
    "grayfill":"#f2f4f6",
    "gridgray":"#cfd6dd",
}

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name):
    pdf = os.path.join(FIG_DIR, name + ".pdf")
    png = os.path.join(FIG_DIR, name + ".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close(fig)
    print("wrote", pdf)


def box(ax, x, y, w, h, text, fc, ec=None, fs=9, weight="normal",
        tc=None, rounding=0.018, lw=1.1, align="center"):
    ec = ec or C["muted"]
    tc = tc or C["ink"]
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(patch)
    ha = "center" if align == "center" else "left"
    tx = x + w / 2 if align == "center" else x + 0.012
    ax.text(tx, y + h / 2, text, ha=ha, va="center",
            fontsize=fs, color=tc, weight=weight, zorder=3, linespacing=1.25)
    return patch


def arrow(ax, p0, p1, color=None, lw=1.6, style="-|>", ls="-", mut=11, rad=0.0):
    color = color or C["slate"]
    a = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=mut,
        linewidth=lw, color=color, zorder=1,
        connectionstyle=f"arc3,rad={rad}", linestyle=ls)
    ax.add_patch(a)


# ============================================================================
# 1. Architecture
# ============================================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")

    # Column headers (centred over each column)
    headers = [
        (1.125, "Heterogeneous\nEvidence"),
        (3.525, "Modality-Specific\nExtraction"),
        (5.925, "Ontology-Validated\nKnowledge Graph"),
        (8.650, "Graph-Centered\nReasoning"),
    ]
    for hx, ht in headers:
        ax.text(hx, 5.95, ht, ha="center", va="center",
                fontsize=10.5, weight="bold", color=C["blue"])

    # --- Column 1: evidence sources ---
    box(ax, 0.15, 3.55, 1.95, 1.95,
        "Textual sources\n\n FIR  •  Court judgment\n Post-mortem  •  Lab\n Deposition  •  SOCO",
        C["blue_l"], ec=C["blue"], fs=8.2, align="left")
    box(ax, 0.15, 1.05, 1.95, 1.95,
        "Image sources\n\n Bloodstain  •  Wound\n Fingerprint  •  Ballistics\n Tool mark  •  Document",
        C["teal_l"], ec=C["teal"], fs=8.2, align="left")

    # --- Column 2: extractors ---
    box(ax, 2.55, 3.7, 1.95, 1.65,
        "Schema-registry\ntext extraction\n\n2-stage:\nentities $\\rightarrow$ relations",
        C["blue_l"], ec=C["blue"], fs=8.4)
    box(ax, 2.55, 1.15, 1.95, 1.65,
        "Strategy-based\nimage analysis\n\nCV features +\nLLM vision",
        C["teal_l"], ec=C["teal"], fs=8.4)

    # --- Column 3: ontology gate (centred between the extractors) + KG ---
    box(ax, 4.85, 2.95, 2.20, 0.95,
        "Ontology validation\n38 node / 48 rel types",
        C["amber_l"], ec=C["amber"], fs=8.6)
    box(ax, 4.85, 0.70, 2.20, 1.95, "", C["grayfill"], ec=C["slate"])
    ax.text(5.95, 1.95, "Property graph\n(Neo4j)", ha="center", va="center",
            fontsize=9.2, weight="bold", color=C["ink"], linespacing=1.25)
    ax.text(5.95, 1.18, "entities, relations,\nper-node embeddings",
            ha="center", va="center", fontsize=7.8, color=C["ink"],
            linespacing=1.3)

    # --- Column 4: reasoning modules ---
    rlabels = [
        ("Semantic search\n(vector KNN)", C["violet_l"], C["violet"]),
        ("NL$\\rightarrow$Cypher querying", C["violet_l"], C["violet"]),
        ("Cross-case entity\nresolution (SAME_AS)", C["violet_l"], C["violet"]),
        ("Temporal reconstruction\n& hypothesis support", C["violet_l"], C["violet"]),
    ]
    ry = 4.55
    for txt, fc, ec in rlabels:
        box(ax, 7.5, ry, 2.3, 0.86, txt, fc, ec=ec, fs=8.2)
        ry -= 1.07

    # Arrows: evidence -> extractors
    arrow(ax, (2.12, 4.5), (2.53, 4.5), color=C["blue"])
    arrow(ax, (2.12, 2.0), (2.53, 2.0), color=C["teal"])
    # extractors -> ontology validation (both converge symmetrically into the gate)
    arrow(ax, (4.52, 4.40), (4.83, 3.65), color=C["blue"], rad=-0.10)
    arrow(ax, (4.52, 2.35), (4.83, 3.15), color=C["teal"], rad=0.10)
    # ontology validation -> KG
    arrow(ax, (5.95, 2.93), (5.95, 2.67), color=C["amber"])
    # KG -> reasoning (distribution bus; connector kept high, clear of feedback)
    arrow(ax, (7.07, 2.40), (7.28, 2.40), color=C["slate"])
    ax.plot([7.3, 7.3], [1.77, 4.98], color=C["slate"], lw=1.4, zorder=0)
    for yy in (4.98, 3.91, 2.84, 1.77):
        arrow(ax, (7.3, yy), (7.48, yy), color=C["slate"], lw=1.3, mut=9)

    # Feedback: reasoning writes hypotheses back to KG (dashed, in the freed zone)
    arrow(ax, (7.50, 1.50), (7.08, 1.05), color=C["violet"], ls=(0, (4, 3)),
          lw=1.3, rad=0.28)
    ax.text(7.0, 0.40, "hypotheses written back\n& embedded", ha="center",
            va="center", fontsize=7.0, color=C["violet"], style="italic")

    _save(fig, "architecture")


# ============================================================================
# 2. Ontology overview
# ============================================================================
def fig_ontology_overview():
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")

    ax.text(6, 6.35, "Forensic Ontology: 38 node types in six semantic groups, "
            "48 typed relationships", ha="center", fontsize=10.5,
            weight="bold", color=C["ink"])

    groups = [
        ("Core case", C["blue_l"], C["blue"],
         "Case, Person, Location,\nEvidence, Weapon, Vehicle,\nCrimeType, TimeEvent"),
        ("Legal / judicial", C["violet_l"], C["violet"],
         "LegalSection, Verdict,\nCourtOrder"),
        ("Medical / lab", C["red_l"], C["red"],
         "InjuryPattern, CauseOfDeath,\nToxicologyResult, OrganFinding,\nSample, TestResult, DNAProfile,\nChemicalCompound"),
        ("Scene / procedural", C["amber_l"], C["amber"],
         "CrimeScene, PhysicalEvidence,\nChainOfCustody, Statement"),
        ("Forensic image", C["teal_l"], C["teal"],
         "BloodstainPattern, Stain,\nFingerprintPattern, Minutiae,\nWoundPattern, BallisticsPattern,\nToolMarkPattern, Forgery..."),
        ("Reasoning", C["slate_l"], C["slate"],
         "Hypothesis,\nImpactMechanism,\nExperiment"),
    ]
    # 3 x 2 grid of group cards
    positions = [(0.2, 3.55), (4.15, 3.55), (8.1, 3.55),
                 (0.2, 0.95), (4.15, 0.95), (8.1, 0.95)]
    w, h = 3.65, 2.25
    for (gx, gy), (title, fc, ec, members) in zip(positions, groups):
        box(ax, gx, gy, w, h, "", fc, ec=ec, lw=1.3, rounding=0.03)
        ax.text(gx + 0.16, gy + h - 0.26, title, ha="left", va="center",
                fontsize=9.8, weight="bold", color=ec)
        ax.text(gx + 0.16, gy + h / 2 - 0.18, members, ha="left", va="center",
                fontsize=8.0, color=C["ink"], linespacing=1.4)

    # Relationship-family ribbon
    fam = ("Relationship families:  involvement (INVOLVED_IN, WITNESSED_BY)  |  "
           "spatial/temporal (OCCURRED_AT, PRECEDES)  |  causal (LED_TO_DEATH, "
           "CAUSE_OF_DEATH_OF, WOUND_CAUSED_BY)  |  evidential (HAS_EVIDENCE, "
           "SUPPORTS, CONTRADICTS)  |  cross-case (SAME_AS)")
    ax.text(6, 0.42, fam, ha="center", va="center", fontsize=8.6,
            color=C["muted"], wrap=True, style="italic")

    _save(fig, "ontology_overview")


# ============================================================================
# 3. Multimodal workflow
# ============================================================================
def fig_multimodal_workflow():
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")

    # Text branch (top)
    ax.text(0.95, 6.05, "Text branch", fontsize=10, weight="bold", color=C["blue"])
    tb = [
        (0.2, "Document\n+ source type"),
        (2.1, "Stage 1:\nentity extraction"),
        (4.0, "Stage 2:\nrelation extraction"),
    ]
    for x, t in tb:
        box(ax, x, 4.7, 1.7, 1.0, t, C["blue_l"], ec=C["blue"], fs=8.2)
    for x0 in (1.9, 3.8):
        arrow(ax, (x0, 5.2), (x0 + 0.2, 5.2), color=C["blue"])

    # Image branch (bottom)
    ax.text(0.95, 2.55, "Image branch", fontsize=10, weight="bold", color=C["teal"])
    ib = [
        (0.2, "Forensic\nimage"),
        (2.1, "CV features\n(OpenCV)"),
        (4.0, "LLM vision +\nsynthesis"),
    ]
    for x, t in ib:
        box(ax, x, 1.15, 1.7, 1.0, t, C["teal_l"], ec=C["teal"], fs=8.2)
    for x0 in (1.9, 3.8):
        arrow(ax, (x0, 1.65), (x0 + 0.2, 1.65), color=C["teal"])

    # Ontology validation gate
    box(ax, 6.05, 2.95, 1.7, 1.45,
        "Ontology\nvalidation\n+ type-safe\ncoercion",
        C["amber_l"], ec=C["amber"], fs=8.2)
    arrow(ax, (5.7, 5.2), (6.6, 4.4), color=C["blue"], rad=-0.18)
    arrow(ax, (5.7, 1.65), (6.6, 2.95), color=C["teal"], rad=0.18)

    # Fusion / cross-reference
    box(ax, 8.05, 2.95, 1.75, 1.45,
        "Case graph\n$G_c$\n\ncross-modal\nlinks",
        C["grayfill"], ec=C["slate"], fs=8.4, weight="bold")
    arrow(ax, (7.77, 3.68), (8.03, 3.68), color=C["slate"])

    # graceful degradation note
    ax.text(5.0, 0.35,
            "Either branch may be absent: a text-only or image-only case still "
            "produces a valid (partial) case graph (graceful degradation).",
            ha="center", fontsize=7.8, color=C["muted"], style="italic")

    # dedup + embed note above fusion
    ax.text(8.92, 4.62, "+ dedup, embed,\nincremental resolution",
            ha="center", fontsize=7.2, color=C["slate"], style="italic")

    _save(fig, "multimodal_workflow")


# ============================================================================
# 4. Real-case F1 per entity type
# ============================================================================
def fig_real_case_f1():
    # Measured metrics: full system, mean over 3 runs (gpt-4.1, temp 0.1),
    # 28 documented Indian criminal cases (two-annotator gold).
    # F1 error bars = standard deviation.
    types = ["Person", "CrimeType", "Weapon/\nmethod", "Location",
             "TimeEvent", "Overall\n(micro)"]
    P = [0.82, 0.89, 0.79, 0.19, 0.85, 0.72]
    R = [0.95, 0.94, 0.66, 0.44, 0.91, 0.86]
    F = [0.88, 0.91, 0.72, 0.27, 0.88, 0.78]
    Fsd = [0.00, 0.01, 0.01, 0.03, 0.01, 0.01]

    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    x = np.arange(len(types)); w = 0.26
    ax.bar(x - w, P, w, label="Precision", color=C["blue"], edgecolor="white", linewidth=0.6)
    ax.bar(x,     R, w, label="Recall",    color=C["teal"], edgecolor="white", linewidth=0.6)
    ax.bar(x + w, F, w, yerr=Fsd, label="F1 ($\\pm$sd, 3 runs)", color=C["amber"],
           edgecolor="white", linewidth=0.6,
           error_kw=dict(ecolor=C["ink"], elinewidth=0.9, capsize=2.5))

    for i in range(len(types)):
        for off, val in ((-w, P[i]), (0, R[i])):
            ax.text(x[i] + off, val + 0.02, f"{val:.2f}", ha="center",
                    va="bottom", fontsize=8.2, color=C["ink"])
        ax.text(x[i] + w, F[i] + Fsd[i] + 0.02, f"{F[i]:.2f}", ha="center",
                va="bottom", fontsize=8.2, color=C["ink"])

    ax.axvline(len(types) - 1.5, color=C["gridgray"], lw=1.0, ls=(0, (4, 3)))
    ax.set_xticks(x); ax.set_xticklabels(types, fontsize=9.8)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score", fontsize=11)
    ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.tick_params(labelsize=9.5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C["gridgray"], lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(ncol=3, frameon=False, fontsize=9.6, loc="upper center",
              bbox_to_anchor=(0.5, 1.12))
    _save(fig, "real_case_f1")


def fig_ablation():
    # Measured ablation: ontology-guided full system vs. naive JSON baseline,
    # F1 per type + micro/macro, mean over 3 runs on the 28 real cases.
    types = ["Person", "Crime\nType", "Weapon", "Location", "Time\nEvent",
             "Micro", "Macro"]
    base = [0.88, 0.75, 0.00, 0.13, 0.91, 0.68, 0.53]
    full = [0.88, 0.91, 0.72, 0.27, 0.88, 0.78, 0.73]

    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    x = np.arange(len(types)); w = 0.38
    ax.bar(x - w/2, base, w, label="Baseline (naive JSON, no ontology)",
           color=C["slate"], edgecolor="white", linewidth=0.6)
    ax.bar(x + w/2, full, w, label="Full (ontology-guided)",
           color=C["blue"], edgecolor="white", linewidth=0.6)
    for i in range(len(types)):
        ax.text(x[i] - w/2, base[i] + 0.012, f"{base[i]:.2f}", ha="center",
                va="bottom", fontsize=8.2, color=C["ink"])
        ax.text(x[i] + w/2, full[i] + 0.012, f"{full[i]:.2f}", ha="center",
                va="bottom", fontsize=8.2, color=C["ink"])
    ax.axvline(len(types) - 2.5, color=C["gridgray"], lw=1.0, ls=(0, (4, 3)))
    ax.set_xticks(x); ax.set_xticklabels(types, fontsize=9.2)
    ax.set_ylim(0, 1.18); ax.set_ylabel("F1", fontsize=10.5)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C["gridgray"], lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(ncol=1, frameon=False, fontsize=9.0, loc="upper right",
              bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Ontology guidance is decisive: it recovers Weapon and lifts "
                 "CrimeType, Location, and both micro-\nand macro-F1, while matching "
                 "the baseline on Person/TimeEvent", fontsize=9.0, color=C["ink"], pad=8)
    _save(fig, "ablation")


# ============================================================================
# 5. Image-evaluation summary (honest negatives)
# ============================================================================
def fig_image_eval_summary():
    # task, score, metric label, ground-truth alignment category
    rows = [
        ("Bloodstain",  1.00, "F1",        "aligned"),
        ("Wound",       0.67, "F1",        "partial"),
        ("Ballistics",  0.50, "accuracy",  "partial"),
        ("Document",    0.05, "F1",        "hard task"),
        ("Tool marks",  0.02, "accuracy",  "misaligned"),
        ("Fingerprint", 0.02, "accuracy",  "misaligned"),
    ]
    names  = [r[0] for r in rows]
    scores = [r[1] for r in rows]
    metric = [r[2] for r in rows]
    align  = [r[3] for r in rows]

    cmap = {"aligned": C["green"], "partial": C["amber"],
            "hard task": C["red"], "misaligned": C["red"]}
    colors = [cmap[a] for a in align]

    fig, ax = plt.subplots(figsize=(10.6, 3.8))
    y = np.arange(len(names))[::-1]
    ax.barh(y, scores, color=colors, edgecolor="white", height=0.64, linewidth=0.6)
    for yi, s, m in zip(y, scores, metric):
        ax.text(s + 0.02, yi, f"{s:.2f} {m}", va="center",
                ha="left", fontsize=15.5, color=C["ink"])
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=17.0)
    ax.set_xlim(0, 1.40); ax.set_xlabel("Evaluation score", fontsize=16.5)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="x", labelsize=15.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=C["gridgray"], lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # reliability threshold guide
    ax.axvline(0.8, color=C["green"], lw=1.0, ls=(0, (4, 3)), alpha=0.6)

    legend = [
        mpatches.Patch(color=C["green"], label="GT aligned"),
        mpatches.Patch(color=C["amber"], label="partial / proxy GT"),
        mpatches.Patch(color=C["red"],   label="misaligned / hard"),
    ]
    ax.legend(handles=legend, frameon=False, fontsize=14.0,
              loc="lower right", bbox_to_anchor=(1.0, 0.05),
              handlelength=1.3, handletextpad=0.5, labelspacing=0.35)
    ax.set_title("Image analysis is reliable only where ground truth is "
                 "task-aligned", fontsize=15.5, color=C["ink"], pad=8)
    _save(fig, "image_eval_summary")


# ============================================================================
# 6. Cross-case entity resolution
# ============================================================================
def fig_entity_resolution():
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.6); ax.axis("off")

    # Stage 1: per-case islands with duplicate mentions
    ax.text(1.5, 5.5, "1. Per-case islands", fontsize=9.6, weight="bold",
            color=C["blue"], ha="center", va="top")
    box(ax, 0.2, 3.5, 2.6, 1.25,
        "Case A\n“Justice R. Banerjee”\n“cyanide”", C["blue_l"],
        ec=C["blue"], fs=8.0)
    box(ax, 0.2, 1.4, 2.6, 1.25,
        "Case B\n“R. Banerjee, J.”\n“KCN (cyanide)”", C["blue_l"],
        ec=C["blue"], fs=8.0)

    # Stage 2: embedding blocking
    ax.text(4.625, 5.5, "2. Embedding\nblocking", fontsize=9.6,
            weight="bold", color=C["violet"], ha="center", va="top",
            linespacing=1.3)
    box(ax, 3.7, 2.35, 1.85, 1.5,
        "cosine KNN\nover node\nembeddings\n→ candidate\npairs", C["violet_l"],
        ec=C["violet"], fs=8.0)
    arrow(ax, (2.82, 4.1), (3.68, 3.4), color=C["blue"], rad=-0.12)
    arrow(ax, (2.82, 2.0), (3.68, 2.9), color=C["blue"], rad=0.12)

    # Stage 3: verification
    ax.text(7.075, 5.5, "3. Verification", fontsize=9.6, weight="bold",
            color=C["amber"], ha="center", va="top")
    box(ax, 6.0, 2.1, 2.15, 2.0,
        "name similarity\n+ role match\n+ shared context\n\nconservative;\nreject weak\nor conflicting",
        C["amber_l"], ec=C["amber"], fs=8.0)
    arrow(ax, (5.57, 3.1), (5.98, 3.1), color=C["violet"])

    # Stage 4: non-destructive SAME_AS
    ax.text(10.15, 5.5, "4. Non-destructive\nlink", fontsize=9.6,
            weight="bold", color=C["teal"], ha="center", va="top",
            linespacing=1.3)
    box(ax, 8.7, 3.5, 2.9, 1.0, "Case A entity", C["blue_l"], ec=C["blue"], fs=8.2)
    box(ax, 8.7, 1.5, 2.9, 1.0, "Case B entity", C["blue_l"], ec=C["blue"], fs=8.2)
    arrow(ax, (10.15, 3.48), (10.15, 2.52), color=C["teal"], style="<|-|>", lw=2.0)
    ax.text(10.32, 3.0, "SAME_AS\nscore, basis", ha="left", va="center",
            fontsize=7.8, color=C["teal"], weight="bold")
    arrow(ax, (8.18, 3.35), (8.66, 3.92), color=C["amber"], rad=0.12)
    arrow(ax, (8.18, 2.85), (8.66, 2.08), color=C["amber"], rad=-0.12)

    ax.text(6.0, 0.45,
            "Provenance is preserved: original per-case nodes are never merged "
            "or deleted — only linked, so every fact keeps its source.",
            ha="center", fontsize=7.8, color=C["muted"], style="italic")
    _save(fig, "entity_resolution")


# ============================================================================
# 7. Example evidence chain
# ============================================================================
def fig_evidence_chain():
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(-0.45, 6.4); ax.axis("off")

    def node(x, y, label, fc, ec, w=2.0, h=0.78, fs=8.4):
        box(ax, x - w / 2, y - h / 2, w, h, label, fc, ec=ec, fs=fs,
            weight="bold", rounding=0.05)
        return (x, y)

    # nodes (left-to-right reading order)
    node(1.5, 3.3, "Case", C["grayfill"], C["slate"])
    node(4.0, 5.0, "Person\n(victim)", C["blue_l"], C["blue"])
    node(4.0, 2.15, "Person\n(accused)", C["blue_l"], C["blue"])
    node(6.7, 5.0, "CauseOfDeath", C["red_l"], C["red"])
    node(6.7, 3.3, "Evidence\n(toxicology)", C["amber_l"], C["amber"])
    node(6.7, 0.75, "TimeEvent\n(meal served)", C["teal_l"], C["teal"])
    node(9.3, 0.75, "TimeEvent\n(collapse)", C["teal_l"], C["teal"])
    node(10.4, 4.15, "Hypothesis", C["violet_l"], C["violet"], w=2.1)

    def edge(p0, p1, label, color, rad=0.0, ls="-", lw=1.6, style="-|>",
             lpos=None, fs=7.2):
        arrow(ax, p0, p1, color=color, rad=rad, ls=ls, lw=lw, style=style)
        if lpos is not None:
            ax.text(lpos[0], lpos[1], label, ha="center", va="center",
                    fontsize=fs, color=color, weight="bold")

    # case -> actors (labels kept to the left, clear of the arrows)
    edge((2.45, 3.55), (3.05, 4.7), "INVOLVED_IN", C["blue"], rad=-0.1,
         lpos=(1.95, 4.55))
    edge((2.45, 3.05), (3.05, 2.35), "INVOLVED_IN", C["blue"], rad=0.1,
         lpos=(2.0, 2.78))
    # medical cause attaches to the victim
    edge((5.7, 5.0), (5.0, 5.0), "CAUSE_OF_DEATH_OF", C["red"],
         lpos=(5.35, 5.62), fs=6.7)
    # case evidence -> cause of death
    edge((2.5, 3.3), (5.7, 3.3), "HAS_EVIDENCE", C["amber"], lpos=(4.05, 3.52))
    edge((6.7, 3.69), (6.7, 4.61), "LED_TO_DEATH", C["red"],
         lpos=(7.45, 4.15), fs=6.7)
    # timeline (HAPPENED_ON routed clearly below the accused box)
    edge((1.75, 2.9), (5.72, 0.82), "HAPPENED_ON", C["teal"], rad=-0.55,
         lpos=(4.55, 0.5))
    edge((7.7, 0.75), (8.3, 0.75), "PRECEDES", C["teal"], lpos=(8.0, 1.48))
    # supporting evidence -> hypothesis (auditable edges)
    edge((7.7, 3.45), (9.35, 4.0), "SUPPORTS", C["green"], rad=-0.12,
         lpos=(8.55, 3.5))
    edge((7.75, 4.95), (9.4, 4.45), "SUPPORTS", C["green"], rad=0.1,
         lpos=(8.5, 5.05))
    edge((9.6, 1.15), (10.15, 3.7), "SUPPORTS", C["green"], rad=-0.22,
         lpos=(10.45, 2.4))

    # legend for edge semantics
    leg = [
        Line2D([0], [0], color=C["green"], lw=2, label="supporting edge"),
        Line2D([0], [0], color=C["red"], lw=2, label="causal (medical) edge"),
        Line2D([0], [0], color=C["blue"], lw=2, label="case / actor edge"),
        Line2D([0], [0], color=C["teal"], lw=2, label="temporal edge"),
    ]
    ax.legend(handles=leg, frameon=False, fontsize=8.2, ncol=4,
              loc="lower center", bbox_to_anchor=(0.5, 0.0),
              columnspacing=1.6, handlelength=1.6)
    ax.text(6, 6.15, "Evidence chain: actors, medical cause, timeline, and a "
            "hypothesis with auditable supporting edges", ha="center",
            fontsize=9.6, weight="bold", color=C["ink"])
    _save(fig, "evidence_chain")


if __name__ == "__main__":
    fig_architecture()
    fig_ontology_overview()
    fig_multimodal_workflow()
    fig_real_case_f1()
    fig_ablation()
    fig_image_eval_summary()
    fig_entity_resolution()
    fig_evidence_chain()
    print("\nAll figures written to", os.path.abspath(FIG_DIR))
