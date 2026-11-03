"""
Figures for the resource version of the paper (ecir_submission/idiomQa_resource.tex).
Every figure is computed from the released data, so re-running this script after
the re-judged scores are merged updates all of them.

  fig_da_chain.pdf        DA revisions in the UGC circulars (Temporal motivation)
  fig_transcription.pdf   who was wrong when the VLM and OCR texts disagreed, by page type,
                          and the kinds of VLM error the hand check found
  fig_profile.pdf         questions per type, gold-answer length, OM issue years
  fig_rq1.pdf             Answer Correctness with 95% CIs (bounds: closed book, oracle) beside\n                          the other five RAGAS metrics
  fig_rq2.pdf             Answer Correctness by question type and system

    python scripts/paper/make_figures.py     (omAgentic env)
"""
import glob
import json
import os
from pathlib import Path

import frontmatter
import matplotlib
import matplotlib.dates
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = Path("ecir_submission/figures")
# Until the re-judged scores are merged, every data figure describes the previous dataset version
# (before the 2026-11-03 gold fixes: 71 Adversarial), which is the version the scores were judged on.
# Set PREVIOUS_VERSION = False after merging to switch to the current dataset and its types.
PREVIOUS_VERSION = os.environ.get("PREVIOUS_VERSION", "1") == "1"
DATASET = ("data/eval/v4/idiom_qa_v4.json.bak_pre_gold_fixes_20261103" if PREVIOUS_VERSION
           else "data/eval/v4/idiom_qa_v4.json")
RAGAS = os.environ.get("RAGAS_DIR", "data/eval/eval_results/ragas_deepseek_full475_matched/detailed_scores")
TEXTWIDTH = 4.8  # inches, LNCS text width (12.2 cm)

# Reference palette (dataviz skill, light mode), validated for adjacent pairs.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeXGyreTermes", "Nimbus Roman", "Liberation Serif"],
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "hatch.color": "white", "hatch.linewidth": 0.8, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

SYSTEMS = [("naive_rag", "Naive RAG"), ("hybrid", "Hybrid RAG"), ("rerank", "Rerank RAG"),
           ("hyde", "HyDE"), ("crag", "CRAG"), ("adaptive_rag", "Adaptive-RAG"),
           ("deep_rag", "DeepRAG"), ("r3_rag", "R3-RAG"), ("search_r1", "Search-R1"),
           ("gear", "GEAR"), ("graphrag", "GraphRAG"), ("idiom_rag", "IDiOM-RAG")]
TYPES = ["Factoid", "Conceptual", "Relational", "Multi-part", "Adversarial", "CrossHop", "Temporal",
         "Aggregation"]


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def fig_da_chain():
    """DA revisions recorded in the circulars, by effective date, for staff on the pre-revised
    (6th CPC) and revised (7th CPC) pay scales (gold answer of the Temporal DA item)."""
    sixth = [("2012-01-01", 58, 65), ("2012-07-01", 65, 72), ("2013-07-01", 80, 90),
             ("2015-07-01", 113, 119), ("2018-01-01", 139, 142)]
    seventh = [("2021-07-01", 28, 31), ("2022-01-01", 31, 34), ("2024-01-01", 46, 50)]
    fig, ax = plt.subplots(figsize=(3.4, 1.3))
    for series, color, marker, label in ((sixth, ORANGE, "s", "old pay scale"),
                                         (seventh, BLUE, "o", "new pay scale")):
        d = pd.to_datetime([s[0] for s in series])
        for x, (_, old, new) in zip(d, series):
            ax.plot([x, x], [old, new], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(d, [s[2] for s in series], marker, color=color, markersize=4.5, markeredgecolor="white",
                markeredgewidth=0.8, label=label)
        for k, (x, (_, old, new)) in enumerate(zip(d, series)):
            dx = {0: -5, 1: 5}.get(k, 0)  # the first two points of each series sit close
            dy = -11 if series is sixth and k == 1 else 4  # 72% goes below, clear of the 90% point
            ax.annotate(f"{new}%", (x, new), xytext=(dx, dy), textcoords="offset points",
                        ha={-5: "right", 5: "left"}.get(dx, "center"), fontsize=6.5, color=INK2)
    ax.set_ylabel("DA (% of pay)")
    ax.set_ylim(0, 160)
    grid(ax)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3,
              columnspacing=1.2, fontsize=6.5, borderaxespad=0.2)
    save(fig, "fig_da_chain.pdf")


def fig_jrf():
    """Monthly JRF base fellowship quoted by the JU notices of Dec 2024 - Jun 2025 (issue dates).
    The national rate became Rs 37,000 from 01.01.2023 (DST OMs of 18.10.2023 and 14.02.2024)."""
    notices = {37: ["2024-12-23", "2025-01-30", "2025-02-05", "2025-02-05", "2025-02-17", "2025-02-21",
                    "2025-02-24", "2025-03-18", "2025-04-09", "2025-04-25", "2025-05-29", "2025-06-09"],
               31: ["2025-01-09", "2025-01-16", "2025-02-19", "2025-06-02"],
               25: ["2025-04-09", "2025-04-17", "2025-04-29", "2025-05-14", "2025-05-27"]}
    style = {37: (BLUE, "o", "Rs 37,000, the revised rate"), 31: (ORANGE, "s", "Rs 31,000, the rate before 2023"),
             25: (MUTED, "^", "Rs 25,000, state-funded projects")}
    fig, ax = plt.subplots(figsize=(3.4, 1.3))
    for amt, dates in notices.items():
        d = pd.to_datetime(dates)
        k = pd.Series(d).groupby(d).cumcount().to_numpy()  # notices issued on the same day sit side by side
        col, mk, label = style[amt]
        ax.plot(d + pd.to_timedelta(3 * k, unit="D"), [amt] * len(d), mk, color=col, markersize=4.2, markeredgecolor="white",
                markeredgewidth=0.6, label=label, linestyle="none")
    ax.set_ylabel("JRF per month\n(Rs thousand)")
    ax.set_ylim(22, 41)
    ax.set_yticks([25, 31, 37])
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.set_xlim(pd.Timestamp("2024-12-01"), pd.Timestamp("2025-07-01"))
    ax.set_xlabel("issue date (December 2024 to June 2025)")
    grid(ax)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, handletextpad=0.3,
              fontsize=6.3, borderaxespad=0.2)
    save(fig, "fig_jrf.pdf")


def fig_transcription():
    """Hand check of VLM-vs-OCR disagreements (counts from the review exports)."""
    # Scanned pages combine the 26 with an embedded scanner-OCR layer and the 82 read with EasyOCR.
    pages = ["Originally digital\nPDF (31 pages)", "Scanned PDF\n(108 pages)"]
    ocr = np.array([17, 296 + 607])
    vlm_wrong = np.array([1, 13 + 19])
    vlm_omit = np.array([1, 14 + 74])
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.7), gridspec_kw={"width_ratios": [1.45, 1]})
    y = np.arange(len(pages))[::-1]
    total = ocr + vlm_wrong + vlm_omit
    left = np.zeros(len(pages))
    for vals, color, hatch, label in ((ocr, MUTED, "", "OCR text wrong"), (vlm_wrong, BLUE, "", "VLM wrong"),
                                      (vlm_omit, ORANGE, "////", "VLM omitted")):
        a.barh(y, vals / total * 100, left=left, height=0.55, color=color, label=label, edgecolor="white",
               linewidth=1, hatch=hatch)
        left += vals / total * 100
    for yi, n in zip(y, total):
        a.text(101, yi, f"n={n}", va="center", fontsize=6.3, color=INK2)
    a.set_yticks(y, pages)
    a.set_xlim(0, 100)
    a.set_xlabel("share of disagreements (%)")
    a.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=6.3,
             handlelength=0.9, columnspacing=0.8, borderaxespad=0.2)
    kinds = ["Omitted text", "Wrong number", "Wrong issue date", "Wrong word"]
    counts = [78, 23, 12, 10]
    yb = np.arange(len(kinds))[::-1]
    b.barh(yb, counts, height=0.55, color=BLUE)
    for yi, c in zip(yb, counts):
        b.text(c + 2, yi, str(c), va="center", fontsize=6.5, color=INK2)
    b.set_yticks(yb, kinds)
    b.set_xlim(0, 95)
    b.set_xlabel("VLM errors corrected")
    grid(b, "x")
    b.set_title("VLM errors by kind", loc="left", color=INK)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_transcription.pdf")


def fig_profile():
    qs = json.load(open(DATASET, encoding="utf-8"))
    df = pd.DataFrame({"type": [q["question_type"] for q in qs],
                       "alen": [len(q["expected_output"].split()) for q in qs]})
    years = []
    for f in glob.glob("data/DataFiles_v2/**/*_om_text_files/*.txt", recursive=True):
        m = frontmatter.load(f).metadata
        inst = Path(f).parts[2].upper()
        try:
            years.append((inst, int(str(m.get("issue_date"))[:4])))
        except ValueError:
            pass
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.75), gridspec_kw={"width_ratios": [1, 1.15, 1.1]})
    counts = df["type"].value_counts().reindex(TYPES)
    y = np.arange(len(TYPES))[::-1]
    colors = [BLUE] * 5 + [ORANGE] * 3
    a.barh(y, counts.values, height=0.6, color=colors)
    for yi, n in zip(y, counts.values):
        a.text(n + 2, yi, str(n), va="center", fontsize=6.3, color=INK2)
    a.set_yticks(y, TYPES)
    a.set_xlim(0, 115)
    a.set_title("(a) questions", loc="left", color=INK)
    grid(a, "x")
    data = [df.loc[df["type"] == t, "alen"].values for t in TYPES]
    bp = b.boxplot(data[::-1], vert=False, widths=0.55, patch_artist=True, showfliers=False,
                   medianprops={"color": INK, "linewidth": 1}, whiskerprops={"color": MUTED, "linewidth": 0.7},
                   capprops={"color": MUTED, "linewidth": 0.7}, boxprops={"linewidth": 0})
    for patch, col in zip(bp["boxes"], colors[::-1]):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    b.set_xscale("log")
    b.set_xticks([10, 30, 100, 300], ["10", "30", "100", "300"])
    b.set_yticks([])
    b.set_title("(b) gold-answer words", loc="left", color=INK)
    grid(b, "x")
    insts = ["UGC", "CSIR", "IACS", "JU"]
    bins = np.arange(2012, 2027)
    bottom = np.zeros(len(bins) - 1)
    for inst, col, hatch in zip(insts, (BLUE, ORANGE, AQUA, YELLOW), ("", "////", "", "...")):
        h, _ = np.histogram([yr for i, yr in years if i == inst], bins=bins)
        c.bar(bins[:-1] + 0.5, h, width=0.85, bottom=bottom, color=col, label=inst, edgecolor="white",
              linewidth=0.6, hatch=hatch)
        bottom += h
    c.set_xticks([2012, 2016, 2020, 2025], ["2012", "2016", "2020", "2025"])
    c.set_title("(c) OMs by issue year", loc="left", color=INK)
    c.legend(frameon=False, loc="upper left", fontsize=6.3, handlelength=0.8, labelspacing=0.2)
    grid(c)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_profile.pdf")


def load_ragas():
    out = {}
    for s, _ in SYSTEMS + [("closed_book", ""), ("gold_context", "")]:
        out[s] = pd.read_csv(f"{RAGAS}/{s}_ragas_detailed.csv")
    return out


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


METRICS = [("answer_correctness", "Answer\ncorrectness"), ("answer_similarity", "Answer\nsimilarity"),
           ("faithfulness", "Faithful-\nness"), ("answer_relevancy", "Answer\nrelevancy"),
           ("context_precision", "Context\nprecision"), ("context_recall", "Context\nrecall")]


def fig_rq1(r):
    """One dot panel per RAGAS metric, all sharing the system rows (sorted by Answer Correctness):
    mean with 95% bootstrap CI, and the closed-book and oracle runs as bands where they apply."""
    XLO = 0.4
    order = sorted(SYSTEMS, key=lambda s: r[s[0]]["answer_correctness"].mean())
    fig, axes = plt.subplots(1, len(METRICS), figsize=(TEXTWIDTH, 2.3), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    y = np.arange(len(order))
    for ax, (m, title) in zip(axes, METRICS):
        for ref, col in (("closed_book", MUTED), ("gold_context", AQUA)):
            if m not in r[ref]:
                continue
            mean, lo, hi = boot_ci(r[ref][m])
            if mean < XLO:  # closed book lies far left of the systems: note its value instead
                ax.annotate(f"\u2190 closed book {mean:.2f}".replace("0.", "."), (XLO + 0.01, -0.45),
                            fontsize=5.4, color=INK2, va="center")
                continue
            ax.axvspan(lo, hi, color=col, alpha=0.18, linewidth=0)
            ax.axvline(mean, color=col, linewidth=1)
        stats = [boot_ci(r[s][m]) for s, _ in order]
        for yi, (s, _), (mean, lo, hi) in zip(y, order, stats):
            col = BLUE if s == "idiom_rag" else INK2
            ax.plot([lo, hi], [yi, yi], color=col, linewidth=1.1, solid_capstyle="round")
            ax.plot(mean, yi, "o", color=col, markersize=3.4, markeredgecolor="white", markeredgewidth=0.6)
        ax.set_title(title, fontsize=6.3, color=INK2)
        ax.set_xlim(XLO, 0.9)
        ax.set_xticks([0.5, 0.7, 0.9], [".5", ".7", ".9"], fontsize=6)
        grid(ax, "x")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [lbl for _, lbl in order])
    axes[0].get_yticklabels()[[s for s, _ in order].index("idiom_rag")].set_color(BLUE)
    axes[0].set_ylim(-1.1, len(order) - 0.4)
    save(fig, "fig_rq1.pdf")


def current_types(r):
    """Question type of each scored row from the current dataset (re-typed items move)."""
    qs = json.load(open(DATASET, encoding="utf-8"))
    for d in r.values():
        d["question_type"] = [qs[int(i) - 1]["question_type"] for i in d["index"]]


def fig_rq2(r):
    cols = [lbl for _, lbl in SYSTEMS] + ["Closed book", "Gold evidence"]
    keys = [s for s, _ in SYSTEMS] + ["closed_book", "gold_context"]
    M = np.array([[r[k].loc[r[k]["question_type"] == t, "answer_correctness"].mean() for k in keys]
                  for t in TYPES])
    n = [int((r["naive_rag"]["question_type"] == t).sum()) for t in TYPES]
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.25))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=0.9, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.6,
                    color="white" if M[i, j] > 0.55 else INK)
    ax.axvline(len(SYSTEMS) - 0.5, color="white", linewidth=2)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(TYPES)), [f"{t} ({k})" for t, k in zip(TYPES, n)])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, length=0)
    save(fig, "fig_rq2.pdf")


def fig_retrieval():
    """(a) nearest-sibling similarity between OMs; (b) single-document Hit@1 vs Recall@10."""
    from sentence_transformers import SentenceTransformer
    files = sorted(glob.glob("data/DataFiles_v2/**/*_om_text_files/*.txt", recursive=True))
    texts = [frontmatter.load(f).content[:2000] for f in files]
    emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu").encode(texts, normalize_embeddings=True)
    S = emb @ emb.T
    np.fill_diagonal(S, -1)
    nn = S.max(axis=1)
    qs = json.load(open(DATASET, encoding="utf-8"))
    sd = {f"q{i + 1:03d}" for i, q in enumerate(qs) if q["question_type"] in TYPES[:4]}
    qrels = {}
    for line in open("data/eval/v4/qrels/idiom_qa_v4.qrels"):
        q, _, d, rel = line.split()
        if int(rel) > 0:
            qrels.setdefault(q.lower(), set()).add(d)
    runs = [("bm25", "BM25"), ("dense-bge-small", "Dense (bge-small)"), ("dense-bge-m3", "Dense (bge-m3)"),
            ("hybrid", "Hybrid"), ("hybrid+bge-reranker", "Hybrid + reranker")]
    res = []
    for key, label in runs:
        ranked = {}
        for line in open(f"data/eval/eval_results/retrieval_qrels_v4/runs/{key}.run"):
            q, _, d, rank, *_ = line.split()
            ranked.setdefault(q.lower(), []).append((int(rank), d))
        hit1 = r10 = n = 0
        for q in sd & set(qrels):
            docs = [d for _, d in sorted(ranked.get(q, []))]
            n += 1
            hit1 += bool(docs[:1]) and docs[0] in qrels[q]
            r10 += len(set(docs[:10]) & qrels[q]) / len(qrels[q])
        res.append((label, hit1 / n, r10 / n))
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.65), gridspec_kw={"width_ratios": [1, 1.25]})
    a.hist(nn, bins=np.arange(0.6, 1.001, 0.025), color=BLUE, edgecolor="white", linewidth=0.6)
    share = (nn >= 0.9).mean()
    a.axvline(0.9, color=MUTED, linewidth=0.8, linestyle="--")
    a.text(0.61, a.get_ylim()[1] * 0.85, f"{share:.0%} of OMs have a\nsibling at \u2265 0.9", ha="left",
           fontsize=6.3, color=INK2)
    a.set_xlabel("similarity to the closest other OM")
    a.set_ylabel("OMs")
    a.set_title("(a) near-identical sibling notices", loc="left", color=INK)
    grid(a)
    y = np.arange(len(res))[::-1]
    for yi, (label, h1, r10) in zip(y, res):
        b.plot([h1, r10], [yi, yi], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
        b.plot(h1, yi, "D", color=ORANGE, markersize=4, markeredgecolor="white", markeredgewidth=0.8, zorder=2)
        b.plot(r10, yi, "o", color=BLUE, markersize=4.5, markeredgecolor="white", markeredgewidth=0.8, zorder=2)
    b.plot([], [], "D", color=ORANGE, markersize=4, label="right OM ranked first (Hit@1)")
    b.plot([], [], "o", color=BLUE, label="right OM in top 10 (R@10)")
    b.set_yticks(y, [r_[0] for r_ in res])
    b.set_xlim(0.65, 1.0)
    b.set_xlabel("single-document questions")
    b.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=6.3, ncol=1,
             handletextpad=0.2, borderaxespad=0.1)
    grid(b, "x")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_retrieval.pdf")
    print("nearest-sibling >=0.9:", round(share, 3), "| retrieval:", [(l, round(h, 3), round(r, 3)) for l, h, r in res])


def main():
    os.chdir(Path(__file__).resolve().parents[2])
    fig_da_chain()
    fig_jrf()
    fig_transcription()
    fig_profile()
    fig_retrieval()
    r = load_ragas()
    if not PREVIOUS_VERSION:
        current_types(r)
    fig_rq1(r)
    fig_rq2(r)


if __name__ == "__main__":
    main()
