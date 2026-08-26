"""The three figures that carry the rate-law audit.

Each one states a claim the campaign's own measurements make, and each is
chosen so the claim is visible rather than argued:

1. the adapter is nowhere near the information limit its rate is supposed to
   measure -- a log-log scatter against the identity line, where the identity
   line *is* the hypothesis and every arm sits three orders of magnitude off it;
2. the supervised token count predicts the rate in opposite directions within
   and between receivers -- one panel, per-receiver fits against same-corpus
   connectors, which is the only form that shows both readings of one dataset;
3. the campaign scored candidates on the axis where a corpus statistic can win
   and never on the axis its claim lives -- candidates placed by what they
   explain on each axis, beside what that cost on a held-out receiver.

Palette and chrome follow the validated light-mode reference: categorical slots
blue/orange/aqua/violet (checked all-pairs, worst CVD dE 9.2, worst
normal-vision dE 16.3), hairline grid one shade off the surface, thin marks,
selective direct labels.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")


def _style(axes) -> None:
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, linewidth=0.6, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)
    for label in (*axes.get_xticklabels(), *axes.get_yticklabels()):
        label.set_color(INK_SECONDARY)


def _figure(width: float, height: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(width, height), facecolor=SURFACE, dpi=200)
    return plt, figure


def _receiver_label(key: str) -> str:
    return {
        "mistral_7b_base": "Mistral 7B",
        "llama31_8b_base": "Llama 3.1 8B",
        "qwen25_7b_base": "Qwen2.5 7B",
        "qwen3_8b_base": "Qwen3 8B",
        "qwen25_math_7b_base": "Qwen2.5-Math 7B",
        "gemma2_9b_base": "Gemma 2 9B",
        "gemma2_9b_instruct": "Gemma 2 9B it",
    }.get(str(key), str(key).replace("_", " "))


def _finite(rows: Iterable[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
    kept = []
    for row in rows:
        try:
            values = [float(row[key]) for key in keys]
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) and value > 0 for value in values):
            kept.append(row)
    return kept


def adapter_against_information(arms: list[dict[str, Any]], path: Path) -> Path:
    """How many times larger the adapter is than the information it carries.

    Plotting the two bit counts against an identity line wastes the frame:
    they are decades apart, so the data sits in one corner. The ratio puts the
    claim on the y-axis and the reference at one is the hypothesis the
    project's title makes.

    Both charges are drawn because "you only counted the held-out split" is
    the first objection. The generous charge extrapolates the measured bits
    per token to every row of the training corpus, which is the largest
    corpus the adapter could be said to serve, and it moves the answer from
    three orders of magnitude to two rather than closing it.
    """
    usable = _finite(
        arms,
        "adapter_bits_at_r90",
        "ceiling_heldout_bits_saved",
        "heldout_tokens",
        "train_response_tokens",
        "corpus_rows",
    )
    rows = []
    for row in usable:
        adapter = float(row["adapter_bits_at_r90"])
        saved = float(row["ceiling_heldout_bits_saved"])
        # The stored train and held-out blocks both score 256 rows, so the
        # corpus charge has to be built from tokens per row times the rows the
        # arm actually trained on.
        corpus_tokens = (
            float(row["train_response_tokens"]) / 256.0 * float(row["corpus_rows"])
        )
        corpus = saved * corpus_tokens / float(row["heldout_tokens"])
        rows.append((adapter / saved, adapter / corpus))
    rows.sort()
    plt, figure = _figure(7.8, 4.9)
    axes = figure.add_subplot(111)
    _style(axes)
    axes.grid(axis="x", visible=False)
    index = list(range(len(rows)))
    axes.axhline(1.0, color=INK, linewidth=1.4, zorder=2)
    axes.scatter(
        index, [row[0] for row in rows], s=26, facecolor=SERIES[0],
        edgecolor=SURFACE, linewidth=0.8, zorder=4,
        label="charged against the held-out split",
    )
    axes.scatter(
        index, [row[1] for row in rows], s=26, facecolor=SERIES[1],
        edgecolor=SURFACE, linewidth=0.8, zorder=3,
        label="charged against every row of the training corpus",
    )
    axes.set_yscale("log")
    axes.set_xlim(-1.5, len(rows) + 7)
    axes.set_ylim(0.35, max(row[0] for row in rows) * 6)
    axes.set_xticks([])
    held = sorted(row[0] for row in rows)
    corpus = sorted(row[1] for row in rows)
    axes.text(
        0.5, 1.25, "an adapter that carries exactly the bits it saves",
        transform=axes.get_yaxis_transform(), ha="center", va="bottom",
        fontsize=8.5, color=INK,
    )
    for values, colour in ((held, SERIES[0]), (corpus, SERIES[1])):
        axes.annotate(
            f"median {values[len(values) // 2]:,.0f}x",
            xy=(len(rows) - 1, values[len(values) // 2]),
            xytext=(8, 0), textcoords="offset points",
            fontsize=8.5, color=colour, va="center",
        )
    axes.set_xlabel(
        f"{len(rows)} model x corpus arms, sorted", color=INK_SECONDARY, fontsize=9
    )
    axes.set_ylabel(
        "serialized adapter / bits it saves", color=INK_SECONDARY, fontsize=9
    )
    axes.set_title(
        "Even charged against the whole corpus, the adapter is 100x its own information",
        color=INK, fontsize=11.5, loc="left", pad=24,
    )
    axes.legend(
        frameon=False, fontsize=8.5, loc="upper left",
        labelcolor=INK_SECONDARY, handletextpad=0.4, borderpad=0.2,
    )
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return Path(path)


def _ordinary_least_squares(
    xs: list[float], ys: list[float]
) -> tuple[float, float] | None:
    if len(xs) < 3:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    return slope, mean_y - slope * mean_x


def token_count_reversal(arms: list[dict[str, Any]], path: Path) -> Path:
    """One variable, two readings, opposite signs.

    Left: inside a receiver, more supervised tokens means a larger rate. Right:
    the same corpus on two receivers, oriented so the more verbose tokenizer
    comes first -- the rate falls. Overlaying the two on one panel was tried
    and rejected: the connectors turn into spaghetti and the reader cannot see
    that most of them descend.
    """
    import itertools

    usable = _finite(arms, "train_response_tokens", "r_star_bits_per_value")
    plt, figure = _figure(11.0, 4.8)
    grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.0), wspace=0.24)
    axes = figure.add_subplot(grid[0, 0])
    _style(axes)
    receivers = sorted({str(row["model_key"]) for row in usable})
    colours = {
        name: SERIES[index % len(SERIES)] for index, name in enumerate(receivers)
    }
    for name in receivers:
        group = [row for row in usable if str(row["model_key"]) == name]
        xs = [math.log2(float(row["train_response_tokens"])) for row in group]
        ys = [float(row["r_star_bits_per_value"]) for row in group]
        axes.scatter(
            xs, ys, s=26, facecolor=colours[name], edgecolor=SURFACE,
            linewidth=0.9, zorder=4,
            label=f"{_receiver_label(name)} (n={len(group)})",
        )
        fit = _ordinary_least_squares(xs, ys)
        if fit:
            slope, intercept = fit
            edge = [min(xs), max(xs)]
            axes.plot(
                edge, [intercept + slope * value for value in edge],
                color=colours[name], linewidth=1.8, zorder=3,
            )
    axes.set_xlabel(
        "supervised tokens over 256 training rows (log2)",
        color=INK_SECONDARY, fontsize=9,
    )
    axes.set_ylabel(
        "R*(0.90), bits per adapter value", color=INK_SECONDARY, fontsize=9
    )
    axes.set_title(
        "Within a receiver: more tokens, larger adapter",
        color=INK, fontsize=11.5, loc="left", pad=10,
    )
    axes.legend(
        frameon=False, fontsize=8, loc="lower right", labelcolor=INK_SECONDARY,
        handletextpad=0.4, borderpad=0.2,
    )

    pairs = figure.add_subplot(grid[0, 1])
    _style(pairs)
    by_corpus: dict[str, list[dict[str, Any]]] = {}
    for row in usable:
        by_corpus.setdefault(str(row["dataset_key"]), []).append(row)
    deltas = []
    for group in by_corpus.values():
        for left, right in itertools.combinations(group, 2):
            if float(left["train_response_tokens"]) < float(
                right["train_response_tokens"]
            ):
                left, right = right, left
            deltas.append(
                (
                    math.log2(
                        float(left["train_response_tokens"])
                        / float(right["train_response_tokens"])
                    ),
                    float(left["r_star_bits_per_value"])
                    - float(right["r_star_bits_per_value"]),
                )
            )
    below = sum(1 for _, value in deltas if value < 0)
    pairs.axhline(0.0, color=INK, linewidth=1.4, zorder=2)
    pairs.scatter(
        [x for x, _ in deltas], [y for _, y in deltas], s=30,
        facecolor=SERIES[0], edgecolor=SURFACE, linewidth=0.9, zorder=4,
    )
    # Two bands, not a fitted line. Pairs whose tokenizers agree to within a
    # tenth are the built-in null -- there is no receiver difference to detect
    # and the sign should be near chance. Pairs that really differ are the
    # test. Band captions live in the corner block, because a label wide
    # enough to read is wider than the band it names.
    edges = [0.0, 0.15, max(x for x, _ in deltas) + 1e-6]
    captions = ("tokenizers within 11%", "tokenizers differing by more")
    lines = []
    for (low, high), caption in zip(zip(edges, edges[1:]), captions):
        inside = [y for x, y in deltas if low <= x < high]
        if not inside:
            continue
        mean = sum(inside) / len(inside)
        share = sum(1 for value in inside if value < 0) / len(inside)
        pairs.plot(
            [low, high], [mean, mean], color=SERIES[1], linewidth=2.6, zorder=5,
        )
        lines.append(f"{caption}: {share:.0%} below zero (n={len(inside)})")
    spread = max(abs(y) for _, y in deltas)
    pairs.set_ylim(-spread * 1.15, spread * 1.6)
    pairs.set_xlabel(
        "extra tokens the first receiver spends on the same text (log2 ratio)",
        color=INK_SECONDARY, fontsize=9,
    )
    pairs.set_ylabel(
        "difference in R*(0.90)", color=INK_SECONDARY, fontsize=9
    )
    pairs.set_title(
        "Between receivers: more tokens, smaller adapter",
        color=INK, fontsize=11.5, loc="left", pad=10,
    )
    pairs.text(
        0.018, 0.975,
        f"{below} of {len(deltas)} same-corpus receiver pairs fall",
        transform=pairs.transAxes, ha="left", va="top", fontsize=9, color=INK,
    )
    pairs.text(
        0.018, 0.895,
        "band means:\n" + "\n".join(f"  {line}" for line in lines),
        transform=pairs.transAxes, ha="left", va="top",
        fontsize=8.5, color=SERIES[1],
    )
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return Path(path)


# Named on the axis figure: the incumbent, the two controls that need no
# model, and the strongest measurement on each axis. Everything else stays an
# unlabelled point -- a name on all 46 is unreadable.
_LABELLED = {
    "correction_channel_bits": ("correction channel rate", (-10, 6)),
    "dataset_fisher_log_volume": ("Fisher log-volume", (10, -2)),
    "fisher_logdet": ("Fisher log-det", (10, -10)),
    "base_head_hidden_alignment": ("head-subspace alignment", (10, -3)),
    "base_codelength_bits_per_token": ("base code length", (10, 4)),
    "tokenizer_fertility_relative": ("tokens per character", (-10, 6)),
    # The two text statistics land on the same spot, so one label names both.
    "text_cross_row_redundancy": (
        "text redundancy and supervised token count", (-10, 10)
    ),
}
# Anything measured from the text alone returns the same number on every
# receiver. Its cross-receiver correlation is a tie-break over a column of
# zeros, so it is drawn at zero, where it belongs.
_TEXT_ONLY = {"text_cross_row_redundancy", "train_response_tokens"}


def scored_on_the_wrong_axis(
    gate: list[dict[str, Any]],
    cross: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    path: Path,
) -> Path:
    """Where each candidate carries signal, and what that was worth.

    Left: the axis the discovery gate scored on against the axis the claim is
    about. A bar chart of either axis alone hides that they disagree. Right:
    the price on a receiver the fit has never seen, which is the only number a
    reader outside the project has to care about.
    """
    within = {
        str(row["candidate"]): abs(float(row["within_model_spearman"]))
        for row in gate
        if row.get("within_model_spearman") is not None
        and math.isfinite(float(row["within_model_spearman"]))
    }
    across = {
        str(row["candidate"]): abs(float(row["cross_receiver_spearman"]))
        for row in cross
        if row.get("cross_receiver_spearman") is not None
        and math.isfinite(float(row["cross_receiver_spearman"]))
    }
    for name in _TEXT_ONLY:
        if name in across:
            across[name] = 0.0
    shared = sorted(set(within) & set(across))
    plt, figure = _figure(11.6, 5.2)
    grid = figure.add_gridspec(1, 2, width_ratios=(1.5, 1.0), wspace=0.42)
    axes = figure.add_subplot(grid[0, 0])
    _style(axes)
    axes.axhspan(0, 0.30, color=GRID, alpha=0.5, zorder=0)
    for name in shared:
        named = name in _LABELLED
        axes.scatter(
            within[name], across[name],
            s=48 if named else 20,
            facecolor=SERIES[1] if name in _TEXT_ONLY else SERIES[0],
            edgecolor=SURFACE, linewidth=0.9,
            alpha=0.95 if named else 0.4,
            zorder=4 if named else 3,
        )
    for name, (label, (dx, dy)) in _LABELLED.items():
        if name not in within or name not in across:
            continue
        axes.annotate(
            label, (within[name], across[name]),
            textcoords="offset points", xytext=(dx, dy),
            fontsize=8, color=INK_SECONDARY,
            ha="right" if dx < 0 else "left",
        )
    axes.set_xlim(0, 1.02)
    axes.set_ylim(-0.10, 0.78)
    axes.set_xlabel(
        "|rho| within a receiver  --  the axis the gate scored",
        color=INK_SECONDARY, fontsize=9,
    )
    axes.set_ylabel(
        "|rho| across receivers on one corpus  --  the axis the claim is about",
        color=INK_SECONDARY, fontsize=9,
    )
    axes.set_title(
        "Candidates were ranked on the axis a text statistic can win",
        color=INK, fontsize=11.5, loc="left", pad=10,
    )
    axes.text(
        0.985, 0.025,
        f"{len(shared)} candidates - 67 arms - 55 receiver pairs\n"
        "shaded: no measurable receiver dependence",
        transform=axes.transAxes, ha="right", va="bottom",
        fontsize=8.5, color=INK_MUTED, linespacing=1.5,
    )
    axes.scatter(
        [], [], s=48, facecolor=SERIES[0], edgecolor=SURFACE,
        label="measured against the frozen model",
    )
    axes.scatter(
        [], [], s=48, facecolor=SERIES[1], edgecolor=SURFACE,
        label="measured from the text alone (identical on every receiver)",
    )
    axes.legend(
        frameon=False, fontsize=8, loc="upper left", labelcolor=INK_SECONDARY,
        handletextpad=0.4, borderpad=0.2,
    )

    bars = figure.add_subplot(grid[0, 1])
    _style(bars)
    bars.grid(axis="y", visible=False)
    rows = sorted(comparison, key=lambda row: -float(row["receiver_held_out_rmse"]))
    labels = [
        str(row["model"]).replace("correction_channel_bits", "channel rate")
        for row in rows
    ]
    values = [float(row["receiver_held_out_rmse"]) for row in rows]
    best = min(values)
    colours = [
        SERIES[1] if value == best else (AXIS if "receiver mean" in name else SERIES[0])
        for name, value in zip(labels, values)
    ]
    positions = list(range(len(rows)))
    bars.barh(positions, values, height=0.55, color=colours, zorder=3)
    bars.set_yticks(positions)
    bars.set_yticklabels(labels, fontsize=8.5)
    for index, value in zip(positions, values):
        bars.text(
            value + max(values) * 0.02, index, f"{value:.3f}",
            va="center", fontsize=8.5, color=INK_SECONDARY,
        )
    bars.set_xlim(0, max(values) * 1.22)
    bars.set_xlabel(
        "error on a receiver the fit never saw (bits per value)",
        color=INK_SECONDARY, fontsize=9,
    )
    bars.set_title(
        "and two zlib calls won", color=INK, fontsize=11.5, loc="left", pad=10,
    )
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)
    return Path(path)


def write_audit_figures(result: dict[str, Any], out_dir: Path) -> list[Path]:
    """All three figures of the rate-law audit, from one analysis result."""
    out_dir = Path(out_dir)
    return [
        adapter_against_information(result["arms"], out_dir / "adapter_vs_information.png"),
        token_count_reversal(result["arms"], out_dir / "token_count_reversal.png"),
        scored_on_the_wrong_axis(
            result["candidate_gate"],
            result["cross_receiver"],
            result["model_comparison"],
            out_dir / "scored_on_the_wrong_axis.png",
        ),
    ]
