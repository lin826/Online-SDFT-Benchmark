"""Aggregation, qualitative selection, and plotting.

This module reads completed chronological traces. It never participates in an
action or update, which keeps evaluation-only gold fields out of methods.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .config import ACTIONS, METHODS, PHASE_LENGTH


def mean_ci(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return {"mean": float(array[0]), "std": 0.0, "ci95": 0.0}
    std = float(array.std(ddof=1))
    return {
        "mean": float(array.mean()),
        "std": std,
        "ci95": float(1.96 * std / math.sqrt(len(array))),
    }


def summarize_metrics(metrics: list[dict]) -> dict:
    metric_names = [
        key
        for key in metrics[0]
        if key not in {"seed", "method"}
    ]
    return {
        method: {
            metric: mean_ci(
                [
                    float(row[metric])
                    for row in metrics
                    if row["method"] == method
                ]
            )
            for metric in metric_names
        }
        for method in METHODS
    }


def find_qualitative_examples(
    rollouts: list[dict],
    limit: int = 8,
) -> list[dict]:
    """Select later steps uniquely solved by Online-SDFT."""
    def public_feedback(row: dict) -> dict:
        feedback = dict(row["feedback"])
        if "match_status" in feedback:
            feedback["observed_surface_match_status"] = feedback.pop(
                "match_status"
            )
        return feedback

    grouped = defaultdict(dict)
    for row in rollouts:
        grouped[(row["seed"], row["t"])][row["method"]] = row

    qualitative = []
    for (seed, step), rows in sorted(grouped.items()):
        if (
            step > PHASE_LENGTH
            and len(rows) == len(METHODS)
            and rows["Online-SDFT"]["correct_online"] == 1
            and sum(
                rows[method]["correct_online"]
                for method in METHODS[:-1]
            )
            == 0
        ):
            qualitative.append(
                {
                    "seed": seed,
                    "t": step,
                    "regime": rows["Online-SDFT"]["regime"],
                    "category": rows["Online-SDFT"]["category"],
                    "notification_title": rows["Online-SDFT"][
                        "notification_title"
                    ],
                    "notification_body": rows["Online-SDFT"][
                        "notification_body"
                    ],
                    "gold_action_scoring_only": rows["Online-SDFT"][
                        "gold_action_scoring_only"
                    ],
                    "methods": {
                        method: {
                            "action": rows[method]["action"],
                            "feedback": public_feedback(rows[method]),
                            "teacher_rollout": rows[method][
                                "teacher_rollout"
                            ],
                            "sdft_evidence_reliability": rows[method].get(
                                "sdft_evidence_reliability"
                            ),
                            "sdft_fusion_weights": rows[method].get(
                                "sdft_fusion_weights"
                            ),
                            "rft_candidate_action": rows[method].get(
                                "rft_candidate_action"
                            ),
                            "rft_accepted": rows[method].get("rft_accepted"),
                            "rft_reason": rows[method].get("rft_reason"),
                        }
                        for method in METHODS
                    },
                }
            )
        if len(qualitative) >= limit:
            break
    return qualitative


def _rft_diagnostic_counts(rows: list[dict]) -> dict:
    """Aggregate auditable RFT filtering and optimizer counts."""
    allowed_reasons = {
        None,
        "accepted",
        "teacher_mismatch",
        "ambiguous_unverified",
        "censored_unknown",
    }
    observed_reasons = {row.get("rft_reason") for row in rows}
    if not observed_reasons <= allowed_reasons:
        raise ValueError("unknown RFT audit reason")
    if any(
        row.get("rft_accepted") is True
        and row.get("rft_reason") != "accepted"
        for row in rows
    ):
        raise ValueError("accepted RFT rows must use the accepted reason")
    attempted = sum(row.get("rft_candidate_action") is not None for row in rows)
    accepted = sum(row.get("rft_accepted") is True for row in rows)
    rejected = sum(row.get("rft_accepted") is False for row in rows)
    rejection_reasons = Counter(
        row["rft_reason"]
        for row in rows
        if row.get("rft_accepted") is False
        and row.get("rft_reason") is not None
    )
    if attempted != accepted + rejected:
        raise ValueError("RFT attempted rows must partition into accepted/rejected")
    if rejected != sum(rejection_reasons.values()):
        raise ValueError("RFT rejection reasons must account for every rejection")
    proposal_actions = [
        row["rft_candidate_action"]
        for row in rows
        if row.get("rft_candidate_action") is not None
    ]
    accepted_actions = [
        row["rft_candidate_action"]
        for row in rows
        if row.get("rft_accepted") is True
    ]
    if any(action not in ACTIONS for action in proposal_actions):
        raise ValueError("RFT proposal counts contain an unknown route")
    if any(action not in ACTIONS for action in accepted_actions):
        raise ValueError("RFT accepted counts contain an unknown route")
    proposal_counts = {
        action: proposal_actions.count(action)
        for action in ACTIONS
    }
    accepted_counts = {
        action: accepted_actions.count(action)
        for action in ACTIONS
    }
    if sum(proposal_counts.values()) != attempted:
        raise ValueError("RFT proposal route counts must account for every attempt")
    if sum(accepted_counts.values()) != accepted:
        raise ValueError("RFT accepted route counts must account for every acceptance")
    entropies = []
    for row in rows:
        candidate = row.get("rft_candidate_action")
        entropy = row.get("rft_candidate_entropy")
        if candidate is None:
            if entropy is not None:
                raise ValueError("RFT proposal entropy requires a sampled route")
            continue
        if entropy is None or isinstance(entropy, bool):
            raise ValueError("RFT sampled routes require proposal entropy")
        value = float(entropy)
        if (
            not math.isfinite(value)
            or not 0.0 <= value <= math.log(len(ACTIONS)) + 1e-12
        ):
            raise ValueError(
                "RFT proposal entropy is outside the categorical range"
            )
        entropies.append(value)
    if len(entropies) != attempted:
        raise ValueError("RFT proposal entropy must account for every attempt")
    update_count = max(
        (int(row.get("rft_update_index") or 0) for row in rows),
        default=0,
    )
    return {
        "attempted": attempted,
        "accepted": accepted,
        "rejected": rejected,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "acceptance_rate": accepted / attempted if attempted else None,
        "proposal_counts": proposal_counts,
        "accepted_counts": accepted_counts,
        "mean_proposal_entropy": (
            float(np.mean(entropies)) if entropies else None
        ),
        "update_count": update_count,
        "censored_unknown": sum(
            row.get("rft_reason") == "censored_unknown"
            for row in rows
        ),
        "pending_after_horizon": sum(
            row.get("lesson_status") == "pending_after_horizon"
            for row in rows
        ),
    }


def summarize_rft_diagnostics(rollouts: list[dict]) -> dict:
    """Return per-seed and total RFT rejection diagnostics."""
    rft_rows = [row for row in rollouts if row.get("method") == "RFT"]
    by_seed = defaultdict(list)
    for row in rft_rows:
        by_seed[int(row["seed"])].append(row)
    per_seed = {
        str(seed): _rft_diagnostic_counts(rows)
        for seed, rows in sorted(by_seed.items())
    }
    total = _rft_diagnostic_counts(rft_rows)
    total["update_count"] = sum(
        counts["update_count"] for counts in per_seed.values()
    )
    return {"per_seed": per_seed, "total": total}


def write_compact_results(
    output_dir: Path,
    config: dict,
    metrics: list[dict],
    rollouts: list[dict],
) -> dict:
    summary = summarize_metrics(metrics)
    qualitative = find_qualitative_examples(rollouts)
    (output_dir / "qualitative_examples.json").write_text(
        json.dumps(qualitative, indent=2) + "\n"
    )
    payload = {
        "config": config,
        "summary": summary,
        "qualitative_examples": len(qualitative),
        "rft_diagnostics": summarize_rft_diagnostics(rollouts),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    return summary


# Frozen baselines are muted so Online-SDFT carries the visual emphasis.
METHOD_COLORS = {
    "Base": "#d2d3d6",
    "ICL": "#b7bac0",
    "RAG": "#969ba4",
    "REINFORCE": "#d97706",
    "RFT": "#75639a",
    "Online-SDFT": "#4a7a3e",
}
# The three frozen baselines land on nearly the same numbers, so they also get
# distinct dash patterns to stay readable where the curves overlap.
METHOD_DASHES = {
    "Base": (0, ()),
    "ICL": (0, (5, 2)),
    "RAG": (0, (1.6, 1.8)),
    "REINFORCE": (0, ()),
    "RFT": (0, (4, 1.7)),
    "Online-SDFT": (0, ()),
}
INK = "#1a1a1a"
MUTED = "#777777"
GRID = "#e0e0e0"
FIELD = "#f6f5f8"
INTER_FONT = Path(__file__).resolve().parents[1] / "figures/fonts/inter/Inter-Variable.ttf"
INTER_STATIC_FONTS = tuple(
    INTER_FONT.parent / name
    for name in ("Inter-Medium.ttf", "Inter-SemiBold.ttf", "Inter-Bold.ttf")
)
PHASE_LABELS = ("weekday", "on-call", "off-hours")


def _apply_plot_style(matplotlib) -> None:
    if INTER_FONT.exists():
        from matplotlib import font_manager

        for font_path in (INTER_FONT, *INTER_STATIC_FONTS):
            if font_path.exists():
                font_manager.fontManager.addfont(font_path)
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Helvetica Neue",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "cm",
            "font.size": 9,
            "figure.facecolor": FIELD,
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.9,
            "axes.labelcolor": MUTED,
            "axes.labelsize": 8.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": 600,
            "axes.titlecolor": INK,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "savefig.facecolor": FIELD,
        }
    )


def _draw_phase_bands(axis, upper: float) -> None:
    """Shade the three preference regimes instead of drawing bare cut lines."""
    for index in range(3):
        start = index * PHASE_LENGTH
        end = start + PHASE_LENGTH
        if index % 2 == 1:
            axis.axvspan(start, end, color="#f4f5f7", zorder=0)
        axis.text(
            start + PHASE_LENGTH / 2,
            upper,
            PHASE_LABELS[index],
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=MUTED,
        )


def write_figures(
    summary: dict,
    curves: list[dict],
    figure_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    _apply_plot_style(matplotlib)
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    # Historical artifacts may predate a newly added method. Fresh benchmark
    # runs still contain the complete registry, while figures-only remains
    # able to render an explicitly labeled historical result set.
    ordered = [method for method in METHODS if method in summary]
    positions = np.arange(len(ordered))

    figure, axes = plt.subplots(1, 2, figsize=(8.1, 3.45), sharey=True)
    figure.text(
        0.5, 0.985, "ONLINE PERSONALIZATION RESULTS",
        ha="center", va="top", fontsize=7, color=MUTED, fontweight=600,
    )
    figure.suptitle(
        "Performance over the full interaction stream",
        y=0.925, fontsize=13.5, color=INK, fontweight=600,
    )
    panels = (
        (
            axes[0],
            [100 * summary[m]["online_accuracy"]["mean"] for m in ordered],
            [100 * summary[m]["online_accuracy"]["ci95"] for m in ordered],
            "Online accuracy",
            "prequential % matching sampled preference  (higher is better)",
            "{:.1f}",
        ),
        (
            axes[1],
            [summary[m]["cum_regret"]["mean"] for m in ordered],
            [summary[m]["cum_regret"]["ci95"] for m in ordered],
            "Cumulative utility gap",
            "Σ utility gap to utility-optimal route  (lower is better)",
            "{:.1f}",
        ),
    )
    for axis, values, errors, title, subtitle, fmt in panels:
        axis.barh(
            positions,
            values,
            xerr=errors,
            height=0.6,
            color=[METHOD_COLORS[m] for m in ordered],
            error_kw={
                "ecolor": MUTED,
                "elinewidth": 0.9,
                "capsize": 2.5,
                "capthick": 0.9,
            },
            zorder=3,
        )
        headroom = max(v + e for v, e in zip(values, errors)) * 1.16
        for position, value, error in zip(positions, values, errors):
            axis.text(
                value + error + headroom * 0.02,
                position,
                fmt.format(value),
                va="center",
                ha="left",
                fontsize=8,
                color=INK,
            )
        axis.set_xlim(0, headroom)
        axis.set_title(title, loc="left", pad=10, fontweight=600)
        axis.set_xlabel(subtitle, labelpad=7)
        axis.grid(axis="x", zorder=0)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
        axis.margins(y=0.08)

    axes[0].set_yticks(positions, ordered)
    for label, method in zip(axes[0].get_yticklabels(), ordered):
        label.set_color(
            METHOD_COLORS[method]
            if method in {"REINFORCE", "RFT", "Online-SDFT"}
            else MUTED
        )
        label.set_fontweight(600 if method == "Online-SDFT" else 400)
    axes[0].invert_yaxis()
    figure.tight_layout(pad=0.7, rect=(0, 0, 1, 0.84))
    figure.subplots_adjust(wspace=0.14)
    figure.savefig(
        figure_dir / "bandit_accuracy.png",
        dpi=220,
        bbox_inches="tight", facecolor=FIELD,
    )
    plt.close(figure)

    accuracy_by_step = defaultdict(list)
    regret_by_step = defaultdict(list)
    for row in curves:
        key = (row["method"], int(row["t"]))
        accuracy_by_step[key].append(float(row["cum_accuracy"]))
        regret_by_step[key].append(float(row["cum_regret"]))

    figure, axes = plt.subplots(1, 2, figsize=(7.7, 3.1))
    handles = []
    for method in ordered:
        steps = sorted(step for name, step in accuracy_by_step if name == method)
        samples = [np.asarray(accuracy_by_step[(method, s)]) for s in steps]
        mean = np.array([s.mean() for s in samples]) * 100
        ci = (
            np.array(
                [
                    0.0
                    if len(s) == 1
                    else 1.96 * s.std(ddof=1) / math.sqrt(len(s))
                    for s in samples
                ]
            )
            * 100
        )
        emphasis = method == "Online-SDFT"
        (line,) = axes[0].plot(
            steps,
            mean,
            color=METHOD_COLORS[method],
            label=method,
            lw=2.2 if emphasis else 1.3,
            linestyle=METHOD_DASHES[method],
            zorder=4 if emphasis else 3,
        )
        handles.append(line)
        axes[0].fill_between(
            steps,
            mean - ci,
            mean + ci,
            color=METHOD_COLORS[method],
            alpha=0.14 if emphasis else 0.07,
            lw=0,
            zorder=2,
        )

        regret_steps = sorted(
            step for name, step in regret_by_step if name == method
        )
        regret_mean = [
            float(np.mean(regret_by_step[(method, s)])) for s in regret_steps
        ]
        axes[1].plot(
            regret_steps,
            regret_mean,
            color=METHOD_COLORS[method],
            lw=2.2 if emphasis else 1.3,
            linestyle=METHOD_DASHES[method],
            zorder=4 if emphasis else 3,
        )

    axes[0].set(
        xlim=(0, 3 * PHASE_LENGTH),
        ylim=(0, 100),
        xlabel="online decisions",
        ylabel="cumulative accuracy (%)",
    )
    axes[0].set_title("Prequential accuracy so far", loc="left", pad=18, fontweight="medium")
    _draw_phase_bands(axes[0], 101)

    axes[1].set(xlim=(0, 3 * PHASE_LENGTH), xlabel="online decisions")
    axes[1].set_ylabel("cumulative utility regret")
    axes[1].set_title(
        "Utility-optimal gap so far",
        loc="left",
        pad=18,
        fontweight="medium",
    )
    _draw_phase_bands(axes[1], axes[1].get_ylim()[1])

    for axis in axes:
        axis.grid(axis="y", zorder=1)
        axis.set_axisbelow(True)

    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=len(ordered),
        bbox_to_anchor=(0.5, -0.03),
        handlelength=1.5,
        columnspacing=1.6,
        handletextpad=0.5,
    )
    figure.tight_layout(pad=0.6, rect=(0, 0.05, 1, 1))
    figure.subplots_adjust(wspace=0.22)
    figure.savefig(
        figure_dir / "bandit_learning_curves.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def replot_from_outputs(output_dir: Path, figure_dir: Path) -> None:
    """Redraw the published figures from stored artifacts, without the model."""
    summary = json.loads(
        (output_dir / "summary.json").read_text()
    )["summary"]
    with (output_dir / "learning_curves.csv").open() as handle:
        curves = list(csv.DictReader(handle))
    write_figures(summary, curves, figure_dir)
