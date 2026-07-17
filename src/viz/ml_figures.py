"""Model-evaluation figures for the regression task.

Drawn from the per-row (actual, predicted) samples that train_models.py persists,
not reconstructed from summary metrics - a diagnostic interpolated between an
RMSE and an R^2 would be an illustration, not a result.

Run (after training):
    python -m src.viz.ml_figures
"""

from __future__ import annotations

import json
import logging
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import resolve

LOGGER = logging.getLogger("ml_figures")

FIGURE_DPI = 150
PALETTE = ["#2b5f8a", "#c1553b", "#4a8c5f", "#8a6fb0"]
GREY, RED = "#888", "#c1553b"


def _save(fig, name: str) -> None:
    target = resolve("outputs") / name
    fig.tight_layout()
    fig.savefig(target, dpi=FIGURE_DPI)
    plt.close(fig)
    LOGGER.info("Wrote %s", target)


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def load_results() -> dict:
    with open(resolve("outputs") / "model_results.json") as handle:
        return json.load(handle)


def load_predictions(split_tag: str, model_name: str) -> pd.DataFrame | None:
    path = resolve("outputs") / "predictions" / f"{split_tag}__{_slug(model_name)}.parquet"
    if not path.exists():
        LOGGER.warning("No predictions for %s / %s", split_tag, model_name)
        return None
    return pd.read_parquet(path)


def get_split(payload: dict, tag: str) -> dict | None:
    for block in payload["splits"]:
        if block["split"] == tag:
            return block
    return None


def plot_profile_actual_vs_predicted(split_tag: str = "primary") -> None:
    """THE key figure: does the model reproduce the shape of the day?

    R^2 says the models add almost nothing over the link-history baseline. This
    figure says what they *do* learn: the bimodal commuter curve, at roughly the
    right amplitude, biased low in level. Both statements are true and the report
    needs both.
    """
    payload = load_results()
    block = get_split(payload, split_tag)
    if not block:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5))
    actual_drawn = False
    for index, result in enumerate(block["results"]):
        predictions = load_predictions(split_tag, result["model"])
        if predictions is None:
            continue
        grouped = predictions.groupby("hour").agg(
            actual=("actual", "mean"), predicted=("prediction", "mean")
        )
        if not actual_drawn:
            ax.plot(grouped.index, grouped["actual"], marker="o", color="#222",
                    linewidth=2.4, label="Actual", zorder=5)
            actual_drawn = True
        ax.plot(grouped.index, grouped["predicted"], marker="s", markersize=4,
                linestyle="--", color=PALETTE[index % len(PALETTE)],
                label=f"{result['model']}")

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean motor vehicles per hour")
    # Title states what the lines actually show, not what the design hoped.
    # Only GBT tracks the curve; the linear model is close to flat, which is the
    # same story its 0.14% skill tells. Every model sits below the actual.
    ax.set_title("Predicted vs Actual Daily Profile (held-out future years)\n"
                 "only GBT tracks the shape; the linear model is nearly flat; "
                 "all are biased low")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, "ml_profile_actual_vs_predicted.png")


def plot_predicted_vs_actual(split_tag: str = "primary",
                             model_name: str = "Gradient-Boosted Trees") -> None:
    predictions = load_predictions(split_tag, model_name)
    if predictions is None:
        return

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    limit = float(np.percentile(predictions["actual"], 99.5))
    hexbin = ax.hexbin(predictions["actual"], predictions["prediction"],
                       gridsize=45, bins="log", cmap="Blues",
                       extent=(0, limit, 0, limit))
    ax.plot([0, limit], [0, limit], "--", color=RED, linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Actual flow (veh/h)")
    ax.set_ylabel("Predicted flow (veh/h)")
    ax.set_title(f"Predicted vs Actual - {model_name}\n"
                 "(axes clipped at the 99.5th percentile; log colour scale)")
    ax.legend(fontsize=8)
    fig.colorbar(hexbin, ax=ax, label="Observations (log)")
    _save(fig, "ml_predicted_vs_actual.png")


def plot_residuals_by_hour(split_tag: str = "primary",
                           model_name: str = "Gradient-Boosted Trees") -> None:
    """Residual by hour - exposes systematic bias that RMSE averages away."""
    predictions = load_predictions(split_tag, model_name)
    if predictions is None:
        return

    predictions = predictions.assign(residual=predictions["prediction"] - predictions["actual"])
    grouped = predictions.groupby("hour")["residual"].agg(["mean", "std", "count"])

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(grouped.index, grouped["mean"], color=["#c1553b" if v < 0 else "#4a8c5f"
                                                  for v in grouped["mean"]])
    ax.axhline(0, color="#222", linewidth=1)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean residual (predicted - actual, veh/h)")
    ax.set_title(f"Residuals by Hour - {model_name}\n"
                 "every hour below zero: the model systematically under-predicts")
    ax.set_xticks(grouped.index)
    ax.grid(alpha=0.25, axis="y")
    _save(fig, "ml_residuals_by_hour.png")


def plot_model_comparison() -> None:
    payload = load_results()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for axis, (split_tag, title) in zip(
        axes[:2], [("primary", "Primary (train ≤2019 → 2023-25, spans COVID)"),
                   ("control", "Control (train ≤2016 → 2017-19, no break)")]
    ):
        block = get_split(payload, split_tag)
        if not block:
            continue
        names = [r["model"] for r in block["results"]]
        r2 = [r["r2"] for r in block["results"]]
        positions = range(len(names))

        axis.bar(list(positions), r2, color="#2b5f8a", label="Model R²")
        # Both baselines. Without the link-history line an R² of 0.86 reads as a
        # triumph; with it, the gap is the whole story.
        axis.axhline(block["baselines"]["link_history"]["r2"], linestyle="--", color=RED,
                     label=f"Baseline: link's own history "
                           f"({block['baselines']['link_history']['r2']:.3f})")
        axis.axhline(max(block["baselines"]["global_mean"]["r2"], 0), linestyle=":", color=GREY,
                     label="Baseline: global mean (≈0)")
        axis.set_xticks(list(positions))
        axis.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
        axis.set_ylim(0, 1)
        axis.set_ylabel("R² on held-out years")
        axis.set_title(title, fontsize=9)
        axis.legend(fontsize=7)

    # Model Efficiency: R² per training second.
    block = get_split(payload, "primary")
    if block:
        names = [r["model"] for r in block["results"]]
        efficiency = [r.get("r2_per_train_second") or 0 for r in block["results"]]
        axes[2].bar(range(len(names)), efficiency, color="#7aa6c2")
        axes[2].set_xticks(range(len(names)))
        axes[2].set_xticklabels(names, rotation=20, ha="right", fontsize=7)
        axes[2].set_ylabel("R² per training second")
        axes[2].set_title("Model Efficiency\n(the brief's metric)", fontsize=9)

    fig.suptitle("Model Comparison - R² is dominated by the link-history baseline", fontsize=11)
    target = resolve("outputs") / "ml_model_comparison.png"
    fig.savefig(target, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", target)


def plot_skill_decomposition() -> None:
    """The honest headline: how much does the ML add over knowing the road?"""
    payload = load_results()
    fig, (ax_stack, ax_share) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    block = get_split(payload, "primary")
    if not block:
        return
    best = max(block["results"], key=lambda r: r["r2"])
    reference = block["baselines"]["link_history"]["r2"]

    # Stacked: what R^2 = 0.86 is actually made of.
    ax_stack.bar(["Best model's R²"], [reference], color="#7aa6c2",
                 label=f"Explained by knowing WHICH ROAD ({reference:.3f})")
    ax_stack.bar(["Best model's R²"], [best["r2"] - reference], bottom=[reference],
                 color="#2b5f8a", label=f"Added by the model ({best['r2'] - reference:+.4f})")
    ax_stack.bar(["Best model's R²"], [1 - best["r2"]], bottom=[best["r2"]],
                 color="#e5e5e5", label=f"Unexplained ({1 - best['r2']:.3f})")
    ax_stack.set_ylim(0, 1)
    ax_stack.set_ylabel("Share of variance in hourly flow")
    ax_stack.set_title(f"What {best['model']}'s R² = {best['r2']:.3f} Is Made Of")
    ax_stack.legend(fontsize=8, loc="lower right")

    # Share of the baseline's residual variance each model explains.
    #
    # The splits do NOT contain the same models - CV-tuned runs on primary only -
    # so the x axis is built from the union, keyed by model name. Indexing by
    # position instead would silently leave the CV bar unlabelled and,
    # worse, could align a primary bar against a different control model.
    all_names: list[str] = []
    for block in payload["splits"]:
        for result in block["results"]:
            if result["model"] not in all_names:
                all_names.append(result["model"])

    for index, (split_tag, offset) in enumerate([("primary", -0.2), ("control", 0.2)]):
        current = get_split(payload, split_tag)
        if not current:
            continue
        by_name = {r["model"]: r for r in current["results"]}
        positions, shares = [], []
        for i, name in enumerate(all_names):
            if name not in by_name:
                continue  # model not run on this split; leave a gap rather than shift
            positions.append(i + offset)
            shares.append(100 * (by_name[name].get("baseline_residual_variance_explained") or 0))
        ax_share.bar(positions, shares, width=0.38, color=PALETTE[index], label=split_tag)

    ax_share.axhline(0, color="#222", linewidth=1)
    ax_share.set_xticks(range(len(all_names)))
    ax_share.set_xticklabels([n.replace(" (CV-tuned, 20% sample)", "\n(CV-tuned, 20% samp.)")
                              for n in all_names], rotation=20, ha="right", fontsize=6.5)
    ax_share.set_ylabel("% of the baseline's residual variance explained")
    ax_share.set_title("Actual Forecasting Skill\n"
                       "below zero = worse than assuming the road behaves as it always has",
                       fontsize=9)
    ax_share.legend(fontsize=8)
    ax_share.grid(alpha=0.25, axis="y")
    _save(fig, "ml_skill_decomposition.png")


def plot_feature_importance() -> None:
    path = resolve("outputs") / "feature_importance.json"
    if not path.exists():
        LOGGER.warning("No feature_importance.json - run training first.")
        return
    with open(path) as handle:
        data = pd.DataFrame(json.load(handle)).head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(data["feature"], data["importance"], color="#2b5f8a")
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title("What Predicts Hourly Flow Before the Hour Is Counted")
    _save(fig, "ml_feature_importance.png")


def plot_lr_coefficients() -> None:
    path = resolve("outputs") / "lr_coefficients.json"
    if not path.exists():
        LOGGER.warning("No lr_coefficients.json - run training first.")
        return
    with open(path) as handle:
        payload = json.load(handle)
    data = pd.DataFrame(payload["coefficients"]).head(15).iloc[::-1]

    colours = ["#4a8c5f" if v > 0 else "#c1553b" for v in data["coefficient"]]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(data["feature"], data["coefficient"], color=colours)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("Linear Regression coefficient (scaled features)")
    ax.set_title("Direction of Effect on Hourly Flow\n"
                 "green = pushes flow up, red = down")
    _save(fig, "ml_lr_coefficients.png")


def plot_split_profile() -> None:
    """The time-aware split: sizes and the distribution shift across it.

    The two splits are NOT comparable on R^2 and this figure is why: their test
    blocks have very different target spread, which the global-mean baseline's
    RMSE measures directly.
    """
    payload = load_results()
    fig, (ax_rows, ax_spread) = plt.subplots(1, 2, figsize=(12, 4.6))

    tags = [b["split"] for b in payload["splits"]]
    train = [b["train_rows"] for b in payload["splits"]]
    test = [b["test_rows"] for b in payload["splits"]]
    positions = range(len(tags))

    ax_rows.bar([p - 0.2 for p in positions], train, width=0.38, color="#2b5f8a", label="Train")
    ax_rows.bar([p + 0.2 for p in positions], test, width=0.38, color="#7aa6c2", label="Test")
    ax_rows.set_xticks(list(positions))
    ax_rows.set_xticklabels([
        f"{b['split']}\ntrain<={b['info']['train_max_year']} -> "
        f"{b['info']['test_min_year']}-{b['info']['test_max_year'] or '25'}"
        for b in payload["splits"]], fontsize=8)
    ax_rows.set_ylabel("Rows")
    ax_rows.set_title("Time-Aware Split Sizes")
    ax_rows.legend(fontsize=8)
    for pos, value in zip(positions, train):
        ax_rows.text(pos - 0.2, value, f"{value/1e6:.2f}M", ha="center", va="bottom", fontsize=7)
    for pos, value in zip(positions, test):
        ax_rows.text(pos + 0.2, value, f"{value/1e3:.0f}k", ha="center", va="bottom", fontsize=7)

    # Target spread per split, via the global-mean baseline's RMSE (= the test
    # block's SD). This is the number that makes cross-split R^2 meaningless.
    spread = [b["baselines"]["global_mean"]["rmse"] for b in payload["splits"]]
    bars = ax_spread.bar(list(positions), spread, color="#c1553b")
    ax_spread.set_xticks(list(positions))
    ax_spread.set_xticklabels(tags)
    ax_spread.set_ylabel("Test-block target SD (veh/h)")
    ax_spread.set_title("Why R² Cannot Be Compared Across Splits\n"
                        "a wider-spread test block inflates R² for the same relative error",
                        fontsize=9)
    for bar, value in zip(bars, spread):
        ax_spread.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}",
                       ha="center", va="bottom", fontsize=9)
    _save(fig, "ml_split_profile.png")


def plot_error_by_road_class() -> None:
    """Where the model's error actually lands - by road class."""
    payload = load_results()
    block = get_split(payload, "primary")
    if not block:
        return
    best = max(block["results"], key=lambda r: r["r2"])
    predictions = load_predictions("primary", best["model"])
    if predictions is None or "road_category" not in predictions.columns:
        LOGGER.warning("No road_category in predictions; skipping.")
        return

    predictions = predictions.assign(
        abs_error=(predictions["prediction"] - predictions["actual"]).abs(),
        residual=predictions["prediction"] - predictions["actual"],
    )
    grouped = predictions.groupby("road_category").agg(
        mae=("abs_error", "mean"), bias=("residual", "mean"),
        actual=("actual", "mean"), n=("actual", "size"),
    ).sort_values("actual", ascending=False)

    fig, (ax_mae, ax_bias) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax_mae.bar(grouped.index, grouped["mae"], color="#2b5f8a", label="MAE")
    ax_mae.plot(grouped.index, grouped["actual"], "o--", color="#c1553b",
                label="Mean actual flow")
    ax_mae.set_ylabel("veh/h")
    ax_mae.set_xlabel("Road category")
    ax_mae.set_title(f"{best['model']}: Error Follows Flow\n"
                     "absolute error is largest where traffic is largest", fontsize=9)
    ax_mae.legend(fontsize=8)

    colours = ["#c1553b" if v < 0 else "#4a8c5f" for v in grouped["bias"]]
    ax_bias.bar(grouped.index, grouped["bias"], color=colours)
    ax_bias.axhline(0, color="#222", linewidth=1)
    ax_bias.set_ylabel("Mean residual (predicted - actual)")
    ax_bias.set_xlabel("Road category")
    ax_bias.set_title("Bias by Road Class\nbelow zero = under-predicting that class",
                      fontsize=9)
    for i, (value, n) in enumerate(zip(grouped["bias"], grouped["n"])):
        ax_bias.text(i, value, f"n={n:,}", ha="center",
                     va="bottom" if value >= 0 else "top", fontsize=7)
    _save(fig, "ml_error_by_road_class.png")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stdout
    )
    if not (resolve("outputs") / "model_results.json").exists():
        LOGGER.error("Train first: python -m src.ml.train_models")
        return 1

    plot_profile_actual_vs_predicted()
    plot_predicted_vs_actual()
    plot_residuals_by_hour()
    plot_model_comparison()
    plot_skill_decomposition()
    plot_feature_importance()
    plot_lr_coefficients()
    plot_split_profile()
    plot_error_by_road_class()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
