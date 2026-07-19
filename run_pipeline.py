"""Run the pipeline end to end, in dependency order.

    python run_pipeline.py                  # everything
    python run_pipeline.py --skip-ml        # measurement + figures only
    python run_pipeline.py --from process   # re-run from the Spark core onward

Stage timings are printed at the end as the brief's Algorithmic Efficiency
evidence.

Thermal note: `ml` is the only heavy stage. `config/settings.yaml` sets
`spark.master: local[4]` rather than local[*] because an all-core run pinned the
development laptop for 44 minutes. Raise it on a machine with real cooling.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time

LOGGER = logging.getLogger("pipeline")

STEPS = [
    ("ingest", "src.ingest.download_counts", "Download DfT counts (skipped if present)"),
    ("process", "src.process.clean_counts", "Clean 5.3M hourly counts -> Parquet"),
    ("process", "src.process.build_profiles", "Aggregate per-link daily profiles"),
    ("db", "src.db.load_db", "Load SQLite warehouse"),
    ("ml", "src.ml.train_models", "Train and compare LR / RF / GBT"),
    ("viz", "src.viz.eda_figures", "EDA figures"),
    ("viz", "src.viz.ml_figures", "Model evaluation figures"),
]

GROUP_ORDER = ["ingest", "process", "db", "ml", "viz"]


def run(module: str) -> tuple[bool, float]:
    started = time.perf_counter()
    result = subprocess.run([sys.executable, "-m", module])
    return result.returncode == 0, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start_group", choices=GROUP_ORDER, default="ingest")
    parser.add_argument("--skip-ml", action="store_true",
                        help="Skip training - the only heavy stage.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stdout
    )

    start_index = GROUP_ORDER.index(args.start_group)
    timings: list[tuple[str, float, bool]] = []

    for group, module, description in STEPS:
        if GROUP_ORDER.index(group) < start_index:
            continue
        if group == "ml" and args.skip_ml:
            LOGGER.info("SKIP  %s", description)
            continue

        LOGGER.info("=" * 70)
        LOGGER.info("RUN   %s  (%s)", description, module)
        ok, seconds = run(module)
        timings.append((description, seconds, ok))
        if not ok:
            LOGGER.error("%s FAILED - stopping.", description)
            return 1
        LOGGER.info("OK    %s in %.1fs", description, seconds)

    LOGGER.info("=" * 70)
    LOGGER.info("Stage timings (Algorithmic Efficiency evidence):")
    for description, seconds, ok in timings:
        LOGGER.info("  %-48s %8.1fs  %s", description, seconds, "ok" if ok else "FAILED")
    LOGGER.info("Total: %.1fs", sum(t[1] for t in timings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
