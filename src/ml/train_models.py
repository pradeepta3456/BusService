"""Train and compare three MLlib regressors on hourly traffic flow.

Comparison discipline: identical features, identical split, identical target for
all three. Only the learner varies, so a difference in score is attributable to
the model rather than to the data it saw.

  * LinearRegression - interpretable baseline; its coefficients say which
    pre-count signals move flow, and in which direction.
  * RandomForestRegressor - non-linear, gives feature importances.
  * GBTRegressor - sequential boosting; usually strongest, always slowest, which
    is exactly why the brief's Model Efficiency metric (score per training
    second) exists.

Evaluated per the brief's regression requirement: RMSE, MAE, R^2. Two trivial
baselines are computed rather than assumed - predicting the global mean, and
predicting each link's own historical mean. The second is the one that matters:
a model that cannot beat "assume this road behaves as it always has" has found
nothing an urban planner did not already know.

Both the primary split (spans COVID) and the control split (does not) are run, so
"the model cannot forecast" and "the pandemic changed the roads" stay separable.

Run:
    python -m src.ml.train_models
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame, functions as F

from src.config import load_config, resolve
from src.ml.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_WITH_REASON,
    LABEL_COLUMN,
    LEAKY_COLUMNS,
    NUMERIC_FEATURES,
    prepare,
)
from src.spark_session import get_spark

LOGGER = logging.getLogger("train_models")


def assert_no_leakage() -> None:
    """Fail loudly if an outcome component ever reaches the feature list."""
    leaked = set(LEAKY_COLUMNS) & set(ALL_FEATURES)
    if leaked:
        raise RuntimeError(
            "LEAKAGE: target components used as features: "
            + ", ".join(f"{c} ({LEAKY_COLUMNS[c]})" for c in sorted(leaked))
        )


def build_stages(scale: bool) -> list:
    stages: list = []
    for column in CATEGORICAL_FEATURES:
        stages.append(
            StringIndexer(inputCol=column, outputCol=f"{column}_idx", handleInvalid="keep")
        )
        stages.append(
            OneHotEncoder(inputCol=f"{column}_idx", outputCol=f"{column}_ohe", handleInvalid="keep")
        )
    assembler_out = "features_raw" if scale else "features"
    stages.append(
        VectorAssembler(
            inputCols=NUMERIC_FEATURES + [f"{c}_ohe" for c in CATEGORICAL_FEATURES],
            outputCol=assembler_out,
            handleInvalid="keep",
        )
    )
    if scale:
        # Only the linear model needs scaling; trees are scale-invariant.
        stages.append(StandardScaler(inputCol="features_raw", outputCol="features", withMean=False))
    return stages


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def evaluate(name: str, model, test: DataFrame, seconds: float, split_tag: str) -> dict:
    predictions = model.transform(test).cache()

    scores = {}
    for metric in ["rmse", "mae", "r2"]:
        scores[metric] = RegressionEvaluator(
            labelCol=LABEL_COLUMN, predictionCol="prediction", metricName=metric
        ).evaluate(predictions)

    # Persist a sample of (actual, predicted) for diagnostic figures. A
    # sample, not the lot: the test block is ~1M rows and a scatter plot cannot
    # show more than a few thousand points honestly anyway.
    (
        predictions
        .select(F.col(LABEL_COLUMN).alias("actual"), F.col("prediction"), "hour", "road_category")
        .sample(fraction=min(1.0, 20000 / max(predictions.count(), 1)), seed=42)
        .toPandas()
        .to_parquet(resolve("outputs") / "predictions" / f"{split_tag}__{_slug(name)}.parquet",
                    index=False)
    )
    predictions.unpersist()

    return {
        "model": name,
        "split": split_tag,
        "rmse": round(scores["rmse"], 2),
        "mae": round(scores["mae"], 2),
        "r2": round(scores["r2"], 4),
        "train_seconds": round(seconds, 2),
        # Model Efficiency (brief): quality bought per second spent. R^2 is used
        # because RMSE is unbounded, so "RMSE per second" has no sensible sign.
        "r2_per_train_second": round(scores["r2"] / seconds, 5) if seconds > 0 else None,
    }


def trivial_baselines(train: DataFrame, test: DataFrame) -> dict:
    """Two constant/naive predictors, computed rather than asserted.

    global_mean - predict the training mean flow for everything.
    link_history - predict each count point's own training-block mean. This is
        the honest bar: it is what a planner already has without any model.
    """
    mean_flow = train.select(F.avg(LABEL_COLUMN)).collect()[0][0]
    results = {}

    for tag, prediction_col in [
        ("global_mean", F.lit(float(mean_flow))),
        ("link_history", F.col("cp_hist_mean_flow")),
    ]:
        scored = test.withColumn("prediction", prediction_col.cast("double"))
        row = scored.select(
            F.sqrt(F.avg(F.pow(F.col("prediction") - F.col(LABEL_COLUMN), 2))).alias("rmse"),
            F.avg(F.abs(F.col("prediction") - F.col(LABEL_COLUMN))).alias("mae"),
        ).collect()[0]
        # R^2 against the TEST mean, matching MLlib's definition.
        test_mean = test.select(F.avg(LABEL_COLUMN)).collect()[0][0]
        ss = scored.select(
            F.sum(F.pow(F.col(LABEL_COLUMN) - F.col("prediction"), 2)).alias("ss_res"),
            F.sum(F.pow(F.col(LABEL_COLUMN) - F.lit(test_mean), 2)).alias("ss_tot"),
        ).collect()[0]
        r2 = 1 - (ss["ss_res"] / ss["ss_tot"]) if ss["ss_tot"] else float("nan")
        results[tag] = {
            "rmse": round(row["rmse"], 2),
            "mae": round(row["mae"], 2),
            "r2": round(r2, 4),
        }
    return results


def add_skill_decomposition(results: list[dict], baselines: dict) -> None:
    """Attach the share of the link-history baseline's residual variance explained.

    R^2 alone is misleading here. The link-history baseline - "assume this road
    behaves as it always has" - already scores ~0.86, because most of the
    variance in hourly flow is between links (a motorway carries 50x a country
    lane) rather than within a link across the day. A model's R^2 is therefore
    dominated by something a planner already knows for free.

    This quantity is the model's gain expressed against what is actually left to
    explain:

        (r2_model - r2_link_history) / (1 - r2_link_history)

    NAMING MATTERS. It is the share of the LINK-HISTORY BASELINE'S RESIDUAL
    variance explained - not exactly a within-link R^2, because cp_hist_mean_flow
    is the *training* link mean, not the test link mean. Close, but not identical,
    and the report must not claim the stronger thing.
    """
    reference = baselines["link_history"]["r2"]
    remaining = 1.0 - reference
    for result in results:
        gain = result["r2"] - reference
        result["r2_gain_over_link_history"] = round(gain, 5)
        result["baseline_residual_variance_explained"] = (
            round(gain / remaining, 5) if remaining > 0 else None
        )


def run_split(spark, cfg: dict, split_tag: str, **split_years) -> dict:
    seed = cfg["ml"]["seed"]
    LOGGER.info("=" * 70)
    LOGGER.info("SPLIT '%s': %s", split_tag, split_years)

    train, test, info = prepare(spark, **split_years)
    train = train.cache()
    test = test.cache()
    n_train, n_test = train.count(), test.count()
    LOGGER.info("train rows: %s | test rows: %s", f"{n_train:,}", f"{n_test:,}")
    if n_train == 0 or n_test == 0:
        LOGGER.error("Empty block; skipping split '%s'.", split_tag)
        return {}

    baselines = trivial_baselines(train, test)
    for tag, metrics in baselines.items():
        LOGGER.info("baseline '%s': %s", tag, metrics)

    rf_cfg = cfg["models"]["random_forest"]
    gbt_cfg = cfg["models"]["gbt"]
    estimators = [
        ("Linear Regression",
         LinearRegression(featuresCol="features", labelCol=LABEL_COLUMN, maxIter=50), True),
        ("Random Forest",
         RandomForestRegressor(featuresCol="features", labelCol=LABEL_COLUMN,
                               numTrees=rf_cfg["num_trees"], maxDepth=rf_cfg["max_depth"],
                               seed=seed), False),
        ("Gradient-Boosted Trees",
         GBTRegressor(featuresCol="features", labelCol=LABEL_COLUMN,
                      maxIter=gbt_cfg["max_iter"], maxDepth=gbt_cfg["max_depth"],
                      seed=seed), False),
    ]

    results, fitted = [], {}
    for name, estimator, scale in estimators:
        pipeline = Pipeline(stages=build_stages(scale) + [estimator])
        started = time.perf_counter()
        model = pipeline.fit(train)
        elapsed = time.perf_counter() - started
        fitted[name] = model
        result = evaluate(name, model, test, elapsed, split_tag)
        results.append(result)
        LOGGER.info("%s -> %s", name, result)

    add_skill_decomposition(results, baselines)
    for result in results:
        LOGGER.info(
            "%s explains %.2f%% of the link-history baseline's residual variance",
            result["model"], 100 * (result["baseline_residual_variance_explained"] or 0),
        )

    return {
        "split": split_tag,
        "info": info,
        "train_rows": n_train,
        "test_rows": n_test,
        "baselines": baselines,
        "results": results,
        "_fitted": fitted,
        "_train": train,
    }


def save_interpretation(fitted: dict) -> None:
    """Persist RF importances and LR coefficients for the report figures.

    The first run discarded the fitted models, so neither existed. They answer
    different questions and are both kept: importance says how much a feature
    contributed, the signed coefficient says in which direction it pushes.
    """
    rf_model = fitted["Random Forest"]
    assembler = [s for s in rf_model.stages if isinstance(s, VectorAssembler)][0]
    names = assembler.getInputCols()
    importances = sorted(
        zip(names, rf_model.stages[-1].featureImportances.toArray()),
        key=lambda pair: pair[1], reverse=True,
    )
    with open(resolve("outputs") / "feature_importance.json", "w") as handle:
        json.dump(
            [{"feature": n, "importance": round(float(v), 5)} for n, v in importances],
            handle, indent=2,
        )
    LOGGER.info("Top features: %s", [(n, round(float(v), 3)) for n, v in importances[:5]])

    lr_model = fitted["Linear Regression"]
    lr_assembler = [s for s in lr_model.stages if isinstance(s, VectorAssembler)][0]
    lr_names = lr_assembler.getInputCols()
    coefficients = sorted(
        zip(lr_names, lr_model.stages[-1].coefficients.toArray()),
        key=lambda pair: abs(pair[1]), reverse=True,
    )
    with open(resolve("outputs") / "lr_coefficients.json", "w") as handle:
        json.dump(
            {
                "intercept": float(lr_model.stages[-1].intercept),
                "note": "Scaled features. Positive coefficient => pushes predicted flow up.",
                "coefficients": [{"feature": n, "coefficient": round(float(v), 5)}
                                 for n, v in coefficients],
            },
            handle, indent=2,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cv", action="store_true", help="Skip the CrossValidator run.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", stream=sys.stdout
    )
    cfg = load_config()
    seed = cfg["ml"]["seed"]
    folds = cfg["ml"]["cv_folds"]

    assert_no_leakage()
    LOGGER.info("Leakage check passed. Features: %s", len(ALL_FEATURES))
    for column, reason in EXCLUDED_WITH_REASON.items():
        LOGGER.info("  excluded '%s': %s", column, reason)

    (resolve("outputs") / "predictions").mkdir(parents=True, exist_ok=True)
    spark = get_spark("train_models")
    payload: dict = {"features": ALL_FEATURES,
                     "excluded": EXCLUDED_WITH_REASON,
                     "leaky_never_used": LEAKY_COLUMNS,
                     "splits": []}
    try:
        primary = run_split(spark, cfg, "primary", **cfg["split"]["primary"])
        control = run_split(spark, cfg, "control", **cfg["split"]["control"])

        if primary:
            save_interpretation(primary["_fitted"])

        # CrossValidator inside the training block only. Tuning against the
        # future block would be the same leak as training on it.
        #
        # Fitted on a sample: CV refits folds x grid = 3 x 4 = 12 times, which at
        # full size dominates the whole run for a hyperparameter answer a sample
        # gives just as well. The resulting score is labelled as sampled wherever
        # it is reported, and is not comparable like-for-like with the
        # full-data models above.
        if not args.skip_cv and primary:
            cv_fraction = cfg["ml"]["cv_sample_fraction"]
            cv_train = (
                primary["_train"].sample(fraction=cv_fraction, seed=seed)
                if cv_fraction < 1.0 else primary["_train"]
            )
            LOGGER.info("CV on %.0f%% sample of the training block (%s rows)",
                        100 * cv_fraction, f"{cv_train.count():,}")

            rf = RandomForestRegressor(featuresCol="features", labelCol=LABEL_COLUMN, seed=seed)
            grid = (
                ParamGridBuilder()
                .addGrid(rf.numTrees, [20, 30])
                .addGrid(rf.maxDepth, [6, 8])
                .build()
            )
            cross_validator = CrossValidator(
                estimator=Pipeline(stages=build_stages(False) + [rf]),
                estimatorParamMaps=grid,
                evaluator=RegressionEvaluator(labelCol=LABEL_COLUMN,
                                              predictionCol="prediction", metricName="rmse"),
                numFolds=folds, seed=seed, parallelism=2,
            )
            started = time.perf_counter()
            cv_model = cross_validator.fit(cv_train)
            elapsed = time.perf_counter() - started
            _, test_block, _ = prepare(spark, **cfg["split"]["primary"])
            cv_result = evaluate(f"Random Forest (CV-tuned, {cv_fraction:.0%} sample)",
                                 cv_model.bestModel, test_block, elapsed, "primary")
            best_rf = cv_model.bestModel.stages[-1]
            cv_result["chosen_params"] = {
                "numTrees": best_rf.getNumTrees, "maxDepth": best_rf.getMaxDepth()
            }
            cv_result["trained_on_sample"] = cv_fraction
            add_skill_decomposition([cv_result], primary["baselines"])
            LOGGER.info("CV-tuned -> %s", cv_result)
            primary["results"].append(cv_result)

        for block in (primary, control):
            if not block:
                continue
            block.pop("_fitted", None)
            block.pop("_train", None)
            payload["splits"].append(block)

        with open(resolve("outputs") / "model_results.json", "w") as handle:
            json.dump(payload, handle, indent=2)
        LOGGER.info("Wrote %s", resolve("outputs") / "model_results.json")
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
