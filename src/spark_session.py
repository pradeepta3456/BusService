"""Single place where the SparkSession is configured.

Partition count, memory and the local master live here rather than being
repeated per script, so the optimisation evidence in the report describes one
real configuration.
"""

from __future__ import annotations

from pyspark.sql import SparkSession

from src.config import PROJECT_ROOT, load_config


def get_spark(app_suffix: str | None = None) -> SparkSession:
    cfg = load_config()["spark"]
    name = cfg["app_name"] + (f" - {app_suffix}" if app_suffix else "")

    spark = (
        SparkSession.builder
        .appName(name)
        # Configurable rather than hard-coded local[*]: the executor-thread count
        # is the main lever on how hard this pins a laptop. See settings.yaml.
        .master(cfg.get("master", "local[*]"))
        # >=4 shuffle partitions is a brief requirement; 8 matches local cores
        # and keeps partitions large enough to avoid task-scheduling overhead.
        .config("spark.sql.shuffle.partitions", cfg["shuffle_partitions"])
        .config("spark.driver.memory", cfg["driver_memory"])
        .config("spark.sql.session.timeZone", "UTC")
        # Auto-broadcast the small dimension tables (road links, regions)
        # against the 5.3M-row count table.
        .config("spark.sql.autoBroadcastJoinThreshold", 32 * 1024 * 1024)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    checkpoint_dir = PROJECT_ROOT / "data" / "processed" / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    spark.sparkContext.setCheckpointDir(str(checkpoint_dir))
    return spark
