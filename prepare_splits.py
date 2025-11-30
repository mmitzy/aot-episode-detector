from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, max as spark_max, rand, when, lit
)

# ---------- CONFIG ---------- #

PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_CSV = str(PROJECT_ROOT / "data" / "index.csv")

TRAIN_CSV = str(PROJECT_ROOT / "data" / "train_core.csv")
VAL_CSV   = str(PROJECT_ROOT / "data" / "val_core.csv")
TEST_CSV  = str(PROJECT_ROOT / "data" / "test_core.csv")

CORE_MIN = 0.20
CORE_MAX = 0.80
CORE_LEN = CORE_MAX - CORE_MIN   # 0.60

TRAIN_CORE_FRAC = 0.80          
TRAIN_WINDOW_LEN = TRAIN_CORE_FRAC * CORE_LEN # 0.48

# ---------------------------- #

def flatten_spark_csv(tmp_dir_str, final_path_str):
    tmp_dir = Path(tmp_dir_str)
    final_path = Path(final_path_str)
    part_files = list(tmp_dir.glob("part-*.csv"))
    if not part_files:
        raise RuntimeError(f"No part-*.csv files found in {tmp_dir}")
    final_path.write_bytes(part_files[0].read_bytes())


def main():
    spark = (
        SparkSession.builder
        .appName("aot-episode-detector-splits")
        .getOrCreate()
    )

    df = spark.read.csv(INDEX_CSV, header=True, inferSchema=True)

    # Episode stats: max_time_sec and max_frame_idx
    ep_stats = (
        df.groupBy("episode_id")
        .agg(
            spark_max("time_sec").alias("max_time_sec"),
            spark_max("frame_idx").alias("max_frame_idx"),
        )
    )

    # Compute episode_frac per frame (0..1 position in episode)
    df2 = df.join(ep_stats, on="episode_id", how="inner")
    df2 = df2.withColumn("episode_frac", col("time_sec") / col("max_time_sec"))

    # Tag regions (core vs border)
    df2 = df2.withColumn(
        "region",
        when(col("episode_frac") < CORE_MIN, lit("early_border"))
        .when(col("episode_frac") > CORE_MAX, lit("late_border"))
        .otherwise(lit("core"))
    )

    # Keep only core frames for classifier splitting for now
    core_df = df2.filter(col("region") == "core")

    # For each episode, pick a random t_start
    max_start = CORE_MIN + (CORE_LEN - TRAIN_WINDOW_LEN)  # 0.20 + (0.60 - 0.48) = 0.32

    ep_splits = (
        ep_stats
        .select("episode_id")  # one row per ep
        .withColumn("split_start_frac", CORE_MIN + rand(seed=42) * (max_start - CORE_MIN))
    )

    # Join split_start_frac back to each core frame
    core_df = core_df.join(ep_splits, on="episode_id", how="inner")

    # Define train/eval inside core:
    train_cond = (
        (col("episode_frac") >= col("split_start_frac")) &
        (col("episode_frac") <= (col("split_start_frac") + lit(TRAIN_WINDOW_LEN)))
    )

    core_df = core_df.withColumn(
        "split_role",
        when(train_cond, lit("train")).otherwise(lit("eval"))
    )

    # Within eval frames, randomly assign to val/test
    eval_df = core_df.filter(col("split_role") == "eval")
    eval_df = eval_df.withColumn("eval_rand", rand(seed=123))

    val_df = eval_df.filter(col("eval_rand") <= 0.5).drop("eval_rand")
    test_df = eval_df.filter(col("eval_rand") > 0.5).drop("eval_rand")

    train_df = core_df.filter(col("split_role") == "train")

    print("Counts (core only):")
    print("  train:", train_df.count())
    print("  val:  ", val_df.count())
    print("  test: ", test_df.count())

    cols_to_keep = ["episode_id", "frame_path", "frame_idx", "time_sec"]
    train_df = train_df.select(*cols_to_keep)
    val_df   = val_df.select(*cols_to_keep)
    test_df  = test_df.select(*cols_to_keep)

    # Write out CSVs
    (train_df
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(TRAIN_CSV + "_tmp"))

    (val_df
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(VAL_CSV + "_tmp"))

    (test_df
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", True)
        .csv(TEST_CSV + "_tmp"))

    flatten_spark_csv(TRAIN_CSV + "_tmp", TRAIN_CSV)
    flatten_spark_csv(VAL_CSV   + "_tmp", VAL_CSV)
    flatten_spark_csv(TEST_CSV  + "_tmp", TEST_CSV)

    print("\nWrote:")
    print("  train_core.csv ->", TRAIN_CSV)
    print("  val_core.csv   ->", VAL_CSV)
    print("  test_core.csv  ->", TEST_CSV)

    spark.stop()


if __name__ == "__main__":
    main()
