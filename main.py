
# CLUSTER COMPUTING INT315

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import verify_full_environment
from src.spark_session import get_spark_session
from src.data_loader import load_flight_dataset
from src.data_quality import run_full_data_quality_audit, save_quality_reports
from src.data_cleaning import run_data_cleaning
from src.feature_engineering import run_feature_engineering
from src.model_training import run_model_training
from src.model_evaluation import run_model_evaluation
from src.hyperparameter_tuning import run_hyperparameter_tuning
from src.final_dashboard import run_final_completion_step


def run_step_2_quality_audit() -> None:
    """Executes Step 2: Data Quality Analysis."""
    print("\n" + "=" * 70)
    print("   STEP 2: DATA QUALITY ANALYSIS")
    print("=" * 70 + "\n")

    spark = None
    try:
        print("[*] Initializing SparkSession for Quality Audit...")
        spark = get_spark_session(
            app_name="AirlineDelayPrediction-QualityAudit",
            master="local[*]",
            driver_memory="4g"
        )
        print(f"[✓] SparkSession active (Spark Version: {spark.version})")

        dataset_file = PROJECT_ROOT / "dataset" / "flight_data_2024.csv"
        df = load_flight_dataset(spark, dataset_file, infer_schema=True)

        report = run_full_data_quality_audit(df)

        out_dir = PROJECT_ROOT / "output"
        save_quality_reports(report, out_dir)

        print("\n" + "=" * 70)
        print(" [✓] STEP 2 COMPLETE: DATA QUALITY AUDIT COMPLETED")
        print("=" * 70 + "\n")

    finally:
        if spark is not None:
            spark.stop()
            print("[*] SparkSession stopped cleanly.")


def run_step_3_cleaning() -> None:
    """Executes Step 3: Distributed Data Cleaning."""
    print("\n" + "=" * 70)
    print("   STEP 3: DISTRIBUTED DATA CLEANING")
    print("=" * 70 + "\n")
    run_data_cleaning()
    print("\n" + "=" * 70)
    print(" [✓] STEP 3 COMPLETE: CLEANED DATASET SAVED TO PARQUET")
    print("=" * 70 + "\n")


def run_step_4_features() -> None:
    """Executes Step 4: Feature Engineering using PySpark MLlib."""
    print("\n" + "=" * 70)
    print("   STEP 4: PYSPARK MLLIB FEATURE ENGINEERING")
    print("=" * 70 + "\n")
    run_feature_engineering()
    print("\n" + "=" * 70)
    print(" [✓] STEP 4 COMPLETE: PIPELINE MODEL & TRAIN/TEST SPLIT SAVED")
    print("=" * 70 + "\n")


def run_step_5_training() -> None:
    """Executes Step 5: Regression Model Training using PySpark MLlib."""
    print("\n" + "=" * 70)
    print("   STEP 5: REGRESSION MODEL TRAINING & EVALUATION")
    print("=" * 70 + "\n")
    run_model_training()
    print("\n" + "=" * 70)
    print(" [✓] STEP 5 COMPLETE: ALL MODELS TRAINED, EVALUATED & SAVED")
    print("=" * 70 + "\n")


def run_step_6_evaluation() -> None:
    """Executes Step 6: Model Evaluation & Performance Comparison."""
    print("\n" + "=" * 70)
    print("   STEP 6: MODEL EVALUATION & PERFORMANCE COMPARISON")
    print("=" * 70 + "\n")
    run_model_evaluation()
    print("\n" + "=" * 70)
    print(" [✓] STEP 6 COMPLETE: EVALUATION CHARTS & REPORT GENERATED")
    print("=" * 70 + "\n")


def run_step_7_tuning() -> None:
    """Executes Step 7: Hyperparameter Tuning using PySpark MLlib."""
    print("\n" + "=" * 70)
    print("   STEP 7: HYPERPARAMETER TUNING USING PYSPARK MLLIB")
    print("=" * 70 + "\n")
    run_hyperparameter_tuning()
    print("\n" + "=" * 70)
    print(" [✓] STEP 7 COMPLETE: TUNED MODELS & COMPARISON REPORT SAVED")
    print("=" * 70 + "\n")


def run_step_8_dashboard() -> None:
    """Executes Step 8: Final Dashboard & Project Completion."""
    print("\n" + "=" * 70)
    print("   STEP 8: FINAL DASHBOARD & PROJECT COMPLETION")
    print("=" * 70 + "\n")
    run_final_completion_step()
    print("\n" + "=" * 70)
    print(" [✓] STEP 8 COMPLETE: ALL DASHBOARDS, REPORTS & CHARTS SAVED")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Airline Delay Prediction Pipeline")
    parser.add_argument(
        "--step",
        type=int,
        choices=[2, 3, 4, 5, 6, 7, 8],
        default=None,
        help="Specify which step to run (2: Quality, 3: Cleaning, 4: Features, 5: Models, 6: Evaluation, 7: Tuning, 8: Dashboard). Omit to run full pipeline."
    )
    args = parser.parse_args()

    print("\n" + "=" * 74)
    print("       AIRLINE DELAY PREDICTION — DISTRIBUTED PYSPARK PIPELINE")
    print("=" * 74 + "\n")

    env_ok = verify_full_environment()
    if not env_ok:
        print("\n[!] Setup halted: Environment requirements not satisfied.")
        sys.exit(1)

    try:
        if args.step == 2:
            run_step_2_quality_audit()
        elif args.step == 3:
            run_step_3_cleaning()
        elif args.step == 4:
            run_step_4_features()
        elif args.step == 5:
            run_step_5_training()
        elif args.step == 6:
            run_step_6_evaluation()
        elif args.step == 7:
            run_step_7_tuning()
        elif args.step == 8:
            run_step_8_dashboard()
        else:
            run_step_2_quality_audit()
            run_step_3_cleaning()
            run_step_4_features()
            run_step_5_training()
            run_step_6_evaluation()
            run_step_8_dashboard()

        print("\n" + "=" * 74)
        print("  ALL REQUESTED PIPELINE STAGES COMPLETED SUCCESSFULLY")
        print("=" * 74 + "\n")

    except Exception as e:
        print(f"\n[!] Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
