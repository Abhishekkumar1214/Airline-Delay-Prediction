# Airline Delay Prediction — Distributed PySpark MLlib Pipeline

A production-grade, distributed machine learning and data engineering project built with **Apache Spark (PySpark MLlib)** and Python. This pipeline ingests, cleans, audits, and engineers features across **7,079,081 Bureau of Transportation Statistics (BTS 2024)** flight records to train and benchmark distributed regression models that predict flight arrival delays (`arr_delay` in minutes) strictly from pre-flight scheduled parameters.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Technologies Used](#-technologies-used)
3. [Dataset Information](#-dataset-information)
4. [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
5. [Installation & Prerequisites](#-installation--prerequisites)
6. [Pipeline Execution Guide (Steps 1–8)](#-pipeline-execution-guide-steps-18)
7. [Model Performance Summary & Leaderboard](#-model-performance-summary--leaderboard)
8. [Interactive Dashboards & Visualizations](#-interactive-dashboards--visualizations)
9. [Sample Real-Time Inference Workflow](#-sample-real-time-inference-workflow)
10. [Windows Compatibility & JVM Optimizations](#-windows-compatibility--jvm-optimizations)

---

## ✈️ Project Overview
Commercial flight delays cost airlines and passengers tens of billions of dollars each year. The core challenge is predicting arrival delay **before departure** using only scheduled operational metadata (scheduled departure hour, operating airline carrier, origin, destination, scheduled flight distance, calendar day/month), without leaking unobservable post-flight attributions (such as carrier delay, weather delay, or mid-air actual tracking info).

### Key Accomplishments:
- **Big Data Scale**: Audited and processed **7,079,081 raw records** down to **6,965,267 clean flights**.
- **Data Leakage Prevention**: Strict **80/20 train-test split executed first**; StringIndexer, OneHotEncoder, and VectorAssembler pipeline fitted exclusively on the 5.57M training set.
- **High-Dimensional Sparse Features**: Encoded into **723 feature dimensions** (`SparseVector`).
- **Distributed MLlib Algorithms**: Trained and benchmarked **Linear Regression**, **Decision Tree Regressor**, and **Random Forest Regressor**.
- **Model Winner**: **Decision Tree Regressor** achieved lowest RMSE (**57.2093 min**), lowest MAE (**26.1686 min**), and highest R² (**0.0336**).
- **Executive Visualizations & Dashboards**: 7 high-resolution charts in `visualizations/final/`, an interactive HTML5 dashboard with custom client-side flight delay estimator, and a Streamlit app.

---

## 🛠️ Technologies Used
- **Distributed Computing**: Apache Spark 4.0.5, PySpark 4.2.0 (DataFrame, SQL, MLlib)
- **Runtime Environment**: Python 3.13, Java OpenJDK 17.0.12 (64-bit)
- **Data Serialization**: Apache Parquet (compressed columnar storage)
- **Data Visualization**: Matplotlib 3.10, Seaborn styles, Chart.js (HTML5)
- **Dashboard Interfaces**: Interactive HTML5/CSS3 Dashboard, Streamlit
- **OS Platform**: Microsoft Windows 11 (with custom Java Unsafe NativeIO patch)

---

## 📊 Dataset Information
- **Source**: United States Department of Transportation — Bureau of Transportation Statistics (BTS)
- **Dataset File**: `dataset/flight_data_2024.csv` (~1.3 GB raw CSV)
- **Raw Dimensions**: 7,079,081 rows × 35 columns
- **Cleaned Dimensions**: 6,965,267 rows × 10 columns (`output/cleaned_flights/` in Parquet)
- **Train Split (80%)**: 5,572,312 rows (`output/train_data/`)
- **Test Split (20%)**: 1,392,955 rows (`output/test_data/`)
- **Target Label**: `arr_delay` (continuous numerical arrival delay in minutes; negative values represent early arrivals).
- **Strictly Excluded Post-Flight Columns**: `weather_delay`, `nas_delay`, `carrier_delay`, `security_delay`, `late_aircraft_delay`, `dep_delay`, `arr_time`.

---

## 📁 Project Architecture & Directory Structure

```text
airline-prediction/
│
├── dataset/
│   └── flight_data_2024.csv             # Raw BTS 2024 flight dataset (~1.3 GB)
│
├── src/                                 # Modular enterprise Python package
│   ├── __init__.py                      # Package initializer
│   ├── utils.py                         # Environment check & Windows path handling
│   ├── spark_session.py                 # SparkSession builder & Windows NativeIO patch
│   ├── data_loader.py                   # PySpark CSV reader & schema inspector
│   ├── data_quality.py                  # Single-pass 35-column quality audit engine
│   ├── data_cleaning.py                 # Distributed cleaning (cancellations & nulls)
│   ├── feature_engineering.py           # MLlib Pipeline (StringIndexer, OHE, Assembler)
│   ├── model_training.py                # Regression training & evaluation module
│   ├── model_evaluation.py              # Comparative metrics & visualization generator
│   ├── hyperparameter_tuning.py         # Grid search & validation split tuning module
│   ├── final_dashboard.py               # Step 8 dashboard & report generation engine
│   ├── dashboard.py                     # Streamlit application script
│   │
│   ├── 02_data_quality.py               # Standalone runner: Step 2 Quality Audit
│   ├── 03_data_cleaning.py              # Standalone runner: Step 3 Cleaning
│   ├── 04_feature_engineering.py        # Standalone runner: Step 4 Feature Engineering
│   ├── 05_model_training.py             # Standalone runner: Step 5 Model Training
│   ├── 06_model_evaluation.py           # Standalone runner: Step 6 Evaluation Charts
│   ├── 07_hyperparameter_tuning.py      # Standalone runner: Step 7 Hyperparameter Tuning
│   └── 08_final_dashboard.py            # Standalone runner: Step 8 Dashboard & Report
│
├── dashboard/
│   └── index.html                       # Standalone interactive HTML5 executive dashboard
│
├── models/                              # Trained Spark MLlib Model Artifacts
│   ├── linear_regression/               # Baseline Linear Regression model
│   ├── decision_tree/                   # Baseline Decision Tree model (Winner)
│   ├── random_forest/                   # Baseline Random Forest model
│   ├── tuned_linear_regression/         # Tuned ElasticNet Linear Regression model
│   ├── tuned_decision_tree/             # Tuned Decision Tree model (depth=5)
│   └── tuned_random_forest/             # Tuned Random Forest model
│
├── output/                              # Generated datasets, pipelines & reports
│   ├── cleaned_flights/                 # 6.96M cleaned Parquet partitions
│   ├── feature_pipeline_model/          # Fitted MLlib PipelineModel
│   ├── train_data/                      # 5.57M transformed train records (Parquet)
│   ├── test_data/                       # 1.39M transformed test records (Parquet)
│   ├── dashboard.html                   # Mirrored interactive HTML dashboard
│   ├── model_comparison.csv             # Benchmark metrics CSV (RMSE, MAE, R², Time)
│   ├── final_dashboard_metrics.json     # Extracted aggregations & prediction samples
│   ├── data_quality_report.txt          # Step 2 full audit report
│   ├── data_cleaning_report.txt         # Step 3 cleaning summary report
│   ├── feature_engineering_report.txt   # Step 4 train/test split report
│   ├── model_training_report.txt        # Step 5 training benchmark report
│   ├── model_evaluation_report.txt      # Step 6 evaluation interpretation report
│   └── final_project_report.txt         # Step 8 comprehensive 15-section final report
│
├── visualizations/                      # High-resolution charts (300 DPI)
│   ├── 00_model_comparison_dashboard.png # Multi-panel summary dashboard
│   ├── 01_rmse_comparison.png           # RMSE bar chart
│   ├── 02_mae_comparison.png            # MAE bar chart
│   ├── 03_r2_score_comparison.png       # R² score bar chart
│   ├── 04_training_time_comparison.png  # Training execution time chart
│   ├── 05_prediction_time_comparison.png# Prediction throughput time chart
│   └── final/                           # Step 8 Publication-Ready Charts
│       ├── 01_flight_delay_distribution.png
│       ├── 02_early_vs_delayed_flights.png
│       ├── 03_avg_delay_by_airline.png
│       ├── 04_avg_delay_by_departure_hour.png
│       ├── 05_top10_airports_by_delay.png
│       ├── 06_actual_vs_predicted_delay.png
│       └── 07_model_performance_comparison.png
│
├── main.py                              # Master orchestrator entry point
├── requirements.txt                     # Python dependencies
└── README.md                            # Comprehensive project guide
```

---

## ⚙️ Installation & Prerequisites

### 1. Java Runtime (OpenJDK 17 recommended)
Apache Spark requires a 64-bit Java Virtual Machine.
Verify your Java setup:
```powershell
java -version
```
Ensure `JAVA_HOME` points to your JDK installation:
```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-17"
```

### 2. Python Environment
Install project dependencies:
```powershell
pip install -r requirements.txt
```
*(Dependencies: `pyspark==4.2.0`, `matplotlib>=3.8.0`, `numpy>=1.26.0`, `streamlit>=1.30.0`)*

---

## 🚀 Pipeline Execution Guide (Steps 1–8)

You can run individual stages using `main.py --step <N>` or run the standalone step scripts:

### Execute Complete End-to-End Pipeline:
```powershell
python main.py
```

### Execute Individual Project Steps:
- **Step 1: Environment & Spark Verification**
  ```powershell
  python -c "from src.utils import verify_full_environment; verify_full_environment()"
  ```
- **Step 2: Data Quality Analysis (35-Column Audit)**
  ```powershell
  python main.py --step 2
  # Or: python src/02_data_quality.py
  ```
- **Step 3: Distributed Data Cleaning (Parquet Output)**
  ```powershell
  python main.py --step 3
  # Or: python src/03_data_cleaning.py
  ```
- **Step 4: Feature Engineering with MLlib Pipeline**
  ```powershell
  python main.py --step 4
  # Or: python src/04_feature_engineering.py
  ```
- **Step 5: Regression Model Training**
  ```powershell
  python main.py --step 5
  # Or: python src/05_model_training.py
  ```
- **Step 6: Model Evaluation & Visualizations**
  ```powershell
  python main.py --step 6
  # Or: python src/06_model_evaluation.py
  ```
- **Step 7: Hyperparameter Tuning**
  ```powershell
  python main.py --step 7
  # Or: python src/07_hyperparameter_tuning.py
  ```
- **Step 8: Final Dashboard, Charts & Final Report**
  ```powershell
  python main.py --step 8
  # Or: python src/08_final_dashboard.py
  ```

---

## 🏆 Model Performance Summary & Leaderboard

All models were evaluated on the **exact same untouched test set of 1,392,955 flights**:

| Model Architecture | Training Dataset | RMSE (min) | MAE (min) | $R^2$ Score | Train Time | Prediction Time | Selection Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Decision Tree Regressor** | **100% Train (5,572,312 rows)** | **57.2093** | **26.1686** | **0.0336** | 90.65 s | 1.21 s | 🏆 **Overall Winner** |
| **Random Forest Regressor** | 20% Sample (1,114,314 rows) | 57.2696 | 26.1747 | 0.0316 | 47.58 s | 1.96 s | Competitive Accuracy |
| **Linear Regression** | 100% Train (5,572,312 rows) | 57.5854 | 26.5728 | 0.0209 | **5.75 s** | **0.78 s** | Highest Throughput |

### Key Findings & Aviation Domain Context:
1. **Decision Tree Regressor Wins**: Achieved the lowest RMSE and highest $R^2$, capturing non-linear congestion thresholds (e.g., peak evening arrival hours at major hubs like DFW and SFO).
2. **Why $R^2 \approx 0.02 - 0.035$**: Using *strictly pre-flight scheduled parameters* predicts baseline operational delay risk without relying on unobservable real-time tactical events (en-route weather convective storms, ground stop programs, mechanical turnarounds). This guarantees authentic, leakage-free generalization.
3. **Linear Regression Throughput**: Trains on 5.57 million flights in just 5.75 seconds and scores 1.39 million flights in 0.78 seconds.

---

## 🖥️ Interactive Dashboards & Visualizations

### Option 1: Standalone Interactive HTML5 Dashboard (Zero Dependencies)
Open `dashboard/index.html` or `output/dashboard.html` directly in any modern browser:
```powershell
# Open directly in Windows default browser
Start-Process "dashboard/index.html"
```
Or serve locally via Python's built-in HTTP server:
```powershell
python -m http.server 8501 --directory dashboard
# Open http://localhost:8501 in your browser
```

**Interactive Dashboard Features:**
- **KPI Counters**: 6.96M cleaned flights, 723 feature dimensions, +7.10 min average delay, 62% early arrival rate.
- **Interactive Chart.js Visuals**: Doughnut punctuality chart, hourly delay trend, carrier comparison, and airport delay rankings.
- **Client-Side Flight Delay Estimator**: Real-time delay calculator predicting minutes and risk levels for any carrier, route, departure hour, and distance.

### Option 2: Streamlit Dashboard
If Streamlit is installed:
```powershell
streamlit run src/dashboard.py
```

### Publication-Quality Charts Saved in `visualizations/final/`:
1. `01_flight_delay_distribution.png`: Bimodal arrival delay distribution with early and delayed zones.
2. `02_early_vs_delayed_flights.png`: Punctuality breakdown (62.0% early, 1.8% on-time, 15.4% minor delay, 20.8% delayed $\ge 15$m).
3. `03_avg_delay_by_airline.png`: Average arrival delay across 15 major US passenger airlines.
4. `04_avg_delay_by_departure_hour.png`: Delay propagation curve showing the progression from 5:00 AM (-2.2 min) to 20:00 PM (+15.5 min).
5. `05_top10_airports_by_delay.png`: Ranking of top 10 delayed US destination airports (DFW, SFO, PHL, HPN, SJU, CLT, MIA, ICT, MSN, BDL).
6. `06_actual_vs_predicted_delay.png`: Actual vs Predicted scatter plot and residual error distribution.
7. `07_model_performance_comparison.png`: 4-panel comparison of RMSE, MAE, $R^2$, and training execution times.

---

## ✈️ Sample Real-Time Inference Workflow

The saved pipeline model (`output/feature_pipeline_model/`) and best model (`models/decision_tree/`) can score real-time flight itineraries:

| Flight ID | Airline | Route | Departure | Distance | Predicted Delay | Punctuality Risk & Operational Assessment |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **DL101** | Delta Air Lines | ATL → LGA | 07:00 | 762 mi | **-2.14 min** | **[LOW RISK]** Favorable early morning schedule; minimal congestion. |
| **AA404** | American Airlines | DFW → ORD | 19:30 | 802 mi | **+47.93 min** | **[ELEVATED RISK]** Peak evening turnaround congestion at DFW hub. |
| **UA820** | United Airlines | SFO → DEN | 14:15 | 967 mi | **+8.29 min** | **[MODERATE RISK]** Afternoon hub operations; within FAA 15m tolerance. |
| **F9250** | Frontier Airlines | MCO → PHL | 13:00 | 861 mi | **+25.76 min** | **[ELEVATED RISK]** Mid-day turnaround buffer pressure on budget carrier. |
| **WN12** | Southwest Airlines | DAL → HOU | 06:15 | 239 mi | **-7.95 min** | **[LOW RISK]** Short-haul early departure with clean airspace. |

---

## 🪟 Windows Compatibility & JVM Optimizations

Running Apache Spark natively on Windows often triggers `AccessDeniedException` during `FileOutputCommitter` because Hadoop's `NativeIO.Windows.access()` check fails under standard user tokens in user/OneDrive directories.

### Implemented Fix in `src/spark_session.py`:
```python
def patch_windows_hadoop(spark: SparkSession) -> None:
    if os.name == "nt":
        # Uses Java reflection & sun.misc.Unsafe via Py4J to set
        # org.apache.hadoop.util.Shell.WINDOWS = False
        # Routes all file checks through standard JVM java.io.File methods
```
This guarantees 100% reliable distributed writes for all Parquet datasets and MLlib models without requiring administrative privileges, `winutils.exe` permissions, or Cygwin.

---

## 📜 Project Deliverables & Reports
- **Final Project Report**: [`output/final_project_report.txt`](output/final_project_report.txt)
- **Interactive Dashboard**: [`dashboard/index.html`](dashboard/index.html)
- **Model Comparison Metrics**: [`output/model_comparison.csv`](output/model_comparison.csv)
- **Data Quality Report**: [`output/data_quality_report.txt`](output/data_quality_report.txt)
- **Cleaning Report**: [`output/data_cleaning_report.txt`](output/data_cleaning_report.txt)
- **Feature Engineering Report**: [`output/feature_engineering_report.txt`](output/feature_engineering_report.txt)
- **Evaluation Report**: [`output/model_evaluation_report.txt`](output/model_evaluation_report.txt)
