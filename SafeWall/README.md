# SafeWall

**SafeWall** is an XGBoost-based runtime prediction system for HPC jobs that adds
configurable buffer strategies to reduce premature job termination caused by walltime limit violations.

---

## Background

HPC batch schedulers (PBS, Slurm, etc.) require users to declare a **walltime limit** before
submitting a job. If the job's actual execution time exceeds that limit, the scheduler kills
the job unconditionally — discarding all work done up to that point.

In practice, users over-request walltime to stay safe. This leads to poor cluster utilization
because nodes are reserved for time that jobs will never use. At the same time, users who
under-request — either accidentally or to jump the queue — face silent job termination.

SafeWall addresses this by:

1. Training a regression model on historical job data to predict actual runtime.
2. Adding a data-driven **buffer** to each prediction, providing a safety margin calibrated
   to the model's own uncertainty.
3. Comparing five buffer families to expose the trade-off between underestimation risk
   (job kills) and overestimation waste (idle reserved nodes).

---

## Dataset

The dataset contains HPC jobs from the **Theta** supercomputer at Argonne National Laboratory,
covering a **7-year period (2017–2023)**. The scheduler is Cobalt; MaxNodes = 4,392.

### Files

| File | Description |
|---|---|
| `data/theta_7year_worklog.swf` | Raw workload trace in Standard Workload Format (SWF) |
| `data/theta_7year_dataset.csv` | Preprocessed CSV with engineered features; input to the notebook |

The SWF format is a plain-text log format standardized for HPC workload traces. Each line
encodes one job with fields such as submit time, requested processors, requested walltime,
actual runtime, and exit status. The CSV is derived from the SWF by computing per-user
historical statistics (previous runtimes, accuracy metrics) as additional features.

User IDs and group IDs are anonymized (hashed) in both files.

### Target Variable

`RUNTIME_SECONDS` — actual wall-clock runtime of the job in seconds.

---

## Features

The model uses 17 numeric features. The `user`, `group_id`, and `cens` columns are
excluded from model training (user ID is used separately for personalized buffer computation).

| Feature | Description |
|---|---|
| `submit_time` | Job submission time (seconds from start of trace) |
| `NODES_USED` | Number of nodes allocated to the job |
| `requested_processors` | Number of CPU cores requested |
| `WALLTIME_SECONDS` | Walltime limit declared by the user (seconds) |
| `ELIGIBLE_WAIT_SECONDS` | Time the job spent waiting in the queue (seconds) |
| `EXIT_STATUS` | Job exit code (0 = normal completion) |
| `RUNTIME1` | Runtime of the user's most recent previous job (seconds) |
| `RUNTIME2` | Runtime of the user's second most recent previous job (seconds) |
| `RUNTIME12` | Mean of `RUNTIME1` and `RUNTIME2` |
| `Amax` | Maximum historical accuracy for this user (min/max ratio of pred vs actual) |
| `Aaverage` | Mean historical accuracy for this user |
| `Tlongest` | Longest historical runtime for this user (seconds) |
| `Tlongest10` | Mean of the 10 longest historical runtimes for this user |
| `Taverage` | Mean historical runtime for this user (seconds) |
| `Taverage10` | Mean of the 10 most recent historical runtimes for this user |
| `Tpercentile25` | 25th percentile of historical runtimes for this user |
| `Accuracy` | Accuracy of the user's walltime request for the current job |
| `cens` | Censoring indicator (always 0 in this dataset — no censored jobs) |

Rows where any feature is missing are dropped before training (~14.5% of records,
primarily first-time users who lack historical accuracy statistics).

---

## Methodology

### Model

**XGBoost** (`XGBRegressor`) with histogram-based tree building (`tree_method='hist'`).
XGBoost handles the mixed feature types, high skew in the target variable, and sparse
historical features (NaN for new users) well without requiring explicit preprocessing.

### Hyperparameter Tuning

`RandomizedSearchCV` over 50 random combinations from a grid covering:

- `n_estimators`: 100–500
- `learning_rate`: 0.01–0.20
- `max_depth`: 3–11
- `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`

5-fold cross-validation, scored by R². `random_state=42` throughout for reproducibility.

### Train-Test Split

80% training / 20% test, random split with `random_state=42`.

---

## Buffer Strategies

After fitting the base model, five families of buffer strategies are tested.
Each strategy adds a non-negative offset to the raw model prediction before
comparing against actual runtimes.

### 1. Percentage-Based

```
buffered_walltime = prediction * (1 + p)
```

`p` tested: 5%, 10%, 15%, 20%, 25%, 30%.
The buffer grows proportionally with predicted duration, which can produce large
absolute overestimates for very long jobs.

### 2. Fixed Value

```
buffered_walltime = prediction + c
```

`c` tested: 30, 60, 120, 180, 300, 600 seconds.
A uniform safety margin independent of job length. Tends to help short jobs most.

### 3. Adaptive

```
buffered_walltime = prediction * (1 + p_short)  if prediction <= 3600 s
buffered_walltime = prediction * (1 + p_long)   if prediction >  3600 s
```

Configurations tested: (5%, 10%), (10%, 15%), (10%, 20%).
Applies a smaller buffer to short jobs and a larger buffer to long jobs, motivated
by the observation that long jobs carry higher absolute uncertainty.

### 4. Error-Based

```
buffered_walltime = prediction + percentile(training_absolute_errors, p)
```

`p` tested: 50th, 75th, 90th, 95th.
The buffer is derived directly from the model's own training residuals. Adding the
P90 training error as a fixed offset means the model would have correctly bounded
~90% of training jobs.

### 5. User-Based (Personalized)

A personalized buffer is computed for each user from their own historical prediction
errors in the training set:

```
buffered_walltime = prediction + user_error_percentile(user, p)
```

Users with fewer than 3 historical training jobs fall back to the global
error-percentile buffer.

Two variants are tested:

- **Full History**: uses all of the user's training-set jobs.
- **Rolling Window (last 5 jobs)**: uses only the 5 most recent training-set jobs,
  making the buffer more responsive to recent behavioral change.

---

## Requirements

```
python >= 3.8
pandas
numpy
matplotlib
seaborn
xgboost >= 1.6
scikit-learn
joblib
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn joblib
```

---

## Usage

1. Place `theta_7year_dataset.csv` in the `data/` directory.
2. Open and run `SafeWall.ipynb` from top to bottom.
3. Results are saved to `data/predictions_with_buffers.csv`.

```bash
jupyter notebook SafeWall.ipynb
```

The notebook is self-contained. All random seeds are fixed (`random_state=42`)
so results are reproducible across runs.

---

## File Structure

```
SafeWall/
├── SafeWall.ipynb                    # Main notebook
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
└── data/
    ├── theta_7year_dataset.csv       # Preprocessed feature CSV (input)
    ├── theta_7year_worklog.swf       # Raw Theta 7-year SWF workload trace
    └── predictions_with_buffers.csv  # Model output with all buffer variants (generated)
```

---

## Citation

If you use this code or dataset in your research, please cite the accompanying paper
(JSSPP 2026).

## License

MIT
