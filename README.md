# When Does the Road Fill?

**Forecasting hourly traffic demand on Britain's road network**
ST5011CEM Big Data Programming Project

A planner sizing infrastructure already knows how much traffic a road carries on average.
They need to know **when it fills, and by how much**. This project forecasts hourly demand
from 26 years of DfT counts, and asks how much of that forecast is real skill rather than
already-known road identity.

---

## Data

| Dataset | Scale |
|---|---|
| DfT raw hourly counts | **5,269,632 rows × 35 cols** — 43,401 count points, 2000–2025 |
| DfT AADF | 600,551 rows — annual average daily flow |

Open Government Licence v3.0, no API key. **The 100k requirement is met 53× by one real
table**; no synthetic augmentation anywhere. 1.0 GB CSV → 71 MB Parquet partitioned by year.
After cleaning: **5,269,609 rows**. DfT counts run a 12-hour day (07:00–18:59) — there is no
night-time data.

---

## The headline result

**R² = 0.861 is not a good model. It is a lookup table.**

| primary split (train ≤2019 → forecast 2023–25) | RMSE | R² | skill vs baseline |
|---|---|---|---|
| **Baseline: the link's own history** | **363.1** | **0.857** | — |
| Linear Regression | 362.8 | 0.857 | +0.14% |
| Random Forest | 365.8 | 0.855 | **−1.47%** ✗ |
| Gradient-Boosted Trees | **357.4** | **0.861** | +3.14% |
| Random Forest (CV-tuned) | 364.9 | 0.855 | **−0.98%** ✗ |

A trivial baseline — *assume each road behaves as it always has* — scores 0.857. The best
model adds **+0.0045**. Two models are **worse than doing nothing**.

Feature importance says why: **77% is the link's own history, and `hour` carries 0.009** —
in a peak-hour project. Because the target is *absolute* flow, the model only has to identify
the road, and between-link variance (a motorway carries 50× a lane) swamps the time-of-day
signal.

**The level of a road's traffic is trivially predictable; the shape of its day is the real
question, and this target cannot ask it.** The fix is identified and half-built: predict flow
normalised to each link's own mean (`peak_index`, already in `fact_hourly_profile`), against
which the baseline scores zero by construction.

Other findings, reported as found:

- **COVID did not break forecasting** — pre-2020 → 2023–25 works about as well as a
  within-decade forecast. Not the expected result.
- **The model ranking inverts between splits** (GBT best on primary, RF on control). A
  ranking that flips is noise.
- **R² is not comparable across the splits** — their test blocks have SD 962.6 vs 752.9.
- Normalised per link, the **AM peak (1.240) marginally exceeds the PM peak (1.237)**,
  reversing the raw-flow ordering — the raw result was a composition effect.
- Motorways peak at **07:00** and flat; minor roads at **08:00** and sharp, with a 15:00
  school-run bump. The sharpest link in Britain carries **4.2× its own average**.

---

## Leakage discipline

Verified on the real data:

```
two_wheeled + cars + buses + LGVs + all_HGVs == all_motor_vehicles
    on 5,269,608 / 5,269,609 rows  (100.0000%)
```

The mode columns are the target's **components**, not correlates (`cars_and_taxis` alone is
r = 0.991). All are barred; `assert_no_leakage()` aborts the run if one returns, and a test
proves the guard fires. Also excluded: **`year`** (it *is* the split variable) and
**`link_length_km`** (null for 100% of minor roads, 0% of major — road class encoded as
missingness).

The split is **by year, never random**. Two are run: primary (≤2019 → 2023–25, spans COVID)
and control (≤2016 → 2017–19, no break) — otherwise "cannot forecast" and "COVID changed the
roads" are indistinguishable.

---

## Architecture

```
INGESTION (Python — I/O-bound; Spark adds nothing)
└── download_counts.py     DfT raw hourly (1.0 GB) + AADF (150 MB)
PROCESSING (PySpark — the distributed core)
├── clean_counts.py        try_cast · hour window · calendar → Parquet by year
└── build_profiles.py      per-link daily profile + peak_index (window fn)
STORAGE (SQLite)                  ML (MLlib)
dim_count_point / dim_region      leakage-safe features · time-aware split
fact_hourly_profile               LR / RF / GBT + CrossValidator
VISUALISATION — small aggregates → pandas → matplotlib (23 figures)
```

**Tool choice.** Spark for the 5.27M-row clean and the profile aggregation — 1.0 GB does not
fit in pandas beside a JVM on 8 GB. Plain `requests` for two HTTP GETs. Only the ~520k-row
profile reaches SQLite; loading the full fact would be the wrong tool for no benefit.
Plotting only ever touches aggregates of a few hundred rows.

**Spark optimisation.** `shuffle.partitions = 8`; Parquet `partitionBy("year")` turns the
time-aware split into partition pruning; history lookup `broadcast()` against the 5.27M-row
fact; train/test cached; `peak_index` via a window function rather than groupBy + join.

**Security.** Every query binds `?` placeholders. `tests/test_queries.py` fires tautology,
statement-termination and UNION payloads and asserts nothing is dropped. No credentials —
SQLite is keyless and the DfT source is open.

---

## Run

Requires **Python 3.13** and **Java 17+**.

```bash
pip install -r requirements.txt
python -m src.ingest.download_counts   # 1.2 GB, one-off, no key
python run_pipeline.py --skip-ml       # clean + profiles + db + figures (~2 min)
python run_pipeline.py                 # + training (~34 min, heavy)
python -m pytest tests/ -q             # 25 tests
```

> **Thermal note.** `config/settings.yaml` sets `local[4]` / `2g` deliberately — an all-core
> run pinned every core for 44 minutes and overheated the dev laptop. Training is the only
> heavy stage. Raise both on a machine with real cooling.

**Spark UI screenshot** (outstanding): run `python -m src.process.clean_counts` and open
`localhost:4040` → Stages **while it runs** — the UI dies with the application.

---

## Layout

```
config/settings.yaml   hour window · target · split years · model sizes · Spark
run_pipeline.py        orchestrator with per-stage timings
src/{ingest,process,db,ml,viz}/
tests/                 features · clean · queries   (25 tests)
docs/                  architecture · schema · report_draft.md
outputs/               23 figures + model_results.json
```

Report: `docs/report_draft.md`.

## Licence & ethics

DfT GB Road Traffic Counts — Open Government Licence v3.0. Counts are aggregate and
non-personal. The ethical exposure is in *use*: forecasting demand where flow is already
highest tends to entrench existing patterns rather than serve suppressed demand — which is
why the modal-shift figure is included.
