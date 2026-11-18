# Member 1 – Dataset exploration, preprocessing and feature engineering

## Run order
```bash
pip install -r requirements.txt
# run everything from the repository root. Raw data is read from data/raw/train and data/raw/student_test
# (set the PATHSNAP_DATA environment variable only if the data lives somewhere else)

python eda/01_inventory.py             # experiments/eda/dataset_inventory.csv (+ per-partition summary)
python eda/03_network_stats.py         # network connectivity tables
python build_dataset.py --partition train
python build_dataset.py --partition student_test     # data/processed/*  (feature tables, labels, diagnostics)
python eda/06_candidate_radius.py      # train-only choice of search radius R and cap K
python eda/02_quality_notes.py         # experiments/eda/data_quality_notes.md (numbers generated from the files)
python eda/04_plots.py                 # visualizations/eda/*.png
python eda/05_folium_maps.py           # visualizations/eda/maps/*.html   (needs folium)
```
Total runtime is about 3 minutes.

## Where things go (matches the team folder layout)
| Content | Folder |
|---|---|
| Raw dataset (not committed) | `data/raw/train`, `data/raw/student_test` |
| Feature tables, labels, per-trip diagnostics | `data/processed/` (large tables are git-ignored; regenerate with `build_dataset.py`) |
| Feature/candidate configuration | `artifacts/configs/feature_config.json` |
| EDA tables and `data_quality_notes.md` | `experiments/eda/` |
| EDA figures and Folium maps | `visualizations/eda/` (`maps/` inside) |
| Pipeline code | `src/` (modules), `build_dataset.py`, `eda/` (scripts) |

## What each module does
| File | Purpose |
|---|---|
| `src/data_io.py` | Load one trip (`.track/.nodes/.arcs/.route`), list trips, save/load tables |
| `src/geo.py` | Per-trip local projection, haversine, point-to-segment distance, bearings |
| `src/preprocess.py` | Track cleaning; never drops valid rows; keeps `obs_idx` |
| `src/network.py` | Arc geometry, lengths, bearings, in/out degrees, duplicate flags |
| `src/candidates.py` | KD-tree candidate generation with exact distance filter and fallback radius |
| `src/labels.py` | Reference route to per-GPS-point labels by monotone DP alignment (training/evaluation only) |
| `src/features.py` | One row per (GPS observation, candidate arc) with 25 features; `build_trip_tables()` |
| `src/viz.py` | `make_trip_map()` Folium map (reused by Members 3 and 4) |
| `src/config.py` | R = 50 m, K = 15, fallback 150 m, etc. |

## Hand-off to the team
**Member 2 (models):** read `data/processed/features_<partition>`. Group CV on `trip_id`. Use `FEATURE_COLS` from
`src/features.py` as inputs and `y` as the target. Train and evaluate on rows with `label_valid == 1` only (3 train trips
have incomplete reference routes). Fit any scaler inside each training fold. Baseline to beat: "nearest arc" top-1
accuracy is about 0.53 on valid train points (`points_<partition>.nearest_is_label`).

**Member 3 (HMM):** each candidate row is a hidden-state option for observation `obs_idx`; `arc_id` is the raw arc id.
Candidate sets are capped at 15 and a few points have only fallback candidates (`is_fallback`). Road networks are
disconnected (89 of 90 trips); reference routes always stay within one weak component, so transitions across
components should be impossible or heavily penalised.

**Member 4 (API):** at inference call
```python
trip = load_trip(partition, trip_id, load_route=False)      # route must never be loaded in the student pipeline
cand_df, point_df, meta = build_trip_tables(trip, cfg, with_labels=False)
```
and keep the same `cfg` (saved in `artifacts/configs/feature_config.json`). `make_trip_map(trip, out_html, predicted_arcs=...)`
renders the result map.

## Known limitations (state these in the report)
- Reference routes of train trips 00000002, 00000033, 00000054 are incomplete (label_valid is low). Student-test labels are 99.9% valid.
- Labels are derived by alignment, so label noise exists; the candidate miss rate on reliable points is about 2% at R = 50 m, K = 15.
- Local equirectangular projection is accurate for the few-km search scale, not for very long trips.
