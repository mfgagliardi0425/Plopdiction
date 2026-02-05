# Repository Overview

This document explains how the project is structured, how data flows through the system, and how to run the key workflows (scraping, dataset building, training, and evaluation).

## 1) Purpose

This repository builds NBA game prediction models with a focus on **spread (ATS) performance**. It collects historical game data, scrapes closing spreads from ESPN, engineers features, trains ML models to predict game margins, and evaluates ATS accuracy. It also supports edge‑threshold filtering to increase ATS win rates by betting only when the model has sufficient edge over the market line.

## 2) High‑Level Architecture

**Data Sources**
- ESPN scoreboard pages → historical schedules and game links.
- ESPN game pages → closing spreads (away‑team line) and team names.
- Local JSON game data → raw game stats (from prior data ingestion).

**Core ML Flow**
1. **Scrape closing spreads** from ESPN game pages into cache.
2. **Build dataset** with engineered features + market spread.
3. **Train models** to predict game margin.
4. **Evaluate ATS** accuracy using the closing spread.
5. **Edge thresholding** to improve ATS accuracy by selecting only high‑edge games.

## 3) Key Directories and Files

### Data Fetching
- [src/data_fetching/espn_spread_scraper.py](src/data_fetching/espn_spread_scraper.py)
  - Scrapes ESPN for closing spreads.
  - Saves to:
    - [odds_cache/espn_closing_spreads.json](odds_cache/espn_closing_spreads.json) (simple date+home key)
    - [odds_cache/espn_closing_spreads_detailed.json](odds_cache/espn_closing_spreads_detailed.json) (full matchup)

### Feature/Model Pipeline
- [src/ml/build_dataset_optimized.py](src/ml/build_dataset_optimized.py)
  - Builds dataset from local game JSONs and ESPN spread cache.
  - Outputs: [ml_data/games_optimized.csv](ml_data/games_optimized.csv)

- [src/ml/train_model.py](src/ml/train_model.py)
  - Trains Ridge, GB, RF models on the dataset.
  - Includes ATS evaluation using `market_spread`.
  - Outputs: [ml_data/best_model_with_spreads.joblib](ml_data/best_model_with_spreads.joblib)

- [src/ml/edge_threshold_eval.py](src/ml/edge_threshold_eval.py)
  - Evaluates ATS accuracy for different edge thresholds.

### Caches and Data
- [odds_cache/espn_closing_spreads.json](odds_cache/espn_closing_spreads.json)
- [odds_cache/espn_closing_spreads_detailed.json](odds_cache/espn_closing_spreads_detailed.json)
- [ml_data/games_optimized.csv](ml_data/games_optimized.csv)
- [ml_data/best_model_with_spreads.joblib](ml_data/best_model_with_spreads.joblib)

## 4) Data Flow (Detailed)

### 4.1 Scrape ESPN Closing Spreads
The ESPN scraper reads each day’s scoreboard, extracts all game IDs, and then scrapes each game page for the odds block. The closing spread is pulled from the **away team line**. The result is stored in two caches:

- **Simple cache**: `YYYY-MM-DD_HOME_TEAM` → `closing_spread_away`
- **Detailed cache**: `YYYY-MM-DD_Away_@_Home` → structured JSON

Key file: [src/data_fetching/espn_spread_scraper.py](src/data_fetching/espn_spread_scraper.py)

### 4.2 Build Dataset
The dataset builder loads local game JSON data, computes team features, and joins the ESPN spread for each matchup via the **detailed cache**. The final CSV includes:

- Team rolling stats
- Rest/back‑to‑back indicators
- `market_spread` (away‑team line)
- `actual_margin` (home − away)

Key file: [src/ml/build_dataset_optimized.py](src/ml/build_dataset_optimized.py)

### 4.3 Train Model
The model predicts **game margin**. The ATS evaluation compares predicted away margin to the closing spread.

Key file: [src/ml/train_model.py](src/ml/train_model.py)

### 4.4 ATS Edge Thresholding
For ATS, we compute edge:

$$
\text{edge} = \hat{M}_{away} - \text{spread}
$$

- If edge ≥ threshold → bet away
- If edge ≤ −threshold → bet home

Key file: [src/ml/edge_threshold_eval.py](src/ml/edge_threshold_eval.py)

## 5) How to Run the Full Pipeline

### Step 1 — Scrape ESPN Closing Spreads
```
python src/data_fetching/espn_spread_scraper.py --start 2025-12-01 --end 2026-01-31 --delay 0.5
```

### Step 2 — Build Dataset
```
python src/ml/build_dataset_optimized.py
```

### Step 3 — Train Model (ATS + Winner Metrics)
```
python src/ml/train_model.py --data ml_data/games_optimized.csv --cutoff 2026-01-16 --model-out ml_data/best_model_with_spreads.joblib
```

### Step 4 — Edge Threshold Testing
```
python src/ml/edge_threshold_eval.py
```

## 6) Interpreting Results

- **Winner Accuracy**: % of games where predicted margin sign matches actual winner.
- **ATS Accuracy**: % of games where predicted outcome beats the closing spread.
- **Edge Threshold**: Higher thresholds = fewer bets, higher ATS accuracy.

Example (test split):
- Threshold 0 → 82 bets, 76.8% ATS
- Threshold 5 → 50 bets, 86.0% ATS
- Threshold 10 → 29 bets, 96.6% ATS

## 7) Common Pitfalls

- Missing spreads: ensure [odds_cache/espn_closing_spreads_detailed.json](odds_cache/espn_closing_spreads_detailed.json) is populated.
- If `market_spread` is mostly zeros in the dataset, the join failed. Rebuild after scraping.
- Team name mismatches: the detailed cache is matched by date + normalized team names.

## 8) Recommended Next Improvements

- Add more robust spread matching (team IDs or ESPN game IDs).
- Optimize model directly for ATS objective (classification on cover vs no cover).
- Add betting ROI and line movement analysis.
- Add confidence calibration and bankroll management.

## 9) Quick Reference

- Scraper: [src/data_fetching/espn_spread_scraper.py](src/data_fetching/espn_spread_scraper.py)
- Dataset: [src/ml/build_dataset_optimized.py](src/ml/build_dataset_optimized.py)
- Training: [src/ml/train_model.py](src/ml/train_model.py)
- Edge eval: [src/ml/edge_threshold_eval.py](src/ml/edge_threshold_eval.py)
