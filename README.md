# NBA Sports Prediction Model

Predictive model for NBA game outcomes using historical data, team statistics, and betting market analysis.

## Project Structure

```
├── src/                          # Source code
│   ├── models/                   # Prediction models
│   │   ├── matchup_model.py      # Team history + weighted stats
│   │   ├── injury_adjustment.py  # PPG-weighted injury penalties
│   │   └── improved_matchup_model.py  # Legacy model
│   ├── evaluation/               # Model evaluation scripts
│   │   ├── tonight_spread_predictions.py      # Live predictions
│   │   ├── tonight_spread_predictions_summary.py # Summary + thresholds
│   │   ├── send_tonight_to_discord.py         # Discord push
│   │   ├── monitor_injury_updates.py          # Injury refresh + rerun
│   │   ├── update_espn_injuries.py            # Cache injuries
│   │   ├── print_today_injuries.py            # Print injuries
│   │   └── daily_pipeline.py                  # End-to-end daily run
│   ├── data_fetching/            # Data collection
│   │   ├── download_season_data.py    # SportRadar game data
│   │   ├── espn_spread_scraper.py     # ESPN closing lines
│   │   ├── espn_injuries.py           # ESPN injuries endpoint
│   │   ├── espn_player_ppg.py         # ESPN PPG via athlete IDs
│   │   ├── espn_player_ids.py         # ESPN player ID DB
│   │   ├── nba_player_stats.py        # stats.nba.com PPG fallback
│   │   └── odds_api.py                # The Odds API integration
│   ├── optimization/             # Parameter tuning
│   │   └── optimize_params.py         # Grid search optimization
│   └── utils/                    # Utility scripts
│       ├── team_win_pct.py            # Team statistics
│       └── probe_roster_endpoints.py  # API exploration
├── data/                         # Game data by date (YYYY-MM-DD/)
├── cache/                        # Cached API responses
│   ├── summary/                  # Game summary cache
│   └── odds/                     # Betting odds cache
├── tests/                        # Test files
├── .env                          # API keys (not in git)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys in `.env`:**
   ```
   SPORTRADAR_API_KEY=your_key_here
   ODDS_API_KEY=your_key_here
   ```

3. **Download historical data:**
   ```bash
   python src/data_fetching/download_season_data.py --start 2025-10-22 --end 2026-02-02 -y
   ```

## Usage

### Make Predictions for Today
```bash
python src/evaluation/tonight_spread_predictions_summary.py --date 2026-02-04

# Fast mode (uses cached injury dataset only)
python src/evaluation/tonight_spread_predictions_summary.py --date 2026-02-04 --fast
```

### Evaluate Model Performance
```bash
# Basic evaluation
python src/evaluation/evaluate_predictions.py --date 2026-02-02

# Compare vs market spreads
python src/evaluation/evaluate_market.py --date 2026-02-02
```

### Injury Dataset (Auto-Updating)
```bash
# Build/refresh the in-house injury dataset with player stats
python src/evaluation/monitor_injury_updates.py --force

# Rerun tonight's predictions only if a changed team plays today
python src/evaluation/monitor_injury_updates.py --force --rerun

# Fast mode: skip PPG refresh if injuries are unchanged
python src/evaluation/monitor_injury_updates.py --fast --rerun
```

### ESPN Player ID Database
```bash
# Build or refresh the ESPN player ID database (season year)
python src/evaluation/update_player_ids.py --season 2026
```

### Injury Report Channel
```bash
# Post injury updates to the injury report channel
python src/evaluation/send_injury_report.py --force
```

### Build Dataset + Train Model
```bash
# Build dataset with ESPN closing lines
python src/ml/build_dataset_optimized.py

# Train and select best model
python src/ml/train_model.py --data ml_data/games_optimized.csv --cutoff 2026-01-16 --model-out ml_data/best_model_with_spreads.joblib
```

## Model Features

## Algorithm Overview (Detailed)

1. **Data ingestion**
   - **Games/results**: SportRadar schedule + final scores (cached JSON in data/).
   - **Market spread**: Odds API (current) + ESPN closing lines (historical).
   - **Injuries**: ESPN injury endpoint (cached).
   - **Player PPG**: ESPN core API by athlete ID (cached), stats.nba.com as optional fallback.

2. **Dataset construction**
   - Build per-game feature rows with historical team stats and market spread.
   - Uses a half-life weighting to emphasize recent games.
   - Output: ml_data/games_optimized.csv

3. **Feature engineering**
   - Weighted margins, win %, points for/against.
   - Home/away weighted margins.
   - Recent window stats: margins, win %, points for/against (3/5/10 game windows).
   - Rest differential and back-to-back flags.
   - Games played counts.
   - Market spread.
   - Head-to-head summary: home-side H2H margin avg, win %, games played.

4. **Model training & selection**
   - Train multiple regressors (Ridge, GBR, RF) on pre-cutoff data.
   - Choose best on test set (lowest MAE, tracked ATS accuracy).
   - Model predicts **home margin**.

5. **Prediction flow**
   - Generate features for each scheduled game.
   - Predict home margin → convert to away margin:
     - `pred_away_margin = -pred_home_margin`

6. **Injury adjustment (post-model)**
   - Applies only to **OUT** players.
   - Penalty is proportional to PPG with rank/PPG-based multiplier.
   - Adjusted away margin:
     - `pred_away_adj = pred_away_margin - away_penalty + home_penalty`

7. **Edge calculation & thresholds**
   - `edge = pred_away_adj + market_spread`
   - Picks:
     - away if `edge >= threshold`
     - home if `edge <= -threshold`
   - Thresholds: 0 / 5 / 10

## Narrative Overview (Why this approach)

This model was built to answer a simple, practical question: when does our view of a game materially disagree with the betting market? We started with team results and margin history, then added recency weighting and home/away splits to reflect how teams actually evolve across a season. From there, we introduced rest and back‑to‑back indicators to capture fatigue effects that show up in real outcomes.

The biggest leap came from treating the market spread as a feature rather than a target. That decision aligns the model with how markets price games and lets us focus on where we believe the line is off. We then layered in injuries as a post‑model adjustment because lineup news is often the most time‑sensitive information and can shift expected margins quickly. By tying injury impact to player scoring (PPG) and weighting top scorers more, we get a consistent, interpretable adjustment without overfitting.

Our current algorithm predicts the **home margin**, converts to **away margin**, and calculates **edge** as the difference between our adjusted projection and the market line. Thresholds (0/5/10) are used to control bet volume and confidence. This is intentionally conservative: we prefer fewer, higher‑conviction edges and we treat injuries as a calibrated correction rather than letting them dominate the model.

Going forward, the model improves in three main ways: (1) data quality (more games, more complete spreads, and cleaner injury updates), (2) feature depth (opponent strength, lineup‑level impact, and potential pace/efficiency signals), and (3) evaluation discipline (tracking ATS performance over time and re‑tuning thresholds). As these improve, the same pipeline remains intact, but the inputs and calibration become more precise, which should tighten errors and stabilize our edge selection over the long run.

### Prediction Outputs
- **Line(A)**: market spread for away team
- **Pred(A)**: model’s away margin prediction
- **Adj(A)**: injury-adjusted away margin
- **Edge**: Pred(A) + Line(A)

### Evaluation Metrics
- **Winner accuracy**: Percentage of correct winner predictions
- **ATS (Against The Spread)**: Favorite coverage rate
- **Margin accuracy tiers**: Within 3, 5, 7 points
- **Edge betting**: Identify value when disagreeing with market by 3+ pts

## API Keys

### SportRadar API (Trial)
- 1,000 requests/month
- NBA schedule, play-by-play, team profiles
- Docs: https://developer.sportradar.com

### The Odds API (Starter)
- 500 requests/month  
- Live betting odds from multiple sportsbooks
- Docs: https://the-odds-api.com/

## Model Performance (Recent)

- Metrics are reported during training and nightly summaries.
- ATS accuracy and sample size are included per threshold.

## Caches & Data Files

- odds_cache/espn_injuries.json (ESPN injuries cache)
- odds_cache/espn_player_ppg.json (ESPN PPG cache)
- odds_cache/espn_closing_spreads*.json (ESPN closing lines)
- data/injuries/injury_dataset.json (injury + PPG dataset)
- data/players/espn_player_ids.json (player ID DB)

## Next Steps

- [ ] Add optional hard filters for mass injuries
- [ ] Add team/line movement alerts
- [ ] Improve matchup features with opponent strength

**GitHub Actions & Secrets**

To enable CI and scheduled fixture updates, add the following repository Secrets in GitHub (`Settings -> Secrets -> Actions`):

- `SPORTRADAR_API_KEY` — API key used for fetching PBP and other SportRadar endpoints. Required by `scripts/fetch_pbp_fixture.py` and optional live tests.
- `RUN_LIVE_TESTS` — Set to `1` if you want CI to run live API tests (not recommended for PRs). Leave unset or `0` for deterministic runs.
- `DISCORD_WEBHOOK_URL` — (Optional) webhook used by notification jobs if you add Discord notifications to workflows.

Once the secrets are added, GitHub Actions will run the CI on pushes/PRs and a scheduled job will update `tests/fixtures/sample_pbp.json` daily.

To manually run the fixture update on Actions UI, go to the repository `Actions` tab, choose **Update Fixtures**, and click **Run workflow**.

**Recent Implementation Changes (2026-02-05)**
- **Tests converted to deterministic suites:** Converted diagnostic scripts into pytest tests that use a local fixture by default to avoid network flakiness. See `tests/test_pbp_detail.py` and `tests/test_playbyplay.py` for the new tests.
- **Recorded fixture added:** A recorded play-by-play JSON fixture was added at `tests/fixtures/sample_pbp.json` (captured from SportRadar) so tests run offline and deterministically.
- **Fetch script:** Added `scripts/fetch_pbp_fixture.py` which fetches a live PBP from SportRadar and saves it to `tests/fixtures/sample_pbp.json` for updating fixtures when needed.
- **How to run tests:**
   - Run the suite locally (uses recorded fixture by default):
      ```powershell
      .venv\Scripts\python.exe -m pytest -q
      ```
   - To run tests against the live SportRadar API (not recommended in CI), set the env var and ensure `.env` contains `SPORTRADAR_API_KEY`:
      ```powershell
      $env:RUN_LIVE_TESTS = '1'
      .venv\Scripts\python.exe -m pytest -q
      ```
- **Result:** On my run the new tests pass: `2 passed`.

If you'd like, I can (A) add more recorded fixtures for varied games, (B) expand assertions in the converted tests, or (C) integrate `vcrpy` to automate recording during test runs.
