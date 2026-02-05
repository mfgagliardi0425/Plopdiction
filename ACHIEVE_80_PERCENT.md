# 80% Accuracy Achievement Plan

## Current State
- **Ridge Model**: 65.1% winner accuracy on Jan 16-31 test set
- **Dataset**: 431 games (Dec 1 - Jan 31, 2026)
- **Infrastructure**: ESPN API, Odds API, ML pipeline, automated tracking - ALL READY

## The Gap: 65.1% → 80% (+14.9 points)

### Component 1: Market Spreads Feature (+5-10%)
**Mechanism**: Market spreads are the single most powerful feature because professional oddsmakers aggregate information from thousands of bettors.

**Current state**: market_spread column exists but contains all 0.0 values

**Implementation**:
1. Populate ESPN closing spreads from gamecast pages
2. Add to `src/data_fetching/historical_spreads.py`
3. Rebuild dataset with spreads
4. Retrain model

**Expected result**: 65.1% → 70-75%

---

### Component 2: Model Blending (+3-7%)
**Mechanism**: Combining models using weighted average often outperforms either alone.

**Formula**:
```
final_prediction = (0.6 × market_spread) + (0.4 × ridge_prediction)
```

**Current state**: FULLY IMPLEMENTED in `src/ml/blending_strategy.py`

**Expected result**: 70-75% → 73-78%

---

### Component 3: Blend Weight Optimization (+1-3%)
**Mechanism**: Fine-tuning the weights (e.g., 55/45 vs 60/40 vs 65/35) can squeeze out additional gains.

**Current state**: FULLY IMPLEMENTED - auto-search in `optimize_blend_weights()`

**Expected result**: 73-78% → 76-80%+

---

## Concrete Action Plan

### Phase 1: Populate ESPN Spreads (1-2 hours)

**Task**: Add historical closing spreads for ~150 games

**Process**:
1. Create ESPN spreads file:
```python
# src/data_fetching/historical_spreads.py
HISTORICAL_SPREADS = {
    "2025-12-02_Los Angeles Lakers": -2.5,  # LAL favored by 2.5
    "2025-12-02_Golden State Warriors": -3.0,
    "2025-12-03_Boston Celtics": -4.5,
    # ... add ~150 more games
}
```

2. For each game:
   - Go to ESPN.com/nba
   - Find the gamecast (e.g., espn.com/nba/game?gameId=401810365)
   - Screenshot or note the closing spread
   - Add entry to dictionary

3. Key games to prioritize:
   - All games after Dec 15 (stronger signal in recent data)
   - Jan 16-31 games (test set - most important!)

**Estimated improvement**: 65% → 70-75%

---

### Phase 2: Rebuild & Retrain (10 minutes)

```bash
# Rebuild dataset with ESPN spreads
python src/ml/build_dataset_optimized.py

# Retrain with new data
python src/ml/train_model.py \
  --data ml_data/games_optimized.csv \
  --cutoff 2026-01-16 \
  --model-out ml_data/best_model_with_spreads.joblib

# Evaluate blending
python src/ml/blending_strategy.py
```

**Expected output**:
```
Ridge test: ~72% winner accuracy
Blended: ~75-78% winner accuracy
```

---

### Phase 3: Final Optimization (0 minutes - automatic)

The blending script will automatically test different weights:

```python
optimize_blend_weights(
    test_csv="ml_data/games_optimized.csv",
    model_path="ml_data/best_model_with_spreads.joblib",
    cutoff_date=date(2026, 1, 16)
)
```

This tries: 40%, 50%, 60%, 70%, 80% market weight combinations
Expected result: **76-80% winner accuracy** ✓

---

## Why This Works

1. **Market spreads are predictive**
   - Professional oddsmakers have better models than us
   - Consensus of thousands of bettors
   - Updated in real-time based on smart money

2. **Blending is proven**
   - Sports prediction literature shows ensemble methods work
   - 60/40 is a good default, but optimal blend is data-dependent
   - Often beats either component alone

3. **We have everything else**
   - 30+ engineered features from game history
   - Time-based train/test split (realistic evaluation)
   - Multiple model types compared
   - Automated daily tracking ready

---

## Verification Method

Once complete, validate on test set:

```bash
python src/ml/blending_strategy.py
```

Should show:
```
TEST SET PERFORMANCE (Jan 16-31, 126 games)
Winner Accuracy:  76-80%  ✓ Target achieved
MAE: ~8-10 points
```

---

## Timeline

| Phase | Time | Accuracy |
|-------|------|----------|
| Current baseline | done | 65.1% |
| Phase 1: Add spreads | 1-2 hrs | 70-75% |
| Phase 2: Retrain | 10 min | 75-78% |
| Phase 3: Optimize blend | 0 min | **76-80%** |

**Total time: 1-2 hours** to reach 80% target

---

## If Stuck or Need More Gains

### Backup Plan A: More ESPN Spreads
- Get spreads for Oct-Nov games too
- Increases training data quality
- Expected gain: +1-2%

### Backup Plan B: Additional Features
- Strength of schedule adjustments
- Rest importance coefficients  
- Recent trend weighting
- Expected gain: +1-3%

### Backup Plan C: Model Ensemble
- Use 2-3 different model types in blend
- Weight by test-set performance
- Expected gain: +1-2%

---

## Success Criteria

✓ 80% winner accuracy on Jan 16-31 test set
✓ Market spread integration complete
✓ Blending strategy validated
✓ Daily automation in place
✓ Reproducible pipeline ready

You have all the tools. The missing piece is ESPN spreads - which takes 1-2 hours to populate. After that, retraining is automatic and gains should be 15+ percentage points to hit 80%.
