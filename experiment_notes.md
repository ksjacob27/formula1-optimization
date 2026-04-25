# Formula 1 Pit Stop Optimization — Experiment Notes

## Project Goal

Apply deep learning to simulate F1 tire degradation and recommend optimal pit stop windows. The model predicts future lap times based on recent lap history, then compares "stay out on worn tyres" vs "pit for fresh tyres" to identify the lap where pitting first becomes net beneficial.

The broader motivation: F1 pit stop strategy is one of the highest-leverage decisions in a race. Teams use proprietary simulation software to make these calls. This project attempts to replicate a simplified version of that capability using publicly available data and sequence modeling.

### Success Criteria
- **Primary:** >10% MAE improvement over moving average baseline on 2024 test races
- **Secondary:** ≥70% of pit window recommendations within ±2 laps of actual pit stop

---

## Data Pipeline

### Source
FastF1 (Python library, v3.8.2). Wraps the official Formula 1 Livetiming API and provides cleaned lap-by-lap data including timing, tyre compounds, driver info, and weather. HTTP cache stored in `cache/fastf1_http_cache.sqlite` — essential for development since re-fetching a full season takes 30–60 minutes.

### Seasons Fetched
| Season | Laps Saved | Rounds | Usage |
|--------|-----------|--------|-------|
| 2022 | 16,650 | 22 (full season) | Training |
| 2023 | 19,590 | 22 (full season) | Training |
| 2024 | 4,183 | 1–5 only | Test set (held out entirely) |

2024 is capped at round 5 (Chinese GP) because that was the most recent data available when the project started.

### Data Loading Per Race
For each race:
1. `fastf1.get_session(year, round, 'R')` — loads the race session
2. `session.load(telemetry=False, weather=True, messages=False)` — fetches lap + weather data (telemetry skipped to save time/space)
3. `session.laps.pick_quicklaps()` — removes outlier laps (explained below)
4. Filter to dry compounds: SOFT, MEDIUM, HARD only
5. Merge nearest weather reading onto each lap via `pd.merge_asof` on lap start time

### What pick_quicklaps() Does
FastF1's `pick_quicklaps()` removes laps that are statistical outliers — typically:
- The pit entry lap (in-lap): driver lifts off early, lap time is slow
- The pit exit lap (out-lap): driver on cold tyres, lap time is slow
- Laps behind safety car or virtual safety car
- Formation laps, first lap of the race

**Important consequence for this project:** After `pick_quicklaps()`, the first observable lap after a pit stop has TyreLife=2 or TyreLife=3 (not TyreLife=1), because TyreLife=1 (the out-lap) was filtered out. This caused a bug in pit stop detection that was fixed later (see Bugs section).

**Used in Runs 1–2.** For Run 3 and onwards, `pick_quicklaps()` was replaced with explicit filtering (see Run 3 notes).

### Run 3 Filtering Change
In Run 3, `pick_quicklaps()` was replaced with three explicit filters to preserve the slow degraded laps at the end of stints — the hypothesis being that this signal is critical for the pit window optimizer:

```python
# pick_quicklaps() — removes outlier laps including slow end-of-stint laps (used in Runs 1–2)
# laps = session.laps.pick_quicklaps()

# Explicit filtering (Run 3+): keeps slow degraded laps; only removes known artifacts
laps = session.laps
laps = laps[laps['IsAccurate'] == True]     # valid timing data only
laps = laps[laps['TrackStatus'] == '1']     # entirely green-flag (filters SC/VSC/yellow/red)
laps = laps[laps['PitInTime'].isna()]       # drop pit-in laps (driver lifts to enter pit)
```

This change was **also applied in `strategy/pit_optimizer.py`'s `prepare_race()`** so that the strategy evaluation uses the same data distribution as training.

The same change removed out-laps (TyreLife=1) implicitly: PitInTime.isna() drops the in-lap; FastF1 doesn't record an out-lap row with PitInTime filled, but the very first lap of a new stint (TyreLife=1) often has TrackStatus != '1' because the pit lane has no TrackStatus — it varies. In practice the out-lap appears to still be present in the filtered data.

### Feature Engineering
| Feature | Description | How Computed |
|---------|-------------|-------------|
| LapTimeSeconds | Lap time in seconds | `LapTime.dt.total_seconds()` |
| StintLength | Laps on current tyre set | Directly from FastF1 `TyreLife` column |
| FuelLoad | Estimated fuel remaining (kg) | `110 - LapNumber * 1.6`, clipped at 0 |
| AirTemp | Ambient air temperature (°C) | From nearest weather reading |
| TrackTemp | Track surface temperature (°C) | From nearest weather reading |
| Compound_SOFT | 1 if on SOFT tyres | One-hot from `Compound` column |
| Compound_MEDIUM | 1 if on MEDIUM tyres | One-hot from `Compound` column |
| Compound_HARD | 1 if on HARD tyres | One-hot from `Compound` column |

**Total input features: 8. Target: LapTimeSeconds of the next lap.**

Fuel load is estimated rather than measured. Real F1 teams know the exact fuel load; we estimate it using a fixed burn rate of 1.6 kg/lap and starting weight of 110 kg. This is a simplification — real burn rate varies with throttle application.

### Normalization
`MinMaxScaler` from scikit-learn applied **per race** for each continuous feature: LapTimeSeconds, StintLength, FuelLoad, AirTemp, TrackTemp. Binary compound columns are left as 0/1.

**Why per-race (not global)?** Different circuits have vastly different baseline lap times (Monaco ~75s vs Monza ~83s vs Bahrain ~97s). Normalizing globally would mean the model sees unnormalized differences between circuits as signal, which it should not. Per-race normalization forces the model to learn *relative* degradation patterns within a race, which generalize better.

**Limitation:** Scalers are fitted on the training race and applied to both train and test sequences from that race. For inference, scalers fitted on the test race data itself are used — this is technically a mild form of look-ahead but is unavoidable given the per-race normalization design.

### Sequence Building
- **Window length:** `SEQUENCE_LENGTH = 10` laps
- **Method:** Sliding window. For a stint of N laps, produces N-10 samples.
- **Grouping:** Sequences are built per driver per stint. A stint boundary is detected when TyreLife decreases between consecutive laps (indicating a pit stop occurred).
- **Minimum stint length:** Stints ≤10 laps produce no sequences and are discarded.
- **Sample format:** X shape `(10, 8)` — 10 consecutive laps, 8 features each. y shape `(1,)` — the LapTimeSeconds value on lap 11 (the next lap after the window).

### Train/Test Data Sizes
| Version | Training Data | Training Sequences | Notes |
|---------|--------------|-------------------|-------|
| v1 | 2023 only (pick_quicklaps) | ~14,000 | Runs 1 |
| v2 | 2022 + 2023 (pick_quicklaps) | 28,115 | Runs 2 |
| v3 | 2022 + 2023 (TrackStatus='1') | 28,981 | Run 3 — test distribution also changed |

Doubling the training set by adding 2022 data significantly changed which model architectures worked (see Experiment Results). The v3 filtering change produced slightly more training sequences overall but a different test distribution — Run 3 results cannot be directly compared to Runs 1–2.

---

## Model Architectures

### Shared Training Config (BASE CONFIG)
| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 32 | |
| Max epochs | 50 | |
| Learning rate | 1e-3 | Adam optimizer |
| Optimizer | Adam | |
| LR Scheduler | ReduceLROnPlateau | patience=3, factor=0.5 |
| Gradient clipping | max_norm=1.0 | Clips gradient norm to prevent explosion |
| Early stopping patience | 7 | On validation loss |
| Val split | 10% | Random split of training set each run |
| Loss function | MSELoss | |
| Device | Apple MPS (M-series) | Falls back to CPU if unavailable |

The LR scheduler halves the learning rate when val loss doesn't improve for 3 epochs. Combined with early stopping at patience=7, this typically produces natural convergence. The 1.0 gradient clip is important for LSTM stability — LSTMs are prone to exploding gradients when sequences exhibit long-range dependencies.

### TireLSTM
Standard 2-layer LSTM. Takes only the **last timestep** output — the intuition being that the final hidden state summarizes all prior sequence information.

```
Input: (batch, 10, 8)
LSTM(input_size=8, hidden_size=64, num_layers=2, dropout=0.2, batch_first=True)
  → output: (batch, 10, 64)
Take last timestep: output[:, -1, :]  → (batch, 64)
Dropout(0.2)
Linear(64, 1)
  → output: (batch,)  — scalar prediction
```

Total parameters (hidden=64, layers=2): ~66,000

### TireGRU
Structurally identical to TireLSTM but uses GRU cells instead of LSTM cells.

```
Input: (batch, 10, 8)
GRU(input_size=8, hidden_size=64, num_layers=2, dropout=0.2, batch_first=True)
  → output: (batch, 10, 64)
Take last timestep: output[:, -1, :]  → (batch, 64)
Dropout(0.2)
Linear(64, 1)
  → output: (batch,)
```

**Why GRU vs LSTM?**
- LSTM has 3 gates (input, forget, output) + cell state = 4 weight matrices per layer
- GRU has 2 gates (reset, update) = 3 weight matrices per layer
- Fewer parameters → less overfitting risk on moderate-sized datasets
- GRU converged in ~21 epochs vs LSTM's ~30–34 epochs in our experiments
- For short sequences (10 laps), the cell state memory advantage of LSTM is less relevant
- Empirically, GRU consistently outperformed all LSTM variants in our experiments

Total parameters (hidden=64, layers=2): ~50,000

### TireLSTMAttention
LSTM with additive (Bahdanau-style) attention over all 10 timesteps. Instead of discarding the 9 intermediate hidden states, the attention mechanism learns a weighted combination of all of them.

```
Input: (batch, 10, 8)
LSTM(input_size=8, hidden_size=64, num_layers=2, dropout=0.2, batch_first=True)
  → out: (batch, 10, 64)
Linear(64, 1) applied to each timestep → scores: (batch, 10, 1)
Softmax over dim=1 → weights: (batch, 10, 1)   [sum to 1 across timesteps]
Weighted sum: (weights * out).sum(dim=1) → context: (batch, 64)
Dropout(0.2)
Linear(64, 1)
  → output: (batch,)
```

**Motivation:** In a 10-lap window, some laps may contain more degradation signal than others (e.g., lap 8 of a stint shows clearer wear than lap 2). Attention allows the model to up-weight those informative timesteps.

**Result:** Performed worst of all variants (MAE 0.1173, -22.5% vs baseline). Analysis: with only 10 timesteps, the variation across hidden states is too small for the attention layer to find a meaningful signal. The additional parameters overfit. Attention mechanisms typically benefit sequences of 20+ timesteps.

Total parameters (hidden=64, layers=2): ~67,000 (attention layer adds ~65 params — almost identical to base LSTM but with fundamentally different pooling)

---

## Experiment Results

### Moving Average Baseline
Predicts next lap time = mean of the 10 laps in the input window. Uses feature index 0 (LapTimeSeconds) directly from the input tensor. No learned parameters.

**Baseline MAE: 0.095744** — all improvement percentages are relative to this value.

This is a meaningful baseline because lap times within a stint don't vary wildly — the average of the last 10 laps is a reasonable guess for the next lap. The model must learn degradation trends (laps getting slower as tyres wear) to beat it.

---

### Run 1 — Training data: 2023 only (~14k sequences)

First systematic multi-experiment run. Four LSTM variants + GRU baseline.

#### Full Results Table
| Run | hidden | layers | dropout | Best Val Loss | Test MAE | Test MSE | Stopped Epoch | vs Baseline |
|-----|--------|--------|---------|--------------|----------|----------|---------------|-------------|
| lstm_baseline | 64 | 2 | 0.2 | 0.009238 | 0.091396 | 0.015167 | 34 | +4.5% |
| lstm_large | 128 | 2 | 0.2 | 0.009403 | 0.115789 | 0.021299 | 32 | -20.9% |
| lstm_deep | 64 | 3 | 0.3 | 0.008649 | 0.101632 | 0.017809 | 30 | -6.2% |
| lstm_high_dropout | 64 | 2 | 0.4 | 0.009107 | 0.065553 | 0.009645 | 31 | **+31.5%** |
| **gru_baseline** | **64** | **2** | **0.2** | **0.007723** | **0.057515** | **0.009226** | **21** | **+39.9%** |

#### Key Observations — Run 1
- **lstm_baseline (+4.5%):** Barely beats the naive baseline. The model is learning something but not much. Val loss converged around 0.0092.
- **lstm_large (-20.9%):** Bigger model = worse results. Hidden size 128 doubled the parameter count and overfit on the ~14k training samples. Val loss was still decreasing slowly when early stopping triggered at epoch 32 — a sign of slow, noisy convergence rather than clean learning.
- **lstm_deep (-6.2%):** 3-layer LSTM with dropout 0.3. Deeper architecture overfit similarly to the wider one. Best val loss was 0.00865 but test MAE was 0.1016 — significant gap between val and test performance indicates overfitting.
- **lstm_high_dropout (+31.5%):** Same architecture as baseline LSTM but dropout 0.4 instead of 0.2. Dramatic improvement. Confirms that regularization is the main bottleneck, not capacity. Best result among all LSTM variants in this run.
- **gru_baseline (+39.9%):** Outperformed all LSTMs cleanly. Notably stopped at epoch 21 — fastest convergence of any model. Best val loss 0.0077 was the lowest of all models. GRU's simpler gating generalizes better to this 10-lap sequence structure.

#### Run 1 Checkpoint (saved as `checkpoint_results/gru_v1/`)
The GRU from this run (MAE=0.0575) was saved as a permanent checkpoint before the data expansion. This represents the best result on 2023-only training data.

---

### Run 2 — Training data: 2022+2023 (28k sequences) + attention variant

Doubled training data and added the attention architecture. Results were mixed — some models improved significantly, others got worse (training variance).

#### Full Results Table
| Run | hidden | layers | dropout | Best Val Loss | Test MAE | Test MSE | Stopped Epoch | vs Baseline |
|-----|--------|--------|---------|--------------|----------|----------|---------------|-------------|
| lstm_baseline | 64 | 2 | 0.2 | 0.008418 | 0.104395 | 0.018229 | 32 | -9.0% |
| lstm_large | 128 | 2 | 0.2 | 0.009653 | 0.109246 | 0.019193 | 33 | -14.1% |
| lstm_deep | 64 | 3 | 0.3 | 0.009494 | 0.081883 | 0.013090 | 31 | **+14.5%** |
| lstm_high_dropout | 64 | 2 | 0.4 | 0.009600 | 0.069796 | 0.010266 | 33 | +27.1% |
| lstm_attention | 64 | 2 | 0.2 | 0.008947 | 0.117315 | 0.021287 | 28 | -22.5% |
| **gru_baseline** | **64** | **2** | **0.2** | **0.008665** | **0.058888** | **0.009088** | **24** | **+38.5%** |

#### Key Observations — Run 2
- **lstm_baseline (-9.0%):** Got worse despite more data. Val loss was 0.0084 (better than Run 1's 0.0092) but test MAE increased to 0.1044. This suggests the model is fitting some circuit-specific pattern in the training data that doesn't transfer to 2024 test races. Also significant training variance — results are sensitive to random weight initialization and val split.
- **lstm_large (-14.1%):** Still overfitting, still worse than baseline. More data helped slightly (Run 1: -20.9% → Run 2: -14.1%) but not enough to recover. The 128-unit hidden layer has too many parameters for this task.
- **lstm_deep (+14.5%):** The most dramatic improvement across runs. Went from -6.2% (Run 1) to +14.5% (Run 2). The 3-layer architecture was starved for data — once the training set doubled, it could learn meaningful depth. This is the clearest evidence that data quantity was the main limiting factor for deeper LSTMs.
- **lstm_high_dropout (+27.1%):** Slightly worse than Run 1's +31.5%, but still strong. Consistent performer regardless of dataset size.
- **lstm_attention (-22.5%):** Worst performer in either run. The attention layer introduces ~65 extra parameters and an additional non-linearity, but with only 10 timesteps the attention weights collapse toward uniform — there isn't enough variation across hidden states to learn a discriminative weighting.
- **gru_baseline (+38.5%):** Slightly lower than Run 1's +39.9% (within normal training variance). Stopped at epoch 24, still one of the fastest to converge. Remains the clear winner.

#### Cross-Run Comparison (same architecture, different data)
| Run | lstm_baseline | lstm_deep | lstm_high_dropout | gru_baseline |
|-----|--------------|-----------|-------------------|--------------|
| 2023 only | +4.5% | -6.2% | +31.5% | +39.9% |
| 2022+2023 | -9.0% | **+14.5%** | +27.1% | +38.5% |

The `lstm_deep` result is the most informative: doubling the data turned a negative result (+14.5% improvement) into a positive one. This directly supports the argument that deeper LSTMs can work but are data-hungry.

---

### Run 3 — Safety car filtering (TrackStatus='1') on 2022+2023 data

**Hypothesis:** `pick_quicklaps()` was stripping the slow, degraded laps at the end of stints — the very signal the optimizer needs to detect that a driver should pit. Replacing it with explicit TrackStatus/IsAccurate/PitInTime filtering would preserve those laps, reduce the systematic late-prediction bias, and improve strategy accuracy.

**What changed:** `pick_quicklaps()` replaced by three explicit filters in both `data/pipeline.py` and `strategy/pit_optimizer.py`. Data was re-fetched and re-preprocessed.

**New data sizes after re-fetch:**
| Season | Laps | Change |
|--------|------|--------|
| 2022 | 17,163 | +513 vs Run 2 |
| 2023 | 20,004 | +414 vs Run 2 |
| 2024 (rounds 1–5) | 4,168 | ~same |
| Training sequences | 28,981 | +866 vs Run 2 |

**Moving average baseline MAE: 0.060526** (down from 0.095744 in Runs 1–2). The test set changed because 2024 laps are now filtered by TrackStatus='1' too — fewer safety car laps means the remaining laps are closer to typical race pace, reducing baseline variance. Improvement percentages below are relative to the new 0.060526 baseline and **cannot be directly compared** to Run 1/2 percentages.

#### Full Results Table
| Run | hidden | layers | dropout | Test MAE | Test MSE | vs Baseline (new) |
|-----|--------|--------|---------|----------|----------|-------------------|
| lstm_baseline | 64 | 2 | 0.2 | 0.111593 | — | -84.4% |
| lstm_large | 128 | 2 | 0.2 | 0.084005 | — | -38.8% |
| lstm_deep | 64 | 3 | 0.3 | 0.094183 | — | -55.6% |
| lstm_high_dropout | 64 | 2 | 0.4 | 0.046429 | — | +23.3% |
| lstm_attention | 64 | 2 | 0.2 | 0.105996 | — | -75.1% |
| **gru_baseline** | **64** | **2** | **0.2** | **0.036339** | **0.004397** | **+40.0%** |

#### Key Observations — Run 3
- **Nearly all LSTM variants degraded sharply** (negative improvement percentages). The new data distribution is harder to learn from — preserving end-of-stint slow laps adds high-variance samples that LSTM struggles to model.
- **lstm_high_dropout (+23.3%):** The one LSTM that remained positive, again vindicating the importance of regularization. High dropout prevents overfitting to the noisy slow-lap tail.
- **gru_baseline (+40.0%):** GRU remained strong and matched its Run 1–2 performance almost exactly (~+39–40% across all three runs). Its simpler gating is robust to the changed data distribution.
- **The baseline itself changed (0.060526 vs 0.095744):** This makes cross-run comparisons unreliable in percentage terms. In absolute MAE, GRU improved: 0.0589 (Run 2) → 0.0363 (Run 3). But this partly reflects an easier test set, not purely a better model.

#### Strategy Evaluation — Run 3 vs Run 2

| Round | Circuit | Driver | Run 2 Predicted | Run 3 Predicted | Actual | Run 2 Error | Run 3 Error |
|-------|---------|--------|-----------------|-----------------|--------|-------------|-------------|
| R1 | Bahrain | VER | 35 | 32 | 19 / 39 | 4 ✗ | 7 ✗ |
| R2 | Saudi Arabia | VER | 13 | 13 | 10 | 3 ✗ | 3 ✗ |
| R3 | Australia | NOR | 35 | 35 | 16 / 42 | 7 ✗ | 7 ✗ |
| R4 | Japan | VER | 16 ✓ | 31 | 18 / 36 | 2 ✓ | 5 ✗ |
| R5 | China | VER | 20 | 18 | 15 | 5 ✗ | 3 ✗ |

**Run 3 strategy accuracy: 0% (0/5) — worse than Run 2's 20% (1/5).**

The one correct prediction (R4 Japan) regressed from error 2 → error 5. The hypothesis was not validated: preserving slow end-of-stint laps did not fix the late-prediction bias; it made it worse for most races.

#### Post-Experiment Analysis
- **Why did strategy accuracy drop?** The filtering change affects both training data and the strategy evaluation data. With TrackStatus='1' applied in `prepare_race()`, the strategy optimizer sees a different slice of each 2024 race — SC laps are now absent, so the stint context the model seeds from is different. The model was also never trained on slow degraded laps before, so seeing them in inference created distribution mismatch.
- **The raw MAE improvement is real but misleading** — the test set is simply a different (easier, more uniform) distribution. A model trained on this data compares favorably to a moving average on the same data, but that doesn't translate to better strategy timing.
- **Finding for the report:** This is a useful negative result. Trying to preserve tire degradation signal by changing the data pipeline backfired because: (1) the test distribution also changed, invalidating direct comparison; (2) LSTM models are less robust to the added noise; (3) strategy evaluation quality is the true downstream metric, and it declined.
- **Recommendation:** Revert or document this as experimental. Run 2 results (20% strategy accuracy, GRU MAE=0.0589) remain the best production-level results.

---

### Report Narrative Arc (Suggested Structure)

| Step | Experiment | Result | Key Takeaway for Report |
|------|-----------|--------|------------------------|
| 1 | Moving average baseline | MAE 0.0957 | Establishes the difficulty of the task |
| 2 | LSTM baseline (hidden=64) | +4.5% | Standard approach — marginal improvement |
| 3 | LSTM large (hidden=128) | -20.9% | Demonstrated overfitting — capacity is not the bottleneck |
| 4 | LSTM deep (3 layers) | -6.2% → +14.5% | Data quantity was the bottleneck for deeper models |
| 5 | LSTM high dropout | +31.5% | Regularization was the right direction |
| 6 | LSTM + attention | -22.5% | Architectural innovation — failed at short sequence lengths |
| 7 | GRU baseline | +39.9% / +38.5% | Simpler gating outperforms all LSTM variants |
| 8 | More training data | Mixed | Helped deeper models; GRU already strong |
| 9 | Safety car lap filtering (Run 3) | GRU MAE +40%, strategy 0% | Negative result: changing data pipeline hurt strategy accuracy despite lower MAE |
| 10 | Eval expansion + per-circuit pit loss | 9.5% on 42 evaluations | Revealed original 20% was a lucky 5-race sample; mean error 8.0 laps |
| 11 | Delta prediction (planned) | TBD | Target ΔLapTime instead of absolute; hypothesis: fixes late-bias by modeling slope directly |

---

## Strategy Evaluation

### Overview
After training, the best model (GRU baseline) is used to evaluate pit stop strategy on the 5 held-out 2024 races. The optimizer simulates the race from each candidate pit lap forward and determines the optimal timing.

### Method in Detail

**Step 1 — Load race data:**
`prepare_race(year, round, driver)` fetches the race from FastF1 (using cache), applies the same compound filtering and weather merge as training data, and returns raw lap data for the specific driver.

**Step 2 — Normalize:**
`normalize_driver_race()` fits a fresh MinMaxScaler on the driver's race data (same per-race normalization approach as training).

**Step 3 — Simulate two futures per candidate pit lap:**

`predict_worn(model, laps_norm, start_lap, n_future, compound)`:
- Seeds from the last 10 observed laps
- Auto-regressively predicts `n_future` laps ahead
- TyreLife increments by 1 each step (tyres continue wearing)
- FuelLoad decreases by `1.6/110` (normalized) each step
- Compound one-hot held constant (same tyres)

`predict_fresh(model, laps_norm, start_lap, n_future, compound)`:
- Seeds from the earliest low-TyreLife laps of the target compound (TyreLife < 0.2 normalized)
- If insufficient fresh-tyre history exists, seeds from current laps but resets TyreLife to 0
- TyreLife increments from 0 each step
- FuelLoad decreases identically

**Step 4 — Compute delta:**
```
pit_total  = pit_loss_normalized + sum(predict_fresh predictions)
stay_total = sum(predict_worn predictions)
delta      = pit_total - stay_total
```
Where `pit_loss_normalized = 22.0s / lap_time_max_seconds`.

**Step 5 — Find pit windows:**
Scan delta values across all candidate pit laps. A pit window is a contiguous region where delta < 0 (pitting is faster than staying out). The **recommended pit lap is the first lap delta crosses below zero** (first crossover point), not the local minimum — strategically, you pit as soon as it becomes beneficial, not at the deepest point.

**Why not local minimum?** Waiting for the minimum means staying on worn tyres longer than necessary after the crossover. Every lap after the crossover where you haven't pitted is net time lost.

### Constants
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Pit loss | 22.0 seconds | Approximate time lost driving through pit lane + stop |
| Max prediction horizon | 20 laps | Limits computational cost; accuracy degrades beyond ~20 laps |
| Sequence length | 10 laps | Must match training configuration |
| Fuel burn rate | 1.6 kg/lap | Standard estimate used in training data too |

---

### Bugs Found and Fixed During Strategy Evaluation

#### Bug 1 — Pit Stop Detection (TyreLife Threshold)

**Problem:** Two races (R2, R3) returned "No actual pit laps found" and were skipped from accuracy evaluation.

**Root cause:** The original pit detection filter was:
```python
actual_pit_laps = laps_raw[(laps_raw['TyreLife'] <= 2) & (laps_raw['LapNumber'] > 3)]
```
`pick_quicklaps()` removes the out-lap (TyreLife=1) and sometimes the second lap too (TyreLife=2). After filtering, the first observable lap of a new stint had TyreLife=3, not ≤2. So the filter missed those pit stops entirely.

**Debug output that revealed the issue:**
Added temporary debug prints:
```
TyreLife values: [1.0, 2.0, 3.0, ... 43.0]   ← R2: TyreLife never resets in filtered data!
Min lap with TyreLife<=3: [1  2  3  10]        ← Lap 10 has TyreLife=3 (out-lap filtered)
```
For R2 (Saudi Arabia), TyreLife went continuously from 1 to 43 with no visible reset, because the in-lap and out-lap were both removed by `pick_quicklaps()`. Lap 10 was the first clean lap after the pit, with TyreLife=3.

**Fix:**
```python
# OLD — missed races where out-lap was filtered
actual_pit_laps = laps_raw[(laps_raw['TyreLife'] <= 2) & (laps_raw['LapNumber'] > 3)]

# NEW — detects TyreLife decrease, robust regardless of how many laps were filtered
_prev_tyre      = laps_raw['TyreLife'].shift(1)
actual_pit_laps = laps_raw[(laps_raw['TyreLife'] < _prev_tyre) & (laps_raw['LapNumber'] > 3)]
```
Same fix applied in both `evaluate_strategy_across_races()` and `plot_strategy()`.

#### Bug 2 — Pit Window Recommendation (Local Minimum vs First Crossover)

**Problem:** The green dashed lines on the strategy plots marked the deepest point of the delta curve (local minimum) as the "best" pit lap within each window.

**Why it was wrong:** The local minimum is the lap where the *benefit of having pitted* is maximized — but this doesn't account for the time already lost by staying out past the crossover. The strategically correct answer is the first lap where pitting becomes net-positive.

**Fix:**
```python
# OLD — local minimum within each window
best_lap = int(window_data.loc[window_data['delta'].idxmin(), 'pit_lap'])

# NEW — first crossover (start of the window)
best_lap = window_start
```
Note: the primary recommendation (`best_pit_lap`) was already using the first crossover correctly via `first_pit_lap`. Only the per-window `best` used for plot annotations was wrong.

---

### TyreLife Debug Output Per Race (Pre-Fix)
This output helped diagnose the detection issue. The "Min lap with TyreLife<=3" line shows which laps had fresh tyres — laps 1–3 are always the race start, subsequent entries are post-pit.

```
R1 VER: TyreLife values: [2.0 → 20.0]
        Min lap with TyreLife<=3: [19 20 39 40]
        → TyreLife <= 2 correctly detected pits at laps 19, 39

R2 VER: TyreLife values: [1.0 → 43.0]  (no reset visible!)
        Min lap with TyreLife<=3: [1 2 3 10]
        → Lap 10 has TyreLife=3 — pit around lap 7-8, both laps filtered
        → TyreLife <= 2 found NO pits after lap 3 → "No actual pit laps found"

R3 NOR: TyreLife values: [2.0 → 26.0]
        Min lap with TyreLife<=3: [2 3 16 42]
        → Laps 16 and 42 are post-pit, but TyreLife=3, not ≤2
        → TyreLife <= 2 found NO pits → "No actual pit laps found"

R4 VER: TyreLife values: [2.0 → 19.0]
        Min lap with TyreLife<=3: [18 19 36 37]
        → TyreLife <= 2 correctly detected pits at laps 18, 36

R5 VER: TyreLife values: [1.0 → 33.0]
        Min lap with TyreLife<=3: [1 2 3 15 16]
        → TyreLife <= 2 correctly detected pit at lap 15
```

---

### Eval Run 1 — Initial 5-race eval (GRU Baseline, 2024 Rounds 1–5)

> **Superseded by Eval Run 2.** This 5-race sample was statistically thin and happened to include R4 Japan — one of the model's strongest performances. The expanded 42-race eval (below) gives the authoritative accuracy figure.

Model loaded: `results/best_gru_baseline.pt` (trained on 2022+2023, MAE=0.0589)

#### Full Results
| Round | Circuit | Driver | Predicted Pit | Actual Pit(s) | Error | Within ±2 |
|-------|---------|--------|--------------|---------------|-------|-----------|
| R1 | Bahrain | VER | 35 | 19, 39 | 4 (vs 39) | ✗ |
| R2 | Saudi Arabia | VER | 13 | 10 | 3 | ✗ |
| R3 | Australia | NOR | 35 | 16, 42 | 7 (vs 42) | ✗ |
| R4 | Japan | VER | 16 | 18, 36 | 2 | ✓ |
| R5 | China | VER | 20 | 15 | 5 | ✗ |

**Overall accuracy (within ±2 laps): 20% (1 of 5 races)**
Secondary success criterion (≥70%): ✗ FAILED

#### Pit Window Detection Per Race

**R1 — Bahrain (VER, 57 laps, SOFT → HARD):**
```
Lap time range: 4.68s, Lap time max: 97.28s
Pit loss normalized: 0.2261
Pit windows detected: 2
  Window 1: laps 35–36 (best: lap 35)
  Window 2: laps 39–46 (best: lap 39)
Primary recommendation: lap 35
Actual pits: [19, 39], Error: 4 laps
```

**R2 — Saudi Arabia (VER, 50 laps, MEDIUM → HARD):**
```
Lap time range: 3.73s, Lap time max: 95.50s
Pit loss normalized: 0.2304
Pit windows detected: 1
  Window 1: laps 13–39 (best: lap 13)
Primary recommendation: lap 13
Actual pits: [10], Error: 3 laps
```

**R3 — Australia (NOR, 58 laps, MEDIUM → HARD):**
```
Lap time range: 3.27s, Lap time max: 83.18s
Pit loss normalized: 0.2645
Pit windows detected: 1
  Window 1: laps 35–47 (best: lap 35)
Primary recommendation: lap 35
Actual pits: [16, 42], Error: 7 laps (vs lap 42)
```

**R4 — Japan (VER, 53 laps, MEDIUM → HARD):**
```
Lap time range: 5.52s, Lap time max: 99.23s
Pit loss normalized: 0.2217
Pit windows detected: 2
  Window 1: laps 16–23 (best: lap 16)
  Window 2: laps 30–42 (best: lap 30)
Primary recommendation: lap 16
Actual pits: [18, 36], Error: 2 laps — ✓ CORRECT
```

**R5 — China (VER, 56 laps, MEDIUM → HARD):**
```
Lap time range: 3.12s, Lap time max: 101.53s
Pit loss normalized: 0.2167
Pit windows detected: 1
  Window 1: laps 20–35 (best: lap 20)
Primary recommendation: lap 20
Actual pits: [15], Error: 5 laps
```

#### Debug Delta Output at Pit Lap 20 (for reference — shows model internals)
```
R1 (VER, Bahrain):
  Stay preds (laps 20–24): [0.5755, 0.6113, 0.6396, 0.6565, 0.6667]
  Pit preds  (laps 20–24): [0.6200, 0.6413, 0.6556, 0.6631, 0.6672]
  Stay total (20 laps): 11.3915
  Pit total  (20 laps): 13.7398
  Delta: +2.3482  → staying out is faster at lap 20 (correct — VER actually stayed out until lap 19)

R2 (VER, Saudi):
  Stay preds (laps 20–24): [0.2916, 0.3203, 0.3453, 0.3627, 0.3758]
  Pit preds  (laps 20–24): [0.2923, 0.2919, 0.2901, 0.2865, 0.2822]
  Stay total: 8.1957
  Pit total:  5.1895
  Delta: -3.0062  → pitting is much faster at lap 20 (but VER already pitted at lap 10!)

R4 (VER, Japan):
  Stay preds (laps 20–24): [0.4684, 0.4919, 0.5123, 0.5213, 0.5287]
  Pit preds  (laps 20–24): [0.4809, 0.4769, 0.4736, 0.4716, 0.4710]
  Stay total: 9.8203
  Pit total:  9.3746
  Delta: -0.4457  → pitting is marginally faster at lap 20 (model correctly identified early pit window)
```

#### Systematic Error Analysis
The model consistently recommends pit windows **3–7 laps later than actual.** This is a directional bias, not random noise. The most likely causes:

1. **Conservative degradation curves:** The `predict_worn` function auto-regressively builds degradation — but if the model learned average degradation across compounds and circuits, it will underestimate aggressive early-stint wear on softer compounds or hotter tracks.
2. **Missing undercut dynamics:** The model has no knowledge of competitor positions or undercut/overcut strategy. Real teams often pit earlier than the "optimal" lap to take strategic track position. Our model can only see tire time, not racing position.
3. **Out-lap filtering effect:** `pick_quicklaps()` removes out-laps. The model never saw genuine "first lap on fresh tyres" data, so `predict_fresh` may underestimate how fast a car is on the very first lap of a new stint — making fresh tyres look slightly worse than they are.
4. **Per-race normalization inference issue:** During strategy evaluation, the scaler is fitted on all of the driver's laps for that race. In a real-time system, laps later in the race haven't happened yet — this is a mild lookahead that may slightly distort predictions.

#### Success Criteria Summary (Eval Run 1 — 5 races)
- ✅ **Primary PASSED** — GRU achieves ~39% MAE improvement over moving average baseline
- ❌ **Secondary FAILED** — 20% strategy accuracy (1/5 races within ±2 laps)

---

### Eval Run 2 — Expanded 42-race eval + per-circuit pit loss (2024, all rounds)

**Changes from Eval Run 1:**
1. **Per-circuit pit loss** — replaced the fixed `PIT_LOSS_SECONDS = 22.0` with a per-round dict (`PIT_LOSS_BY_ROUND_2024`) using published F1 strategy estimates. Key differences: Saudi Arabia 19s (short pit lane), Monaco 24s, Canada 17s, Imola 26s, Las Vegas 16s, Singapore 28s.
2. **Expanded from 5 → 42 evaluations** — all 24 rounds of 2024, VER and NOR as evaluation drivers. Rounds where a driver had no clean laps (e.g., Brazil R21 — wet race, TrackStatus filtering removed all laps) were skipped automatically.
3. **Plotting disabled by default** — `evaluate_strategy_across_races(..., plot=False)` to avoid 48 plot windows.

**Results: 9.5% accuracy (4/42 race-driver combinations within ±2 laps)**

| Round | Circuit | Driver | Predicted | Actual | Error | ✓ |
|-------|---------|--------|-----------|--------|-------|---|
| R1 | Bahrain | NOR | 13 | 15 | 2 | ✓ |
| R6 | Miami | NOR | 32 | 33 | 1 | ✓ |
| R8 | Monaco | VER | 55 | 54 | 1 | ✓ |
| R22 | Las Vegas | VER | 13 | 13 | **0** | ✓ |

**Stats across 42 evaluations:**
- Mean error: 8.0 laps
- Median error: 5.5 laps
- VER: 2/21 ✓ (9.5%), NOR: 2/21 ✓ (9.5%)

**Notable failures:**
| Round | Circuit | Driver | Error | Notes |
|-------|---------|--------|-------|-------|
| R2 | Saudi Arabia | VER | 19 | Model predicts very late (lap 29, actual lap 10) |
| R15 | Netherlands | VER | 23 | Zandvoort's tight layout; unusual degradation |
| R20 | Mexico | VER | 23 | Model predicts lap 51, actual lap 28 |
| R12 | Britain | Both | 15 | Silverstone high-speed wear not captured |
| R7 | Imola | NOR | 23 | Sprint weekend — race data distorted |

**Crashes (no clean laps found):**
- R3 VER (Australia) — VER retired, insufficient green-flag laps
- R9 VER (Canada) — similar issue
- R21 VER + NOR (Brazil) — wet race, TrackStatus='1' filtered all laps

**Key findings:**
1. **The original 20% (1/5) was optimistic** — based on a lucky sample. The true accuracy on 42 evaluations is 9.5%, a substantially more reliable estimate.
2. **Per-circuit pit loss helped on unusual circuits** — Monaco, Las Vegas, and Miami are all within ±2 laps. These are precisely the circuits where the 22s default would have been most wrong. But it couldn't compensate for the fundamental late-bias on standard circuits.
3. **Mean error jumped to 8.0 laps** — far worse than the 3–7 lap range observed in Eval Run 1. The expanded set exposed many more circuits where the model is catastrophically late.
4. **The late-bias is the dominant failure mode** — model consistently recommends staying out too long before pitting. Suggests `predict_worn` underestimates degradation slope, causing the delta crossover to arrive too late.
5. **Brazil R21 limitation** — TrackStatus='1' filtering removes all laps from wet/heavy-SC races. A known limitation of the filtering approach.

#### Updated Success Criteria (Eval Run 2 — 42 races)
- ✅ **Primary PASSED** — GRU achieves ~40% MAE improvement over moving average baseline
- ❌ **Secondary FAILED** — 9.5% strategy accuracy (4/42 race-driver combos within ±2 laps)

**Next experiment: delta prediction** — change the training target from absolute lap time to the change in lap time (`ΔLapTime = LapTime[t+1] − LapTime[t]`). Hypothesis: forces the model to explicitly learn degradation slope, which should reduce the late-prediction bias that drives the 8-lap mean error.

---

## Checkpoints

### `checkpoint_results/gru_v1/`
Frozen snapshot of the best GRU trained on **2023 data only**. Saved before the 2022 data was added so that the v1 results are permanently preserved.

| File | Description |
|------|-------------|
| best_gru.pt | GRU weights (hidden=64, layers=2, dropout=0.2, trained on 2023) |
| results_gru.json | Training metrics: MAE=0.0575, MSE=0.0092, best_val_loss=0.0077 |
| predictions_gru.png | Predicted vs actual lap times (first 200 test samples) |
| training_curves.png | Train/val loss curves for LSTM and GRU |
| mae_comparison.png | Bar chart comparing baseline MAE, LSTM MAE, GRU MAE |
| strategy_LEC_2024_r2.png | Strategy plot R2 (originally LEC, later changed to VER) |
| strategy_SAI_2024_r3.png | Strategy plot R3 (originally SAI, later changed to NOR) |
| strategy_VER_2024_r1.png | Strategy plot R1 VER |
| strategy_VER_2024_r4.png | Strategy plot R4 VER |
| strategy_VER_2024_r5.png | Strategy plot R5 VER |
| strategy_summary.csv | Per-race accuracy summary |

> **Note:** The gru_v1 checkpoint was made before two bugs were fixed (TyreLife detection and local minimum recommendation). The plots in this checkpoint reflect those bugs. MAE of 0.0575 was from 2023-only training; the current GRU trained on 2022+2023 achieves MAE=0.0589 (very similar due to training variance).

---

## Future Work / Improvements Not Yet Tried

| Idea | Status | Expected Impact | Rationale |
|------|--------|----------------|-----------|
| **Delta prediction (ΔLapTime target)** | **→ IN PROGRESS** | **High** | **Forces model to explicitly learn degradation slope. Addresses the 8-lap mean error late-bias. Changes target from absolute lap time to lap-to-lap change.** |
| ~~Safety car lap filtering~~ | DONE — negative | ~~Medium~~ | Tried in Run 3. Strategy accuracy dropped 20% → 0% on 5-race sample. Expanded 42-race eval confirmed 9.5% true accuracy. |
| ~~Per-circuit pit loss~~ | DONE — partial | Medium | Added `PIT_LOSS_BY_ROUND_2024` dict. Helped unusual-layout circuits (Monaco, Las Vegas, Miami all ✓). Didn't fix fundamental late-bias. |
| GRU with high dropout (0.4) | Pending | Medium | lstm_high_dropout showed +31.5%; never tried on GRU which is already the strongest architecture |
| Longer sequence length (15–20 laps) | Pending | Medium–High | More degradation context before prediction; also makes attention viable (failed at 10 steps) |
| Compound-specific degradation models | Pending | High | SOFT/MEDIUM/HARD have very different wear profiles; shared model averages across them, diluting degradation signal |
| Multi-stop strategy simulation | Pending | High | Current optimizer assumes one future pit; many races are 2-stop, optimizer currently ignores second stop |
| More training data (2021 and earlier) | Pending | Medium | lstm_deep proved more data helps; GRU likely improves too |
| AdamW optimizer + weight decay | Pending | Low–Medium | Better regularization than dropout alone; may stabilize LSTM variants |
| Competitor position as feature | Pending | High | Would allow model to account for undercut/overcut strategy — real teams pit early for track position |
| Track-specific models | Pending | Medium | Each circuit has different degradation characteristics; per-circuit tuning |
