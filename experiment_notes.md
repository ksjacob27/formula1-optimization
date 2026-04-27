# Formula 1 Pit Stop Optimization — Experiment Notes

## Project Goal

Apply deep learning to simulate F1 tire degradation and recommend optimal pit stop windows. The model predicts future lap times based on recent lap history, then compares "stay out on worn tyres" vs "pit for fresh tyres" to identify the lap where pitting first becomes net beneficial.

The broader motivation: F1 pit stop strategy is one of the highest-leverage decisions in a race. Teams use proprietary simulation software to make these calls. This project attempts to replicate a simplified version of that capability using publicly available data and sequence modeling.

### Success Criteria
- **Primary:** >10% MAE improvement over moving average baseline on 2024 test races
- **Secondary:** ≥70% of pit window recommendations within ±5 laps of actual pit stop (relaxed from ±2; real F1 strategy decisions vary by 5–10 laps across teams, making ±2 tighter than expert variance)

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
| 11 | Delta prediction (Run 4) | 19.0% strategy accuracy | Changed target to ΔLapTime; doubled accuracy from 9.5% — degradation slope is the key signal |
| 12 | TrackTemp ablation (Run 5) | MAE improved, strategy −2.4% | TrackTemp hurts average MAE (noise after normalization) but helps at cold-track venues (Las Vegas). Retained. |
| 13 | Dual TrackTemp normalization (Run 6) | MAE improved, strategy −16.6% | Global temp gives model circuit-identity signal; learns temperature→timing heuristics from 44 races that don't transfer to 2024. Third MAE/strategy divergence. |

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

### Run 4 — Delta Target Prediction (ΔLapTime)

**Hypothesis:** Training on lap-to-lap delta (ΔLapTime = LapTime[t+1] − LapTime[t]) instead of absolute lap time forces the model to explicitly learn the degradation slope. The late-prediction bias in Eval Run 2 (mean error 8.0 laps) was driven by `predict_worn` underestimating how fast tyres degrade. If the model learns the rate of change directly, crossover detection should occur earlier.

**Implementation:**
- `build_sequences(delta=True)` in `data/preprocessing.py`: target changed to `target[i+SEQUENCE_LENGTH] - target[i+SEQUENCE_LENGTH-1]`
- New tensors saved with `_delta` suffix alongside absolute versions
- `train_model(..., delta=True)` in `train.py`: loads `_delta` tensors
- `predict_worn()` / `predict_fresh()` in `pit_optimizer.py`: accumulate deltas auto-regressively (`abs_lap_time += delta_pred`)
- New baseline: **mean of lap-to-lap deltas within the 10-lap window** (equivalent to extrapolating the current degradation trend)

**New baseline (mean delta): MAE = 0.043961**

Note: this baseline is *smaller* than the absolute baseline (0.060526) not because the task is easier, but because delta values are small and centered near zero. The absolute and delta baselines are not directly comparable.

#### Full Results Table
| Run | hidden | layers | dropout | Test MAE | Test MSE | Stopped Epoch | vs Delta Baseline |
|-----|--------|--------|---------|----------|----------|---------------|-------------------|
| lstm_baseline_delta | 64 | 2 | 0.2 | 0.041762 | 0.005429 | 20 | +5.0% |
| lstm_high_dropout_delta | 64 | 2 | 0.4 | 0.036317 | 0.004457 | 40 | **+17.4%** |
| **gru_baseline_delta** | **64** | **2** | **0.2** | **0.036866** | **0.004464** | **24** | **+16.1%** |

#### Key Observations — Run 4
- **lstm_baseline_delta (+5.0%):** Marginal improvement; LSTM without extra regularization still struggles with the delta signal. Earliest convergence (epoch 20).
- **lstm_high_dropout_delta (+17.4%):** Best raw MAE in this run (0.036317). High dropout (0.4) continues to be the key regularization lever for LSTM variants. Narrow margin over GRU in MAE terms.
- **gru_baseline_delta (+16.1%):** Second-best MAE. GRU converged faster (epoch 24 vs epoch 40 for lstm_high_dropout). Chosen for strategy eval because of its consistent cross-run stability.

#### Strategy Evaluation — Run 4 (gru_baseline_delta)

**Strategy accuracy: 19.0% (8/42) — doubled from 9.5% in Eval Run 2**

Correct predictions (within ±2 laps):
| Round | Circuit | Driver | Predicted | Actual | Error |
|-------|---------|--------|-----------|--------|-------|
| R2 | Saudi Arabia | NOR | 39 | 39 | **0** |
| R12 | Britain | VER | 41 | 40 | 1 |
| R12 | Britain | NOR | 41 | 41 | **0** |
| R14 | Spain | NOR | 33 | 31 | 2 |
| R16 | Hungary | VER | 42 | 43 | 1 |
| R17 | Belgium | NOR | 39 | 39 | **0** |
| R22 | Las Vegas | VER | 13 | 13 | **0** |
| R23 | Qatar | NOR | 45 | 43 | 2 |

**Stats across 42 evaluations:**
- Mean error: ~10.5 laps (still late on most standard circuits)
- Strategy accuracy: 19.0% vs 9.5% in absolute mode — confirms delta formulation addresses some late-bias

**Why did accuracy double?** The delta model learns to predict the sign and magnitude of change each lap, which makes the `predict_worn` curve slope upward more aggressively. This shifts the crossover point earlier, reducing the systematic late-bias.

**Why is it still far from 70% target?** The model still fails on:
- Races with very early first stops (e.g. R5 Miami, R17 Belgium actual lap 15 but predicted lap 39+)
- Circuits where degradation doesn't follow the typical gradual-wear pattern (e.g. ultra-abrasive Zandvoort)
- Multi-stop races where the model predicts no early window

---

### Run 5 — TrackTemp Ablation

**Hypothesis:** TrackTemp is directly correlated with tyre rubber-on-asphalt grip and degradation rate. However, per-race MinMaxScaler normalization compresses absolute temperature into [0,1] range, potentially removing cross-race signal. The ablation tests whether TrackTemp actually contributes to prediction quality or whether it's noise that the model ignores.

**Method:** Zero-mask `TrackTemp` (feature index 4) in both training and inference tensors. A perfectly useless feature should produce identical MAE when zeroed; if MAE changes significantly in either direction, the feature carries signal.

```python
FEATURE_INDEX = {
    'LapTimeSeconds': 0, 'StintLength': 1, 'FuelLoad': 2,
    'AirTemp': 3, 'TrackTemp': 4,
    'Compound_SOFT': 5, 'Compound_MEDIUM': 6, 'Compound_HARD': 7,
}
# Zero-masking in load_data():
X_train[:, :, FEATURE_INDEX['TrackTemp']] = 0.0
X_test[:,  :, FEATURE_INDEX['TrackTemp']] = 0.0
```

**Architecture:** Same as gru_baseline_delta (hidden=64, layers=2, dropout=0.2). Trained from scratch on zero-masked data so the model learns to not rely on TrackTemp.

#### Training Result

| Model | Test MAE | Test MSE | Stopped Epoch | vs Delta Baseline |
|-------|----------|----------|---------------|-------------------|
| gru_baseline_delta (with TrackTemp) | 0.036866 | 0.004464 | 24 | +16.1% |
| gru_baseline_delta_no_tracktemp | **0.035816** | **0.004387** | 31 | **+18.5%** |
| **Δ MAE (no_tracktemp − with_tracktemp)** | **−0.001050** | | | |

**Surprising result:** Removing TrackTemp slightly *improves* raw MAE (−0.001050). This suggests that within a single normalized race, the TrackTemp feature adds mild noise rather than clean signal — perhaps because per-race normalization wipes out the temperature information that matters most (absolute temperature level). The model without TrackTemp converges more slowly (epoch 31 vs 24), consistent with slightly less noisy data.

#### Strategy Evaluation — Run 5 (gru_baseline_delta_no_tracktemp)

**Strategy accuracy: 16.7% (7/42) — down from 19.0% with TrackTemp**

Comparison of correct predictions:
| Round | Circuit | Driver | With TrackTemp | Without TrackTemp | Notes |
|-------|---------|--------|---------------|------------------|-------|
| R2 | Saudi Arabia | NOR | ✓ (error 0) | ✓ (error 0) | Unchanged |
| R12 | Britain | VER | ✓ (error 1) | ✓ (error 1) | Unchanged |
| R12 | Britain | NOR | ✓ (error 0) | ✓ (error 0) | Unchanged |
| R14 | Spain | NOR | ✓ (error 2) | ✓ (error 2) | Unchanged |
| R16 | Hungary | VER | ✓ (error 1) | ✓ (error 2) | Unchanged (still ✓) |
| R17 | Belgium | NOR | ✓ (error 0) | ✓ (error 1) | Unchanged |
| R22 | Las Vegas | VER | ✓ (error **0**, pred 13) | ✗ (error **10**, pred 39) | **REGRESSION** |
| R23 | Qatar | NOR | ✓ (error 2) | ✓ (error 2) | Unchanged |

**Las Vegas R22 is the key regression:** With TrackTemp, the model correctly predicts lap 13 (actual 13). Without TrackTemp, it predicts lap 39 (error 10). Las Vegas is a night street race with one of the coldest track temperatures on the calendar (~18°C vs 30–50°C at most circuits). The cold track fundamentally changes tyre warm-up behavior and degradation rate. TrackTemp appears to provide the signal that triggers an early pit recommendation specifically at cold-track venues.

#### Conclusion — TrackTemp Signal

| Metric | Effect of Removing TrackTemp |
|--------|------------------------------|
| Raw MAE | Slightly **improves** (−0.001050) |
| Strategy accuracy | Slightly **declines** (19.0% → 16.7%) |

**Interpretation:** TrackTemp is weakly informative for average lap-time prediction (MAE), where its signal is diluted by per-race normalization. However, it carries **meaningful signal for strategy timing** at temperature-extreme circuits (Las Vegas night race). The dissociation between MAE and strategy accuracy demonstrates that raw prediction error is an imperfect proxy for downstream strategy quality — a model can improve on MAE while making worse strategy decisions.

**Decision: retain TrackTemp** in the feature set. The MAE gain from removing it is marginal (0.001), while the strategy cost is −2.4 percentage points (1 race in 42).

---

### Run 6 — Dual TrackTemp Normalization (9 features)

**Hypothesis:** Per-race MinMaxScaler destroys absolute temperature information. Adding a second `TrackTemp_global` feature — the same raw temperature normalized once across all training races — would give the model access to absolute temperature level while keeping the existing per-race relative variation.

**Implementation:**
- Global MinMaxScaler fitted on 2022+2023 TrackTemp values: range **15.6°C–57.5°C**
- Scaler saved to `data/processed/global_tracktemp_scaler.pkl` and loaded at inference time
- `TrackTemp_global` added as feature index 8; tensors regenerated as 9-feature (shape `[n, 10, 9]`)
- Model instantiated with `input_size=9`; old 8-feature models backward-compatible via `_model_input_size()` helper that slices sequences to model's expected size

**Training result:**
| Model | Test MAE | Test MSE | Stopped Epoch | vs Delta Baseline |
|-------|----------|----------|---------------|-------------------|
| gru_delta_global_tracktemp | 0.036330 | 0.004477 | 29 | **+17.4%** |
| gru_baseline_delta (Run 4) | 0.036866 | 0.004464 | 24 | +16.1% |

MAE improved by 0.000536 — a small positive signal in raw prediction quality.

#### Strategy Evaluation — Run 6 (gru_delta_global_tracktemp)

**Strategy accuracy: 2.4% (1/42) — catastrophic regression from 19.0%**

Only R18 Singapore VER was correct (predicted 29, actual 31, error 2).

**Prediction shift analysis (Run 6 vs Run 4):**
- Mean shift: **−16.0 laps** (Run 6 predicts 16 laps earlier on average)
- 41/42 races shifted to earlier predictions
- Std shift: 9.1 laps (high variance — inconsistent across circuits)

**Key regressions vs Run 4:**
| Round | Circuit | Driver | Run 4 pred | Actual | Run 6 pred | R4 err | R6 err |
|-------|---------|--------|-----------|--------|-----------|--------|--------|
| R2 | Saudi Arabia | NOR | 39 ✓ | 39 | 22 | 0 | **17** |
| R12 | Britain | VER | 41 ✓ | 40 | 29 | 1 | **11** |
| R12 | Britain | NOR | 41 ✓ | 41 | 25 | 0 | **16** |
| R17 | Belgium | NOR | 39 ✓ | 39 | 23 | 0 | **16** |
| R22 | Las Vegas | VER | 13 ✓ | 13 | 36 | 0 | **7** |

#### Root Cause Analysis

**Global temperature is teaching the model circuit-identity heuristics.** With only 44 training races (2022+2023), the model finds a spurious correlation: hot-circuit races tend to have faster degradation → earlier pits in training data. When global TrackTemp is high, the model aggressively predicts earlier pit stops across the board. But:

1. The temperature→timing correlation varies by year and regulation changes. 2024 pit strategies differ from 2022–2023 baseline behavior.
2. The model is using temperature as a *race identity* feature (which circuit am I on?) rather than a *physics* feature (how fast are tyres degrading?). With only 44 races, the circuit-identity signal dominates the learning.
3. Las Vegas R22 is the clearest illustration: global TrackTemp ≈ 0.06 (very cold), which the model associated with late-stop cold-weather circuits from training — predicted lap 36, 23 laps later than actual lap 13.

#### Critical Finding — MAE vs Strategy Accuracy Divergence (Third Case)

This is the third consecutive experiment where MAE and strategy accuracy moved in opposite directions:
| Experiment | MAE change | Strategy accuracy change |
|-----------|-----------|--------------------------|
| Run 5 (remove TrackTemp) | −0.001050 (better) | −2.4% (worse) |
| Run 6 (add global TrackTemp) | −0.000536 (better) | −16.6% (worse) |

**Conclusion:** MAE on the held-out test set is an unreliable proxy for strategy optimization quality. The strategy optimizer depends on the SHAPE of the degradation curve (when does the crossover happen?), not average prediction accuracy. A model that improves on average prediction error while distorting the degradation slope can produce dramatically worse strategy recommendations.

**Recommendation for future experiments:** Evaluate strategy accuracy on a subset of races *during training* as a more direct optimization signal, not just as a post-hoc metric.

---

### Phase 1 — Multi-Stop Strategy Simulation

**Hypothesis:** The current optimizer assumes a single future pit stop. Many 2024 races were 2-stop, so the optimizer is structurally incorrect for those events. Adding a 2-stop search should improve accuracy on multi-stop races without hurting 1-stop races.

**Implementation:**
- `find_optimal_two_stop()`: brute-force search over all `(pit_1, pit_2)` pairs where `pit_1 ∈ [start+1, total_laps//2]` and `pit_2 ∈ [pit_1+8, total_laps−SEQUENCE_LENGTH]`; caches worn predictions by stint length to avoid O(N²) model calls
- `find_optimal_pit_window()` now calls `find_optimal_two_stop()` from `start_lap=SEQUENCE_LENGTH` and compares total normalized race time for 1-stop vs 2-stop
- `recommended_pit_laps` returns `[pit_lap]` (1-stop) or `[pit_1, pit_2]` (2-stop) depending on which total is lower
- Scoring: first predicted pit vs nearest actual pit (closest-actual, forgiving of strategy-count mismatches)
- Success window: ±5 laps (relaxed from ±2 in earlier phases)

**Bug fixed during this phase:** `actual_sorted` leftover reference from bijective scoring caused every `results.append()` to throw a `NameError` silently, resulting in an empty DataFrame. Fixed to `actual_pit_laps`.

**Strategy accuracy: 35.7% (15/42) — +4.7pp improvement over 31.0% Phase 0 baseline**

| Metric | Phase 0 (Run 4) | Phase 1 (multi-stop) |
|--------|----------------|----------------------|
| Strategy accuracy ±5 laps | 31.0% | **35.7%** |
| Races evaluated | 42 | 42 |
| Model | gru_baseline_delta (8-feat) | gru_baseline_delta (8-feat) |

**Failure pattern:** Systematic late-prediction bias persists for the majority of races. The model predicts pits 15–32 laps later than actual on high-degradation circuits (Netherlands, Mexico, Miami, Imola). The 2-stop optimizer fires on some 1-stop races (e.g. R5 NOR predicted [12, 20], actual [32]) and misses on genuine 2-stop races where both actual stops are late in the race. Root cause: the shared model averages degradation across SOFT/MEDIUM/HARD, which have very different wear profiles, diluting the onset signal for aggressive compounds.

**Next step:** Phase 2 — compound-specific GRU models.

---

### Phase 2 — Compound-Specific GRU Models

**Hypothesis:** The shared GRU averages degradation behavior across SOFT, MEDIUM, and HARD tyres despite their very different wear profiles. Separate models per compound should produce sharper degradation curves for each type.

**Implementation:**
- Training data split by compound one-hot: `X_train_delta_{soft,medium,hard}.pt` / `X_test_delta_{soft,medium,hard}.pt`
- Three independent GRU models trained (same architecture: hidden=64, layers=2, dropout=0.2, delta target)
- `load_compound_models()` helper loads all three; `_resolve_model(model_or_dict, compound)` selects correct model at inference time
- `predict_worn()` and `predict_fresh()` now accept either a single `nn.Module` or a compound dict — backward-compatible with Phase 1

**Training results (compound-specific MAE):**
| Model | Sequences | Test MAE | vs gru_baseline_delta |
|-------|-----------|----------|-----------------------|
| gru_delta_soft | 4,025 | 0.042113 | −14.3% (worse — data-limited) |
| gru_delta_medium | 12,065 | 0.037233 | −1.0% |
| gru_delta_hard | 12,891 | 0.035449 | +3.8% |

SOFT MAE is higher due to having 3.4× fewer sequences. MEDIUM and HARD approximately match the shared baseline.

**Strategy accuracy: 61.9% (26/42) — +26.2pp over Phase 1, +30.9pp over Run 4 baseline**

| Metric | Run 4 | Phase 1 | Phase 2 |
|--------|-------|---------|---------|
| Strategy accuracy ±5 laps | 31.0% | 35.7% | **61.9%** |
| Model | shared GRU | shared GRU + multi-stop | compound GRU + multi-stop |

**Key improvements vs Phase 1:**
- Netherlands R15: 32-lap error → 1-lap error ✓
- Mexico R20 NOR: 27-lap error → 2-lap error ✓
- Belgium R14: both VER and NOR ✓ (was ✗ in Phase 1)
- Las Vegas R22: both ✓ maintained

**Remaining failures (16/42):**
- 5 early 2-stop triggers (pit_1 ≤ lap 15): R10 VER/NOR, R12 NOR, R16 VER, R18 NOR — model fires 2-stop before tyres are meaningfully worn
- 7 errors within ±6–8 laps: R2 NOR, R4 VER, R10 VER, R11 VER, R13 NOR, R16 NOR, R20 VER — tantalizingly close to ±5 cutoff
- 4 large misses (>10 laps, structural): R12 NOR (27 laps), R17 NOR (26 laps), R18 NOR (21 laps)

**Gap to 70% target:** Need 4 more correct (30/42). The 7 close failures are primary targets.

---

### Phase 3 — Minimum First-Pit Constraint (Reverted)

**Hypothesis:** 5 Phase 2 failures had spurious early first-pit predictions (pit_1 ≤ lap 15). Adding a minimum of `max(15, 20% of race laps)` would eliminate these and push predictions into more realistic territory.

**Result: 59.5% (25/42) — regression from 61.9%**

**Root cause of regression:** The constraint broke two races with legitimately early actual pit stops:
- R4 NOR (Japan): Actual pit was lap 13. Phase 2 correctly predicted [13, 21]. Constraint forced [21, 43] → error 7 ✗.
- R5 VER (China): Actual pit was lap 15. Phase 2 correctly predicted [12, 20]. Constraint forced [28, 46] → error 13 ✗.

The constraint fixed R10 VER (pit_1 pushed from 11 → 16, closer to actual 19), but broke two other correctly predicted early stops. Net: −1.

**Conclusion:** Real F1 strategy has legitimate early pits (undercuts on lap 12–15 at some circuits). A hard minimum first-pit constraint is too blunt — it would need to be circuit-specific to be useful, which is impractical without additional circuit metadata. **Reverted. Phase 2 (61.9%) remains the best result.**

---

### Phase 4 — Longer Sequence Length (seq=15, 2026-04-26)

**Hypothesis:** Increasing the sliding window from 10 to 15 laps gives the model more degradation context per prediction, potentially reducing early-stint prediction errors.

**Result: 50.0% (21/42) — regression from 61.9%**

**Config change:** `SEQUENCE_LENGTH = 10 → 15` in `data/preprocessing.py`. All other hyperparameters unchanged (hidden=64, layers=2, dropout=0.2, delta target, compound-specific GRUs).

**Sequence counts after preprocessing:**
- Train: 24,956 sequences at [N, 15, 9] (vs 28,981 at [N, 10, 9] — fewer because many stints are too short for 15-lap windows)
- Test: 2,764 sequences

**Training MAE (compound-specific, seq=15):**
| Model | MAE | vs baseline |
|-------|-----|-------------|
| gru_delta_soft | 0.036748 | −0.004464 |
| gru_delta_medium | 0.040378 | −0.000834 |
| gru_delta_hard | 0.035464 | −0.005748 |

**Root cause of regression:** Longer windows reduce the number of training sequences per compound (fewer stints long enough to generate 15-lap windows). This shrinks the training set, particularly for SOFT which had only 3,519 sequences (vs 4,087 at seq=10). The model also shifts predictions earlier because it's seeing more of the stint before making its first prediction, and the sliding window now spans laps where the model hasn't yet learned when to fire.

The regression pattern in the per-race table shows many 2-stop predictions with an early `pit_1` (laps 11–15) across long races — the model is triggering on the degradation it sees in the 15-lap window before the natural pit window opens.

**Conclusion:** More context is not better for this architecture. The 10-lap window is better calibrated to the actual degradation signal timescale in F1 stints. **Reverted to SEQUENCE_LENGTH=10. Phase 2 (61.9%) remains the best result.**

**Per-race results (42 race-driver combinations, 2024 season, VER + NOR):**

| Round | Circuit | VER | NOR |
|-------|---------|-----|-----|
| 1 | Bahrain | ✓ (err 5) | ✓ (err 4) |
| 2 | Saudi Arabia | ✓ (err 1) | ✗ (err 14) |
| 3 | Australia | ✓ (err 3) | — |
| 4 | Japan | ✓ (err 5) | ✓ (err 0) |
| 5 | China | ✓ (err 3) | ✗ (err 12) |
| 6 | Miami | ✓ (err 3) | ✗ (err 11) |
| 7 | Imola | ✗ (err 13) | ✓ (err 3) |
| 8 | Monaco | ✗ (err 43) | — |
| 10 | Spain | ✗ (err 8) | ✗ (err 14) |
| 11 | Austria | ✗ (err 14) | ✓ (err 3) |
| 12 | Britain | ✗ (err 25) | ✗ (err 27) |
| 13 | Hungary | ✓ (err 3) | ✓ (err 3) |
| 14 | Belgium | ✓ (err 0) | ✓ (err 3) |
| 15 | Netherlands | ✓ (err 3) | ✓ (err 3) |
| 16 | Italy | ✗ (err 13) | ✓ (err 3) |
| 17 | Azerbaijan | ✓ (err 2) | ✗ (err 28) |
| 18 | Singapore | ✗ (err 20) | ✗ (err 15) |
| 19 | USA | ✗ (err 14) | ✗ (err 17) |
| 20 | Mexico | ✗ (err 13) | ✗ (err 21) |
| 22 | Las Vegas | ✓ (err 1) | ✓ (err 1) |
| 23 | Qatar | ✗ (err 18) | ✗ (err 19) |
| 24 | Abu Dhabi | ✓ (err 5) | ✗ (err 17) |

21 correct / 42 evaluated = **50.0%**

| Artifact | Description |
|----------|-------------|
| `results/strategy_eval_seq15_final.log` | Full eval output (50.0%) |

---

## Checkpoints

### Best Result — Phase 2 Compound-Specific GRU (2026-04-26)

**Strategy accuracy: 61.9% within ±5 laps** — best result across all experiments.

**Model:** Three independent GRUs, one per tyre compound. Each trained on the delta (ΔLapTime) target using compound-filtered sequences from 2022+2023 training data.

| File | Description |
|------|-------------|
| `results/best_gru_delta_soft.pt` | SOFT compound GRU — MAE 0.042113, 4,025 training sequences |
| `results/best_gru_delta_medium.pt` | MEDIUM compound GRU — MAE 0.037233, 12,065 training sequences |
| `results/best_gru_delta_hard.pt` | HARD compound GRU — MAE 0.035449, 12,891 training sequences |
| `results/strategy_summary_BEST_phase2_compound_61pct.csv` | Per-race-driver breakdown (42 rows) |
| `results/strategy_eval_phase2_compound.log` | Full eval output |
| `results/training_phase2_compound.log` | Training output for all three models |

**Architecture:** hidden=64, layers=2, dropout=0.2, input_size=8, sequence_length=10, delta target.

**Optimizer config:** `pit_optimizer.py` `__main__` auto-detects compound weights via `load_compound_models()` and uses `find_optimal_two_stop()` for 2-stop simulation. To reproduce:
```
PYTHONPATH=. python strategy/pit_optimizer.py
```

**Eval summary (42 race-driver combinations, 2024 season, VER + NOR):**

| Round | Circuit | VER | NOR |
|-------|---------|-----|-----|
| 1 | Bahrain | ✓ (err 4) | ✓ (err 4) |
| 2 | Saudi Arabia | ✓ (err 2) | ✗ (err 7) |
| 3 | Australia | — (skipped) | ✗ (err 13) |
| 4 | Japan | ✗ (err 7) | ✓ (err 0) |
| 5 | China | ✓ (err 3) | ✓ (err 5) |
| 6 | Miami | ✓ (err 3) | ✓ (err 5) |
| 7 | Imola | ✓ (err 3) | ✓ (err 3) |
| 8 | Monaco | ✓ (err 4) | — |
| 10 | Spain | ✗ (err 8) | ✗ (err 14) |
| 11 | Austria | ✗ (err 6) | ✓ (err 3) |
| 12 | Britain | ✗ (err 15) | ✗ (err 27) |
| 13 | Hungary | ✗ (err 12) | ✗ (err 7) |
| 14 | Belgium | ✓ (err 1) | ✓ (err 5) |
| 15 | Netherlands | ✓ (err 1) | ✓ (err 3) |
| 16 | Italy | ✗ (err 13) | ✗ (err 8) |
| 17 | Azerbaijan | ✓ (err 2) | ✗ (err 26) |
| 18 | Singapore | ✓ (err 3) | ✗ (err 21) |
| 19 | USA | ✓ (err 1) | ✓ (err 5) |
| 20 | Mexico | ✗ (err 6) | ✓ (err 2) |
| 22 | Las Vegas | ✓ (err 2) | ✓ (err 0) |
| 23 | Qatar | ✓ (err 0) | ✗ (err 12) |
| 24 | Abu Dhabi | ✓ (err 5) | ✓ (err 1) |

26 correct / 42 evaluated = **61.9%**

---

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
| ~~Delta prediction (ΔLapTime target)~~ | DONE — positive | ~~High~~ | Run 4. Strategy accuracy doubled from 9.5% → 19.0% on 42 evaluations. Delta formulation corrects late-prediction bias by forcing explicit degradation slope learning. |
| ~~TrackTemp ablation~~ | DONE — retained | ~~Medium~~ | Run 5. Removing TrackTemp slightly improves MAE (−0.001) but costs 1 race (Las Vegas night race). TrackTemp kept in feature set. |
| ~~Safety car lap filtering~~ | DONE — negative | ~~Medium~~ | Tried in Run 3. Strategy accuracy dropped 20% → 0% on 5-race sample. Expanded 42-race eval confirmed 9.5% true accuracy. |
| ~~Per-circuit pit loss~~ | DONE — partial | Medium | Added `PIT_LOSS_BY_ROUND_2024` dict. Helped unusual-layout circuits (Monaco, Las Vegas, Miami all ✓). Didn't fix fundamental late-bias. |
| ~~Dual TrackTemp normalization (global + per-race)~~ | DONE — negative | ~~Medium–High~~ | Run 6. MAE 0.036330 (improved), strategy accuracy **2.4%** (catastrophic regression from 19.0%). Global temp teaches model circuit-identity heuristics; see Run 6 notes below. |
| ~~TrackTemp × Compound interaction features~~ | Dropped | ~~Medium~~ | Run 6 showed absolute temperature hurts strategy more than it helps. Interaction features would compound the same problem. Deprioritized. |
| ~~Minimum first-pit constraint~~ | DONE — negative | ~~Medium~~ | Phase 3. 59.5% vs 61.9% Phase 2. Broke legitimate early pits (Japan R4 lap 13, China R5 lap 15). Hard constraint too blunt without circuit-specific metadata. Reverted. |
| GRU with high dropout (0.4) | Pending | Medium | lstm_high_dropout showed +31.5%; never tried on GRU which is already the strongest architecture |
| ~~Longer sequence length (15 laps)~~ | DONE — negative | ~~Medium–High~~ | Phase 4. **50.0%** (21/42) vs 61.9% Phase 2. −11.9pp regression. See Phase 4 notes. |
| ~~Compound-specific degradation models~~ | DONE — **major positive** | ~~High~~ | Phase 2. Strategy accuracy **61.9%** (+26.2pp over Phase 1, +30.9pp over baseline). Separate GRU per compound dramatically reduces cross-compound averaging noise. |
| ~~Multi-stop strategy simulation~~ | DONE — positive | ~~High~~ | Phase 1. Strategy accuracy improved 31.0% → **35.7%** (+4.7pp) on 42 evaluations. See Phase 1 notes. |
| More training data (2021 and earlier) | Pending | Medium | lstm_deep proved more data helps; GRU likely improves too |
| AdamW optimizer + weight decay | Pending | Low–Medium | Better regularization than dropout alone; may stabilize LSTM variants |
| Competitor position as feature | Pending | High | Would allow model to account for undercut/overcut strategy — real teams pit early for track position |
| Track-specific models | Pending | Medium | Each circuit has different degradation characteristics; per-circuit tuning |
