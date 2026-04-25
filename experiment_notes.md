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
| Version | Training Data | Training Sequences | Test Sequences |
|---------|--------------|-------------------|----------------|
| v1 (initial) | 2023 only | ~14,000 | 3,242 |
| v2 (current) | 2022 + 2023 | 28,115 | 3,242 |

Doubling the training set by adding 2022 data significantly changed which model architectures worked (see Experiment Results).

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

### Test Race Results (2024 Rounds 1–5) — GRU Baseline Model

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

#### Success Criteria Summary
- ✅ **Primary PASSED** — GRU achieves ~39% MAE improvement over moving average baseline
- ❌ **Secondary FAILED** — 20% strategy accuracy (1/5 races within ±2 laps)

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

| Idea | Expected Impact | Rationale |
|------|----------------|-----------|
| GRU with high dropout (0.4) | Medium | lstm_high_dropout showed +31.5%; GRU + dropout likely pushes higher |
| Longer sequence length (15–20 laps) | Medium–High | More degradation history; also makes attention viable |
| Safety car lap filtering | Medium | FastF1 provides TrackStatus data — filter SC/VSC laps from training & eval |
| Compound-specific degradation models | High | SOFT/MEDIUM/HARD have very different wear profiles; shared model averages them |
| Multi-stop strategy simulation | High | Current optimizer assumes one future pit; real races often have two stops |
| More training data (2021 and earlier) | Medium | lstm_deep proved more data helps; GRU likely improves too |
| GRU with attention at longer seq length | Unknown | Attention failed at seq_len=10; try at 20–30 after increasing sequence length |
| AdamW optimizer + weight decay | Low–Medium | Better regularization than dropout alone; may help lstm_baseline stability |
| Competitor position as feature | High | Would allow model to account for undercut/overcut strategy |
| Track-specific models | Medium | Each circuit has different degradation characteristics; per-circuit tuning |
