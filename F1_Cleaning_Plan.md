# F1 Podium Prediction — Data Cleaning Plan (for Claude Code)

## 0. Context

- **Project:** CSEN 903 Milestone 1. Predict whether a driver finishes on the podium (`positionOrder <= 3`) using only information known **after qualifying and before the race starts** (the "cutoff").
- **Data:** Kaggle *Formula 1 World Championship (1950–2024)*, the Ergast schema: 14 CSVs, with missing values written as the literal text `\N`.
- **Notebook:** `F1_Podium_Prediction.ipynb` already exists. This plan **replaces §5 (Cleaning)** and updates §3–§8 where they depend on cleaning. Keep the notebook's conventions:
  - `RAW` is never modified;
  - cleaning works on copies;
  - figures are saved through `savefig()`;
  - open decisions are marked `TODO(decide)`.
- **Every decision below has already been made by the user.** Implement them exactly as written. If the data contradicts an assumption here, **stop and report** rather than silently choosing something else.

---

## 1. Scope switch: modern vs all data (required)

Add one config flag at the top of the notebook:

```python
DATA_SCOPE = "modern"        # "modern" = 2006–2024 (main choice) | "all" = 1950–2024 (comparison trial)
SCOPE_YEARS = {"modern": (2006, 2024), "all": (1950, 2024)}
```

- **"modern" (main pipeline):** 2006–2024. This is the knockout-qualifying era, with consistent qualifying, field sizes and data coverage.
- **"all" (comparison trial):** every season. It exists only to **prove in the evaluation** that the 2006+ choice is better.
- **The comparison must be fair:** both scopes are evaluated on the **same validation seasons (2021–2022) and the same test seasons (2023–2024)**. Only the training seasons differ:
  - modern: train 2006–2020;
  - all: train 1950–2020.
- Wrap the whole pipeline (cleaning → modelling table → models) so it can run once per scope with the same model settings. Write a comparison table to `outputs/scope_comparison.csv` with these columns:
  - scope;
  - train rows;
  - % of feature cells missing;
  - validation and test ROC-AUC, PR-AUC and F1 for LR, RF and the FFNN.
- Plot the comparison (`figures/scope_comparison.png`).
- **Everything below applies to both scopes** unless it says "(all only)" or "(modern only)".

---

## 2. Fixed steps (no real alternative, so implement directly)

1. **Load raw data:** read every CSV with `dtype=str, keep_default_na=False` into `RAW`, and never modify it.
2. **Sentinel audit before cleaning:**
   - count `\N`, empty and whitespace-only cells per column;
   - flag untrimmed whitespace;
   - this is already in §3; keep it.
3. **Sentinels → NaN:** on a copy, `strip()` every text cell, apply `html.unescape()`, then turn `\N`, `""`, `NULL`, `NaN` and `None` into `NaN`.
4. **Semantic typing:**
   - IDs and counts → `Int64`;
   - points and speeds → `float64`;
   - dates → `datetime64`;
   - times of day → `timedelta`.
5. **Durations → seconds:** convert `q1`, `q2`, `q3`, `fastestLapTime`, `lap_times.time` and `pit_stops.duration`, handling both `m:ss.sss` and `h:mm:ss.sss`. Then assert that the number of missing values after parsing equals the number of sentinels before. Any extra missing value is a parse failure: list it and explain it.
6. **Drop `results.time`:** it mixes the winner's absolute time with gaps (`+5.478`), and it is post-race information anyway.
7. **Missing-value types:** add a table (`outputs/missingness_types.csv`) labelling every column that has missing values as one of:
   - `structural`: the field didn't exist yet (e.g. `fastestLapTime` before 2004);
   - `outcome`: missing because of how the race went (`time`, `milliseconds`, `position` for non-finishers). These columns can **never** be features;
   - `pre-race-informative`: e.g. `q2`/`q3` missing because the driver was knocked out. These become features.
8. **Key checks as assertions** (not just report tables). Run them after every cleaning step that changes rows:
   - primary-key uniqueness and non-null;
   - natural grain, i.e. one row per (raceId, driverId) in results, qualifying, sprint_results and driver_standings;
   - foreign-key integrity (no orphans).
9. **Feature whitelist:** keep an explicit list of allowed features. The modelling table may only contain columns from that list, plus IDs and the target. Replace the current `FORBIDDEN` blacklist check with a whitelist assertion, and keep the inference-time rejection of post-race fields.
10. **Cleaning log:** every rule below writes one row to `outputs/cleaning_log.csv` with these columns: `step, table, column, rule, rows_affected, cells_affected, rows_before, rows_after, notes`.
11. **Before/after report:** for each cleaned table, show:
    - dtype changes;
    - missing % before vs after;
    - row counts before vs after;
    - PK/FK results before vs after.

    Save as `outputs/cleaning_before_after.csv`, plus a figure.

---

## 3. Decisions made by the user (implement exactly)

### D1. Seasons
- Main pipeline: **2006–2024**.
- Comparison trial: **all seasons** (section 1).
- **Verify coverage first.** Print the first season that has data in `qualifying`, `lap_times`, `pit_stops` and `sprint_results` in *this* Kaggle copy. Expected: pit stops from 2011, lap times from 1996, sprints from 2021. Pit-stop-based features are missing before their first season and get a `pit_data_available` flag (see D6).

### D2. Non-starters
- Rule: **drop rows that have no grid slot at the cutoff.** That means:
  - status `Did not qualify`, `Did not prequalify`, `107% Rule` or `Not qualified`, whatever the grid value;
  - status `Excluded` or `Withdrew` **with `grid == 0`**.
- **Keep** `Withdrew` rows that **have a grid slot** (`grid > 0`). In this data, that is how a non-start after qualifying is recorded (there is no separate "Did not start" label from 2006). At the cutoff you would have predicted for these drivers, so dropping them would be selecting on the outcome.
- Log the counts per status per scope.

### D3. Shared drives (matters in "all" only; nothing to resolve in 2006+)
- **Merge to one row per (raceId, driverId):**
  - pre-race fields (`grid`, `constructorId`, `number`) come from the row with the **earliest `resultId`**, i.e. the car the driver started in;
  - `podium = 1` if **any** of his rows has `positionOrder <= 3`;
  - `points` = sum of the rows;
  - `positionOrder` = the minimum (best);
  - `status` = the status of the best-placed row.
- Afterwards:
  - assert the grain is unique;
  - report races whose number of podium rows is not 3, and explain each one (genuine ties or shared podiums in the 1950s).

### D4. Indianapolis 500
- **Exclude** all races whose name contains "Indianapolis" (1950–1960, 11 races) from the data questions Q1–Q3 and from the model.
- Log it and mention it in the report as a scope decision.

### D5. grid = 0
- **2006+:** pit-lane start, so:
  - `grid = field_size + 1`
  - `pitlane_start = 1`
- **Before 2006 (all only):** the slot is unknown, so:
  - `grid = NaN`
  - `grid_unknown = 1`
- `field_size` = max(number of result rows in the race, max grid value in the race). This handles races where the grid number is bigger than the number of rows (Formula 2 cars mixed into the field in some 1950s–80s races).
- Also add `grid_pct = grid / field_size`.
- Apply D2 **before** D5, so pre-race exits with grid 0 are already gone.

### D6. Missing values in features: domain fill first, then median + indicator
- **Step 1, domain fill** (only where the real value is known):
  - rookie / no previous races: `drv_career_starts = 0`, `drv_career_podiums = 0`;
  - first visit to a circuit: `drv_circuit_podium_rate` stays **NaN**, and `drv_circuit_first_visit = 1`. "Never been here" is not the same as "0%";
  - pit-stop features in seasons without pit data: NaN, and `pit_data_available = 0`;
  - non-sprint weekend: sprint features NaN, `is_sprint_weekend = 0` (see D14).
- **Step 2:** the preprocessing pipeline imputes whatever is still missing with the **median fitted on the training split only**, plus missing-indicator columns (`SimpleImputer(strategy="median", add_indicator=True)`).
- Log, per feature, how many cells were filled by step 1 and how many by step 2.

### D7. Round-1 standings
- For round 1 of each season, standings features come from **last season's final standings** (rank, and share of the leader's points; see D13). Add `is_round_1 = 1`.
- Rookies and new teams with no previous season: NaN + flag, handled by D6 step 2.
- Teams that were renamed are matched through `constructor_family` (D12).
- Leakage check: last season's final standings are dated before this race. Add an assertion.

### D8. Timing outliers: per-group robust z-score plus hard limits
- **Method:** modified z-score `z = 0.6745 × (x − median) / MAD` (Iglewicz & Hoaglin, 1993), computed **within each group**:
  - `pit_stops.duration_sec`: grouped by race;
  - `q1_sec`, `q2_sec`, `q3_sec`: grouped by race and session.
- **Flag** values where |z| > 3.5. Also flag anything outside these hard limits:
  - pit stop > 120 s or < 1.5 s;
  - qualifying time > 2 × session median.
  - If MAD = 0, use the hard limits only.
- **Action:**
  - set the flagged cell to NaN, but **never delete the row**;
  - keep a boolean `<col>_outlier` column;
  - log the counts, plus examples such as the `16:42.640` Q1 time (raceId 255) and red-flag stops.
- Show before/after distributions (histograms) in the notebook.

### D9. Lap-time cleaning (for past-race pace features only)
- Per race, **remove**:
  1. lap 1;
  2. pit in-laps and out-laps, matched from `pit_stops` on (raceId, driverId, lap) and lap + 1;
  3. laps slower than **107% of that race's median green-flag lap**. These are safety car, VSC and red-flag laps. Compute the race median after removing 1 and 2.
- Then:
  - compute each driver's **median** remaining lap per race;
  - `pace_ratio = driver_median / best_driver_median_in_race` (1.000 = fastest);
  - drivers with fewer than 5 clean laps in a race → NaN for that race.
- These numbers are used **only as past-race features** (`shift(1)` on chronological order, rolling over previous races). **Never** use current-race lap data.
- Log laps removed per filter.

### D10. Target
- `podium = (positionOrder <= 3).astype(int)`, computed **after** D3.
- Assertions:
  - no NaN in the target;
  - in "modern", every race has exactly 3 positives (list any exception).

### D11. Status codebook
- Create `codebooks/status_map.csv` mapping **every** status label in `status.csv` to exactly one group: `finished`, `lapped`, `mechanical`, `incident`, `non_start`, `disqualified` or `ambiguous`.
  - `finished`: "Finished".
  - `lapped`: "+N Lap(s)".
  - `mechanical`: Engine, Gearbox, Hydraulics, Brakes, Power Unit, ERS, Turbo, Electrical, …
  - `incident`: Accident, Collision, Spun off, Collision damage, …
  - `non_start`: Withdrew, Did not qualify, …
  - `disqualified`: Disqualified, Excluded.
  - `ambiguous`: Retired, Technical, Puncture, Wheel, Wheel nut, Tyre, Damage, Debris, Vibration(s), Rear/Front wing, Undertray, Handling, Seat, Out of fuel, …
- Assert that all labels are mapped and none are duplicated.
- **Before finalising, show the user the full mapping table for review.**
- **Q3:** compute the mechanical-retirement rate per constructor **twice**:
  - (a) ambiguous counted as not mechanical;
  - (b) ambiguous counted as mechanical.

  Report both rankings side by side and their Spearman rank correlation.
- Finish and reliability features (e.g. `drv_finish_rate_lastN`) use this codebook: finished or lapped = finished.

### D12. Constructor lineage
- Create `codebooks/constructor_lineage.csv` with columns `constructorId, constructor_name, constructor_family, note`. At least these chains:
  - Minardi → Toro Rosso → AlphaTauri → RB
  - Jordan → Midland → Spyker → Force India → Racing Point → Aston Martin
  - Sauber → BMW Sauber → Sauber → Alfa Romeo → (Kick) Sauber
  - Benetton → Renault → Lotus F1 → Renault → Alpine
  - BAR → Honda → Brawn → Mercedes (judgement call: same factory, new owner. Note it.)
  - Stewart → Jaguar → Red Bull
- Unmapped constructors map to themselves.
- **List every judgement call** in the `note` column and **show the table to the user for review.**
- Use it as follows:
  - `constructor_family` for rolling team-form features, round-1 standings matching (D7) and Q3 grouping;
  - the original name for display only.

### D13. Points and standings
- Never use raw standings points as features. Use, for driver and constructor, **before the race** (standings after the previous round, or last season's final standings for round 1):
  - `champ_rank`;
  - `pts_share = points / leader_points`, with 0 if leader_points == 0 (round-1 edge case);
  - `gap_to_leader_share = 1 − pts_share`.
- Reason: scale-free across the 2010 scoring change, 2014 double points and half-points races.

### D14. Sprint weekends
- `is_sprint_weekend` = 1 if the race has a `sprint_results` row (or a `sprint_date`).
- `sprint_finish_pct = sprint positionOrder / sprint field size`. The sprint runs before the Sunday race, so this is allowed; NaN on normal weekends.
- Note in the notebook that in 2021–22 the sprint set the Sunday grid, so `grid_minus_quali` is non-zero by design on those weekends.

### D15. Countries and nationalities
- After strip + `html.unescape`, map both sides to **ISO-3166 alpha-3** using two explicit dicts, stored as `codebooks/nationality_iso.csv` and `codebooks/country_iso.csv`:
  - British → GBR; Argentine and Argentinian → ARG; East German → DEU; Rhodesian → ZWE; Monegasque → MCO;
  - dual nationality (e.g. "American-Italian") → the first one listed;
  - circuits: UK → GBR; USA and United States → USA; UAE → ARE; Korea → KOR; …
- Assert there are **zero unmapped values** on either side.
- `home = driver_iso == circuit_iso`, used in Q2.

### D16. Abnormal races
- **Keep every race.** Add `race_shortened = 1` when the winner's laps are < 75% of the race's maximum laps in an average season at that circuit, or when half points were awarded. Examples: 2021 Belgium, 2009 Malaysia.
- This is **for reporting and error analysis only. It is not a model feature**, because you can't know before the start that a race will be cut short.

---

## 4. Order of operations

1. Load `RAW`, then run the before-cleaning audit (sentinels, PK/FK, grain, coverage).
2. Strip + unescape → sentinels to NaN → typing → duration parsing (fixed steps 3–6).
3. Apply scope filter (D1) → exclude Indy 500 (D4) → drop non-starters (D2) → merge shared drives (D3, all only).
4. grid = 0 handling and field size (D5) → timing outliers (D8) → lap-time cleaning (D9).
5. Codebooks: status (D11), constructor lineage (D12), ISO countries (D15). Stop for user review of the D11 and D12 tables.
6. Target (D10), standings representation (D13), round-1 standings (D7), sprint features (D14), abnormal-race flag (D16).
7. Domain fill (D6 step 1). Step 2 happens later, inside the model pipeline.
8. After-cleaning audit + before/after comparison + cleaning log, then run all assertions.
9. Build the modelling table and run the existing leakage tests (§8.1), extended to the new past-race pace and pit features.
10. Run the full pipeline for both scopes and write the scope comparison (section 1).

---

## 5. Deliverables

- An updated `F1_Podium_Prediction.ipynb`. Every cleaning step gets:
  - a short markdown cell: what, why (one line), and which decision (D#);
  - a code cell;
  - a check (an assertion or a printed count).
- `codebooks/`: `status_map.csv`, `constructor_lineage.csv`, `nationality_iso.csv`, `country_iso.csv`.
- `outputs/`: `cleaning_log.csv`, `cleaning_before_after.csv`, `missingness_types.csv`, `scope_comparison.csv`, plus figures (sentinel heatmap before/after, outlier histograms before/after, coverage by season, scope comparison).
- A final notebook cell that prints a decision summary: every D1–D16 with its chosen option and its counts from the cleaning log. This is for the report.

## 6. Acceptance checks (all must pass with Run All, in both scopes)

- [ ] `RAW` is unchanged (compare a hash of `RAW` at the start and the end).
- [ ] No `\N` string remains in any cleaned table.
- [ ] Missing count after typing equals the sentinel count before, except for logged parse failures and D8 outliers.
- [ ] PK, natural grain and FK assertions pass after every row-changing step.
- [ ] Every status, nationality and country is mapped; the codebooks have no duplicates.
- [ ] Target has no NaN; every race in "modern" has exactly 3 podium rows.
- [ ] The modelling table contains only whitelisted features, plus IDs and the target.
- [ ] Leakage tests pass: form, pace and pit features use only earlier races; standings come from the previous round or last season.
- [ ] Imputers and scalers are fitted on the training split only (already demonstrated by preprocessing experiment E2).
- [ ] The scope comparison uses identical validation (2021–22) and test (2023–24) seasons for both scopes.

## 7. Out of scope (do not change in this task)

Model choice, hyperparameter grids, NN architecture, XAI and the Q1/Q2 analysis logic, apart from the inputs they now receive from cleaning. One change outside cleaning is required, because the scope comparison depends on it: set the split to train ≤ 2020, validation 2021–22, test 2023–24.
