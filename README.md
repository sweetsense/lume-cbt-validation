# CBT Fluorimeter Validation -- Data & Code

Reproducibility package for the field validation of a submersible tryptophan-like
fluorescence (TLF) sensor against Compartment Bag Test (CBT) E. coli enumeration,
deployed across drinking-water sites in Rwanda and Kenya.

## Files

| File | Description |
|------|-------------|
| `paired_observations.csv` | Cleaned, paired dataset: 216 sensor-vs-CBT observations with raw and processed features, in-sample and LOOCV predictions. |
| `model_specification.json` | Full model specification: Tobit regression coefficients, normalization statistics, per-sensor baselines, and agreement metrics. |
| `make_figures.py` | Python script to generate publication-quality figures from `paper_data.json` (the full export). |

## Dataset columns (`paired_observations.csv`)

| Column | Description |
|--------|-------------|
| `observation_id` | Unique observation identifier |
| `barcode` | Sensor device barcode (50045, 50053, or 50065) |
| `site_name` | Sampling site name/hierarchy |
| `sample_date` | Date/time of CBT sample collection |
| `water_type` | Water source type (e.g., Source, Treated) |
| `cbt_ecoli_cfu` | Observed E. coli CFU/100 mL from Compartment Bag Test |
| `cbt_censored` | TRUE if CBT result is right-censored (>= 100 CFU) |
| `sensor_mon2_raw` | Raw TLF fluorescence signal (arbitrary units) |
| `sensor_tof_raw` | Raw time-of-flight turbidity signal |
| `water_temp_c` | Water temperature at time of measurement (degrees C) |
| `mon2c_baseline_subtracted` | Temperature-corrected TLF after per-sensor baseline subtraction |
| `mon2c_normalized` | z-scored mon2c_baseline_subtracted (model input) |
| `tof_normalized` | z-scored baseline-subtracted ToF signal (model input) |
| `predicted_log10_cfu_insample` | In-sample Tobit model prediction, log10(CFU + 1) |
| `predicted_log10_cfu_loocv` | Leave-one-out cross-validation prediction, log10(CFU + 1) |
| `loocv_agreement` | TRUE if LOOCV prediction is within the agreement band |
| `who_risk_true` | WHO risk category from CBT (Conformity/Low/Intermediate/Very high) |
| `who_risk_predicted` | WHO risk category from model prediction |

## Model overview

The prediction model is a Tobit regression (right-censored at log10(101) = 2.004)
with EM estimation and ridge regularization (lambda = 0.1). Features are
temperature-corrected fluorescence (`mon2c_n`, baseline-subtracted, z-scored),
turbidity (`tof_n`, baseline-subtracted, z-scored), water temperature (`temp`,
z-scored), per-sensor fixed effects, and per-sensor fluorescence slopes
(8 parameters total). Reference sensor is 50065.

**Temperature correction**: Fluorescence is corrected to a 20 C reference using an
exponential decay model (rho = 0.0164, estimated from clean-water samples).
Temperature is also included as an explicit predictor.

**Per-sensor slopes**: Each sensor receives its own fluorescence coefficient
(via sensor x mon2c_n interaction terms), allowing the model to account for
inter-sensor sensitivity differences.

**Exclusion pipeline**: No statistical screening (no IQR fencing, no Cook's D,
no Kind A outlier removal). Only 4 documented instrument/deployment overrides:
1 sensor fault (cross-sensor validated), 2 turbidity-compromised readings
(ToF >> in-water norm), 1 baseline-transition artifact. See
`cbt-overrides-snapshot.json`.

**Agreement criterion**: Predictions are scored as "agreeing" with the CBT if they
fall within +/- 0.92 log10, derived from combining two independent CBT 95%
confidence intervals in quadrature.

## Reproducing the figures

```bash
pip install numpy matplotlib
python make_figures.py
```

Note: `make_figures.py` expects `paper_data.json` (the full export from
`export-paper-data.js`) in the same directory. To generate figures from the CSV
alone, the paired observations and model specification provide all necessary data.

## Live validation dashboard

Interactive results, time series, and data exploration:
https://validation.thelume.ai/cbt

## License

Data and code are provided for peer review and reproducibility purposes.
Contact the authors for reuse permissions.
