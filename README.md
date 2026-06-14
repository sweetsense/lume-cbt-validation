# CBT Fluorimeter Validation -- Data & Code

Reproducibility package for the field validation of a submersible tryptophan-like
fluorescence (TLF) sensor against Compartment Bag Test (CBT) E. coli enumeration,
deployed across drinking-water sites in Rwanda and Kenya.

## Files

| File | Description |
|------|-------------|
| `paired_observations.csv` | Cleaned, paired dataset: 177 sensor-vs-CBT observations with raw and processed features, in-sample and LOOCV predictions, and WHO risk classifications. |
| `model_specification.json` | Full model specification: Tobit regression coefficients, normalization statistics, temperature correction parameters, per-sensor baselines, and agreement metrics. |
| `make_figures.py` | Python script to generate publication-quality figures (scatter plot, confusion matrices, temperature correction) from the exported data. |

## Dataset columns (`paired_observations.csv`)

| Column | Description |
|--------|-------------|
| `observation_id` | Sequential integer identifier |
| `barcode` | Sensor device barcode (50045, 50053, or 50065) |
| `site_name` | Water sampling site name |
| `sample_date` | Date and time of water sample collection (local time) |
| `water_type` | Type of water source (Treated or Source/untreated) |
| `cbt_ecoli_cfu` | Observed E. coli CFU/100 mL from Compartment Bag Test |
| `cbt_censored` | TRUE if CBT result is right-censored (>= 100 CFU) |
| `sensor_mon2_raw` | Raw TLF fluorescence signal (mon2, arbitrary units) |
| `sensor_tof_raw` | Raw time-of-flight turbidity signal (kcps) |
| `water_temp_c` | Water temperature at time of measurement (degrees C) |
| `mon2c_temp_corrected` | TLF signal after multiplicative temperature correction to 20 C reference |
| `mon2c_normalized` | Temperature-corrected TLF after per-sensor baseline subtraction |
| `tof_normalized` | ToF signal after per-sensor baseline subtraction |
| `predicted_log10_cfu_insample` | In-sample Tobit model prediction, log10(CFU + 1) |
| `predicted_log10_cfu_loocv` | Leave-one-out cross-validation prediction, log10(CFU + 1) |
| `loocv_agreement` | TRUE if LOOCV prediction is within the CBT agreement band |
| `who_risk_true` | WHO risk category from observed CBT result |
| `who_risk_predicted` | WHO risk category from LOOCV prediction |

## Model overview

The prediction model is a Tobit regression (right-censored at log10(101) = 2.004)
with EM estimation. Features are z-scored, temperature-corrected TLF fluorescence
and baseline-subtracted ToF turbidity signal, plus sensor fixed effects.

**Temperature correction**: `mon2c = mon2_raw * exp(-rho * (temp - 20))` where
rho = 0.0140 per degree C, estimated from 101 clean-water (CBT = 0) samples.

**Agreement criterion**: Predictions are scored as "agreeing" with the CBT if they
fall within +/- 0.92 log10, derived from combining two independent CBT 95%
confidence intervals.

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
