# CBT Fluorimeter Validation -- Data & Code

Reproducibility package for the field validation of a submersible tryptophan-like
fluorescence (TLF) sensor against Compartment Bag Test (CBT) E. coli enumeration,
deployed across drinking-water sites in Rwanda and Kenya.

## Files

| File | Description |
|------|-------------|
| `paired_observations.csv` | Cleaned, paired dataset: 205 sensor-vs-CBT observations with raw and processed features, in-sample and LOOCV predictions. |
| `model_specification.json` | Full model specification: Tobit regression coefficients, normalization statistics, temperature correction parameters, per-sensor baselines, and agreement metrics. |
| `make_figures.py` | Python script to generate publication-quality figures from `paper_data.json` (the full export). |

## Dataset columns (`paired_observations.csv`)

| Column | Description |
|--------|-------------|
| `observed_cfu` | Observed E. coli CFU/100 mL from Compartment Bag Test |
| `observed_log10` | log10(observed CFU + 1) |
| `predicted_log10` | In-sample Tobit model prediction, log10(CFU + 1) |
| `predicted_loo_log10` | Leave-one-out cross-validation prediction, log10(CFU + 1) |
| `barcode` | Sensor device barcode (50045, 50053, or 50065) |
| `temperature` | Water temperature at time of measurement (degrees C) |
| `mon2` | Raw TLF fluorescence signal (arbitrary units) |
| `mon2c` | Temperature-corrected TLF signal (to 20 C reference) |
| `mon2c_n` | Temperature-corrected TLF after per-sensor baseline subtraction (z-scored) |
| `tof_n` | Time-of-flight turbidity signal after per-sensor baseline subtraction |
| `censored` | TRUE if CBT result is right-censored (>= 100 CFU) |
| `agree` | TRUE if in-sample prediction is within the agreement band |
| `country` | Country of deployment (Kenya or Rwanda) |

## Model overview

The prediction model is a Tobit regression (right-censored at log10(101) = 2.004)
with EM estimation and ridge regularization (lambda = 0.1). Features are z-scored,
temperature-corrected TLF fluorescence (`mon2c_n`), a quadratic fluorescence term
(`mon2c_n^2`), and per-sensor fixed effects (5 parameters total).

**Temperature correction**: `mon2c = mon2_raw * exp(-rho * (temp - 20))` where
rho = 0.0235 per degree C, estimated from clean-water (CBT = 0) samples.

**Exclusion pipeline**: Fully automated with no manual overrides. IQR fencing
(Q3 + 1.5 * IQR for clean-water samples) excludes instrument outliers. No Cook's
distance or manual override exclusions are applied.

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
