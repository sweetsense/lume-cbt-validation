# CBT Fluorimeter Validation -- Data & Code

Reproducibility package for the field validation of a submersible tryptophan-like
fluorescence (TLF) sensor against Compartment Bag Test (CBT) E. coli enumeration,
deployed across drinking-water sites in Rwanda and Kenya.

## Files

| File | Description |
|------|-------------|
| `paired_observations.csv` | Cleaned, paired dataset: 189 sensor-vs-CBT observations with raw and processed features, in-sample and LOOCV predictions. |
| `model_specification.json` | Full model specification: Tobit regression coefficients, normalization statistics, per-sensor baselines, and agreement metrics. |
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
| `mon2c_n` | Temperature-corrected TLF after per-sensor baseline subtraction |
| `mon2_raw_n` | Raw TLF fluorescence after per-sensor baseline subtraction (used in final model) |
| `tof_n` | Time-of-flight turbidity signal after per-sensor baseline subtraction |
| `censored` | TRUE if CBT result is right-censored (>= 100 CFU) |
| `agree` | TRUE if in-sample prediction is within the agreement band |
| `country` | Country of deployment (Kenya or Rwanda) |

## Model overview

The prediction model is a Tobit regression (right-censored at log10(101) = 2.004)
with EM estimation and ridge regularization (lambda = 0.1). Features are raw
(uncorrected) fluorescence (`mon2_raw`, baseline-subtracted, z-scored), turbidity
(`tof_n`, baseline-subtracted, z-scored), water temperature (`temp`, z-scored),
and per-sensor fixed effects (6 parameters total).

**Temperature as predictor**: Rather than pre-correcting fluorescence for temperature,
the model includes water temperature as an explicit predictor, giving the regression
more flexibility to capture temperature effects.

**Exclusion pipeline**: 13 documented Kind A outlier exclusions (anomalous
high-fluorescence readings in verified clean water, with mon2 values 2.5-3.1x the
sensor's clean-water median) plus automated IQR fencing (Q3 + 1.5 * IQR for
clean-water samples, Q3 + 5 * IQR for extreme values).

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
