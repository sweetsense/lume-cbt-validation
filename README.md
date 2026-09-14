# lume-cbt-validation

Data and code for Knopp, Ecklu, Ross and Thomas, *Field validation of a tryptophan-like fluorescence sensor against the compartment bag test for microbial drinking water quality in rural water treatment programs in East Africa* (Water Research X, revised manuscript WROA-D-26-00467).

Every number in the paper can be recomputed from this repository.

## Contents

| Path | What it holds |
|---|---|
| `data/cbt-datagrid.csv` | mWater compartment bag test (CBT) records, May 25 to June 11, 2026. Enumerator names are replaced by codes and coordinates are removed. |
| `data/cbt_revision/raw/` | Sensor readings for units 50045, 50053 and 50065: full LED and SiPM bias sweeps, time-of-flight, and diagnostics including temperature (gzip JSON). |
| `data/cbt_revision/judgments.json` | Every case decision, with its reason, who made it and when: record exclusions, the one corrected CBT value, the in-water rule, replicates, temperature and baseline choices. |
| `data/cbt_revision/submitted/` | The submitted manuscript's paired table and results, used to reproduce the submitted numbers. |
| `scripts/cbt_revision/pair_soaks.py` | Records, time zones, the in-water rule, bucket immersions, pairing of each CBT record with sensor readings, and sample groups. |
| `scripts/cbt_revision/evaluate.py` | Temperature correction, daily clean-water baseline, models, held-out validation, performance measures, index of agreement, reference uncertainty, comparison models, and the export used by the paper. |
| `paper/` | `paper_data.json`, `paired_observations.csv`, `make_figures.py` and the figures. |
| `data/cbt_revision/*.csv`, `*.json` | Intermediate and final outputs written by the scripts. |

## Reproduce

Requires Python 3.9 or later with numpy, pandas and matplotlib.

```
python3 scripts/cbt_revision/pair_soaks.py
CBT_PAPER_DIR=paper python3 scripts/cbt_revision/evaluate.py step5 export
cd paper && python3 make_figures.py
```

`python3 scripts/cbt_revision/evaluate.py 0` refits the submitted manuscript's model on its own paired table and prints its reported results next to the recomputed ones.

## What is evaluated

The Lume (Virridy) as one method: the instrument's fluorescence, time-of-flight and temperature channels together with the analytics that turn them into an *E. coli* estimate. The analytics correct fluorescence to 20 °C, reference fluorescence and time-of-flight to the unit's clean-water baseline for the day, and fit one Tobit regression of log10(MPN/100 mL + 1), right-censored at 100 MPN/100 mL, with coefficients shared by all units. Performance is measured with the sensor and the sampling day held out together, on 75 samples.
