# Pair-Specific Time Headway Distributions and Covariate-Conditioned AFT Models

Analysis code for the study of car-following behaviour between pedal rickshaws (PR) and
battery-operated three-wheelers (BTW) in Dhaka's non-lane-based, heterogeneous traffic.

The work asks whether aggregating PR and BTW into one vehicle class hides real differences in
how they follow other vehicles, and estimates how kinematics, behaviour and site conditions
shift the headway distribution of each directional follower-leader pair.

**Manuscript:** *Pair-Specific Time Headway Distributions and Covariate-Conditioned AFT Models for
Pedal Rickshaws and Battery Operated Three Wheelers in Dhaka's Non-Lane-Based Heterogeneous
Traffic* - submitted to the Transportation Research Board Annual Meeting.

**Authors:** Tahsin Reza, Md Asif Raihan, Md Musleh Uddin Hasan
Department of Urban and Regional Planning / Accident Research Institute, BUET
[ORCID 0009-0006-0889-3589](https://orcid.org/0009-0006-0889-3589)

## Method in brief

Video data from two sub-saturated corridors in Dhaka yielded 229 PR and 669 BTW follower-leader
observations. Nine probability families were tested against the pooled and traffic-pressure-
stratified samples; six families were then fitted to eight directional follower-leader pairs.
Candidates were ranked by AIC and validated with refitted parametric-bootstrap
Kolmogorov-Smirnov and Anderson-Darling tests plus Cox-Snell residual diagnostics. Accelerated
failure time (AFT) models were then estimated with covariates for kinematics, following
behaviour, site conditions and width-standardised flow.

## Repository layout

```
notebooks/     the analysis pipeline, in run order
scripts/       standalone table and figure generators
development/   the exploratory sequence these were built from
figures/       Cox-Snell diagnostic plots for the selected AFT models
data/          where the input workbook goes (not committed)
```

### Pipeline order

Run the notebooks in `notebooks/` in this order - each reads tables written by the previous step:

| # | Notebook | What it does |
|---|---|---|
| 1 | `Headway.ipynb` | Baseline headway distributions, family fitting and comparison across the pooled and stratified samples |
| 2 | `Headway_41_Distribution_Covariate_Screening.ipynb` | Screens candidate covariates against each fitted distribution |
| 3 | `Headway_41_Sequential_AFT_Selection_VIF.ipynb` | Sequential AFT model selection with VIF checks for multicollinearity |
| 4 | `Headway_Final_AFT_Bootstrap_CoxSnell.ipynb` | Final AFT fits, parametric-bootstrap goodness of fit, Cox-Snell residual diagnostics |
| 5 | `Headway_Maximum_Covariate_Effect_Summary.ipynb` | Summarises the largest covariate effects across validated pairs |
| 6 | `Headway_Final_Run.ipynb` | End-to-end final run producing the reported tables |
| 7 | `Headway_graphs_4x2_CoxSnell_6_per_row.ipynb` | Assembles the Cox-Snell figure panels |

### Scripts

| Script | Purpose |
|---|---|
| `PR_BTW_Final_Figures_3x3_Forest_CoxSnell.py` | Forest plots and Cox-Snell panels for the manuscript |
| `generate_table8_from_T5_T6_data3.py` | Builds Table 8 from the T5/T6 outputs |
| `generate_table8_no_artifact_tool.py` | Same table without the `artifact_tool` dependency |

### development/

The numbered notebooks trace how the final pipeline was arrived at - association tests, early
pair distribution fits, Weibull AFT trials, covariate screening rounds, and successive versions
of the master pipeline. They are kept for provenance and are not needed to reproduce the results.
`development/scratch/` holds untitled working notebooks from the same period.

## Data

The input workbook is **not committed**. The notebooks expect a project folder containing
`data3.xlsx` plus the `Tables/` and `final run/tables/` output folders.

Point the code at that folder with an environment variable:

```bash
set HEADWAY_BASE_DIR=D:\Headway            # Windows
export HEADWAY_BASE_DIR=/path/to/Headway   # macOS / Linux
```

Without it, the notebooks fall back to `D:\Headway`. Several also search a list of candidate
locations before giving up, so a local `data/` copy may be picked up automatically.

Each row of the workbook is one follower-leader observation: the target vehicle's time headway
with its speed, speed difference, lateral placement, leader class, site and flow covariates.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
jupyter lab
```

Note that `scripts/generate_table8_from_T5_T6_data3.py` imports `artifact_tool`, which is not on
PyPI; use `generate_table8_no_artifact_tool.py` instead if you do not have it.

## Related work

- **Bachelor's thesis** - *Flow Characteristics of Traditional Rickshaws and Battery-Operated
  Three-Wheelers in Non-Lane-Based Heterogeneous Traffic of Dhaka*, DURP, BUET, 2026

More at [tahsinreza3109.github.io](https://tahsinreza3109.github.io)
