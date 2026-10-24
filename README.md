# MMA3001-Project: Environmental Feature Importance on Occupancy Modelling

This repository is for the project submission for MMA3001 – Numerical Methods and Machine Learning (Monash University).

It predicts the **headcount of a room from indoor environmental sensor readings** (CO₂, temperature, humidity, formaldehyde, pressure, VOCs and particulate matter) and identifies **which environmental variable is the strongest predictor of occupancy**.

## Engineering problem

Smart buildings need occupancy information to control ventilation, thermal comfort, indoor air quality and energy use. Dedicated occupancy sensors can be unreliable or expensive, while environmental sensors are already installed in many buildings. This project tests whether environmental readings alone can estimate how many people are in a room, and which measurement carries the most information about occupancy.

**Scope:** one room (`941f0352`) and one environmental sensor (`6012002000869`), 29 February – 6 May 2024. These are the only room and sensor whose data overlap in time (see `DATA/README.md`).

## Key results

| Model | Role | RMSE (people) | MAE (people) | R² |
|---|---|---|---|---|
| Polynomial regression (degree 2) | Baseline | 0.869 | 0.667 | 0.272 |
| Gradient Boosting | Alternative | 0.786 | 0.600 | 0.403 |
| Random Forest (default) | Alternative | 0.720 | 0.544 | 0.500 |
| **Random Forest (tuned)** | **Final model** | **0.710** | **0.538** | **0.513** |

**Strongest predictor: CO₂.** It ranks first on all three measures (leave-one-feature-out sensitivity, model feature importance and correlation with headcount). Removing it causes the largest loss in accuracy (R² −0.110). Overall ranking: CO₂, formaldehyde, humidity, temperature, pressure, VOC, then particulate matter (no predictive value). The sensor has no light-level channel.

## Repository structure

```
MMA3001-Project/
├── DATA/                      Input data (not committed; see DATA/README.md)
├── Jupyter Notebooks/
│   └── Occupancy_Modelling.ipynb   Step-by-step analysis with explanations
├── src/
│   └── occupancy_model.py     All functions plus a command-line pipeline
├── docs/                      HTML documentation (generated with pdoc)
├── outputs/                   Result tables (CSV) and figures (PNG)
├── tests/                     Tests
├── requirements.txt           Python dependencies
├── LICENSE
└── README.md
```

## Method

1. **Load data:** occupancy events for room 941f0352; environmental readings for sensor 869, parsed from JSON and timed by *measurement* time (the `createdate` column is the upload time, often months later).
2. **Align:** each 10-minute environmental reading is paired with the **time-weighted average headcount** over the preceding 10 minutes, treating headcount as a step function. Windows with less than 80% occupancy coverage are dropped. This gives 4,009 paired readings.
3. **Split:** 70% training (2,806 readings), 30% test (1,203 readings).
4. **Model:** degree-2 polynomial regression (baseline), Random Forest and Gradient Boosting. The Random Forest is tuned using out-of-bag convergence (300 trees) and 5-fold cross-validated grid search on the training set only.
5. **Select:** the model with the lowest test RMSE becomes the final model.
6. **Sensitivity analysis:** accuracy across train/test split sizes and seeds; leave-one-feature-out on the final model (change in RMSE, MAE and R²).
7. **Predict:** headcount for all 13,800 sensor readings, flagging readings outside the training dates or the training range of any variable.

## Setup

Requires Python 3.10 or later.

```
git clone https://github.com/HarperVicic/MMA3001-Project.git
cd MMA3001-Project
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Then place the three data files in `DATA/` (see `DATA/README.md`).

## Usage

**Run the full pipeline** from the repository root (about 4 minutes; writes all tables and figures to `outputs/`):

```
python src/occupancy_model.py
python src/occupancy_model.py --no-tune      # skip Random Forest tuning (about 1 minute)
python src/occupancy_model.py --data-dir DATA --output-dir outputs
```

**Step through the analysis** in `Jupyter Notebooks/Occupancy_Modelling.ipynb` (select the `.venv` kernel and Run All).

**Use the functions** in your own code:

```python
import sys; sys.path.append("src")
from occupancy_model import load_occupancy, load_environmental, build_modelling_dataset
```

## Outputs

| File | Contents |
|---|---|
| `outputs/predicted_headcount.csv` | Predicted headcount for every sensor reading, with measured headcount, train/test/unlabelled label and reliability flags |
| `outputs/model_performance_metrics.csv` | RMSE, MAE and R² for every model (the script also records training time, prediction time and model size) |
| `outputs/sensitivity_analysis_results.csv` | Change in RMSE, MAE and R² when each variable is removed |
| `outputs/feature_importance_ranking.csv` | Variables ranked by sensitivity, model importance and correlation |
| `outputs/figures/` | Time series, correlation heatmap, polynomial fit, sensitivity analysis, predicted headcount |

## Documentation

HTML documentation for every function (parameters, returns, errors and examples) is in `docs/`; open `docs/index.html` in a browser. To regenerate it after changing the code:

```
pdoc src/occupancy_model.py --docformat numpy -o docs
```

## Testing

The docstring examples can be run as tests:

```
python -m doctest -v src/occupancy_model.py
```

## Limitations

* One room and one sensor: results may not generalise to other rooms, ventilation systems or seasons.
* The registry does not map occupancy rooms, so the room was chosen by correlation with the sensor's CO₂.
* The occupancy data is truncated at 6 May 2024.
* The random train/test split lets readings 10 minutes apart fall on both sides, which can make test scores optimistic.
* Over half of the readings without a measured headcount fall outside the training range (mainly formaldehyde), where the model cannot extrapolate.

## Summary of AI Usage in Project
This project will utilize AI tools, specifically Google Gemini (via Colab integration, 'Gemini-2.5-Flash', July-October 2026) and Microsoft Copilot (web interface, July-October 2026), to assist across various project phases. Gemini will primarily support code generation, debugging, and conceptual explanations, while Copilot will aid in refining written documentation and report structuring.

All AI-generated content, including code and text, will undergo thorough human review, validation, and adaptation to ensure accuracy, efficiency, and alignment with project requirements. The author will maintain full oversight and responsibility for all final decisions, implementations, and conclusions, ensuring the ethical and professional standards expected in engineering are met.

### Claude (Anthropic)

Claude (Anthropic, via the Claude desktop app, October 2026) was also used during the project, for:

* investigating the datasets (identifying the upload-time vs measurement-time issue, the frozen sensors, the truncated occupancy export and the mixing of five rooms into one time series);
* restructuring the notebook code into documented functions (`src/occupancy_model.py`) and writing docstrings;
* implementing the 10-minute time-weighted alignment, the Random Forest tuning and the project outputs section;
* setting up the repository structure and this README.

All Claude-generated code was run and checked against the notebook results, and the author made the modelling decisions (choice of room and sensor, variables, baseline and final model).
