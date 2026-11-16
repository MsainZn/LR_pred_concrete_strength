# Task 3: Machine Learning with Linear Regression
**Predicting the compressive strength of concrete from its recipe and age.**
IU DLBDSIDS01-02 Introduction to Data Science, Oral Assignment.

## Contents
| Path | What it is |
|---|---|
| `Task3_LinearRegression_Concrete.ipynb` | The analysis, already executed. Part 1: ML paradigms, Part 2: how linear regression works, Part 3: statistical analysis first, then model, tuning, test, bias |
| `src/` | The dataset: `concrete.csv` (1,030 rows, 8 features + strength) and the original R file it was exported from |
| `data/` | Identical copy of `src/`. The notebook reads `data/` if it exists, otherwise `src/` |
| `figures/` | The 12 figures as PNGs, numbered in notebook order |
| `outputs/` | Result tables (CSV) and the trained model (`final_model.joblib`) |
| `pyproject.toml`, `uv.lock`, `.python-version` | Environment: Python 3.13 + exact tested library versions |
| `requirements.txt` | Same libraries for a plain `pip install -r` |
| `_old_bodyfat_version/` | The previous body-fat version, kept only so nothing is lost. Safe to delete |

## Setup on a Mac with Apple Silicon
All libraries ship native arm64 wheels (no Rosetta, nothing to compile).

**Option A: uv (recommended)**
```bash
brew install uv
cd ~/Desktop/data-course/Task3_LinearRegression
uv sync
uv run jupyter lab Task3_LinearRegression_Concrete.ipynb
```

**Option B: venv + pip**
```bash
cd ~/Desktop/data-course/Task3_LinearRegression
python3 -m venv .venv && source .venv/bin/activate
pip install .
jupyter lab Task3_LinearRegression_Concrete.ipynb
```
Then choose *Kernel → Restart Kernel and Run All Cells*. The full run takes about 15 seconds.

## Key results (test set: 205 tests of 86 mixes the model never saw)
| Model | MAE (MPa) | RMSE (MPa) | R² |
|---|---|---|---|
| Naive benchmark (always the mean) | 14.4 | 17.8 | 0.00 |
| Plain linear regression | 8.5 | 10.7 | 0.64 |
| Linear regression + log(age) | 5.4 | 7.4 | 0.83 |
| Tuned (log(age), degree 2, Ridge) | 5.2 | 7.4 | 0.83 |

The statistical analysis showed that strength grows with the *logarithm* of age. That one transform did most of the work, and the tuned model is no better than the simple one on unseen mixes.

## Data source
Yeh, I-C. (1998). *Concrete Compressive Strength* [Dataset]. UCI Machine Learning Repository, CC BY 4.0, https://doi.org/10.24432/C5PK67. Paper: *Cement and Concrete Research* 28(12), 1797–1808. The CSV was exported unchanged from the R package *AppliedPredictiveModeling* (Kuhn & Johnson, 2013).
