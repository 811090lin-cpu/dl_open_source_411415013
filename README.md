# dl_open_source_411415013
Deep learning models for traffic volume prediction

# Traffic Flow Prediction

A modular pipeline for traffic forecasting using time-series features.

## Repository Architecture

### Current Structure (Issue Optimized)
- `train.py`: Data loading and cleaning utilities.
- `test.py`: Core inference logic and utility functions for model validation.

### Legacy Structure (Original Version)
- `traffic_train.py`: Integrated script for feature engineering and model training.
- `traffic_test.py`: Inference and evaluation script.

## Pipeline
1. **Load Data**: `train.py` cleans the raw traffic CSV.
2. **Feature Engineering**: `traffic_train.py` (Legacy) or the updated modules compute lags (`lag_1`, `lag_24`) and log-transformed weather features.
3. **Training & Eval**: Run scripts to train the model and generate metrics (MAE/RMSE).

## Quick Start
```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Run training (Original)
python traffic_train.py

# 3. Evaluate model (Original)
python traffic_test.py
