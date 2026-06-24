# dl_open_source_411415013
Deep learning models for traffic volume prediction
# Traffic Flow Prediction

A modular pipeline for traffic forecasting using time-series features.

## Files
- `train.py`: Data loading and cleaning utilities.
- `traffic_train.py`: Integrated script for feature engineering and model training.
- `traffic_test.py`: Inference and evaluation script.

## Pipeline
1. **Load Data**: `train.py` cleans the raw traffic CSV.
2. **Feature Engineering**: `traffic_train.py` computes lags (`lag_1`, `lag_24`) and log-transformed weather features.
3. **Training & Eval**: Run scripts to train the model and generate metrics (MAE/RMSE).

## Quick Start
```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Run training
python traffic_train.py

# 3. Evaluate model
python traffic_test.py
