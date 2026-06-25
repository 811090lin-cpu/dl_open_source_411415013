# dl_open_source_411415013
Deep learning models for traffic volume prediction

# Traffic Flow Prediction

A modular pipeline for traffic forecasting using time-series features.

## Repository Structure

```text
├── Metro_Interstate_Traffic_Volume.csv  # Dataset
├── train.py                             # Current: Data loading & cleaning
├── test.py                              # Current: Core inference & validation logic
├── traffic_train.py                     # Legacy: Integrated training script
├── traffic_test.py                      # Legacy: Inference & evaluation script
│
├── model/                               # Current: Optimized model assets
│   ├── history.json                     # Training history
│   ├── model.keras                      # Model weights
│   ├── model_scaler_X.pkl               # Input scaler
│   └── model_scaler_y.pkl               # Target scaler
│
└── traffic_model/                       # Legacy: Original model assets
    ├── traffic_history.json             # Training history
    ├── traffic_model.keras              # Model weights
    ├── traffic_model_scaler_X.pkl       # Input scaler
    └── traffic_model_scaler_y.pkl       # Target scaler
```

### Current Structure (Issue Optimized)
- `train.py`: Data loading and cleaning utilities.
- `test.py`: Core inference logic and utility functions for model validation.

### Legacy Structure (Original Version)
- `traffic_train.py`: Integrated script for feature engineering and model training.
- `traffic_test.py`: Inference and evaluation script.

## Architectural Philosophy

* **Multi-Scale Temporal Extraction**: By utilizing parallel CNN branches with distinct kernel sizes (3 and 5), the model extracts local fluctuations at different temporal granularities simultaneously. This allows the system to be sensitive to both sudden traffic bursts and gradual volume shifts.
* **Bidirectional Context Awareness**: The integration of Bidirectional LSTM is central to the design. By processing the sequence in both forward and backward directions, the model gains a comprehensive understanding of how past traffic trends and potential future states interact, which is critical for modeling cyclic daily traffic patterns.
* **Hybrid Feature Fusion**: The architecture acts as a hierarchical pipeline—where the CNN acts as a spatial-temporal encoder for raw sequences, and the subsequent LSTM layers interpret these encoded features to model long-term dependencies. The use of Batch Normalization between these stages ensures that the high-level features passed to the LSTM are robust and effectively scaled.

## Quick Start
```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Run training (Original)
python traffic_train.py

# 3. Evaluate model (Original)
python traffic_test.py
