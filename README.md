# dl_open_source_411415013
## Traffic Flow Prediction
Deep learning models for traffic volume prediction
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

## Dataset Overview
The **Metro Interstate Traffic Volume** dataset provides historical hourly traffic records for Interstate 94, documenting the relationship between traffic flow and various meteorological factors.

* **Source**: [Metro Interstate Traffic Volume (Kaggle)](https://www.kaggle.com/datasets/pooriamst/metro-interstate-traffic-volume)

### Feature Definitions
| Feature | Type | Description |
| :--- | :--- | :--- |
| `date_time` | Temporal | The chronological timestamp of the observation. |
| `traffic_volume` | Target | Total number of vehicles passing the station per hour. |
| `holiday` | Categorical | Indicates if the date is a US public holiday or contains a major event. |
| `temp` | Meteorological | Average temperature in Kelvin. |
| `rain_1h` | Meteorological | Amount of precipitation (rain) in millimeters occurred in the hour. |
| `snow_1h` | Meteorological | Amount of snowfall in millimeters occurred in the hour. |
| `clouds_all` | Meteorological | Percentage of cloud cover. |
| `weather_main` | Categorical | Short textual category (e.g., Clear, Clouds, Rain, Snow, Mist). |
| `weather_description` | Categorical | Detailed, specific textual description of the current weather. |

### Feature Categories
* **Temporal Features**: The `date_time` column enables the extraction of seasonal, weekly, and daily cycles.
* **Meteorological Features**: `temp`, `rain_1h`, `snow_1h`, and `clouds_all` provide quantitative environmental data that influence traffic flow.
* **Categorical Features**: `holiday`, `weather_main`, and `weather_description` provide the qualitative context necessary for understanding specific temporal records.

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
