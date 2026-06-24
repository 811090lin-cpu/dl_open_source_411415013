# dl_open_source_411415013
Deep learning models for traffic volume prediction
# Traffic Flow Prediction: Feature Engineering Optimization

This project focuses on enhancing traffic flow prediction accuracy through advanced feature engineering and automated data preprocessing techniques.

---

## Table of Contents
1. [Project Goal](#project-goal)
2. [Data Structure & Pipeline](#data-structure--pipeline)
3. [Optimization: Before vs. After](#optimization-before-vs-after)
4. [Usage](#usage)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Dependencies](#dependencies)

---

## Project Goal
This project aims to resolve data bottlenecks in traffic forecasting models, specifically addressing temporal dependency loss and high-dimensional noise. By implementing structured feature optimization, we achieve a more stable and accurate prediction system.

## Data Structure & Pipeline
The input data consists of raw traffic volume, weather logs, and timestamps. Our pipeline processes them as follows:

* **Raw Data Input**: 
    * `traffic_data.csv`: [Timestamp, Flow_Volume]
    * `weather_data.csv`: [Timestamp, Condition, Precipitation]
* **Data Processing Flow**:
    1. **Alignment**: Merging traffic and weather data by timestamp.
    2. **Transformation**: Applying feature engineering to convert raw inputs into model-ready tensors.



## Optimization: Before vs. After

| Feature | Baseline (Before) | Optimized (After) | Impact |
| :--- | :--- | :--- | :--- |
| **Temporal Context** | Single timestep only | `lag_1` & `lag_24` | Captures cyclic daily/hourly patterns |
| **Weather Encoding** | High-dimensional One-hot | Weighted Numerical Mapping | Reduced noise and feature dimensionality |
| **Data Distribution** | Raw precipitation values | `log1p` transformed values | Stabilized model training & convergence |

## Usage

### 1. Installation
```bash
pip install -r requirements.txt
