# Trajectory Prediction


## Sturcture
```
├── data
│   ├── PandasData
│   │   ├── Modified (Data with Feature)
│   │   ├── Original (Raw data)
│   │   ├── Predicted (Data with FAM prediction)
│   │   └── Sampled (Down-sampled data to target frequency)
│   └── state_data.pkl
├── model (logs)
│   ├── loss_baseline.npy
│   ├── loss_state.npy
│   └── TFT_logs
│       └── lightning_logs
├── notebook
│   ├── data_to_features.ipynb (to use generate features)
│   ├── model_eval.ipynb
│   ├── pred_model.ipynb (Baseline LSTM Model)
│   ├── pred_model_state.ipynb (LSTM with state prediction)
│   ├── state_predicton_models.ipynb (LightGBM for state prediction)
│   └── TFT.ipynb (Temporal Fusion Transformer Model, Pytorch integrated time series forecasting model)
└── src
    ├── constant.py (Constants, threshold for feature generator)
    ├── FeatureGenerator.py (Feature generator pipeline)
    ├── MyDataset.py (Customized Data Loader)
    ├── TraPredModel_new.py (LSTM Model with encoder-decoder structure)
    ├── TraPredModel.py (LSTM Model)
    └── utils.py (Helper functions)

```