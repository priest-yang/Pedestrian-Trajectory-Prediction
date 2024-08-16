# Pedestrian Trajectory Prediction
This project aims at predict the trajectory for pedestrians in a manufacturing plants. 
The data was collected from a virtual environments. 



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

## Model


### [Source Files](src/)
- VQ-VAE
- Temporal Fusion Transformer
- baseline Constnt Velocity Model and baseline LSTM
- FAM for pedestrian state classify
- Feature generator

### Script
The pipeline of data preparation - model training - validation could be found at [`notebook`](nootbook/) folder. 

### Visualization
We built an interactive visualization of trajectory prediction model by **Plotly**. You can found the script at [`visualization`](notebook/visualization.ipynb). 


## Real-time Pipeline in C++ environments

In the future, we will integrate the model into unreal engine. Here we put a simulator in C++ environments as a demo. The program will read off-line features from ``/demo`` folder line-by-line then update a buffer in a **FIFO** way. The **JIT** model will be tested in a process. 


To run demo, in the folder ``pipeline/`` 

- follow the ``libtorch/README.md`` to insall libtorch. 
- change ``-DCMAKE_PREFIX_PATH`` in ``run.sh`` to the path of libtorch
- adjust ``main`` function in ``main.cpp`` for model path, data path, feature dim, prediction windows etc.


run the following commands to test:
```shell
mkdir build && cd build
sh ../run.sh
./real-time-sim
``` 

you will see sth like:
``` shell
Elapsed time: 0.760805 seconds.
Processed 608 lines.
Speed: 799.153 lines per second.
```
