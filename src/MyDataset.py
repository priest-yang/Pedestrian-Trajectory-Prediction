import pandas as pd
import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import torch.utils.data
        
class MyDataset():
    def __init__(self, lookback=None) -> None:
        self.data = None
        self.train = None
        self.test = None
        self.feature_dim = None
        self.lookback = lookback
        self.dataset : list[pd.DataFrame] = []

    def read_data(self, df: pd.DataFrame, agv_col_name = 'AGV_name'):
        ''' Read data from a pandas dataframe and create a dataset for training, if the data is not None, it will be concatenated with the new data
        Args:
            df: A pandas dataframe
            agv_col_name: The column name of the AGV name
        
        '''
        if self.lookback is None:
            raise ValueError("Lookback is not set, use set_lookback() to set the lookback window size")

        agv_list = df[agv_col_name].unique()
        for agv in agv_list:
            cur_data = df[df[agv_col_name] == agv]
            # cur_data = cur_data.select_dtypes(include=[np.number])
            cur_data.drop(columns=[agv_col_name], inplace=True)
            cur_data = cur_data.astype(np.float32)
            if self.feature_dim is None:
                self.feature_dim = cur_data.shape[1]
            else:
                assert self.feature_dim == cur_data.shape[1], f"Feature dimension should be the same. now under {cur_data.shape[1]} features, but previous data has {self.feature_dim} features. \nPrevious columns: {self.dataset[0].columns}. \nGiven features are {cur_data.columns}"
        
            self.dataset.append(cur_data)

        # X, y = self.create_dataset(cur_data.values, lookback=self.lookback)
        # if self.data is None:
        #     self.data = TensorDataset(X, y)
        # else:
        #     self.data = torch.utils.data.ConcatDataset([self.data, TensorDataset(X, y)])

    def normalize_dataset(self):
        if self.dataset is None:
            raise ValueError("Dataset is empty, please read data first")
        
        concatenated_data = pd.concat(self.dataset)
        # normalize
        mean = concatenated_data.mean()
        std = concatenated_data.std()

        # do not normalize first 4 columns
        mean[:4] = 0
        std[:4] = 1

        self.dataset = [(data - mean) / std for data in self.dataset]

        # standardize
        concatenated_data = pd.concat(self.dataset)
        # recompute the min and max in the new combined data
        min_ = concatenated_data.min()
        max_ = concatenated_data.max()

        # do not standardize first 4 columns
        min_[:4] = -500 # -5m is the min value of the first 4 columns
        max_[:4] = 1000 # 10m is the max value of the first 4 columns

        self.dataset = [(data - min_) / (max_ - min_) for data in self.dataset]

        return {'mean': mean, 'std': std, 'min': min_, 'max': max_}

    def generate_data(self, return_list = False, future_steps = None) -> None:
        ''' Generate data for training
        '''
        if self.dataset is None:
            raise ValueError("Dataset is empty, please read data first")
        
        X, y = [], []
        for data in self.dataset:
            X_data, y_data = self.create_dataset(data, lookback=self.lookback, future_steps=future_steps)
            X.append(X_data)
            y.append(y_data)
        
        X_cat = torch.cat(X)
        y_cat = torch.cat(y)
        self.data = TensorDataset(X_cat, y_cat)
        
        if return_list:
            return X, y
    

    @staticmethod
    def create_dataset(dataset, lookback, future_steps=None):
        """Transform a time series into a prediction dataset
        Args:
            dataset: A numpy array of time series, first dimension is the time steps
            lookback: Size of window for prediction
            future_steps: Number of future steps to predict
        """
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.select_dtypes(include=[np.number]).values
        
        if not future_steps:
            future_steps = lookback

        X, y = [], []
        for i in range(len(dataset)-future_steps-lookback+1):
            feature = dataset[i:i+lookback]
            target = dataset[i+lookback:i+lookback+future_steps]
            X.append(feature)
            y.append(target)
        return torch.tensor(X), torch.tensor(y)
    
    
    def split_data(self, frac: float = 0.8, shuffle: bool = True, train_batch_size: int = 4, test_batch_size:int = 16):
        n = len(self.data)
        train_size = int(n * frac)
        test_size = n - train_size

        # train = torch.utils.data.Subset(self.data, range(0, train_size))
        # test = torch.utils.data.Subset(self.data, range(train_size, n))

        # random split
        train, test = torch.utils.data.random_split(self.data, [train_size, test_size])
        
        train = DataLoader(train, batch_size=train_batch_size, shuffle=shuffle)
        test = DataLoader(test, batch_size=test_batch_size, shuffle=False)
        
        self.train = train
        self.test = test

        return train, test
    
    def set_lookback(self, lookback: int):
        self.lookback = lookback

    @staticmethod
    def normalize(data:torch.utils.data.DataLoader, scaler=None):
        """Normalize the data
        Args:
            data: A torch DataLoader
            scaler: A sklearn scaler, if None, MinMaxScaler will be used
        """
        if scaler is None:
            scaler = MinMaxScaler()
        for i, (X, y) in enumerate(data):
            X = X.view(-1, X.shape[-1])
            y = y.view(-1, y.shape[-1])
            scaler.partial_fit(X)
            scaler.partial_fit(y)
        return scaler
    

import pickle
import os
def save_dataset(dataloader: torch.utils.data.DataLoader, file_path: str = None, type: str = 'train'):
    dataset = dataloader.dataset  # Extract dataset
    sampler = dataloader.sampler if dataloader.sampler is not None else None

    # Save dataset (if possible) and dataloader params
    save_dict = {
        "dataset": dataset,  # Only works if dataset is serializable
        "sampler": sampler,  # Only works if sampler is serializable
        "batch_size": dataloader.batch_size,
        "shuffle": dataloader.shuffle if hasattr(dataloader, 'shuffle') else False,
        "num_workers": dataloader.num_workers,
        "drop_last": dataloader.drop_last,
    }

    # Save to a file
    if file_path is None:
        os.makedirs("../data/.cache", exist_ok=True)
        file_path = f"../data/.cache/{type}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)    

    print(f"Saving to {file_path}")
    with open(file_path, "wb") as f:
        pickle.dump(save_dict, f)

# load
from typing import Optional
def load_dataset(file_path: str, batch_size: int = None) -> Optional[torch.utils.data.DataLoader]:
    with open(file_path, "rb") as f:
        load_dict = pickle.load(f)

    # Extract dataset and parameters
    dataset = load_dict["dataset"]
    sampler = load_dict["sampler"]
    batch_size = load_dict["batch_size"] if batch_size is None else batch_size
    shuffle = load_dict["shuffle"]
    num_workers = load_dict["num_workers"]
    drop_last = load_dict["drop_last"]

    # Recreate DataLoader
    from torch.utils.data import DataLoader

    new_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # If a sampler exists, shuffle should be False
        # sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    return new_dataloader