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
            cur_data.drop(columns=["AGV_name"], inplace=True)
            cur_data = cur_data.astype(np.float32)
            if self.feature_dim is None:
                self.feature_dim = cur_data.shape[1]
            else:
                assert self.feature_dim == cur_data.shape[1], f"Feature dimension should be the same. now under {cur_data.shape[1]} features, but previous data has {self.feature_dim} features. Given features are {cur_data.columns}"
        
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
        self.dataset = [(data - mean) / std for data in self.dataset]

        # standardize
        concatenated_data = pd.concat(self.dataset)
        # recompute the min and max in the new combined data
        min_ = concatenated_data.min()
        max_ = concatenated_data.max()
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

        train = torch.utils.data.Subset(self.data, range(0, train_size))
        test = torch.utils.data.Subset(self.data, range(train_size, n))
        
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