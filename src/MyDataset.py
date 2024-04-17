import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

class MyDataset():
    def __init__(self, lookback=None) -> None:
        self.data = None
        self.train = None
        self.test = None
        self.feature_dim = None
        self.lookback = lookback

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
            cur_data = cur_data.select_dtypes(include=[np.number])
            if self.feature_dim is None:
                self.feature_dim = cur_data.shape[1]
            else:
                assert self.feature_dim == cur_data.shape[1], "Feature dimension should be the same"
            
            X, y = self.create_dataset(cur_data.values, lookback=self.lookback)
            if self.data is None:
                self.data = TensorDataset(X, y)
            else:
                self.data = torch.utils.data.ConcatDataset([self.data, TensorDataset(X, y)])
    
    @staticmethod
    def create_dataset(dataset, lookback):
        """Transform a time series into a prediction dataset
        Args:
            dataset: A numpy array of time series, first dimension is the time steps
            lookback: Size of window for prediction
        """
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.select_dtypes(include=[np.number])
            
        X, y = [], []
        for i in range(len(dataset)-lookback):
            feature = dataset[i:i+lookback]
            target = dataset[i+1:i+lookback+1]
            X.append(feature)
            y.append(target)
        return torch.tensor(X), torch.tensor(y)
    
    
    def split_data(self, frac: float = 0.8, shuffle: bool = True, batch_size: int = 4):
        n = len(self.data)
        train_size = int(n * frac)
        test_size = n - train_size

        # Shawn: not sure if we should shuffle here

        # if shuffle:
        #     train, test = torch.utils.data.random_split(self.data, [train_size, test_size])
        # else:
        #     train = torch.utils.data.Subset(self.data, range(0, train_size))
        #     test = torch.utils.data.Subset(self.data, range(train_size, n))
        
        train = torch.utils.data.Subset(self.data, range(0, train_size))
        test = torch.utils.data.Subset(self.data, range(train_size, n))
        
        train = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
        test = DataLoader(test, batch_size=batch_size, shuffle=shuffle)
        
        self.train = train
        self.test = test

        return train, test
    
    def set_lookback(self, lookback: int):
        self.lookback = lookback