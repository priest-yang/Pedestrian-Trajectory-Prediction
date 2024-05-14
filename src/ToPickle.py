import pandas as pd
import os

files = os.listdir(os.path.join('..', 'data', 'PandasData', 'Original'))

for file in files:
    file_path = os.path.join('..', 'data', 'PandasData', 'Original', file)
    data = pd.read_csv(file_path)
    filename = file.split('.')[0]
    out_file = os.path.join('..', 'data', 'PandasData', 'Original', f'{filename}.pkl')
    data.to_pickle(out_file)
