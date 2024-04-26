import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import circmean


keys_to_average = ["User_X", "User_Y", "User_Z", "U_X", "U_Y", "U_Z",
                   "AGV_X",	"AGV_Y", "AGV_Z", "AGV_speed",
                   "GazeOrigin_X", "GazeOrigin_Y", "GazeOrigin_Z",
                   "GazeDirection_X", "GazeDirection_Y", "GazeDirection_Z",	"Confidence"]
keys_of_angular_data = ["User_Pitch", "User_Yaw", "User_Roll", "AGV_Pitch", "AGV_Yaw", "AGV_Roll"]

keys_to_not_average = ['PID', 'SCN', 'TimestampID', "Timestamp", "AGV_name"]
new_keys = ['FrameID']


def average_data_per_framerate(df: pd.DataFrame, framerate: int):
    timestamp_ids = df['TimestampID'].unique()
    count = 0
    new_df = {}
    for key in df.keys():
        new_df[key] = []
    for key in new_keys:
        new_df[key] = []

    del new_df['DatapointID']

    for timestamp_id in tqdm(timestamp_ids):
        frame_id = 1
        timestamp_df = df[df["TimestampID"] == timestamp_id]
        num_datapoints = len(timestamp_df.index)
        window_size = int(72 / framerate)
        start_idx = 0
        end_idx = window_size
        while window_size > 0 and end_idx <= num_datapoints:
            for key in keys_to_average:
                new_df[key].append(np.mean(timestamp_df[key].iloc[start_idx:end_idx]))
            for key in keys_to_not_average:
                new_df[key].append(timestamp_df[key].iloc[end_idx-1])
            for key in keys_of_angular_data:
                new_df[key].append(circmean(timestamp_df[key].iloc[start_idx:end_idx], high=180, low=-180))
            for key in new_keys:
                new_df[key].append(frame_id)

            frame_id += 1
            start_idx = end_idx
            end_idx += window_size

        if start_idx < num_datapoints < end_idx:
            for key in keys_to_average:
                new_df[key].append(np.mean(timestamp_df[key].iloc[start_idx:-1]))
            for key in keys_to_not_average:
                new_df[key].append(timestamp_df[key].iloc[-1])
            for key in keys_of_angular_data:
                new_df[key].append(circmean(timestamp_df[key].iloc[start_idx:-1], high=180, low=-180))
            for key in new_keys:
                new_df[key].append(frame_id)

        # count += 1
        # if count > 3:
        #     break

    for key, value in new_df.items():
        print(key, len(value))
    new_df = pd.DataFrame(new_df)
    return new_df


def main():
    filepath = os.path.join('..', 'data', 'PandasData', 'Original', 'PID001_NSL.pkl')
    df = pd.read_pickle(filepath)
    framerate = 24

    df.drop(columns=['EyeTarget'], inplace=True)
    new_df = average_data_per_framerate(df, framerate)

    out_filepath = os.path.join('..', 'data', 'PandasData', 'Modified', f'PID001_NSL_framerate_{framerate}.csv')
    new_df.to_csv(out_filepath, index=False)

    pickle_filepath = os.path.join('..', 'data', 'PandasData', 'Modified', f'PID001_NSL_framerate_{framerate}.pkl')
    new_df.to_pickle(pickle_filepath)


if __name__ == '__main__':
    main()
