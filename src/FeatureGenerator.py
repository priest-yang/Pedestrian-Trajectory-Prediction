from utils import *
from constant import *

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import pandas as pd
import os
import numpy as np
import math
import os
import glob
from typing import Literal
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')



class FeatureGenerator:
    def __init__(self, raw_data_path = None, save_data_path = None, GT_path = '../../Inter_test/data/behavior_inter_rater_Shawn.csv', GT_only = False, to : Literal['csv', 'pkl']= 'csv') -> None:
        '''
        Args:
            raw_data_path: path to the raw data
            save_data_path: path to save the processed data
            GT_path: path to the ground truth file
            GT_only: if True, only generate ground truth files to accelerate the process
        '''

        
        self.current_directory = os.getcwd()
        if raw_data_path is not None:
            self.data_directory = raw_data_path
        else:
            self.data_directory = os.path.join(
                self.current_directory, "data", "PandasData", "Original")
        if save_data_path is not None:
            self.out_directory = save_data_path
        else:
            self.out_directory = os.path.join(
                self.current_directory, "data", "PandasData", "Modified")

        if not os.path.exists(self.out_directory):
            os.makedirs(self.out_directory)
        for f in glob.glob(os.path.join(self.out_directory, "*.pkl")):
            os.remove(f)

        self.files = os.listdir(self.data_directory)

        self.features = ["AV_distance", "AV_speed", "Wait_time",
                    "Gaze_ratio", "Curb_distance", "Ped_speed"]

        # columns in the data that are important to compute the features
        self.keys = ['TimestampID', 'Timestamp', 'AGV_name',
                'User_X', 'User_Y', 'AGV_X', 'AGV_Y', 'AGV_speed']
        self.eye_data_keys = ['TimestampID', 'EyeTarget']

        self.GT_only = GT_only
        if GT_only:
            if GT_path is None:
                GT_path = '../../Inter_test/data/behavior_inter_rater_Shawn.csv' # default path
            try:
                self.GT = pd.read_csv(GT_path)
            except:
                raise FileNotFoundError("Ground truth file not found")

            self.GT = self.GT[['PID', 'AGV_name', 'Condition']].drop_duplicates()
            test_condition_df = self.GT[['PID', 'Condition']].drop_duplicates(
                subset=['PID', 'Condition'], keep='last')
            self.files = [f"PID{str(row['PID']).zfill(3)}_{row['Condition']}.pkl" for _,
                         row in test_condition_df.iterrows()]
            
        self.to = to


    def generate_features(self, data_aug = False, lidar_range=20, camera_range=15, FRAMERATE = 70, to :Literal['pkl', 'csv'] = None):
        '''
        Generate features for the data
        Args:
            data_aug: if True, augment the data
            lidar_range: range of the lidar
            camera_range: range of the camera
            FRAMERATE: frame rate of the RAW data
            to: format to save the data, either 'pkl' or 'csv'
        '''

        to = to if to in ['pkl', 'csv'] else self.to

        self.frame_rate = FRAMERATE

        for file in tqdm(self.files):
            if file.endswith(".pkl") == False: 
                continue
            
            data_details = file.split("_")
            pid = int(data_details[0][-3:])
            scn = data_details[1].split('.')[0]

            df = pd.read_pickle(os.path.join(self.data_directory, file))

            if self.GT_only:
                AGV_num = self.GT[(self.GT['PID'] == pid) & (
                    self.GT['Condition'] == scn)]['AGV_name'].unique()
                AGV_list = ["AGV"+str(i) for i in AGV_num]
                df = df[df['AGV_name'].isin(AGV_list)]

            out_df = pd.DataFrame()
            # keep non-numerical raw features from original data
            df["Timestamp"] = pd.to_timedelta(df["Timestamp"])
            # Process the data
            if data_aug:
                df = self.data_aug_helper(df, lidar_range=lidar_range, camera_range=camera_range)
                if df.shape[0] == 0: # if the data is empty after data augmentation, skip
                    continue
            
            out_df = self.process_data_gm(df, [
                (self.generate_AGV_User_distance, (), {}),
                (self.generate_AGV_speed, (), {'frame_rate': FRAMERATE}),
                (self.generate_user_speed, (), {'frame_rate': FRAMERATE}),
                (self.generate_wait_time, (), {'H1': 0.2, 'H2': 0.1,
                 'THRESHOLE_ANGLE': GAZING_ANGLE_THRESHOLD, 'frame_rate': FRAMERATE}),
                (self.generate_facing_bool, (), {}),
                (self.generate_distance_to_closest_station, (), {}),
                (self.generate_distance_from_start_and_end_stations,  (), {}),
                (self.generate_facing_stations, (), {}),
                (self.generate_intend_to_cross, (), {}),
                (self.generate_possible_interaction, (), {}),

                (self.select_columns, ("AGV_distance_X", "AGV_distance_Y", "AGV_speed_X", "AGV_speed_Y", "AGV_speed",
                                  "User_speed_X", "User_speed_Y", "User_speed",
                                  "User_velocity_X", "User_velocity_Y",
                                  "Wait_time",
                                  "intent_to_cross", "Gazing_station", "possible_interaction",
                                  "facing_along_sidewalk", "facing_to_road",
                                  'On_sidewalks', 'On_road',
                                  'closest_station', "distance_to_closest_station",
                                  'distance_to_closest_station_X', 'distance_to_closest_station_Y',
                                  'looking_at_AGV',
                                  'start_station_X', 'start_station_Y',
                                  'end_station_X', 'end_station_Y',
                                  'distance_from_start_station_X', 'distance_from_start_station_Y',
                                  'distance_from_end_station_X', 'distance_from_end_station_Y',
                                  'facing_start_station', 'facing_end_station',
                                  # Keep raw features
                                  "GazeDirection_X", "GazeDirection_Y", "GazeDirection_Z",
                                  "AGV_X", "AGV_Y", "User_X", "User_Y",
                                  "AGV_name", "TimestampID", "Timestamp",
                                  'looking_at_closest_station',
                                  ), {}),
            ])

            # # add the eye ralted features
            # out_df["Gaze ratio"] = generate_gaze_ratio(df)
            if to == 'pkl':
                out_filename = os.path.join(self.out_directory, file)
                out_df.to_pickle(out_filename)
            elif to == 'csv':
                out_filename = os.path.join(self.out_directory, file.strip(".pkl") + ".csv")
                out_df.to_csv(out_filename, index=False)
            else:
                raise ValueError("Invalid value for 'to' argument. Should be either 'pkl' or 'csv'")
            
            break # for testing purpose

    @staticmethod
    def process_data_gm(data, pipeline_functions):
        """Process the data for a guided model."""
        for function, arguments, keyword_arguments in pipeline_functions:
            if keyword_arguments and (not arguments):
                data = data.pipe(function, **keyword_arguments)
            elif (not keyword_arguments) and (arguments):
                data = data.pipe(function, *arguments)
            else:
                data = data.pipe(function)
        return data

    @staticmethod
    def select_columns(data, *columns):
        """Select only columns passed as argumentsdef."""
        return data.loc[:, columns]

    # generate the distance between AVG and User
    # Unit: m
    @staticmethod
    def generate_AGV_User_distance(df):
        df["AGV_distance_X"] = (np.abs(df['User_X'] - df['AGV_X']) / 100).tolist()
        df["AGV_distance_Y"] = (np.abs(df['User_Y'] - df['AGV_Y']) / 100).tolist()
        df['AGV_distance'] = np.sqrt(df['AGV_distance_X']**2 + df['AGV_distance_Y']**2)
        return df
    
    # generate the speed of AVG
    # Unit: m/
    @staticmethod
    def generate_AGV_speed(df, frame_rate = None):
        def generate_AGV_speed_helper(df, frame_rate = None):
            df["AGV_speed_X"] = abs(
                df[['AGV_X']] - df[['AGV_X']].shift(1)).values / 100 * frame_rate
            df["AGV_speed_Y"] = abs(
                df[['AGV_Y']] - df[['AGV_Y']].shift(1)).values / 100 * frame_rate
            df["AGV_speed"] = np.sqrt(df["AGV_speed_X"] ** 2 + df["AGV_speed_Y"] ** 2)
            return df

        df = df.groupby("AGV_name").apply(
            generate_AGV_speed_helper, frame_rate = frame_rate).reset_index(drop=True)
        return df
    
    # generate the speed of User
    # Unit: m/s
    @staticmethod
    def generate_user_speed(df, frame_rate = None):
        def generate_user_speed_helper(df, frame_rate = None):
            df["User_speed_X"] = abs(
                df[['User_X']] - df[['User_X']].shift(1)).values / 100. * frame_rate
            df["User_speed_Y"] = abs(
                df[['User_Y']] - df[['User_Y']].shift(1)).values / 100. * frame_rate
            df["User_velocity_X"] = (df[['User_X']] - df[['User_X']].shift(1)) / 100. * frame_rate
            df["User_velocity_Y"] = (df[['User_Y']] - df[['User_Y']].shift(1)) / 100. * frame_rate
            df["User_speed"] = np.sqrt(
                df["User_speed_X"] ** 2 + df["User_speed_Y"] ** 2)
            return df
        
        df = df.groupby('AGV_name').apply(
            generate_user_speed_helper, frame_rate=frame_rate).reset_index(drop=True)
        return df
    

    @staticmethod
    def generate_wait_time(df, H1=0.2, H2=0.1, THRESHOLE_ANGLE=30, frame_rate=None):
        """Generate the wait time feature."""
        from utils import get_direction_normalized, get_angle_between_normalized_vectors
        from constant import ERROR_RANGE, stations, User_trajectory

        # df['User_speed'] = np.sqrt(df['User_speed_X']**2 + df['User_speed_Y']**2)
        df['Wait_State'] = (df.shift(1) + df)['User_speed'] < H1
        df['Wait_time'] = 0

        # add "On side walk" features
        # User in +- error_range of would be accepted (Unit: cm)
        error_range = ERROR_RANGE
        df['On_sidewalks'] = df['User_Y'].apply(lambda x: True if
                                                (x > 8150 - error_range and x <
                                                 8400 + error_range)
                                                or (x > 6045 - error_range and x < 6295 + error_range)
                                                else False)

        # df['On sidewalks'] = True
        df['On_road'] = df['User_Y'].apply(lambda x: True if
                                           (x < 8150 - error_range /
                                            2) and (x > 6295 + error_range/2)
                                           else False)

        # add "Eye contact" features
        angle_in_radians = np.radians(THRESHOLE_ANGLE)
        threshold_COSINE = np.cos(angle_in_radians)
        df['Target_station_position'] = df['AGV_name'].apply(
            lambda x: stations[User_trajectory[int(x[3:])][1]])
        df['User_TargetStation_direction'] = df.apply(lambda x: get_direction_normalized(
            (x['User_X'], x['User_Y']), x['Target_station_position']), axis=1)
        df['User_AGV_direction'] = df.apply(lambda x: get_direction_normalized(
            (x['User_X'], x['User_Y']), (x['AGV_X'], x['AGV_Y'])), axis=1)

        df['User_TargetStation_angle'] = df.apply(lambda x: get_angle_between_normalized_vectors(
            (x['GazeDirection_X'], x['GazeDirection_Y']), x['User_AGV_direction']), axis=1)
        df['User_AGV_angle'] = df.apply(lambda x: get_angle_between_normalized_vectors(
            (x['GazeDirection_X'], x['GazeDirection_Y']), x['User_TargetStation_direction']), axis=1)

        df['looking_at_AGV'] = df.apply(lambda x: True if
                                        x['User_TargetStation_angle'] > threshold_COSINE or x['User_AGV_angle'] > threshold_COSINE else False, axis=1)

        begin_wait_Timestamp = None
        begin_wait_Flag = False
        AGV_passed_Flag = False
        cur_AGV = "AGV1"

        for index, row in df.iterrows():
            if row['AGV_name'] != cur_AGV:  # AGV changed
                cur_AGV = row['AGV_name']
                begin_wait_Timestamp = None
                begin_wait_Flag = False
                AGV_passed_Flag = False
                continue

            if AGV_passed_Flag == True:  # AGV is passed
                continue

            if begin_wait_Flag == False:  # in walking state
                # begin of waiting state
                if row['Wait_State'] and row['On_sidewalks'] and ~row['looking_at_AGV']:
                    begin_wait_Flag = True
                    begin_wait_Timestamp = index - 1 if index > 1 else 1
                    df.loc[index, 'Wait_time'] = (index - begin_wait_Timestamp) / frame_rate
                else:
                    continue
            else:  # in waiting state
                if df.loc[index, 'User_speed'] <= H2:  # still in waiting state
                    df.loc[index, 'Wait_time'] = (index - begin_wait_Timestamp) / frame_rate
                else:  # end of waiting state
                    begin_wait_Flag = False
                    begin_wait_Timestamp = None
                    df.loc[index, 'Wait_time'] = 0
                    AGV_passed_Flag = True  # AGV is passed

        df["Wait_time"] = df['Wait_time'].tolist()
        # print("Count of TimeStamp in Wait State / Walk State:",
        #       df[df['Wait_time'] > 0].shape[0], "/", df[df['Wait_time'] == 0].shape[0])
        return df
    

    @staticmethod
    def generate_facing_bool(df):
        THRESHOLD_ANGLE = 45
        THRESHOLD_COS = np.cos(np.radians(THRESHOLD_ANGLE))

        magnitude = (df['GazeDirection_X']**2 + df['GazeDirection_Y']**2)**0.5
        df['GazeDirection_projected_X'] = (df['GazeDirection_X'] / magnitude)
        df['GazeDirection_projected_Y'] = (df['GazeDirection_Y'] / magnitude)

        df['facing_along_sidewalk'] = (
            df['GazeDirection_projected_X'] > THRESHOLD_COS)

        def facing_road_helper(row):
            if row["User_velocity_Y"] < 0 and row['User_Y'] > 6295:
                # If moving down, then should be looking down
                return -row['GazeDirection_projected_Y'] > THRESHOLD_COS
            elif row["User_velocity_Y"] < -WALK_STAY_THRESHOLD and row['User_Y'] < 6295:
                return False

            if row["User_velocity_Y"] > 0 and row['User_Y'] < 8150:
                # If moving up, should be looking up
                return row['GazeDirection_projected_Y'] > THRESHOLD_COS
            elif row["User_velocity_Y"] > WALK_STAY_THRESHOLD and row['User_Y'] > 8150:
                return False

            # We can assume that they are looking at the road if they are stationary
            return True

        df['facing_to_road'] = df.apply(
            lambda row: facing_road_helper(row), axis=1)

        return df
    

    @staticmethod
    def generate_distance_to_closest_station(df):
        def generate_distance_to_closest_station_helper(row):
            mindis = 1000000
            closest_station = -1
            for station, position in stations.items():
                dis = np.sqrt((row['User_X'] - position[0]) **
                              2 + (row['User_Y'] - position[1])**2)
                if dis < mindis:
                    mindis = dis
                    closest_station = station
                    mindis_X = abs(row['User_X'] - position[0])
                    mindis_Y = abs(row['User_Y'] - position[1])
            return closest_station, mindis, mindis_X, mindis_Y

        df['distance_to_closest_station'] = df.apply(
            lambda row: generate_distance_to_closest_station_helper(row), axis=1)

        df['closest_station'] = df['distance_to_closest_station'].apply(
            lambda x: x[0])
        df['distance_to_closest_station_X'] = df['distance_to_closest_station'].apply(
            lambda x: x[2])
        df['distance_to_closest_station_Y'] = df['distance_to_closest_station'].apply(
            lambda x: x[3])
        df['distance_to_closest_station'] = df['distance_to_closest_station'].apply(
            lambda x: x[1])
        return df



# # Distance from start and end stations
    @staticmethod
    def generate_distance_from_start_and_end_stations(df):
        def generate_station_coordinates(row):
            agv_number = int(row["AGV_name"][3:])
            start_station, end_station = User_trajectory[agv_number]
            start_station_X, start_station_Y = stations[start_station]
            end_station_X, end_station_Y = stations[end_station]
            return start_station_X, start_station_Y, end_station_X, end_station_Y

        def generate_distance_from_stations_helper(row):
            distance_start_X = abs(row['User_X'] - row['start_station_X'])
            distance_start_Y = abs(row['User_Y'] - row['start_station_Y'])
            distance_end_X = abs(row['User_X'] - row['end_station_X'])
            distance_end_Y = abs(row['User_Y'] - row['end_station_Y'])

            return distance_start_X, distance_start_Y, distance_end_X, distance_end_Y

        df['station_coords'] = df.apply(
            lambda row: generate_station_coordinates(row), axis=1)
        df['start_station_X'] = df['station_coords'].apply(lambda x: x[0])
        df['start_station_Y'] = df['station_coords'].apply(lambda x: x[1])
        df['end_station_X'] = df['station_coords'].apply(lambda x: x[2])
        df['end_station_Y'] = df['station_coords'].apply(lambda x: x[3])
        df.drop(columns=['station_coords'], inplace=True)

        df['distance_from_stations'] = df.apply(
            lambda row: generate_distance_from_stations_helper(row), axis=1)
        df['distance_from_start_station_X'] = df['distance_from_stations'].apply(
            lambda x: x[0])
        df['distance_from_start_station_Y'] = df['distance_from_stations'].apply(
            lambda x: x[1])
        df['distance_from_end_station_X'] = df['distance_from_stations'].apply(
            lambda x: x[2])
        df['distance_from_end_station_Y'] = df['distance_from_stations'].apply(
            lambda x: x[3])
        df.drop(columns=['distance_from_stations'], inplace=True)
        return df




    @staticmethod
    def generate_facing_stations(df):
        def dot(vec1, vec2):
            return vec1[0] * vec2[0] + vec1[1] * vec2[1]

        def angle(vec1, vec2):
            return math.acos(dot(vec1, vec2))

        def generate_facing_stations_helper(row):
            agv_number = int(row['AGV_name'][3:])
            start_station, end_station = User_trajectory[agv_number]
            _, start_station_Y = stations[start_station]
            _, end_station_Y = stations[end_station]
            user_y = row['User_Y']
            head_pose = math.radians(row['User_Yaw'])
            head_pose_vector = (-math.cos(head_pose), -math.sin(head_pose))
            start_station_to_user_vector = (0, np.sign(user_y - start_station_Y))
            angle_between_user_and_start = angle(
                head_pose_vector, start_station_to_user_vector)
            facing_start_station = False
            if angle_between_user_and_start > ANGULAR_THRESHOLD_HIGH:
                facing_start_station = True
            end_station_to_user_vector = (0, np.sign(user_y - end_station_Y))
            angle_between_user_and_end = angle(
                head_pose_vector, end_station_to_user_vector)
            facing_end_station = False
            if angle_between_user_and_end > ANGULAR_THRESHOLD_HIGH:
                facing_end_station = True

            return facing_start_station, facing_end_station

        df['facing_stations'] = df.apply(
            lambda row: generate_facing_stations_helper(row), axis=1)
        df['facing_start_station'] = df['facing_stations'].apply(lambda x: x[0])
        df['facing_end_station'] = df['facing_stations'].apply(lambda x: x[1])
        df.drop(columns=['facing_stations'], inplace=True)
        return df
    


    # ## Intend to cross feature
    # based on the following:
    # - θ<Gaze , AGV>  or θ<Gaze , ∃ station> is less than 30°
    # - Speed decrease (Under test)
    # 
    @staticmethod
    def generate_intend_to_cross(df):
        THRESHOLE_ANGLE = 30
        angle_in_radians = np.radians(THRESHOLE_ANGLE)
        threshold_COSINE = np.cos(angle_in_radians)

        df['most_close_station_direction_cos'] = df.apply(
            lambda row: get_most_close_station_direction(row), axis=1)

        df['looking_at_closest_station'] = df['most_close_station_direction_cos'].apply(
            lambda x: x[0] > threshold_COSINE)

        df["Gazing_station"] = df['most_close_station_direction_cos'].apply(
            lambda x: x[1])  # if x[0] > threshold_COSINE else np.nan)

        df['User-AGV_direction_cos'] = df.apply(
            lambda row: get_user_agv_direction_cos(row), axis=1)

        df['acceleration'] = (df[['User_speed_X', 'User_speed_Y']] - df[['User_speed_X', 'User_speed_Y']].shift(1))\
            .apply(lambda row: (row['User_speed_X']**2 + row['User_speed_Y']**2)**0.5, axis=1)

        def intent_to_cross_helper(row):
            THRESHOLD_ANGLE = 30
            THRESHOLD_COS = np.cos(np.radians(THRESHOLD_ANGLE))
            facing_to_road = True

            if row["User_velocity_Y"] < 0 and row['User_Y'] > 6295:
                # If moving down, then should be looking down
                facing_to_road = -row['GazeDirection_projected_Y'] > THRESHOLD_COS
            elif row["User_velocity_Y"] < -WALK_STAY_THRESHOLD and row['User_Y'] < 6295:
                facing_to_road = False

            if row["User_velocity_Y"] > 0 and row['User_Y'] < 8150:
                # If moving up, should be looking up
                facing_to_road = row['GazeDirection_projected_Y'] > THRESHOLD_COS
            elif row["User_velocity_Y"] > WALK_STAY_THRESHOLD and row['User_Y'] > 8150:
                facing_to_road = False

            if ((row['most_close_station_direction_cos'][0] > threshold_COSINE and abs(row['User_Y'] - stations[row['Gazing_station']][1]) > 300) or
                    row['User-AGV_direction_cos'] > threshold_COSINE) and (facing_to_road):
                return True
            else:
                return False

        df['intent_to_cross'] = df.apply(
            lambda row: intent_to_cross_helper(row), axis=1)

        return df
    
    
    @staticmethod
    def generate_possible_interaction(df):
        def generate_possible_interation_helper(df):
            THRESHOLD_PERIOD = 5
            THRESHOLD_DISTANCE = 10
            df['possible_interaction'] = df[['AGV_distance']]\
                .rolling(window=2*THRESHOLD_PERIOD, closed='right').min()\
                .shift(-THRESHOLD_PERIOD) < THRESHOLD_DISTANCE
            return df

        df = df.groupby(by='AGV_name', group_keys=False).apply(
            generate_possible_interation_helper)
        df['possible_interaction'].fillna(False, inplace=True)
        return df
    

    
    @staticmethod
    def data_aug_helper(df, lidar_range=60, camera_range=20):
        # Simulate Lidar, dismiss the data when the AGV is too far away from the user
        df['AGV_Worker_distance'] = (
            (df['User_X'] - df['AGV_X']) ** 2 + (df['User_Y'] - df['AGV_Y']) ** 2) ** 0.5 / 100
        df = df[df['AGV_Worker_distance'] <= lidar_range]
        df = df[df.apply(lambda x: does_line_intersect_rectangles(
            (x['User_X'], x['User_Y']), (x['AGV_X'], x['AGV_Y'])) == False, axis=1)]

        df[df['AGV_Worker_distance'] <= camera_range][['EyeTarget', 'GazeDirection_X',
                                                       'GazeDirection_Y', 'GazeDirection_Z',]] = ("", 0, 0, 0)
        df[df['AGV_Worker_distance'] <= camera_range][['GazeOrigin_X', 'GazeOrigin_Y',
           'GazeOrigin_Z']] = df[df['AGV_Worker_distance'] <= camera_range][['User_X', 'User_Y', 'User_Z']]

        return df
    
    @staticmethod
    def re_sample(df:pd.DataFrame = None, df_path:str = None, target_frame_rate = None):
        """
        Resample the data to the given frame rate
        Args:
            df: data frame to resample
            df_path: path to the data frame
            target_frame_rate: target frame rate
        """

        if df_path:
            if df_path.endswith('.pkl'):
                df = pd.read_pickle(df_path)
            elif df_path.endswith('.csv'):
                df = pd.read_csv(df_path)

        origin_frame_rate = df.groupby('TimestampID').size().mean()

        if origin_frame_rate == target_frame_rate:
            return df
        else:
            if target_frame_rate > origin_frame_rate:
                raise ValueError("Target frame rate should be smaller than the origin frame rate")
            else:
                print(f"Resampling data from {origin_frame_rate} to {target_frame_rate}")
            ratio = int(origin_frame_rate/target_frame_rate)
            df = df.apply(lambda x: x.iloc[::ratio])
            return df
        
    @staticmethod
    def re_sample_dir(dir, target_frame_rate, target_dir = None, to:Literal['csv', 'pkl'] = 'csv'):
        """
        Resample all the files in the directory to the target frame rate
        Args:
            dir: directory of the data
            target_frame_rate: target frame rate
            target_dir: directory to save the resampled data
            to: format to save the resampled data, either 'csv' or 'pkl'
        """
        if target_dir is None:
            target_dir = dir + "_resampled"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print(f"Created directory {target_dir}")
        files = os.listdir(dir)
        for file in files:
            if file.endswith('csv'):
                df = pd.read_csv(os.path.join(dir, file))
            elif file.endswith('pkl'):
                df = pd.read_pickle(os.path.join(dir, file))
            else:
                print(f"Unrecognized file format: {file}")
                continue
            df = FeatureGenerator.re_sample(df=df, target_frame_rate=target_frame_rate)
            if to == 'csv':
                df.to_csv(os.path.join(target_dir, file[:-3] + 'csv'), index=False)
            elif to == 'pkl':
                df.to_pickle(os.path.join(target_dir, file[:-3] + 'pkl'))
            df = None

        return target_dir



if __name__ == "__main__":
    fg = FeatureGenerator(GT_only = False, save_data_path = 'data/test/')
    fg.generate_features(data_aug=False, lidar_range=20, camera_range=15, FRAMERATE = 70, to = 'csv')
    fg.re_sample_dir('data/test/', 10, 'data/test_resampled/')



        

    


    
        


        
        

        