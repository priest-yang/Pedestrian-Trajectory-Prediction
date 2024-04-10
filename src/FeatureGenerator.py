from DataAug import *
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from constant import *
import math
import os
import glob

import warnings
warnings.filterwarnings('ignore')


# df = pd.read_csv("../data/PandasData/Original/PID001_NSL.csv")
# df.columns


# generate the distance between AVG and User
# Unit: m
def generate_AGV_User_distance(df):
    df["AGV distance X"] = (np.abs(df['User_X'] - df['AGV_X']) / 100).tolist()
    df["AGV distance Y"] = (np.abs(df['User_Y'] - df['AGV_Y']) / 100).tolist()
    return df


# generate the speed of AVG
# Unit: m/s

def generate_AGV_speed_helper(df):
    df["AGV speed X"] = abs(
        df[['AGV_X']] - df[['AGV_X']].shift(1)).values / 100
    df["AGV speed Y"] = abs(
        df[['AGV_Y']] - df[['AGV_Y']].shift(1)).values / 100
    df["AGV speed"] = np.sqrt(df["AGV speed X"] ** 2 + df["AGV speed Y"] ** 2)
    df["AGV speed X"].fillna(0, inplace=True)
    df["AGV speed Y"].fillna(0, inplace=True)
    df["AGV speed"].fillna(0, inplace=True)
    return df


def generate_AGV_speed(df):
    df = df.groupby("AGV_name").apply(
        generate_AGV_speed_helper).reset_index(drop=True)
    return df


# generate the speed of User
# Unit: m/s

def generate_user_speed_helper(df):
    df["User speed X"] = abs(
        df[['User_X']] - df[['User_X']].shift(1)).values / 100.
    df["User speed Y"] = abs(
        df[['User_Y']] - df[['User_Y']].shift(1)).values / 100.
    df["User velocity X"] = (df[['User_X']] - df[['User_X']].shift(1)) / 100.
    df["User velocity Y"] = (df[['User_Y']] - df[['User_Y']].shift(1)) / 100.
    df["User speed"] = np.sqrt(
        df["User speed X"] ** 2 + df["User speed Y"] ** 2)
    df["User speed X"].fillna(0, inplace=True)
    df["User speed Y"].fillna(0, inplace=True)
    df["User speed"].fillna(0, inplace=True)
    return df


def generate_user_speed(df):
    df = df.groupby('AGV_name').apply(
        generate_user_speed_helper).reset_index(drop=True)
    return df


User_trajectory = {  # (AGV_name, User_start_station, User_end_station)
                     1: (1, 2),
                     2: (2, 4),
                     3: (4, 3),
                     4: (3, 6),
                     5: (6, 5),
                     6: (5, 6),
                     7: (6, 8),
                     8: (8, 7),
                     9: (7, 8),
    10: (8, 6),
    11: (6, 5),
    12: (5, 6),
    13: (6, 3),
    14: (3, 4),
    15: (4, 2),
    16: (2, 1),
}

# Dictionary with keys = station number and values = their X,Y coordinates.
stations = {1: (1580, 8683),
            2: (1605, 5800),
            3: (5812, 8683),
            4: (5800, 5786),
            5: (7632, 8683),
            6: (7639, 5786),
            7: (13252, 8683),
            8: (13319, 5796)
            }


def get_direction_normalized(start: tuple, end: tuple) -> tuple:
    """
    Returns the normalized direction vector from start to end.
    """
    x = end[0] - start[0]
    y = end[1] - start[1]
    length = np.sqrt(x**2 + y**2)
    return (x/length, y/length)


def get_angle_between_normalized_vectors(v1: tuple, v2: tuple) -> float:
    """
    Returns the angle between two vectors in radians.
    """
    return np.arccos(np.dot(v1, v2))


def generate_wait_time(df, H1=0.2, H2=0.1, THRESHOLE_ANGLE=30):
    df['User_speed'] = np.sqrt(df['User speed X']**2 + df['User speed Y']**2)
    df['Wait State'] = (df.shift(1) + df)['User_speed'] < H1
    df['Wait time'] = 0

    # add "On side walk" features
    # User in +- error_range of would be accepted (Unit: cm)
    error_range = ERROR_RANGE
    df['On sidewalks'] = df['User_Y'].apply(lambda x: True if
                                            (x > 8150 - error_range and x <
                                             8400 + error_range)
                                            or (x > 6045 - error_range and x < 6295 + error_range)
                                            else False)

    # df['On sidewalks'] = True
    df['On road'] = df['User_Y'].apply(lambda x: True if
                                       (x < 8150 - error_range /
                                        2) and (x > 6295 + error_range/2)
                                       else False)
#     df['On road'] = ~df['On sidewalks']

    # add "Eye contact" features
    angle_in_radians = np.radians(THRESHOLE_ANGLE)
    threshold_COSINE = np.cos(angle_in_radians)
    df['Target station position'] = df['AGV_name'].apply(
        lambda x: stations[User_trajectory[int(x[3:])][1]])
    df['User-TargetStation direction'] = df.apply(lambda x: get_direction_normalized(
        (x['User_X'], x['User_Y']), x['Target station position']), axis=1)
    df['User-AGV direction'] = df.apply(lambda x: get_direction_normalized(
        (x['User_X'], x['User_Y']), (x['AGV_X'], x['AGV_Y'])), axis=1)

    df['User-TargetStation angle'] = df.apply(lambda x: get_angle_between_normalized_vectors(
        (x['GazeDirection_X'], x['GazeDirection_Y']), x['User-AGV direction']), axis=1)
    df['User-AGV angle'] = df.apply(lambda x: get_angle_between_normalized_vectors(
        (x['GazeDirection_X'], x['GazeDirection_Y']), x['User-TargetStation direction']), axis=1)

    df['looking_at_AGV'] = df.apply(lambda x: True if
                                    x['User-TargetStation angle'] > threshold_COSINE or x['User-AGV angle'] > threshold_COSINE else False, axis=1)

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
            if row['Wait State'] and row['On sidewalks'] and ~row['looking_at_AGV']:
                begin_wait_Flag = True
                begin_wait_Timestamp = index - 1 if index > 1 else 1
                df.loc[index, 'Wait time'] = index - begin_wait_Timestamp
            else:
                continue
        else:  # in waiting state
            if df.loc[index, 'User_speed'] <= H2:  # still in waiting state
                df.loc[index, 'Wait time'] = index - begin_wait_Timestamp
            else:  # end of waiting state
                begin_wait_Flag = False
                begin_wait_Timestamp = None
                df.loc[index, 'Wait time'] = 0
                AGV_passed_Flag = True  # AGV is passed

    df["Wait time"] = df['Wait time'].tolist()
    # print("Count of TimeStamp in Wait State / Walk State:",
    #       df[df['Wait time'] > 0].shape[0], "/", df[df['Wait time'] == 0].shape[0])
    return df


# ## Intend to cross feature
# based on the following:
# - θ<Gaze , AGV>  or θ<Gaze , ∃ station> is less than 30°
# - Speed decrease (Under test)
# 


def generate_intend_to_cross(df):
    THRESHOLE_ANGLE = 30
    angle_in_radians = np.radians(THRESHOLE_ANGLE)
    threshold_COSINE = np.cos(angle_in_radians)

    # Dictionary with keys = station number and values = their X,Y coordinates.
    stations = {1: (1580, 8683),
                2: (1605, 5800),
                3: (5812, 8683),
                4: (5800, 5786),
                5: (7632, 8683),
                6: (7639, 5786),
                7: (13252, 8683),
                8: (13319, 5796)
                }

    def get_direction_normalized(start: tuple, end: tuple) -> tuple:
        """
        Returns the normalized direction vector from start to end.
        """
        x = end[0] - start[0]
        y = end[1] - start[1]
        length = np.sqrt(x**2 + y**2)
        return (x/length, y/length)

    def get_most_close_station_direction(row):
        """
        Returns (maximum cosine value, corresponding station number)
        """
        max_cos = -1
        most_common_station = np.nan
        for station, position in stations.items():
            direction_normalized = get_direction_normalized(
                (row['User_X'], row['User_Y']), position)
            cosine_gaze_direction = row['GazeDirection_X'] * \
                direction_normalized[0] + \
                row['GazeDirection_Y'] * direction_normalized[1]
            if cosine_gaze_direction > max_cos:
                max_cos = cosine_gaze_direction
                most_common_station = station
        return max_cos, most_common_station

    def get_user_agv_direction_cos(row):
        """
        Returns the cos between direction vector from the user to the AGV.
        """
        direction_normalized = get_direction_normalized(
            (row['User_X'], row['User_Y']), (row['AGV_X'], row['AGV_Y']))
        cosine_gaze_direction = row['GazeDirection_X'] * \
            direction_normalized[0] + \
            row['GazeDirection_Y'] * direction_normalized[1]
        return cosine_gaze_direction

    df['most_close_station_direction_cos'] = df.apply(
        lambda row: get_most_close_station_direction(row), axis=1)

    df['looking_at_closest_station'] = df['most_close_station_direction_cos'].apply(
        lambda x: x[0] > threshold_COSINE)

    df["Gazing_station"] = df['most_close_station_direction_cos'].apply(
        lambda x: x[1])  # if x[0] > threshold_COSINE else np.nan)

    df['User-AGV_direction_cos'] = df.apply(
        lambda row: get_user_agv_direction_cos(row), axis=1)

    df['acceleration'] = (df[['User speed X', 'User speed Y']] - df[['User speed X', 'User speed Y']].shift(1))\
        .apply(lambda row: (row['User speed X']**2 + row['User speed Y']**2)**0.5, axis=1)

    def intent_to_cross_helper(row):
        THRESHOLD_ANGLE = 30
        THRESHOLD_COS = np.cos(np.radians(THRESHOLD_ANGLE))
        facing_to_road = True

        if row["User velocity Y"] < 0 and row['User_Y'] > 6295:
            # If moving down, then should be looking down
            facing_to_road = -row['GazeDirection__projected_Y'] > THRESHOLD_COS
        elif row["User velocity Y"] < -WALK_STAY_THRESHOLD and row['User_Y'] < 6295:
            facing_to_road = False

        if row["User velocity Y"] > 0 and row['User_Y'] < 8150:
            # If moving up, should be looking up
            facing_to_road = row['GazeDirection__projected_Y'] > THRESHOLD_COS
        elif row["User velocity Y"] > WALK_STAY_THRESHOLD and row['User_Y'] > 8150:
            facing_to_road = False

        if ((row['most_close_station_direction_cos'][0] > threshold_COSINE and abs(row['User_Y'] - stations[row['Gazing_station']][1]) > 300) or
                row['User-AGV_direction_cos'] > threshold_COSINE) and (facing_to_road):
            return True
        else:
            return False

    df['intent_to_cross'] = df.apply(
        lambda row: intent_to_cross_helper(row), axis=1)

    return df


# ## Distance to the most close Station


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


# # Facing the start or end station


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


# ## Possible Interaction 
# In the following 10s, assume User doesn't move & AGV doesn't slow down, return
# - True if min(AGV, User) < 5
# - False otherwise


def generate_possible_interaction(df):
    def generate_possible_interation_helper(df):
        THRESHOLD_PERIOD = 5
        THRESHOLD_DISTANCE = 10
        df['AGV distance'] = (df['AGV distance X']**2 +
                              df['AGV distance Y']**2) ** 0.5
        df['possible_interaction'] = df[['AGV distance']]\
            .rolling(window=2*THRESHOLD_PERIOD, closed='right').min()\
            .shift(-THRESHOLD_PERIOD) < THRESHOLD_DISTANCE
        return df

    df = df.groupby(by='AGV_name', group_keys=False).apply(
        generate_possible_interation_helper)
    df['possible_interaction'].fillna(False, inplace=True)
    return df


# ## Facing along sidewalk / to road 


# error_range = 50  #User in +- error_range of would be accepted (Unit: cm)
# df['On sidewalks'] = df['User_Y'].apply(lambda x: True if \
#                                        (x > 8150 - error_range and x < 8400 + error_range) \
#                                        or (x > 6045 - error_range and x < 6259 + error_range)\
#                                         else False)

# # df['On sidewalks'] = True
# df['On road'] = df['User_Y'].apply(lambda x: True if \
#                                        (x < 8150 - error_range/2) and (x > 6259 + error_range/2) \
#                                             else False)


def generate_facing_bool(df):
    THRESHOLD_ANGLE = 45
    THRESHOLD_COS = np.cos(np.radians(THRESHOLD_ANGLE))

    magnitude = (df['GazeDirection_X']**2 + df['GazeDirection_Y']**2)**0.5
    df['GazeDirection_projected_X'] = (df['GazeDirection_X'] / magnitude)
    df['GazeDirection__projected_Y'] = (df['GazeDirection_Y'] / magnitude)

    # df['facing_along_sidewalk'] = df['GazeDirection_projected_X'] > THRESHOLD_COS & \
    #     (df["User speed X"] * df['GazeDirection_projected_X'] > 0)

    # & \  # Facing majorly the sidewalk direction
    df['facing_along_sidewalk'] = (
        df['GazeDirection_projected_X'] > THRESHOLD_COS)
#         ((df["User_Y"] < 8150 - ERROR_RANGE) or (df["User_Y"] > 6295 + ERROR_RANGE)      # Within the sidewalk. I think this should be checked separately
#     ((df["User speed X"] * df['GazeDirection_projected_X']) > 0)                       # This would not work, since user speed is always positive


#     df['facing_to_road'] = (abs(df['GazeDirection__projected_Y']) > THRESHOLD_COS) and \
#     (
#         (df['On road'] and (df["User speed Y"] * df['GazeDirection__projected_Y'] > 0)) or
#         ((df['User_Y'] > 8150) and (-df['GazeDirection__projected_Y'] > THRESHOLD_COS)) or
#         ((df['User_Y'] < 6295) and (df['GazeDirection__projected_Y'] > THRESHOLD_COS))
#     )

    # THE ABOVE CONDITIONS LOOK TOO COMPLICATED. I THINK WE ONLY NEED TO CHECK IF THE ANGLE IS MORE THAN THE THRESHOLD
    # THE ON ROAD CONDITIONS SHOULD BE CHECKED SEPARTELY. HERE, WE ARE ONLY CONCERNED WITH WHETHER THE PARTICIPANT IS LOOKING AT THE ROAD


    def facing_road_helper(row):
        if row["User velocity Y"] < 0 and row['User_Y'] > 6295:
            # If moving down, then should be looking down
            return -row['GazeDirection__projected_Y'] > THRESHOLD_COS
        elif row["User velocity Y"] < -WALK_STAY_THRESHOLD and row['User_Y'] < 6295:
            return False

        if row["User velocity Y"] > 0 and row['User_Y'] < 8150:
            # If moving up, should be looking up
            return row['GazeDirection__projected_Y'] > THRESHOLD_COS
        elif row["User velocity Y"] > WALK_STAY_THRESHOLD and row['User_Y'] > 8150:
            return False

        # We can assume that they are looking at the road if they are stationary
        return True

#     df['facing_to_road'] = abs(df['GazeDirection__projected_Y']) > THRESHOLD_COS
    df['facing_to_road'] = df.apply(
        lambda row: facing_road_helper(row), axis=1)

    return df


# ## Eye Gaze Ratio


# generate the feature of eye gaze ratio
def count(x):
    _count = 0
    _total = 0
    for item in x:
        _total += 1
        if 'AGV_Sphere' in item:
            _count += 1

    return _count / _total


def generate_gaze_ratio(df) -> np.ndarray:
    '''
    input: raw data without aggregation
    output: dataframe contains a column named "Gaze ratio" 
    Computation method:
    For each second, gaze ratio = count(eye_target.contains("AGV")) / count(Frames)
    '''

    eye_data_keys = ['TimestampID', 'EyeTarget']
    eye_data = df[eye_data_keys]
    out_df = eye_data.groupby('TimestampID').agg(count)
    return out_df['EyeTarget'].values


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


def select_columns(data, *columns):
    """Select only columns passed as arguments."""
    return data.loc[:, columns]


def data_aug(df, lidar_range=60, camera_range=20):
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



# re-format to pickle
# for file in files_new: # for file in files:
#     if file == "desktop.ini":
#         continue
#     df = pd.read_csv(os.path.join(data_directory, file))
#     df.to_pickle(os.path.join(data_directory, file.replace(".csv", ".pkl")))


class FeatureGenerator:
    def __init__(self, GT_path = None) -> None:
        self.current_directory = os.getcwd()
        self.data_directory = os.path.join(
            self.current_directory, "..", "data", "PandasData", "Original")

        self.out_directory = os.path.join(
            self.current_directory, "..", "data", "PandasData", "Modified")
        
        if not os.path.exists(self.out_directory):
            os.makedirs(self.out_directory)
        files = glob.glob(os.path.join(self.out_directory, "*.pkl"))
        for f in files:
            os.remove(f)

        self.files = os.listdir(self.data_directory)

        self.features = ["AV_distance", "AV_speed", "Wait_time",
                    "Gaze_ratio", "Curb_distance", "Ped_speed"]

        # columns in the data that are important to compute the features
        self.keys = ['TimestampID', 'Timestamp', 'AGV_name',
                'User_X', 'User_Y', 'AGV_X', 'AGV_Y', 'AGV_speed']
        self.eye_data_keys = ['TimestampID', 'EyeTarget']

        # GT_path = '../../Inter_test/data/behavior_inter_rater_Shawn.csv'
        if GT_path is None:
            GT_path = '../../Inter_test/data/behavior_inter_rater_Shawn.csv'
        
        self.GT = pd.read_csv(GT_path)
        self.GT = self.GT[['PID', 'AGV_name', 'Condition']].drop_duplicates()
        test_condition_df = self.GT[['PID', 'Condition']].drop_duplicates(
            subset=['PID', 'Condition'], keep='last')
        self.files_new = [f"PID{str(row['PID']).zfill(3)}_{row['Condition']}.pkl" for _,
                     row in test_condition_df.iterrows()]
        
    def generate_features(self, lidar_range=20, camera_range=15):

        for file in self.files_new:  # for file in files:
            if file == "desktop.ini":
                continue
            
            data_details = file.split("_")
            pid = int(data_details[0][-3:])
            scn = data_details[1].split('.')[0]
            # print(f'PID{pid:03}, Condition {scn}')

            df = pd.read_pickle(os.path.join(self.data_directory, file))

            AGV_num = self.GT[(self.GT['PID'] == pid) & (
                self.GT['Condition'] == scn)]['AGV_name'].unique()
            AGV_list = ["AGV"+str(i) for i in AGV_num]
            df = df[df['AGV_name'].isin(AGV_list)]
            out_df = pd.DataFrame()

            # Group the dataframe by TimestampID to get the per second data
            grouped = df.groupby("TimestampID")
            aggregated = grouped.mean(numeric_only=True).reset_index()

            # keep non-numerical raw features from original data
            df["Timestamp"] = pd.to_timedelta(df["Timestamp"])
            raw_features = ["Timestamp", "AGV_name", "TimestampID"]
            df_dropped = df.drop_duplicates(subset=['Timestamp'], keep='first')
            aggregated[raw_features] = df_dropped[raw_features].values

            # Process the data
            aggregated = data_aug(aggregated, lidar_range=lidar_range, camera_range=camera_range)
            if aggregated.shape[0] == 0:
                continue
            
            out_df = process_data_gm(aggregated, [
                # (data_aug, (), {'lidar_range': 30, 'camera_range': 5}),

                (generate_AGV_User_distance, (), {}),
                (generate_AGV_speed, (), {}),
                (generate_user_speed, (), {}),
                (generate_wait_time, (), {'H1': 0.2, 'H2': 0.1,
                 'THRESHOLE_ANGLE': GAZING_ANGLE_THRESHOLD}),
                (generate_facing_bool, (), {}),
                (generate_distance_to_closest_station, (), {}),

                # Features for the check functions
                (generate_distance_from_start_and_end_stations,  (), {}),
                (generate_facing_stations, (), {}),

                # new features:
                (generate_intend_to_cross, (), {}),
                (generate_possible_interaction, (), {}),

                (select_columns, ("AGV distance X", "AGV distance Y", "AGV speed X", "AGV speed Y", "AGV speed",
                                  "User speed X", "User speed Y", "User speed",
                                  "User velocity X", "User velocity Y",
                                  "Wait time",
                                  "intent_to_cross", "Gazing_station", "possible_interaction",
                                  "facing_along_sidewalk", "facing_to_road",
                                  'On sidewalks', 'On road',
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

            # out_df = data_aug(out_df)

            # save the dataframe
            out_filename = os.path.join(self.out_directory, file.strip(".pkl") + ".pkl")
            out_df.to_pickle(out_filename)

            # out_filename = os.path.join(out_directory, file.strip(".csv") + ".csv")
            # out_df.to_csv(out_filename)


# df = pd.read_pickle(os.path.join(
#     '../data', 'PandasData/Modified/PID003_NSL.pkl'))
# df.columns = df.columns.str.replace(' ', '_')


# df.head().columns





