import math
import pandas as pd

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

sidewalks = {1: (2625, 15000, 2625, 8150),
             2: (2425, 15000, 2425, 8400),
             3: (2625, 8150, 0, 8150),
             4: (2425, 8400, 0, 8400),
             5: (0, 6295, 16000, 6295),
             6: (0, 6045, 6400, 6045),
             7: (6400, 6045, 6400, 4000),
             8: (6650, 6045, 6650, 4000),
             9: (6650, 6045, 16000, 6045),
             10: (4275, 8150, 9000, 8150),
             11: (4525, 8400, 8750, 8400),
             12: (4525, 15000, 4525, 8400),
             13: (4275, 15000, 4275, 8150),
             14: (8750, 8400, 8750, 15000),
             15: (9000, 8150, 9000, 15000),
             16: (10500, 8150, 10500, 15000),
             17: (10750, 8400, 10750, 15000),
             18: (10750, 8400, 16000, 8400),
             19: (10500, 8150, 16000, 8150)}

agv_to_stations = {1: (1, 2),
                   2: (2, 4),
                   3: (4, 3),
                   4: (3, 6),
                   5: (6, 5),
                   6: (5, 6),
                   7: (6, 8),
                   8: (8, 7)}

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


for i in range(9, 17):
    agv_to_stations[i] = (agv_to_stations[17-i][1], agv_to_stations[17-i][0])


SIDEWALK_1 = {"High": 8400, "Low": 8150}
SIDEWALK_2 = {"High": 6295, "Low": 6045}

STATION_LENGTH = 5.                            # meters
WALK_STAY_THRESHOLD = 0.3
# 0.8m/s as the threshold between walking and waiting

AT_STATION_THRESHOLD = 2.5
CLOSE_TO_STATION_THRESHOLD = 3.0
CLOSE_TO_STATION_THRESHOLD_X = 3.0
CLOSE_TO_STATION_THRESHOLD_Y = 2.0
MARGIN_NEAR_SIDEWALKS = 1.0
# in radius 3m we consider the user is close to the station

GAZING_ANGLE_THRESHOLD = 40
GAZING_ANGLE_THRESHOLD_RADIUS = math.radians(GAZING_ANGLE_THRESHOLD)
GAZING_ANGLE_THRESHOLD_COS = math.cos(GAZING_ANGLE_THRESHOLD_RADIUS)

ERROR_RANGE = 50
state_num_to_name = {0: 'At station',
                     1: 'Approach sidewalk',
                     2: 'Wait',
                     3: 'Move along sidewalk',
                     4: 'Approach station',
                     5: 'Cross',
                     -1: 'Error'}

# Radius to determine whether a worker is at a station
RADIUS_1 = 2.5                                 # meters
# Radius to determine whether a worker is approaching either a station or the sidewalk
RADIUS_2 = 3.5                                 # meters
# Radius to determine if moving along the sidewalk
RADIUS_3 = 6.                                  # meters
# Below this speed, the worker is assumed to be stopped
SPEED_THRESHOLD_LOW = 0.2                      # m/s
# Above this speed, the worker is assumed to be actively moving
SPEED_THRESHOLD_HIGH = 0.8                     # m/s
# Angle between the heading vector and the vector from user to station to determine
# whether the user is looking at the station
ANGULAR_THRESHOLD_LOW = math.pi / 3            # radians
# Angle between the heading vector and the vector from user to station to determine
# whether the user is looking at the road
ANGULAR_THRESHOLD_HIGH = 2 * math.pi / 3       # radians
STATION_LENGTH = 5.                            # meters



H1 = 0.2, 
H2 = 0.1, 
FRAME_RATE = 20

DATA_DF = pd.DataFrame(columns=['PID', 'SCN', 'TimestampID', 'Timestamp', 'DatapointID', 'AGV_name',
       'User_X', 'User_Y', 'User_Z', 'User_Pitch', 'User_Yaw', 'User_Roll',
       'U_X', 'U_Y', 'U_Z', 'AGV_X', 'AGV_Y', 'AGV_Z', 'AGV_Pitch', 'AGV_Yaw',
       'AGV_Roll', 'AGV_speed', 'EyeTarget', 'GazeOrigin_X', 'GazeOrigin_Y',
       'GazeOrigin_Z', 'GazeDirection_X', 'GazeDirection_Y', 'GazeDirection_Z',
       'Confidence'])

DEFAULT_DF = pd.DataFrame(columns=["AGV distance X", "AGV distance Y", "AGV speed X", "AGV speed Y", "AGV speed",
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
                           "User_AVG direction", 
                           # Keep raw features
                           "GazeDirection_X", "GazeDirection_Y", "GazeDirection_Z",
                           "AGV_X", "AGV_Y", "User_X", "User_Y", 
                           "AGV_name", "TimestampID", "Timestamp"])


DEFAULT_FEATURE = {
    "AGV distance X": 0, "AGV distance Y": 0, "AGV speed X": 0, "AGV speed Y": 0, "AGV speed": 0,
    "User speed X": 0, "User speed Y": 0, "User speed": 0, "User velocity X": 0, "User velocity Y": 0,
    "Wait time": 0,
    "intent_to_cross": False, "Gazing_station": -1, "possible_interaction": False, 
    "facing_along_sidewalk": False, "facing_to_road": False,
    'On sidewalks': False, 'On road': False,
    'distance_to_closest_station': 10000000, 'closest_station': -1, 
    'distance_to_closest_station_X': 1000000, 'distance_to_closest_station_Y': 10000000,
    'looking_at_AGV': False, 

    #special feature
    "Gaze ratio": 0, 

    # Station details (For now, we assume that we know the start and end stations)
    # Later, we can compute the minimum of these distance from all stations or something similar
    "start_station_X": -100., "start_station_Y": -100.,
    "end_station_X": -100., "end_station_Y": -100.,
    "distance_from_start_station_X": -100., "distance_from_start_station_Y": -100.,
    "distance_from_end_station_X": -100., "distance_from_end_station_Y": -100.,
    "facing_start_station": False, "facing_end_station": False,

    # Keep raw features
    "GazeDirection_X": 0, "GazeDirection_Y": 0, "GazeDirection_Z": 0,
    "AGV_X": 0, "AGV_Y": 0, "User_X": 0, "User_Y": 0,
    "AGV_name": "AGV", "TimestampID": 0, "Timestamp": None,
    # "User_Yaw": 0,
}


STATE = {"Error", "Wait", "Arrived", "Cross", "Approach", "Start"}

CUSTOM_PALETTE = {
    "Error"    : "gray",
    "Wait"     : "yellow",
    "Arrived"  : "green",
    "Cross"    : "red",
    "Approach Sidewalk" : "purple", 
    # "Approach" : "orange",
    "Move Along Sidewalk" : "orange",
    "Approach Target Station" : "green",
    "Start"    : "blue", 
    "At Station" : "green"
}

CUSTOM_MARKERS = {
    "Error"     : "X", # X
    "Wait"      : "D", # diamond
    "Arrived"   : "*", # star
    "Cross"     : "^", # triangle
    "Approach Sidewalk" : "P", 
    "Move Along Sidewalk" : "o",
    "Approach Target Station" : "P",
    # "Approach"  : "o", # circle
    "Start"     : "P", 
    "At Station" : "*"
}

PLOTLY_CUSTOM_MARKERS = {
    "Error": "x",
    "Wait": "diamond",
    "Arrived": "circle-x",
    "Cross": "triangle-up",
    "Approach Sidewalk": "hexagon",
    "Move Along Sidewalk": "diamond-cross",
    "Approach Target Station": "triangle-down",
    "Start": "circle-x", 
    # "At Station": "circle-x"
}

