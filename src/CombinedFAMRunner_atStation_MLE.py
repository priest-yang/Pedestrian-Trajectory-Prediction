import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from constant import *
from PlotState import *
from Data2Feature import *
from IPython.display import display
# from State_checker import *

# disable the warning of pandas function "DataFrame.at"
import warnings

warnings.filterwarnings("ignore")
from FAM import *


# # warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# # pd.options.mode.chained_assignment = None  # default='warn'


# class FiniteAutomationState:
#     """
#     Class of FAM state
#     member variables:
#     - name: name of the state
#     - features: dictionary of features of the state
#     - constraints_satisfied: boolean indicating whether the constraints are satisfied
#     - next_state: next state object

#     member functions:
#     - check: detect whether the constraints are satisfied
#     - transition: return the next state and the probability of the transition


#     Doc: features
#     {
#     "AGV distance X", "AGV distance Y", "AGV speed X", "AGV speed Y", "AGV speed",
#     "User speed X", "User speed Y", "User speed",
#     "Wait time",
#     "intent_to_cross", "Gazing_station", "possible_interaction",
#     "facing_along_sidewalk", "facing_to_road",
#     'On sidewalks', 'On road',
#     'closest_station', 'distance_to_closest_station',
#     'looking_at_AGV',
#     **raw features**
#     "GazeDirection_X", "GazeDirection_Y", "GazeDirection_Z",
#     "AGV_X", "AGV_Y", "User_X", "User_Y",
#     "AGV_name", "TimestampID", "Timestamp",
#     }
#     """

#     def __init__(self, features=None) -> None:
#         self.name = None
#         self.constraints_satisfied = False
#         if features is None:
#             self.features = DEFAULT_FEATURE
#         else:
#             self.features = features

#     @staticmethod
#     def check(features=None) -> bool:
#         """
#         Detect whether the constraints are satisfied
#         Update the constraints_satisfied variable
#         """
#         assert features is not None, 'Features are not provided'

#         raise NotImplementedError

#     def transition(self):
#         """
#         Return the next state and the probability of the transition
#         Return: next_state, probability
#         """
#         raise NotImplementedError

#     def update_features(self, features):
#         """
#         Update the features of the state
#         """
#         self.features = features


# class ErrorState(FiniteAutomationState):
#     def __init__(self, features=None, S_prev = 'Error') -> None:
#         super().__init__(features)
#         self.name = 'Error'
#         self.S_prev = S_prev
#         self.MLE = pd.DataFrame({
#             'Wait'    : [0.022282, 0.805907, 0.087452, 0.015152, 0.000000, 0.0075], 
#             'At Station'    : [0.950089, 0.018987, 0.000000, 0.000000, 0.252874, 0.0000], 
#             'Approach Sidewalk'    : [0.023619, 0.126582, 0.562738, 0.000000, 0.000000, 0.0050], 
#             'Move Along Sidewalk'    : [0.000446, 0.008439, 0.087452, 0.003367, 0.004598, 0.9200], 
#             'Approach Target Station'    : [0.003565, 0.000000, 0.000000, 0.131313, 0, 0.0650], 
#             'Cross'   : [0.000000, 0.040084, 0.262357, 0.850168, 0.000000, 0.0025], 
#         }, index=['At Station', 'Wait', 'Approach Sidewalk', 'Cross', 'Approach Target Station', 'Move Along Sidewalk'])


#     @staticmethod
#     def check(features=None) -> bool:
#         assert features is not None, 'Features are not provided'
#         return True

#     def transition(self, features=None):
#         # TODO: consider the order
#         features = self.features

#         assert features is not None, 'Features are not provided'

#         if self.S_prev == 'Error':
#             # if StartState.check(features=features):
#             #     return StartState(features), 1
#             if AtStationState.check(features=features):
#                 return AtStationState(features), 1

#             if WaitingState.check(features=features):
#                 return WaitingState(features), 1

#             if CrossingState.check(features=features):
#                 return CrossingState(features), 1

#             if ApproachingSidewalkState.check(features=features):
#                 return ApproachingSidewalkState(features), 1

#             if MovingAlongSidewalkState.check(features=features):
#                 return MovingAlongSidewalkState(features), 1

#             if ApproachingStationState.check(features=features):
#                 return ApproachingStationState(features), 1

#         else:
#             Q = []
#             if AtStationState.check(features=features):
#                 Q.append(('At Station', self.MLE.at[self.S_prev, 'At Station']))
#             if WaitingState.check(features=features):
#                 Q.append(('Wait', self.MLE.at[self.S_prev, 'Wait']))
#             if CrossingState.check(features=features):
#                 Q.append(('Cross', self.MLE.at[self.S_prev, 'Cross']))
#             if ApproachingSidewalkState.check(features=features):
#                 Q.append(('Approach Sidewalk', self.MLE.at[self.S_prev, 'Approach Sidewalk']))
#             if MovingAlongSidewalkState.check(features=features):
#                 Q.append(('Move Along Sidewalk', self.MLE.at[self.S_prev, 'Move Along Sidewalk']))
#             if ApproachingStationState.check(features=features):
#                 Q.append(('Approach Target Station', self.MLE.at[self.S_prev, 'Approach Target Station']))

#             if len(Q) == 0:
#                 return self, 1
#             else:
#                 Q = sorted(Q, key=lambda x: x[1], reverse=True)
#                 max_state = Q[0][0]
#                 if max_state == 'At Station':
#                     return AtStationState(features), 1
#                 if max_state == 'Wait':
#                     return WaitingState(features), 1
#                 if max_state == 'Cross':
#                     return CrossingState(features), 1
#                 if max_state == 'Approach Sidewalk':
#                     return ApproachingSidewalkState(features), 1
#                 if max_state == 'Move Along Sidewalk':
#                     return MovingAlongSidewalkState(features), 1
#                 if max_state == 'Approach Target Station':
#                     return ApproachingStationState(features), 1

#             # Using MLE to get the next state

#         # if ArrivedState.check(features=features):
#         #     return ArrivedState(features), 1


#         return self, 1

# class AtStationState(FiniteAutomationState):
#     def __init__(self, features=None) -> None:
#         super().__init__(features)
#         self.name = 'At Station'

#     @staticmethod
#     def check(features=None) -> bool:
#         assert features is not None, 'Features are not provided'
#         # Constraints:
#         #       1. Be stationary
#         #       2. Be within some small distance of the station
#         #       3. Not be on the road

#         # Check for speed
#         stationary = abs(features['User speed']) <= WALK_STAY_THRESHOLD * 2
#         # Checks for distance from start and end stations
#         # near_station = features['distance_to_closest_station'] < CLOSE_TO_STATION_THRESHOLD * 100
#         near_station_X = features['distance_to_closest_station_X'] < CLOSE_TO_STATION_THRESHOLD_X * 200
#         near_station_Y = features['distance_to_closest_station_Y'] < CLOSE_TO_STATION_THRESHOLD_Y * 200
#         near_station = near_station_X and near_station_Y
#         # Check for on the road
#         on_road = features['On road']

#         if stationary and near_station and (not on_road):
#             return True
#         else:
#             return False

#     def transition(self) -> (FiniteAutomationState, float):
#         features = self.features
#         assert features is not None, 'Features are not provided'

#         # AtStation -> ApproachingSidewalkState
#         if abs(features['User speed']) > WALK_STAY_THRESHOLD \
#                 and \
#                 (features['On sidewalks'] or features['facing_along_sidewalk']):
#             next_state = ApproachingSidewalkState(features)
#             return next_state, 1

#         # AtStation -> WaitingState
#         elif abs(features['User speed']) <= WALK_STAY_THRESHOLD \
#                 and \
#                 features['intent_to_cross'] \
#                 and \
#                 features['possible_interaction']:
#             next_state = WaitingState(features)
#             return next_state, 1

#         # Stay in AtStation State
#         else:
#             return self, 1


# class WaitingState(FiniteAutomationState):
#     def __init__(self, features=None) -> None:
#         super().__init__(features)
#         self.name = 'Wait'

#     @staticmethod
#     def check(features=None) -> bool:
#         # return WaitingStateChecker(features=features)
#         assert features is not None, 'Features are not provided'

#         if abs(features['User speed']) <= WALK_STAY_THRESHOLD and \
#             (features['possible_interaction'] or features['looking_at_AGV'] or features['On road']):
#             return True
#         else:
#             return False

#     def transition(self):
#         features = self.features
#         assert features is not None, 'Features are not provided'

#         # WaitingState -> CrossingState
#         if abs(features['User speed']) > 0.8 * WALK_STAY_THRESHOLD \
#                 and features['On road'] \
#                 and features['facing_to_road']:
#             next_state = CrossingState(features)
#             return next_state, 1

#         # WaitingState -> ApproachingSidewalkState
#         # TODO: Why is the 0.8 multiplier not used here for the speed check?
#         elif abs(features['User speed']) > WALK_STAY_THRESHOLD \
#                 and features['On sidewalks']:
#             next_state = ApproachingSidewalkState(features)
#             return next_state, 1

#         # WaitingState -> MovingAlongSidewalkState
#         elif abs(features['User speed X']) > 0.8 * WALK_STAY_THRESHOLD \
#                 and (features['On sidewalks'] or features['facing_along_sidewalk']):
#             next_state = MovingAlongSidewalkState(features)
#             return next_state, 1

#         # else stay in WaitingState
#         else:
#             return self, 1


# # class ArrivedState(FiniteAutomationState):
# #     def __init__(self, features=None) -> None:
# #         super().__init__(features)
# #         self.name = 'Arrived'

# #     @staticmethod
# #     def check(features=None) -> bool:
# #         assert features is not None, 'Features are not provided'
# #         # Constraints:
# #         #       1. Be stationary
# #         #       2. Be within some small distance of the station and looking towards the station
# #         #       3. Not be on the road

# #         # Check for speed
# #         stationary = abs(features['User speed']) < WALK_STAY_THRESHOLD

# #         # # Checks for distance from end station
# #         # near_end_station = features['distance_from_end_station_X'] < STATION_LENGTH * 100.
# #         # near_end_station = near_end_station and features['distance_from_end_station_Y'] < RADIUS_1 * 100.
# #         # facing_end_station = features['facing_end_station']              # Check for facing the end station
# #         # near_and_facing_end_station = near_end_station and facing_end_station

# #         # Check for on the road
# #         on_road = features['On road']

# #         # Check for looking towards the station
# #         # TODO: I think the near and facing end station condition should ensure that the user is not on the road.
# #         # TODO: So, the not on_road condition may be redundant

# #         # shawn: reduce the near_and_facing_end_station for now:
# #         # From observation, the user may stop and looking around when they arrived. 
# #         # if stationary and near_and_facing_end_station and (not on_road):
# #         #     return True
# #         # else:
# #         #     return False

# #         if stationary and (not on_road):
# #             return True
# #         else:
# #             return False

# #     def transition(self):
# #         features = self.features
# #         assert features is not None, 'Features are not provided'
# #         return self, 1


# class CrossingState(FiniteAutomationState):
#     def __init__(self, features=None) -> None:
#         super().__init__(features)
#         self.name = 'Cross'

#     @staticmethod
#     def check(features=None) -> bool:
#         assert features is not None, 'Features are not provided'

#         # If the worker is moving in the Y direction and is on the road
#         moving = abs(features['User speed Y']) > WALK_STAY_THRESHOLD  # TODO: tune this threshold
#         on_road = features['On road']

#         # shawn: add user's gazing into consideration
#         looking_at_road = features['facing_to_road']
#         looking_at_agv = features['looking_at_AGV']

#         if moving and on_road and (looking_at_road or looking_at_agv):
#             return True
#         return False

#     def transition(self):
#         features = self.features
#         assert features is not None, 'Features are not provided'

#         # CrossingState -> MovingAlongSidewalkState
#         if features['On sidewalks'] \
#                 and \
#                 (abs(features['User speed X']) > 1.5 * abs(features['User speed Y'])\
#                     or( features['facing_along_sidewalk'] and abs(features['User speed X']) > 0.5 * WALK_STAY_THRESHOLD)):
#             next_state = MovingAlongSidewalkState(features)
#             return next_state, 1

#         # CrossingState -> ApproachingStationState
#         if abs(features['User speed']) > WALK_STAY_THRESHOLD \
#                 and \
#                 features['closest_station'] == features['Gazing_station'] \
#                 and \
#                 not features['On road']:
#             next_state = ApproachingStationState(features)
#             return next_state, 1

#         # CrossingState -> WaitingState (wait for AGV)
#         elif abs(features['User speed']) < WALK_STAY_THRESHOLD \
#                 and \
#                 features['possible_interaction'] \
#                 and \
#                 features['looking_at_AGV'] \
#                 and \
#                 features['On road']:
#             next_state = WaitingState(features)
#             return next_state, 1

#         # CrossingState -> ArrivedState
#         elif abs(features['User speed']) < WALK_STAY_THRESHOLD \
#                 and \
#                 (not features['facing_to_road']) \
#                 and \
#                 features['distance_to_closest_station'] <= CLOSE_TO_STATION_THRESHOLD * 100:
#             # next_state = ArrivedState(features)
#             next_state = AtStationState(features)
#             return next_state, 1

#         # else stay in CrossingState
#         else:
#             return self, 1


# class ApproachingSidewalkState(FiniteAutomationState):
#     def __init__(self, features=None) -> None:
#         super().__init__(features)
#         self.name = 'Approach Sidewalk'

#     @staticmethod
#     def check(features=None) -> bool:
#         # return ApproachingSidewalkStateChecker(features=features)

#         assert features is not None, 'Features are not provided'
#         # check whether the constraints are satisfied
#         # near_start_station = features['distance_from_start_station_X'] < STATION_LENGTH * 100
#         # near_start_station = near_start_station and features['distance_from_start_station_Y'] < RADIUS_2 * 100

#         near_start_station = abs(features['distance_to_closest_station_Y']) <= CLOSE_TO_STATION_THRESHOLD_Y * 100 * 2

#         # facing_start_station = features['facing_start_station']  # Check for facing the start station
#         moving = features['User speed Y'] > WALK_STAY_THRESHOLD * 0.3  # TODO: tune this threshold

#         if near_start_station and moving and (not features['On road']):  # and not facing_start_station
#             return True

#         return False

#     def transition(self):
#         features = self.features
#         assert features is not None, 'Features are not provided'

#         near_station_X = features['distance_to_closest_station_X'] < CLOSE_TO_STATION_THRESHOLD_X * 100
#         near_station_Y = features['distance_to_closest_station_Y'] < CLOSE_TO_STATION_THRESHOLD_Y * 100
#         near_station = near_station_X and near_station_Y

#         # ApproachingSidewalkState -> CrossingState
#         if abs(features['User speed Y']) > 0.5 * WALK_STAY_THRESHOLD \
#                 and features['facing_to_road'] \
#                     and features['On road']: # new added
#             next_state = CrossingState(features)
#             return next_state, 1

#         # ApproachingSidewalkState -> WaitingSidewalkState
#         elif abs(features['User speed']) < WALK_STAY_THRESHOLD \
#                 and features['intent_to_cross'] \
#                 and features['possible_interaction']:
#             next_state = WaitingState(features)
#             return next_state, 1

#         # ApproachingSidewalkState -> MovingAlongSidewalkState
#         elif (abs(features['User speed X']) > 1.5 * abs(features['User speed Y'])\
#                     or( features['facing_along_sidewalk'] and abs(features['User speed X']) > WALK_STAY_THRESHOLD))\
#                         and (not near_station or features["facing_along_sidewalk"]):
#             next_state = MovingAlongSidewalkState(features)
#             return next_state, 1

#         # else stay in ApproachingSidewalkState
#         else:
#             return self, 1


# class MovingAlongSidewalkState(FiniteAutomationState):
#     def __init__(self, features=None) -> None:
#         super().__init__(features)
#         self.name = 'Move Along Sidewalk'

#     @staticmethod
#     def check(features=None) -> bool:
#         # return MovingAlongSidewalkStateChecker(features=features)
#         assert features is not None, 'Features are not provided'
#         # check whether the constraints are satisfied
#         moving = features['User speed X'] > WALK_STAY_THRESHOLD * 0.8  # TODO: tune this threshold

#         within_sidewalk = features['distance_from_start_station_Y'] < 500 + MARGIN_NEAR_SIDEWALKS * 100
#         within_sidewalk = within_sidewalk or (features['distance_from_end_station_Y'] < 500 + MARGIN_NEAR_SIDEWALKS * 100)

#         if within_sidewalk and moving:
#             return True

#         return False

#     def transition(self):
#         features = self.features
#         assert features is not None, 'Features are not provided'

#         # MovingAlongSidewalkState -> CrossingState
#         if ( (abs(features['User speed Y']) > 1.5 * abs(features['User speed X']) \
#             or (abs(features['User speed Y']) > WALK_STAY_THRESHOLD) and features['facing_to_road']) )\
#                 and (features['intent_to_cross'] or features['On road']): # new added:
#             next_state = CrossingState(features)
#             return next_state, 1

#         # MovingAlongSidewalkState -> WaitingState
#         elif abs(features['User speed']) < WALK_STAY_THRESHOLD \
#                 and features['intent_to_cross'] \
#                 and features['possible_interaction']:
#             next_state = WaitingState(features)
#             return next_state, 1

#         # MovingAlongSidewalkState -> ApproachingStationState
#         elif (abs(features['User speed']) < WALK_STAY_THRESHOLD or features['looking_at_closest_station'])\
#                 and (not features['facing_to_road']) \
#                 and features['distance_to_closest_station'] <= CLOSE_TO_STATION_THRESHOLD * 200:
#             next_state = ApproachingStationState(features)
#             return next_state, 1

#         # else stay in MovingAlongSidewalkState
#         else:
#             return self, 1


# class ApproachingStationState(FiniteAutomationState):
#     def __init__(self, features=None) -> None:
#         super().__init__(features)
#         self.name = 'Approach Target Station'

#     @staticmethod
#     def check(features=None) -> bool:
#         # return ApproachingStationStateChecker(features=features)
#         assert features is not None, 'Features are not provided'
#         # check whether the constraints are satisfied
#         near_station = features['distance_from_end_station_X'] < STATION_LENGTH * 200
#         near_station = near_station and features['distance_from_end_station_Y'] < CLOSE_TO_STATION_THRESHOLD * 150
#         looking_at_station = features['facing_end_station']


#         if (not features['On road']) and near_station and (features['User speed'] > WALK_STAY_THRESHOLD * 0.2):  # and looking_at_station
#             return True

#         return False

#     def transition(self):
#         features = self.features
#         assert features is not None, 'Features are not provided'

#         # ApproachingStationState -> ArrivedState
#         if abs(features['User speed']) < WALK_STAY_THRESHOLD\
#                 and (not features['facing_to_road']) \
#                     and features['distance_to_closest_station'] <= CLOSE_TO_STATION_THRESHOLD * 300:
#             # next_state = ArrivedState(features)
#             next_state = AtStationState(features)
#             return next_state, 1

#         # Stay in ApproachingStationState
#         else:
#             return self, 1


# # class StartState(FiniteAutomationState):
# #     def __init__(self, features=None) -> None:
# #         super().__init__(features)
# #         self.name = 'Start'
# #         pass

# #     @staticmethod
# #     def check(features=None) -> bool:
# #         # return StartStateChecker(features=features)
# #         assert features is not None, 'Features are not provided'
# #         # check whether the constraints are satisfied

# #         # TODO: check the constraints
# #         # Constraints:
# #         #       1. Be stationary
# #         #       2. Be within some small distance of the station and looking towards the station
# #         #       3. Not be on the road (TODO: Not sure about this feature)

# #         # Check for speed
# #         stationary = abs(features['User speed']) < WALK_STAY_THRESHOLD

# #         # Checks for distance from start and end stations
# #         near_start_station = features['distance_to_closest_station'] < CLOSE_TO_STATION_THRESHOLD * 100

# #         # near_start_station = features['distance_from_start_station_X'] < STATION_LENGTH * 100
# #         # near_start_station = near_start_station and features['distance_from_start_station_Y'] < RADIUS_1 * 100
# #         # facing_start_station = features['facing_start_station']       
# #         # near_and_facing_start_station = near_start_station and facing_start_station

# #         # Check for on the road
# #         on_road = features['On road']

# #         # Check for looking towards the station
# #         # TODO: I think the near and facing start station condition should ensure that the user is not on the road.
# #         # TODO: So, the not on_road condition may be redundant
# #         if stationary and near_start_station and (not on_road):
# #             return True
# #         else:
# #             return False

# #     def transition(self) -> (FiniteAutomationState, float):
# #         features = self.features
# #         assert features is not None, 'Features are not provided'

# #         # StartState -> ApproachingState
# #         if abs(features['User speed']) > WALK_STAY_THRESHOLD \
# #                 and \
# #                 (features['On sidewalks'] or features['facing_along_sidewalk']):
# #             next_state = ApproachingSidewalkState(features)
# #             return next_state, 1

# #         # StartState -> WaitingState
# #         elif abs(features['User speed']) <= WALK_STAY_THRESHOLD \
# #                 and \
# #                 features['intent_to_cross'] \
# #                 and \
# #                 features['possible_interaction']:
# #             next_state = WaitingState(features)
# #             return next_state, 1

# #         # Stay in StartState
# #         else:
# #            return self, 1


# class FiniteAutomationMachine:
#     def __init__(self, features: dict = None) -> None:
#         self.current_state = ErrorState(features)  # initialize at the Unknown state
#         self.next_state = None
#         self.errorFlag = [True, True, False]  # True for satisfied constraints, False otherwise
#         self.S_prev = 'Error'
#         self.probabilityTransitionMatrix = None


#     def run(self, features: dict):
#         '''
#         Transition to the next state
#         inputs: features / row of dataframe
#         '''
#         self.current_state.update_features(features)
#         self.next_state, probability = self.current_state.transition()
#         if probability > 0.8:
#             self.current_state = self.next_state

#         # check whether the constraints are satisfied
#         constraints_satisfied = self.current_state.check(features=self.current_state.features)
#         self.errorFlag = self.errorFlag[1:] + [constraints_satisfied]

#         if not any(self.errorFlag):  # if the constraints are not satisfied for past 3 times
#             self.S_prev = self.current_state.name
#             self.current_state = ErrorState(None, S_prev=self.S_prev)
#             self.errorFlag = [True, True, False]


class MLECombinedFAMRunner:
    def __init__(self, datapath: str = None, savepath: str = None, plot: bool = None, fig_save_dir: str = None,
                 error_flag_size: int = 3) -> None:
        self.MODEL_NAME = 'MLEcombined_FAM'
        self.datapath = datapath
        self.savepath = savepath
        self.plot = plot
        self.fig_save_dir = fig_save_dir
        self.error_flag_size = error_flag_size

        # for real time run
        self.df_under_tst = pd.DataFrame(columns=DEFAULT_FEATURE.keys())
        self.result_df = pd.DataFrame(columns=DEFAULT_FEATURE.keys())
        self.cur_AGV = None
        self.FAM = None
        self.iter = 0
        self.D2F = Data2Feature(data=DATA_DF.copy(), feature=DEFAULT_DF.copy())

    def set_param_(self, datapath: str = None, savepath: str = None, plot: bool = None, fig_save_dir: str = None,
                   error_flag_size: int = None):
        if datapath is not None:
            self.datapath = datapath
        if savepath is not None:
            self.savepath = savepath
        if plot is not None:
            self.plot = plot
        if fig_save_dir is not None:
            self.fig_save_dir = fig_save_dir
        if error_flag_size is not None:
            self.error_flag_size = error_flag_size

    def run(self, datapath: str = None, savepath: str = None, plot: bool = None,
            fig_save_dir: str = None, error_flag_size: int = None, drop_feature: list = None,
            save: bool = False) -> pd.DataFrame:
        """
        Run the FAM on the data
        """
        if datapath is None:
            datapath = self.datapath
        if savepath is None:
            savepath = self.savepath
        if plot is None:
            plot = self.plot
        if fig_save_dir is None:
            fig_save_dir = self.fig_save_dir
        if error_flag_size is None:
            error_flag_size = self.error_flag_size

        assert datapath is not None, 'Datapath is not provided'
        assert savepath is not None, 'Savepath is not provided'

        # if plot:
        #     current_directory = os.getcwd()
        #     save_directory = os.path.join(current_directory, "data", "Plots", self.MODEL_NAME)
        #     if not os.path.exists(save_directory):
        #         os.makedirs(save_directory)
        #         print('Creating directory: ', save_directory)
        #
        #     print('Saving figures to: ', save_directory)

        # load data
        df = pd.read_pickle(datapath)
        result_df = pd.DataFrame()

        # drop features if needed (for experiments)
        if drop_feature is not None:
            df[drop_feature] = 0

        AGV_name_list = df['AGV_name'].unique()

        print("running Combined FAM")
        # print(AGV_name_list)
        for AGV in AGV_name_list:

            df_under_tst = df[df['AGV_name'] == AGV]

            # initialize the FAM
            features = df_under_tst.iloc[1].to_dict()
            FAM = FiniteAutomationMachine(features, error_flag_size=error_flag_size)
            df_under_tst.at[0, 'state'] = FAM.current_state.name

            # run the FAM
            for index, row in df_under_tst.iterrows():
                if index < 1:
                    continue
                features = row.to_dict()
                FAM.run(features)
                df_under_tst.at[index, 'state'] = FAM.current_state.name

            # plot the result
            if plot:
                current_directory = os.getcwd()
                save_directory = os.path.join(current_directory, "data", "Plots", self.MODEL_NAME)
                type = re.search(r'PID(\d+_(NSL|SLD))', datapath).group(1)

                if fig_save_dir is not None:
                    save_directory = fig_save_dir

                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)
                    # print('Creating directory: ', save_directory)

                # print('Saving figures to: ', save_directory)
                plot_FSM_state_scatter(df_under_tst, os.path.join(save_directory, type), key='state')

            result_df = pd.concat([result_df, df_under_tst])

        # save the data
        if save:
            result_df.to_pickle(savepath)
        return result_df

    def run_real_time(self,
                      AGV_distance_X: float = None,
                      AGV_distance_Y: float = None,
                      AGV_speed_X: float = None,
                      AGV_speed_Y: float = None,
                      AGV_speed: float = None,
                      User_speed_X: float = None,
                      User_speed_Y: float = None,
                      User_speed: float = None,
                      User_velocity_X: float = None,
                      User_velocity_Y: float = None,
                      Wait_time: float = None,
                      intent_to_cross: bool = None,
                      Gazing_station: str = None,
                      possible_interaction: bool = None,
                      facing_along_sidewalk: bool = None,
                      facing_to_road: bool = None,
                      On_sidewalks: bool = None,
                      On_road: bool = None,
                      closest_station: str = None,
                      distance_to_closest_station: float = None,
                      distance_to_closest_station_X: float = None,
                      distance_to_closest_station_Y: float = None,
                      looking_at_AGV: bool = None,
                      start_station_X: float = None,
                      start_station_Y: float = None,
                      end_station_X: float = None,
                      end_station_Y: float = None,
                      distance_from_start_station_X: float = None,
                      distance_from_start_station_Y: float = None,
                      distance_from_end_station_X: float = None,
                      distance_from_end_station_Y: float = None,
                      facing_start_station: bool = None,
                      facing_end_station: bool = None,

                      # special feature:
                      Gaze_ratio: float = None,

                      GazeDirection_X: float = None,
                      GazeDirection_Y: float = None,
                      GazeDirection_Z: float = None,
                      AGV_X: float = None,
                      AGV_Y: float = None,
                      User_X: float = None,
                      User_Y: float = None,
                      AGV_name: str = None,
                      TimestampID: str = None,
                      Timestamp: str = None,
                      ) -> None:
        feature = {
            "AGV distance X": AGV_distance_X,
            "AGV distance Y": AGV_distance_Y,
            "AGV speed X": AGV_speed_X,
            "AGV speed Y": AGV_speed_Y,
            "AGV speed": AGV_speed,
            "User speed X": User_speed_X,
            "User speed Y": User_speed_Y,
            "User speed": User_speed,
            "User velocity X": User_velocity_X,
            "User velocity Y": User_velocity_Y,
            "Wait time": Wait_time,
            "intent_to_cross": intent_to_cross,
            "Gazing_station": Gazing_station,
            "possible_interaction": possible_interaction,
            "facing_along_sidewalk": facing_along_sidewalk,
            "facing_to_road": facing_to_road,
            'On sidewalks': On_sidewalks,
            'On road': On_road,
            'closest_station': closest_station,
            'distance_to_closest_station': distance_to_closest_station,
            'distance_to_closest_station_X': distance_to_closest_station_X,
            'distance_to_closest_station_Y': distance_to_closest_station_Y,
            'looking_at_AGV': looking_at_AGV,
            'start_station_X': start_station_X,
            'start_station_Y': start_station_Y,
            'end_station_X': end_station_X,
            'end_station_Y': end_station_Y,
            'distance_from_start_station_X': distance_from_start_station_X,
            'distance_from_start_station_Y': distance_from_start_station_Y,
            'distance_from_end_station_X': distance_from_end_station_X,
            'distance_from_end_station_Y': distance_from_end_station_Y,
            'facing_start_station': facing_start_station,
            'facing_end_station': facing_end_station,

            # special feature:
            'Gaze ratio': Gaze_ratio,
            # Keep raw features
            "GazeDirection_X": GazeDirection_X,
            "GazeDirection_Y": GazeDirection_Y,
            "GazeDirection_Z": GazeDirection_Z,
            "AGV_X": AGV_X,
            "AGV_Y": AGV_Y,
            "User_X": User_X,
            "User_Y": User_Y,
            "AGV_name": AGV_name,
            "TimestampID": TimestampID,
            "Timestamp": Timestamp,
        }
        # update to next FAM
        if self.cur_AGV is not None and AGV_name != self.cur_AGV:
            self.result_df = pd.concat([self.result_df, self.df_under_tst])
            self.iter = 0
            self.df_under_tst = pd.DataFrame(columns=DEFAULT_FEATURE.keys())
            self.cur_AGV = AGV_name
            self.FAM = FiniteAutomationMachine(features=feature)
            self.df_under_tst.loc[self.iter] = DEFAULT_FEATURE
            self.df_under_tst.at[self.iter, 'state'] = self.FAM.current_state.name
            self.iter += 1
        # first shot
        elif self.FAM is None:
            self.cur_AGV = AGV_name
            self.FAM = FiniteAutomationMachine(features=feature)
            self.df_under_tst.loc[self.iter] = DEFAULT_FEATURE
            self.df_under_tst.at[self.iter, 'state'] = self.FAM.current_state.name
            self.iter += 1
        else:
            self.FAM.run(features=feature)
            self.df_under_tst.loc[self.iter] = feature
            self.df_under_tst.at[self.iter, 'state'] = self.FAM.current_state.name
            self.iter += 1

    def run_real_time_raw(self,
                          PID=None,
                          SCN=None,
                          TimestampID=None,
                          Timestamp=None,
                          DatapointID=None,
                          AGV_name=None,
                          User_X=None,
                          User_Y=None,
                          User_Z=None,
                          User_Pitch=None,
                          User_Yaw=None,
                          User_Roll=None,
                          U_X=None,
                          U_Y=None,
                          U_Z=None,
                          AGV_X=None,
                          AGV_Y=None,
                          AGV_Z=None,
                          AGV_Pitch=None,
                          AGV_Yaw=None,
                          AGV_Roll=None,
                          AGV_speed=None,
                          EyeTarget=None,
                          GazeOrigin_X=None,
                          GazeOrigin_Y=None,
                          GazeOrigin_Z=None,
                          GazeDirection_X=None,
                          GazeDirection_Y=None,
                          GazeDirection_Z=None,
                          Confidence=None
                          ):
        # update to next FAM, update feature matrix
        if self.cur_AGV is not None and AGV_name != self.cur_AGV:
            self.D2F = Data2Feature()

        self.D2F.update(
            PID=PID,
            SCN=SCN,
            TimestampID=TimestampID,
            Timestamp=Timestamp,
            DatapointID=DatapointID,
            AGV_name=AGV_name,
            User_X=User_X,
            User_Y=User_Y,
            User_Z=User_Z,
            User_Pitch=User_Pitch,
            User_Yaw=User_Yaw,
            User_Roll=User_Roll,
            U_X=U_X,
            U_Y=U_Y,
            U_Z=U_Z,
            AGV_X=AGV_X,
            AGV_Y=AGV_Y,
            AGV_Z=AGV_Z,
            AGV_Pitch=AGV_Pitch,
            AGV_Yaw=AGV_Yaw,
            AGV_Roll=AGV_Roll,
            AGV_speed=AGV_speed,
            EyeTarget=EyeTarget,
            GazeOrigin_X=GazeOrigin_X,
            GazeOrigin_Y=GazeOrigin_Y,
            GazeOrigin_Z=GazeOrigin_Z,
            GazeDirection_X=GazeDirection_X,
            GazeDirection_Y=GazeDirection_Y,
            GazeDirection_Z=GazeDirection_Z,
            Confidence=Confidence
        )

        feature = self.D2F.get_feature()
        # update to next FAM
        if self.cur_AGV is not None and AGV_name != self.cur_AGV:
            self.result_df = pd.concat([self.result_df, self.df_under_tst])
            self.iter = 0
            self.df_under_tst = pd.DataFrame(columns=DEFAULT_FEATURE.keys())
            self.cur_AGV = AGV_name
            self.FAM = FiniteAutomationMachine(features=feature)
            self.df_under_tst.loc[self.iter] = DEFAULT_FEATURE
            self.df_under_tst.at[self.iter, 'state'] = self.FAM.current_state.name
            self.iter += 1
        # first shot
        elif self.FAM is None:
            self.cur_AGV = AGV_name
            self.FAM = FiniteAutomationMachine(features=feature)
            self.df_under_tst.loc[self.iter] = DEFAULT_FEATURE
            self.df_under_tst.at[self.iter, 'state'] = self.FAM.current_state.name
            self.iter += 1
        else:
            self.FAM.run(features=feature)
            self.df_under_tst.loc[self.iter] = feature
            self.df_under_tst.at[self.iter, 'state'] = self.FAM.current_state.name
            self.iter += 1

        return self.FAM.current_state.name

    def export_result(self, savepath: str = None) -> None:
        if savepath is None:
            savepath = self.savepath
        self.result_df = pd.concat([self.result_df, self.df_under_tst])
        self.result_df.to_pickle(savepath)
        print("export result to ", savepath)


# test for FAM without error states #

class FiniteAutomationMachineWithoutErrorState:
    def __init__(self, features: dict = None) -> None:
        self.current_state = ErrorState(features)  # initialize at the Unknown state
        self.next_state = None
        # self.errorFlag = [True, True, False]  # True for satisfied constraints, False otherwise

    def run(self, features: dict):
        """
        Transition to the next state
        inputs: features / row of dataframe
        """
        self.current_state.update_features(features)
        self.next_state, probability = self.current_state.transition()
        if probability > 0.8:
            self.current_state = self.next_state

        # check whether the constraints are satisfied
        # constraints_satisfied = self.current_state.check(features=self.current_state.features)
        # self.errorFlag = self.errorFlag[1:] + [constraints_satisfied]

        # if not any(self.errorFlag):  # if the constraints are not satisfied for past 3 times
        #     self.current_state = ErrorState(None)
        #     self.errorFlag = [True, True, False]


class FAMRunner:
    def __init__(self, datapath: str = None, savepath: str = None, plot: bool = False,
                 fig_save_dir: str = None) -> None:
        self.MODEL_NAME = 'FAM'

        self.datapath = datapath
        self.savepath = savepath
        self.plot = plot
        self.fig_save_dir = fig_save_dir

    def set_param_(self, datapath: str = None, savepath: str = None, plot: bool = False, fig_save_dir: str = None):
        if datapath is not None:
            self.datapath = datapath
        if savepath is not None:
            self.savepath = savepath
        if plot is not None:
            self.plot = plot
        if fig_save_dir is not None:
            self.fig_save_dir = fig_save_dir

    def run(self, datapath: str = None, savepath: str = None, plot: bool = False,
            fig_save_dir: str = None) -> pd.DataFrame:
        """
        Run the FAM on the data
        """
        if datapath is None:
            datapath = self.datapath
        if savepath is None:
            savepath = self.savepath
        if plot is None:
            plot = self.plot
        if fig_save_dir is None:
            fig_save_dir = self.fig_save_dir

        assert datapath is not None, 'Datapath is not provided'
        assert savepath is not None, 'Savepath is not provided'

        # if plot:
        #     current_directory = os.getcwd()
        #     save_directory = os.path.join(current_directory, "data", "Plots", self.MODEL_NAME)
        #     if not os.path.exists(save_directory):
        #         os.makedirs(save_directory)
        #         print('Creating directory: ', save_directory)
        #
        #     print('Saving figures to: ', save_directory)

        # load data
        df = pd.read_pickle(datapath)
        result_df = pd.DataFrame()

        AGV_name_list = df['AGV_name'].unique()

        for AGV in tqdm(AGV_name_list):

            # if AGV != 'AGV15':
            #     continue

            df_under_tst = df[df['AGV_name'] == AGV]

            # initialize the FAM
            features = df_under_tst.iloc[1].to_dict()
            FAM = FiniteAutomationMachineWithoutErrorState(features)
            df_under_tst.at[0, 'state'] = FAM.current_state.name

            # run the FAM
            for index, row in df_under_tst.iterrows():
                if index < 1:
                    continue
                features = row.to_dict()
                FAM.run(features)
                df_under_tst.at[index, 'state'] = FAM.current_state.name

            # plot the result
            if plot:
                current_directory = os.getcwd()
                save_directory = os.path.join(current_directory, "data", "Plots", self.MODEL_NAME)
                type = re.search(r'PID(\d+_(NSL|SLD))', datapath).group(1)

                if fig_save_dir is not None:
                    save_directory = fig_save_dir

                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)
                    # print('Creating directory: ', save_directory)

                # print('Saving figures to: ', save_directory)
                plot_FSM_state_scatter(df_under_tst, os.path.join(save_directory, type), key='state')

            result_df = pd.concat([result_df, df_under_tst])

        # save the data
        result_df.to_pickle(savepath)
        return result_df

# tester for run_real_time function, running FAM based on features


if __name__ == '__main__':

    files = glob.glob('../data/PandasData/Modified/*.pkl')
    for filepath in files:

        filename = filepath.split("\\")[-1]
        filename = filename.strip('.pkl')
        print(filename)

        runner = MLECombinedFAMRunner(plot=False)
        runner.set_param_(
            datapath=filepath,
            savepath=os.path.join('..', 'data', 'PandasData/Predicted/', f'{filename}_combined_FAM.pkl'),
            plot=False,
            fig_save_dir='../data/Plots/combined_FAM'
        )
        # df = pd.read_pickle(os.path.join('..', 'data', 'PandasData/Modified/PID003_NSL.pkl'))
        # df.columns = df.columns.str.replace(' ', '_')
        # for index, row in df.iterrows():
        #     runner.run_real_time(**row.to_dict())
        #
        # result = runner.result_df
        # result.to_pickle(os.path.join('..', 'data', 'PandasData/Predicted/PID003_NSL_combined_FAM.pkl'))
        # display(result.head(20))

        result = runner.run()
        # result = runner.result_df
        result.to_pickle(os.path.join('..', 'data', 'PandasData/Predicted/', f'{filename}_combined_FAM.pkl'))
        # print(result.head())
