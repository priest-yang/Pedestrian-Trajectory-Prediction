import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from .constant import *
from IPython.display import display

# disable the warning of pandas function "DataFrame.at"
import warnings

warnings.filterwarnings("ignore")
from .FAM import *


class MLECombinedFAMRunner:
    def __init__(self, datapath: str = None, savepath: str = None, plot: bool = None, fig_save_dir: str = None, error_flag_size : int = 3) -> None:
        self.MODEL_NAME = 'MLEcombined_FAM'
        self.datapath = datapath
        self.savepath = savepath
        self.plot = plot
        self.fig_save_dir = fig_save_dir
        self.error_flag_size = error_flag_size

        # for real time run
        self.df_under_tst = pd.DataFrame(columns = DEFAULT_FEATURE.keys())
        self.result_df = pd.DataFrame(columns = DEFAULT_FEATURE.keys())
        self.cur_AGV = None
        self.FAM = None
        self.iter = 0
        # self.D2F = Data2Feature(data=DATA_DF.copy(), feature=DEFAULT_DF.copy())


    def set_param_(self, datapath: str = None, savepath: str = None, plot: bool = None, fig_save_dir: str = None, error_flag_size : int = None):
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
            fig_save_dir: str = None, error_flag_size : int = None, drop_feature : list = None, save : bool = False) -> pd.DataFrame:
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
        
        if save:
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

        #drop features if needed (for experiments)
        if drop_feature is not None:
            df[drop_feature] = 0

        AGV_name_list = df['AGV_name'].unique()

        # print("running Combined FAM")
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

            result_df = pd.concat([result_df, df_under_tst])

        # save the data
        if save:
            result_df.to_pickle(savepath)
        return result_df

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
    def __init__(self, datapath: str = None, savepath: str = None, plot: bool = False, fig_save_dir: str = None) -> None:
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

            result_df = pd.concat([result_df, df_under_tst])

        # save the data
        # result_df.to_pickle(savepath)
        return result_df
    
# tester for run_real_time function, running FAM based on features
    

# if __name__ == '__main__':
#     runner = MLECombinedFAMRunner(plot=False)

#     read_dir = "../data/PandasData/Sampled/"
#     save_dir = "../data/PandasData/Sampled_state/"

#     for file in tqdm(os.listdir(read_dir)):
#         if file.endswith(".pkl"):
#             runner.run(read_dir + file, save=True, savepath=save_dir+file)




