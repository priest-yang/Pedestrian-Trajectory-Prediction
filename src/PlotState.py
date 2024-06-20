
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from src.constant import *
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


def add_sidewalk(fig, ax, x0, y0, x1, y1, showlegend=False):
    ax.add_line(Line2D(xdata=[x0, x1], ydata=[y0, y1], label='sidewalks',
                       linestyle='dashed', linewidth=2, color='black'))
                    # line=dict(color='black', width=2, dash='dash'),
                    # showlegend=showlegend)

    return fig, ax


def draw_rectangle(fig: plt.Figure, ax : plt.Axes, center: tuple, lx: float, ly: float, label: int):
    cx, cy = center
    x0 = cx - lx/2
    y0 = cy - ly/2
    ax.add_patch(
        Rectangle((x0, y0), lx, ly, fill=False, edgecolor='black', linewidth=2, label=str(label))
    )
    dy = -200
    if cy > 8000:
        dy = 150
        
    ax.text(
        cx,
        cy+dy,
        f"{label}",
        bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round'), 
        
    )
    return fig, ax


def plot_FSM_state_scatter(df: pd.DataFrame, save_path, key='state'):
    # for i in range(df['AGV_name'].nunique()):

    assert(df['AGV_name'].nunique() == 1)
    AGV_name = df['AGV_name'].iloc[0]
    agv_num = int(re.search(r'\d+', AGV_name).group())

    fig, ax = plt.subplots(figsize=(28, 10))

    sns.lineplot(data=df, x='AGV_X', y='AGV_Y', ax=ax, sort=False, label='AGV trajectory')
    sns.lineplot(data=df, x='User_X', y='User_Y', ax=ax, sort=False, label='User trajectory')
  
    # sns.scatterplot(data=df, x='AGV_X', y='AGV_Y', ax=ax, hue='FAM_state')
    sns.scatterplot(data=df, x='User_X', y='User_Y', ax=ax, hue=key, style=key, s=200, alpha=.5,
                    palette=CUSTOM_PALETTE, markers=CUSTOM_MARKERS)
    plt.legend(loc='upper right')
    
    for k, v in stations.items():
        fig, ax = draw_rectangle(fig, ax, v, 500, 100, k)
    for i, v in enumerate(sidewalks.values()):
        if i == 0:
            fig, ax = add_sidewalk(fig, ax, *v, showlegend=True)
        else:
            fig, ax = add_sidewalk(fig, ax, *v, showlegend=False)

    plt.xlim(0, 15000) 
    plt.ylim(5000, 10000) 
    # print(agv_to_stations[agv_num])
    ax.set_title(f'AGV{agv_num}, from station {agv_to_stations[agv_num][0]} to station {agv_to_stations[agv_num][1]}')
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'AGV{agv_num}.png'))
    # plt.show()
    plt.close()


# def plot_state(datapath):
    
#     df = pd.read_pickle(datapath)
    
#     current_directory = os.getcwd()
#     save_directory = os.path.join(current_directory, "../data", "Plots", "State")
#     if not os.path.isdir(save_directory):
#         os.mkdir(save_directory)

#     df = pd.read_pickle(os.path.join(datapath))
#     fig_path = os.path.join(save_directory, datapath[:-4])

#     plot_User_AGV_trajectory_scatter(fig_path, df)


