from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from typing import List
import numpy as np
import pandas as pd
from dataclasses import dataclass
import math
import pickle as pkl
from tqdm import tqdm

avm = ArgoverseMap()
encode_type = {"AGENT": 1.0, "AV": 2.0, "OTHERS": 2.0, "MAP": 3.0} 

@dataclass
class Obs:
    """Class for keeping track of an item in inventory."""
    cor_x: List[float]
    cor_y: List[float]
    timestamp: List[float]
    type: str
    def __init__(self, cor_x: List[float], cor_y: List[float], type: str, timestamp: List[float] ):
        self.cor_x = cor_x
        self.cor_y = cor_y
        self.timestamp = timestamp
        self.type = type
        
    def encode(self, origin_time):
        indexes = []
        for i in range(len(self.timestamp)):
            if self.timestamp[i] < origin_time+1e-3:
                indexes.append(i)

        # resample
        sample_rate = 4   
        reversed_indexes = indexes[::-1]
        selected_idx = reversed_indexes[::sample_rate]

        vects = []
        for j in range(len(selected_idx)-1):
            id1 = selected_idx[j]
            id0 = selected_idx[j+1]
            vec = [self.cor_x[id0], self.cor_y[id0], self.cor_x[id1], self.cor_y[id1],  encode_type[self.type], self.timestamp[id1]-origin_time ]
            vects.append(vec)
        return vects
 
def encode_centerline(centerline):
    dist_threshod = 4.0
    tar = []
    last_pt = centerline[0]
    for pt in centerline:
        dist = get_dist(pt, last_pt)
        if dist > dist_threshod:
            tar.append([last_pt[0], last_pt[1], pt[0], pt[1], encode_type['MAP'], 0.0])
            last_pt = pt
    return tar


def get_dist(pt1,pt2):
    return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

def dist_square(pt1, pt2):
    return (pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2

def select_n_from_pool(pool, n, center_x, center_y):
    dist_list = []
    for vec in pool:
        mid_x = (vec[0] + vec[2])/2
        mid_y = (vec[1] + vec[3])/2
        # center around 0,0
        dist = dist_square([mid_x, mid_y], [center_x, center_y])
        dist_list.append(dist)

    pool = [pool[i] for i in np.argsort(dist_list)[:n]]
    return pool

def center_pool(pool, center_x, center_y):
    for i in pool:
        i[0] -= center_x
        i[1] -= center_y
        i[2] -= center_x
        i[3] -= center_y
    return pool

def get_sequence(
    df: pd.DataFrame,
) -> None:
    
    # params
    origin_idx = 19 
    n_map_max = 200
    n_obs_max = 56
    n_seq_max = 256
    future_idx = [29, 39, 49]

    # Seq data
    city_name = df["CITY_NAME"].values[0]

    # Get API for Argo Dataset map
    seq_lane_props = avm.city_lane_centerlines_dict[city_name]

    x_min = min(df["X"])
    x_max = max(df["X"])
    y_min = min(df["Y"])
    y_max = max(df["Y"])

    centerline_list = []
    
    # Get lane centerlines which lie within the range of trajectories
    for lane_id, lane_props in seq_lane_props.items():
        lane_cl = lane_props.centerline
        if (
            np.min(lane_cl[:, 0]) < x_max
            and np.min(lane_cl[:, 1]) < y_max
            and np.max(lane_cl[:, 0]) > x_min
            and np.max(lane_cl[:, 1]) > y_min
        ):
            centerline_list.append(lane_cl)

    # centerline
    frames = df.groupby("TRACK_ID")
    obs_list = []
    # Plot all the tracks up till current frame
    for group_name, group_data in frames:
        object_type = group_data["OBJECT_TYPE"].values[0]

        cor_x = group_data["X"].values
        cor_y = group_data["Y"].values
        dummy_time_stamp = group_data["TIMESTAMP"].values

        obs = Obs(cor_x, cor_y, object_type, dummy_time_stamp)
        obs_list.append(obs)

    # find agent
    agent = None
    av = None
    others = [] 
    for obs in obs_list:
        if obs.type == "AGENT":
            agent = obs
        elif obs.type == "AV":
            av = obs
            others.append(obs)
        else:
            others.append(obs)

    # check data validity
    if (len(av.cor_x) != 50) or (len(agent.cor_x) != 50 ):
        return
    
    origin_time = agent.timestamp[origin_idx]
    av_x = av.cor_x[19]
    av_y = av.cor_y[19]
    agent_x = agent.cor_x[19]
    agent_y = agent.cor_y[19]

    obs_pool = []
    for obs in others:
        vect = obs.encode(origin_time)
        for i in vect:
            obs_pool.append(i)

    # map pool
    map_pool = []
    for centerline in centerline_list:
        if len(centerline)<2:
            continue
        vect =  encode_centerline(centerline)
        if len(vect)>0:
            for i in vect:
                map_pool.append(i)
            
    agent_pool = agent.encode(origin_time)

    # center around av
    agent_pool = center_pool(agent_pool, av_x, av_y)
    obs_pool = center_pool(obs_pool, av_x, av_y)
    map_pool = center_pool(map_pool, av_x, av_y)


    # select around agent
    c_x = agent_x - av_x
    c_y = agent_y - av_y
    obs_pool = select_n_from_pool(obs_pool, n_obs_max, c_x, c_y)
    map_pool =select_n_from_pool(map_pool,  n_map_max, c_x, c_y)
    
    # combine
    x = agent_pool + obs_pool + map_pool

    # truncate
    if len(x)> n_seq_max:
        x = x[:n_seq_max] 
    
    # padding
    while len(x)<n_seq_max:
        pad = np.array([0.0] * 6)
        x = np.append(x, [pad], axis=0)
    x = np.asarray(x)
    
    # ==================== create y
    y = []
    for i in future_idx:
        y.append(agent.cor_x[i]-av_x)
        y.append(agent.cor_y[i]-av_y)
    y = np.asarray(y)
    return x,y
  
#===========================================================
##set root_dir to the correct path to your dataset folder

def prepare_data(src_dir, tar_dir):
    afl = ArgoverseForecastingLoader(src_dir)
    print('Total number of sequences:',len(afl))


    import os
    import glob
    extension = 'csv'
    os.chdir (src_dir)
    result = glob.glob ('*.{}'.format (extension))


    for fn in tqdm(result):
        filename = fn[:-4] #no .csv
        seq_path = f"{src_dir}/{filename}.csv"
        # seq_path = f"{src_dir}/4051.csv"
        x,y = get_sequence(afl.get(seq_path).seq_df)
        with open(f"{tar_dir}/{filename}.pkl", 'wb') as f:
            pkl.dump((x, y), f)
            # print(f"saved {filename}")
    return


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    src_dir = '/home/dalaska/data/forecast_train/train/data'
    tar_dir = '/home/dalaska/train_pkl_test_hahaha'
    prepare_data(src_dir, tar_dir)