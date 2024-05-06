from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from typing import List
import numpy as np
import pandas as pd
from dataclasses import dataclass
import math
import pickle as pkl
from tqdm import tqdm

origin_idx = 19              # index of the frame to split past and future
future_idx = [29, 39, 49]    # index of the future steps
new_t = [0.0, -0.5, -1.0, -1.5, -2.0] # time of the past steps
n_map_max = 180   # maxium length of map sequence 
n_obs_max = 60    # maxium length of 
target_shape = (244, 5)

print(f"origin_idx: {origin_idx}, future_idx: {future_idx}, new_t: {new_t}, n_map_max: {n_map_max}, n_obs_max: {n_obs_max}")


avm = ArgoverseMap()
encode_type = {"AGENT": 5.0, "AV": 2.0, "OTHERS": 2.0, "MAP": 1.0} 


def linear_interpolate(t, x, new_t):
    """
    Performs linear interpolation or extrapolation for a new set of x values.

    :param x: The array of x values (must be monotonically increasing or decreasing).
    :param t: The array of t values corresponding to each x value.
    :param new_x: The new x values to interpolate/extrapolate t values for.
    :return: Interpolated/extrapolated values of t for each new_x.
    """
    if len(t) != len(x):
        raise ValueError("x and t must have the same length.")
    
    new_x = []
    for xi in new_t:
        if xi <= t[0]:
            # Extrapolate to the left
            slope = (x[1] - x[0]) / (t[1] - t[0]) if abs(t[1] - t[0])>1e-6 else 0.0
            
            ti = x[0] + slope * (xi - t[0])
        elif xi >= t[-1]:
            # Extrapolate to the right
            slope = (x[-1] - x[-2]) / (t[-1] - t[-2]) if abs(t[-1] - t[-2])>1e-6 else 0.0
            ti = x[-1] + slope * (xi - t[-1])
        else:
            # Interpolate
            for i in range(len(t) - 1):
                if t[i] <= xi <= t[i+1] or t[i] >= xi >= t[i+1]:
                    slope = (x[i+1] - x[i]) / (t[i+1] - t[i]) if abs(t[i+1] - t[i])>1e-6 else 0.0
                    ti = x[i] + slope * (xi - t[i])
                    break
        new_x.append(ti)
    
    return new_x



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

        # center time stamp relative to origin time
        self.timestamp = [ i - origin_time for i in self.timestamp]

        # select data before origin time
        t = []
        x = []
        y = [] 
        for i in range(len(self.timestamp)):
            if self.timestamp[i] < 1e-3:
                t.append(self.timestamp[i])
                x.append(self.cor_x[i])
                y.append(self.cor_y[i])

        if len(t)<2:
            return []
        
        # interpolate
        new_x = linear_interpolate(t, x, new_t)
        new_y = linear_interpolate(t, y,  new_t)


        vects = []
        for j in range(len(new_t)-1):
            vec = [new_x[j+1], new_y[j+1], new_x[j], new_y[j],  encode_type[self.type]]
            vects.append(vec)
        return vects

def encode_centerline(centerline):
    dist_threshod = 6.0
    tar = []
    last_pt = centerline[0]
    for pt in centerline:
        dist = get_dist(pt, last_pt)
        if dist > dist_threshod:
            tar.append([last_pt[0], last_pt[1], pt[0], pt[1], encode_type['MAP']])
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
    av_x = av.cor_x[origin_idx]
    av_y = av.cor_y[origin_idx]


    # agent pool
    agent_pool = agent.encode(origin_time)

    # # obs pool
    obs_pool = []
    # add agent to obs pool
    for i in agent_pool:
        i_copy = i[:]
        i_copy[4] = encode_type["OTHERS"] 
        obs_pool.append(i_copy)
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
            
    # center around av
    agent_pool = center_pool(agent_pool, av_x, av_y)
    obs_pool = center_pool(obs_pool, av_x, av_y)
    map_pool = center_pool(map_pool, av_x, av_y)


    # select around agent
    pad = [0.0, 0.0, 0.0, 0.0, -1.0]
    if len(obs_pool) > n_obs_max:
        obs_pool = obs_pool[:n_obs_max]
    while len(obs_pool)<n_obs_max:
        obs_pool.append(pad)

    if len(map_pool) > n_map_max:
        map_pool = map_pool[:n_map_max]
    while len(map_pool)<n_map_max:
        map_pool.append(pad)
    
    
    # combine
    if (n_obs_max):
        x = agent_pool + obs_pool + map_pool
    else:
        x = agent_pool + map_pool
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
        x,y = get_sequence(afl.get(seq_path).seq_df)
        if x.shape == target_shape:
            with open(f"{tar_dir}/{filename}.pkl", 'wb') as f:
                pkl.dump((x, y), f)
    return


# -----------------------------------------------------------------------------
if __name__ == "__main__":

    import os
    current_directory = os.getcwd()
    # # depending on the stage call the appropriate function
    src_dir = os.path.join(current_directory, 'sample/data/csv')
    tar_dir = os.path.join(current_directory, 'sample/data/pkl')
    prepare_data(src_dir, tar_dir)
