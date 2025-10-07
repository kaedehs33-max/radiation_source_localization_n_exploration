# this is a lib file for simulation setup

import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import poisson

# create occupancy map
def create_occupancy_map(H=50, W=50, n_obstacles=10, seed=0):
    np.random.seed(seed)
    occ = np.zeros((H, W), dtype=np.uint8)

    # boundaries occupied
    occ[0, :] = 1
    occ[-1, :] = 1
    occ[:, 0] = 1
    occ[:, -1] = 1

    # random rectangular obstacles
    for _ in range(n_obstacles):
        h, w = np.random.randint(3, 8, size=2)
        y = np.random.randint(1, H - h - 1)
        x = np.random.randint(1, W - w - 1)
        occ[y:y+h, x:x+w] = 1
    return occ

# radiation source placement
def find_surface_voxels(occ):
    H, W = occ.shape
    surface = np.zeros_like(occ, dtype=bool)
    for y in range(1, H):
        for x in range(1, W):
            if occ[y, x] == 1 and np.any(occ[y-1:y+2, x-1:x+2] == 0):
                surface[y, x] = True
    return np.argwhere(surface)

def add_radiation_sources(occ, n_sources, resolution):


    surf_coords = find_surface_voxels(occ)

    # place some sources randomly on surface voxels

    src_indices = np.random.choice(len(surf_coords), n_sources, replace=False)
    sources = surf_coords[src_indices]
    sources_z = np.random.uniform(0.5, 1.5, n_sources)  # random z height
    I_list = np.random.uniform(50, 100, size=n_sources) # intensity (bq)

    H, W = occ.shape

    sources_xy = sources * resolution
    sources_xyz = np.column_stack((sources_xy, sources_z))

    return sources, sources_xyz, I_list

# visualization image
def vis_map(occ, sources):

    H, W = occ.shape

    vis_map = np.ones((H, W, 3))
    vis_map[occ == 1] = [0, 0, 0]
    for x, y in sources:
        vis_map[x, y] = [1, 0, 0]  # red source

    return vis_map

# expected count
def exp_count(sensor_xyz, sources_xyz, I_list, r_s=0.1):
    x_s, y_s, z_s = sensor_xyz
    
    total_rate = 0.0
    I_c = 1 - 1 / np.sqrt(1 + r_s**2)
    for j, ((x_src, y_src, z_src), I_j) in enumerate(zip(sources_xyz, I_list)):
        d_xy = np.linalg.norm([x_s - x_src, y_s - y_src])
        d = np.sqrt(d_xy**2 + (z_s - z_src)**2)
        sensed_ratio = 1 - d / np.sqrt(d**2 + r_s**2)
        total_rate += I_j / I_c * sensed_ratio
    return total_rate

# count sampled from poisson distribution
def sim_count(sensor_xyz, sources_xyz, I_list, r_s=0.1):
    total_rate = exp_count(sensor_xyz, sources_xyz, I_list, r_s)
    return np.random.poisson(total_rate)