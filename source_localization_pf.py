import lib_sim_setup as lss
import numpy as np
import matplotlib.pyplot as plt

resolution = 0.1 # [m/voxel]
H = 50
W = 50

n_sources = 10

r_s = 0.1 # gamma sensor radius

extent=[0, W*resolution, 0, H*resolution]

occ_map = lss.create_occupancy_map(H, W, n_obstacles=10, seed=0)

sources, sources_xyz, I_list = lss.add_radiation_sources(occ_map, n_sources, resolution)

vis_map = lss.vis_map(occ_map, sources)

# visualization
fig, ax = plt.subplots()
ax.set_title("occupancy map and riadioactive sources")
ax.imshow(np.transpose(vis_map, (1, 0, 2)), origin='lower', extent=extent)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
plt.show()


x, y, z = 2.5, 3, 1

sensor_xyz = [x, y, z]

count = lss.sim_count(sensor_xyz, sources_xyz, I_list, r_s)
exp_count = lss.exp_count(sensor_xyz, sources_xyz, I_list, r_s)

print(f"({x}, {y}) -> count rate = {count}")
print(f"({x}, {y}) -> expected count rate = {exp_count}")
