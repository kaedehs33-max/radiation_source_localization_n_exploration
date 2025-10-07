import lib_sim_setup as lss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

resolution = 0.1 # [m/voxel]
H = 50
W = 50
h, w = H * resolution, W * resolution

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


# initialize particles

N = 100

particles = lss.initialize_particles(
    N_particles=N,
    r_max=5,
    occ=occ_map,
    resolution=0.1,
    seed=0
)

weights = np.ones(N) / N

fig, ax = plt.subplots()
ax.set_title("particle filter")
ax.imshow(np.transpose(vis_map, (1, 0, 2)), origin='lower', extent=extent)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")

for p in particles:
    r_est = p['r']
    for i in range(r_est):
        xi, yi = p['sources_xy'][i]
        ax.scatter(xi, yi, c='cyan', s=5, alpha=0.3)

plt.show()


