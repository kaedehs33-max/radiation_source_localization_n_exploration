# in this demo, we simulate the gamma count in arbitray position on the map
# count is sampled from poisson distribution
# attenuation is ignored, i.e., obstacles are transparent to gamma rays
# %% import
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# %% create occupancy map 
# index is (x, y), 
# # while during visualization, 
# # map_world(x, y) = occ_map(x, y) = occ_vis(y, x)
# # and vis the occ_vis as occ_map', "lower"

resolution = 0.1  # [m/pixel]


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

occ_map = create_occupancy_map()

# %% radiation source placement
def find_surface_voxels(occ):
    H, W = occ.shape
    surface = np.zeros_like(occ, dtype=bool)
    for y in range(1, H):
        for x in range(1, W):
            if occ[y, x] == 1 and np.any(occ[y-1:y+2, x-1:x+2] == 0):
                surface[y, x] = True
    return np.argwhere(surface)

surf_coords = find_surface_voxels(occ_map)

# place some sources randomly on surface voxels
n_sources = 20
src_indices = np.random.choice(len(surf_coords), n_sources, replace=False)
sources = surf_coords[src_indices]
sources_z = np.random.uniform(0.5, 1.5, n_sources)  # random z height
I_list = np.random.uniform(50, 100, size=n_sources) # intensity (bq)

H, W = occ_map.shape
extent=[0, W*resolution, 0, H*resolution]

vis_map = np.ones((H, W, 3))
vis_map[occ_map == 1] = [0, 0, 0]
for x, y in sources:
    vis_map[x, y] = [1, 0, 0]  # red source

sources_xy = sources * resolution


# %% radiation sensing model (poisson thinning)
sensor_z = 1.0 # sensor hight [m]

# expected count
def exp_count(sensor_xyz, sources_xy, sources_z, I_list, r_s=0.1):
    x_s, y_s, z_s = sensor_xyz
    total_rate = 0.0
    I_c = 1 - 1 / np.sqrt(1 + r_s**2)
    for j, ((x_src, y_src), z_src, I_j) in enumerate(zip(sources_xy, sources_z, I_list)):
        d_xy = np.linalg.norm([x_s - x_src, y_s - y_src])
        d = np.sqrt(d_xy**2 + (z_s - z_src)**2)
        sensed_ratio = 1 - d / np.sqrt(d**2 + r_s**2)
        total_rate += I_j / I_c * sensed_ratio
    return total_rate

# count sampled from poisson distribution
def sim_count(sensor_xyz, sources_xy, sources_z, I_list, r_s=0.1):
    total_rate = exp_count(sensor_xyz, sources_xy, sources_z, I_list, r_s)
    return np.random.poisson(total_rate)

# %% Interactive UI

fig, ax = plt.subplots()
ax.imshow(np.transpose(vis_map, (1, 0, 2)), origin='lower', extent=extent)
ax.set_title("Click to sample count rate")

def onclick(event):
    if event.xdata is None or event.ydata is None:
        return
    x, y = event.xdata * resolution, event.ydata * resolution
    count = sim_count((x, y, sensor_z), sources_xy, sources_z, I_list, r_s=0.1)
    ax.plot(event.xdata, event.ydata, 'bo')
    ax.text(event.xdata+0.1, event.ydata, f"{count}", color='blue', fontsize=8)
    plt.draw()
    print(f"({x}, {y}) -> count rate = {count}")

cid = fig.canvas.mpl_connect('button_press_event', onclick)

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")

plt.show()

# %% count rate map
from matplotlib.colors import LogNorm, PowerNorm

exp_count_rate_map = np.zeros((H, W))

for x in range(H):
    for y in range(W):
        if occ_map[x, y] == 0:  # free space
            exp_count_rate_map[x, y] = exp_count((x * resolution, y * resolution, sensor_z), sources_xy, sources_z, I_list)


fig, ax = plt.subplots()
ax.set_title("Expected count rate map (overlayed on map)")

# base layer: occupancy + source map
ax.imshow(np.transpose(vis_map, (1, 0, 2)), origin='lower', extent=extent)

norm = PowerNorm(gamma=0.5, vmin=np.min(exp_count_rate_map[exp_count_rate_map>0]), vmax=np.max(exp_count_rate_map))
# overlay: expected radiation intensity (semi-transparent)
im = ax.imshow(exp_count_rate_map.T, cmap='plasma', alpha=0.6, norm=norm, origin='lower', extent=extent)

# colorbar for radiation intensity
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=10)

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")

plt.show(block=False)


# %% animation of count rate map
import matplotlib.animation as animation

fig, ax = plt.subplots()
ax.set_title("Simulated count fluctuations over time")
ax.imshow(np.transpose(vis_map, (1, 0, 2)), origin='lower', extent=extent)

# initial Poisson-sampled frame
measured_map = np.random.poisson(exp_count_rate_map)
norm = PowerNorm(gamma=0.5, vmin=np.min(exp_count_rate_map[exp_count_rate_map>0]), vmax=np.max(exp_count_rate_map))
im = ax.imshow(measured_map.T, cmap='plasma', alpha=0.6, norm=norm, origin='lower', extent=extent)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Count rate", fontsize=10)

# update function
def update(frame):
    # random Poisson sampling around the expected count
    measured_map = np.random.poisson(exp_count_rate_map)
    im.set_data(measured_map.T)
    ax.set_title(f"Simulated count fluctuations (frame {frame})")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=50, interval=300, blit=False)

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")

plt.show()