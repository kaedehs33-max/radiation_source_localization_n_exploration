import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from matplotlib.colors import LinearSegmentedColormap
from poisson_disc import Bridson_sampling


H=50
W=50
n_obstacles=3
resolution = 0.1
x_min, x_max = 0, W * resolution
y_min, y_max = 0, H * resolution
extent=[x_min, x_max, y_min, y_max]

seed=0
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

# place some sources randomly
n_sources = 3
d_min = 1.0
sources_pxy = np.random.rand(n_sources, 2) * (W-1, H-1)
sources_xy = sources_pxy * resolution
sources_z = np.ones([n_sources, 1])
sources_xyz = np.column_stack((sources_xy, sources_z))

I_list = np.random.uniform(90, 100, size=n_sources) # intensity (sensed count at 1m)


pink_red = LinearSegmentedColormap.from_list("pink_red", ["#ffc0cb", "#ff0000"])  # light pink → red

plt.imshow(occ.T, cmap='gray_r', interpolation='nearest', extent=extent)
sc = plt.scatter(
    sources_xy[:, 0], sources_xy[:, 1],
    c=I_list.flatten(),         # color by intensity
    cmap=pink_red,                 # choose color map
    s=50,                       # marker size
    edgecolors='black'
)

plt.show()

N_particles = 100

r_max = 5
lambda_b_range=(1, 10)
lambda_shape=50
lambda_scale=2.0

particles = []

# # Precompute decreasing probability for r
# r_probs = np.linspace(1.0, 0.1, r_max)
# r_probs /= np.sum(r_probs)
# for _ in range(N_particles):
#     # sample number of sources
#     r_hypo = np.random.choice(np.arange(1, r_max+1), p=r_probs)

#     # background rate
#     lambda_b_hypo = np.random.uniform(*lambda_b_range)

#     # sources (positions and intensities)
#     sources_xy_hypo = []
#     lambdas_hypo = []
#     for _ in range(r_hypo):
#         # sample point
#         for _ in range(1000):
#             x_pix = np.random.uniform(0, W)
#             y_pix = np.random.uniform(0, H)
#         x = x_pix * resolution
#         y = y_pix * resolution
#         lam = np.random.gamma(lambda_shape, lambda_scale)
#         sources_xy.append((x, y))
#         lambdas_hypo.append(lam)

#     particles.append({
#         'r': r,
#         'lambda_b': lambda_b,
#         'sources_xy': np.array(sources_xy),
#         'lambdas': np.array(lambdas)
#     })
