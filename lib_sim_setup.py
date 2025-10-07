# this is a lib file for simulation setup

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

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
    # sources_z = np.random.uniform(0.5, 1.5, n_sources)  # random z height
    sources_z = np.full(n_sources, 1.0)

    I_list = np.random.uniform(50, 100, size=n_sources) # intensity (bq)

    H, W = occ.shape

    sources_xy = sources * resolution
    sources_xyz = np.column_stack((sources_xy, sources_z))

    return sources, sources_xyz, I_list

# visualization image
def vis_map_create(occ, sources):

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


def sample_free_point_with_clearance(occ, resolution=1.0, r_vox=1, max_tries=10000):
    H, W = occ.shape
    for _ in range(max_tries):
        x_pix = np.random.rand() * (W - 1)
        y_pix = np.random.rand() * (H - 1)
        ix = int(np.clip(int(round(x_pix)), 0, W - 1))
        iy = int(np.clip(int(round(y_pix)), 0, H - 1))

        # sample neighborhood indices (clip to map)
        x0 = max(0, ix - r_vox); x1 = min(W - 1, ix + r_vox)
        y0 = max(0, iy - r_vox); y1 = min(H - 1, iy + r_vox)

        neighborhood = occ[y0:y1+1, x0:x1+1]
        if np.all(neighborhood == 0):   # all free in clearance window
            return x_pix * resolution, y_pix * resolution

    raise RuntimeError("sample_free_point_with_clearance: couldn't find free point")

def initialize_particles(N_particles, r_max, occ, resolution,
                         lambda_b_range=(1, 10),
                         lambda_shape=2.0, lambda_scale=20.0,
                         seed=None):
    """
    Initialize N_particles particles for multi-source estimation.
    Each particle: {r, lambda_b, [(x_i, y_i, lambda_i)...]}
    """
    if seed is not None:
        np.random.seed(seed)
        
    H, W = occ.shape
    particles = []

    # Precompute decreasing probability for r
    r_probs = np.linspace(1.0, 0.1, r_max)
    r_probs /= np.sum(r_probs)

    for _ in range(N_particles):
        # sample number of sources
        r = np.random.choice(np.arange(1, r_max+1), p=r_probs)

        # background rate
        lambda_b = np.random.uniform(*lambda_b_range)

        # sources (positions and intensities)
        sources_xy = []
        lambdas = []
        for _ in range(r):
            # sample valid free point
            for _ in range(1000):
                x_pix = np.random.uniform(0, W)
                y_pix = np.random.uniform(0, H)
                if occ[int(y_pix), int(x_pix)] == 0:
                    break
            x = x_pix * resolution
            y = y_pix * resolution
            lam = np.random.gamma(lambda_shape, lambda_scale)
            sources_xy.append((x, y))
            lambdas.append(lam)

        particles.append({
            'r': r,
            'lambda_b': lambda_b,
            'sources_xy': np.array(sources_xy),
            'lambdas': np.array(lambdas)
        })

    return particles # its a dictionary

def visualize_particles(ax, occ, sources, particles, weights, resolution=0.1):
    H, W = occ.shape
    extent = [0, W * resolution, 0, H * resolution]

    # draw occupancy map and true sources
    vis_map = vis_map_create(occ, sources)
    
    ax.imshow(np.transpose(vis_map, (1, 0, 2)), origin='lower', extent=extent)

    # normalize weights for visualization
    norm_w = (weights - np.min(weights)) / (np.max(weights) - np.min(weights) + 1e-12)

    # draw particle sources with color mapped by weight
    for p, w in zip(particles, norm_w):
        r_est = p['r']
        if r_est == 0:
            continue
        for i in range(r_est):
            x, y = p['sources_xy'][i]
            # color map (you can change cmap name)
            color = plt.cm.viridis(w)  
            ax.scatter(x, y, color=color, s=15, alpha=0.6)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Particle hypotheses (color = weight)")

def normalize_log_weights(log_w):
    # stable softmax: subtract max, exponentiate, normalize
    a = np.max(log_w)
    w = np.exp(log_w - a)
    w /= np.sum(w)
    return w

def extract_particle_sources_and_I(p, default_z=1.0, default_I=1.0):
    """
    Return (sources_xyz, I_list) where:
      - sources_xyz is shape (r_actual, 3)
      - I_list is shape (r_actual,)

    r_actual = min(p['r'], number of sources actually present in arrays)

    Accepts particles with keys:
      - 'sources_xyz' (shape (n,3) or (3,) or (n*3,) )
      - or 'sources_xy' (shape (n,2) or (2,))
    And intensity keys:
      - 'I_list' or 'lambdas' (shape (n,) or (1,))
    If intensities are missing for available sources, they are filled with `default_I`.
    """
    r = int(p.get("r", 0))

    # --- sources -> produce s as shape (n_s, 3) ---
    if "sources_xyz" in p:
        s = np.asarray(p["sources_xyz"], dtype=float)
        # normalize shapes: allow (3,) -> (1,3)
        if s.ndim == 1 and s.size == 3:
            s = s.reshape(1, 3)
        elif s.ndim == 1 and s.size % 3 == 0:
            s = s.reshape(-1, 3)
        # if s.ndim == 2 and s.shape[1] != 3 try to reshape if possible
        elif s.ndim == 2 and s.shape[1] != 3:
            try:
                s = s.reshape(-1, 3)
            except Exception:
                # fallback: take first two columns and append z
                s = np.column_stack((s[:, 0], s[:, 1], np.ones(s.shape[0]) * default_z))
    elif "sources_xy" in p:
        s_xy = np.asarray(p["sources_xy"], dtype=float)
        if s_xy.size == 0:
            s = np.zeros((0, 3), dtype=float)
        elif s_xy.ndim == 1 and s_xy.size == 2:
            s = np.array([[s_xy[0], s_xy[1], default_z]], dtype=float)
        elif s_xy.ndim == 2 and s_xy.shape[1] == 2:
            s = np.column_stack((s_xy, np.ones(s_xy.shape[0]) * default_z))
        else:
            # try to reshape (n,2)
            s_xy = s_xy.reshape(-1, 2)
            s = np.column_stack((s_xy, np.ones(s_xy.shape[0]) * default_z))
    else:
        s = np.zeros((0, 3), dtype=float)

    # --- intensities ---
    if "I_list" in p:
        I = np.asarray(p["I_list"], dtype=float).reshape(-1)
    elif "lambdas" in p:
        I = np.asarray(p["lambdas"], dtype=float).reshape(-1)
    else:
        I = np.zeros((0,), dtype=float)

    # --- determine how many sources we can actually return ---
    n_s = s.shape[0]
    n_I = I.shape[0]
    if n_I == 0:
        # we only can rely on source positions; we'll fill intensities with default_I
        n_take = min(r, n_s)
    else:
        n_take = min(r, n_s, n_I)

    if n_take == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)

    s_out = s[:n_take].astype(float)
    if n_I == 0:
        I_out = np.ones(n_take, dtype=float) * default_I
    else:
        I_out = I[:n_take].astype(float)

    return s_out, I_out
