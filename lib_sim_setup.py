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
    for y in range(1, H-1):
        for x in range(1, W-1):
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

    I_list = np.random.uniform(90, 100, size=n_sources) # intensity (sensed count at 1m)

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
                         lambda_shape=50, lambda_scale=2.0,
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
            # sample point
            for _ in range(1000):
                x_pix = np.random.uniform(0, W)
                y_pix = np.random.uniform(0, H)
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

def systematic_resample_particles(particles, weights):
    """
    Systematic resampling.
    - particles: list of particle dicts
    - weights: 1D numpy array, normalized (sum == 1)

    Returns:
      new_particles: list (len == len(particles)) of deep-ish copies of selected particles.
    """
    import numpy as np

    N = len(particles)
    if N == 0:
        return []

    # cumulative distribution
    cumulative = np.cumsum(weights)
    # positions: systematic samples in (0,1]
    positions = (np.arange(N) + np.random.rand()) / N
    # find corresponding indices
    indexes = np.searchsorted(cumulative, positions, side='right')

    new_particles = []
    for idx in indexes:
        p = particles[int(idx)]
        # shallow-deep copy of particle contents (safe for arrays)
        newp = {
            'r': int(p.get('r', 0)),
            'lambda_b': float(p.get('lambda_b', 0.0)),
            'sources_xy': np.array(p.get('sources_xy', np.zeros((0,2))), copy=True),
            'lambdas': np.array(p.get('lambdas', np.zeros((0,))), copy=True)
        }
        new_particles.append(newp)

    return new_particles

def perturb_particles_simple(particles, occ, resolution,
                             sigma_xy=0.2,        # meters (stddev for x,y)
                             sigma_lambda=10.0,   # intensity units (stddev for lambda)
                             sigma_lambda_b=0.5,  # background rate stddev
                             clip_positions=True):
    """
    Add simple Gaussian perturbation to each particle's parameters (except r).
    - particles: list of particle dicts (each has 'r','lambda_b','sources_xy','lambdas')
    - occ: occupancy map (used to clip to map bounds if clip_positions True)
    - resolution: meters per voxel (used to compute map bounds)
    - sigma_*: standard deviations for Gaussian noise for each variable
    Returns: particles (modified in-place and also returned)
    """
    import numpy as np

    H, W = occ.shape
    x_min, x_max = 0.0, W * resolution
    y_min, y_max = 0.0, H * resolution

    for p in particles:
        # perturb background
        if "lambda_b" in p:
            p["lambda_b"] = float(p.get("lambda_b", 0.0) + sigma_lambda_b * np.random.randn())
            if p["lambda_b"] < 0:
                p["lambda_b"] = 0.0

        # perturb sources positions and intensities
        s_xy = p.get("sources_xy", None)
        lams = p.get("lambdas", None)

        if s_xy is None or lams is None:
            # nothing to perturb
            continue

        # If arrays are empty, skip
        if s_xy.size == 0:
            p["sources_xy"] = np.zeros((0, 2))
            p["lambdas"] = np.zeros((0,))
            p["r"] = 0
            continue

        # add noise to x,y (shape (r,2))
        noise_xy = sigma_xy * np.random.randn(*s_xy.shape)
        s_xy = s_xy.astype(float) + noise_xy

        # add noise to lambdas
        noise_l = sigma_lambda * np.random.randn(lams.shape[0])
        lams = lams.astype(float) + noise_l

        # clip positions to map bounds (if requested)
        if clip_positions:
            s_xy[:, 0] = np.clip(s_xy[:, 0], x_min, x_max)
            s_xy[:, 1] = np.clip(s_xy[:, 1], y_min, y_max)

        # remove sources with non-positive intensity
        keep_mask = lams > 0.0
        if np.any(~keep_mask):
            if np.sum(keep_mask) == 0:
                # all sources died
                p["sources_xy"] = np.zeros((0, 2))
                p["lambdas"] = np.zeros((0,))
                p["r"] = 0
                continue
            else:
                p["sources_xy"] = s_xy[keep_mask]
                p["lambdas"] = lams[keep_mask]
                p["r"] = int(p["sources_xy"].shape[0])
        else:
            p["sources_xy"] = s_xy
            p["lambdas"] = lams
            p["r"] = int(p["sources_xy"].shape[0])

    return particles

def estimate_state_from_particles(particles, weights):
    """
    Estimate number of sources (r_est) and average source positions/intensities.

    Parameters
    ----------
    particles : list of dict
        Each particle has keys 'r', 'sources_xy', 'lambdas'
    weights : np.ndarray
        Normalized particle weights, shape (N,)

    Returns
    -------
    r_est : int
        Estimated number of sources
    sources_est : np.ndarray, shape (r_est, 2)
        Weighted mean positions of estimated sources
    lambdas_est : np.ndarray, shape (r_est,)
        Weighted mean intensities of estimated sources
    """

    import numpy as np

    # estimated number of sources (rounded expectation)
    r_exp = np.sum([w * p["r"] for w, p in zip(weights, particles)])
    r_est = int(np.floor(r_exp + 0.5))

    # select particles with that r
    mask = np.array([p["r"] == r_est for p in particles])
    if not np.any(mask):
        # fallback: choose the most frequent r
        unique, counts = np.unique([p["r"] for p in particles], return_counts=True)
        r_est = unique[np.argmax(counts)]
        mask = np.array([p["r"] == r_est for p in particles])

    sub_particles = [p for p, m in zip(particles, mask) if m]
    sub_weights = weights[mask]
    sub_weights /= np.sum(sub_weights)

    # average over source positions and intensities
    if r_est == 0:
        return 0, np.zeros((0, 2)), np.zeros((0,))

    # Align all source lists by index (1st source, 2nd source, ...)
    # If some particle has fewer sources (shouldn’t happen here, all have r_est)
    # we pad with NaNs and ignore them.
    all_xy = np.full((len(sub_particles), r_est, 2), np.nan)
    all_l = np.full((len(sub_particles), r_est), np.nan)

    for i, p in enumerate(sub_particles):
        sxy = np.asarray(p["sources_xy"])
        lam = np.asarray(p["lambdas"])
        n = min(r_est, sxy.shape[0])
        all_xy[i, :n, :] = sxy[:n, :]
        all_l[i, :n] = lam[:n]

    # weighted mean ignoring NaNs
    sources_est = np.nansum(all_xy * sub_weights[:, None, None], axis=0)
    lambdas_est = np.nansum(all_l * sub_weights[:, None], axis=0)

    return r_est, sources_est, lambdas_est

def _random_point_in_world(occ, resolution):
    """Sample a random point uniformly within map bounds (no free-space restriction)."""
    import numpy as np
    H, W = occ.shape
    x_pix = np.random.rand() * (W - 1)
    y_pix = np.random.rand() * (H - 1)
    x = x_pix * resolution
    y = y_pix * resolution
    return x, y

def death_move_random(particles, weights, p_death=0.2):
    """
    For each particle, with prob p_death remove a random source (if any).
    Modifies particles in-place.
    """
    import numpy as np
    for i, p in enumerate(particles):
        wi = weights[i]
        if p.get('r', 0) > 0 and np.random.rand() < p_death * (1 - wi):
            r = p['r']
            idx = np.random.randint(0, r)
            mask = np.ones(r, dtype=bool)
            mask[idx] = False
            p['sources_xy'] = p['sources_xy'][mask]
            p['lambdas'] = p['lambdas'][mask]
            p['r'] = p['sources_xy'].shape[0]
    return particles

def birth_move_from_particles(particles, weights, occ, resolution, p_birth=0.3,
                              sigma_birth_xy=1.0, lambda_shape=2.0, lambda_scale=20.0):
    """
    For each particle (loop), with prob p_birth propose a birth:
      - choose a particle index j with prob proportional to weights
      - pick one of particle j's sources (or its centroid) as a center
      - sample new source position ~ N(center, sigma_birth_xy)
      - sample lambda from Gamma
    This uses the global particle set (weights) to guide proposals.
    """
    import numpy as np
    N = len(particles)
    if N == 0:
        return particles
    # sample indices for proposals (pre-sample to avoid bias from sequential updates)
    for i, p in enumerate(particles):
        wi = weights[i]
        if np.random.rand() < p_birth * (1 - wi):
            # pick a guiding particle index j
            j = np.random.choice(np.arange(N), p=weights)
            guide = particles[j]
            # choose a center: if guide has sources pick one randomly else pick random free location
            if guide.get('r', 0) > 0:
                src_idx = np.random.randint(0, guide['r'])
                cx, cy = guide['sources_xy'][src_idx]
            else:
                # fallback: random free point
                cx, cy = _random_point_in_world(occ, resolution)
            # sample a perturbed location around center
            x_new = cx + sigma_birth_xy * np.random.randn()
            y_new = cy + sigma_birth_xy * np.random.randn()
            # clip to map bounds
            Hpix, Wpix = occ.shape
            x_new = np.clip(x_new, 0.0, (Wpix-1)*resolution)
            y_new = np.clip(y_new, 0.0, (Hpix-1)*resolution)
            lam = np.random.gamma(lambda_shape, lambda_scale)
            # append to particle p
            if p.get('r', 0) == 0:
                p['sources_xy'] = np.array([[x_new, y_new]])
                p['lambdas'] = np.array([lam])
            else:
                p['sources_xy'] = np.vstack([p['sources_xy'], np.array([x_new,y_new])])
                p['lambdas'] = np.hstack([p['lambdas'], lam])
            p['r'] = p['sources_xy'].shape[0]
    return particles