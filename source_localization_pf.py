import lib_sim_setup as lss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import poisson

resolution = 0.1 # [m/voxel]
H = 50
W = 50
h, w = H * resolution, W * resolution

n_sources = 4

r_s = 0.1 # gamma sensor radius

extent=[0, W*resolution, 0, H*resolution]

occ_map = lss.create_occupancy_map(H, W, n_obstacles=10, seed=0)

sources, sources_xyz, I_list = lss.add_radiation_sources(occ_map, n_sources, resolution)

vis_map = lss.vis_map_create(occ_map, sources)

# visualization
fig, ax = plt.subplots()
ax.set_title("occupancy map and riadioactive sources")
ax.imshow(np.transpose(vis_map, (1, 0, 2)), origin='lower', extent=extent)
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
plt.show()


# initialize particles

N = 100
N_eff_thresh = 0.5 * N  # threshold for resampling

particles = lss.initialize_particles(
    N_particles=N,
    r_max=5,
    occ=occ_map,
    resolution=0.1,
    seed=0
)

weights = np.ones(N) / N
log_weights = np.log(weights)

fig, ax = plt.subplots()
lss.visualize_particles(ax, occ_map, sources, particles, weights, resolution=0.1)

plt.show()

sensor_z = 1.0    # sensor height (m)
num_iters = 20

fig, (ax_particles, ax_estimate) = plt.subplots(1, 2, figsize=(10, 5))

# ----- Left: Particles -----
ax_particles.set_title("Particles and Measurement")
ax_particles.imshow(np.transpose(vis_map, (1, 0, 2)), origin='lower', extent=extent)
ax_particles.set_xlabel("X [m]")
ax_particles.set_ylabel("Y [m]")

# ----- Right: Ground truth vs Estimated -----
ax_estimate.set_title("Estimation vs Truth")
ax_estimate.imshow(np.transpose(vis_map, (1, 0, 2)), origin='lower', extent=extent)
ax_estimate.set_xlabel("X [m]")
ax_estimate.set_ylabel("Y [m]")


for it in range(num_iters):
    # 1) sample measurement point (in meters)
    mx, my = lss.sample_free_point_with_clearance(occ_map, resolution, r_vox=1, max_tries=10000)

    # 2) simulate measurement from true sources
    # use exp_count then Poisson sample (keeps sim deterministic/non-deterministic choice explicit)
    meas_k = lss.sim_count((mx, my, sensor_z), sources_xyz, I_list, r_s=r_s)

    # 3) update particles' log_weights by adding log-likelihood
    for i, p in enumerate(particles):
        s_xyz, I_p = lss.extract_particle_sources_and_I(p)
        lam_pred = lss.exp_count((mx, my, sensor_z), s_xyz, I_p, r_s=r_s)
        # add particle's background if present
        if "lambda_b" in p:
            lam_pred = lam_pred + float(p["lambda_b"])
        lam_pred = max(lam_pred, 1e-12)   # avoid mu==0 for numerical stability
        log_lik = poisson.logpmf(meas_k, mu=lam_pred)
        # accumulate log-likelihood
        log_weights[i] += log_lik

    # 4) compute normalized linear weights for visualization and diagnostics
    weights = lss.normalize_log_weights(log_weights)

    N_eff = 1.0 / np.sum(weights**2) # effective sample size

    if N_eff < N_eff_thresh:
        # perform systematic resampling
        particles = lss.systematic_resample_particles(particles, weights)

        # reset weights to uniform after resampling
        weights = np.ones(N) / N
        log_weights = np.log(weights)   # so you can resume accumulating log-likelihoods

        # apply simple Gaussian perturbation (tune sigma values to taste)
        particles = lss.perturb_particles_simple(particles, occ_map, resolution,
                                                sigma_xy=0.3,
                                                sigma_lambda=15.0,
                                                sigma_lambda_b=0.6)

        print(f"Resampled at iter {it}, N_eff={N_eff:.1f}")

    r_est, sources_est, lambdas_est = lss.estimate_state_from_particles(particles, weights)
    print(f"Iteration {it}: r_est = {r_est}")
    print(f"Sources_est = \n{sources_est}")
    print(f"Lambdas_est = {lambdas_est}")



    ax_particles.collections.clear()
    ax_particles.patches.clear()
    ax_estimate.collections.clear()
    ax_estimate.patches.clear()

    # ----- Update left subplot -----
    lss.visualize_particles(ax_particles, occ_map, sources, particles, weights, resolution=resolution)
   
     # ----- Update right subplot ----- 
    if r_est > 0:
        # use distinct colors per estimated source
        cmap = plt.cm.tab10  # or 'Set1', 'tab20', etc.
        for j, (x_est, y_est) in enumerate(sources_est):
            color = cmap(j % cmap.N)
            ax_estimate.scatter(
                x_est, y_est,
                facecolors='none', edgecolors=color,
                s=200, linewidths=2,
                label=f'Estimated source {j+1}'
            )
    # ax_estimate.legend(loc='upper right')
    plt.pause(0.01)


plt.ioff()
plt.show()





