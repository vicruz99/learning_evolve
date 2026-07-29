# sol_000066 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000043 (state e63f418f) state=0f7fe832 sum of radii=2.070845 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_sum_radii(centers):
    """
    Computes the maximum valid radii for a given set of centers and returns their sum.
    Radii are constrained by distance to boundaries and half the distance to the nearest neighbor.
    """
    c = centers
    # Distance to boundaries: min(x, 1-x, y, 1-y)
    d_bound = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                         np.minimum(c[:, 1], 1.0 - c[:, 1]))
    
    # Pairwise Euclidean distances
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Distance to nearest neighbor / 2
    d_nn = 0.5 * np.min(dists, axis=1)
    
    # Final radii are the limiting factor
    radii = np.minimum(d_bound, d_nn)
    return np.sum(radii), radii

def objective(centers_flat):
    """Objective function for minimizers: minimize negative sum of radii."""
    centers = centers_flat.reshape(N, 2)
    s, _ = compute_sum_radii(centers)
    return -s

def simulate_packing():
    """
    Runs a force-directed simulation to rapidly pack circles into a high-density configuration.
    """
    np.random.seed(42)
    
    # Hexagonal lattice initialization
    cx, cy = [], []
    r0 = 0.08
    y = r0
    row = 0
    while len(cx) < N:
        x = r0 + (row % 2) * r0
        while x <= 1 - r0 and len(cx) < N:
            cx.append(x)
            cy.append(y)
            x += 2 * r0
        y += np.sqrt(3) * r0
        row += 1
        
    centers = np.column_stack([cx, cy])
    centers += np.random.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    radii = np.full(N, 0.03)
    
    forces = np.zeros_like(centers)
    
    for step in range(20000):
        # Compute pairwise overlaps
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(dists, 1.0)
        min_req = radii[:, None] + radii[None, :]
        overlap = np.maximum(0, min_req - dists)
        
        # Repulsive forces proportional to overlap
        f_dir = diff / (dists[:, :, None] + 1e-12)
        f_rep = np.sum(overlap[:, :, None] * f_dir * 50.0, axis=1)
        
        # Boundary repulsion forces
        f_bnd = np.zeros_like(centers)
        r_flat = radii
        mask_l = centers[:, 0] < r_flat
        f_bnd[mask_l, 0] += (r_flat[mask_l] - centers[mask_l, 0]) * 100.0
        mask_r = centers[:, 0] > 1.0 - r_flat
        f_bnd[mask_r, 0] -= (centers[mask_r, 0] - (1.0 - r_flat[mask_r])) * 100.0
        mask_b = centers[:, 1] < r_flat
        f_bnd[mask_b, 1] += (r_flat[mask_b] - centers[mask_b, 1]) * 100.0
        mask_t = centers[:, 1] > 1.0 - r_flat
        f_bnd[mask_t, 1] -= (centers[mask_t, 1] - (1.0 - r_flat[mask_t])) * 100.0
        
        # Integrate dynamics with damping
        forces = 0.5 * forces + f_rep + f_bnd
        centers += forces * 0.02
        centers = np.clip(centers, 0.001, 0.999)
        
        # Grow radii when packing is stable
        if np.max(overlap) < 1e-5:
            radii *= 1.00005
            
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_centers = None
    best_sum = -1.0
    best_radii = None
    
    bounds = [(0.001, 0.999)] * (2 * N)
    
    # Phase 1: Force simulation baseline
    c_sim = simulate_packing()
    s_sim, r_sim = compute_sum_radii(c_sim)
    best_centers = c_sim.copy()
    best_sum = s_sim
    best_radii = r_sim.copy()
    
    # Phase 2: Multi-start Powell refinement
    np.random.seed(123)
    for trial in range(50):
        if trial == 0:
            c0 = best_centers.copy()
        elif trial % 8 == 0:
            # Occasional random restart to explore new basins
            c0 = np.random.rand(N, 2) * 0.8 + 0.1
        else:
            # Perturb best known configuration
            c0 = best_centers + np.random.normal(0, 0.004, (N, 2))
            
        c0 = np.clip(c0, 0.001, 0.999)
        
        try:
            res = minimize(objective, c0.flatten(), method='Powell', bounds=bounds,
                           options={'maxiter': 1500, 'ftol': 1e-12, 'xtol': 1e-12, 'disp': False})
            if -res.fun > best_sum:
                c_opt = res.x.reshape(N, 2)
                s_opt, r_opt = compute_sum_radii(c_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            continue
            
    # Phase 3: Safety shrink to guarantee strict validation compliance
    best_radii *= 0.999995
    
    return best_centers, best_radii, float(np.sum(best_radii))
