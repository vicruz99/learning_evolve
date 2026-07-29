# sol_000342 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 776f37f0) state=f4429c4e sum of radii=0.580512 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def get_lp_matrix(n):
    """Precompute the constraint matrix for the radii LP."""
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    m = len(pairs)
    A = np.zeros((m, n))
    for k, (i, j) in enumerate(pairs):
        A[k, i] = 1.0
        A[k, j] = 1.0
    return A, pairs

def get_initial_centers(n, seed):
    """Generate hexagonal lattice initialization with perturbation."""
    np.random.seed(seed)
    pts = []
    s = 0.22
    for i in range(7):
        for j in range(7):
            x = j * s + (i % 2) * s / 2
            y = i * s * np.sqrt(3) / 2
            if x < 1.0 and y < 1.0:
                pts.append([x, y])
    pts = np.array(pts[:n])
    
    # Normalize to [0,1] with margin
    if pts.size > 0:
        pts -= pts.min(axis=0)
        span = pts.max(axis=0) - pts.min(axis=0)
        if np.any(span > 0):
            pts /= span
        pts *= 0.80
        pts += 0.10
        
    # Add controlled noise
    pts += np.random.randn(n, 2) * 0.015
    return np.clip(pts, 0.02, 0.98)

def compute_optimal_radii(centers, A_lp):
    """Solve LP to maximize sum of radii given fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    # Compute pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Extract upper triangular distances for LP RHS
    b_ub = []
    for i in range(n):
        for j in range(i + 1, n):
            b_ub.append(dists[i, j])
    b_ub = np.array(b_ub)
    
    # Boundary bounds for each radius
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mr = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(mr, 1e-9)))
        
    res = linprog(c_obj, A_ub=A_lp, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return np.maximum(res.x, 0.0)
    return np.full(n, 1e-5)

def optimize_packing():
    """Run multiple trials of the alternating LP/Force optimization."""
    n = 26
    A_lp, _ = get_lp_matrix(n)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    for seed in range(12):
        centers = get_initial_centers(n, seed)
        radii = np.full(n, 0.05)
        
        # Simulation parameters
        k_rep = 60.0
        
        for step in range(600):
            # 1. Optimize radii via LP
            radii = compute_optimal_radii(centers, A_lp)
            
            # 2. Compute repulsive forces
            diff = centers[:, None, :] - centers[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, 1e-12)
            dist = np.maximum(dist, 1e-6) # Prevent singularity
            
            force_mag = k_rep * (radii[:, None] + radii[None, :]) / (dist**2)
            forces = np.sum(diff * force_mag[:, :, None], axis=1)
            
            # 3. Update positions with cooling schedule
            dt = 0.008 / (1.0 + step * 0.005)
            centers += forces * dt
            
            # 4. Enforce boundary constraints strictly
            for i in range(n):
                centers[i, 0] = np.clip(centers[i, 0], radii[i], 1.0 - radii[i])
                centers[i, 1] = np.clip(centers[i, 1], radii[i], 1.0 - radii[i])
                
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
    return best_centers, best_radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run the circle packing optimization and return the best result.
    Returns: (centers, radii, sum_radii)
    """
    centers, radii = optimize_packing()
    
    # Final safety check: ensure strict non-overlap with tolerance
    # The LP and clipping guarantee this, but we clamp radii slightly if needed
    # to pass the 1e-12 validation tolerance robustly.
    n = centers.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if d < radii[i] + radii[j] - 1e-12:
                # Adjust radii down proportionally to fix tiny violations
                ratio = d / (radii[i] + radii[j] + 1e-12)
                radii[i] *= ratio
                radii[j] *= ratio
                
    return centers, radii, float(np.sum(radii))
