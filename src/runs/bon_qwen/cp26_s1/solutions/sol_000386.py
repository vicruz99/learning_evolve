# sol_000386 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 916b0b30) state=04aa24a5 sum of radii=2.153954 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import distance_matrix

def compute_max_radii(centers):
    """Solve LP to find radii that maximize sum given fixed centers."""
    n = centers.shape[0]
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        for val in [centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1]]:
            A_ub.append(row)
            b_ub.append(val)
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    dists = distance_matrix(centers, centers)
    np.fill_diagonal(dists, np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Maximize sum(r) <=> Minimize -sum(r)
    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*n, method='highs')
    if res.success:
        return res.x
    else:
        return np.zeros(n)

def relax_centers(centers, radii, n_steps=100, dt=0.015):
    """Adjust centers to resolve overlaps and respect boundaries using force dynamics."""
    n = centers.shape[0]
    centers = centers.copy()
    
    for _ in range(n_steps):
        forces = np.zeros((n, 2))
        
        # Boundary forces
        for i in range(n):
            for axis, coord in enumerate([centers[i, 0], centers[i, 1]]):
                if coord < radii[i]:
                    forces[i, axis] += (radii[i] - coord) * 40.0
                elif coord > 1.0 - radii[i]:
                    forces[i, axis] -= (coord - (1.0 - radii[i])) * 40.0
                    
        # Pairwise repulsion forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                d2 = dx*dx + dy*dy
                d = np.sqrt(d2)
                r_sum = radii[i] + radii[j]
                
                if d < r_sum:
                    overlap = r_sum - d
                    force_mag = overlap * 50.0
                    if d > 1e-9:
                        fx = (dx / d) * force_mag
                        fy = (dy / d) * force_mag
                        forces[i] += [fx, fy]
                        forces[j] += [-fx, -fy]
                    else:
                        # Degenerate case: same center
                        forces[i] += [1.0, 0.0]
                        forces[j] += [-1.0, 0.0]
                        
        centers += forces * dt
        centers = np.clip(centers, 0.0, 1.0)
        
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    n = 26
    
    # Initialize centers in a dense hexagonal-like grid
    centers = []
    idx = 0
    # 5 columns, 6 rows gives 30 spots, we take first 26
    for r in range(6):
        for c in range(5):
            if idx >= n:
                break
            x = 0.12 + c * 0.18
            y = 0.12 + r * 0.16 + (c % 2) * 0.08
            centers.append([x, y])
            idx += 1
        if idx >= n:
            break
    centers = np.array(centers)
    
    # Initialize with small radii
    radii = np.ones(n) * 0.01
    
    best_sum = 0.0
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    # Alternating optimization
    for it in range(400):
        # 1. Maximize radii for current centers via LP
        radii = compute_max_radii(centers)
        curr_sum = np.sum(radii)
        
        # Track best
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
        # 2. Relax centers to make room for larger radii
        # Adaptive steps: more early on, fewer later for fine-tuning
        steps = 150 if it < 100 else 60
        centers = relax_centers(centers, radii, n_steps=steps, dt=0.012)
        
        # 3. Periodic perturbation to escape local minima
        if it % 40 == 39:
            noise_scale = 0.03 * (0.5 + 0.5 * (1 - it/400))
            centers += np.random.normal(0, noise_scale, centers.shape)
            centers = np.clip(centers, 0.0, 1.0)
            
    # Final projection to ensure strict validity within tolerance
    # Recompute radii one last time with best centers
    final_radii = compute_max_radii(best_centers)
    final_sum = np.sum(final_radii)
    
    return best_centers, final_radii, final_sum
