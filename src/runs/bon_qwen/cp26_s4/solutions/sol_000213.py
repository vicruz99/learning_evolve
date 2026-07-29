# sol_000213 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 722eaafb) state=f233125d sum of radii=2.352279 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def compute_max_radii(centers):
    """
    Computes the optimal radii for a given set of centers using Linear Programming.
    Maximizes sum of radii subject to non-overlap and boundary constraints.
    """
    n = centers.shape[0]
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Distance to boundaries: min(dist to 0, dist to 1)
    b = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    b = np.maximum(b, 0.0)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # LP Setup
    # Maximize sum(r_i) => Minimize -sum(r_i)
    c_obj = -np.ones(n)
    
    # Constraints: r_i + r_j <= dist(i, j) for all i < j
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            
    m = len(pairs)
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    
    for idx, (i, j) in enumerate(pairs):
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        b_ub[idx] = dists[i, j]
        
    bounds = [(0.0, b[i]) for i in range(n)]
    
    # Solve LP
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except:
        pass
        
    return np.zeros(n)

def run_packing():
    n = 26
    rng = np.random.RandomState(42)
    
    # Initialize with hexagonal pattern
    # Estimate spacing for good density
    s = 0.215
    centers = []
    row = 0
    while len(centers) < n:
        for col in range(10):
            cx = col * s + (row % 2) * s / 2
            cy = row * s * np.sqrt(3) / 2
            if cx <= 1.0 and cy <= 1.0:
                centers.append([cx, cy])
        row += 1
        
    centers = np.array(centers[:n])
    
    # Center and scale to fit comfortably within [0,1]
    min_c = np.min(centers, axis=0)
    max_c = np.max(centers, axis=0)
    size = max_c - min_c
    scale = 0.90 / np.max(size)
    centers *= scale
    centers += (1.0 - scale * size) / 2.0
    
    # Ensure strictly inside boundaries
    centers = np.clip(centers, 0.02, 0.98)
    
    best_centers = centers.copy()
    best_sum = 0.0
    
    step = 0.03
    for it in range(2500):
        radii = compute_max_radii(centers)
        curr_sum = np.sum(radii)
        
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers.copy()
            
        # Identify bottleneck (circle with smallest radius)
        min_idx = np.argmin(radii)
        r = radii[min_idx]
        cx, cy = centers[min_idx]
        
        # Check distances to walls
        gaps = [cx - r, 1.0 - (cx + r), cy - r, 1.0 - (cy + r)]
        min_gap = min(gaps)
        
        if min_gap <= 1e-4:
            # Move away from the closest wall
            if abs(gaps[0] - min_gap) < 1e-9:
                centers[min_idx, 0] += step
            elif abs(gaps[1] - min_gap) < 1e-9:
                centers[min_idx, 0] -= step
            elif abs(gaps[2] - min_gap) < 1e-9:
                centers[min_idx, 1] += step
            elif abs(gaps[3] - min_gap) < 1e-9:
                centers[min_idx, 1] -= step
        else:
            # Move away from the closest neighbor
            dists = np.linalg.norm(centers - centers[min_idx], axis=1)
            dists[min_idx] = np.inf
            neighbor_idx = np.argmin(dists)
            
            vec = centers[min_idx] - centers[neighbor_idx]
            dist = np.linalg.norm(vec)
            if dist > 1e-9:
                centers[min_idx] += (vec / dist) * step
                
        # Keep centers inside
        centers = np.clip(centers, 0.02, 0.98)
        
        # Random perturbation to escape local minima
        if rng.rand() < 0.05:
            idx = rng.randint(n)
            centers[idx] += rng.randn(2) * step * 0.3
            centers = np.clip(centers, 0.02, 0.98)
            
        # Decay step size
        if it % 100 == 0 and step > 0.001:
            step *= 0.95
            
    final_radii = compute_max_radii(best_centers)
    return best_centers, final_radii, np.sum(final_radii)
