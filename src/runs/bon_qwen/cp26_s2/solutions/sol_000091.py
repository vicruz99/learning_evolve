# sol_000091 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8101c7b4) state=1742e617 sum of radii=2.500446 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def calculate_max_radii(centers):
    """Solves the LP to find radii that maximize the sum of radii for fixed centers."""
    n = centers.shape[0]
    b = np.zeros((1, n))
    A_eq = np.zeros((0, n))
    b_eq = np.zeros(0)
    
    # Objective: maximize sum of radii -> minimize -sum(radii)
    c_obj = np.ones(n) * -1
    
    # Bounds: 0 <= r_i
    bounds = [(0, None) for _ in range(n)]
    
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    A_ub = np.zeros((4 * n, n))
    b_ub = np.zeros(4 * n)
    
    for i in range(n):
        x, y = centers[i]
        dist_to_boundary = min(x, 1 - x, y, 1 - y)
        
        # r_i <= dist_to_boundary
        A_ub[i, i] = 1
        b_ub[i] = dist_to_boundary
        
        # Lower boundary (redundant with 0 but explicit)
        A_ub[n + i, i] = -1
        b_ub[n + i] = 0

    # Pairwise distance constraints: r_i + r_j <= ||c_i - c_j||
    n_pairs = n * (n - 1) // 2
    A_ub_pairwise = np.zeros((n_pairs, n))
    b_ub_pairwise = np.zeros(n_pairs)
    
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            A_ub_pairwise[k, i] = 1
            A_ub_pairwise[k, j] = 1
            b_ub_pairwise[k] = dist
            k += 1
            
    # Combine constraints
    A_total = np.vstack([A_ub, A_ub_pairwise])
    b_total = np.concatenate([b_ub, b_ub_pairwise])
    
    # Solve LP
    try:
        res = linprog(c_obj, A_ub=A_total, b_ub=b_total, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            return None
    except Exception:
        return None

def get_packing_score(centers):
    radii = calculate_max_radii(centers)
    if radii is not None and all(r >= 0 for r in radii):
        return np.sum(radii), radii
    return 0.0, None

def run_packing():
    n = 26
    # 1. Initial Grid Setup (5x5 + 1 in gap)
    centers = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx < n - 1:
                centers[idx] = [0.1 + 0.2 * i, 0.1 + 0.2 * j]
                idx += 1
    # Place 26th circle in the center gap
    centers[n-1] = [0.2, 0.2]
    
    best_centers = centers.copy()
    best_score = get_packing_score(centers)[0]
    
    # 2. Simulated Annealing Optimization
    T = 0.01
    step_size = 0.05
    
    for i in range(1000):
        # Cool down
        T = max(T * 0.995, 1e-6)
        step_size = max(0.001, step_size * 0.995)
        
        # Perturb a random circle
        idx = np.random.randint(n)
        new_center = best_centers[idx].copy() + np.random.uniform(-step_size, step_size, 2)
        new_center = np.clip(new_center, 0.01, 0.99) # Keep away from boundary
        
        candidate_centers = best_centers.copy()
        candidate_centers[idx] = new_center
        
        score, radii = get_packing_score(candidate_centers)
        if score is None: continue
            
        diff = score - best_score
        if diff > 0 or np.random.random() < np.exp(diff / T):
            best_centers = candidate_centers
            best_score = score
            if i % 50 == 0:
                pass # Silent iteration
                
    # Final refinement with small steps
    T = 0.001
    step_size = 0.005
    for i in range(500):
        idx = np.random.randint(n)
        new_center = best_centers[idx].copy() + np.random.uniform(-step_size, step_size, 2)
        new_center = np.clip(new_center, 0.005, 0.995)
        
        candidate_centers = best_centers.copy()
        candidate_centers[idx] = new_center
        score, _ = get_packing_score(candidate_centers)
        if score > best_score:
            best_centers = candidate_centers
            best_score = score

    final_radii = calculate_max_radii(best_centers)
    if final_radii is None:
        final_radii = np.zeros(n)
        
    return best_centers, final_radii, float(np.sum(final_radii))
