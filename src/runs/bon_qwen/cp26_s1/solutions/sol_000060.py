# sol_000060 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 608ae89b) state=2d9a1b0c sum of radii=2.620561 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(params, N, weight):
    """
    Computes the objective value: -sum(radii) + weight * penalty
    Penalty accounts for boundary violations and circle overlaps.
    """
    pts = params.reshape(-1, 3)
    centers = pts[:, :2]
    radii = pts[:, 2]
    
    score = -np.sum(radii)
    penalty = 0.0
    
    # Boundary penalty: circles must stay within [0,1]x[0,1]
    for i in range(N):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        # r <= x, r <= 1-x, r <= y, r <= 1-y
        penalty += max(0.0, r - x)**2
        penalty += max(0.0, r - (1.0 - x))**2
        penalty += max(0.0, r - y)**2
        penalty += max(0.0, r - (1.0 - y))**2
        
    # Overlap penalty: distance between centers must be >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            overlap = radii[i] + radii[j] - dist
            if overlap > 0.0:
                penalty += overlap**2
                
    return score + weight * penalty

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    N = 26
    np.random.seed(42)
    
    # 1. Initialization: Hexagonal-ish grid layout
    centers = []
    for i in range(5):
        for j in range(5):
            centers.append([0.1 + j * 0.2, 0.1 + i * 0.2])
    centers.append([0.5, 0.5])  # 26th circle
    
    # Add small random noise to break symmetry
    centers = np.array(centers) + np.random.randn(N, 2) * 0.01
    centers = np.clip(centers, 0.02, 0.98)  # Keep strictly inside initially
    
    # Initial radii: start moderately sized
    radii = np.full(N, 0.08) + np.random.randn(N) * 0.01
    radii = np.clip(radii, 0.03, 0.12)
    
    # Flatten to 1D parameter array for scipy: [x1, y1, r1, x2, y2, r2, ...]
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = centers[i, 0]
        params[3*i+1] = centers[i, 1]
        params[3*i+2] = radii[i]
        
    # Bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    
    # 2. Optimization with increasing penalty weights
    current_params = params.copy()
    weight_schedule = [10.0, 50.0, 200.0, 1000.0, 5000.0, 20000.0]
    
    for w in weight_schedule:
        res = minimize(
            compute_objective, 
            current_params, 
            args=(N, w),
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
        )
        current_params = res.x
        
    # 3. Extract and format results
    final_centers = current_params.reshape(-1, 3)[:, :2]
    final_radii = current_params.reshape(-1, 3)[:, 2]
    
    # 4. Post-processing: Strict validity enforcement
    # Iteratively clamp radii to satisfy constraints exactly within tolerance
    for _ in range(100):
        changed = False
        # Boundary constraints
        for i in range(N):
            r_max = min(
                final_centers[i, 0], 
                1.0 - final_centers[i, 0],
                final_centers[i, 1], 
                1.0 - final_centers[i, 1]
            )
            if final_radii[i] > r_max + 1e-14:
                final_radii[i] = r_max
                changed = True
                
        # Overlap constraints
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt(np.sum((final_centers[i] - final_centers[j])**2))
                sum_r = final_radii[i] + final_radii[j]
                if sum_r > dist + 1e-12:
                    # Shrink both proportionally to maintain sum reduction fairness
                    shrink = (sum_r - dist) / 2.0
                    final_radii[i] -= shrink
                    final_radii[j] -= shrink
                    changed = True
        if not changed:
            break
            
    # Ensure non-negative radii after clamping
    final_radii = np.maximum(final_radii, 0.0)
    
    sum_radii = float(np.sum(final_radii))
    return final_centers, final_radii, sum_radii
