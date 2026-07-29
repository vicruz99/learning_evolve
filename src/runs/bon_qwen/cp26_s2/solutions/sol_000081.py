# sol_000081 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3e058973) state=211aee7d sum of radii=2.576445 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # --- Initialization ---
    # Start with a 5x5 grid pattern for 25 circles
    centers = np.zeros((n, 2))
    
    # Generate grid points
    grid_points = []
    for i in range(5):
        for j in range(5):
            grid_points.append([0.1 + i * 0.2, 0.1 + j * 0.2])
    
    # Add a 26th point in the center of the square
    grid_points.append([0.5, 0.5])
    
    # Convert to numpy array
    centers = np.array(grid_points)
    
    # Small initial radii to ensure feasibility (no overlaps initially)
    # Since grid spacing is 0.2 and radii 0.01, 2r = 0.02 < 0.2, so no overlap.
    radii = np.ones(n) * 0.01
    
    # Flatten state vector for scipy: [x1, y1, x2, y2, ..., x26, y26, r1, r2, ..., r26]
    x0 = np.concatenate([centers.flatten(), radii])
    
    # --- Bounds ---
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    # --- Constraints ---
    cons = []
    
    # Helper to get indices
    def idx_x(i): return 2 * i
    def idx_y(i): return 2 * i + 1
    def idx_r(i): return 2 * n + i
    
    # 1. Boundary Constraints: Circle must be inside [0, 1]x[0, 1]
    # x - r >= 0  =>  r - x <= 0 (ineq: fun >= 0) -> x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    
    for i in range(n):
        # Left wall
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[idx_x(i)] - v[idx_r(i)]})
        # Right wall
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[idx_x(i)] - v[idx_r(i)]})
        # Bottom wall
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[idx_y(i)] - v[idx_r(i)]})
        # Top wall
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[idx_y(i)] - v[idx_r(i)]})

    # 2. Overlap Constraints: dist(i, j)^2 >= (r_i + r_j)^2
    # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (
                    (v[idx_x(i)] - v[idx_x(j)])**2 + 
                    (v[idx_y(i)] - v[idx_y(j)])**2 - 
                    (v[idx_r(i)] + v[idx_r(j)])**2
                )
            })
            
    # --- Objective Function ---
    # Minimize negative sum of radii
    def objective(v):
        r = v[2*n:]
        return -np.sum(r)
        
    # --- Optimization ---
    # Use SLSQP method which handles constraints well
    # We add a seed for reproducibility of the random part if any, 
    # but here the start is deterministic. 
    # We might run multiple restarts if needed, but one good start is usually sufficient.
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'ftol': 1e-9, 'maxiter': 1000})
    
    if not res.success:
        # Fallback or warning, though we proceed with best found
        pass
        
    # Extract results
    best_centers = res.x[:2*n].reshape((n, 2))
    best_radii = res.x[2*n:]
    
    # Ensure radii are non-negative (numerical safety)
    best_radii = np.maximum(best_radii, 0.0)
    
    total_sum = np.sum(best_radii)
    
    return best_centers, best_radii, total_sum

# For local testing/validation
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # print(f"Radii: {radii}")
    # print(f"Centers:\n{centers}")
