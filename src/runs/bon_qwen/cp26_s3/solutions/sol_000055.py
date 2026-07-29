# sol_000055 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a7088d37) state=4a85cf54 sum of radii=1.801530 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initial Configuration: Hexagonal Lattice
    # We generate 26 points in a staggered pattern to approximate optimal packing.
    # A hexagonal packing is denser than a square grid.
    # We choose row counts to sum to 26: 5 + 4 + 5 + 4 + 5 + 3 = 26.
    row_counts = [5, 4, 5, 4, 5, 3] 
    pts = []
    y = 0
    dy = np.sqrt(3)/2
    
    for r in row_counts:
        x_coords = np.linspace(0, r-1, r)
        # Shift every other row to create hexagonal packing
        if len(pts) % 2 == 1:
            x_coords += 0.5
        for x in x_coords:
            pts.append([x, y])
        y += dy
    
    pts = np.array(pts)
    # Normalize to fit within [0.1, 0.9] roughly to leave room for radii
    min_c = pts.min(axis=0)
    max_c = pts.max(axis=0)
    pts = (pts - min_c) / (max_c - min_c) * 0.7 + 0.15
    
    centers_init = pts
    radii_init = np.ones(n) * 0.05
    
    # 2. Objective Function with Penalty
    # We maximize the sum of radii by minimizing the negative sum.
    # Constraints (boundaries and non-overlap) are enforced via penalty terms.
    def objective(vars):
        centers = np.column_stack((vars[0::3], vars[1::3]))
        radii = vars[2::3]
        
        # Objective: minimize negative sum of radii
        obj = -np.sum(radii)
        
        # Penalty weight (high value to enforce constraints strictly)
        penalty = 10000.0
        
        # Boundary violations: circle must be inside [0,1]x[0,1]
        # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
        v_bound = 0.0
        v_bound += np.sum(np.maximum(0, radii - centers[:, 0])**2)
        v_bound += np.sum(np.maximum(0, centers[:, 0] + radii - 1)**2)
        v_bound += np.sum(np.maximum(0, radii - centers[:, 1])**2)
        v_bound += np.sum(np.maximum(0, centers[:, 1] + radii - 1)**2)
        
        # Overlap violations: dist(i,j) >= r_i + r_j
        # Compute pairwise distances efficiently
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        # Overlap amount (positive if overlapping)
        overlap = r_sum - dists
        overlap = np.maximum(0, overlap)
        
        # Sum of squared overlaps (upper triangle to avoid double counting)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        v_overlap = np.sum((overlap[mask])**2)
        
        return obj + penalty * (v_bound + v_overlap)

    # 3. Optimization
    x0 = np.zeros(3 * n)
    x0[0::3] = centers_init[:, 0]
    x0[1::3] = centers_init[:, 1]
    x0[2::3] = radii_init
    
    # Bounds: x, y in [0, 1], r >= 0
    bounds = [(0, 1), (0, 1), (0, None)] * n
    
    # Run L-BFGS-B optimization
    # L-BFGS-B is suitable for large-scale bound-constrained optimization
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 5000, 'ftol': 1e-12})
    
    final_centers = np.column_stack((res.x[0::3], res.x[1::3]))
    final_radii = res.x[2::3]
    final_radii = np.maximum(final_radii, 0)
    
    return final_centers, final_radii, np.sum(final_radii)
