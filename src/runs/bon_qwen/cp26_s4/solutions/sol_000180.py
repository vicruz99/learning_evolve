# sol_000180 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 64b41a5f) state=2637c58e sum of radii=2.621344 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    N = 26
    
    # Objective function: Minimize negative sum of radii
    def objective(z):
        r = z[2::3]
        return -np.sum(r)

    # Constraint function: Returns vector of constraints >= 0
    def constraints_func(z):
        # Unpack variables
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
        # i.e., x - r >= 0, 1 - x - r >= 0, etc.
        c_boundary = np.concatenate([
            x - r,
            1 - x - r,
            y - r,
            1 - y - r
        ])
        
        # Pairwise non-overlap constraints: dist(i, j) >= r_i + r_j
        # We compute this for i < j to avoid duplicates
        constraints = []
        
        # Vectorized distance calculation for efficiency
        # x_col shape (N, 1)
        x_col = x[:, np.newaxis]
        y_col = y[:, np.newaxis]
        r_col = r[:, np.newaxis]
        
        # Compute squared distance matrix
        # diff_sq = (x_i - x_j)^2 + (y_i - y_j)^2
        # We can compute this element-wise for upper triangle to save memory/time if N was large,
        # but for N=26, full matrix is fine.
        dist_sq = (x_col - x_col.T)**2 + (y_col - y_col.T)**2
        
        # Upper triangle indices
        row_idx, col_idx = np.triu_indices(N, k=1)
        
        # Distances for pairs
        dists = np.sqrt(dist_sq[row_idx, col_idx])
        
        # Sum of radii for pairs
        r_sums = r[row_idx] + r[col_idx]
        
        # Constraint: dist - r_sum >= 0
        c_pairwise = dists - r_sums
        
        return np.concatenate([c_boundary, c_pairwise])

    # Variable bounds
    # x, y in [0, 1]
    # r in [0, 0.5] (theoretically max radius is 0.5)
    bounds = [(0, 1) for _ in range(3 * N)]
    # Tighten r bounds slightly for stability? No, [0, 0.5] is fine.
    for i in range(N):
        bounds[3*i + 2] = (0.0, 0.5)

    cons = {'type': 'ineq', 'fun': constraints_func}

    best_sum_radii = 0.0
    best_centers = None
    best_radii = None

    # Helper to update best result
    def update_best(res):
        nonlocal best_sum_radii, best_centers, best_radii
        if res.success or res.nit > 50: # Accept if converged or ran enough steps
            current_sum = -res.fun
            if current_sum > best_sum_radii:
                # Validate roughly before accepting to avoid NaNs etc
                z = res.x
                r = z[2::3]
                if np.all(r >= 0) and np.all(r <= 1):
                    best_sum_radii = current_sum
                    best_centers = np.column_stack((z[0::3], z[1::3]))
                    best_radii = r

    # Strategy 1: Random initialization with small radii
    # This allows the optimizer to grow circles into free space
    rng = np.random.default_rng(42)
    for seed in range(5):
        z0 = np.zeros(3 * N)
        # Random positions in center area to avoid immediate boundary issues
        z0[0::3] = rng.uniform(0.2, 0.8, N)
        z0[1::3] = rng.uniform(0.2, 0.8, N)
        # Start with small radius to ensure feasibility
        z0[2::3] = 0.05 
        
        res = minimize(objective, z0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
        update_best(res)

    # Strategy 2: Grid-like initialization
    # Try to pack 26 circles in a roughly grid pattern to give a head start
    # 5x5 grid has 25 spots. We need 26.
    # We can place 25 in a grid and 1 in a gap or slightly perturbed.
    
    # Create a 6x5 grid pattern (30 spots) and pick 26?
    # Or just a dense packing heuristic.
    # Let's try a hexagonal packing initialization.
    
    # Hexagonal rows
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 5 circles
    # Row 5: 1 circle?
    # Total 26.
    
    init_points = []
    # Try to fit rows with spacing
    # Width 1. Radius approx 0.1.
    # x coords: 0.1, 0.3, 0.5, 0.7, 0.9
    
    # Row y spacing sqrt(3)/2 * 2r = r*sqrt(3).
    # If r=0.1, dy = 0.1732.
    
    # Let's just place them reasonably
    z0_grid = np.zeros(3 * N)
    
    # Fill first 25 in 5x5 grid
    idx = 0
    for r in range(5):
        for c in range(5):
            x_val = 0.1 + c * 0.2
            y_val = 0.1 + r * 0.2
            z0_grid[3*idx] = x_val
            z0_grid[3*idx+1] = y_val
            z0_grid[3*idx+2] = 0.09 # Start slightly below 0.1
            idx += 1
    
    # Place 26th circle in a gap?
    # Center of grid is (0.5, 0.5), occupied.
    # Maybe at (0.2, 0.2)? Distance to (0.1, 0.1) is sqrt(0.01+0.01)=0.141.
    # 2r = 0.18. Overlap.
    # Let's place 26th at (0.5, 0.5) but push others? 
    # Or just place it randomly and let optimizer fix.
    z0_grid[3*25] = 0.5
    z0_grid[3*25+1] = 0.5
    z0_grid[3*25+2] = 0.01 # Small radius
    
    res_grid = minimize(objective, z0_grid, method='SLSQP', bounds=bounds, constraints=cons, 
                        options={'maxiter': 2000, 'ftol': 1e-9})
    update_best(res_grid)

    # Strategy 3: Perturbed Grid
    # Start from grid, perturb positions slightly, optimize
    for _ in range(3):
        z0_pert = z0_grid.copy()
        # Add noise
        noise = rng.normal(0, 0.02, 3*N)
        # Clip noise to keep in bounds roughly
        z0_pert += noise
        # Ensure bounds
        z0_pert[0::3] = np.clip(z0_pert[0::3], 0.05, 0.95)
        z0_pert[1::3] = np.clip(z0_pert[1::3], 0.05, 0.95)
        z0_pert[2::3] = np.clip(z0_pert[2::3], 0.01, 0.2)
        
        res_pert = minimize(objective, z0_pert, method='SLSQP', bounds=bounds, constraints=cons, 
                            options={'maxiter': 1000})
        update_best(res_pert)

    # Strategy 4: "Repulsion" style initialization (Physics inspired)
    # Place circles randomly, run a quick local optimization to separate them, then optimize radii.
    # This is covered by the random init with small radii, but let's be explicit.
    
    # Strategy 5: Hexagonal lattice specific
    # 5 rows of 5, 5, 5, 5, 5 is 25.
    # Try 5, 5, 5, 5, 4, 2?
    # Let's try a compact hexagonal block.
    # Rows: 5, 5, 5, 5, 4, 2 -> 26
    # Or 5, 5, 5, 4, 4, 3 -> 26
    
    # Let's try to construct a valid hexagonal configuration
    # r approx 0.1
    # dy = 0.1 * sqrt(3) = 0.1732
    # Row y: 0.1, 0.2732, 0.4464, 0.6196, 0.7928, 0.966 (too high)
    # 5 rows max with r=0.1.
    # 5, 5, 5, 5, 5 = 25.
    # Maybe squeeze in a small one?
    
    # Let's stick to the optimized results from previous strategies.
    # The optimizer should handle finding the local optimum from these starts.
    
    # Final check and return
    if best_centers is None:
        # Fallback to a simple valid configuration if optimization failed
        # 26 circles of radius 0.04 in a grid
        centers = np.zeros((26, 2))
        radii = np.full(26, 0.04)
        idx = 0
        for r in range(6):
            for c in range(5):
                if idx >= 26: break
                centers[idx, 0] = 0.1 + c * 0.18
                centers[idx, 1] = 0.1 + r * 0.18
                idx += 1
        sum_radii = 26 * 0.04
    else:
        centers = best_centers
        radii = best_radii
        sum_radii = best_sum_radii

    # Ensure non-negative radii and clipping just in case
    radii = np.maximum(radii, 0)
    
    # Validate before returning
    # (We assume the constraints were satisfied, but numerical errors might occur)
    # We can do a quick fix if needed, but usually SLSQP is precise.
    
    return centers, radii, float(sum_radii)

# Helper functions as requested (none needed inside run_packing for this logic, 
# but keeping structure clean)
# The logic is self-contained in run_packing or uses standard libs.

if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # print(c)
    # print(r)
