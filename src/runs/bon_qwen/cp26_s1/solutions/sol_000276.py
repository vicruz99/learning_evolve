# sol_000276 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 085da352) state=603a645a sum of radii=2.575002 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    def compute_sum_radii(centers, radii):
        return np.sum(radii)

    def validate_internal(centers, radii):
        """Internal validation to check feasibility during optimization."""
        # Check boundary
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x < r - 1e-7 or x > 1 - r + 1e-7 or y < r - 1e-7 or y > 1 - r + 1e-7:
                return False
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
                if dist < radii[i] + radii[j] - 1e-7:
                    return False
        return True

    def get_hexagonal_initialization(r_start, jitter=0):
        """Generates a hexagonal grid initialization."""
        centers = []
        # Approximate rows for 26 circles in a square
        # Pattern: 4, 5, 4, 5, 4, 4 = 26
        rows_counts = [4, 5, 4, 5, 4, 4]
        
        # Spacing based on radius
        x_step = 2 * r_start
        y_step = np.sqrt(3) * r_start
        
        current_y = r_start
        for row_idx, count in enumerate(rows_counts):
            for k in range(count):
                # Offset odd rows
                if row_idx % 2 == 1:
                    x = r_start + x_step / 2 + k * x_step
                else:
                    x = r_start + k * x_step
                
                # Add jitter if specified
                if jitter > 0:
                    x += np.random.uniform(-jitter, jitter)
                    current_y_curr = current_y + np.random.uniform(-jitter, jitter)
                else:
                    current_y_curr = current_y
                
                centers.append([x, current_y_curr])
            current_y += y_step
            
        return np.array(centers)

    def objective(vars_flat):
        """Minimize negative sum of radii."""
        # vars_flat is [x1, y1, r1, x2, y2, r2, ...]
        # Radii are at indices 2, 5, 8, ...
        radii = vars_flat[2::3]
        return -np.sum(radii)

    def bound_constraints(vars_flat):
        """
        Boundary constraints:
        x_i >= r_i  => x_i - r_i >= 0
        x_i <= 1 - r_i => 1 - x_i - r_i >= 0
        y_i >= r_i
        y_i <= 1 - r_i
        """
        vars_2d = vars_flat.reshape(n, 3)
        x = vars_2d[:, 0]
        y = vars_2d[:, 1]
        r = vars_2d[:, 2]
        
        c1 = x - r
        c2 = 1.0 - x - r
        c3 = y - r
        c4 = 1.0 - y - r
        
        return np.concatenate([c1, c2, c3, c4])

    def overlap_constraints(vars_flat):
        """
        Non-overlap constraints:
        dist(i, j)^2 >= (r_i + r_j)^2
        """
        vars_2d = vars_flat.reshape(n, 3)
        x = vars_2d[:, 0]
        y = vars_2d[:, 1]
        r = vars_2d[:, 2]
        
        # Vectorized calculation
        # dx matrix
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        
        # r_sum matrix
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        r_sum_sq = r_sum**2
        
        # We only need constraints for i < j
        # Create a mask for upper triangle
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        
        # Constraint values: dist_sq - r_sum_sq >= 0
        constraints = (dist_sq - r_sum_sq)[mask]
        return constraints

    # Setup bounds for x, y in [0, 1] and r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    constraints = [
        {'type': 'ineq', 'fun': bound_constraints},
        {'type': 'ineq', 'fun': overlap_constraints}
    ]

    best_centers = None
    best_radii = None
    max_sum = -1.0

    # Multi-start optimization
    # Try a few different initial radii and configurations
    initial_radii_candidates = [0.09, 0.095, 0.085]
    
    np.random.seed(42) # Reproducibility

    for r_start in initial_radii_candidates:
        for jitter in [0, 0.01]:
            # Generate initial centers
            centers_init = get_hexagonal_initialization(r_start, jitter=jitter)
            # Adjust centers to ensure they are within bounds initially if jitter pushed them out
            centers_init = np.clip(centers_init, 0, 1)
            
            # Ensure initial radii are valid for these centers (reduce if necessary)
            r_init = np.full(n, r_start)
            
            # Flatten initial guess
            x0 = []
            for i in range(n):
                x0.extend([centers_init[i, 0], centers_init[i, 1], r_init[i]])
            x0 = np.array(x0)

            try:
                res = minimize(
                    objective, 
                    x0, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
                )
                
                if res.success or res.fun < 0: # If converged or improved
                    current_sum = -res.fun
                    if current_sum > max_sum:
                        # Extract solution
                        sol_2d = res.x.reshape(n, 3)
                        cands = sol_2d[:, :2]
                        rad = sol_2d[:, 2]
                        
                        # Basic sanity check (might fail due to numerical precision in solver)
                        # We rely on the final validation in the main return, 
                        # but let's prefer valid intermediate results if possible.
                        # However, SLSQP constraints are 'ineq' >= 0, so small violations might exist.
                        
                        max_sum = current_sum
                        best_centers = cands
                        best_radii = rad
                        
            except Exception:
                continue

    if best_centers is None:
        # Fallback to a simple valid grid if optimization fails completely
        # 5x5 grid minus 4 circles? Or just small random circles.
        # Let's create a valid small packing.
        centers = np.random.rand(n, 2)
        radii = np.full(n, 0.01)
        best_centers = centers
        best_radii = radii

    # Final refinement: ensure strict validity and clamp if needed
    # The solver might give results very close to boundary.
    # We can slightly reduce radii to ensure validate_packing passes.
    
    # Check for any slight violations and fix
    # Since SLSQP handles constraints, violations should be tiny (< 1e-12 ideally).
    # But let's be safe.
    
    # If there are overlaps, reduce radii slightly.
    # A simple heuristic: scale down radii by a tiny factor if needed.
    # But better to just return the result as SLSQP is robust.
    
    # Ensure radii are non-negative
    best_radii = np.maximum(best_radii, 0.0)
    
    # Ensure centers are within [0,1]
    best_centers = np.clip(best_centers, 0.0, 1.0)

    # Final validation check (internal)
    # If invalid, scale down radii by 0.999 until valid
    scale = 1.0
    while not validate_internal(best_centers, best_radii * scale):
        scale *= 0.999
        if scale < 0.1: # Emergency stop
            break
            
    final_radii = best_radii * scale
    final_sum = np.sum(final_radii)

    return best_centers, final_radii, final_sum
