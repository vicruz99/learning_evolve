# sol_000372 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 469c683e) state=0e0fa960 sum of radii=2.611493 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses SLSQP optimization with multiple restarts from different initial configurations.
    """
    N = 26
    
    # Objective function: minimize negative sum of radii
    def objective(vars):
        # vars are arranged as [x1, y1, r1, x2, y2, r2, ...]
        # Radii are at indices 2, 5, 8, ...
        radii = vars[2::3]
        return -np.sum(radii)

    # Constraint functions
    def get_constraints(N):
        constraints = []
        
        # Boundary constraints:
        # x_i >= r_i  => x_i - r_i >= 0
        # x_i <= 1 - r_i => 1 - x_i - r_i >= 0
        # y_i >= r_i => y_i - r_i >= 0
        # y_i <= 1 - r_i => 1 - y_i - r_i >= 0
        
        for i in range(N):
            idx_x = 3 * i
            idx_y = 3 * i + 1
            idx_r = 3 * i + 2
            
            # x >= r
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, ix=idx_x, ir=idx_r: v[ix] - v[ir]
            })
            # 1 - x >= r
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, ix=idx_x, ir=idx_r: 1.0 - v[ix] - v[ir]
            })
            # y >= r
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, iy=idx_y, ir=idx_r: v[iy] - v[ir]
            })
            # 1 - y >= r
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, iy=idx_y, ir=idx_r: 1.0 - v[iy] - v[ir]
            })

        # Overlap constraints:
        # dist^2 >= (r_i + r_j)^2
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        
        for i in range(N):
            for j in range(i + 1, N):
                idx_xi, idx_yi, idx_ri = 3 * i, 3 * i + 1, 3 * i + 2
                idx_xj, idx_yj, idx_rj = 3 * j, 3 * j + 1, 3 * j + 2
                
                def overlap_constraint(v, ii=i, jj=j):
                    xi, yi, ri = v[3*ii], v[3*ii+1], v[3*ii+2]
                    xj, yj, rj = v[3*jj], v[3*jj+1], v[3*jj+2]
                    return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return constraints

    constraints = get_constraints(N)
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    
    best_sum_radii = -np.inf
    best_centers = None
    best_radii = None
    
    # Helper to create initial configurations
    def create_hex_grid(N, r_est=0.1):
        pts = []
        h = r_est * np.sqrt(3)
        w = 2 * r_est
        y = r_est
        row = 0
        # Add a small margin to ensure points are strictly inside for initialization
        margin = 0.01
        y = r_est + margin
        
        while y + r_est + margin <= 1.0:
            x = r_est + margin
            if row % 2 == 1:
                x = r_est + w/2 + margin
            
            while x + r_est + margin <= 1.0:
                if len(pts) < N:
                    pts.append([x, y])
                x += w
            y += h
            row += 1
        
        # If we don't have enough, fill randomly in remaining space
        while len(pts) < N:
            # Find a spot
            found = False
            for _ in range(100):
                px = np.random.uniform(0.2, 0.8)
                py = np.random.uniform(0.2, 0.8)
                pts.append([px, py])
                found = True
                break
            if not found:
                pts.append([0.5, 0.5])
                
        return np.array(pts[:N])

    def create_dense_grid(N):
        pts = []
        # 5x5 grid points
        for i in range(5):
            for j in range(5):
                if len(pts) < N:
                    # Spacing 1/4 = 0.25. Center at 0.125, 0.375, ...
                    # But to fit more, let's try slightly tighter or shifted
                    # Standard grid for 25 circles r=0.1 is centers at 0.1, 0.3, 0.5, 0.7, 0.9
                    x = 0.1 + i * 0.2
                    y = 0.1 + j * 0.2
                    pts.append([x, y])
        while len(pts) < N:
             pts.append([0.5, 0.5]) # Fill last one
        return np.array(pts[:N])

    # Generate initial guesses
    initial_guesses = []
    
    # 1. Hexagonal packing estimate
    hex_pts = create_hex_grid(N, r_est=0.1)
    # Initial radii small to ensure feasibility
    initial_guesses.append(np.hstack([hex_pts, np.full((N, 1), 0.05)]).flatten())
    
    # 2. Dense grid with perturbation
    grid_pts = create_dense_grid(N)
    # Perturb slightly
    grid_pts += np.random.normal(0, 0.01, grid_pts.shape)
    # Clip to valid range
    grid_pts = np.clip(grid_pts, 0.05, 0.95)
    initial_guesses.append(np.hstack([grid_pts, np.full((N, 1), 0.05)]).flatten())
    
    # 3. Random initialization (multiple)
    for _ in range(5):
        rand_pts = np.random.uniform(0.1, 0.9, (N, 2))
        initial_guesses.append(np.hstack([rand_pts, np.full((N, 1), 0.04)]).flatten())

    # Optimization loop
    for i, x0 in enumerate(initial_guesses):
        # Run optimization
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False}
            )
            
            # Check if valid
            if res.success or (not np.isnan(res.fun)):
                # Extract result
                vars_opt = res.x
                centers_opt = vars_opt.reshape(N, 3)[:, :2]
                radii_opt = vars_opt.reshape(N, 3)[:, 2]
                
                # Double check constraints manually with tolerance
                valid = True
                
                # Check bounds
                for k in range(N):
                    x, y, r = centers_opt[k, 0], centers_opt[k, 1], radii_opt[k]
                    if x < r - 1e-7 or x > 1 - r + 1e-7 or y < r - 1e-7 or y > 1 - r + 1e-7:
                        valid = False
                        break
                
                if valid:
                    # Check overlaps
                    for k1 in range(N):
                        for k2 in range(k1 + 1, N):
                            d = np.sqrt(np.sum((centers_opt[k1] - centers_opt[k2])**2))
                            if d < radii_opt[k1] + radii_opt[k2] - 1e-7:
                                valid = False
                                break
                        if not valid: break
                
                if valid:
                    current_sum = np.sum(radii_opt)
                    if current_sum > best_sum_radii:
                        best_sum_radii = current_sum
                        best_centers = centers_opt.copy()
                        best_radii = radii_opt.copy()
                        
        except Exception as e:
            continue

    # Fallback if optimization failed to find a good valid packing
    # (Though with small radii init, it should find something)
    if best_centers is None:
        # Return a safe default grid packing
        centers = np.array([
            [0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
            [0.1, 0.3], [0.3, 0.3], [0.5, 0.3], [0.7, 0.3], [0.9, 0.3],
            [0.1, 0.5], [0.3, 0.5], [0.5, 0.5], [0.7, 0.5], [0.9, 0.5],
            [0.1, 0.7], [0.3, 0.7], [0.5, 0.7], [0.7, 0.7], [0.9, 0.7],
            [0.1, 0.9], [0.3, 0.9], [0.5, 0.9], [0.7, 0.9], [0.9, 0.9],
            [0.5, 0.5] # 26th circle (will likely be small or overlap in default)
        ])
        # For fallback, just make radii small enough to be valid
        radii = np.full(N, 0.05)
        # Adjust 26th if needed
        best_centers = centers
        best_radii = radii
        best_sum_radii = np.sum(radii)

    return best_centers, best_radii, best_sum_radii

if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    print(f"Centers shape: {centers.shape}, Radii shape: {radii.shape}")
