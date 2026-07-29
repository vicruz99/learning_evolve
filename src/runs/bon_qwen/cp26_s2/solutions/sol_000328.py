# sol_000328 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cc22fbce) state=7d06509d sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

# Global constant for number of circles
N_CIRCLES = 26

def run_packing():
    """
    Runs the circle packing optimization to find centers and radii for 26 circles
    in a unit square maximizing the sum of radii.
    """
    best_centers = None
    best_radii = None
    best_sum_radii = -1.0

    # Run multiple times with different seeds to avoid local minima
    num_restarts = 5
    
    for seed in range(num_restarts):
        # 1. Initialization
        centers = np.random.rand(N_CIRCLES, 2)
        # Start with small valid radii to allow expansion
        # 0.02 is small enough to not overlap with random centers usually,
        # but if they do, the optimizer will fix it.
        radii = np.full(N_CIRCLES, 0.02)
        
        # Initial variable vector: [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.concatenate([centers.flatten(), radii])

        # 2. Define Bounds
        # x, y in [0, 1], r in [0, 0.5] (since diameter <= 1)
        bounds = []
        for _ in range(N_CIRCLES):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r

        # 3. Define Constraints
        constraints_list = []

        # Boundary constraints: circle must be inside square
        # x - r >= 0  =>  x - r >= 0
        # 1 - x - r >= 0
        # y - r >= 0
        # 1 - y - r >= 0
        for i in range(N_CIRCLES):
            idx_x = 3 * i
            idx_y = 3 * i + 1
            idx_r = 3 * i + 2
            
            # Constraint: x - r >= 0
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[3*i] - v[3*i+2],
                'jac': lambda v, i=i: jacobian_boundary_x_min(v, i)
            })
            
            # Constraint: 1 - x - r >= 0
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2],
                'jac': lambda v, i=i: jacobian_boundary_x_max(v, i)
            })
            
            # Constraint: y - r >= 0
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[3*i+1] - v[3*i+2],
                'jac': lambda v, i=i: jacobian_boundary_y_min(v, i)
            })
            
            # Constraint: 1 - y - r >= 0
            constraints_list.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2],
                'jac': lambda v, i=i: jacobian_boundary_y_max(v, i)
            })

        # Overlap constraints: dist^2 >= (r1 + r2)^2
        # (x1-x2)^2 + (y1-y2)^2 - (r1+r2)^2 >= 0
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                constraints_list.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: overlap_constraint_fun(v, i, j),
                    'jac': lambda v, i=i, j=j: overlap_constraint_jac(v, i, j)
                })

        # 4. Objective Function: Minimize -sum(radii)
        def objective(v):
            radii = v[2*N_CIRCLES:]
            return -np.sum(radii)

        # 5. Optimization
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'ftol': 1e-9, 'maxiter': 2000, 'disp': False}
            )
            
            if res.success or res.nit > 0:
                final_radii = res.x[2*N_CIRCLES:]
                current_sum = np.sum(final_radii)
                
                # Verify validity manually just in case
                final_centers = res.x[:2*N_CIRCLES].reshape((N_CIRCLES, 2))
                
                # Clean up tiny negative radii due to float errors
                final_radii = np.maximum(final_radii, 0.0)
                
                if validate_packing(final_centers, final_radii):
                    if current_sum > best_sum_radii:
                        best_sum_radii = current_sum
                        best_centers = final_centers
                        best_radii = final_radii
        except Exception:
            continue

    # Fallback if optimization failed (should not happen)
    if best_centers is None:
        return np.zeros((N_CIRCLES, 2)), np.zeros(N_CIRCLES), 0.0

    return best_centers, best_radii, best_sum_radii

# --- Helper functions for constraints and Jacobians ---

def jacobian_boundary_x_min(v, i):
    # Constraint: x_i - r_i >= 0
    # d/dx_i = 1, d/dr_i = -1
    jac = np.zeros(3 * N_CIRCLES)
    jac[3 * i] = 1.0
    jac[3 * i + 2] = -1.0
    return jac

def jacobian_boundary_x_max(v, i):
    # Constraint: 1 - x_i - r_i >= 0
    # d/dx_i = -1, d/dr_i = -1
    jac = np.zeros(3 * N_CIRCLES)
    jac[3 * i] = -1.0
    jac[3 * i + 2] = -1.0
    return jac

def jacobian_boundary_y_min(v, i):
    # Constraint: y_i - r_i >= 0
    # d/dy_i = 1, d/dr_i = -1
    jac = np.zeros(3 * N_CIRCLES)
    jac[3 * i + 1] = 1.0
    jac[3 * i + 2] = -1.0
    return jac

def jacobian_boundary_y_max(v, i):
    # Constraint: 1 - y_i - r_i >= 0
    # d/dy_i = -1, d/dr_i = -1
    jac = np.zeros(3 * N_CIRCLES)
    jac[3 * i + 1] = -1.0
    jac[3 * i + 2] = -1.0
    return jac

def overlap_constraint_fun(v, i, j):
    # Constraint: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
    xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
    
    dx = xi - xj
    dy = yi - yj
    dr = ri + rj
    
    return (dx*dx + dy*dy - dr*dr)

def overlap_constraint_jac(v, i, j):
    # Gradient of (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2
    jac = np.zeros(3 * N_CIRCLES)
    
    xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
    xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
    
    dx = xi - xj
    dy = yi - yj
    dr = ri + rj
    
    # Partial derivatives for i
    jac[3*i] = 2.0 * dx          # d/dxi
    jac[3*i+1] = 2.0 * dy        # d/dyi
    jac[3*i+2] = -2.0 * dr       # d/dri
    
    # Partial derivatives for j
    jac[3*j] = -2.0 * dx         # d/dxj
    jac[3*j+1] = -2.0 * dy       # d/dyj
    jac[3*j+2] = -2.0 * dr       # d/drj
    
    return jac

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        return False

    if np.isnan(radii).any():
        return False

    # Check if radii are nonnegative
    for i in range(n):
        if radii[i] < 0:
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Use a small epsilon for float comparison
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
            if dist < radii[i] + radii[j] - 1e-9:
                return False

    return True

if __name__ == "__main__":
    # This block is for local testing if needed, 
    # but the function run_packing is the entry point.
    pass
