# sol_000073 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ed1177e6) state=c8904ee5 sum of radii=1.780575 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def generate_initial_centers(n):
    """
    Generates a staggered hexagonal-like initial guess for n centers.
    """
    centers = []
    # Approximate number of rows
    # For n=26, we can try a pattern like 6, 5, 6, 5, 4
    rows = [6, 5, 6, 5, 4]
    current_n = 0
    
    y_spacing = 1.0 / 6.0
    y_current = y_spacing # Start a bit away from bottom
    
    for i, count in enumerate(rows):
        x_spacing = 1.0 / (count + 1)
        # Stagger offset
        offset = 0.0 if i % 2 == 0 else x_spacing / 2.0
        
        for j in range(count):
            if current_n >= n:
                break
            x = (j + 1) * x_spacing + offset
            # Keep within [0,1]
            x = np.clip(x, 0.1, 0.9)
            y = y_current
            centers.append([x, y])
            current_n += 1
        y_current += y_spacing * 1.5 # Vertical spacing for staggered rows
        
    if len(centers) < n:
        # Fill remaining randomly if pattern fell short
        while len(centers) < n:
            centers.append([np.random.rand(), np.random.rand()])
            
    return np.array(centers[:n])

def solve_radii_lp(centers):
    """
    Given fixed centers, solve the LP to maximize sum of radii.
    Maximize sum(r_i)
    Subject to:
    r_i >= 0
    r_i <= x_i
    r_i <= 1 - x_i
    r_i <= y_i
    r_i <= 1 - y_i
    r_i + r_j <= ||c_i - c_j|| for all i < j
    """
    n = centers.shape[0]
    
    # Variables: r_0, ..., r_{n-1}
    # Objective: Maximize sum(r) => Minimize -sum(r)
    c_obj = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # We need to collect all linear constraints.
    
    rows_A = []
    rows_b = []
    
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        # r_i <= x  => r_i - x <= 0
        row = np.zeros(n)
        row[i] = 1.0
        rows_A.append(row)
        rows_b.append(x)
        
        # r_i <= 1-x => r_i <= 1-x
        rows_A.append(row)
        rows_b.append(1.0 - x)
        
        # r_i <= y
        rows_A.append(row)
        rows_b.append(y)
        
        # r_i <= 1-y
        rows_A.append(row)
        rows_b.append(1.0 - y)
        
    # Pairwise constraints: r_i + r_j <= dist_ij
    # Precompute distances to avoid recomputing in loop if possible, 
    # but for n=26 it's fast enough to do on fly or precompute.
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            dists[i, j] = d
            dists[j, i] = d
            
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows_A.append(row)
            rows_b.append(d)
            
    A_ub = np.array(rows_A)
    b_ub = np.array(rows_b)
    
    # Bounds for r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    # method='highs' is robust
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return -res.fun, res.x # Return sum of radii and radii
    else:
        # Fallback to small radii if LP fails (shouldn't happen)
        return 0.0, np.zeros(n)

def compute_force_gradient(centers, radii):
    """
    Computes a heuristic force to move centers to increase the sum of radii.
    The idea: if r_i is constrained by neighbor j, moving i away from j helps.
    Force is sum of vectors pointing away from active constraints.
    """
    n = centers.shape[0]
    forces = np.zeros_like(centers)
    
    # Soften tolerance for active constraints
    tol = 1e-4
    
    for i in range(n):
        # Boundary gradients
        # If r_i is close to boundary constraint, push away from boundary
        x, y = centers[i]
        r = radii[i]
        
        # Check lower x
        if r > x - tol:
            forces[i, 0] -= 1.0
        # Check upper x
        if r > (1.0 - x) - tol:
            forces[i, 0] += 1.0
        # Check lower y
        if r > y - tol:
            forces[i, 1] -= 1.0
        # Check upper y
        if r > (1.0 - y) - tol:
            forces[i, 1] += 1.0
            
        # Pairwise gradients
        for j in range(n):
            if i == j: continue
            # Constraint: r_i + r_j <= dist
            # If active, force i away from j
            # Distance
            diff = centers[i] - centers[j]
            dist = np.sqrt(np.sum(diff**2))
            if dist < 1e-9: dist = 1e-9
            unit_vec = diff / dist
            
            if (radii[i] + radii[j]) > dist - tol:
                # Repel
                forces[i] += unit_vec
                
    return forces

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    
    # 1. Initial Configuration
    centers = generate_initial_centers(n)
    
    # 2. Iterative Optimization
    # We perform a loop of: Solve LP for radii -> Compute Forces -> Update Centers
    best_sum_r = 0.0
    best_centers = centers.copy()
    best_radii = np.zeros(n)
    
    # Parameters for optimization
    lr = 0.01 # Learning rate for position updates
    max_iter = 200
    
    for step in range(max_iter):
        # Solve for optimal radii given current centers
        sum_r, radii = solve_radii_lp(centers)
        
        if sum_r > best_sum_r:
            best_sum_r = sum_r
            best_centers = centers.copy()
            best_radii = radii.copy()
            
        # Compute forces to improve positions
        forces = compute_force_gradient(centers, radii)
        
        # Apply forces (Gradient Ascent on sum of radii via position change)
        # Normalize forces to prevent exploding steps
        max_f = np.max(np.abs(forces))
        if max_f > 1e-6:
            forces = forces / max_f
            
        centers += forces * lr
        
        # Keep centers strictly inside [0, 1] with some margin to allow radii
        # But actually centers can be anywhere, LP handles radii limits.
        # However, to keep radii positive, centers shouldn't be exactly at boundary if radius > 0.
        # We clip to [epsilon, 1-epsilon]
        eps = 1e-4
        centers = np.clip(centers, eps, 1.0 - eps)
        
        # Decay learning rate
        lr *= 0.98
        
    # Final verification and cleanup
    # Ensure radii are valid for the final centers (LP guarantees this, but numerical noise exists)
    final_sum, final_radii = solve_radii_lp(best_centers)
    
    # Sanity check and fix for any tiny violations
    # If LP was solved correctly, violations shouldn't exist.
    
    return best_centers, final_radii, final_sum

# Helper to ensure no closures
def dummy_helper():
    pass
