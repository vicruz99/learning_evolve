# sol_000261 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8d1f387b) state=f2e4896d sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def objective_unequal(v, n):
    """Objective function to minimize: -sum(radii)"""
    r_sum = 0.0
    for i in range(n):
        r_sum += v[3*i + 2]
    return -r_sum

def make_bounds(n):
    """Create bounds for variables [x0, y0, r0, ..., x25, y25, r25]"""
    bounds = []
    for i in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
    return bounds

def make_constraints(n):
    """Create constraints for boundary and non-overlap"""
    cons = []
    
    # Boundary constraints
    # x_i >= r_i  => x_i - r_i >= 0
    # 1 - x_i >= r_i => 1 - x_i - r_i >= 0
    # y_i >= r_i
    # 1 - y_i >= r_i
    for i in range(n):
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        # 1 - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # 1 - y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        
    # Overlap constraints
    # dist^2 >= (r_i + r_j)^2
    # dist^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            def overlap_constraint(v, i=i, j=j):
                c1 = np.array([v[3*i], v[3*i+1]])
                c2 = np.array([v[3*j], v[3*j+1]])
                r1 = v[3*i+2]
                r2 = v[3*j+2]
                dist_sq = np.sum((c1 - c2)**2)
                return dist_sq - (r1 + r2)**2
            cons.append({'type': 'ineq', 'fun': overlap_constraint})
            
    return cons

def get_initial_guess(n):
    """Generate initial guess for centers and radii"""
    # We try to fit n circles in a hexagonal pattern
    # Start with a dense grid and pick points
    
    # Generate hexagonal lattice points
    lattice = []
    # Parameters for lattice
    # We want points to be somewhat spread out
    # Let's use a spacing that allows ~26 points
    # A 5x5 grid has 25 points.
    # Let's try to generate a grid of 6x5 (30 points) and remove 4.
    
    points = []
    # 6 rows, 5 cols
    # Row spacing: roughly 1/6 = 0.166
    # Col spacing: roughly 1/5 = 0.2
    # But hexagonal shift
    
    # Let's just create a 6x5 grid with hexagonal shift
    # Rows 0..5
    # Cols 0..4
    # x = c * 0.2 + 0.1
    # y = r * 0.16 + 0.08
    # Shift odd rows by 0.1
    
    pts = []
    for r_idx in range(6):
        for c_idx in range(5):
            x = c_idx * 0.2 + 0.1
            y = r_idx * 0.16 + 0.08
            if r_idx % 2 == 1:
                x += 0.1
            pts.append([x, y])
    
    # We have 30 points. Take first 26.
    # To make them valid initial radii, set small radius.
    # But to help optimization, maybe set radius based on nearest neighbor distance?
    # Just set 0.05.
    
    selected_pts = pts[:n]
    centers = np.array(selected_pts)
    radii = np.full(n, 0.05)
    
    # Flatten
    v = np.zeros(3 * n)
    for i in range(n):
        v[3*i] = centers[i, 0]
        v[3*i+1] = centers[i, 1]
        v[3*i+2] = radii[i]
        
    return v

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Get initial guess
    v0 = get_initial_guess(n)
    
    # Define problem
    bounds = make_bounds(n)
    constraints = make_constraints(n)
    
    # Optimize
    # We might need to run multiple times or use a robust method.
    # SLSQP is good.
    
    # Try optimization
    res = minimize(
        objective_unequal,
        v0,
        args=(n,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
    )
    
    if res.success or res.fun < 0: # Fun is negative sum
        # Extract results
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i, 0] = res.x[3*i]
            centers[i, 1] = res.x[3*i+1]
            radii[i] = res.x[3*i+2]
            
        sum_radii = np.sum(radii)
        
        # Validate locally (optional but good for debugging)
        # But we trust the constraints.
        
        return centers, radii, sum_radii
    else:
        # Fallback to a valid packing if optimization fails
        # 5x5 grid of radius 0.1 + 1 small circle
        centers_fb = np.zeros((n, 2))
        radii_fb = np.zeros(n)
        
        # 25 circles in 5x5 grid
        idx = 0
        for r in range(5):
            for c in range(5):
                centers_fb[idx, 0] = 0.1 + c * 0.2
                centers_fb[idx, 1] = 0.1 + r * 0.2
                radii_fb[idx] = 0.1
                idx += 1
        
        # 26th circle: place in a gap?
        # Gap at (0.2, 0.2) is surrounded by (0.1,0.1), (0.3,0.1), (0.1,0.3), (0.3,0.3).
        # Distance to (0.1, 0.1) is sqrt(0.01+0.01) = 0.1414.
        # Max radius = 0.1414 - 0.1 = 0.0414.
        # Let's place at (0.2, 0.2) with r=0.04.
        centers_fb[idx, 0] = 0.2
        centers_fb[idx, 1] = 0.2
        radii_fb[idx] = 0.04
        
        sum_fb = np.sum(radii_fb)
        return centers_fb, radii_fb, sum_fb
