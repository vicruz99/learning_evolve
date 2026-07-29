# sol_000224 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state df9a626f) state=b8be346f sum of radii=2.340535 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(x, n):
    """Maximize sum of radii -> minimize negative sum"""
    return -np.sum(x[2::3])

def make_boundary_constraints(i):
    """Factory for boundary constraint functions for circle i"""
    def c1(x):
        return x[3*i] - x[3*i+2]
    def c2(x):
        return 1.0 - x[3*i] - x[3*i+2]
    def c3(x):
        return x[3*i+1] - x[3*i+2]
    def c4(x):
        return 1.0 - x[3*i+1] - x[3*i+2]
    return [
        {'type': 'ineq', 'fun': c1},
        {'type': 'ineq', 'fun': c2},
        {'type': 'ineq', 'fun': c3},
        {'type': 'ineq', 'fun': c4}
    ]

def make_pair_constraint(i, j):
    """Factory for non-overlap constraint function between circles i and j"""
    def c(x):
        dx = x[3*i] - x[3*j]
        dy = x[3*i+1] - x[3*j+1]
        dr = x[3*i+2] + x[3*j+2]
        return dx*dx + dy*dy - dr*dr
    return {'type': 'ineq', 'fun': c}

def build_all_constraints(n):
    """Constructs the full list of constraints for N circles"""
    constraints = []
    for i in range(n):
        constraints.extend(make_boundary_constraints(i))
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(make_pair_constraint(i, j))
    return constraints

def compute_exact_radii(centers):
    """Computes the maximum feasible radii for given centers"""
    n = centers.shape[0]
    radii = np.empty(n)
    for i in range(n):
        # Distance to boundaries
        r = min(centers[i, 0], 1.0 - centers[i, 0], 
                centers[i, 1], 1.0 - centers[i, 1])
        # Half distance to nearest neighbor
        for j in range(n):
            if i == j:
                continue
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist / 2.0 < r:
                r = dist / 2.0
        radii[i] = r
    return radii

def run_packing():
    n = 26
    
    # 1. Initialize on a hexagonal lattice
    sp = 0.22
    pts = []
    for row in range(8):
        for col in range(8):
            x = col * sp + (row % 2) * sp / 2.0
            y = row * sp * np.sqrt(3) / 2.0
            if x < 1.0 and y < 1.0:
                pts.append((x, y))
    pts = np.array(pts[:n])
    
    # Flatten initial state: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = pts[i, 0]
        x0[3*i+1] = pts[i, 1]
        x0[3*i+2] = 0.01  # Small initial radius to start feasible
    
    # Bounds for each variable
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)] * n
    
    # Constraints
    constraints = build_all_constraints(n)
    
    # 2. Optimize using SLSQP
    res = minimize(
        objective, 
        x0, 
        args=(n,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False}
    )
    
    # 3. Extract and post-process
    centers = res.x[:3*n].reshape(n, 3)[:, :2]
    radii = compute_exact_radii(centers)
    total_sum = float(np.sum(radii))
    
    return centers, radii, total_sum
