# sol_000269 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e18200b9) state=5b238a1f sum of radii=2.410953 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    np.random.seed(42)
    
    # --- Stage 1: Initialize with a dense hexagonal lattice ---
    # Rows configuration: 5, 6, 5, 6, 4 circles to sum to 26
    row_counts = [5, 6, 5, 6, 4]
    y_positions = np.linspace(0.08, 0.92, 5)
    
    centers = []
    r_init = 0.08
    
    for i, count in enumerate(row_counts):
        # Shift x-coordinates for even-indexed rows to create hexagonal packing
        shift = r_init * 0.5 if i % 2 == 1 else 0.0
        x_positions = np.linspace(shift + r_init, 1 - shift - r_init, count)
        for x in x_positions:
            centers.append([x, y_positions[i]])
            
    centers = np.array(centers)
    radii = np.full(n, r_init)
    
    # --- Stage 2: Numerical Optimization ---
    # Variables: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.5)] * n
    
    # Non-overlap constraints: dist >= r_i + r_j
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def make_constraint(i, j):
                def constraint(vars):
                    x_i, y_i, r_i = vars[3*i], vars[3*i+1], vars[3*i+2]
                    x_j, y_j, r_j = vars[3*j], vars[3*j+1], vars[3*j+2]
                    dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                    return dist - (r_i + r_j)
                return constraint
            cons.append({'type': 'ineq', 'fun': make_constraint(i, j)})
            
        # Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
        def make_boundary_constraints(k):
            c = []
            def c1(vars): return vars[3*k] - vars[3*k+2]
            def c2(vars): return 1 - vars[3*k] - vars[3*k+2]
            def c3(vars): return vars[3*k+1] - vars[3*k+2]
            def c4(vars): return 1 - vars[3*k+1] - vars[3*k+2]
            return [{'type': 'ineq', 'fun': c1}, {'type': 'ineq', 'fun': c2},
                    {'type': 'ineq', 'fun': c3}, {'type': 'ineq', 'fun': c4}]
        cons.extend(make_boundary_constraints(i))

    res = minimize(lambda v: -sum(v[3*i+2] for i in range(n)), x0, 
                   method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 1000, 'ftol': 1e-12})
    
    # Extract optimized values
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    for i in range(n):
        final_centers[i] = [res.x[3*i], res.x[3*i+1]]
        final_radii[i] = res.x[3*i+2]
        
    # Ensure non-negative radii
    final_radii = np.maximum(final_radii, 0)
    
    return final_centers, final_radii, float(np.sum(final_radii))
