# sol_000252 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5bb01f44) state=f491403e sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def boundary_constraint_x_lower(v, i):
    return v[3*i] - v[3*i+2]

def boundary_constraint_x_upper(v, i):
    return 1.0 - v[3*i] - v[3*i+2]

def boundary_constraint_y_lower(v, i):
    return v[3*i+1] - v[3*i+2]

def boundary_constraint_y_upper(v, i):
    return 1.0 - v[3*i+1] - v[3*i+2]

def boundary_constraint_radius(v, i):
    return v[3*i+2]

def overlap_constraint(v, i, j):
    x1, y1, r1 = v[3*i], v[3*i+1], v[3*i+2]
    x2, y2, r2 = v[3*j], v[3*j+1], v[3*j+2]
    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
    sum_r_sq = (r1 + r2)**2
    return dist_sq - sum_r_sq

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None

    def objective(v):
        return -np.sum(v[2::3])

    # Precompute constraints list
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': boundary_constraint_x_lower, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': boundary_constraint_x_upper, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': boundary_constraint_y_lower, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': boundary_constraint_y_upper, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': boundary_constraint_radius, 'args': (i,)})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq', 'fun': overlap_constraint, 'args': (i, j)})

    bounds = [(0.0, 1.0) for _ in range(3 * n)]
    for i in range(n):
        bounds[3*i+2] = (0.0, 0.5)

    # Try multiple random seeds / perturbations to find global optimum
    for seed in range(10):
        np.random.seed(42 + seed)
        
        # Initial guess: Hexagonal grid
        pts = []
        r_approx = 0.09 
        spacing = 2 * r_approx
        y = r_approx
        row_idx = 0
        while len(pts) < n:
            if row_idx % 2 == 0:
                x = r_approx
                while x <= 1.0 - r_approx:
                    pts.append((x, y))
                    x += spacing
            else:
                x = r_approx + spacing/2
                while x <= 1.0 - r_approx:
                    pts.append((x, y))
                    x += spacing
            y += spacing * math.sqrt(3) / 2
            row_idx += 1
            if row_idx > 10: break
        
        pts = pts[:n]
        
        v0 = np.zeros(3 * n)
        for i in range(n):
            if i < len(pts):
                v0[3*i] = pts[i][0]
                v0[3*i+1] = pts[i][1]
            else:
                v0[3*i] = np.random.rand()
                v0[3*i+1] = np.random.rand()
            v0[3*i+2] = r_approx
        
        # Add noise to break symmetry and escape local optima
        v0 += np.random.normal(0, 0.005, size=v0.shape)
        
        try:
            res = opt.minimize(
                objective,
                v0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False}
            )
            
            current_sum = -res.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = np.zeros((n, 2))
                best_radii = np.zeros(n)
                for i in range(n):
                    best_centers[i] = [res.x[3*i], res.x[3*i+1]]
                    best_radii[i] = res.x[3*i+2]
        except Exception:
            pass

    if best_centers is None:
        best_centers = np.random.rand(n, 2) * 0.5 + 0.25
        best_radii = np.ones(n) * 0.05
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, float(best_sum)
