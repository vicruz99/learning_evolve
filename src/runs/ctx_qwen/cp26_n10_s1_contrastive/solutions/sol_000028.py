# sol_000028 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000015 (state cc21d5f7) state=84968b47 sum of radii=2.621850 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(x):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def compute_constraints(x):
    """Compute all inequality constraints: boundary and separation."""
    N = 26
    xc = x[0::3]
    yc = x[1::3]
    r = x[2::3]
    
    cons = []
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    cons.append(xc - r)
    cons.append(1.0 - xc - r)
    cons.append(yc - r)
    cons.append(1.0 - yc - r)
    
    # Pairwise non-overlap: dist_sq >= (r_i + r_j)^2
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = xc[i_idx] - xc[j_idx]
    dy = yc[i_idx] - yc[j_idx]
    r_sum = r[i_idx] + r[j_idx]
    cons.append(dx*dx + dy*dy - r_sum*r_sum)
    
    return np.concatenate(cons)

def run_packing():
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_sum = -np.inf
    best_x = None
    
    # Run multiple restarts with diverse initial layouts
    num_restarts = 50
    
    for k in range(num_restarts):
        layout_type = k % 3
        seed = k * 17 + 3
        np.random.seed(seed)
        
        # Generate initial positions based on layout type
        if layout_type == 0:  # Hexagonal lattice
            pts = []
            r_l = 0.04
            dy = np.sqrt(3) * r_l
            dx = 2.0 * r_l
            y = r_l
            row = 0
            while len(pts) < n:
                x_off = r_l if row % 2 == 1 else 0.0
                x = r_l + x_off
                while x <= 1.0 - r_l and len(pts) < n:
                    pts.append([x, y])
                    x += dx
                y += dy
                row += 1
            pts = np.array(pts[:n])
        elif layout_type == 1:  # 5x5 Grid + 1
            pts = []
            for i in range(5):
                for j in range(5):
                    pts.append([0.1 + i*0.2, 0.1 + j*0.2])
            pts.append([0.5, 0.5])
            pts = np.array(pts[:n])
        else:  # Random dense
            pts = np.random.uniform(0.1, 0.9, (n, 2))
            
        # Perturb to break symmetry and ensure strict feasibility
        pts += np.random.uniform(-0.02, 0.02, pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        
        # Small initial radii guarantee feasible start
        r_init = np.full(n, 0.02)
        x0 = np.zeros(3 * n)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = r_init
        
        try:
            res = minimize(
                compute_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 2000, 'ftol': 1e-12}
            )
            
            # Accept if constraints are satisfied (within numerical tolerance)
            if np.min(compute_constraints(res.x)) >= -1e-7:
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_x = res.x.copy()
        except Exception:
            continue
            
    # Fallback strictly feasible configuration if optimization fails entirely
    if best_x is None:
        best_x = np.zeros(3 * n)
        best_x[0::3] = np.tile(np.linspace(0.1, 0.9, 5), 5)
        best_x[1::3] = np.repeat(np.linspace(0.1, 0.9, 5), 5)
        best_x[2::3] = 0.01
        
    # High-precision refinement phase
    # Add tiny perturbation to escape potential shallow local minima
    best_x += np.random.uniform(-1e-5, 1e-5, best_x.shape)
    try:
        res_final = minimize(
            compute_objective,
            best_x,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 5000, 'ftol': 1e-14}
        )
        if np.min(compute_constraints(res_final.x)) >= -1e-8:
            best_x = res_final.x
    except Exception:
        pass
        
    # Extract and format results
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = best_x[3*i]
        centers[i, 1] = best_x[3*i+1]
        radii[i] = best_x[3*i+2]
        
    return centers, radii, float(np.sum(radii))
