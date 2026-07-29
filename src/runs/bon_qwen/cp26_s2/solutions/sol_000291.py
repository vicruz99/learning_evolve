# sol_000291 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a4dfceb8) state=174c2b4f sum of radii=2.452410 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def solve_radii_lp(cx, cy):
    """Solve for radii that maximize sum(r) given fixed centers using Linear Programming."""
    n = len(cx)
    # Objective: maximize sum(r) => minimize -sum(r)
    c_obj = -np.ones(n)
    
    # Constraints setup: A_ub @ r <= b_ub
    n_pairs = n * (n - 1) // 2
    n_boundary = 4 * n
    n_cons = n_pairs + n_boundary
    
    A_ub = np.zeros((n_cons, n))
    b_ub = np.zeros(n_cons)
    
    idx = 0
    # Pairwise non-overlap: r_i + r_j <= dist(c_i, c_j)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
            
    # Boundary constraints: r_i <= cx_i, r_i <= 1-cx_i, r_i <= cy_i, r_i <= 1-cy_i
    for i in range(n):
        A_ub[idx, i] = 1.0; b_ub[idx] = cx[i]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - cx[i]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = cy[i]; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - cy[i]; idx += 1
        
    # Solve LP (highs solver is robust and fast for this size)
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
    if res.success:
        return res.x
    return None

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    
    # Multiple restarts to find global optimum
    for restart in range(10):
        np.random.seed(42 + restart)
        # Initial random placement with margin from edges
        cx = np.random.uniform(0.2, 0.8, n)
        cy = np.random.uniform(0.2, 0.8, n)
        
        for step in range(250):
            r = solve_radii_lp(cx, cy)
            if r is None:
                break
                
            curr_sum = np.sum(r)
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = np.column_stack((cx.copy(), cy.copy()))
                best_radii = r.copy()
                
            # Force-directed relaxation for centers
            forces = np.zeros((n, 2))
            for i in range(n):
                for j in range(i + 1, n):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist = np.hypot(dx, dy)
                    if dist > 1e-9:
                        # Push apart if touching or slightly overlapping
                        if dist < r[i] + r[j] + 1e-5:
                            repulsion = 0.015
                            fx = repulsion * dx / dist
                            fy = repulsion * dy / dist
                            forces[i, 0] += fx
                            forces[i, 1] += fy
                            forces[j, 0] -= fx
                            forces[j, 1] -= fy
                            
            # Boundary repulsion to keep circles inside
            for i in range(n):
                if r[i] > cx[i] - 1e-5: forces[i, 0] += 0.015
                if r[i] > 1 - cx[i] - 1e-5: forces[i, 0] -= 0.015
                if r[i] > cy[i] - 1e-5: forces[i, 1] += 0.015
                if r[i] > 1 - cy[i] - 1e-5: forces[i, 1] -= 0.015
                
            cx += forces[:, 0]
            cy += forces[:, 1]
            cx = np.clip(cx, 0.0, 1.0)
            cy = np.clip(cy, 0.0, 1.0)
            
    # Ensure strict non-negativity and valid types
    best_radii = np.maximum(best_radii, 0.0)
    
    return best_centers, best_radii, best_sum
