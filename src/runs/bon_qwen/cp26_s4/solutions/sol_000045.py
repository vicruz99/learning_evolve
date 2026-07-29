# sol_000045 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4e4d202b) state=af1019f0 sum of radii=1.336065 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings

warnings.filterwarnings('ignore')

N_CIRCLES = 26

def objective_func(x):
    """Maximize sum of radii (minimize negative sum)"""
    return -np.sum(x[2::3])

def constraint_func(x):
    """Vectorized inequality constraints: boundaries and non-overlap"""
    n = N_CIRCLES
    pts = x.reshape(-1, 3)
    xy = pts[:, :2]
    r = pts[:, 2]
    
    # Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
    b = np.concatenate([
        xy[:, 0] - r,
        1.0 - xy[:, 0] - r,
        xy[:, 1] - r,
        1.0 - xy[:, 1] - r
    ])
    
    # Pairwise non-overlap: dist >= r_i + r_j
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = r[:, None] + r[None, :]
    overlap = dist - r_sum
    
    # Extract strictly upper triangle to avoid duplicates and self-pairs
    triu_idx = np.triu_indices(n, k=1)
    return np.concatenate([b, overlap[triu_idx]])

def compute_lp_radii(centers, n):
    """Solve LP to maximize sum of radii for fixed centers"""
    c_lp = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        x, y = centers[i]
        for _ in range(4):
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
        b_ub.extend([x, 1.0 - x, y, 1.0 - y])
        
    # Pairwise constraints: r_i + r_j <= dist(i,j)
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    bounds_r = [(0.0, None)] * n
    try:
        lp_res = linprog(c_lp, A_ub=np.array(A_ub), b_ub=np.array(b_ub), 
                         bounds=bounds_r, method='highs')
        return lp_res.x if lp_res.success else None
    except Exception:
        return None

def run_packing():
    n = N_CIRCLES
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    # Try multiple random starts to avoid poor local minima
    for trial in range(5):
        centers = np.zeros((n, 2))
        # 5x5 grid initialization
        for i in range(5):
            for j in range(5):
                centers[i*5+j] = [0.1 + 0.2*i, 0.1 + 0.2*j]
        centers[25] = [0.5, 0.5]
        
        # Add perturbation
        centers += np.random.uniform(-0.02, 0.02, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        
        x0 = np.zeros(3*n)
        x0[::3] = centers[:, 0]
        x0[1::3] = centers[:, 1]
        x0[2::3] = 0.15  # Initial radii
        
        bounds = [(0.0, 1.0)]*2*n + [(0.0, 0.5)]*n
        
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, 
                           constraints={'type': 'ineq', 'fun': constraint_func},
                           options={'maxiter': 500, 'ftol': 1e-10, 'disp': False})
            
            opt_centers = res.x[:2*n].reshape(n, 2)
            
            # Refine radii using LP for guaranteed feasibility and optimality
            radii = compute_lp_radii(opt_centers, n)
            if radii is not None:
                current_sum = np.sum(radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = opt_centers.copy()
                    best_radii = radii.copy()
        except Exception:
            continue
            
    # Fallback configuration
    if best_centers is None:
        centers = np.zeros((n, 2))
        for i in range(5):
            for j in range(5):
                centers[i*5+j] = [0.1 + 0.2*i, 0.1 + 0.2*j]
        centers[25] = [0.5, 0.5]
        best_centers = centers
        best_radii = compute_lp_radii(centers, n) or np.full(n, 0.1)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, best_sum
