import numpy as np
import scipy.optimize as opt

def objective(vars):
    """Objective: maximize sum of radii (minimize negative sum)"""
    return -np.sum(vars[2::3])

def constr_boundary(vars, n):
    """Boundary constraints: circles must lie within [0,1]x[0,1]"""
    pts = vars.reshape(-1, 3)
    x, y, r = pts[:, 0], pts[:, 1], pts[:, 2]
    # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    return np.hstack([x - r, 1 - x - r, y - r, 1 - y - r])

def constr_overlap(vars, n):
    """Non-overlap constraints: squared distance >= squared sum of radii"""
    pts = vars.reshape(-1, 3)
    dx = pts[:, 0, np.newaxis] - pts[np.newaxis, :, 0]
    dy = pts[:, 1, np.newaxis] - pts[np.newaxis, :, 1]
    dr = pts[:, 2, np.newaxis] + pts[np.newaxis, :, 2]
    dist_sq = dx**2 + dy**2
    rad_sum_sq = dr**2
    
    # Extract only upper triangular pairs (i < j) to avoid duplicates and self-comparison
    iu, jv = np.triu_indices(n, k=1)
    return (dist_sq[iu, jv] - rad_sum_sq[iu, jv])

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Variable bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    # Constraint objects
    bc = opt.NonlinearConstraint(constr_boundary, 0, np.inf, args=(n,))
    oc = opt.NonlinearConstraint(constr_overlap, 0, np.inf, args=(n,))
    cons = [bc, oc]
    
    # Multi-start optimization to escape local minima
    for seed in range(8):
        rng = np.random.RandomState(seed)
        
        # Initial guess: structured 5x5 grid with small random perturbation
        x_grid = np.linspace(0.1, 0.9, 5)
        y_grid = np.linspace(0.1, 0.9, 5)
        cx, cy = np.meshgrid(x_grid, y_grid)
        centers_init = np.vstack([cx.ravel(), cy.ravel()]).T[:n]
        
        # Perturb and clip to ensure initial feasibility
        centers_init += rng.uniform(-0.012, 0.012, centers_init.shape)
        centers_init = np.clip(centers_init, 0.05, 0.95)
        
        radii_init = np.ones(n) * 0.04
        x0 = np.hstack([centers_init.ravel(), radii_init])
        
        try:
            res = opt.minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons, 
                options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False}
            )
            if res.success:
                curr_sum = np.sum(res.x[2::3])
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_centers = res.x.reshape(-1, 3)[:, :2]
                    best_radii = res.x.reshape(-1, 3)[:, 2]
        except Exception:
            continue
            
    # Fallback to a known valid grid packing if optimization fails
    if best_centers is None:
        centers = np.linspace(0.1, 0.9, 5).repeat(5).reshape(-1, 1)
        centers = np.hstack([centers, np.tile(np.linspace(0.1, 0.9, 5), 5).reshape(-1, 1)][:26])
        radii = np.ones(26) * 0.09
        return centers, radii, np.sum(radii)
        
    return best_centers, best_radii, best_sum