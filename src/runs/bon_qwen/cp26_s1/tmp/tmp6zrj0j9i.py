import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
# Precompute indices for upper triangle to efficiently extract pairwise constraints
_MASK_INDICES = tuple(np.triu_indices(N_CIRCLES, k=1))

def compute_constraints(v):
    """
    Computes all constraint violations for the current configuration.
    Returns an array where each element corresponds to a constraint value that must be >= 0.
    """
    centers = np.column_stack((v[0::3], v[1::3]))
    r = v[2::3]
    
    cons = []
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    cons.append(centers[:, 0] - r)
    cons.append(1.0 - centers[:, 0] - r)
    cons.append(centers[:, 1] - r)
    cons.append(1.0 - centers[:, 1] - r)
    
    # Non-overlap constraints: dist(i,j) - r_i - r_j >= 0 for all i < j
    dists = np.sqrt(np.sum((centers[:, None] - centers[None, :])**2, axis=2))
    cons.append(dists[_MASK_INDICES] - r[_MASK_INDICES[0]] - r[_MASK_INDICES[1]])
    
    return np.concatenate(cons)

def objective(v):
    """Objective function: minimize negative sum of radii (equivalent to maximizing sum)"""
    return -np.sum(v[2::3])

def run_packing():
    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * (2 * N_CIRCLES) + [(0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_sum = 0.0
    best_v = None
    
    inits = []
    
    # 1. Hexagonal-like grid initialization (promotes dense packing)
    hex_pts = []
    y = 0.12
    row_parity = 0
    while len(hex_pts) < N_CIRCLES:
        x = 0.12 + (0.11 if row_parity else 0)
        while x < 0.88 and len(hex_pts) < N_CIRCLES:
            hex_pts.append([x, y])
            x += 0.22
        y += 0.19
        row_parity = 1 - row_parity
    inits.append(np.array(hex_pts[:N_CIRCLES]))
    
    # 2. Multiple random initializations to escape local optima
    for seed in [1, 42, 123, 456, 789, 1024]:
        np.random.seed(seed)
        inits.append(np.random.rand(N_CIRCLES, 2))
        
    for cfg in inits:
        v0 = np.zeros(3 * N_CIRCLES)
        v0[0::3] = cfg[:, 0]
        v0[1::3] = cfg[:, 1]
        v0[2::3] = 0.04  # Start with small radii to ensure initial feasibility
        
        # Clamp initial radii to strictly satisfy boundary constraints
        for k in range(N_CIRCLES):
            max_r = min(v0[3*k], 1.0 - v0[3*k], v0[3*k+1], 1.0 - v0[3*k+1])
            v0[3*k+2] = min(max_r, 0.04)
            
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-10})
            if res.success:
                s = np.sum(res.x[2::3])
                if s > best_sum:
                    c = np.column_stack((res.x[0::3], res.x[1::3]))
                    r = res.x[2::3]
                    # Verify against the provided validation function
                    if validate_packing(c, r):
                        best_sum = s
                        best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is not None:
        c = np.column_stack((best_v[0::3], best_v[1::3]))
        r = best_v[2::3]
        return c, r, np.sum(r)
    else:
        # Fallback configuration (should not be reached given robust initialization)
        c = np.column_stack((np.random.rand(N_CIRCLES), np.random.rand(N_CIRCLES)))
        r = np.ones(N_CIRCLES) * 0.01
        return c, r, 0.26