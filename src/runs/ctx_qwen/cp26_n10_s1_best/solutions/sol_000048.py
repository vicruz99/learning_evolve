# sol_000048 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000029 (state 9a6065a6) state=e87f33be sum of radii=2.620761 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

N = 26

def objective(vars):
    """Minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def constraints(vars):
    """Compute all inequality constraints g(vars) >= 0."""
    xi = vars[0::3]
    yi = vars[1::3]
    ri = vars[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c_boundary = np.concatenate([xi - ri, 1.0 - xi - ri, yi - ri, 1.0 - yi - ri])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = xi[:, None] - xi[None, :]
    dy = yi[:, None] - yi[None, :]
    dr = ri[:, None] + ri[None, :]
    
    i_idx, j_idx = np.tril_indices(N, -1)
    c_overlap = dx[i_idx, j_idx]**2 + dy[i_idx, j_idx]**2 - dr[i_idx, j_idx]**2
    
    return np.concatenate([c_boundary, c_overlap])

def get_initial_guess(seed, angle_deg, r_start):
    """Generate a feasible initial configuration from a rotated hexagonal lattice."""
    np.random.seed(seed)
    centers = []
    y = r_start
    row = 0
    
    # Generate hex grid points
    while len(centers) < N + 5:
        x = r_start if row % 2 == 0 else 2 * r_start
        while x + r_start <= 1.0 and len(centers) < N + 5:
            centers.append([x, y])
            x += 2 * r_start
        y += math.sqrt(3) * r_start
        row += 1
        
    centers = np.array(centers[:N])
    
    # Rotate around square center
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx = centers[:, 0] - 0.5
    dy = centers[:, 1] - 0.5
    centers[:, 0] = dx * cos_a - dy * sin_a + 0.5
    centers[:, 1] = dx * sin_a + dy * cos_a + 0.5
    
    # Add controlled perturbation and clip
    centers += np.random.uniform(-0.01, 0.01, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    # Compute strictly feasible initial radii
    radii = np.zeros(N)
    for i in range(N):
        min_d = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
        for j in range(N):
            if i != j:
                d = math.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if d < min_d:
                    min_d = d
        radii[i] = min_d * 0.55  # Conservative factor guarantees strict feasibility
        
    x0 = np.zeros(3 * N)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = radii
    return x0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_val = -np.inf
    best_vars = None
    
    # Explore rotated lattice alignments
    angles = list(range(0, 41, 5))
    
    for seed in range(35):
        angle = angles[seed % len(angles)]
        r_start = 0.09 + np.random.uniform(0, 0.015)
        x0 = get_initial_guess(seed, angle, r_start)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
            
            if not np.isnan(res.fun):
                curr = -res.fun
                # Strict feasibility check
                c = constraints(res.x)
                if np.min(c) >= -1e-8 and curr > best_val:
                    best_val = curr
                    best_vars = res.x.copy()
        except Exception:
            pass
            
    # Local refinement to escape shallow minima
    if best_vars is not None:
        for _ in range(8):
            x0 = best_vars + np.random.normal(0, 1e-4, 3 * N)
            x0[0::3] = np.clip(x0[0::3], 0.0, 1.0)
            x0[1::3] = np.clip(x0[1::3], 0.0, 1.0)
            x0[2::3] = np.clip(x0[2::3], 0.0, 0.5)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
                
                if not np.isnan(res.fun):
                    curr = -res.fun
                    c = constraints(res.x)
                    if np.min(c) >= -1e-8 and curr > best_val:
                        best_val = curr
                        best_vars = res.x.copy()
            except Exception:
                pass

    # Extract results
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3]
    
    # Deterministic repair to guarantee validator compliance
    for _ in range(50):
        valid = True
        for i in range(N):
            if radii[i] < 0: 
                valid = False; break
            if centers[i,0] - radii[i] < -1e-10 or centers[i,0] + radii[i] > 1.0 + 1e-10: 
                valid = False; break
            if centers[i,1] - radii[i] < -1e-10 or centers[i,1] + radii[i] > 1.0 + 1e-10: 
                valid = False; break
                
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    if math.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1]) < radii[i] + radii[j] - 1e-10:
                        valid = False; break
                if not valid: break
                
        if valid: break
        radii *= 0.999  # Minimal shrinkage to fix numerical drift
        
    return centers, radii, float(np.sum(radii))
