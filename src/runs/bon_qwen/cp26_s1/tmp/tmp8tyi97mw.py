import numpy as np
from scipy.optimize import minimize

def compute_loss(params, n, lam):
    """Computes the objective: -sum(radii) + penalty for constraint violations."""
    centers = params[:n*2].reshape(n, 2)
    radii = params[n*2:]
    
    # Base objective: minimize negative sum of radii
    score = -np.sum(radii)
    
    # Penalty for negative radii
    score += lam * np.sum(np.maximum(0, -radii)**2)
    
    # Penalty for boundary violations
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
        v = np.array([r - x, r - (1.0 - x), r - y, r - (1.0 - y)])
        score += lam * np.sum(np.maximum(0, v)**2)
        
    # Penalty for pairwise overlaps
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            ov = radii[i] + radii[j] - d
            if ov > 0:
                score += lam * ov**2
    return score

def check_validity(params, n):
    """Checks if a configuration satisfies all constraints within tolerance."""
    centers = params[:n*2].reshape(n, 2)
    radii = params[n*2:]
    
    if np.any(np.isnan(centers)) or np.any(np.isnan(radii)):
        return False, 0.0
        
    for i in range(n):
        if radii[i] < -1e-10:
            return False, 0.0
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-10 or x + r > 1 + 1e-10 or y - r < -1e-10 or y + r > 1 + 1e-10:
            return False, 0.0
            
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist < radii[i] + radii[j] - 1e-10:
                return False, 0.0
                
    return True, float(np.sum(radii))

def run_packing():
    N = 26
    best_params = None
    best_sum = -1.0
    
    # Bounds for L-BFGS-B
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    
    # Generate initial configurations
    inits = []
    
    # 1. Hexagonal lattice packing
    rows = [6, 5, 6, 5, 4]
    xs, ys = [], []
    y_curr = 0.05
    for idx, cnt in enumerate(rows):
        y = y_curr + idx * 0.18 * 0.866025
        w = (cnt - 1) * 0.16
        xs_row = np.linspace(0.5 - w / 2, 0.5 + w / 2, cnt)
        if idx % 2 == 1:
            xs_row += 0.08
        xs.extend(xs_row)
        ys.extend([y] * cnt)
    inits.append((np.array(xs), np.array(ys)))
    
    # 2. Randomized starts
    for s in [10, 20, 30, 40, 50]:
        np.random.seed(s)
        cx = 0.5 + (np.random.rand(N) - 0.5) * 0.7
        cy = 0.5 + (np.random.rand(N) - 0.5) * 0.7
        inits.append((cx, cy))
        
    for cx, cy in inits:
        p0 = np.concatenate([cx, cy, np.full(N, 0.06)])
        
        try:
            # Pass 1: Find feasible region
            res1 = minimize(compute_loss, p0, args=(N, 1000.0), method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-12})
            
            # Pass 2: Tighten constraints and maximize radii
            res2 = minimize(compute_loss, res1.x, args=(N, 5000.0), method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})
            
            valid, s = check_validity(res2.x, N)
            if valid and s > best_sum:
                best_sum = s
                best_params = res2.x
        except Exception:
            pass
            
    # Fallback initialization if needed
    if best_params is None:
        grid_x, grid_y = np.meshgrid(np.linspace(0.15, 0.85, 6), np.linspace(0.15, 0.85, 6))
        p0 = np.concatenate([grid_x.ravel()[:N], grid_y.ravel()[:N], np.full(N, 0.05)])
        res = minimize(compute_loss, p0, args=(N, 5000.0), method='L-BFGS-B', bounds=bounds)
        best_params = res.x
        best_sum = np.sum(best_params[2*N:])
        
    centers = best_params[:2 * N].reshape(N, 2)
    radii = best_params[2 * N:]
    return centers, radii, float(best_sum)