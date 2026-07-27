import numpy as np
from scipy.optimize import minimize

def compute_objective(vars, W):
    """
    Computes the objective function: -sum(radii) + penalty term.
    Vectorized for performance.
    """
    n = 26
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    
    obj = -np.sum(radii)
    
    # Boundary penalties: circles must be inside [0,1]x[0,1]
    # x >= r  => r - x <= 0
    p1 = np.maximum(0, radii - centers[:,0])**2
    # x <= 1-r => x + r - 1 <= 0
    p2 = np.maximum(0, centers[:,0] + radii - 1)**2
    # y >= r
    p3 = np.maximum(0, radii - centers[:,1])**2
    # y <= 1-r
    p4 = np.maximum(0, centers[:,1] + radii - 1)**2
    obj += W * np.sum(p1 + p2 + p3 + p4)
    
    # Overlap penalties: dist(i,j) >= r_i + r_j
    # Compute all pairwise distances efficiently
    diff = centers[:, None] - centers[None, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf) # Ignore self-distance
    
    sum_radii_pair = radii[:, None] + radii[None, :]
    violations = np.maximum(0, sum_radii_pair - dists)
    
    # Sum of squared violations (counts each pair twice due to symmetry, which is fine)
    obj += W * np.sum(violations**2)
    
    return obj

def post_optimize_scaling(centers, radii):
    """
    Uniformly scales up radii by the maximum allowable slack margin.
    """
    n = len(radii)
    min_slack = 1.0
    
    # Check boundary slacks
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        min_slack = min(min_slack, x - r, 1 - x - r, y - r, 1 - y - r)
        
    # Check overlap slacks
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1])
            # Slack decreases by 2*delta for overlaps
            min_slack = min(min_slack, (d - radii[i] - radii[j]) / 2)
            
    if min_slack > 1e-9:
        radii = radii + min_slack
        
    return centers, radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Prepare initial configurations
    inits = []
    
    # 1. Grid pattern
    coords = np.linspace(0.1, 0.9, 5)
    gc = np.array([(x, y) for x in coords for y in coords])[:n]
    inits.append((gc, np.full(n, 0.05)))
    
    # 2. Hexagonal pattern
    hc = []
    y = 0.1
    while len(hc) < n and y <= 0.9:
        x = 0.1
        shift = 0.1 if len(hc) % 2 == 1 else 0.0
        while x + shift <= 0.9:
            if len(hc) < n:
                hc.append([x + shift, y])
            x += 0.2
        y += 0.173205 # sqrt(3)/2 * 0.2
    hc = np.array(hc[:n])
    inits.append((hc, np.full(n, 0.08)))
    
    # 3. Random patterns
    np.random.seed(42)
    for _ in range(10):
        rc = np.random.rand(n, 2) * 0.6 + 0.2
        rr = np.full(n, 0.05)
        inits.append((rc, rr))
        
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    weights = [100, 500, 2000, 10000]
    
    for ic, ir in inits:
        vars0 = np.concatenate([ic.flatten(), ir])
        current_vars = vars0.copy()
        
        # Iteratively increase penalty weight
        for W in weights:
            res = minimize(
                compute_objective,
                current_vars,
                args=(W,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            current_vars = res.x
            
        co = current_vars[:2*n].reshape(n, 2)
        ro = current_vars[2*n:]
        
        # Validate constraints
        valid = True
        for i in range(n):
            x, y = co[i]
            r = ro[i]
            if x < r - 1e-7 or x > 1 - r - 1e-7 or y < r - 1e-7 or y > 1 - r - 1e-7:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.hypot(co[i,0] - co[j,0], co[i,1] - co[j,1])
                    if d < ro[i] + ro[j] - 1e-7:
                        valid = False
                        break
                if not valid:
                    break
                    
        if valid:
            current_sum = np.sum(ro)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = co.copy()
                best_radii = ro.copy()
                
    # Final polishing step: scale up radii by slack margin
    if best_centers is not None:
        best_centers, best_radii = post_optimize_scaling(best_centers, best_radii)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, best_sum