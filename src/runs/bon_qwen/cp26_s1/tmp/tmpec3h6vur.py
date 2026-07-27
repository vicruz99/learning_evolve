import numpy as np
from scipy.optimize import minimize

def compute_loss(vars, N):
    """
    Computes the negative sum of radii plus penalties for constraint violations.
    """
    centers = vars[:2*N].reshape(N, 2)
    radii = vars[2*N:]
    
    sum_r = np.sum(radii)
    penalty = 0.0
    
    # Boundary constraints: circles must stay inside [0,1]x[0,1]
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        # Distance to each boundary
        for b in [x, 1-x, y, 1-y]:
            if b < r:
                penalty += 1e5 * (r - b)**2
                
    # Overlap constraints: circles must not overlap
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist < radii[i] + radii[j]:
                penalty += 1e5 * (radii[i] + radii[j] - dist)**2
                
    return -sum_r + penalty

def get_initial_guess(N, seed):
    """
    Generates an initial configuration based on a hexagonal lattice.
    """
    np.random.seed(seed)
    pts = []
    spacing = 0.17
    
    # Generate hexagonal grid points
    for row in range(8):
        for col in range(8):
            x = 0.05 + col * spacing + (row % 2) * spacing / 2
            y = 0.05 + row * spacing * np.sqrt(3) / 2
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append([x, y])
                
    # Fallback to random if not enough points (unlikely)
    if len(pts) < N:
        while len(pts) < N:
            pts.append([np.random.rand(), np.random.rand()])
            
    # Shuffle and select exactly N points
    np.random.shuffle(pts)
    pts = pts[:N]
    
    # Initial radius
    r_init = 0.07
    return np.array([p[0] for p in pts] + [p[1] for p in pts] + [r_init]*N)

def run_packing():
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    """
    N = 26
    best_loss = np.inf
    best_vars = None
    
    # Bounds for optimization: x,y in [0,1], r in [0, 0.5]
    bounds = [(0, 1)]*(2*N) + [(0, 0.5)]*N
    
    # Run optimization with multiple random restarts
    for seed in range(15):
        vars0 = get_initial_guess(N, seed)
        res = minimize(compute_loss, vars0, args=(N,), method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 10000, 'ftol': 1e-10, 'gtol': 1e-6})
        if res.fun < best_loss:
            best_loss = res.fun
            best_vars = res.x
            
    centers = best_vars[:2*N].reshape(N, 2)
    radii = best_vars[2*N:]
    
    # Ensure strict validity within floating point tolerance
    radii = np.maximum(radii, 0.0)
    centers = np.clip(centers, 0.0, 1.0)
    
    return centers, radii, np.sum(radii)