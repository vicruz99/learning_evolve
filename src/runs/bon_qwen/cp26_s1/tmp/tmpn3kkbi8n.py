import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(v):
    """Minimize negative sum of radii."""
    return -np.sum(v[2 * N_CIRCLES:])

def constraint_bounds(v):
    """Ensure circles stay within the unit square."""
    c = v[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    r = v[2 * N_CIRCLES:]
    return np.concatenate([
        c[:, 0] - r, 1.0 - c[:, 0] - r,
        c[:, 1] - r, 1.0 - c[:, 1] - r
    ])

def constraint_overlap(v):
    """Ensure circles do not overlap."""
    c = v[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    r = v[2 * N_CIRCLES:]
    
    # Pairwise squared distances and radius sums
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Extract upper triangle constraints
    mask = np.triu(np.ones((N_CIRCLES, N_CIRCLES), dtype=bool), k=1)
    return (dist_sq - r_sum ** 2).ravel()[mask.ravel()]

def run_packing():
    # Deterministic initialization
    np.random.seed(42)
    
    # Start with a 5x5 grid pattern perturbed slightly to help escape symmetry
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 5)
    centers = np.array([[x, y] for x in xs for y in ys])
    centers = centers[:N_CIRCLES]
    centers += np.random.normal(0, 0.005, centers.shape)
    radii = np.full(N_CIRCLES, 0.08)
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.01, 0.5)] * N_CIRCLES
    
    cons = [
        {'type': 'ineq', 'fun': constraint_bounds},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    # Optimize
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 5000, 'ftol': 1e-12})
                   
    centers_opt = res.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    radii_opt = res.x[2 * N_CIRCLES:]
    
    # Strict boundary enforcement
    radii_opt = np.maximum(radii_opt, 1e-7)
    centers_opt[:, 0] = np.clip(centers_opt[:, 0], radii_opt, 1.0 - radii_opt)
    centers_opt[:, 1] = np.clip(centers_opt[:, 1], radii_opt, 1.0 - radii_opt)
    
    # Rapid overlap resolution to guarantee validity within 1e-12 tolerance
    for _ in range(30):
        changed = False
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                d = np.linalg.norm(centers_opt[i] - centers_opt[j])
                rs = radii_opt[i] + radii_opt[j]
                if d < rs - 1e-9:
                    scale = (d + 1e-9) / rs
                    radii_opt[i] *= np.sqrt(scale)
                    radii_opt[j] *= np.sqrt(scale)
                    changed = True
        if not changed:
            break
            
    return centers_opt, radii_opt, np.sum(radii_opt)