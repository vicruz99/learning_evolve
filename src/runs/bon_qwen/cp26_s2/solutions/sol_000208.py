# sol_000208 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8fb167b6) state=c1118d9c sum of radii=0.815557 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def relax_positions(centers, radii, n, max_iter=1000, tol=1e-8):
    """
    Iteratively resolves overlaps and boundary violations by pushing circles apart.
    Modifies centers in place and returns them for convenience.
    """
    for _ in range(max_iter):
        max_ov = 0.0
        
        # Pairwise overlap resolution
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                
                # Avoid division by zero
                if dist_sq < 1e-16:
                    dist = 1e-8
                    nx, ny = 0.0, 0.0
                else:
                    dist = np.sqrt(dist_sq)
                    nx, ny = dx / dist, dy / dist
                    
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    if overlap > max_ov:
                        max_ov = overlap
                    move = overlap * 0.5
                    centers[i, 0] += nx * move
                    centers[i, 1] += ny * move
                    centers[j, 0] -= nx * move
                    centers[j, 1] -= ny * move
        
        # Boundary constraint enforcement
        for i in range(n):
            if centers[i, 0] < radii[i]:
                centers[i, 0] = radii[i]
            elif centers[i, 0] > 1.0 - radii[i]:
                centers[i, 0] = 1.0 - radii[i]
                
            if centers[i, 1] < radii[i]:
                centers[i, 1] = radii[i]
            elif centers[i, 1] > 1.0 - radii[i]:
                centers[i, 1] = 1.0 - radii[i]
                
        if max_ov < tol:
            break
            
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    
    # Hexagonal initial placement: 5 rows with counts [6, 5, 6, 5, 4]
    counts = [6, 5, 6, 5, 4]
    dx = 0.14
    dy = 0.12
    x_min = (1.0 - 6 * dx) / 2.0
    y_min = (1.0 - 5 * dy) / 2.0
    
    for r_idx, count in enumerate(counts):
        y = y_min + r_idx * dy + 0.02
        x_start = x_min + (r_idx % 2) * (dx / 2.0)
        for c in range(count):
            centers[idx] = [x_start + c * dx, y]
            idx += 1
            
    # Deterministic perturbation to break symmetry and aid optimization
    np.random.seed(42)
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    radii = np.full(n, 0.02)
    
    # Phase 1: Controlled Growth & Relaxation
    # Radii grow slowly while positions adjust to minimize overlaps
    growth_steps = 3000
    for step in range(growth_steps):
        growth = 1.0 + 0.0002 * (1.0 - step / growth_steps)
        radii *= growth
        relax_positions(centers, radii, n, max_iter=20, tol=1e-5)
        
    # Phase 2: Fine-Tuning
    # Scale radii up uniformly and relax aggressively to reach packing limits
    for _ in range(300):
        radii *= 1.0005
        relax_positions(centers, radii, n, max_iter=100, tol=1e-6)
        
    # Phase 3: Strict Final Relaxation
    # Ensure all constraints are satisfied within tight numerical tolerance
    relax_positions(centers, radii, n, max_iter=5000, tol=1e-9)
    
    # Safety clipping
    centers = np.clip(centers, 1e-7, 1.0 - 1e-7)
    radii = np.maximum(radii, 1e-7)
    
    return centers, radii, np.sum(radii)
