# sol_000216 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 722eaafb) state=d88cabaf sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def compute_repulsion(centers, radii):
    """Compute pairwise repulsive forces based on overlap depth."""
    n = centers.shape[0]
    forces = np.zeros_like(centers)
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[j] - centers[i]
            dist = np.linalg.norm(diff)
            if dist < 1e-9:
                dist = 1e-9
                diff = np.array([1.0, 0.0])
            
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                # Force proportional to overlap, directed along separation vector
                f = overlap * diff / dist
                forces[i] -= f
                forces[j] += f
    return forces

def apply_boundary_forces(centers, radii, forces):
    """Apply repulsive forces to keep circles inside the unit square."""
    n = centers.shape[0]
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Left boundary
        if x < r:
            forces[i, 0] += (r - x) * 5.0
        # Right boundary
        if x > 1 - r:
            forces[i, 0] -= (x - (1 - r)) * 5.0
        # Bottom boundary
        if y < r:
            forces[i, 1] += (r - y) * 5.0
        # Top boundary
        if y > 1 - r:
            forces[i, 1] -= (y - (1 - r)) * 5.0
    return forces

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    n = 26
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.02)
    
    # Initial structured placement (5x5 grid + 1)
    idx = 0
    for r in range(5):
        for c in range(5):
            if idx < n:
                centers[idx] = [0.1 + c * 0.2, 0.1 + r * 0.2]
                idx += 1
    if idx < n:
        centers[idx] = [0.5, 0.5]
        
    dt = 0.4
    growth = 1.0005
    
    # Main expansion and relaxation loop
    for step in range(5000):
        radii *= growth
        
        forces = compute_repulsion(centers, radii)
        forces = apply_boundary_forces(centers, radii, forces)
        
        centers += forces * dt
        centers = np.clip(centers, radii[:, None], 1 - radii[:, None])
        
        # Escape local minima with occasional random kicks
        if step % 500 == 0:
            noise = np.random.normal(0, 0.005, centers.shape)
            centers += noise
            centers = np.clip(centers, radii[:, None], 1 - radii[:, None])
            
        # Gradually decrease dynamics for convergence
        if step % 200 == 0:
            dt *= 0.95
            growth *= 0.997
            
    # Final strong relaxation to settle positions
    for _ in range(1000):
        forces = compute_repulsion(centers, radii)
        forces = apply_boundary_forces(centers, radii, forces)
        centers += forces * dt * 0.1
        centers = np.clip(centers, radii[:, None], 1 - radii[:, None])
        
    # Safety check: compute minimum clearance to all other circles and boundaries
    min_clearance = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            clearance = dist - radii[i] - radii[j]
            if clearance < min_clearance:
                min_clearance = clearance
                
        for k in range(2):
            clearance = centers[i, k] - radii[i]
            if clearance < min_clearance:
                min_clearance = clearance
            clearance = 1.0 - centers[i, k] - radii[i]
            if clearance < min_clearance:
                min_clearance = clearance
                
    # If numerical overlaps persist, shrink radii uniformly to guarantee validity
    if min_clearance < -1e-9:
        factor = 1.0 + min_clearance / (2.0 * np.max(radii) + 1e-9)
        radii *= factor
        centers = np.clip(centers, radii[:, None], 1 - radii[:, None])
        
    return centers, radii, float(np.sum(radii))
