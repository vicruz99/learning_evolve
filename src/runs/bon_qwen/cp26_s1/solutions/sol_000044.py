# sol_000044 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ff99986a) state=d264f398 sum of radii=2.235828 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np


def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    Uses force-based iterative optimization with hexagonal initialization.
    """
    n = 26
    
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    # Try multiple starting configurations
    for seed_idx in range(8):
        centers, radii = initialize_packing(n, seed_idx)
        centers, radii = optimize_packing(centers, radii, n)
        
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
    
    return best_centers, best_radii, best_sum


def initialize_packing(n: int, seed_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Initialize circle positions using various patterns."""
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.04
    
    np.random.seed(seed_idx)
    
    # Pattern 0: Hexagonal 5-row pattern (6,5,6,5,4)
    # Pattern 1: Grid-based 5x5 + 1
    # Pattern 2-7: Perturbed versions of patterns 0 and 1
    
    use_hex = seed_idx % 2 == 0
    
    if use_hex:
        centers, radii, _ = hexagonal_init(n, seed_idx)
    else:
        centers, radii, _ = grid_init(n, seed_idx)
    
    return centers, radii


def hexagonal_init(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Hexagonal pattern initialization."""
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.05
    
    row_configs = [
        (6, 0.10, 0.07),
        (5, 0.24, 0.15),
        (6, 0.38, 0.07),
        (5, 0.52, 0.15),
        (4, 0.66, 0.07)
    ]
    
    idx = 0
    for ncols, y_base, x_off in row_configs:
        for c in range(ncols):
            if idx >= n:
                break
            x = x_off + c * 0.17
            if seed > 0:
                x += np.random.uniform(-0.03, 0.03)
                y_base += np.random.uniform(-0.03, 0.03)
            x = max(0.05, min(0.95, x))
            y_base = max(0.05, min(0.95, y_base))
            centers[idx] = [x, y_base]
            idx += 1
    
    return centers, radii, 0.0


def grid_init(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Grid-based initialization."""
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.05
    
    # 5x5 grid + 1 extra circle
    idx = 0
    for row in range(5):
        for col in range(5):
            if idx >= n:
                break
            x = 0.1 + col * 0.2
            y = 0.1 + row * 0.2
            if seed > 0:
                x += np.random.uniform(-0.02, 0.02)
                y += np.random.uniform(-0.02, 0.02)
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            centers[idx] = [x, y]
            idx += 1
    
    # Place extra circle in a gap
    if idx < n:
        centers[idx] = [0.95, 0.1]
        idx += 1
    
    return centers, radii, 0.0


def optimize_packing(centers: np.ndarray, radii: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Force-based optimization to maximize sum of radii."""
    
    for outer_iter in range(3000):
        # Phase 1: Expand each radius to maximum possible given current positions
        for i in range(n):
            max_r = min(
                centers[i, 0], 
                1 - centers[i, 0],
                centers[i, 1], 
                1 - centers[i, 1]
            )
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, dist - radii[j])
            radii[i] = max(max_r, 1e-10)
        
        # Phase 2: Resolve overlaps with repulsion forces
        for _ in range(80):
            forces = np.zeros_like(centers)
            max_overlap = 0.0
            
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[j, 0] - centers[i, 0]
                    dy = centers[j, 1] - centers[i, 1]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq)
                    req = radii[i] + radii[j]
                    
                    if dist < req and dist > 1e-10:
                        overlap = req - dist
                        nx = dx / dist
                        ny = dy / dist
                        f = overlap * 0.4
                        forces[i, 0] -= nx * f
                        forces[i, 1] -= ny * f
                        forces[j, 0] += nx * f
                        forces[j, 1] += ny * f
                        max_overlap = max(max_overlap, overlap)
            
            # Adaptive step size
            step = 0.03
            if outer_iter > 2000:
                step = 0.01
            centers = centers + forces * step
            
            # Enforce boundaries strictly
            for i in range(n):
                r = radii[i]
                centers[i, 0] = max(r + 1e-12, min(1.0 - r - 1e-12, centers[i, 0]))
                centers[i, 1] = max(r + 1e-12, min(1.0 - r - 1e-12, centers[i, 1]))
            
            if max_overlap < 1e-10:
                break
    
    # Final cleanup: ensure no overlaps by computing exact max radii
    for i in range(n):
        max_r = min(
            centers[i, 0], 
            1 - centers[i, 0],
            centers[i, 1], 
            1 - centers[i, 1]
        )
        for j in range(n):
            if i == j:
                continue
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
        radii[i] = max(max_r, 1e-10)
    
    # Adjust centers to be exactly at valid positions
    for i in range(n):
        r = radii[i]
        centers[i, 0] = max(r + 1e-12, min(1.0 - r - 1e-12, centers[i, 0]))
        centers[i, 1] = max(r + 1e-12, min(1.0 - r - 1e-12, centers[i, 1]))
    
    return centers, radii
