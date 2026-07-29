# sol_000008 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5f9158d7) state=f165f31f sum of radii=1.231052 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    n = 26
    # Initialize positions in a hexagonal pattern
    pos = np.zeros((n, 2))
    idx = 0
    y_start = 0.08
    dy = 0.173205
    x_start = 0.08
    dx = 0.16
    row = 0
    
    while idx < n:
        y = y_start + row * dy
        num_cols = 5 if row % 2 == 0 else 4
        for col in range(num_cols):
            if idx >= n:
                break
            x = x_start + col * dx + (dx * 0.5 if row % 2 == 1 else 0)
            pos[idx] = [x, y]
            idx += 1
        row += 1
        if y > 0.95:
            break
            
    # Add small random perturbation to break symmetry and avoid local traps
    pos += np.random.randn(*pos.shape) * 0.005
    pos = np.clip(pos, 0.02, 0.98)
    
    r = 0.05
    r_step = 1e-4
    damping = 0.4
    
    # Iterative expansion and relaxation
    for it in range(80000):
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion
        diff = pos[:, None, :] - pos[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Overlap amount
        overlaps = np.maximum(0, 2 * r - dists)
        safe_dists = np.where(dists > 1e-8, dists, 1e-8)
        dirs = diff / safe_dists[:, :, None]
        p_forces = overlaps[:, :, None] * dirs
        
        forces += np.sum(p_forces, axis=1)
        
        # Wall repulsion
        forces[:, 0] += np.maximum(0, r - pos[:, 0])
        forces[:, 0] -= np.maximum(0, pos[:, 0] - (1 - r))
        forces[:, 1] += np.maximum(0, r - pos[:, 1])
        forces[:, 1] -= np.maximum(0, pos[:, 1] - (1 - r))
        
        max_force = np.max(np.linalg.norm(forces, axis=1))
        
        if max_force < 1e-8:
            r += r_step
        else:
            pos += forces * damping
            pos = np.clip(pos, 1e-4, 1 - 1e-4)
            
    # Calculate final feasible radius
    dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists[np.tril_indices(n, -1)])
    
    min_wall = min(np.min(pos[:, 0]), np.min(pos[:, 1]), 
                   np.min(1 - pos[:, 0]), np.min(1 - pos[:, 1]))
    
    r_final = min(min_pair, min_wall) / 2
    
    centers = pos
    radii = np.full(n, r_final)
    return centers, radii, float(sum(radii))
