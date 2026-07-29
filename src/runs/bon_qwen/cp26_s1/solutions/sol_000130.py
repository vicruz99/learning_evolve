# sol_000130 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5da4630c) state=33f3ee1e sum of radii=2.084434 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    N = 26
    np.random.seed(42)
    
    # Initialize centers on a perturbed grid for good spatial distribution
    pos = np.zeros((N, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx < N:
                pos[idx] = [0.1 + j * 0.18, 0.1 + i * 0.18]
                idx += 1
    pos += np.random.normal(0, 0.02, pos.shape)
    pos = np.clip(pos, 0.05, 0.95)
    
    r = np.full(N, 0.03)
    
    # Simulation parameters
    dt = 0.0008
    repulsion_k = 2000.0
    wall_k = 5000.0
    damping = 0.85
    growth_rate = 0.3
    steps = 80000
    
    vel = np.zeros((N, 2))
    
    for step in range(steps):
        # Compute pairwise distance matrix
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(dist_sq)
        np.fill_diagonal(dist, np.inf)
        
        # Compute forces
        forces = np.zeros((N, 2))
        
        # Wall repulsion forces
        left_over = np.maximum(0, r - pos[:, 0])
        right_over = np.maximum(0, r - (1.0 - pos[:, 0]))
        forces[:, 0] += wall_k * (left_over - right_over)
        
        top_over = np.maximum(0, r - pos[:, 1])
        bottom_over = np.maximum(0, r - (1.0 - pos[:, 1]))
        forces[:, 1] += wall_k * (top_over - bottom_over)
        
        # Circle-circle repulsion forces
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        overlap = np.maximum(0, r_sum - dist)
        
        safe_dist_sq = np.maximum(dist_sq, 1e-10)
        force_mag = repulsion_k * overlap / safe_dist_sq
        forces += np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Exploration noise with exponential cooling
        temp = 0.002 * np.exp(-step / 25000)
        if step > steps - 20000:
            growth_rate = 0.0
            damping = 0.95
            temp = 0.0
            
        vel += np.random.normal(0, temp, pos.shape)
        vel = vel * damping + forces * dt
        pos += vel * dt
        pos = np.clip(pos, 1e-6, 1.0 - 1e-6)
        
        # Compute clearance for adaptive radius growth
        wall_clear = np.minimum(
            np.minimum(pos[:, 0] - r, 1.0 - pos[:, 0] - r),
            np.minimum(pos[:, 1] - r, 1.0 - pos[:, 1] - r)
        )
        circle_clear = np.min(dist - r_sum, axis=1)
        clear = np.minimum(wall_clear, circle_clear)
        
        # Grow radii based on locally available space
        r += growth_rate * clear
        r = np.maximum(r, 1e-6)
        
    return pos, r, np.sum(r)
