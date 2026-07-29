# sol_000045 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1f1389a1) state=c8b86381 sum of radii=0.523900 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

# Final Code
import numpy as np

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Configurations to try
    configs = []
    
    # 1. Random
    configs.append(np.random.uniform(0.1, 0.9, (n, 2)))
    
    # 2. Grid
    grid_c = []
    for r in range(5):
        for c in range(5):
            grid_c.append([0.1 + c*0.18, 0.1 + r*0.18])
    grid_c.append([0.5, 0.5])
    configs.append(np.array(grid_c))
    
    # 3. Hexagonal
    hex_c = []
    y = 0.08
    row = 0
    while len(hex_c) < n:
        x = 0.08
        shift = 0.08 * (row % 2)
        while x < 0.92 and len(hex_c) < n:
            hex_c.append([x + shift, y])
            x += 0.16
        y += 0.08 * np.sqrt(3)
        row += 1
    configs.append(np.array(hex_c))
    
    # Parameters
    dt = 0.002
    growth_step = 0.00015
    force_k = 100.0
    wall_k = 200.0
    max_iter = 5000
    
    for config in configs:
        current_centers = config.copy()
        current_centers += np.random.uniform(-0.01, 0.01, (n, 2))
        current_centers = np.clip(current_centers, 0.02, 0.98)
        
        current_radii = np.ones(n) * 0.02
        
        for step in range(max_iter):
            current_radii += growth_step
            
            # Forces
            c1 = current_centers[:, np.newaxis, :]
            c2 = current_centers[np.newaxis, :, :]
            diffs = c1 - c2
            dists_sq = np.sum(diffs**2, axis=2)
            dists = np.sqrt(dists_sq)
            dists_safe = np.where(dists_sq > 1e-10, dists, 1e-5)
            dirs = diffs / dists_safe[:, :, np.newaxis]
            
            min_dists = current_radii[:, np.newaxis] + current_radii[np.newaxis, :]
            overlaps = min_dists - dists
            force_mags = np.maximum(0, overlaps) * force_k
            force_vectors = force_mags[:, :, np.newaxis] * dirs
            forces = np.sum(force_vectors, axis=1)
            
            x = current_centers[:, 0]
            y = current_centers[:, 1]
            r = current_radii
            
            forces[:, 0] += wall_k * np.maximum(0, r - x)
            forces[:, 0] -= wall_k * np.maximum(0, x - (1.0 - r))
            forces[:, 1] += wall_k * np.maximum(0, r - y)
            forces[:, 1] -= wall_k * np.maximum(0, y - (1.0 - r))
            
            current_centers += forces * dt
            current_centers = np.clip(current_centers, 0.0, 1.0)
            
            if step % 600 == 0 and step > 0:
                current_centers += np.random.uniform(-0.005, 0.005, (n, 2))
                current_centers = np.clip(current_centers, 0.01, 0.99)

            # Check validity
            if step % 500 == 0:
                dists_f = np.sqrt(np.sum((current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :])**2, axis=2))
                min_dists_f = current_radii[:, np.newaxis] + current_radii[np.newaxis, :]
                overlaps_f = min_dists_f - dists_f
                np.fill_diagonal(overlaps_f, 0)
                max_ov = np.max(overlaps_f)
                
                w_ov = np.maximum(0, current_radii - current_centers[:, 0])
                w_ov += np.maximum(0, current_centers[:, 0] - (1.0 - current_radii))
                w_ov += np.maximum(0, current_radii - current_centers[:, 1])
                w_ov += np.maximum(0, current_centers[:, 1] - (1.0 - current_radii))
                max_w_ov = np.max(w_ov)
                
                if max_ov < 1e-5 and max_w_ov < 1e-5:
                    s = np.sum(current_radii)
                    if s > best_sum:
                        best_sum = s
                        best_centers = current_centers.copy()
                        best_radii = current_radii.copy()

    if best_centers is None:
        r = 0.08
        centers = np.zeros((n, 2))
        idx = 0
        y = r
        while idx < n:
            x = r
            while x < 1.0 - r and idx < n:
                centers[idx] = [x, y]
                x += 2*r
                idx += 1
            y += 2*r
        radii = np.ones(n) * r
        best_sum = np.sum(radii)
        best_centers = centers
        best_radii = radii

    return best_centers, best_radii, float(best_sum)
