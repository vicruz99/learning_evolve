# sol_000296 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fc92aa36) state=1ca5c3c3 sum of radii=2.208735 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    N = 26
    
    # Initialize centers in the safe interior and small radii
    centers = np.random.uniform(0.15, 0.85, (N, 2))
    radii = np.full(N, 0.01)
    
    num_growth_steps = 6000
    relax_iters = 80
    growth_rate = 0.00002
    lr = 0.5
    
    for step in range(num_growth_steps):
        radii += growth_rate
        # Decay learning rate to settle into tight packing
        lr_step = lr * max(0.05, 1.0 - step / num_growth_steps)
        
        for _ in range(relax_iters):
            # Vectorized pairwise repulsion
            diffs = centers[:, None, :] - centers[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            min_dists = radii[:, None] + radii[None, :]
            overlaps = np.maximum(min_dists - dists, 0.0)
            overlaps[np.diag_indices(N)] = 0.0
            
            safe_dists = np.maximum(dists, 1e-12)
            norm_diffs = diffs / safe_dists[:, :, None]
            forces = np.sum(overlaps[:, :, None] * norm_diffs, axis=1)
            
            # Boundary repulsion forces
            left = np.maximum(radii - centers[:, 0], 0.0)
            right = np.maximum(centers[:, 0] - (1.0 - radii), 0.0)
            bottom = np.maximum(radii - centers[:, 1], 0.0)
            top = np.maximum(centers[:, 1] - (1.0 - radii), 0.0)
            
            forces[:, 0] += left * 15.0 - right * 15.0
            forces[:, 1] += bottom * 15.0 - top * 15.0
            
            # Update and strictly clamp to feasible region
            centers += forces * lr_step
            centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
            centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
            
    # Final safety adjustment: compute tight radii from final positions
    r_bound = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                         np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    dists[np.diag_indices(N)] = np.inf
    r_pair_min = np.min(dists / 2.0, axis=1)
    
    # Combine constraints and apply tolerance margin
    final_radii = np.minimum(r_bound, r_pair_min) * 0.9999
    
    # Re-clamp centers to match final radii exactly
    centers[:, 0] = np.clip(centers[:, 0], final_radii, 1.0 - final_radii)
    centers[:, 1] = np.clip(centers[:, 1], final_radii, 1.0 - final_radii)
    
    total_sum = float(np.sum(final_radii))
    return centers, final_radii, total_sum
