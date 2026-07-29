# sol_000388 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 916b0b30) state=758614d4 sum of radii=0.173348 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def get_min_dist(centers):
    """Compute the minimum distance between any pair of circles and boundaries."""
    n = centers.shape[0]
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 2.0)
    min_pair = np.min(dists)
    min_bound = np.min(np.concatenate([centers, 1 - centers], axis=1))
    return min(min_pair, min_bound)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_overall_centers = None
    best_overall_d = 0.0

    for seed in range(8):
        np.random.seed(seed)
        
        # Hexagonal initialization
        pts = []
        for i in range(7):
            for j in range(7):
                x = i * 0.14
                y = j * 0.14 * np.sqrt(3) / 2 + (0.07 * np.sqrt(3) / 2 if i % 2 == 1 else 0)
                if 0 <= x <= 1 and 0 <= y <= 1:
                    pts.append([x, y])
        while len(pts) < n:
            pts.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
        
        centers = np.array(pts[:n])
        # Add small random perturbation to break symmetry
        centers += np.random.uniform(-0.008, 0.008, centers.shape)
        centers = np.clip(centers, 0.01, 0.99)

        curr_d = get_min_dist(centers)
        best_centers = centers.copy()
        best_d = curr_d

        step = 0.012
        for _ in range(4000):
            d_target = curr_d * 1.001
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion
            for i in range(n):
                for j in range(i + 1, n):
                    diff = centers[i] - centers[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-8:
                        diff = np.random.uniform(-0.01, 0.01, 2)
                        dist = np.linalg.norm(diff)
                    if dist < d_target:
                        force_mag = (d_target - dist) * 5.0 / dist
                        f_vec = diff * force_mag
                        forces[i] += f_vec
                        forces[j] -= f_vec

                # Boundary repulsion
                if centers[i, 0] < d_target:
                    forces[i, 0] += (d_target - centers[i, 0]) * 20.0
                if centers[i, 0] > 1 - d_target:
                    forces[i, 0] -= (d_target - (1 - centers[i, 0])) * 20.0
                if centers[i, 1] < d_target:
                    forces[i, 1] += (d_target - centers[i, 1]) * 20.0
                if centers[i, 1] > 1 - d_target:
                    forces[i, 1] -= (d_target - (1 - centers[i, 1])) * 20.0

            centers += forces * step
            centers = np.clip(centers, 0, 1)

            new_d = get_min_dist(centers)
            if new_d > best_d:
                best_d = new_d
                best_centers = centers.copy()
                curr_d = new_d
            elif np.random.rand() < 0.06:
                # Shake to escape local minima
                centers += np.random.uniform(-0.02, 0.02, centers.shape)
                centers = np.clip(centers, 0, 1)
                curr_d = get_min_dist(centers)

            step *= 0.9992

        if best_d > best_overall_d:
            best_overall_d = best_d
            best_overall_centers = best_centers.copy()

    # Final radius calculation
    r = best_overall_d / 2
    radii = np.full(n, r)
    sum_radii = n * r
    
    return best_overall_centers, radii, sum_radii
