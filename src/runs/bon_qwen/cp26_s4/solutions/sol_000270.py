# sol_000270 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f6ad2c92) state=2add860a sum of radii=2.586728 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Helper to create initial hexagonal configuration
    def get_initial_config(seed=0):
        rng = np.random.RandomState(seed)
        # Hexagonal rows: 6, 5, 6, 5, 4
        rows = [6, 5, 6, 5, 4]
        centers = []
        for r_idx, cnt in enumerate(rows):
            y = 0.12 + r_idx * 0.16 + rng.uniform(-0.02, 0.02)
            for c_idx in range(cnt):
                x = 0.12 + c_idx * 0.18 + (0.09 if r_idx % 2 == 1 else 0.0) + rng.uniform(-0.02, 0.02)
                centers.append([x, y])
        centers = np.array(centers)
        # Clamp to valid range
        centers = np.clip(centers, 0.01, 0.99)
        radii = np.full(n, 0.08)
        return np.concatenate([centers.flatten(), radii])

    # Vectorized constraint function
    def constraints(vars):
        c = vars[:2*n].reshape(n, 2)
        r = vars[2*n:]
        cons = []
        # Boundary constraints: c - r >= 0, 1 - c - r >= 0
        cons.append(c[:, 0] - r)
        cons.append(r)
        cons.append(1.0 - c[:, 0] - r)
        cons.append(c[:, 1] - r)
        cons.append(1.0 - c[:, 1] - r)
        
        # Overlap constraints: dist - r_i - r_j >= 0
        # Vectorized distance matrix
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        overlaps = r[:, np.newaxis] + r[np.newaxis, :]
        # Only lower triangle to avoid duplicates
        mask = np.tril(np.ones((n, n), dtype=bool), -1)
        cons.append(dist[mask] - overlaps[mask])
        
        return np.concatenate(cons)

    def objective(vars):
        return -np.sum(vars[2*n:])

    bnds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': constraints}
    
    best_result = None
    best_sum = -np.inf
    
    # Try multiple seeds to find global optimum
    for seed in range(5):
        x0 = get_initial_config(seed=seed)
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons_dict,
                           options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            if res.success and res.fun < best_sum:
                best_sum = res.fun
                best_result = res
        except Exception:
            continue
            
    if best_result is None:
        # Fallback to basic config
        x0 = get_initial_config(seed=0)
        best_result = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons_dict,
                               options={'maxiter': 2000, 'ftol': 1e-9})
        
    final_centers = best_result.x[:2*n].reshape(n, 2)
    final_radii = best_result.x[2*n:]
    
    # Post-processing: strict validity check and minor adjustment
    # SLSQP might land exactly on boundary. We ensure strict satisfaction within tolerance.
    # Check overlaps and shrink radii slightly if needed
    for _ in range(10):
        valid = True
        min_gap = 1.0
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((final_centers[i] - final_centers[j])**2))
                gap = dist - (final_radii[i] + final_radii[j])
                if gap < min_gap:
                    min_gap = gap
            # Check boundaries
            for axis in range(2):
                if final_centers[i, axis] - final_radii[i] < 1e-14:
                    min_gap = min(min_gap, -1e-10)
                if final_centers[i, axis] + final_radii[i] > 1 - 1e-14:
                    min_gap = min(min_gap, -1e-10)
                    
        if min_gap < -1e-12:
            # Violation found, shrink all radii slightly to resolve
            shrink_factor = 0.999
            final_radii *= shrink_factor
        else:
            break
            
    return final_centers, final_radii, float(np.sum(final_radii))
