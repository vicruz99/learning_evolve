# sol_000103 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 15bab5cf) state=55c0b72e sum of radii=2.459437 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
ALPHA = 1500.0

def objective(params):
    """
    Objective function: maximize sum of radii subject to non-overlap and boundary constraints.
    Uses a penalty method with squared violation terms.
    """
    centers = params[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = params[2*N_CIRCLES:]

    # Pairwise distances (N x N matrix)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)  # Ignore self-distances

    # Overlap penalty: max(0, r_i + r_j - d_ij)^2 summed over i < j
    rad_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    overlaps = rad_sum - dists
    triu_idx = np.triu_indices(N_CIRCLES, k=1)
    overlap_pen = np.sum(np.maximum(0, overlaps[triu_idx])**2)

    # Boundary penalties: max(0, r - dist_to_wall)^2
    left_pen = np.sum(np.maximum(0, radii - centers[:, 0])**2)
    right_pen = np.sum(np.maximum(0, radii - (1.0 - centers[:, 0]))**2)
    bottom_pen = np.sum(np.maximum(0, radii - centers[:, 1])**2)
    top_pen = np.sum(np.maximum(0, radii - (1.0 - centers[:, 1]))**2)

    bound_pen = left_pen + right_pen + bottom_pen + top_pen

    # Minimize negative sum of radii + penalty
    return -np.sum(radii) + ALPHA * (overlap_pen + bound_pen)


def run_packing():
    n = N_CIRCLES
    best_params = None
    best_loss = np.inf

    inits = []

    # 1. Hexagonal lattice initialization (dense packing geometry)
    pts = []
    y = 0.1
    row = 0
    while y < 0.9 and len(pts) < n:
        x = 0.1 + (0.08 if row % 2 == 1 else 0)
        while x < 0.9 and len(pts) < n:
            pts.append([x, y])
            x += 0.16
        y += 0.12
        row += 1
    while len(pts) < n:
        pts.append([np.random.uniform(0.2, 0.8), np.random.uniform(0.2, 0.8)])
    pts = np.array(pts[:n])
    inits.append(np.concatenate([pts.flatten(), np.ones(n) * 0.08]))

    # 2. Random initialization
    np.random.seed(42)
    pts2 = np.random.rand(n, 2) * 0.6 + 0.2
    inits.append(np.concatenate([pts2.flatten(), np.ones(n) * 0.08]))

    # 3. Grid initialization
    gs = np.linspace(0.15, 0.85, 6)
    pts3 = np.array(np.meshgrid(gs, gs)).T.reshape(-1, 2)
    pts3 = pts3[:n]
    inits.append(np.concatenate([pts3.flatten(), np.ones(n) * 0.08]))

    # Bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0.0, 1.0) for _ in range(2*n)] + [(0.0, 0.5) for _ in range(n)]

    # Phase 1: Global search from multiple starts
    for init in inits:
        res = minimize(objective, init, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 8000, 'ftol': 1e-12})
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x.copy()

    # Phase 2: Local refinement via perturbed restarts
    for _ in range(4):
        pert = best_params.copy()
        pert[:2*n] += np.random.normal(0, 0.005, 2*n)
        pert[:2*n] = np.clip(pert[:2*n], 0.01, 0.99)
        res = minimize(objective, pert, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 4000, 'ftol': 1e-12})
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x.copy()

    centers = best_params[:2*n].reshape(n, 2)
    radii = best_params[2*n:]

    # Apply safety margin to strictly satisfy validator tolerance
    radii *= 0.995

    return centers, radii, float(np.sum(radii))
