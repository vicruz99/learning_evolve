# sol_000057 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000035 (state 7f27ef90) state=d1736a1a sum of radii=2.620777 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def objective(params):
    return -np.sum(params[2 * N:])

def constraints(params):
    c = params[:2 * N].reshape(N, 2)
    r = params[2 * N:]
    
    vals = np.empty(4 * N + NUM_PAIRS)
    vals[:N] = c[:, 0] - r
    vals[N:2*N] = 1.0 - c[:, 0] - r
    vals[2*N:3*N] = c[:, 1] - r
    vals[3*N:4*N] = 1.0 - c[:, 1] - r
    
    dx = c[:, 0][:, None] - c[:, 0][None, :]
    dy = c[:, 1][:, None] - c[:, 1][None, :]
    dists = np.sqrt(dx*dx + dy*dy)
    r_sum = r[:, None] + r[None, :]
    
    vals[4*N:] = dists[I_IDX, J_IDX] - r_sum[I_IDX, J_IDX]
    return vals

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None

    def optimize_from(start_c, start_r):
        x0 = np.concatenate([start_c.flatten(), start_r])
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons_dict,
                           options={'maxiter': 20000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                c_opt = res.x[:2*N].reshape(N, 2)
                r_opt = res.x[2*N:]
                r_opt = np.maximum(r_opt, 0.0)
                for i in range(N):
                    max_r = min(c_opt[i, 0], 1.0 - c_opt[i, 0], c_opt[i, 1], 1.0 - c_opt[i, 1])
                    if r_opt[i] > max_r:
                        r_opt[i] = max_r
                return c_opt, r_opt, np.sum(r_opt)
        except Exception:
            pass
        return None, None, 0.0

    def generate_inits():
        inits = []
        # Hexagonal patterns with varying spacing and margin
        for s in np.linspace(0.17, 0.23, 6):
            for m in [0.03, 0.04, 0.05, 0.06]:
                rng = np.random.RandomState(len(inits) * 17 + 5)
                c = np.zeros((N, 2))
                idx = 0
                y = m
                row = 0
                while idx < N and y < 1.0 - m:
                    x = m + (row % 2) * s / 2.0
                    while x < 1.0 - m and idx < N:
                        c[idx] = [x + rng.uniform(-0.012, 0.012), y + rng.uniform(-0.012, 0.012)]
                        idx += 1
                        x += s
                    y += s * math.sqrt(3) / 2.0
                    row += 1
                while idx < N:
                    c[idx] = rng.uniform(0.1, 0.9, 2)
                    idx += 1
                r = np.full(N, 0.025)
                inits.append((c, r))
                
        # Grid patterns
        for step in np.linspace(0.17, 0.23, 6):
            rng = np.random.RandomState(len(inits) * 17 + 5)
            c = np.zeros((N, 2))
            idx = 0
            y = step
            while y < 1.0 and idx < N:
                x = step
                while x < 1.0 and idx < N:
                    c[idx] = [x + rng.uniform(-0.012, 0.012), y + rng.uniform(-0.012, 0.012)]
                    idx += 1
                    x += step
                y += step
            while idx < N:
                c[idx] = rng.uniform(0.1, 0.9, 2)
                idx += 1
            r = np.full(N, 0.025)
            inits.append((c, r))
            
        # Random starts
        for i in range(20):
            rng = np.random.RandomState(i + 500)
            c = rng.uniform(0.15, 0.85, (N, 2))
            r = np.full(N, 0.035)
            inits.append((c, r))
            
        return inits

    inits = generate_inits()
    
    for c0, r0 in inits:
        opt_c, opt_r, s = optimize_from(c0, r0)
        if opt_c is not None and s > best_sum:
            best_sum = s
            best_centers = opt_c.copy()
            best_radii = opt_r.copy()
            
            # Local refinement to escape shallow local minima
            for _ in range(6):
                pert_c = best_centers + np.random.randn(N, 2) * 0.002
                pert_c = np.clip(pert_c, 0.02, 0.98)
                pert_r = best_radii + np.random.randn(N) * 0.0005
                pert_r = np.clip(pert_r, 0.001, 0.5)
                rc, rr, rs = optimize_from(pert_c, pert_r)
                if rc is not None and rs > best_sum:
                    best_sum = rs
                    best_centers = rc.copy()
                    best_radii = rr.copy()

    # Final strict validity check and numerical adjustment
    if best_centers is not None:
        margin = 1e-9
        for i in range(N):
            x, y = best_centers[i]
            max_r = min(x, 1-x, y, 1-y) - margin
            best_radii[i] = min(best_radii[i], max(0, max_r))
            
        for _ in range(100):
            changed = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = best_centers[i,0] - best_centers[j,0]
                    dy = best_centers[i,1] - best_centers[j,1]
                    d = math.hypot(dx, dy)
                    if d < best_radii[i] + best_radii[j] - 1e-12:
                        excess = best_radii[i] + best_radii[j] - d + 1e-10
                        best_radii[i] -= excess/2
                        best_radii[j] -= excess/2
                        changed = True
            if not changed:
                break
        best_sum = np.sum(best_radii)

    # Fallback (should not be reached)
    if best_centers is None:
        centers = np.random.rand(N, 2) * 0.6 + 0.2
        radii = np.full(N, 0.02)
        return centers, radii, float(np.sum(radii))
        
    return best_centers, best_radii, float(best_sum)
