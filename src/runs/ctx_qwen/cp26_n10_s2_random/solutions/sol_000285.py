# sol_000285 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000227 (state 324f8d76) state=9ac3ee4c sum of radii=2.622183 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

# Precompute constant LP constraint matrix structure
def _build_lp_struct():
    num_pairs = N * (N - 1) // 2
    A = np.zeros((num_pairs + 4 * N, N))
    pairs = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            pairs.append((i, j))
            idx += 1
    for i in range(N):
        for _ in range(4):
            A[idx, i] = 1.0
            idx += 1
    return A, pairs

A_LP, PAIRS = _build_lp_struct()

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers. Returns radii and duals."""
    n = centers.shape[0]
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    b = np.zeros(A_LP.shape[0])
    k = 0
    for i, j in PAIRS:
        b[k] = dists[i, j]
        k += 1
    for i in range(n):
        b[k] = centers[i, 0]; k += 1
        b[k] = 1.0 - centers[i, 0]; k += 1
        b[k] = centers[i, 1]; k += 1
        b[k] = 1.0 - centers[i, 1]; k += 1
        
    try:
        res = linprog(-np.ones(n), A_ub=A_LP, b_ub=b, bounds=[(0, None)]*n, method='highs')
        if res.success:
            try:
                duals = res.marginals.ineqlin
            except AttributeError:
                try:
                    duals = res.ineqlin.marginals
                except AttributeError:
                    duals = np.zeros_like(b)
            return res.x, duals
    except Exception:
        pass
    return None, None

def compute_grad(centers, duals):
    """Computes exact gradient of sum(radii) w.r.t centers using LP duals."""
    grad = np.zeros_like(centers)
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    k = 0
    for i, j in PAIRS:
        mu = duals[k]
        if mu > 1e-9:
            d = dists[i, j]
            if d > 1e-9:
                v = (centers[i] - centers[j]) / d
                grad[i] += mu * v
                grad[j] -= mu * v
        k += 1
        
    bp = len(PAIRS)
    for i in range(N):
        grad[i, 0] += duals[bp + 4*i] - duals[bp + 4*i + 1]
        grad[i, 1] += duals[bp + 4*i + 2] - duals[bp + 4*i + 3]
    return grad

def obj_grad(v):
    """Objective and gradient for L-BFGS-B: minimizes negative sum of radii."""
    centers = v.reshape(N, 2)
    centers = np.clip(centers, 0.001, 0.999)
    radii, duals = solve_lp(centers)
    if radii is None:
        return 0.0, np.zeros_like(v)
    val = -np.sum(radii)
    g = compute_grad(centers, duals)
    return val, -g.flatten()

def generate_inits(rng):
    """Generates diverse initial center configurations."""
    inits = []
    # Hexagonal patterns with varying row counts
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [4, 5, 6, 5, 6], [5, 4, 6, 6, 5]
    ]
    for pat in patterns:
        for r0 in [0.088, 0.094, 0.101]:
            c = []
            y = r0
            for ri, cnt in enumerate(pat):
                sh = r0 if ri % 2 else 0.0
                x = r0 + sh
                for _ in range(cnt):
                    c.append([x + rng.normal(0, 0.003), y + rng.normal(0, 0.003)])
                    x += 2.0 * r0
                y += r0 * np.sqrt(3.0)
            inits.append(np.clip(np.array(c[:N]), 0.05, 0.95))
            
    # Corner-heavy starts
    for _ in range(8):
        c = rng.uniform(0.15, 0.85, (N, 2))
        c[:4] = [[0.1,0.1], [0.9,0.1], [0.1,0.9], [0.9,0.9]]
        inits.append(np.clip(c + rng.normal(0, 0.004, c.shape), 0.05, 0.95))
        
    # Force-directed spreads
    for _ in range(12):
        c = rng.uniform(0.2, 0.8, (N, 2))
        for _ in range(500):
            f = np.zeros_like(c)
            for i in range(N):
                for j in range(i+1, N):
                    dv = c[i]-c[j]; d = np.linalg.norm(dv)
                    if d < 0.22 and d > 1e-5:
                        ff = (0.22-d)*0.04/d
                        f[i] += dv*ff; f[j] -= dv*ff
            c += f; c = np.clip(c, 0.1, 0.9)
        inits.append(c)
    return inits

def coordinate_refine(centers, rng):
    """Optimizes each circle's position independently to escape local minima."""
    c = centers.copy()
    improved = True
    for _ in range(4):
        if not improved: break
        improved = False
        for i in range(N):
            def obj_i(p):
                tmp = c.copy()
                tmp[i] = np.clip(p, 0.002, 0.998)
                r, _ = solve_lp(tmp)
                return -np.sum(r) if r is not None else 0.0
                
            current_val = obj_i(c[i])
            try:
                res = minimize(obj_i, c[i], method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9})
                if res.fun < current_val - 1e-8:
                    c[i] = np.clip(res.x, 0.005, 0.995)
                    improved = True
            except Exception:
                pass
    return c

def repair_packing(centers, radii):
    """Deterministic repair to guarantee strict validation compliance."""
    radii = radii.copy()
    for _ in range(200):
        changed = False
        # Boundary clamping
        for i in range(N):
            mr = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
            if radii[i] > mr + 1e-11:
                radii[i] = mr; changed = True
                
        # Overlap resolution
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i]-centers[j])
                req = radii[i] + radii[j]
                if d < req - 1e-11:
                    shrink = (req - d)/2.0 + 1e-9
                    radii[i] -= shrink; radii[j] -= shrink; changed = True
        if not changed: break
    return np.maximum(radii, 0.0)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_lbfgs = [(0.005, 0.995)] * (2 * N)
    bounds_slsqp = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    best_c = None
    best_r = None
    best_sum = -1.0
    
    inits = generate_inits(rng)
    
    # Phase 1: L-BFGS-B center optimization from diverse starts
    for c0 in inits:
        try:
            res = minimize(obj_grad, c0.flatten(), jac=True, method='L-BFGS-B', bounds=bounds_lbfgs,
                          options={'maxiter': 4000, 'ftol': 1e-13})
            s = -res.fun
            if s > best_sum:
                best_sum = s
                best_c = res.x.reshape(N, 2)
        except Exception:
            pass
            
    if best_c is None:
        best_c = inits[0]
    best_r, _ = solve_lp(best_c)
    best_sum = np.sum(best_r)
    
    # Phase 2: Coordinate-wise refinement (very effective for packing)
    best_c = coordinate_refine(best_c, rng)
    best_r, _ = solve_lp(best_c)
    best_sum = np.sum(best_r)
    
    # Phase 3: Adaptive Basin Hopping
    for step in range(70):
        noise = 0.008 * (0.91 ** (step // 5))
        cp = best_c + rng.normal(0, noise, best_c.shape)
        cp = np.clip(cp, 0.01, 0.99)
        
        try:
            res = minimize(obj_grad, cp.flatten(), jac=True, method='L-BFGS-B', bounds=bounds_lbfgs,
                          options={'maxiter': 2000, 'ftol': 1e-13})
            s = -res.fun
            if s > best_sum:
                best_sum = s
                best_c = res.x.reshape(N, 2)
                best_r, _ = solve_lp(best_c)
                
                # Coordinate refine on new basin
                best_c = coordinate_refine(best_c, rng)
                best_r, _ = solve_lp(best_c)
                best_sum = np.sum(best_r)
        except Exception:
            pass
            
    # Phase 4: SLSQP Joint Polish for numerical precision
    def obj_joint(v):
        return -np.sum(v[2*N:])
        
    def cons_joint(v):
        c = v[:2*N].reshape(N, 2)
        r = v[2*N:]
        con = np.concatenate([
            c[:,0] - r, 1.0 - c[:,0] - r,
            c[:,1] - r, 1.0 - c[:,1] - r
        ])
        i, j = np.triu_indices(N, 1)
        dx = c[i,0] - c[j,0]
        dy = c[i,1] - c[j,1]
        dr = r[i] + r[j]
        con = np.concatenate([con, dx**2 + dy**2 - dr**2])
        return con
        
    v0 = np.concatenate([best_c.flatten(), best_r])
    try:
        res = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_slsqp,
                      constraints={'type':'ineq','fun':cons_joint},
                      options={'maxiter': 10000, 'ftol': 1e-14})
        if np.min(cons_joint(res.x)) >= -1e-8:
            s = np.sum(res.x[2*N:])
            if s > best_sum:
                best_c = res.x[:2*N].reshape(N,2)
                best_r = res.x[2*N:]
                best_sum = s
    except Exception:
        pass
        
    # Phase 5: Final Verification & Repair
    lp_r, _ = solve_lp(best_c)
    if lp_r is not None and np.sum(lp_r) > best_sum:
        best_r = lp_r
        best_sum = np.sum(lp_r)
        
    radii = repair_packing(best_c.copy(), best_r.copy())
    return best_c, radii, float(np.sum(radii))
