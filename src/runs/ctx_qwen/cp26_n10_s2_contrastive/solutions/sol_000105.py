# sol_000105 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000094 (state 4cf54399) state=f1d2f00a sum of radii=2.524228 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    dx = centers[:, 0][:, None] - centers[:, 0][None, :]
    dy = centers[:, 1][:, None] - centers[:, 1][None, :]
    dists = np.hypot(dx, dy)
    
    A_ub = np.zeros((len(I_IDX), n))
    A_ub[np.arange(len(I_IDX)), I_IDX] = 1.0
    A_ub[np.arange(len(I_IDX)), J_IDX] = 1.0
    b_ub = dists[I_IDX, J_IDX]
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(1e-9, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def objective_centers(centers_flat):
    """Objective for center optimization: negative sum of radii."""
    centers = centers_flat.reshape(N, 2)
    _, s = solve_lp_radii(centers)
    return -s

def generate_initials(num_init):
    """Generate diverse initial center configurations."""
    inits = []
    rng = np.random.default_rng(42)
    
    for _ in range(num_init):
        style = rng.choice(['hex', 'grid', 'rand'])
        c = np.zeros((N, 2))
        
        if style == 'hex':
            sp = rng.uniform(0.15, 0.22)
            idx = 0
            y = sp / 2
            row = 0
            while idx < N and y < 1.0 - sp / 2:
                x = sp / 2 + (row % 2) * sp / 2
                while x < 1.0 - sp / 2 and idx < N:
                    c[idx] = [x, y]
                    x += sp
                    idx += 1
                y += sp * np.sqrt(3) / 2
                row += 1
            while idx < N:
                c[idx] = rng.uniform(0.1, 0.9, 2)
                idx += 1
        elif style == 'grid':
            step = rng.uniform(0.17, 0.23)
            idx = 0
            y = step / 2
            while y < 1.0 - step / 2 and idx < N:
                x = step / 2
                while x < 1.0 - step / 2 and idx < N:
                    c[idx] = [x, y]
                    x += step
                    idx += 1
                y += step
            while idx < N:
                c[idx] = rng.uniform(0.1, 0.9, 2)
                idx += 1
        else:
            c = rng.uniform(0.1, 0.9, (N, 2))
            
        c += rng.normal(0, 0.006, c.shape)
        inits.append(np.clip(c, 0.02, 0.98))
        
    return inits

def run_packing():
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Broad initialization & filtering
    inits = generate_initials(40)
    candidates = []
    for c in inits:
        r, s = solve_lp_radii(c)
        candidates.append((s, c, r))
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Adaptive basin hopping on top candidates
    top_k = min(5, len(candidates))
    rng = np.random.default_rng(123)
    
    for s0, c0, r0 in candidates[:top_k]:
        curr_c = c0.copy()
        curr_r = r0.copy()
        curr_s = s0
        step = 0.012
        
        for iteration in range(400):
            noise = step * (0.997 ** (iteration // 50))
            c_pert = curr_c + rng.normal(0, noise, curr_c.shape)
            c_pert = np.clip(c_pert, 0.005, 0.995)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            
            if s_pert > curr_s:
                curr_c, curr_r, curr_s = c_pert, r_pert, s_pert
                step = min(0.02, step * 1.03)
                
                # Lightweight coordinate refinement after improvement
                if iteration % 5 == 0:
                    for dim in range(2):
                        for i in range(N):
                            base = curr_c[i, dim]
                            for delta in [0.005, -0.005]:
                                cand = curr_c.copy()
                                cand[i, dim] = np.clip(base + delta, 0.005, 0.995)
                                _, s_cand = solve_lp_radii(cand)
                                if s_cand > curr_s:
                                    curr_c = cand
                                    curr_s = s_cand
                                    curr_r, _ = solve_lp_radii(curr_c)
            else:
                step = max(1e-4, step * 0.96)
                
            if curr_s > best_sum:
                best_sum = curr_s
                best_centers = curr_c.copy()
                best_radii = curr_r.copy()
                
    if best_centers is None:
        best_centers = inits[0]
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # Phase 3: Final Nelder-Mead polish on centers
    try:
        res = minimize(objective_centers, best_centers.flatten(), method='Nelder-Mead',
                       options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 3000})
        co = res.x.reshape(N, 2)
        ro, s_final = solve_lp_radii(co)
        if s_final > best_sum:
            best_sum = s_final
            best_centers = co.copy()
            best_radii = ro.copy()
    except Exception:
        pass
        
    # Phase 4: Strict post-processing to guarantee validator compliance
    c_final = best_centers.copy()
    r_final = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(c_final[i, 0], 1.0 - c_final[i, 0], 
                 c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], mx - 1e-10)
        r_final[i] = max(r_final[i], 0.0)
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(c_final[i, 0] - c_final[j, 0], c_final[i, 1] - c_final[j, 1])
                if d < r_final[i] + r_final[j] - 1e-12:
                    exc = r_final[i] + r_final[j] - d
                    r_final[i] -= exc / 2.0
                    r_final[j] -= exc / 2.0
                    changed = True
        if not changed:
            break
            
    r_final = np.maximum(r_final, 0.0)
    return c_final, r_final, float(np.sum(r_final))
