import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)
A_ub_lp = np.zeros((NUM_PAIRS, N))
A_ub_lp[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(NUM_PAIRS), J_IDX] = 1.0

def solve_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    bounds = []
    for i in range(N):
        x, y = centers[i]
        ub = min(x, 1.0-x, y, 1.0-y)
        bounds.append((0.0, max(0.0, ub - 1e-12)))
    try:
        res = linprog(-np.ones(N), A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(N, 0.01)

def make_strict(centers, radii):
    """Deterministically resolve overlaps and boundary violations."""
    for i in range(N):
        ub = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        if radii[i] > ub:
            radii[i] = max(0.0, ub - 1e-9)
    for _ in range(50):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            if d < radii[i] + radii[j] - 1e-12:
                exc = radii[i] + radii[j] - d
                radii[i] -= exc/2.0
                radii[j] -= exc/2.0
                changed = True
        if not changed:
            break
    return centers, np.maximum(radii, 0.0)

def objective_slsqp(x):
    return -np.sum(x[2::3])

def constraints_slsqp(x):
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    c = np.empty(4 * N + NUM_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4*N:] = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    return c

def polish_slsqp(centers, radii):
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * N
    x0 = np.zeros(3 * N)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = np.maximum(radii * 0.98, 1e-5)
    try:
        res = minimize(objective_slsqp, x0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': constraints_slsqp},
                       options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
        if res.success:
            c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
            r_opt = solve_lp(c_opt)
            return c_opt, r_opt, np.sum(r_opt)
    except Exception:
        pass
    return centers, radii, np.sum(radii)

def run_packing():
    rng = np.random.default_rng(42)
    best_sum = 0.0
    best_c = None
    best_r = None

    # Generate diverse initial configurations
    starts = []
    # 1. Hexagonal lattices with varying spacings
    for s in np.linspace(0.14, 0.26, 15):
        c = np.zeros((N, 2))
        idx = 0
        y = s/2
        row = 0
        while idx < N and y < 1.0 - s/2:
            x = s/2 + (row%2)*s/2
            while x < 1.0 - s/2 and idx < N:
                c[idx] = [x, y]
                idx += 1
                x += s
            y += s * np.sqrt(3)/2
            row += 1
        while idx < N:
            c[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
        starts.append(c + rng.normal(0, 0.008, c.shape))
        
    # 2. Random uniform placements
    for _ in range(30):
        starts.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    # Evaluate and pick best start
    for sc in starts:
        sc = np.clip(sc, 0.02, 0.98)
        r = solve_lp(sc)
        s = np.sum(r)
        if s > best_sum:
            best_sum = s
            best_c = sc.copy()
            best_r = r.copy()

    if best_c is None:
        best_c = starts[0]
        best_sum, best_r = np.sum(solve_lp(best_c)), solve_lp(best_c)

    curr_c = best_c.copy()
    curr_r = best_r.copy()
    curr_s = best_sum
    
    # Basin Hopping + Coordinate Descent
    T = 0.015
    for step in range(8000):
        T = 0.015 * np.exp(-step / 1500.0)
        scale = T * 3.0
        
        # 1. Random subset perturbation (Simulated Annealing)
        n_p = rng.choice([1, 2, 3])
        idxs = rng.choice(N, n_p, replace=False)
        new_c = curr_c.copy()
        new_c[idxs] += rng.normal(0, scale, (n_p, 2))
        new_c = np.clip(new_c, 0.01, 0.99)
        new_r = solve_lp(new_c)
        new_s = np.sum(new_r)
        
        if new_s > curr_s or rng.random() < np.exp((new_s - curr_s) / max(T, 1e-6)):
            curr_c, curr_r, curr_s = new_c, new_r, new_s
            if new_s > best_sum:
                best_sum = new_s
                best_c = new_c.copy()
                best_r = new_r.copy()
                
        # 2. Coordinate Descent (Greedily move each circle to improve LP sum)
        if step % 5 == 0:
            improved = True
            while improved:
                improved = False
                for i in range(N):
                    best_move_c = curr_c.copy()
                    best_move_s = curr_s
                    for dx, dy in [(scale, 0), (-scale, 0), (0, scale), (0, -scale)]:
                        trial_c = curr_c.copy()
                        trial_c[i, 0] = np.clip(trial_c[i, 0] + dx, 0.01, 0.99)
                        trial_c[i, 1] = np.clip(trial_c[i, 1] + dy, 0.01, 0.99)
                        trial_r = solve_lp(trial_c)
                        trial_s = np.sum(trial_r)
                        if trial_s > best_move_s:
                            best_move_s = trial_s
                            best_move_c = trial_c
                    if best_move_s > curr_s:
                        curr_c = best_move_c
                        curr_r = solve_lp(curr_c)
                        curr_s = best_move_s
                        improved = True
                        if curr_s > best_sum:
                            best_sum = curr_s
                            best_c = curr_c.copy()
                            best_r = curr_r.copy()
                            
        # 3. Periodic gradient-based polishing
        if step % 500 == 0:
            best_c, best_r, best_sum = polish_slsqp(best_c, best_r)
            curr_c, curr_r, curr_s = best_c, best_r, best_sum

    # Final high-precision polish
    best_c, best_r, best_sum = polish_slsqp(best_c, best_r)
    
    # Strict post-processing to guarantee validator compliance
    c_final, r_final = make_strict(best_c.copy(), best_r.copy())
    return c_final, r_final, float(np.sum(r_final))