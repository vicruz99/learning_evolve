# sol_000121 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000066 (state 7dd8b726) state=66e65a0c sum of radii=2.615046 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
TRIU_IDX = np.triu_indices(N, k=1)

def obj_eq(vars_array):
    return -vars_array[-1]

def con_eq(vars_array):
    c = vars_array[:2*N].reshape(N, 2)
    R = vars_array[2*N]
    con = np.concatenate([
        c[:, 0] - R,
        1.0 - c[:, 0] - R,
        c[:, 1] - R,
        1.0 - c[:, 1] - R
    ])
    
    dx = c[:, 0:1] - c[:, 0:1].T
    dy = c[:, 1:2] - c[:, 1:2].T
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1.0)
    
    return np.concatenate([con, d2[TRIU_IDX] - 4.0 * R * R])

def obj_uneq(vars_array):
    return -np.sum(vars_array[:N])

def con_uneq(vars_array):
    r = vars_array[:N]
    u = vars_array[N:2*N]
    v = vars_array[2*N:3*N]
    
    x = r + (1.0 - 2.0 * r) * u
    y = r + (1.0 - 2.0 * r) * v
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d2 = dx**2 + dy**2
    
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    rs2 = rs**2
    
    return d2[TRIU_IDX] - rs2[TRIU_IDX]

def generate_configs():
    configs = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 4, 6, 4, 6], [5, 7, 5, 5, 4], 
        [7, 5, 5, 5, 4], [5, 5, 5, 5, 6]
    ]
    
    for pat in patterns:
        pts = []
        y = 0.1
        row = 0
        for cnt in pat:
            shift = 0.1 if row % 2 else 0.0
            x = 0.1 + shift
            for _ in range(cnt):
                if len(pts) < N:
                    pts.append([x, y])
                x += 0.2
            y += 0.173205
            row += 1
        configs.append(np.array(pts[:N]))
        
    for cfg in configs[:4]:
        for angle in [0.15, 0.3, -0.15, 0.05]:
            c_val, s_val = np.cos(angle), np.sin(angle)
            rot = cfg @ np.array([[c_val, -s_val], [s_val, c_val]])
            rot -= rot.min(axis=0)
            rot /= rot.max(axis=0)
            rot = rot * 0.8 + 0.1
            configs.append(rot)
            
    np.random.seed(42)
    for _ in range(8):
        configs.append(np.random.uniform(0.15, 0.85, (N, 2)))
        
    return configs

def run_packing():
    configs = generate_configs()
    
    best_R = 0.0
    best_c = None
    
    bounds_eq = [(0.0, 1.0)] * (2 * N) + [(0.09, 0.15)]
    cons_eq = {'type': 'ineq', 'fun': con_eq}
    
    for cfg in configs:
        x0 = np.concatenate([cfg.flatten(), [0.095]])
        try:
            res = minimize(obj_eq, x0, method='SLSQP', bounds=bounds_eq,
                           constraints=cons_eq, options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False})
            if np.isfinite(res.fun):
                R_val = -res.fun
                if R_val > best_R:
                    if np.min(con_eq(res.x)) > -1e-6:
                        best_R = R_val
                        best_c = res.x[:2*N].reshape(N, 2).copy()
        except Exception:
            pass
            
    if best_c is None:
        best_c = configs[0]
        best_R = 0.095
        
    r0 = np.full(N, best_R)
    u0 = (best_c[:, 0] - r0) / (1.0 - 2.0 * r0)
    v0 = (best_c[:, 1] - r0) / (1.0 - 2.0 * r0)
    u0 = np.clip(u0, 0.0, 1.0)
    v0 = np.clip(v0, 0.0, 1.0)
    
    x0_uneq = np.concatenate([r0, u0, v0])
    bounds_uneq = [(1e-4, 0.5)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    cons_uneq = {'type': 'ineq', 'fun': con_uneq}
    
    final_best_sum = 0.0
    final_best_vars = None
    
    np.random.seed(123)
    for trial in range(6):
        v0_trial = x0_uneq.copy()
        if trial > 0:
            v0_trial[N:] += np.random.uniform(-0.02, 0.02, 2 * N)
            v0_trial[:N] *= np.random.uniform(0.97, 1.03, N)
            v0_trial = np.clip(v0_trial, [1e-4]*N + [0.0]*(2*N), [0.5]*N + [1.0]*(2*N))
            
        try:
            res = minimize(obj_uneq, v0_trial, method='SLSQP', bounds=bounds_uneq,
                           constraints=cons_uneq, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            if np.isfinite(res.fun):
                s = -res.fun
                c_vals = con_uneq(res.x)
                if np.min(c_vals) > -1e-5 and s > final_best_sum:
                    final_best_sum = s
                    final_best_vars = res.x.copy()
        except Exception:
            pass
            
    if final_best_vars is None:
        final_best_vars = x0_uneq
        
    r_out = final_best_vars[:N]
    u_out = final_best_vars[N:2*N]
    v_out = final_best_vars[2*N:3*N]
    
    x_out = r_out + (1.0 - 2.0 * r_out) * u_out
    y_out = r_out + (1.0 - 2.0 * r_out) * v_out
    centers_out = np.column_stack((x_out, y_out))
    
    c_vals = con_uneq(final_best_vars)
    min_viol = np.min(c_vals)
    if min_viol < -1e-9:
        alpha = 1.0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers_out[i] - centers_out[j])
                r_sum = r_out[i] + r_out[j]
                if d < r_sum:
                    alpha = min(alpha, d / r_sum)
        r_out *= alpha * 0.999999
        x_out = r_out + (1.0 - 2.0 * r_out) * u_out
        y_out = r_out + (1.0 - 2.0 * r_out) * v_out
        centers_out = np.column_stack((x_out, y_out))
        
    return centers_out, r_out, float(np.sum(r_out))
