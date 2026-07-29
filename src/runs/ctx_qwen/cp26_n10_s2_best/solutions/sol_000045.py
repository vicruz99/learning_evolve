# sol_000045 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000016 (state 585439f0) state=7d855308 sum of radii=2.613917 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def objective(vars):
    """Minimize negative sum of radii."""
    return -np.sum(vars[2*N:])

def compute_constraints(vars):
    """Compute all inequality constraints (must return values >= 0)."""
    x = vars[:N]
    y = vars[N:2*N]
    r = vars[2*N:]
    
    c = np.empty(4 * N + NUM_PAIRS)
    
    # Boundary constraints
    c[:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    # Pairwise non-overlap constraints
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    dists = np.sqrt(dx*dx + dy*dy)
    r_sum = r[I_IDX] + r[J_IDX]
    c[4*N:] = dists - r_sum
    
    return c

def generate_init(seed, init_type):
    """Generate a valid initial configuration."""
    np.random.seed(seed)
    if init_type == 'hex':
        r0 = 0.085
        centers = []
        y = r0
        row = 0
        while len(centers) < N:
            x_start = r0 if row % 2 == 0 else 2 * r0
            x = x_start
            while x <= 1 - r0:
                centers.append([x, y])
                x += 2 * r0
            y += np.sqrt(3) * r0
            row += 1
        centers = np.array(centers[:N])
        centers += np.random.uniform(-0.015, 0.015, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        return np.concatenate([centers.flatten(), np.full(N, 0.05)])
        
    elif init_type == 'grid':
        xs = np.linspace(0.12, 0.88, 6)
        ys = np.linspace(0.15, 0.85, 5)
        centers = []
        for yy in ys:
            for xx in xs:
                if len(centers) < N:
                    centers.append([xx, yy])
        centers = np.array(centers[:N])
        centers += np.random.uniform(-0.02, 0.02, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        return np.concatenate([centers.flatten(), np.full(N, 0.05)])
        
    else: # random sequential
        centers = np.random.uniform(0.15, 0.85, (N, 2))
        r = np.full(N, 0.03)
        for i in range(N):
            for j in range(i):
                d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                r[i] = min(r[i], d - r[j])
            r[i] = min(r[i], centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        return np.concatenate([centers.flatten(), r])

def run_packing():
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    best_sum = -1.0
    best_vars = None
    
    # Create diverse initial guesses
    inits = []
    for i in range(5):
        inits.append(generate_init(i, 'hex'))
        inits.append(generate_init(i, 'grid'))
        inits.append(generate_init(i, 'random'))
        
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            
            # Check improvement and feasibility
            current_sum = -res.fun
            if current_sum > best_sum:
                c_val = compute_constraints(res.x)
                if np.all(c_val >= -1e-7):
                    best_sum = current_sum
                    best_vars = res.x.copy()
        except Exception:
            continue
            
    # Extract best solution
    x = best_vars[:N]
    y = best_vars[N:2*N]
    r = best_vars[2*N:]
    
    # Strict boundary enforcement
    for i in range(N):
        r[i] = min(r[i], x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        
    # Iterative overlap resolution
    for _ in range(10):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j]:
                    shrink = (r[i] + r[j] - d) / 2.0 + 1e-8
                    r[i] = max(0.0, r[i] - shrink)
                    r[j] = max(0.0, r[j] - shrink)
                    changed = True
        if not changed:
            break
            
    centers = np.column_stack([x, y])
    return centers, r, float(np.sum(r))
