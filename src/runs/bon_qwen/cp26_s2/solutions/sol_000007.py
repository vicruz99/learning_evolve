# sol_000007 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f294fc76) state=adad36c7 sum of radii=2.607995 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(state):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(state[2::3])

def boundary_constraints(state):
    """Ensure all circles remain inside the unit square."""
    N = 26
    out = np.empty(4 * N)
    for i in range(N):
        i3 = 3 * i
        out[4*i]   = state[i3] - state[i3+2]          # x - r >= 0
        out[4*i+1] = 1.0 - state[i3] - state[i3+2]    # x + r <= 1
        out[4*i+2] = state[i3+1] - state[i3+2]        # y - r >= 0
        out[4*i+3] = 1.0 - state[i3+1] - state[i3+2]  # y + r <= 1
    return out

def overlap_constraints(state):
    """Ensure no two circles overlap."""
    N = 26
    num_pairs = N * (N - 1) // 2
    out = np.empty(num_pairs)
    idx = 0
    for i in range(N):
        i3 = 3 * i
        xi, yi, ri = state[i3], state[i3+1], state[i3+2]
        for j in range(i + 1, N):
            j3 = 3 * j
            dx = xi - state[j3]
            dy = yi - state[j3+1]
            out[idx] = dx*dx + dy*dy - (ri + state[j3+2])**2
            idx += 1
    return out

def run_packing():
    N = 26
    best_state = None
    best_sum = -1.0
    
    # Variable bounds: x in [0,1], y in [0,1], r in [1e-6, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    
    cons = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': overlap_constraints}
    ]
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Standard 5x5 grid + 1 center circle
    c1 = []
    for i in range(5):
        for j in range(5):
            c1.append((0.1 + i*0.2, 0.1 + j*0.2))
    c1.append((0.5, 0.5))
    inits.append(c1)
    
    # 2. Staggered hexagonal-like layout
    c2 = []
    ys = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
    for k, y in enumerate(ys):
        n_in_row = 6 if k % 2 == 0 else 5
        x_start = 0.05 if n_in_row == 6 else 0.1
        for m in range(n_in_row):
            x = x_start + m * (0.9 / (n_in_row - 1)) if n_in_row > 1 else 0.5
            c2.append((x, y))
        if len(c2) >= 26:
            break
    inits.append(c2[:26])
    
    # 3. Randomized starts to escape local optima
    np.random.seed(42)
    for _ in range(5):
        c_rand = []
        for _ in range(26):
            c_rand.append((np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)))
        inits.append(c_rand)

    # Optimization loop over multiple starts
    for start_coords in inits:
        state0 = np.zeros(3 * N)
        for i, (x, y) in enumerate(start_coords):
            state0[3*i] = x
            state0[3*i+1] = y
            state0[3*i+2] = 0.05
            
        # Add small perturbation
        state0 += np.random.normal(0, 0.005, state0.shape)
        
        try:
            res = minimize(objective, state0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            
            current_sum = -res.fun
            if current_sum > best_sum:
                # Verify feasibility within tolerance
                bc = boundary_constraints(res.x)
                oc = overlap_constraints(res.x)
                if np.all(bc >= -1e-7) and np.all(oc >= -1e-7):
                    best_state = res.x.copy()
                    best_sum = current_sum
        except Exception:
            continue
            
    # Fallback to a valid grid packing if optimization fails
    if best_state is None:
        best_state = np.zeros(3 * N)
        for i in range(25):
            best_state[3*i] = 0.1 + (i % 5) * 0.2
            best_state[3*i+1] = 0.1 + (i // 5) * 0.2
            best_state[3*i+2] = 0.1
        best_state[3*25] = 0.5
        best_state[3*25+1] = 0.5
        best_state[3*25+2] = 0.05

    # Extract results
    centers = np.array([[best_state[3*i], best_state[3*i+1]] for i in range(N)])
    radii = best_state[2::3].copy()
    
    # Final safety margin to guarantee validator passes
    bc = boundary_constraints(best_state)
    oc = overlap_constraints(best_state)
    if np.any(bc < 1e-8) or np.any(oc < 1e-8):
        radii *= 0.9999
        best_state[2::3] = radii
        
    return centers, radii, float(np.sum(radii))
