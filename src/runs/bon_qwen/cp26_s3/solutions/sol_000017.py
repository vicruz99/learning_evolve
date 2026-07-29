# sol_000017 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0f0997f0) state=c5a3ccf2 sum of radii=0.260000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # Objective: minimize negative sum of radii
    def objective(x_vars):
        # x_vars is [x1, y1, r1, x2, y2, r2, ...]
        # radii are at indices 2, 5, 8, ...
        return -np.sum(x_vars[2::3])

    # Constraints
    def make_constraints(n):
        constraints = []
        # Boundary constraints
        for i in range(n):
            # x >= r  => x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
            # x <= 1-r => 1 - r - x >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i+2] - v[3*i]})
            # y >= r
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
            # y <= 1-r
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i+2] - v[3*i+1]})
            # r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2]})

        # Overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                def overlap(v, i=i, j=j):
                    xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                    xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                    # (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
                    return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                constraints.append({'type': 'ineq', 'fun': overlap})
        return constraints

    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    constraints_list = make_constraints(n)

    # Helper to generate initial guesses
    def get_guesses():
        guesses = []
        
        # Guess 1: Random valid start (small radii)
        np.random.seed(42)
        centers = np.random.rand(26, 2)
        radii = np.ones(26) * 0.01
        guesses.append((centers, radii))

        # Guess 2: Grid based (5x5 + 1)
        centers = []
        radii = []
        r0 = 0.05
        for i in range(5):
            for j in range(5):
                centers.append([0.1 + i*0.2, 0.1 + j*0.2])
                radii.append(r0)
        centers.append([0.2, 0.2]) # 26th circle
        radii.append(r0)
        guesses.append((np.array(centers), np.array(radii)))

        # Guess 3: Hexagonal-ish dense packing
        centers = []
        radii = []
        r0 = 0.06
        rows_counts = [5, 4, 5, 4, 5, 3]
        y = r0
        for i, count in enumerate(rows_counts):
            if count > 1:
                span = 1.0 - 2*r0
                step = span / (count - 1)
                xs = [r0 + k*step for k in range(count)]
            else:
                xs = [0.5]
            for x in xs:
                centers.append([x, y])
                radii.append(r0)
            y += np.sqrt(3) * r0
        
        # Ensure 26 circles
        while len(centers) < 26:
            centers.append([0.5, 0.5])
            radii.append(r0)
        centers = np.array(centers[:26])
        radii = np.array(radii[:26])
        guesses.append((centers, radii))
        
        return guesses

    guesses = get_guesses()
    
    for k, (init_c, init_r) in enumerate(guesses):
        # Flatten
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = init_c[i, 0]
            x0[3*i+1] = init_c[i, 1]
            x0[3*i+2] = init_r[i]
        
        # Clip to bounds
        x0[0::3] = np.clip(x0[0::3], 0.0, 1.0)
        x0[1::3] = np.clip(x0[1::3], 0.0, 1.0)
        x0[2::3] = np.clip(x0[2::3], 0.0, 0.5)

        try:
            res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, 
                              constraints=constraints_list, 
                              options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            if res.success:
                current_sum = -res.fun
                # Extract
                c = res.x[0::3].reshape(-1, 2)
                r = res.x[2::3]
                
                # Validate strict feasibility for safety
                valid = True
                for i in range(n):
                    if r[i] < 1e-9: valid = False
                    if c[i,0] < r[i] + 1e-9 or c[i,0] > 1-r[i] - 1e-9: valid = False
                    if c[i,1] < r[i] + 1e-9 or c[i,1] > 1-r[i] - 1e-9: valid = False
                
                if valid:
                    for i in range(n):
                        for j in range(i+1, n):
                            d2 = (c[i,0]-c[j,0])**2 + (c[i,1]-c[j,1])**2
                            if d2 < (r[i]+r[j])**2 - 1e-9:
                                valid = False
                                break
                        if not valid: break
                
                if valid and current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = c
                    best_radii = r
        except Exception:
            pass

    if best_centers is None:
        # Fallback
        centers = np.zeros((26, 2))
        radii = np.ones(26) * 0.01
        idx = 0
        for i in range(5):
            for j in range(5):
                centers[idx] = [0.1 + i*0.2, 0.1 + j*0.2]
                idx += 1
        if idx < 26:
            centers[idx] = [0.2, 0.2]
        best_sum = 0.26
        best_centers = centers
        best_radii = radii

    return best_centers, best_radii, float(best_sum)
