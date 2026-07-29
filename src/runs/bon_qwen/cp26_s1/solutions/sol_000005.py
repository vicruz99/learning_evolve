# sol_000005 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ada29bac) state=6c19cc72 sum of radii=2.599917 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Target configuration: 5-6-5-6-4 staggered layout
    # Estimated initial radius for this layout is around 0.095-0.098
    r_init = 0.096
    spacing = 2 * r_init
    row_spacing = spacing * np.sqrt(3) / 2
    
    centers = []
    # 5 rows layout: 5, 6, 5, 6, 4 circles
    # Row 1 (5 circles)
    y1 = r_init
    x1 = np.linspace(r_init, 1 - r_init, 5)
    centers.extend([(x, y1) for x in x1])
    
    # Row 2 (6 circles) - staggered
    y2 = y1 + row_spacing
    x2 = np.linspace(r_init + r_init, 1 - r_init - r_init, 6) 
    # Adjusted span to fit 6 circles
    width_6 = 1 - 2*r_init
    step_6 = width_6 / 5 
    # Centers for 6 circles: start at r_init + r_init/2 ? 
    # Let's just space them evenly within [r_init, 1-r_init]
    # But they need to be staggered relative to row 1.
    # Row 1 centers: r, r+s, r+2s, r+3s, r+4s
    # Row 2 centers: r+s/2, r+s/2+s, ...
    # Let's recalculate to ensure they are within bounds and staggered.
    
    # Reset and use a robust grid generation for hexagonal packing
    centers = []
    
    # Layout parameters
    rows_config = [5, 6, 5, 6, 4]
    r_est = 0.095
    
    y_pos = r_est
    for count in rows_config:
        x_positions = np.linspace(r_est, 1 - r_est, count)
        # Stagger odd-indexed rows (1, 3, 5 in 0-indexed list: 1, 3)
        # Actually, let's stagger rows 1, 3 (indices 1, 3)
        if len(centers) % 2 == 1: # Row indices 1, 3 (0, 1, 2, 3, 4)
             # Shift by half step?
             # Simple approach: linspace with count, but shifted?
             # Better: generate grid and let optimizer fix it.
             pass
        centers.extend(list(zip(x_positions, [y_pos]*count)))
        y_pos += row_spacing
        
    # If we didn't generate 26, adjust. The config 5+6+5+6+4 = 26.
    # Re-generate carefully
    centers = []
    r_est = 0.09
    dy = 2 * r_est * np.sqrt(3) / 2
    
    y_curr = r_est
    # Row 0: 5
    xs = np.linspace(r_est, 1-r_est, 5)
    centers.extend(zip(xs, [y_curr]*5))
    
    # Row 1: 6
    y_curr += dy
    # To fit 6, we might need to compress or shift. 
    # 6 circles of radius 0.09 need width 12*0.09 = 1.08 > 1.
    # So 6 circles of radius 0.09 is impossible in a straight line.
    # We rely on the optimizer to adjust radii and positions.
    # Let's place them roughly evenly.
    xs = np.linspace(r_est, 1-r_est, 6) 
    centers.extend(zip(xs, [y_curr]*6))
    
    # Row 2: 5
    y_curr += dy
    xs = np.linspace(r_est, 1-r_est, 5)
    centers.extend(zip(xs, [y_curr]*5))
    
    # Row 3: 6
    y_curr += dy
    xs = np.linspace(r_est, 1-r_est, 6)
    centers.extend(zip(xs, [y_curr]*6))
    
    # Row 4: 4
    y_curr += dy
    xs = np.linspace(r_est, 1-r_est, 4)
    centers.extend(zip(xs, [y_curr]*4))
    
    centers = np.array(centers)
    radii = np.full(n, r_est)
    
    # Optimization
    def objective(vars):
        # vars: [x1, y1, r1, x2, y2, r2, ...]
        # We want to maximize sum of radii -> minimize negative sum
        r = vars[2::3]
        return -np.sum(r)

    def constraint_boundary(vars):
        # x - r >= 0 => r - x <= 0
        # 1 - x - r >= 0 => x + r - 1 <= 0
        # Same for y
        c = []
        for i in range(n):
            idx = 3*i
            x, y, r = vars[idx], vars[idx+1], vars[idx+2]
            c.append(x - r) # x >= r
            c.append(1 - x - r) # x + r <= 1
            c.append(y - r) # y >= r
            c.append(1 - y - r) # y + r <= 1
            c.append(r) # r >= 0
        return c

    def constraint_overlap(vars):
        c = []
        for i in range(n):
            for j in range(i + 1, n):
                idx_i = 3*i
                idx_j = 3*j
                xi, yi, ri = vars[idx_i], vars[idx_i+1], vars[idx_i+2]
                xj, yj, rj = vars[idx_j], vars[idx_j+1], vars[idx_j+2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                # dist >= ri + rj  => dist^2 >= (ri + rj)^2
                # But for inequality constraint g(x) >= 0, we can use dist - (ri+rj) >= 0
                # However, sqrt is expensive. 
                # dist_sq - (ri + rj)^2 >= 0
                c.append(dist_sq - (ri + rj)**2)
        return c

    # Initial guess
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]

    bounds = [(0, 1) if i % 3 != 2 else (0, 0.5) for i in range(3*n)]
    # Radii can't be larger than 0.5

    constraints = []
    
    # Boundary constraints
    cons = {'type': 'ineq', 'fun': constraint_boundary}
    constraints.append(cons)
    
    # Overlap constraints
    cons_overlap = {'type': 'ineq', 'fun': constraint_overlap}
    constraints.append(cons_overlap)

    # Optimize
    # SLSQP is good for constrained problems
    try:
        res = scipy.optimize.minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        final_vars = res.x
    except Exception as e:
        # Fallback to initial guess if optimization fails
        final_vars = x0

    # Extract results
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    for i in range(n):
        idx = 3*i
        final_centers[i, 0] = final_vars[idx]
        final_centers[i, 1] = final_vars[idx+1]
        final_radii[i] = final_vars[idx+2]

    # Ensure radii are non-negative (clipping small negative values due to numerical error)
    final_radii = np.maximum(final_radii, 0)
    
    # Final check and correction
    # If any radius is effectively zero, it's fine, but we want valid packing.
    # The optimizer should handle this.
    
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

if __name__ == "__main__":
    centers, radii, total = run_packing()
    print(f"Sum of radii: {total}")
    # Basic validation print
    for i in range(len(radii)):
        print(f"Circle {i}: ({centers[i,0]:.4f}, {centers[i,1]:.4f}), r={radii[i]:.4f}")
