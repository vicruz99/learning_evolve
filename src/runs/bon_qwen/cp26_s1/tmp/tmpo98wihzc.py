import numpy as np
from scipy.optimize import minimize

def get_hexagonal_centers(n, r_init, square_size=1.0):
    """
    Generates initial centers for n circles in a hexagonal packing pattern.
    """
    centers = []
    # Estimate number of rows needed
    # Approximate area per circle in hex packing is 2*sqrt(3)*r^2
    # Total area n * 2*sqrt(3)*r^2 approx square_size^2
    # This is a rough estimate, we will just generate until we have n
    # or fit within bounds.
    
    # Heuristic for rows and cols
    # For n=26, maybe 6 rows?
    # Let's try to fit as many as possible in a hex grid
    
    # Vertical spacing in hex packing: r * sqrt(3)
    # Horizontal spacing: 2*r
    # But we will just place them with initial r_init and let optimizer scale.
    
    # Let's construct a grid
    row = 0
    while len(centers) < n:
        # x starts at r_init
        x = r_init
        # Shift x for odd rows to create hex pattern
        if row % 2 == 1:
            x = r_init + r_init # shift by r? No, shift by r makes horizontal dist r?
            # In hex packing, horizontal distance between neighbors in same row is 2r.
            # Vertical distance is r*sqrt(3).
            # Horizontal shift between rows is r.
            pass 
        
        # Actually, standard hex lattice:
        # Row i: y = r + i * r * sqrt(3)
        # Cols in even rows: x = r, 3r, 5r...
        # Cols in odd rows: x = 2r, 4r, 6r... (shifted by r)
        
        y = r_init + row * r_init * np.sqrt(3)
        
        if y + r_init > square_size:
            break
            
        # Determine x start
        if row % 2 == 1:
            x_start = r_init * 2 # Shifted
        else:
            x_start = r_init
            
        col = 0
        x = x_start
        while x + r_init <= square_size and len(centers) < n:
            centers.append([x, y])
            x += 2 * r_init
            col += 1
        row += 1
        
    # If we didn't get enough, just add random points near center
    # But for 26 and r=0.09, we should get enough.
    # 0.09 + 5*0.09*1.732 = 0.09 + 0.78 = 0.87 < 1. So 6 rows fit.
    # Width: 0.09 + 4*0.18 = 0.81 < 1. So 5 cols fit.
    # 6*5 = 30. So we will have plenty.
    
    return np.array(centers[:n])

def validate_packing(centers, radii):
    """
    Validation function provided in prompt
    """
    n = centers.shape[0]
    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
        elif np.isnan(radii[i]):
            return False
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

def objective(vars, n):
    # vars is flat array [x1, y1, r1, x2, y2, r2, ...]
    # We want to maximize sum(r), so minimize -sum(r)
    radii = vars[2::3]
    return -np.sum(radii)

def constraints_builder(n):
    constraints = []
    
    # Boundary constraints
    # x_i - r_i >= 0  =>  vars[3*i] - vars[3*i+2] >= 0
    # 1 - x_i - r_i >= 0 => 1 - vars[3*i] - vars[3*i+2] >= 0
    # y_i - r_i >= 0 => vars[3*i+1] - vars[3*i+2] >= 0
    # 1 - y_i - r_i >= 0 => 1 - vars[3*i+1] - vars[3*i+2] >= 0
    
    # Note: SLSQP expects func(x) >= 0
    # So x - r >= 0 is fine.
    
    for i in range(n):
        idx_x = 3 * i
        idx_y = 3 * i + 1
        idx_r = 3 * i + 2
        
        # x >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx_x] - v[idx_r]
        })
        # 1 - x >= r  =>  1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[idx_x] - v[idx_r]
        })
        # y >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx_y] - v[idx_r]
        })
        # 1 - y >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[idx_y] - v[idx_r]
        })
        
        # r >= 0 (handled by bounds, but can add constraint if needed. Bounds are better)
        
    # Overlap constraints
    # dist(i, j) >= r_i + r_j
    # sqrt((xi-xj)^2 + (yi-yj)^2) - (ri + rj) >= 0
    # To avoid sqrt singularity, we can use squared form?
    # dist^2 >= (ri+rj)^2
    # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    # This is smooth and differentiable.
    
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi = 3 * i
            idx_yi = 3 * i + 1
            idx_ri = 3 * i + 2
            idx_xj = 3 * j
            idx_yj = 3 * j + 1
            idx_rj = 3 * j + 2
            
            def make_ineq(i_idx_x, i_idx_y, i_idx_r, j_idx_x, j_idx_y, j_idx_r):
                def ineq(v):
                    dx = v[i_idx_x] - v[j_idx_x]
                    dy = v[i_idx_y] - v[j_idx_y]
                    ri = v[i_idx_r]
                    rj = v[j_idx_r]
                    return dx*dx + dy*dy - (ri + rj)*(ri + rj)
                return ineq
            
            constraints.append({
                'type': 'ineq',
                'fun': make_ineq(idx_xi, idx_yi, idx_ri, idx_xj, idx_yj, idx_rj)
            })
            
    return constraints

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    square_size = 1.0
    
    # Initial radius guess. 0.09 is safe.
    r_init = 0.09
    
    # Generate initial centers
    centers_init = get_hexagonal_centers(n, r_init, square_size)
    
    # Construct initial variables vector
    # [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[3 * i] = centers_init[i, 0]
        x0[3 * i + 1] = centers_init[i, 1]
        x0[3 * i + 2] = r_init
        
    # Bounds
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    constraints = constraints_builder(n)
    
    # Optimization options
    # SLSQP is generally good for this.
    # maxiter can be increased if needed.
    opt_res = minimize(
        objective,
        x0,
        args=(n,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False}
    )
    
    # Extract results
    best_x = opt_res.x
    
    # There might be cases where optimizer fails or finds poor local minima.
    # We can try to improve by restarting or perturbation if needed, 
    # but with good init, one run might suffice.
    # Let's add a simple check and maybe a second run with perturbation if result is poor?
    # However, for the purpose of this task, let's trust the optimizer.
    
    # Reshape
    centers = best_x.reshape(n, 3)[:, :2]
    radii = best_x.reshape(n, 3)[:, 2]
    
    # Ensure radii are non-negative (optimizer should respect bounds)
    radii = np.maximum(radii, 0.0)
    
    # Ensure strict validity (fix minor numerical errors)
    # If validation fails, we might need to shrink radii slightly.
    # But the constraints should enforce it.
    # Let's verify.
    if not validate_packing(centers, radii):
        # Fallback: shrink radii slightly to fix overlaps/boundaries
        # This is a safety measure.
        scale = 0.999
        while not validate_packing(centers, radii * scale) and scale > 0.5:
            scale -= 0.01
        radii = radii * scale
        # Re-verify
        if not validate_packing(centers, radii):
            # If still fails, force valid state by reducing radii more aggressively
            # or just return what we have (might fail validation check by user).
            # Let's try to fix by reducing radii until valid.
            r_factor = 1.0
            while not validate_packing(centers, radii * r_factor) and r_factor > 0.0:
                r_factor *= 0.9
            radii = radii * r_factor

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# To run and test locally (not part of submission)
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(c, r)}")