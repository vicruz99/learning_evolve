# sol_000340 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6a87b209) state=e45fa2f6 sum of radii=1.235000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def validate_packing(centers, radii):
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0: return False
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialize centers in a hexagonal pattern
    r_init = 0.09
    centers = []
    row, col = 0, 0
    
    while len(centers) < n:
        # Hexagonal spacing
        x = col * 2 * r_init * np.cos(np.pi/6) + (row % 2) * r_init * np.cos(np.pi/6)
        y = row * 2 * r_init * np.sin(np.pi/6)
        
        if 0 <= x <= 1 and 0 <= y <= 1:
            centers.append([x, y])
        
        col += 1
        if col > 5: # Reset row if too wide
            col = 0
            row += 1
            # Shift x back for next row logic
            if row > 10: break

    centers = np.array(centers[:n])
    # Shift to center in [0,1]
    centers = centers - centers.min(axis=0) + 0.1

    radii = np.full(n, r_init)
    
    # 2. Optimization loop
    # We maximize the sum of radii by pushing circles apart
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    # Use a simple force-based relaxation to find local optimum
    # We want to maximize r such that dist(i,j) >= 2r
    # Equivalently, minimize overlap for a fixed large r, then increase r
    
    current_r = 0.095
    
    # Initial relaxation
    for _ in range(500):
        # Compute repulsive forces
        forces = np.zeros_like(centers)
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 2 * current_r:
                    # Repel
                    overlap = (2 * current_r - dist) / (2 * current_r)
                    force = diff / (dist + 1e-6) * overlap * 10.0
                    forces[i] += force
                    forces[j] -= force
        
        # Update positions
        centers += forces * 0.01
        
        # Clamp to boundary [current_r, 1-current_r]
        centers = np.clip(centers, current_r, 1 - current_r)
        
    # 3. Gradual increase of radius
    while current_r < 0.11:
        increased = False
        for _ in range(200):
            forces = np.zeros_like(centers)
            for i in range(n):
                for j in range(i + 1, n):
                    diff = centers[i] - centers[j]
                    dist = np.linalg.norm(diff)
                    req_dist = 2 * current_r
                    if dist < req_dist:
                        overlap = (req_dist - dist) / req_dist
                        force = diff / (dist + 1e-6) * overlap * 10.0
                        forces[i] += force
                        forces[j] -= force
            
            centers += forces * 0.005
            centers = np.clip(centers, current_r, 1 - current_r)
            
        # Check if we can increase radius
        min_dist = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                min_dist = min(min_dist, dist)
            # Check boundary
            for d in range(2):
                min_dist = min(min_dist, centers[i][d] - current_r)
                min_dist = min(min_dist, (1 - current_r) - centers[i][d])
        
        if min_dist > 1e-4:
            current_r += 0.0001
            increased = True
            
        if not increased:
            break
            
    best_centers = centers
    best_radii = np.full(n, current_r)
    
    # 4. Final refinement using scipy to maximize sum of radii
    # Since we likely have equal radii, we can optimize the single parameter r
    # but the geometry might benefit from slight variations.
    # Let's assume equal radii for the final polish as it's robust.
    
    def objective(params):
        # params is flat array of centers
        c = np.reshape(params, (n, 2))
        # We want to maximize r, so we minimize -r. 
        # But r is not in params. 
        # Instead, we define a function that checks feasibility for a given r.
        pass

    # Actually, a simpler approach:
    # Fix the current configuration, calculate the max possible r.
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(best_centers[i] - best_centers[j])
            min_dist = min(min_dist, dist / 2)
        for d in range(2):
            min_dist = min(min_dist, (best_centers[i][d]) / 2) # dist to 0
            min_dist = min(min_dist, (1 - best_centers[i][d]) / 2) # dist to 1
            
    final_r = min_dist
    final_radii = np.full(n, final_r)
    final_centers = best_centers
    
    # Validation check
    if not validate_packing(final_centers, final_radii):
        # Fallback to a simple grid if optimization failed
        r = 0.09
        final_centers = np.zeros((26, 2))
        final_radii = np.full(26, r)
        idx = 0
        for i in range(5):
            for j in range(6):
                if idx < 26:
                    final_centers[idx] = [0.1 + i*0.18, 0.1 + j*0.16]
                    idx += 1
        
    return final_centers, final_radii, np.sum(final_radii)

# Execute to verify
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(c, r)}")
