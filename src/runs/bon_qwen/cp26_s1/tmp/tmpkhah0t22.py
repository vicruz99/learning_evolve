import numpy as np
from scipy.optimize import minimize
import math
import copy

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # --- Helper Functions ---
    
    def calculate_energy(centers, radii, penalty_weight=1000.0):
        """
        Calculates the objective function value.
        Minimizes: -sum(radii) + penalty_weight * (overlap_penalty + boundary_penalty)
        """
        # Objective: minimize negative sum of radii (maximize sum)
        obj = -np.sum(radii)
        
        # Overlap penalty
        overlap_penalty = 0.0
        for i in range(n):
            xi, yi = centers[i]
            ri = radii[i]
            # Boundary penalties
            # Left: x - r >= 0 => r - x <= 0. Violation if r > x
            if ri > xi:
                overlap_penalty += (ri - xi) ** 2
            # Right: x + r <= 1 => r - (1-x) <= 0. Violation if r > 1-x
            if ri > (1.0 - xi):
                overlap_penalty += (ri - (1.0 - xi)) ** 2
            # Bottom: y - r >= 0 => r - y <= 0. Violation if r > y
            if ri > yi:
                overlap_penalty += (ri - yi) ** 2
            # Top: y + r <= 1 => r - (1-y) <= 0. Violation if r > 1-y
            if ri > (1.0 - yi):
                overlap_penalty += (ri - (1.0 - yi)) ** 2
            
            # Pairwise overlap penalties (only check j > i to avoid double counting)
            for j in range(i + 1, n):
                xj, yj = centers[j]
                rj = radii[j]
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                sum_radii = ri + rj
                if dist < sum_radii:
                    overlap_penalty += (sum_radii - dist) ** 2
        
        return obj + penalty_weight * overlap_penalty

    def get_valid_params(centers, radii):
        """Flattens centers and radii into a single vector for optimizer."""
        return np.concatenate([centers.flatten(), radii])

    def set_valid_params(params, centers, radii):
        """Updates centers and radii from the flattened vector."""
        centers[:] = params[:2*n].reshape((n, 2))
        radii[:] = params[2*n:]

    def run_optimization(initial_centers, initial_radii, seed=0):
        """
        Runs a single optimization instance.
        """
        np.random.seed(seed)
        
        # Perturb initial centers slightly to avoid symmetry traps
        # But keep them within valid bounds roughly
        perturbation_scale = 0.005
        current_centers = initial_centers.copy() + np.random.uniform(-perturbation_scale, perturbation_scale, size=(n, 2))
        # Clamp to [0, 1]
        current_centers = np.clip(current_centers, 0, 1)
        
        current_radii = initial_radii.copy()
        
        # Define bounds for optimizer
        # x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for _ in range(n):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((1e-5, 0.5)) # r (small positive lower bound)
            
        initial_params = get_valid_params(current_centers, current_radii)
        
        # Use L-BFGS-B
        try:
            res = minimize(
                calculate_energy,
                initial_params,
                args=(1000.0,), # penalty weight
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-9}
            )
            
            final_params = res.x
            final_centers = final_params[:2*n].reshape((n, 2))
            final_radii = final_params[2*n:]
            
            return final_centers, final_radii
            
        except Exception:
            return None, None

    # --- Main Logic ---

    # 1. Generate initial hexagonal packing
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.09)
    
    row_radius = 0.09
    dy = math.sqrt(3) * row_radius
    dx = 2 * row_radius
    
    y = row_radius
    count = 0
    row_idx = 0
    
    # Fill grid
    # We need 26 circles. 
    # Pattern: 5, 4, 5, 4, 5, 3 (sum 26) or similar.
    # Let's just iterate rows until we have 26.
    
    while count < n:
        x = row_radius
        # Shift x for odd rows (1, 3, 5...)
        if row_idx % 2 == 1:
            x += row_radius
        
        # Place circles in this row
        while x + row_radius <= 1.0 and count < n:
            centers[count, 0] = x
            centers[count, 1] = y
            x += dx
            count += 1
        
        y += dy
        row_idx += 1
        
    # Adjust initial radii to be safe
    # A radius of 0.09 might be too large for this specific layout near boundaries
    # Let's reduce it slightly to ensure valid start
    initial_radii = np.full(n, 0.08)

    best_centers = None
    best_radii = None
    best_sum = -1.0

    # 2. Run optimization multiple times with different seeds
    num_runs = 10
    for i in range(num_runs):
        c, r = run_optimization(centers, initial_radii, seed=i)
        if c is not None:
            current_sum = np.sum(r)
            # Check validity roughly (energy should be low)
            # We can re-check with the validation function logic manually or just trust low energy
            # But to be safe, let's compute a validity score
            is_valid = True
            
            # Check bounds
            for k in range(n):
                if r[k] < 0 or c[k][0] < 0 or c[k][0] > 1 or c[k][1] < 0 or c[k][1] > 1:
                    is_valid = False
                    break
                if c[k][0] - r[k] < -1e-5 or c[k][0] + r[k] > 1 + 1e-5 or \
                   c[k][1] - r[k] < -1e-5 or c[k][1] + r[k] > 1 + 1e-5:
                    is_valid = False
                    break
            
            if is_valid:
                # Check overlaps
                for k in range(n):
                    for m in range(k + 1, n):
                        dist = np.sqrt(np.sum((c[k] - c[m])**2))
                        if dist < r[k] + r[m] - 1e-5:
                            is_valid = False
                            break
                    if not is_valid:
                        break
            
            if is_valid and current_sum > best_sum:
                best_sum = current_sum
                best_centers = c.copy()
                best_radii = r.copy()

    # Fallback to initial if optimization failed (unlikely)
    if best_centers is None:
        return centers, initial_radii, np.sum(initial_radii)
        
    return best_centers, best_radii, best_sum

# Validation function provided in prompt (for context, not included in solution block)
import numpy as np # ensure np is available

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

# To run and print result
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(c, r)}")