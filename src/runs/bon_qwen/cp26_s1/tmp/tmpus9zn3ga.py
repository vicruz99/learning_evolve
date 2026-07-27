import numpy as np
from scipy.optimize import minimize


def compute_energy(params, n, alpha):
    """Compute objective: -sum(radii) + alpha * overlap_penalty"""
    centers = np.empty((n, 2))
    radii = np.empty(n)
    
    for i in range(n):
        u = params[3 * i]
        v = params[3 * i + 1]
        r = params[3 * i + 2]
        radii[i] = r
        if r < 0.5:
            scale = 1.0 - 2.0 * r
            centers[i, 0] = r + u * scale
            centers[i, 1] = r + v * scale
        else:
            centers[i, 0] = 0.5
            centers[i, 1] = 0.5
    
    penalty = 0.0
    for i in range(n):
        ri = radii[i]
        xi = centers[i, 0]
        yi = centers[i, 1]
        for j in range(i + 1, n):
            rj = radii[j]
            dx = xi - centers[j, 0]
            dy = yi - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = ri + rj - dist
            if overlap > 0:
                penalty += overlap * overlap
    
    return -np.sum(radii) + alpha * penalty


def force_based_simulation(n, centers, radii, max_iter=80000):
    """Run force-based expansion simulation to get a good initial packing."""
    k_repulse = 500.0
    dt_expand = 0.00015
    dt_move = 0.002
    damping = 0.7
    
    velocities = np.zeros((n, 2))
    
    for iteration in range(max_iter):
        forces = np.zeros((n, 2))
        
        # Compute repulsive forces from overlaps
        for i in range(n):
            ri = radii[i]
            xi = centers[i, 0]
            yi = centers[i, 1]
            
            for j in range(i + 1, n):
                rj = radii[j]
                dx = centers[j, 0] - xi
                dy = centers[j, 1] - yi
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = ri + rj
                
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    force = k_repulse * overlap / dist
                    fx = force * dx
                    fy = force * dy
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy
            
            # Boundary repulsion
            if xi - ri < 0:
                forces[i, 0] += k_repulse * (ri - xi)
            if xi + ri > 1:
                forces[i, 0] -= k_repulse * (xi + ri - 1)
            if yi - ri < 0:
                forces[i, 1] += k_repulse * (ri - yi)
            if yi + ri > 1:
                forces[i, 1] -= k_repulse * (yi + ri - 1)
        
        # Update velocities with damping
        velocities = velocities * damping + forces * dt_move
        centers += velocities * dt_move
        
        # Enforce hard boundary constraints
        for i in range(n):
            r = radii[i]
            centers[i, 0] = np.clip(centers[i, 0], r, 1.0 - r)
            centers[i, 1] = np.clip(centers[i, 1], r, 1.0 - r)
        
        # Expand radii gradually
        radii += dt_expand
        
        # Every 500 iterations, reduce expansion rate
        if iteration > 0 and iteration % 10000 == 0:
            dt_expand *= 0.5
    
    return centers, radii


def initialize_hexagonal(n, pattern_index=0):
    """Initialize circles in a hexagonal-like pattern."""
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.025)
    
    # Different row arrangements for 26 circles
    patterns = [
        [6, 5, 6, 5, 4],
        [5, 6, 5, 6, 4],
        [5, 5, 6, 5, 5],
        [4, 6, 6, 6, 4],
        [6, 6, 6, 4, 4],
        [5, 5, 5, 6, 5],
    ]
    
    pattern = patterns[pattern_index % len(patterns)]
    num_rows = len(pattern)
    
    idx = 0
    for row in range(num_rows):
        num_cols = pattern[row]
        y = (row + 0.5) / num_rows
        for col in range(num_cols):
            if row % 2 == 0:
                x = (col + 0.5) / num_cols
            else:
                x = (col + 0.5) / num_cols + 0.5 / num_cols
            x = np.clip(x, 0.08, 0.92)
            centers[idx] = [x, y]
            idx += 1
    
    return centers, radii


def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    
    # Try multiple patterns and seeds
    for pattern_idx in range(6):
        for seed in range(5):
            np.random.seed(seed)
            
            # Initialize
            centers, radii = initialize_hexagonal(n, pattern_idx)
            
            # Add small random perturbation
            centers += np.random.randn(n, 2) * 0.01
            centers = np.clip(centers, 0.05, 0.95)
            
            # Run force-based simulation
            centers, radii = force_based_simulation(n, centers.copy(), radii.copy(), max_iter=50000)
            
            # Convert to optimization parameters
            params = np.zeros(3 * n)
            for i in range(n):
                r = radii[i]
                x, y = centers[i]
                if r < 0.5:
                    scale = 1.0 - 2.0 * r
                    u = (x - r) / scale if scale > 1e-10 else 0.5
                    v = (y - r) / scale if scale > 1e-10 else 0.5
                else:
                    u = 0.5
                    v = 0.5
                params[3 * i] = np.clip(u, 0, 1)
                params[3 * i + 1] = np.clip(v, 0, 1)
                params[3 * i + 2] = r
            
            # Bounds for optimization
            bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.44)] * n
            
            # Gradient-based refinement with high penalty
            result = minimize(
                compute_energy,
                params,
                args=(n, 5000000),
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 80000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            
            # Extract candidate solution
            cand_centers = np.zeros((n, 2))
            cand_radii = np.zeros(n)
            for i in range(n):
                u = result.x[3 * i]
                v = result.x[3 * i + 1]
                r = result.x[3 * i + 2]
                cand_radii[i] = r
                if r < 0.5:
                    scale = 1.0 - 2.0 * r
                    cand_centers[i, 0] = r + u * scale
                    cand_centers[i, 1] = r + v * scale
                else:
                    cand_centers[i, 0] = 0.5
                    cand_centers[i, 1] = 0.5
            
            # Check validity and update best
            valid = True
            for i in range(n):
                x, y = cand_centers[i]
                r = cand_radii[i]
                if x - r < -1e-8 or x + r > 1 + 1e-8 or y - r < -1e-8 or y + r > 1 + 1e-8:
                    valid = False
                    break
                for j in range(i + 1, n):
                    dist = np.sqrt(np.sum((cand_centers[i] - cand_centers[j]) ** 2))
                    if dist < cand_radii[i] + cand_radii[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                s = np.sum(cand_radii)
                if s > best_sum:
                    best_sum = s
                    best_centers = cand_centers.copy()
                    best_radii = cand_radii.copy()
    
    return best_centers, best_radii, best_sum