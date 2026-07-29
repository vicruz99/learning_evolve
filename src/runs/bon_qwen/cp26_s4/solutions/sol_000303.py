# sol_000303 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d28721c0) state=6841448e sum of radii=1.968691 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Run multiple trials to escape local optima
    for trial in range(4):
        centers = np.zeros((n, 2))
        idx = 0
        # Hexagonal packing pattern: 6, 5, 6, 5, 4 circles per row sums to 26
        pattern = [6, 5, 6, 5, 4]
        
        margin = 0.05
        spacing = 0.9 / 6.0
        
        for r_idx, count in enumerate(pattern):
            y = margin + r_idx * spacing * math.sqrt(3)
            x_start = margin + (6 - count) * spacing / 2.0
            for c_idx in range(count):
                if idx < n:
                    centers[idx, 0] = x_start + c_idx * spacing
                    centers[idx, 1] = y
                    idx += 1
                    
        # Add small random perturbation to break symmetry
        centers += np.random.randn(n, 2) * 0.01
        centers[:, 0] = np.clip(centers[:, 0], 0.05, 0.95)
        centers[:, 1] = np.clip(centers[:, 1], 0.05, 0.95)
        
        radii = np.ones(n) * 0.02
        
        # Iterative expansion and resolution
        for expand_step in range(300):
            radii += 0.00025
            
            lr = 0.05
            for step in range(40):
                grad = np.zeros_like(centers)
                
                # Compute repulsive forces for overlaps
                for i in range(n):
                    for j in range(i+1, n):
                        dx = centers[i, 0] - centers[j, 0]
                        dy = centers[i, 1] - centers[j, 1]
                        dist = math.hypot(dx, dy)
                        min_d = radii[i] + radii[j]
                        
                        if dist < min_d and dist > 1e-9:
                            overlap = min_d - dist
                            nx = dx / dist
                            ny = dy / dist
                            force = overlap * 2.0
                            grad[i, 0] += nx * force
                            grad[i, 1] += ny * force
                            grad[j, 0] -= nx * force
                            grad[j, 1] -= ny * force
                            
                # Boundary repulsion
                for i in range(n):
                    for dim in range(2):
                        if centers[i, dim] < radii[i]:
                            grad[i, dim] += (radii[i] - centers[i, dim]) * 5.0
                        if centers[i, dim] > 1 - radii[i]:
                            grad[i, dim] -= (centers[i, dim] - (1 - radii[i])) * 5.0
                            
                centers += lr * grad
                centers[:, 0] = np.clip(centers[:, 0], 0.001, 0.999)
                centers[:, 1] = np.clip(centers[:, 1], 0.001, 0.999)
                lr *= 0.95
                
            # Check validity
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    if math.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1]) < radii[i]+radii[j] - 1e-7:
                        valid = False
                        break
                if not valid: break
                if centers[i,0]-radii[i] < -1e-7 or centers[i,0]+radii[i] > 1+1e-7 or \
                   centers[i,1]-radii[i] < -1e-7 or centers[i,1]+radii[i] > 1+1e-7:
                    valid = False
                    break
                    
            if not valid:
                radii -= 0.0001
                
        # Final shrinkage to guarantee strict validity
        for _ in range(100):
            s = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    d = math.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                    if d < radii[i]+radii[j]:
                        s = max(s, (radii[i]+radii[j]-d)/2 + 1e-7)
                for dim in range(2):
                    if centers[i,dim] < radii[i]: s = max(s, radii[i]-centers[i,dim]+1e-7)
                    if centers[i,dim] > 1-radii[i]: s = max(s, centers[i,dim]-(1-radii[i])+1e-7)
            if s > 0: radii -= s
            else: break
            
        radii = np.maximum(radii, 0)
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
    return best_centers, best_radii, best_sum
