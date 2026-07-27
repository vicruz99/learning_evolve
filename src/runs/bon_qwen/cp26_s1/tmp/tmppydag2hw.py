import numpy as np
import math

def run_packing():
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_radii: float sum of all radii
    """
    n_circles = 26
    
    # 1. Initialization: Hexagonal lattice pattern
    # This provides a good initial dense configuration
    points = []
    # Generate points in a hexagonal pattern
    # Spacing 2.0 (corresponds to radius 1.0 for touching circles)
    for j in range(10):
        for i in range(10):
            x = i * 2.0 + (j % 2) * 1.0
            y = j * math.sqrt(3)
            points.append((x, y))
    
    # Select first 26 points
    selected_points = np.array(points[:n_circles])
    
    # Normalize to fit in [0.05, 0.95] to leave margin
    x_min, x_max = selected_points[:, 0].min(), selected_points[:, 0].max()
    y_min, y_max = selected_points[:, 1].min(), selected_points[:, 1].max()
    
    if x_max == x_min: x_max += 1
    if y_max == y_min: y_max += 1
    
    # Scale to fit in width 0.9 (leaving 0.05 margin on each side)
    scale_x = 0.9 / (x_max - x_min)
    scale_y = 0.9 / (y_max - y_min)
    scale = min(scale_x, scale_y)
    
    cx = (x_max + x_min) / 2.0
    cy = (y_max + y_min) / 2.0
    
    centers = np.zeros((n_circles, 2))
    for k in range(n_circles):
        dx = selected_points[k, 0] - cx
        dy = selected_points[k, 1] - cy
        centers[k, 0] = 0.5 + dx * scale
        centers[k, 1] = 0.5 + dy * scale
    
    centers = np.clip(centers, 0, 1)
    
    # Initial radii: small to ensure no overlap
    radii = np.ones(n_circles) * 0.02
    
    # 2. Optimization Loop
    max_iter = 10000
    expansion_rate = 0.0004
    damping = 0.5
    repulsion_strength = 10.0
    
    for iteration in range(max_iter):
        # Check overlaps and compute forces
        max_overlap = 0.0
        moves = np.zeros_like(centers)
        
        # Boundary constraints
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            # Left
            if x - r < 0:
                moves[i, 0] += r - x
                if r - x > max_overlap: max_overlap = r - x
            # Right
            elif x + r > 1:
                moves[i, 0] -= (x + r - 1)
                if x + r - 1 > max_overlap: max_overlap = x + r - 1
            # Bottom
            if y - r < 0:
                moves[i, 1] += r - y
                if r - y > max_overlap: max_overlap = r - y
            # Top
            elif y + r > 1:
                moves[i, 1] -= (y + r - 1)
                if y + r - 1 > max_overlap: max_overlap = y + r - 1
        
        # Circle-Circle constraints
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = math.sqrt(dist_sq)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    if overlap > max_overlap:
                        max_overlap = overlap