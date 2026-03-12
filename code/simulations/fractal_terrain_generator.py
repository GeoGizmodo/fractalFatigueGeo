#!/usr/bin/env python3
"""
Fractal Terrain Generation using Fractional Brownian Motion
Based on the technical report's fractal analysis methodology
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class FractalTerrainGenerator:
    """Generate synthetic terrain with controlled fractal dimensions"""
    
    def __init__(self, size=512, pixel_size=10.0):
        """
        Initialize terrain generator
        
        Args:
            size: Grid size (pixels)
            pixel_size: Physical size per pixel (meters)
        """
        self.size = size
        self.pixel_size = pixel_size
        
    def generate_fbm_terrain(self, target_fractal_dim, roughness_factor=1.0):
        """
        Generate terrain using fractional Brownian motion
        
        Args:
            target_fractal_dim: Target fractal dimension (2.0-2.5)
            roughness_factor: Controls terrain amplitude
            
        Returns:
            elevation_grid: 2D elevation array (meters)
        """
        # Convert fractal dimension to Hurst exponent
        # For 2D surfaces: H = 3 - D
        hurst = 3.0 - target_fractal_dim
        
        # Generate white noise
        noise = np.random.randn(self.size, self.size)
        
        # Create frequency grid
        freqs = np.fft.fftfreq(self.size)
        fx, fy = np.meshgrid(freqs, freqs)
        
        # Avoid division by zero at DC component
        f_magnitude = np.sqrt(fx**2 + fy**2)
        f_magnitude[0, 0] = 1e-10
        
        # Apply fractal scaling in frequency domain
        # Power spectral density: S(k) ∝ k^(-β)
        # For fBm: β = 2H + 1 = 2(3-D) + 1 = 7 - 2D
        beta = 7 - 2 * target_fractal_dim
        
        # Create filter
        filter_func = f_magnitude ** (-beta/2)
        filter_func[0, 0] = 0  # Remove DC component
        
        # Apply filter in frequency domain
        noise_fft = np.fft.fft2(noise)
        filtered_fft = noise_fft * filter_func
        
        # Convert back to spatial domain
        terrain = np.real(np.fft.ifft2(filtered_fft))
        
        # NORMALIZE RMS HEIGHT (isolate spectral structure from amplitude)
        # This ensures all terrains have the same RMS height regardless of D
        terrain = terrain - np.mean(terrain)  # Remove mean
        rms = np.sqrt(np.mean(terrain**2))    # Calculate RMS
        if rms > 0:
            terrain = terrain / rms * roughness_factor  # Normalize to target RMS
        else:
            terrain = terrain * roughness_factor
        
        # Ensure positive elevations (add offset)
        terrain = terrain - np.min(terrain) + 10.0
        
        return terrain
    
    def compute_fractal_dimension(self, elevation_grid):
        """
        Compute fractal dimension using differential box-counting
        Implements the exact algorithm from the technical report
        
        Args:
            elevation_grid: 2D elevation array
            
        Returns:
            fractal_dimension: Computed fractal dimension
            r_squared: Quality of linear fit
        """
        # Normalize elevation to [0, 1] range
        z_min, z_max = np.min(elevation_grid), np.max(elevation_grid)
        if z_max == z_min:
            return 2.0, 1.0  # Flat surface
            
        z_norm = (elevation_grid - z_min) / (z_max - z_min)
        
        # Box sizes (powers of 2 from 2 to 128)
        box_sizes = [2, 4, 8, 16, 32, 64, 128]
        
        # Filter box sizes to ensure they fit in the grid
        nrows, ncols = z_norm.shape
        valid_box_sizes = [eps for eps in box_sizes if eps < min(nrows, ncols)]
        
        if len(valid_box_sizes) < 3:
            return 2.0, 0.0  # Not enough scales for reliable fit
        
        box_counts = []
        
        for eps in valid_box_sizes:
            total_boxes = 0
            
            # Iterate over all possible box positions
            for i in range(0, nrows - eps + 1, eps):
                for j in range(0, ncols - eps + 1, eps):
                    # Extract box region
                    box_region = z_norm[i:i+eps, j:j+eps]
                    
                    # Calculate elevation range in box
                    z_min_box = np.min(box_region)
                    z_max_box = np.max(box_region)
                    delta_z = z_max_box - z_min_box
                    
                    # Vertical box height (cubic boxes)
                    h = eps / max(nrows, ncols)
                    
                    # Count vertical boxes
                    n_vertical = max(1, int(np.ceil(delta_z / h)))
                    total_boxes += n_vertical
            
            box_counts.append(total_boxes)
        
        # Linear regression on log-log plot
        log_eps = np.log(valid_box_sizes)
        log_N = np.log(box_counts)
        
        # Fit line: log(N) = -D * log(eps) + C
        reg = LinearRegression()
        reg.fit(log_eps.reshape(-1, 1), log_N)
        
        fractal_dimension = -reg.coef_[0]
        r_squared = reg.score(log_eps.reshape(-1, 1), log_N)
        
        # Ensure fractal dimension is in valid range for surfaces
        fractal_dimension = np.clip(fractal_dimension, 2.0, 3.0)
        
        return fractal_dimension, r_squared
    
    def generate_terrain_set(self, target_dimensions, roughness_factor=1.0):
        """
        Generate a set of terrains with specified fractal dimensions
        
        Args:
            target_dimensions: List of target fractal dimensions
            roughness_factor: Controls terrain amplitude
            
        Returns:
            terrains: Dictionary of {target_D: (terrain, actual_D, r_squared)}
        """
        terrains = {}
        
        for target_D in target_dimensions:
            print(f"Generating terrain with target D = {target_D:.2f}")
            
            # Generate terrain
            terrain = self.generate_fbm_terrain(target_D, roughness_factor)
            
            # Verify fractal dimension
            actual_D, r_squared = self.compute_fractal_dimension(terrain)
            
            print(f"  Actual D = {actual_D:.3f}, R² = {r_squared:.3f}")
            
            terrains[target_D] = {
                'terrain': terrain,
                'actual_D': actual_D,
                'r_squared': r_squared,
                'target_D': target_D
            }
        
        return terrains
    
    def visualize_terrains(self, terrains, save_path=None):
        """Visualize generated terrains"""
        n_terrains = len(terrains)
        fig, axes = plt.subplots(2, n_terrains, figsize=(4*n_terrains, 8))
        
        if n_terrains == 1:
            axes = axes.reshape(2, 1)
        
        for i, (target_D, data) in enumerate(terrains.items()):
            terrain = data['terrain']
            actual_D = data['actual_D']
            r_squared = data['r_squared']
            
            # Elevation map
            im1 = axes[0, i].imshow(terrain, cmap='terrain', origin='lower')
            axes[0, i].set_title(f'Target D={target_D:.2f}\nActual D={actual_D:.3f}')
            axes[0, i].set_xlabel('X (pixels)')
            axes[0, i].set_ylabel('Y (pixels)')
            plt.colorbar(im1, ax=axes[0, i], label='Elevation (m)')
            
            # Cross-section
            mid_row = terrain.shape[0] // 2
            x_coords = np.arange(terrain.shape[1]) * self.pixel_size
            axes[1, i].plot(x_coords, terrain[mid_row, :])
            axes[1, i].set_title(f'Cross-section (R²={r_squared:.3f})')
            axes[1, i].set_xlabel('Distance (m)')
            axes[1, i].set_ylabel('Elevation (m)')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Always save the plot
        if save_path is None:
            save_path = "fractal_terrains_auto.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved terrain visualization: {save_path}")
        
        plt.show()

if __name__ == "__main__":
    # Test the fractal terrain generator
    generator = FractalTerrainGenerator(size=256, pixel_size=10.0)
    
    # Target fractal dimensions from the technical report
    target_dimensions = [2.05, 2.20, 2.35, 2.50]
    
    # Generate terrain set
    terrains = generator.generate_terrain_set(target_dimensions, roughness_factor=50.0)
    
    # Visualize results
    generator.visualize_terrains(terrains, 'fractal_terrains.png')
    
    print("\nTerrain generation complete!")
    print("Generated terrains with controlled fractal dimensions")