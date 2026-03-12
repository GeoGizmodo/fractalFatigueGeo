#!/usr/bin/env python3
"""
Fatigue Analysis using Miner's Cumulative Damage Rule
Implements the exact equations from the technical report
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FatigueAnalyzer:
    """Fatigue analysis using Miner's Rule from technical report"""
    
    def __init__(self, material_params=None):
        """
        Initialize with material parameters from technical report
        
        Args:
            material_params: Dictionary of material parameters
        """
        if material_params is None:
            # Default parameters from technical report
            self.params = {
                'sigma_ref_min': 400e6,  # Reference stress amplitude (Pa) - cast iron
                'sigma_ref_max': 500e6,  # Reference stress amplitude (Pa) - spring steel
                'm_min': 5,              # Fatigue exponent - cast iron
                'm_max': 8,              # Fatigue exponent - high-strength steel
                'terrain_adaptive': True  # Use terrain-adaptive parameters
            }
        else:
            self.params = material_params
    
    def get_terrain_adaptive_params(self, fractal_dimension):
        """
        Get terrain-adaptive fatigue parameters
        Higher fractal dimension -> more severe conditions -> lower reference stress
        
        Args:
            fractal_dimension: Terrain fractal dimension (2.0-2.5)
            
        Returns:
            sigma_ref: Reference stress amplitude (Pa)
            m: Fatigue exponent
        """
        if not self.params['terrain_adaptive']:
            return self.params['sigma_ref_min'], self.params['m_min']
        
        # Normalize fractal dimension to [0, 1]
        D_norm = (fractal_dimension - 2.0) / 0.5
        D_norm = np.clip(D_norm, 0, 1)
        
        # Linear interpolation between min and max values
        # Higher D -> lower reference stress (more severe conditions)
        sigma_ref = self.params['sigma_ref_max'] - D_norm * (self.params['sigma_ref_max'] - self.params['sigma_ref_min'])
        
        # Higher D -> lower fatigue exponent (more sensitive to stress)
        m = self.params['m_max'] - D_norm * (self.params['m_max'] - self.params['m_min'])
        
        return sigma_ref, m
    
    def calculate_stress_cycles(self, stress_amplitude, distance_km, terrain_wavelength):
        """
        Calculate number of stress cycles from technical report
        n = (distance × 1000) / λ
        
        Args:
            stress_amplitude: Stress amplitude array (Pa)
            distance_km: Mission distance (km)
            terrain_wavelength: Characteristic wavelength (m)
            
        Returns:
            n_cycles: Number of stress cycles
        """
        return (distance_km * 1000) / terrain_wavelength
    
    def basquin_law_cycles_to_failure(self, stress_amplitude, sigma_ref, m):
        """
        Calculate cycles to failure using Basquin's Law
        N = (σ_ref / σ)^m
        
        Args:
            stress_amplitude: Applied stress amplitude (Pa)
            sigma_ref: Reference stress amplitude (Pa)
            m: Fatigue exponent
            
        Returns:
            N: Cycles to failure
        """
        # Avoid division by zero
        stress_amplitude = np.maximum(stress_amplitude, 1e3)  # Minimum 1 kPa
        
        return (sigma_ref / stress_amplitude) ** m
    
    def miners_rule_damage(self, n_cycles, N_failure):
        """
        Calculate fatigue damage using Miner's Rule
        D = n / N
        
        Args:
            n_cycles: Applied cycles
            N_failure: Cycles to failure
            
        Returns:
            damage: Fatigue damage (0-1, where 1 = failure)
        """
        return n_cycles / N_failure
    
    def analyze_fatigue_from_response(self, response_data, frequency_analysis, 
                                    fractal_dimension, distance_km, terrain_wavelength):
        """
        Complete fatigue analysis from vehicle response data
        
        Args:
            response_data: Vehicle response from simulator
            frequency_analysis: Frequency analysis results
            fractal_dimension: Terrain fractal dimension
            distance_km: Mission distance (km)
            terrain_wavelength: Characteristic terrain wavelength (m)
            
        Returns:
            fatigue_results: Dictionary containing fatigue analysis
        """
        # Get terrain-adaptive material parameters
        sigma_ref, m = self.get_terrain_adaptive_params(fractal_dimension)
        
        # Extract stress amplitude data
        stress_amplitude = response_data['stress_amplitude']
        
        # Calculate RMS stress for representative analysis
        rms_stress = np.sqrt(np.mean(stress_amplitude**2))
        max_stress = np.max(np.abs(stress_amplitude))
        
        # Calculate number of cycles
        n_cycles = self.calculate_stress_cycles(stress_amplitude, distance_km, terrain_wavelength)
        
        # Calculate cycles to failure using RMS stress
        N_failure_rms = self.basquin_law_cycles_to_failure(rms_stress, sigma_ref, m)
        N_failure_max = self.basquin_law_cycles_to_failure(max_stress, sigma_ref, m)
        
        # Calculate fatigue damage
        damage_rms = self.miners_rule_damage(n_cycles, N_failure_rms)
        damage_max = self.miners_rule_damage(n_cycles, N_failure_max)
        
        # Rainflow counting approximation for more accurate damage
        # Use simplified approach: assume all cycles at RMS level
        damage_equivalent = damage_rms
        
        # Calculate fatigue life (missions to failure)
        if damage_equivalent > 0:
            missions_to_failure = 1.0 / damage_equivalent
        else:
            missions_to_failure = np.inf
        
        return {
            'fractal_dimension': fractal_dimension,
            'sigma_ref': sigma_ref,
            'fatigue_exponent': m,
            'rms_stress': rms_stress,
            'max_stress': max_stress,
            'n_cycles': n_cycles,
            'N_failure_rms': N_failure_rms,
            'N_failure_max': N_failure_max,
            'damage_rms': damage_rms,
            'damage_max': damage_max,
            'damage_equivalent': damage_equivalent,
            'missions_to_failure': missions_to_failure,
            'distance_km': distance_km,
            'terrain_wavelength': terrain_wavelength,
            'total_vibration_energy': frequency_analysis['total_vibration_energy'],
            'total_stress_energy': frequency_analysis['total_stress_energy']
        }
    
    def compare_terrain_fatigue(self, terrain_results_dict):
        """
        Compare fatigue results across different terrain types
        
        Args:
            terrain_results_dict: Dictionary of {fractal_dim: fatigue_results}
            
        Returns:
            comparison_data: Summary comparison data
        """
        fractal_dims = []
        damages = []
        missions_to_failure = []
        vibration_energies = []
        stress_energies = []
        rms_stresses = []
        
        for D, results in terrain_results_dict.items():
            fractal_dims.append(D)
            damages.append(results['damage_equivalent'])
            missions_to_failure.append(results['missions_to_failure'])
            vibration_energies.append(results['total_vibration_energy'])
            stress_energies.append(results['total_stress_energy'])
            rms_stresses.append(results['rms_stress'])
        
        # Sort by fractal dimension
        sort_idx = np.argsort(fractal_dims)
        fractal_dims = np.array(fractal_dims)[sort_idx]
        damages = np.array(damages)[sort_idx]
        missions_to_failure = np.array(missions_to_failure)[sort_idx]
        vibration_energies = np.array(vibration_energies)[sort_idx]
        stress_energies = np.array(stress_energies)[sort_idx]
        rms_stresses = np.array(rms_stresses)[sort_idx]
        
        # Calculate relative wear rates (normalized to smoothest terrain)
        relative_damage = damages / damages[0] if damages[0] > 0 else damages
        relative_life_reduction = missions_to_failure[0] / missions_to_failure
        
        return {
            'fractal_dimensions': fractal_dims,
            'damages': damages,
            'missions_to_failure': missions_to_failure,
            'relative_damage': relative_damage,
            'relative_life_reduction': relative_life_reduction,
            'vibration_energies': vibration_energies,
            'stress_energies': stress_energies,
            'rms_stresses': rms_stresses
        }
    
    def visualize_fatigue_comparison(self, comparison_data, save_path=None):
        """Visualize fatigue comparison across terrain types"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        D = comparison_data['fractal_dimensions']
        
        # Damage vs Fractal Dimension
        axes[0, 0].semilogy(D, comparison_data['damages'], 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Fractal Dimension D')
        axes[0, 0].set_ylabel('Fatigue Damage per Mission')
        axes[0, 0].set_title('Fatigue Damage vs Terrain Complexity')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Missions to Failure vs Fractal Dimension
        axes[0, 1].semilogy(D, comparison_data['missions_to_failure'], 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Fractal Dimension D')
        axes[0, 1].set_ylabel('Missions to Failure')
        axes[0, 1].set_title('Component Life vs Terrain Complexity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Relative Life Reduction
        axes[0, 2].plot(D, comparison_data['relative_life_reduction'], 'go-', linewidth=2, markersize=8)
        axes[0, 2].set_xlabel('Fractal Dimension D')
        axes[0, 2].set_ylabel('Life Reduction Factor')
        axes[0, 2].set_title('Relative Life Reduction')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Vibration Energy vs Fractal Dimension
        axes[1, 0].semilogy(D, comparison_data['vibration_energies'], 'mo-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Fractal Dimension D')
        axes[1, 0].set_ylabel('Total Vibration Energy')
        axes[1, 0].set_title('Vibration Energy vs Terrain Complexity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # RMS Stress vs Fractal Dimension
        axes[1, 1].semilogy(D, comparison_data['rms_stresses'], 'co-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Fractal Dimension D')
        axes[1, 1].set_ylabel('RMS Stress (Pa)')
        axes[1, 1].set_title('RMS Stress vs Terrain Complexity')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Power Law Fit for Damage vs Fractal Dimension
        # Fit: damage ∝ (D - 2)^α
        D_shifted = D - 2.0
        valid_idx = (comparison_data['damages'] > 0) & (D_shifted > 0)
        
        if np.sum(valid_idx) >= 2:
            log_D_shifted = np.log(D_shifted[valid_idx])
            log_damage = np.log(comparison_data['damages'][valid_idx])
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_D_shifted, log_damage)
            
            # Plot fit
            D_fit = np.linspace(D[0], D[-1], 100)
            D_shifted_fit = D_fit - 2.0
            damage_fit = np.exp(intercept) * (D_shifted_fit ** slope)
            
            axes[1, 2].loglog(D_shifted, comparison_data['damages'][valid_idx], 'ko', markersize=8, label='Data')
            axes[1, 2].loglog(D_shifted_fit, damage_fit, 'r-', linewidth=2, 
                             label=f'Fit: ∝(D-2)^{slope:.2f}\nR²={r_value**2:.3f}')
            axes[1, 2].set_xlabel('D - 2.0')
            axes[1, 2].set_ylabel('Fatigue Damage')
            axes[1, 2].set_title('Power Law Relationship')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'Insufficient data\nfor power law fit', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        
        # Always save the plot
        if save_path is None:
            save_path = "fatigue_comparison_auto.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved fatigue comparison: {save_path}")
        
        plt.show()
        
        return slope if 'slope' in locals() else None

if __name__ == "__main__":
    # Test the fatigue analyzer
    analyzer = FatigueAnalyzer()
    
    # Create synthetic test data
    test_fractal_dims = [2.05, 2.20, 2.35, 2.50]
    test_results = {}
    
    for D in test_fractal_dims:
        # Simulate increasing stress with fractal dimension
        base_stress = 50e6  # 50 MPa base stress
        stress_factor = (D - 2.0) * 2 + 1  # 1x to 2x multiplier
        rms_stress = base_stress * stress_factor
        
        # Create mock response data
        mock_response = {
            'stress_amplitude': np.random.normal(rms_stress, rms_stress*0.2, 1000)
        }
        
        # Create mock frequency analysis
        mock_freq_analysis = {
            'total_vibration_energy': rms_stress**2 * 1e-12,  # Scaled for visualization
            'total_stress_energy': rms_stress**2 * 1e-6
        }
        
        # Analyze fatigue
        result = analyzer.analyze_fatigue_from_response(
            mock_response, mock_freq_analysis, D, 
            distance_km=100, terrain_wavelength=20
        )
        
        test_results[D] = result
    
    # Compare results
    comparison = analyzer.compare_terrain_fatigue(test_results)
    
    # Visualize
    power_law_exponent = analyzer.visualize_fatigue_comparison(comparison, 'fatigue_analysis_test.png')
    
    print("Fatigue analysis test complete!")
    if power_law_exponent is not None:
        print(f"Power law exponent: {power_law_exponent:.3f}")
    
    # Print summary table
    print("\nFatigue Analysis Summary:")
    print("Fractal D | Damage/Mission | Missions to Failure | Life Reduction")
    print("-" * 65)
    for i, D in enumerate(comparison['fractal_dimensions']):
        damage = comparison['damages'][i]
        missions = comparison['missions_to_failure'][i]
        reduction = comparison['relative_life_reduction'][i]
        print(f"{D:8.2f} | {damage:13.2e} | {missions:15.1f} | {reduction:12.1f}x")