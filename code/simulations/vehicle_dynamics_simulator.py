#!/usr/bin/env python3
"""
Vehicle Dynamics Simulation using Quarter-Car Model
Implements the exact equations from the technical report and patent
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

class QuarterCarSimulator:
    """Quarter-car suspension model from technical report"""
    
    def __init__(self, vehicle_params=None):
        """
        Initialize with HMMWV parameters from technical report
        
        Args:
            vehicle_params: Dictionary of vehicle parameters
        """
        if vehicle_params is None:
            # HMMWV parameters from technical report
            self.params = {
                'ms': 3402 / 4,  # Sprung mass per wheel (kg) - total mass / 4
                'mu': 100,       # Unsprung mass (kg) - wheel, tire, etc.
                'ks': 36300,     # Spring stiffness (N/m)
                'cs': 2500,      # Damping coefficient (N·s/m)
                'kt': 200000,    # Tire stiffness (N/m)
                'g': 9.81        # Gravitational acceleration (m/s²)
            }
        else:
            self.params = vehicle_params
        
        # Calculate derived parameters
        self.omega_n = np.sqrt(self.params['ks'] / self.params['ms'])  # Natural frequency
        self.zeta = self.params['cs'] / (2 * np.sqrt(self.params['ks'] * self.params['ms']))  # Damping ratio
        
        print(f"Vehicle Parameters:")
        print(f"  Natural frequency: {self.omega_n:.2f} rad/s ({self.omega_n/(2*np.pi):.2f} Hz)")
        print(f"  Damping ratio: {self.zeta:.3f}")
    
    def terrain_wavelength_model(self, roughness_index):
        """
        Terrain wavelength model from technical report
        λ = 50 × exp(-0.035 × R)
        
        Args:
            roughness_index: Terrain roughness index (0-100)
            
        Returns:
            wavelength: Characteristic wavelength (meters)
        """
        return 50.0 * np.exp(-0.035 * roughness_index)
    
    def road_amplitude_model(self, roughness_index):
        """
        Road amplitude model from technical report
        A_road = 0.01 + (R/100) × 0.14
        
        Args:
            roughness_index: Terrain roughness index (0-100)
            
        Returns:
            amplitude: Road amplitude (meters)
        """
        return 0.01 + (roughness_index / 100.0) * 0.14
    
    def transmissibility_function(self, frequency_ratio):
        """
        Transmissibility function from technical report
        T = √(1 + (2ζr)²) / √((1-r²)² + (2ζr)²)
        
        Args:
            frequency_ratio: r = ω/ωn
            
        Returns:
            transmissibility: Transmissibility magnitude
        """
        r = frequency_ratio
        zeta = self.zeta
        
        numerator = np.sqrt(1 + (2 * zeta * r)**2)
        denominator = np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
        
        return numerator / denominator
    
    def simulate_terrain_response(self, terrain_profile, vehicle_speed, pixel_size):
        """
        Simulate vehicle response to terrain profile
        
        Args:
            terrain_profile: 1D terrain elevation profile (meters)
            vehicle_speed: Vehicle speed (m/s)
            pixel_size: Physical size per pixel (meters)
            
        Returns:
            response_data: Dictionary containing simulation results
        """
        # Create distance and time arrays
        distances = np.arange(len(terrain_profile)) * pixel_size
        dt = pixel_size / vehicle_speed
        time = np.arange(len(terrain_profile)) * dt
        
        # Calculate terrain derivatives for excitation
        terrain_velocity = np.gradient(terrain_profile, dt)
        
        # Set up state-space representation of quarter-car model
        # States: [z_s, z_s_dot, z_u, z_u_dot] (sprung mass pos/vel, unsprung mass pos/vel)
        
        def quarter_car_dynamics(state, t, road_input, road_velocity):
            """Quarter-car differential equations"""
            z_s, z_s_dot, z_u, z_u_dot = state
            
            # Interpolate road input at current time
            if t >= len(road_input) * dt:
                z_r = road_input[-1]
                z_r_dot = 0
            else:
                idx = int(t / dt)
                if idx >= len(road_input) - 1:
                    z_r = road_input[-1]
                    z_r_dot = 0
                else:
                    z_r = road_input[idx]
                    z_r_dot = road_velocity[idx]
            
            # Forces
            F_spring = self.params['ks'] * (z_u - z_s)
            F_damper = self.params['cs'] * (z_u_dot - z_s_dot)
            F_tire = self.params['kt'] * (z_r - z_u)
            
            # Equations of motion
            z_s_ddot = (F_spring + F_damper) / self.params['ms']
            z_u_ddot = (-F_spring - F_damper + F_tire) / self.params['mu']
            
            return [z_s_dot, z_s_ddot, z_u_dot, z_u_ddot]
        
        # Initial conditions (at rest)
        initial_state = [terrain_profile[0], 0, terrain_profile[0], 0]
        
        # Solve differential equation
        solution = odeint(quarter_car_dynamics, initial_state, time, 
                         args=(terrain_profile, terrain_velocity))
        
        z_s = solution[:, 0]      # Sprung mass position
        z_s_dot = solution[:, 1]  # Sprung mass velocity
        z_u = solution[:, 2]      # Unsprung mass position
        z_u_dot = solution[:, 3]  # Unsprung mass velocity
        
        # Calculate accelerations
        z_s_ddot = np.gradient(z_s_dot, dt)
        z_u_ddot = np.gradient(z_u_dot, dt)
        
        # Calculate forces
        suspension_force = self.params['ks'] * (z_u - z_s) + self.params['cs'] * (z_u_dot - z_s_dot)
        tire_force = self.params['kt'] * (terrain_profile - z_u)
        
        # Calculate stress amplitude (from technical report)
        body_acceleration_g = z_s_ddot / self.params['g']  # In g units
        stress_amplitude = np.abs(body_acceleration_g) * self.params['g'] * self.params['ks']  # Pa
        
        return {
            'time': time,
            'distance': distances,
            'terrain_profile': terrain_profile,
            'sprung_mass_pos': z_s,
            'sprung_mass_vel': z_s_dot,
            'sprung_mass_acc': z_s_ddot,
            'unsprung_mass_pos': z_u,
            'unsprung_mass_vel': z_u_dot,
            'unsprung_mass_acc': z_u_ddot,
            'suspension_force': suspension_force,
            'tire_force': tire_force,
            'body_acceleration_g': body_acceleration_g,
            'stress_amplitude': stress_amplitude,
            'vehicle_speed': vehicle_speed,
            'dt': dt
        }
    
    def compute_power_spectral_density(self, signal_data, dt, nperseg=None):
        """
        Compute power spectral density of a signal
        
        Args:
            signal_data: Time series data
            dt: Time step
            nperseg: Length of each segment for Welch's method
            
        Returns:
            frequencies: Frequency array (Hz)
            psd: Power spectral density
        """
        if nperseg is None:
            nperseg = min(len(signal_data) // 4, 1024)
        
        frequencies, psd = signal.welch(signal_data, fs=1/dt, nperseg=nperseg, 
                                       window='hann', noverlap=nperseg//2)
        
        return frequencies, psd
    
    def analyze_frequency_response(self, response_data):
        """
        Analyze frequency content of vehicle response
        
        Args:
            response_data: Output from simulate_terrain_response
            
        Returns:
            frequency_analysis: Dictionary containing spectral analysis
        """
        dt = response_data['dt']
        
        # Compute PSDs for key signals
        freq_acc, psd_acc = self.compute_power_spectral_density(
            response_data['sprung_mass_acc'], dt)
        freq_force, psd_force = self.compute_power_spectral_density(
            response_data['suspension_force'], dt)
        freq_stress, psd_stress = self.compute_power_spectral_density(
            response_data['stress_amplitude'], dt)
        
        # Calculate RMS values
        rms_acceleration = np.sqrt(np.mean(response_data['sprung_mass_acc']**2))
        rms_force = np.sqrt(np.mean(response_data['suspension_force']**2))
        rms_stress = np.sqrt(np.mean(response_data['stress_amplitude']**2))
        
        # Find dominant frequencies
        peak_freq_acc = freq_acc[np.argmax(psd_acc)]
        peak_freq_force = freq_force[np.argmax(psd_force)]
        
        return {
            'frequencies_acc': freq_acc,
            'psd_acceleration': psd_acc,
            'frequencies_force': freq_force,
            'psd_force': psd_force,
            'frequencies_stress': freq_stress,
            'psd_stress': psd_stress,
            'rms_acceleration': rms_acceleration,
            'rms_force': rms_force,
            'rms_stress': rms_stress,
            'peak_frequency_acc': peak_freq_acc,
            'peak_frequency_force': peak_freq_force,
            'total_vibration_energy': np.trapz(psd_acc, freq_acc),
            'total_force_energy': np.trapz(psd_force, freq_force),
            'total_stress_energy': np.trapz(psd_stress, freq_stress)
        }
    
    def visualize_response(self, response_data, frequency_analysis, save_path=None):
        """Visualize vehicle response and frequency analysis"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        time = response_data['time']
        distance = response_data['distance']
        
        # Time domain plots
        axes[0, 0].plot(distance, response_data['terrain_profile'], 'k-', label='Terrain')
        axes[0, 0].plot(distance, response_data['sprung_mass_pos'], 'b-', label='Sprung Mass')
        axes[0, 0].plot(distance, response_data['unsprung_mass_pos'], 'r-', label='Unsprung Mass')
        axes[0, 0].set_xlabel('Distance (m)')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].set_title('Vehicle Response to Terrain')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(distance, response_data['sprung_mass_acc'], 'b-')
        axes[0, 1].set_xlabel('Distance (m)')
        axes[0, 1].set_ylabel('Acceleration (m/s²)')
        axes[0, 1].set_title(f'Body Acceleration (RMS: {frequency_analysis["rms_acceleration"]:.2f} m/s²)')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(distance, response_data['suspension_force'], 'g-')
        axes[1, 0].set_xlabel('Distance (m)')
        axes[1, 0].set_ylabel('Force (N)')
        axes[1, 0].set_title(f'Suspension Force (RMS: {frequency_analysis["rms_force"]:.0f} N)')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(distance, response_data['stress_amplitude'], 'm-')
        axes[1, 1].set_xlabel('Distance (m)')
        axes[1, 1].set_ylabel('Stress (Pa)')
        axes[1, 1].set_title(f'Stress Amplitude (RMS: {frequency_analysis["rms_stress"]:.0f} Pa)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Frequency domain plots
        axes[2, 0].loglog(frequency_analysis['frequencies_acc'], frequency_analysis['psd_acceleration'])
        axes[2, 0].set_xlabel('Frequency (Hz)')
        axes[2, 0].set_ylabel('PSD (m²/s⁴/Hz)')
        axes[2, 0].set_title(f'Acceleration PSD (Peak: {frequency_analysis["peak_frequency_acc"]:.2f} Hz)')
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].loglog(frequency_analysis['frequencies_force'], frequency_analysis['psd_force'])
        axes[2, 1].set_xlabel('Frequency (Hz)')
        axes[2, 1].set_ylabel('PSD (N²/Hz)')
        axes[2, 1].set_title(f'Force PSD (Peak: {frequency_analysis["peak_frequency_force"]:.2f} Hz)')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Always save the plot
        if save_path is None:
            save_path = "vehicle_response_auto.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved vehicle response: {save_path}")
        
        plt.show()

if __name__ == "__main__":
    # Test the quarter-car simulator
    simulator = QuarterCarSimulator()
    
    # Create a simple sinusoidal terrain for testing
    distance = np.linspace(0, 1000, 1000)  # 1 km
    wavelength = 20  # meters
    amplitude = 0.1  # meters
    terrain = amplitude * np.sin(2 * np.pi * distance / wavelength)
    
    # Simulate vehicle response
    vehicle_speed = 15  # m/s (54 km/h)
    pixel_size = 1.0   # 1 meter per pixel
    
    response = simulator.simulate_terrain_response(terrain, vehicle_speed, pixel_size)
    freq_analysis = simulator.analyze_frequency_response(response)
    
    # Visualize results
    simulator.visualize_response(response, freq_analysis, 'vehicle_response_test.png')
    
    print("Vehicle dynamics simulation test complete!")