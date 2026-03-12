"""
Vehicle Parameter Library for Fractal Terrain Fatigue Validation
================================================================

This library generates quarter-car suspension parameters spanning the parameter
space typical of ground vehicles from motorcycles to heavy trucks.

Parameter ranges are derived from:
1. Wong, J.Y. "Theory of Ground Vehicles" (4th ed., 2008)
2. OpenVD: Open Vehicle Dynamics library (Mendes, 2020)
3. Published vehicle ride dynamics literature

PARAMETER RANGES (from literature):
- Sprung mass (ms): 150-2500 kg per wheel [Wong 2008, Ch. 7]
- Unsprung mass ratio (mu/ms): 0.10-0.15 [Wong 2008, p. 437]
- Natural frequency (fn): 0.8-2.5 Hz [ride comfort literature]
- Damping ratio (ζ): 0.20-0.40 [typical suspension design]
- Tire stiffness (kt): ~10× suspension stiffness [Wong 2008, p. 437]

Quarter-car model parameters:
- ms: Sprung mass (vehicle body mass per wheel) [kg]
- mu: Unsprung mass (wheel/tire/suspension mass) [kg]
- ks: Suspension spring stiffness [N/m]
- kt: Tire vertical stiffness [N/m]
- cs: Suspension damping coefficient [N·s/m]
- zeta: Damping ratio (dimensionless)

REFERENCES:
- Wong, J.Y. (2008). Theory of Ground Vehicles (4th ed.). Wiley.
- Mendes, A. (2020). OpenVD: Open Vehicle Dynamics. 
  https://github.com/andresmendes/openvd
"""

import numpy as np
from scipy.stats import qmc

class VehicleLibrary:
    """Library of vehicle suspension parameters for validation studies."""
    
    @staticmethod
    def generate_ensemble(n_vehicles=50, seed=42):
        """
        Generate ensemble of vehicles spanning realistic parameter space.
        
        Uses Latin Hypercube Sampling to ensure good coverage of parameter space.
        
        Args:
            n_vehicles (int): Number of vehicles to generate
            seed (int): Random seed for reproducibility
        
        Returns:
            dict: Vehicle ID -> parameter dictionary
        """
        np.random.seed(seed)
        
        # Parameter ranges from literature (Wong 2008, OpenVD)
        ms_range = (150, 2500)      # kg - sprung mass per wheel
        fn_range = (0.8, 2.5)       # Hz - natural frequency (ride comfort)
        zeta_range = (0.20, 0.40)   # dimensionless - damping ratio
        mu_ratio_range = (0.10, 0.15)  # dimensionless - unsprung/sprung mass
        kt_ks_ratio = 10.0          # tire stiffness ~10x suspension (Wong 2008)
        
        vehicles = {}
        
        # Latin Hypercube Sampling for better parameter space coverage
        sampler = qmc.LatinHypercube(d=4, seed=seed)
        samples = sampler.random(n=n_vehicles)
        
        # Scale samples to parameter ranges
        ms_samples = samples[:, 0] * (ms_range[1] - ms_range[0]) + ms_range[0]
        fn_samples = samples[:, 1] * (fn_range[1] - fn_range[0]) + fn_range[0]
        zeta_samples = samples[:, 2] * (zeta_range[1] - zeta_range[0]) + zeta_range[0]
        mu_ratio_samples = samples[:, 3] * (mu_ratio_range[1] - mu_ratio_range[0]) + mu_ratio_range[0]
        
        for i in range(n_vehicles):
            ms = ms_samples[i]
            fn = fn_samples[i]
            zeta = zeta_samples[i]
            mu_ratio = mu_ratio_samples[i]
            
            # Calculate derived parameters
            omega_n = 2 * np.pi * fn
            ks = ms * omega_n**2
            mu = ms * mu_ratio
            kt = kt_ks_ratio * ks
            cs = 2 * zeta * np.sqrt(ks * ms)
            
            vehicles[f'Vehicle_{i+1:03d}'] = {
                'ms': ms,
                'mu': mu,
                'ks': ks,
                'kt': kt,
                'cs': cs,
                'zeta': zeta,
                'omega_n': omega_n,
                'f_n': fn,
                'c_crit': 2 * np.sqrt(ks * ms),
                'mass_ratio': mu_ratio,
                'total_mass': ms + mu,
                'source': 'Generated from parameter ranges (Wong 2008, OpenVD)'
            }
        
        return vehicles
    
    @staticmethod
    def get_canonical_vehicles():
        """
        Returns dictionary of 10 canonical vehicle parameter sets.
        
        These represent specific vehicle types with documented parameters.
        
        Returns:
            dict: Vehicle name -> parameter dictionary
        """
        vehicles = {}
        
        # 1. MOTORCYCLE - Light vehicle, stiff suspension
        # Derived from parameter ranges in Wong (2008) and vehicle dynamics literature
        vehicles['Motorcycle'] = {
            'ms': 150.0,      # kg (total ~200 kg, 75% sprung)
            'mu': 15.0,       # kg (10% mass ratio)
            'ks': 18000.0,    # N/m (stiff for handling)
            'kt': 180000.0,   # N/m (10x suspension stiffness)
            'zeta': 0.30,
            'source': 'Derived from parameter ranges in Wong (2008) Ch. 7 and vehicle dynamics literature'
        }
        
        # 2. COMPACT CAR - Small passenger vehicle
        # Based on Wong Problem 7.6, scaled down
        vehicles['Compact_Car'] = {
            'ms': 300.0,      # kg
            'mu': 35.0,       # kg
            'ks': 20000.0,    # N/m
            'kt': 160000.0,   # N/m
            'zeta': 0.30,
            'source': 'Scaled from Wong (2008) Problem 7.6'
        }
        
        # 3. SEDAN (PASSENGER CAR) - Wong Problem 7.6
        # Direct measurement from literature
        vehicles['Sedan'] = {
            'ms': 454.5,      # kg (1000 lb)
            'mu': 45.45,      # kg (100 lb, 10% mass ratio)
            'ks': 22000.0,    # N/m (125 lb/in)
            'kt': 176000.0,   # N/m (1000 lb/in)
            'zeta': 0.30,
            'source': 'Wong (2008) "Theory of Ground Vehicles", Problem 7.6, p.515'
        }
        
        # 4. FAMILY SUV - Mid-size utility vehicle
        # Derived from parameter ranges in Wong (2008) and vehicle dynamics literature
        vehicles['SUV'] = {
            'ms': 600.0,      # kg
            'mu': 70.0,       # kg
            'ks': 30000.0,    # N/m
            'kt': 250000.0,   # N/m
            'zeta': 0.30,
            'source': 'Derived from parameter ranges in Wong (2008) Ch. 7 and vehicle dynamics literature'
        }
        
        # 5. LIGHT TRUCK / PICKUP - Commercial light vehicle
        # Derived from parameter ranges in Wong (2008) and vehicle dynamics literature
        vehicles['Light_Truck'] = {
            'ms': 700.0,      # kg
            'mu': 80.0,       # kg
            'ks': 32000.0,    # N/m
            'kt': 280000.0,   # N/m
            'zeta': 0.30,
            'source': 'Derived from parameter ranges in Wong (2008) Ch. 7 and vehicle dynamics literature'
        }
        
        # 6. HMMWV - Military utility vehicle
        # Current simulation baseline
        vehicles['HMMWV'] = {
            'ms': 850.0,      # kg
            'mu': 85.0,       # kg (10% mass ratio)
            'ks': 35000.0,    # N/m
            'kt': 350000.0,   # N/m (10x suspension stiffness)
            'zeta': 0.30,
            'source': 'Current study baseline parameters'
        }
        
        # 7. SEDAN (OPENVD) - Alternative sedan configuration
        # From OpenVD Vehicle-Dynamics-Vertical
        vehicles['Sedan_OpenVD'] = {
            'ms': 1000.0,     # kg
            'mu': 100.0,      # kg (10% mass ratio)
            'ks': 48000.0,    # N/m
            'kt': 480000.0,   # N/m (10x suspension stiffness)
            'zeta': 0.30,     # Adjusted from cs=4000 N·s/m
            'source': 'OpenVD Vehicle-Dynamics-Vertical, Template2DOF.m'
        }
        
        # 8. DELIVERY VAN - Commercial vehicle
        # Derived from parameter ranges in Wong (2008) and vehicle dynamics literature
        vehicles['Delivery_Van'] = {
            'ms': 1200.0,     # kg
            'mu': 120.0,      # kg
            'ks': 50000.0,    # N/m (stiffer for cargo)
            'kt': 400000.0,   # N/m
            'zeta': 0.30,
            'source': 'Derived from parameter ranges in Wong (2008) Ch. 7 and vehicle dynamics literature'
        }
        
        # 9. HEAVY TRUCK - Large commercial vehicle
        # Scaled from Wong automobile example
        vehicles['Heavy_Truck'] = {
            'ms': 1500.0,     # kg
            'mu': 150.0,      # kg
            'ks': 55000.0,    # N/m (very stiff for load capacity)
            'kt': 450000.0,   # N/m
            'zeta': 0.30,
            'source': 'Scaled from Wong (2008) Example 7.1, heavy vehicle parameters'
        }
        
        # 10. BUS - Large passenger vehicle
        # Derived from parameter ranges in Wong (2008) and vehicle dynamics literature
        vehicles['Bus'] = {
            'ms': 2000.0,     # kg
            'mu': 180.0,      # kg
            'ks': 60000.0,    # N/m (stiff for passenger comfort)
            'kt': 500000.0,   # N/m
            'zeta': 0.30,
            'source': 'Derived from parameter ranges in Wong (2008) Ch. 7 and vehicle dynamics literature'
        }
        
        # Calculate derived parameters for all vehicles
        for name, params in vehicles.items():
            ms = params['ms']
            mu = params['mu']
            ks = params['ks']
            kt = params['kt']
            zeta = params['zeta']
            
            # Natural frequency (rad/s)
            omega_n = np.sqrt(ks / ms)
            
            # Natural frequency (Hz)
            f_n = omega_n / (2 * np.pi)
            
            # Damping coefficient from damping ratio
            cs = 2 * zeta * np.sqrt(ks * ms)
            
            # Critical damping
            c_crit = 2 * np.sqrt(ks * ms)
            
            # Mass ratio
            mass_ratio = mu / ms
            
            # Add derived parameters
            params['cs'] = cs
            params['omega_n'] = omega_n
            params['f_n'] = f_n
            params['c_crit'] = c_crit
            params['mass_ratio'] = mass_ratio
            params['total_mass'] = ms + mu
        
        return vehicles
    
    @staticmethod
    def get_all_vehicles():
        """Alias for get_canonical_vehicles() for backward compatibility."""
        return VehicleLibrary.get_canonical_vehicles()
    
    @staticmethod
    def get_vehicle(name):
        """
        Get parameters for a specific canonical vehicle.
        
        Args:
            name (str): Vehicle name (e.g., 'Sedan', 'HMMWV')
        
        Returns:
            dict: Vehicle parameters
        """
        vehicles = VehicleLibrary.get_canonical_vehicles()
        if name not in vehicles:
            raise ValueError(f"Vehicle '{name}' not found. Available: {list(vehicles.keys())}")
        return vehicles[name]
    
    @staticmethod
    def print_ensemble_summary(n_vehicles=50):
        """Print summary statistics of vehicle ensemble."""
        vehicles = VehicleLibrary.generate_ensemble(n_vehicles=n_vehicles)
        
        ms_vals = [v['ms'] for v in vehicles.values()]
        fn_vals = [v['f_n'] for v in vehicles.values()]
        zeta_vals = [v['zeta'] for v in vehicles.values()]
        mu_ratio_vals = [v['mass_ratio'] for v in vehicles.values()]
        
        print("\n" + "="*100)
        print(f"VEHICLE ENSEMBLE SUMMARY (N = {n_vehicles})")
        print("="*100)
        print(f"{'Parameter':<30} {'Min':<15} {'Max':<15} {'Mean':<15} {'Std':<15}")
        print("-"*100)
        print(f"{'Sprung mass ms (kg)':<30} {min(ms_vals):<15.1f} {max(ms_vals):<15.1f} {np.mean(ms_vals):<15.1f} {np.std(ms_vals):<15.1f}")
        print(f"{'Natural frequency fn (Hz)':<30} {min(fn_vals):<15.2f} {max(fn_vals):<15.2f} {np.mean(fn_vals):<15.2f} {np.std(fn_vals):<15.2f}")
        print(f"{'Damping ratio ζ':<30} {min(zeta_vals):<15.3f} {max(zeta_vals):<15.3f} {np.mean(zeta_vals):<15.3f} {np.std(zeta_vals):<15.3f}")
        print(f"{'Mass ratio mu/ms':<30} {min(mu_ratio_vals):<15.3f} {max(mu_ratio_vals):<15.3f} {np.mean(mu_ratio_vals):<15.3f} {np.std(mu_ratio_vals):<15.3f}")
        print("-"*100)
        print("Parameter ranges from: Wong (2008) Theory of Ground Vehicles, OpenVD library")
        print("Sampling method: Latin Hypercube Sampling for uniform parameter space coverage")
        print("="*100 + "\n")
    
    @staticmethod
    def print_summary():
        """Print summary table of canonical vehicles."""
        vehicles = VehicleLibrary.get_canonical_vehicles()
        
        print("\n" + "="*100)
        print("CANONICAL VEHICLE LIBRARY - QUARTER-CAR SUSPENSION PARAMETERS")
        print("="*100)
        print(f"{'Vehicle':<20} {'ms (kg)':<10} {'mu (kg)':<10} {'ks (N/m)':<12} {'kt (N/m)':<12} {'fn (Hz)':<10} {'ζ':<8}")
        print("-"*100)
        
        for name, params in vehicles.items():
            print(f"{name:<20} {params['ms']:<10.1f} {params['mu']:<10.1f} "
                  f"{params['ks']:<12.0f} {params['kt']:<12.0f} "
                  f"{params['f_n']:<10.2f} {params['zeta']:<8.2f}")
        
        print("-"*100)
        print(f"Total vehicles: {len(vehicles)}")
        print(f"Mass range: {min(v['ms'] for v in vehicles.values()):.0f} - {max(v['ms'] for v in vehicles.values()):.0f} kg")
        print(f"Frequency range: {min(v['f_n'] for v in vehicles.values()):.2f} - {max(v['f_n'] for v in vehicles.values()):.2f} Hz")
        print("="*100 + "\n")
    
    @staticmethod
    def print_sources():
        """Print detailed source information for canonical vehicles."""
        vehicles = VehicleLibrary.get_canonical_vehicles()
        
        print("\n" + "="*100)
        print("CANONICAL VEHICLE PARAMETER SOURCES")
        print("="*100)
        
        for name, params in vehicles.items():
            print(f"\n{name}:")
            print(f"  Source: {params['source']}")
            print(f"  Parameters: ms={params['ms']:.1f} kg, mu={params['mu']:.1f} kg, "
                  f"ks={params['ks']:.0f} N/m, kt={params['kt']:.0f} N/m")
            print(f"  Natural frequency: {params['f_n']:.2f} Hz")
        
        print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    # Print summaries when run directly
    print("\n" + "="*100)
    print("VEHICLE LIBRARY TEST")
    print("="*100)
    
    # Test canonical vehicles
    VehicleLibrary.print_summary()
    
    # Test ensemble generation
    VehicleLibrary.print_ensemble_summary(n_vehicles=50)
    
    print("\nTesting ensemble generation...")
    ensemble = VehicleLibrary.generate_ensemble(n_vehicles=50)
    print(f"✓ Successfully generated {len(ensemble)} vehicles")
    print(f"✓ First vehicle: {list(ensemble.keys())[0]}")
    print(f"✓ Parameters: ms={ensemble['Vehicle_001']['ms']:.1f} kg, fn={ensemble['Vehicle_001']['f_n']:.2f} Hz")
