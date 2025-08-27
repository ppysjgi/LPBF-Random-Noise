import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.optimize import minimize
from autograd import grad, jacobian, hessian
import autograd.numpy as anp 
import cvxpy as cp
import os

# ===============================
# 1. Heat Transfer Model Class
# ===============================
import autograd.numpy as anp

class HeatTransferModel:
    def __init__(self, domain_size=(0.01, 0.01), grid_size=(101, 101), dt=1e-5, material_params=None):
        self.Lx, self.Ly = domain_size
        self.nx, self.ny = grid_size
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)
        self.x = anp.linspace(0, self.Lx, self.nx)
        self.y = anp.linspace(0, self.Ly, self.ny)
        self.X, self.Y = anp.meshgrid(self.x, self.y)
        self.dt = dt
        self.nt = None  # will be set in simulate()
        self.debug_counter = 0  # For diagnostic output
        
        default_params = {
            'T0': 21.0,           # Initial temperature (°C)
            'alpha': 5e-6,        # Thermal diffusivity (m²/s)
            'rho': 7800.0,        # Density (kg/m³)
            'cp': 500.0,          # Specific heat capacity (J/(kg·K))
            'k': 20.0,            # Thermal conductivity (W/(m·K)
            'T_melt': 1500.0,     # Melting temperature (°C)
            'thickness': 0.00017  # Plate thickness (m)
        }
        self.material = default_params if material_params is None else material_params
        self.T = self.material['T0'] * anp.ones((self.ny, self.nx))
        
        # Calculate the heat capacity term (not used directly now)
        self.heat_capacity = self.material['rho'] * self.material['cp']
        
        # Add plate thickness (in meters)
        self.thickness = self.material['thickness']

    def reset(self):
        """Reset the temperature field to initial conditions"""
        self.T = self.material['T0'] * anp.ones((self.ny, self.nx))
    
    def _laplacian(self, T):
        """
        Compute the Laplacian of T using a five-point stencil.
        """
        lap_inner = ((T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / (self.dx**2) +
                     (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / (self.dy**2))
        lap = anp.pad(lap_inner, pad_width=((1,1),(1,1)), mode='constant', constant_values=0)
        return lap

    def sawtooth_trajectory(self, t, params):
        v = params['v']
        A = params['A']
        y0 = params['y0']
        period = params['period']
        noise_sigma = params.get('noise_sigma', 0.0)
        max_noise = 0.00005  # 0.05 mm
        noise_sigma = anp.clip(noise_sigma, 0.0, max_noise)
        omega = 2 * anp.pi / period
        # Average speed factor for sawtooth (approximate)
        avg_speed_factor = anp.sqrt(1 + (2 * A * omega / anp.pi) ** 2)
        t_scaled = t * avg_speed_factor
        raw_x = v * t_scaled
        k = 1000  # sharpness parameter
        x = raw_x - (raw_x - (self.Lx - 0.0005)) * (1 / (1 + anp.exp(-k * (raw_x - (self.Lx - 0.0005)))))
        y = y0 + A * (2/anp.pi) * anp.arcsin(anp.sin(omega * t_scaled))
        if noise_sigma > 0:
            import numpy as np
            x = x + noise_sigma * np.random.randn()
            y = y + noise_sigma * np.random.randn()
        tx = v
        ty = (2 * A * omega / anp.pi) * anp.cos(omega * t_scaled)
        norm = anp.sqrt(tx**2 + ty**2)
        return x, y, tx / norm, ty / norm

    def swirl_trajectory(self, t, params):
        v = params['v']
        A = params['A']
        y0 = params['y0']
        fr = params['fr']
        om = 2 * anp.pi * fr
        noise_sigma = params.get('noise_sigma', 0.0)
        max_noise = 0.00005  # 0.05 mm
        noise_sigma = anp.clip(noise_sigma, 0.0, max_noise)
        # Average speed factor for swirl (circle/ellipse): sqrt(1 + (A*om/v)^2)
        avg_speed_factor = anp.sqrt(1 + (A * om / v) ** 2)
        t_scaled = t * avg_speed_factor
        raw_x = v * t_scaled + A * anp.sin(om * t_scaled)
        k = 1000
        x = raw_x - (raw_x - (self.Lx - 0.0005)) * (1 / (1 + anp.exp(-k * (raw_x - (self.Lx - 0.0005)))))
        y = y0 + A * anp.cos(om * t_scaled)
        if noise_sigma > 0:
            import numpy as np
            x = x + noise_sigma * np.random.randn()
            y = y + noise_sigma * np.random.randn()
        tx = v + A * om * anp.cos(om * t_scaled)
        ty = -A * om * anp.sin(om * t_scaled)
        norm = anp.sqrt(tx**2 + ty**2)
        return x, y, tx / norm, ty / norm

    def compute_temperature_gradients(self, T=None):
        """Calculate spatial gradients of temperature field"""
        if T is None:
            T = self.T
            
        grad_x_left = (T[:, 1:2] - T[:, 0:1]) / self.dx
        grad_x_interior = (T[:, 2:] - T[:, :-2]) / (2 * self.dx)
        grad_x_right = (T[:, -1:] - T[:, -2:-1]) / self.dx
        grad_x = anp.concatenate([grad_x_left, grad_x_interior, grad_x_right], axis=1)

        grad_y_top = (T[1:2, :] - T[0:1, :]) / self.dy
        grad_y_interior = (T[2:, :] - T[:-2, :]) / (2 * self.dy)
        grad_y_bottom = (T[-1:, :] - T[-2:-1, :]) / self.dy
        grad_y = anp.concatenate([grad_y_top, grad_y_interior, grad_y_bottom], axis=0)

        grad_mag = anp.sqrt(grad_x**2 + grad_y**2)
        return grad_x, grad_y, grad_mag

    def _gaussian_source(self, x_src, y_src, heat_params):

        power = heat_params.get('Q', 200.0)  # Power in Watts
        
        # Use r0 if provided, otherwise use sigma_x/sigma_y
        if 'r0' in heat_params:
            sigma_x = heat_params['r0'] / 2.0
            sigma_y = heat_params['r0'] / 2.0
        else:
            sigma_x = heat_params.get('sigma_x', 1.5e-3)  # Default 1.5 mm
            sigma_y = heat_params.get('sigma_y', 1.5e-3)  # Default 1.5 mm
        
        # Apply absorptivity from material properties
        absorbed_power = power * self.material['absorptivity']
        
        # Plain Gaussian function without normalization - to match original script
        G = np.exp(-(((self.X - x_src)**2) / (2 * sigma_x**2) +
                     ((self.Y - y_src)**2) / (2 * sigma_y**2)))
        
        # Scale by amplitude (similar to source_amplitude in original code)
        return G * absorbed_power

    def simulate(self, parameters, start_x=0.0, end_x=None, use_gaussian=True, verbose=False):

        laser_params, heat_params = parameters
        sawtooth_params, swirl_params = laser_params
        
        self.reset()
        if end_x is None:
            end_x = self.Lx

        fixed_nt = 3000  # Increased from 500 to 3000 time steps
        self.nt = fixed_nt

        # Use either the provided thermal diffusivity or compute it from k if needed
        if 'alpha' not in self.material and 'k' in self.material:
            alpha = self.material['k'] / (self.material['rho'] * self.material['cp'])
        else:
            alpha = self.material['alpha']
        
        T = self.T.copy()
        for n in range(self.nt):
            t = n * self.dt

            # Calculate laser positions and directions
            x1, y1, tx1, ty1 = self.sawtooth_trajectory(t, sawtooth_params)
            x2, y2, tx2, ty2 = self.swirl_trajectory(t, swirl_params)
            
            # Compute the heat source from each laser using the modified Gaussian
            S1 = self._gaussian_source(x1, y1, heat_params[0])
            S2 = self._gaussian_source(x2, y2, heat_params[1])
            S_total = S1 + S2
            #print("heat added from opt is:", anp.max(S_total))
            if verbose:
                print("S_total", anp.max(S_total))
            
            # Diffusion update (explicit finite difference update)
            lap = self._laplacian(T)
            T_diff = T + self.dt * alpha * lap 
            #print("thickness", self.thickness)
            source_increment = self.dt * S_total / (self.dx * self.dy * self.thickness * self.material['rho'] * self.material['cp'])
            #print("max source_increment", anp.max(source_increment))
            # Update temperature field
            T_new = T_diff + source_increment
            #print("max T", anp.max(T_new))
            
            # Enforce fixed boundary conditions (constant initial temperature)
            T_new[0, :] = self.material['T0']
            T_new[-1, :] = self.material['T0']
            T_new[:, 0] = self.material['T0']
            T_new[:, -1] = self.material['T0']
            
            if verbose:
                print("max T_new", anp.max(T_new))
            
            T = T_new
              
        
        self.T = T
        return self.T

    def compute_temperature_gradients(self, T=None):
        """Calculate spatial gradients of temperature field"""
        if T is None:
            T = self.T
            
        # Compute gradient in the x-direction with central differences
        grad_x_left = (T[:, 1:2] - T[:, 0:1]) / self.dx
        grad_x_interior = (T[:, 2:] - T[:, :-2]) / (2 * self.dx)
        grad_x_right = (T[:, -1:] - T[:, -2:-1]) / self.dx
        grad_x = anp.concatenate([grad_x_left, grad_x_interior, grad_x_right], axis=1)

        # Compute gradient in the y-direction with central differences
        grad_y_top = (T[1:2, :] - T[0:1, :]) / self.dy
        grad_y_interior = (T[2:, :] - T[:-2, :]) / (2 * self.dy)
        grad_y_bottom = (T[-1:, :] - T[-2:-1, :]) / self.dy
        grad_y = anp.concatenate([grad_y_top, grad_y_interior, grad_y_bottom], axis=0)

        # Calculate gradient magnitude
        grad_mag = anp.sqrt(grad_x**2 + grad_y**2)
        return grad_x, grad_y, grad_mag

# ===============================
# 2. Trajectory Optimizer Class
# ===============================
    
class TrajectoryOptimizer:
    def __init__(self, model, initial_params=None, bounds=None, x_range=(0.0, 0.01)):
        self.model = model
        self.initial_params = initial_params
        self.bounds = bounds
        self.x_range = x_range
        # Now allow Q, r0, v, and noise_sigma to be optimized (exclude only y0)
        # Add noise_sigma as an optimizable parameter if present in bounds, or always include it
        self.param_names = [k for k in initial_params.keys() if k not in ['sawtooth_y0', 'swirl_y0']]
        # Always include noise_sigma if present in bounds, even if not in initial_params
        if self.bounds and 'noise_sigma' in self.bounds and 'noise_sigma' not in self.param_names:
            self.param_names.append('noise_sigma')

    def parameters_to_array(self, params_dict):
        """Convert dictionary of parameters to flat array for optimizer."""
        return anp.array([params_dict[name] for name in self.param_names])
    
    def array_to_parameters(self, params_array):
        """Convert flat array from optimizer to dictionary of parameters."""
        return {name: params_array[i] for i, name in enumerate(self.param_names)}
    
    def unpack_parameters(self, params_array):
        """
        Unpack flat parameter array into structured format for model simulation.
        Returns tuple of (laser_params, heat_params).
        """
        params_dict = self.array_to_parameters(params_array)

        # Use optimized values for v, Q, r0, and noise_sigma
        noise_sigma = params_dict.get('noise_sigma', self.initial_params.get('noise_sigma', 0.0))
        sawtooth_params = {
            'v': params_dict['sawtooth_v'],
            'A': params_dict['sawtooth_A'],
            'y0': self.initial_params['sawtooth_y0'],  # y0 still fixed
            'period': params_dict['sawtooth_period'],
            'noise_sigma': noise_sigma
        }

        swirl_params = {
            'v': params_dict['swirl_v'],
            'A': params_dict['swirl_A'],
            'y0': self.initial_params['swirl_y0'],      # y0 still fixed
            'fr': params_dict['swirl_fr'],
            'noise_sigma': noise_sigma
        }

        # Use optimized Q and r0 for each laser
        heat_params = (
            {'Q': params_dict['sawtooth_Q'], 'r0': params_dict['sawtooth_r0']},
            {'Q': params_dict['swirl_Q'], 'r0': params_dict['swirl_r0']}
        )

        laser_params = (sawtooth_params, swirl_params)
        return laser_params, heat_params
    
    def objective_function(self, params_array):
        # Add parameter validation
        for i, name in enumerate(self.param_names):
            if anp.abs(params_array[i]) > 1e6:  # Detect unreasonable values
                print(f"Warning: Parameter {name} has extreme value: {params_array[i]}")
                return 1e10  # Return large penalty value
                
        # Unpack parameters into required format for model
        laser_params, heat_params = self.unpack_parameters(params_array)
        
        # Determine if using Gaussian or Goldak model based on parameters
        use_gaussian = 'r0' in heat_params[0]
        
        # Run simulation with current parameters
        T = self.model.simulate((laser_params, heat_params), 
                                start_x=self.x_range[0], 
                                end_x=self.x_range[1],
                                use_gaussian=use_gaussian)
        
        # Compute temperature gradients
        _, _, grad_mag = self.model.compute_temperature_gradients(T)
        
        # Compute squared L2 norm of gradient magnitudes
        cost = anp.sum(grad_mag**2)
        
        # Normalize by domain size
        cost = cost / (self.model.nx * self.model.ny)
        
        return cost

    def objective_max_gradient(self, params_array):
        """
        Alternative objective: minimize maximum temperature gradient.
        This can reduce local hot spots and extreme gradients.
        """
        laser_params, heat_params = self.unpack_parameters(params_array)
        
        # Determine if using Gaussian or Goldak model
        use_gaussian = 'r0' in heat_params[0]
        
        # Run simulation
        T = self.model.simulate((laser_params, heat_params), 
                                start_x=self.x_range[0], 
                                end_x=self.x_range[1],
                                use_gaussian=use_gaussian)
        
        # Compute temperature gradients
        _, _, grad_mag = self.model.compute_temperature_gradients(T)
        
        # Use maximum gradient as objective
        cost = anp.max(grad_mag)
        
        return cost

    def objective_path_focused(self, params_array):
        """
        Alternative objective: focus on gradients along the laser paths only.
        This emphasizes the regions that matter most for melt pool quality.
        """
        laser_params, heat_params = self.unpack_parameters(params_array)
        sawtooth_params, swirl_params = laser_params
        
        # Determine if using Gaussian or Goldak model
        use_gaussian = 'r0' in heat_params[0]
        
        # Run simulation
        T = self.model.simulate((laser_params, heat_params), 
                                start_x=self.x_range[0], 
                                end_x=self.x_range[1],
                                use_gaussian=use_gaussian)
        
        # Compute temperature gradients
        _, _, grad_mag = self.model.compute_temperature_gradients(T)
        
        # Generate points along laser paths
        times = anp.linspace(0, self.model.nt * self.model.dt, 50)  # Sample points
        path_points = []
        
        for t in times:
            # Get positions of both lasers at time t
            x1, y1, _, _ = self.model.sawtooth_trajectory(t, sawtooth_params)
            x2, y2, _, _ = self.model.swirl_trajectory(t, swirl_params)
            
            # Add points to list
            path_points.append((x1, y1))
            path_points.append((x2, y2))
        
        # Compute gradients at path points
        path_gradients = []
        dx, dy = self.model.dx, self.model.dy
        
        for x, y in path_points:
            # Find closest grid points
            i = int(anp.clip(y / dy, 0, self.model.ny - 1))
            j = int(anp.clip(x / dx, 0, self.model.nx - 1))
            
            # Get gradient at this point
            path_gradients.append(grad_mag[i, j])
        
        # Use mean of path gradients as objective
        path_gradients = anp.array(path_gradients)
        cost = anp.mean(path_gradients)
        
        return cost

    def objective_thermal_uniformity(self, params_array):
        """
        Alternative objective: promote thermal uniformity via temperature variance
        plus gradient minimization. This helps create more uniform melt pools.
        """
        laser_params, heat_params = self.unpack_parameters(params_array)
        
        # Determine if using Gaussian or Goldak model
        use_gaussian = 'r0' in heat_params[0]
        
        # Run simulation
        T = self.model.simulate((laser_params, heat_params), 
                                start_x=self.x_range[0], 
                                end_x=self.x_range[1],
                                use_gaussian=use_gaussian)
        
        # Compute gradient cost
        _, _, grad_mag = self.model.compute_temperature_gradients(T)
        grad_cost = anp.mean(grad_mag**2)
        
        # Compute temperature variance in melted region
        # Assuming T > T0 + 100 is melted (or use T_melt if available)
        melt_temp = self.model.material.get('T_melt', self.model.material['T0'] + 100)
        melt_mask = T > melt_temp
        if anp.sum(melt_mask) > 0:
            T_melt = T[melt_mask]
            T_variance = anp.var(T_melt)
        else:
            T_variance = 0.0
        
        # Combine costs with weights
        cost = 0.7 * grad_cost + 0.3 * T_variance
        
        return cost

    def objective_max_temp_difference(self, params_array):
        """
        Alternative objective: minimize maximum temperature difference
        in the melt pool. This helps ensure uniform material properties.
        """
        laser_params, heat_params = self.unpack_parameters(params_array)
        
        # Determine if using Gaussian or Goldak model
        use_gaussian = 'r0' in heat_params[0]
        
        # Run simulation
        T = self.model.simulate((laser_params, heat_params), 
                            start_x=self.x_range[0], 
                            end_x=self.x_range[1],
                            use_gaussian=use_gaussian)
        
        # Identify the melt pool region
        # Assuming T > T_melt is melted
        melt_temp = self.model.material.get('T_melt', self.model.material['T0'] + 100)
        melt_mask = T > melt_temp
        
        if anp.sum(melt_mask) > 0:
            T_melt = T[melt_mask]
            # Maximum temperature difference in melt pool
            max_temp_diff = anp.max(T_melt) - anp.min(T_melt)
        else:
            # Penalize if nothing is melting
            max_temp_diff = 1000.0  
        
        return max_temp_diff
    
    def get_inequality_constraints(self):
        """
        Create individual inequality constraint functions in the form g(x) <= 0.
        
        Returns:
            inequality_constraints: List of inequality constraint functions
        """
        inequality_constraints = []
        
        # 1. Minimum distance between lasers constraint
        def min_distance_constraint(x):
            laser_params, _ = self.unpack_parameters(x)
            min_distance = self._calculate_min_laser_distance(laser_params)
            min_allowed_distance = 0.0005  # 0.5mm minimum separation
            return min_allowed_distance - min_distance  # g(x) <= 0 form
        
        # 2. Laser path boundary constraints
        def sawtooth_max_y_constraint(x):
            params_dict = self.array_to_parameters(x)
            sawtooth_y_max = params_dict['sawtooth_y0'] + params_dict['sawtooth_A']
            return sawtooth_y_max - self.model.Ly  # Must be <= 0
        
        def sawtooth_min_y_constraint(x):
            params_dict = self.array_to_parameters(x)
            sawtooth_y_min = params_dict['sawtooth_y0'] - params_dict['sawtooth_A']
            return -sawtooth_y_min  # -y_min <= 0 means y_min >= 0
        
        def swirl_max_y_constraint(x):
            params_dict = self.array_to_parameters(x)
            swirl_y_max = params_dict['swirl_y0'] + params_dict['swirl_A']
            return swirl_y_max - self.model.Ly
        
        def swirl_min_y_constraint(x):
            params_dict = self.array_to_parameters(x)
            swirl_y_min = params_dict['swirl_y0'] - params_dict['swirl_A']
            return -swirl_y_min
        
        # 3. Energy density constraints
        def max_energy_sawtooth_constraint(x):
            params_dict = self.array_to_parameters(x)
            Q = params_dict['Q']
            energy_density = Q / (params_dict['sawtooth_v'] * 0.0002)  # Assuming h=0.2mm
            max_energy_density = 100.0  # J/mm²
            return energy_density - max_energy_density
        
        def min_energy_sawtooth_constraint(x):
            params_dict = self.array_to_parameters(x)
            Q = params_dict['Q']
            energy_density = Q / (params_dict['sawtooth_v'] * 0.0002)
            min_energy_density = 10.0  # J/mm²
            return min_energy_density - energy_density
        
        def max_energy_swirl_constraint(x):
            params_dict = self.array_to_parameters(x)
            Q = params_dict['Q']
            energy_density = Q / (params_dict['swirl_v'] * 0.0002)
            max_energy_density = 100.0  # J/mm²
            return energy_density - max_energy_density
        
        def min_energy_swirl_constraint(x):
            params_dict = self.array_to_parameters(x)
            Q = params_dict['Q']
            energy_density = Q / (params_dict['swirl_v'] * 0.0002)
            min_energy_density = 10.0  # J/mm²
            return min_energy_density - energy_density
        
        # Add bounds as inequality constraints
        bound_constraints = []
        for i, name in enumerate(self.param_names):
            if name in self.bounds:
                lb, ub = self.bounds[name]
                
                # Lower bound: lb - x_i <= 0 (ensures x_i >= lb)
                def make_lb_constraint(idx, bound):
                    return lambda x: bound - x[idx]
                
                # Upper bound: x_i - ub <= 0 (ensures x_i <= ub)
                def make_ub_constraint(idx, bound):
                    return lambda x: x[idx] - bound
                
                bound_constraints.append(make_lb_constraint(i, lb))
                bound_constraints.append(make_ub_constraint(i, ub))
        
        # Combine all constraints
        # You can comment out constraints you don't want to enforce
        inequality_constraints.extend([
            #min_distance_constraint,
            #sawtooth_max_y_constraint,
            #sawtooth_min_y_constraint,
            #swirl_max_y_constraint,
            #swirl_min_y_constraint,
            #max_energy_sawtooth_constraint,
            #min_energy_sawtooth_constraint,
            #max_energy_swirl_constraint,
            #min_energy_swirl_constraint
        ])
        
        # Add bound constraints
        inequality_constraints.extend(bound_constraints)
        
        return inequality_constraints
    
    def constraint_functions(self, params_array):
        params_dict = self.array_to_parameters(params_array)
        laser_params, heat_params = self.unpack_parameters(params_array)
        sawtooth_params, swirl_params = laser_params
        
        constraints = []
        
        # Minimum distance between lasers constraint
        min_distance = self._calculate_min_laser_distance(laser_params)
        min_allowed_distance = 0.0005  # 0.5mm minimum separation
        constraints.append(min_allowed_distance - min_distance)  # min_dist - actual_dist <= 0
        
        # Constraint on laser path parameters to stay within domain
        # Ensure sawtooth amplitude + y0 stays in domain
        sawtooth_y_max = params_dict['sawtooth_y0'] + params_dict['sawtooth_A']
        sawtooth_y_min = params_dict['sawtooth_y0'] - params_dict['sawtooth_A']
        constraints.append(sawtooth_y_max - self.model.Ly)  # Must be <= 0
        constraints.append(-sawtooth_y_min)  # -y_min <= 0 means y_min >= 0
        
        # Same for swirl
        swirl_y_max = params_dict['swirl_y0'] + params_dict['swirl_A']
        swirl_y_min = params_dict['swirl_y0'] - params_dict['swirl_A']
        constraints.append(swirl_y_max - self.model.Ly)
        constraints.append(-swirl_y_min)
        
        # Energy density constraint
        # Typical energy density E = Q/(v*h) where h is hatch spacing (simplified)
        Q = params_dict['Q']
        energy_density_sawtooth = Q / (params_dict['sawtooth_v'] * 0.0002)  # Assuming h=0.2mm
        energy_density_swirl = Q / (params_dict['swirl_v'] * 0.0002)
        
        max_energy_density = 100.0  # J/mm² (adjust based on material)
        min_energy_density = 10.0    # J/mm²
        
        #constraints.append(energy_density_sawtooth - max_energy_density)
        #constraints.append(min_energy_density - energy_density_sawtooth)
        #constraints.append(energy_density_swirl - max_energy_density)
        #constraints.append(min_energy_density - energy_density_swirl)
        
        return anp.array(constraints)

    def _calculate_min_laser_distance(self, laser_params):
        """
        Calculate minimum distance between the two lasers over simulation time.
        More robust implementation with better time sampling.
        
        Args:
            laser_params: Tuple of (sawtooth_params, swirl_params)
            
        Returns:
            min_dist: Minimum distance between lasers during simulation
        """
        sawtooth_params, swirl_params = laser_params
        
        # Estimate total simulation time
        slowest_v = anp.minimum(sawtooth_params['v'], swirl_params['v'])
        total_time = (self.x_range[1] - self.x_range[0]) / slowest_v
        
        # Use more samples to better capture minimum distance
        # Higher frequency components need finer sampling
        n_samples = 500  # Increased from 100
        t_samples = anp.linspace(0, total_time, n_samples)
        
        # Calculate distances at each time point
        distances = anp.zeros(n_samples)
        
        for i, t in enumerate(t_samples):
            # Get positions at time t
            x1, y1, _, _ = self.model.sawtooth_trajectory(t, sawtooth_params)
            x2, y2, _, _ = self.model.swirl_trajectory(t, swirl_params)
            
            # Calculate Euclidean distance
            distances[i] = anp.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # Find minimum distance
        min_dist = anp.min(distances)
        
        # Optional debug output
        # if min_dist < 0.001:
        #     min_idx = anp.argmin(distances)
        #     t_min = t_samples[min_idx]
        #     print(f"Warning: Minimum distance {min_dist*1000:.2f}mm at t={t_min:.3f}s")
        
        return min_dist
    
    def optimize_with_scipy(self, objective_type='standard', method='SLSQP', max_iterations=100):
        # Select objective function
        objective_functions = {
            'standard': self.objective_function,
            'max_gradient': self.objective_max_gradient,
            'path_focused': self.objective_path_focused,
            'thermal_uniformity': self.objective_thermal_uniformity,
            'max_temp_difference': self.objective_max_temp_difference
        }
        
        if objective_type not in objective_functions:
            raise ValueError(f"Unknown objective type: {objective_type}. "
                            f"Choose from: {list(objective_functions.keys())}")
        
        objective_func = objective_functions[objective_type]
        
        # Convert initial parameters to array
        initial_params_array = self.parameters_to_array(self.initial_params)
        
        # Set up constraints if using SLSQP or COBYLA
        constraints = None
        if method.upper() in ['SLSQP', 'COBYLA']:
            constraints = [{
                'type': 'ineq',
                'fun': lambda x: self.constraint_functions(x)  # Convert g(x)<=0 to -g(x)>=0
            }]
        
        # Run optimization
        print(f"Starting optimization with {objective_type} objective using scipy method: {method}")
        bounds_list = [(self.bounds[name][0], self.bounds[name][1]) for name in self.param_names]
        
        from scipy.optimize import minimize
        
        result = minimize(
            objective_func,
            initial_params_array,
            method=method,
            constraints=constraints,
            bounds=bounds_list,
            options={
                'maxiter': max_iterations, 
                'disp': True,
                'xtol': 1e-10,  # Tighter tolerance on parameter changes'
                'ftol': 1e-10   # Tighter tolerance on function value changes
                # Method-specific options can be added here:
                # 'ftol': 1e-6,       # Function tolerance for L-BFGS-B, TNC
                # 'eps': 1e-6,        # Step size for finite difference approximation
                # 'finite_diff_rel_step': 1e-6  # Relative step size for finite differencing
            }
        )
        
        # Convert result to dictionary
        optimized_params = self.array_to_parameters(result.x)
        
        print("Optimization complete.")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Iterations: {result.nit}")
        print("\nOptimized Parameters:")
        for name, value in optimized_params.items():
            print(f"  {name}: {value:.6f}")
        
        return result, optimized_params
    def optimize(self, objective_type='standard', method='SLSQP', max_iterations=100):
        return self.optimize_with_scipy(
            objective_type=objective_type, 
            method=method, 
            max_iterations=max_iterations
        )
    
    def perform_sensitivity_analysis(self, best_params, objective_type='standard'):
        # Select objective function based on type
        objective_functions = {
            'standard': self.objective_function,
            'max_gradient': self.objective_max_gradient,
            'path_focused': self.objective_path_focused,
            'thermal_uniformity': self.objective_thermal_uniformity,
            'max_temp_difference': self.objective_max_temp_difference
        }
        
        objective_func = objective_functions[objective_type]
        
        best_params_array = self.parameters_to_array(best_params)
        base_obj = objective_func(best_params_array)
        
        sensitivity_data = {}
        
        print("\n=== Sensitivity Analysis ===")
        print(f"Base objective value: {base_obj:.6e}")
        
        # Test sensitivity for each parameter
        for i, name in enumerate(self.param_names):
            # Create perturbed parameter sets
            perturb_amount = max(best_params_array[i] * 0.05, 1e-6)  # 5% perturbation
            
            # +5% perturbation
            params_plus = best_params_array.copy()
            params_plus[i] += perturb_amount
            obj_plus = objective_func(params_plus)
            
            # -5% perturbation
            params_minus = best_params_array.copy()
            params_minus[i] -= perturb_amount
            obj_minus = objective_func(params_minus)
            
            # Calculate sensitivity (finite difference approximation)
            sensitivity_plus = (obj_plus - base_obj) / perturb_amount
            sensitivity_minus = (base_obj - obj_minus) / perturb_amount
            
            # Average sensitivity
            avg_sensitivity = (sensitivity_plus + sensitivity_minus) / 2
            
            sensitivity_data[name] = {
                'value': best_params[name],
                'sensitivity': avg_sensitivity,
                'rel_sensitivity': avg_sensitivity * best_params[name] / base_obj
            }
            
            print(f"Parameter: {name}")
            print(f"  Value: {best_params[name]:.6f}")
            print(f"  Absolute Sensitivity: {avg_sensitivity:.6e}")
            print(f"  Relative Sensitivity: {sensitivity_data[name]['rel_sensitivity']:.6e}")
        
        # Sort parameters by relative sensitivity
        sorted_params = sorted(sensitivity_data.items(), 
                              key=lambda x: abs(x[1]['rel_sensitivity']), 
                              reverse=True)
        
        print("\nParameters ranked by sensitivity:")
        for i, (name, data) in enumerate(sorted_params):
            print(f"{i+1}. {name}: {abs(data['rel_sensitivity']):.6e}")
        
        return sensitivity_data
    
 
# ===============================
# 3. Visualization Class
# ===============================
class Visualization:
    def __init__(self, model):
        self.model = model

    def compare_simulations(self, initial_params, optimized_params, use_gaussian=True, y_crop=None, show_full_domain=True):
        
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        
        # Create a temporary optimizer with the correct parameters
        temp_optimizer = TrajectoryOptimizer(self.model, initial_params=initial_params, bounds=None)
        
        # Unpack parameters
        initial_array = temp_optimizer.parameters_to_array(initial_params)
        optimized_array = temp_optimizer.parameters_to_array(optimized_params)
        
        laser_init, heat_init = temp_optimizer.unpack_parameters(initial_array)
        laser_opt, heat_opt = temp_optimizer.unpack_parameters(optimized_array)

        v_init1 = laser_init[0]['v']
        v_init2 = laser_init[1]['v']
        v_opt1 = laser_opt[0]['v']
        v_opt2 = laser_opt[1]['v']

        # Run simulations
        T_init = self.model.simulate((laser_init, heat_init), use_gaussian=use_gaussian)
        T_opt = self.model.simulate((laser_opt, heat_opt), use_gaussian=use_gaussian)

        # Set the color scale bar for the animation to the highest temperature seen during the simulation
        T_min = 0
        T_max = max(np.max(T_init), np.max(T_opt))

        # Create a better temperature colormap
        colors = [(0, 0, 0.3), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0), (1, 1, 1)]
        cmap_temp = LinearSegmentedColormap.from_list('thermal', colors)
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # Assign subplots
        ax1 = fig.add_subplot(gs[0, 0])  # Initial temperature
        ax2 = fig.add_subplot(gs[0, 1])  # Optimized temperature
        ax3 = fig.add_subplot(gs[0, 2])  # Path comparison
        ax4 = fig.add_subplot(gs[1, 0])  # Initial gradient
        ax5 = fig.add_subplot(gs[1, 1])  # Optimized gradient
        # ax6 = fig.add_subplot(gs[1, 2])  # Melt pool metrics (REMOVED)
        
        # Define the full physical extent for proper rendering
        full_extent = [0, self.model.Lx * 1000, 0, self.model.Ly * 1000]  # [x_min, x_max, y_min, y_max] in mm
        
        # Define view limits
        if show_full_domain:
            # Show the entire domain
            x_min, x_max = 0, self.model.Lx * 1000  # mm
            y_min, y_max = 0, self.model.Ly * 1000  # mm
        else:
            # Show the cropped view
            x_min, x_max = 0, self.model.Lx * 1000  # mm
            if y_crop is None:
                y_crop = (0.002, 0.008)  # default crop in meters
            y_min, y_max = y_crop[0] * 1000, y_crop[1] * 1000  # mm
        

        # --- Arc-length-based resampling for physically accurate path visualization ---
        def resample_path(path, v, dt):
            diffs = np.diff(path, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            arc_length = np.concatenate([[0], np.cumsum(seg_lengths)])
            total_length = arc_length[-1]
            n_points = int(np.ceil(total_length / (v * dt)))
            if n_points < 2:
                n_points = 2
            target_lengths = np.linspace(0, total_length, n_points)
            x = np.interp(target_lengths, arc_length, path[:, 0])
            y = np.interp(target_lengths, arc_length, path[:, 1])
            return np.stack([x, y], axis=1)

        # Use fine time step for noisy path
        dt = self.model.dt if hasattr(self.model, 'dt') else 1e-5
        sim_duration = self.model.nt * dt  # Actual simulation duration

        t_points_init1 = np.arange(0, sim_duration, dt)
        t_points_init2 = np.arange(0, sim_duration, dt)
        t_points_opt1 = np.arange(0, sim_duration, dt)
        t_points_opt2 = np.arange(0, sim_duration, dt)

        # Generate noisy paths
        init_saw_path = np.array([self.model.sawtooth_trajectory(t, laser_init[0])[:2] for t in t_points_init1])
        init_swirl_path = np.array([self.model.swirl_trajectory(t, laser_init[1])[:2] for t in t_points_init2])
        opt_saw_path = np.array([self.model.sawtooth_trajectory(t, laser_opt[0])[:2] for t in t_points_opt1])
        opt_swirl_path = np.array([self.model.swirl_trajectory(t, laser_opt[1])[:2] for t in t_points_opt2])

        # Resample at constant arc length intervals
        init_saw_path = resample_path(init_saw_path, v_init1, dt)
        init_swirl_path = resample_path(init_swirl_path, v_init2, dt)
        opt_saw_path = resample_path(opt_saw_path, v_opt1, dt)
        opt_swirl_path = resample_path(opt_swirl_path, v_opt2, dt)
        
        # Convert to mm
        init_saw_path_mm = init_saw_path * 1000
        init_swirl_path_mm = init_swirl_path * 1000
        opt_saw_path_mm = opt_saw_path * 1000
        opt_swirl_path_mm = opt_swirl_path * 1000
        
        # Generate physical coordinates for contours
        x_phys = np.linspace(0, self.model.Lx, self.model.nx) * 1000  # mm
        y_phys = np.linspace(0, self.model.Ly, self.model.ny) * 1000  # mm
        X_phys, Y_phys = np.meshgrid(x_phys, y_phys)

        # Compute gradients for initial and optimized temperature fields
        _, _, grad_init = self.model.compute_temperature_gradients(T_init)
        _, _, grad_opt = self.model.compute_temperature_gradients(T_opt)
        
        # Display temperature field using the full domain for both rendering and extent
        im1 = ax1.imshow(T_init, extent=full_extent, origin='lower', 
                        cmap=cmap_temp, interpolation='bilinear', aspect='auto')
        
        # Plot laser paths on temperature field with increased visibility
        ax1.plot(init_saw_path_mm[:, 0], init_saw_path_mm[:, 1], 'w-', linewidth=2.5, alpha=0.9)
        ax1.plot(init_swirl_path_mm[:, 0], init_swirl_path_mm[:, 1], 'g-', linewidth=2.5, alpha=0.9)


        # Add markers for current positions with increased visibility
        ax1.plot(init_saw_path_mm[-1, 0], init_saw_path_mm[-1, 1], 'wo', markersize=10, markeredgecolor='k', markeredgewidth=1.5)
        ax1.plot(init_swirl_path_mm[-1, 0], init_swirl_path_mm[-1, 1], 'go', markersize=10, markeredgecolor='k', markeredgewidth=1.5)
        # Add legend for the 15mm marker
        ax1.legend(loc='upper right', fontsize=9, frameon=True)
        
        # Add metrics text
        melt_temp = self.model.material.get('T_melt', self.model.material['T0'] + 100)
        melt_mask_init = T_init > melt_temp
        melt_count_init = np.sum(melt_mask_init)
        max_temp_init = np.max(T_init)

        if melt_count_init > 0:
            T_melt_init = T_init[melt_mask_init]
            cv_init = np.std(T_melt_init) / np.mean(T_melt_init) * 100
            metrics_text = f"Max Temp: {max_temp_init:.0f}°C\nMelt Pool: {melt_count_init} pts\nCV: {cv_init:.1f}%"
        else:
            metrics_text = f"Max Temp: {max_temp_init:.0f}°C"
        
        text_box = ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', 
                                                        alpha=0.7))
        # Explicitly set text color to white
        text_box.set_color('white')
        
        ax1.set_title(f'Initial Temperature (Max: {np.max(T_init):.0f}°C)')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        fig.colorbar(im1, ax=ax1, label='Temperature (°C)')
        
        # Plot optimized temperature field
        im2 = ax2.imshow(T_opt, extent=full_extent, origin='lower', 
                        cmap=cmap_temp, interpolation='bilinear', aspect='auto')
        
        # Plot laser paths on temperature field with increased visibility
        ax2.plot(opt_saw_path_mm[:, 0], opt_saw_path_mm[:, 1], 'w-', linewidth=2.5, alpha=0.9)
        ax2.plot(opt_swirl_path_mm[:, 0], opt_swirl_path_mm[:, 1], 'g-', linewidth=2.5, alpha=0.9)
        
        # Add markers for current positions with increased visibility
        ax2.plot(opt_saw_path_mm[-1, 0], opt_saw_path_mm[-1, 1], 'wo', markersize=10, markeredgecolor='k', markeredgewidth=1.5)
        ax2.plot(opt_swirl_path_mm[-1, 0], opt_swirl_path_mm[-1, 1], 'go', markersize=10, markeredgecolor='k', markeredgewidth=1.5)
        
        # Add metrics text
        melt_mask_opt = T_opt > melt_temp
        melt_count_opt = np.sum(melt_mask_opt)
        max_temp_opt = np.max(T_opt)

        if melt_count_opt > 0:
            T_melt_opt = T_opt[melt_mask_opt]
            cv_opt = np.std(T_melt_opt) / np.mean(T_melt_opt) * 100
            metrics_text = f"Max Temp: {max_temp_opt:.0f}°C\nMelt Pool: {melt_count_opt} pts\nCV: {cv_opt:.1f}%"
        else:
            metrics_text = f"Max Temp: {max_temp_opt:.0f}°C"
        
        text_box = ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', 
                                                        alpha=0.7))
        # Explicitly set text color to white
        text_box.set_color('white')
        
        ax2.set_title(f'Optimized Temperature (Max: {np.max(T_opt):.0f}°C)')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        fig.colorbar(im2, ax=ax2, label='Temperature (°C)')
        
        # Plot path comparison
        ax3.plot(init_saw_path_mm[:, 0], init_saw_path_mm[:, 1], 'b-', label='Initial Sawtooth', linewidth=2)
        ax3.plot(init_swirl_path_mm[:, 0], init_swirl_path_mm[:, 1], 'g-', label='Initial Swirl', linewidth=2)
        ax3.plot(opt_saw_path_mm[:, 0], opt_saw_path_mm[:, 1], 'r-', label='Optimized Sawtooth', linewidth=2)
        ax3.plot(opt_swirl_path_mm[:, 0], opt_swirl_path_mm[:, 1], 'm-', label='Optimized Swirl', linewidth=2)
        
        # Add markers
        ax3.plot(init_saw_path_mm[-1, 0], init_saw_path_mm[-1, 1], 'bo', markersize=8, markeredgecolor='k')
        ax3.plot(init_swirl_path_mm[-1, 0], init_swirl_path_mm[-1, 1], 'go', markersize=8, markeredgecolor='k')
        ax3.plot(opt_saw_path_mm[-1, 0], opt_saw_path_mm[-1, 1], 'ro', markersize=8, markeredgecolor='k')
        ax3.plot(opt_swirl_path_mm[-1, 0], opt_swirl_path_mm[-1, 1], 'mo', markersize=8, markeredgecolor='k')
        
        ax3.set_title('Laser Path Comparison')
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot gradient fields
        im4 = ax4.imshow(grad_init/1000, extent=full_extent, origin='lower', 
                        cmap='viridis', interpolation='bilinear', aspect='auto')
        
        # Plot paths on gradient field
        ax4.plot(init_saw_path_mm[:, 0], init_saw_path_mm[:, 1], 'w-', linewidth=1.5, alpha=0.7)
        ax4.plot(init_swirl_path_mm[:, 0], init_swirl_path_mm[:, 1], 'g-', linewidth=1.5, alpha=0.7)
        
        ax4.set_title(f'Initial Gradient (Max: {np.max(grad_init)/1000:.0f}°C/mm)')
        ax4.set_xlabel('X (mm)')
        ax4.set_ylabel('Y (mm)')
        ax4.set_xlim(x_min, x_max)
        ax4.set_ylim(y_min, y_max)
        fig.colorbar(im4, ax=ax4, label='Gradient (°C/mm)')
        
        # Plot optimized gradient field
        im5 = ax5.imshow(grad_opt/1000, extent=full_extent, origin='lower', 
                        cmap='viridis', interpolation='bilinear', aspect='auto')
        
        # Plot paths on gradient field
        ax5.plot(opt_saw_path_mm[:, 0], opt_saw_path_mm[:, 1], 'w-', linewidth=1.5, alpha=0.7)
        ax5.plot(opt_swirl_path_mm[:, 0], opt_swirl_path_mm[:, 1], 'g-', linewidth=1.5, alpha=0.7)
        
        ax5.set_title(f'Optimized Gradient (Max: {np.max(grad_opt)/1000:.0f}°C/mm)')
        ax5.set_xlabel('X (mm)')
        ax5.set_ylabel('Y (mm)')
        ax5.set_xlim(x_min, x_max)
        ax5.set_ylim(y_min, y_max)
        fig.colorbar(im5, ax=ax5, label='Gradient (°C/mm)')
        
        plt.suptitle('LPBF Dual Laser Scan Optimization: Initial vs. Optimized', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.subplots_adjust(bottom=0.1, wspace=0.25, hspace=0.3)  # Adjust spacing

        # Add a label below the figure to explain the scan path colors (matching the animation)
        fig.text(
            0.5, 0.01,
            "White line: Sawtooth scan path    |    Green line: Swirl scan path",
            ha='center', va='bottom', fontsize=11, color='black'
        )

        plt.show()
        
        return fig
    
    def animate_optimization_results(self, model, initial_params, optimized_params, fps=20, 
                               use_gaussian=True, y_crop=None, show_full_domain=True, save_gif=False,
                               filename='laser_optimization.gif', max_frames=200, simulation_duration=0.1):
        """
        Enhanced animation with fixed rendering issues.
        
        Parameters:
        -----------
        model: object
            The simulation model
        initial_params: dict
            Dictionary of initial parameters
        optimized_params: dict
            Dictionary of optimized parameters
        fps: int
            Frames per second for animation
        use_gaussian: bool
            Whether to use Gaussian or Goldak heat source model
        y_crop: tuple or None
            (y_min, y_max) in meters to crop the y-axis. If None, uses (0, model.Ly).
            Only applied if show_full_domain is False.
        show_full_domain: bool
            If True, shows the entire domain regardless of y_crop setting
        save_gif: bool
            Whether to save animation as GIF
        filename: str
            Filename for saving GIF
        max_frames: int
            Maximum number of frames to include in animation
        simulation_duration: float
            Duration to run the simulation in seconds (default: 0.01)
            Increase this value to make the animation run longer and extend further in x-direction
        """
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Patch
        
        # Create a better temperature colormap
        colors = [(0, 0, 0.3), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0), (1, 1, 1)]
        cmap_temp = LinearSegmentedColormap.from_list('thermal', colors)
        
        # Create optimizer and unpack parameters
        temp_optimizer = TrajectoryOptimizer(model, initial_params=initial_params, bounds=None)
        
        initial_array = temp_optimizer.parameters_to_array(initial_params)
        optimized_array = temp_optimizer.parameters_to_array(optimized_params)

        initial_laser_params, initial_heat_params = temp_optimizer.unpack_parameters(initial_array)
        optimized_laser_params, optimized_heat_params = temp_optimizer.unpack_parameters(optimized_array)

        v_init1 = initial_laser_params[0]['v']
        v_init2 = initial_laser_params[1]['v']
        v_opt1 = optimized_laser_params[0]['v']
        v_opt2 = optimized_laser_params[1]['v']

        # Run simulations
        T_init = self.model.simulate((initial_laser_params, initial_heat_params), use_gaussian=use_gaussian)
        T_opt = self.model.simulate((optimized_laser_params, optimized_heat_params), use_gaussian=use_gaussian)

        # Set the color scale bar for the animation to the highest temperature seen during the simulation
        T_min = 0
        T_max = max(np.max(T_init), np.max(T_opt))

        # Define physical coordinate extents
        full_extent = [0, model.Lx * 1000, 0, model.Ly * 1000]  # [x_min, x_max, y_min, y_max] in mm
        
        # Define view limits based on parameters
        if show_full_domain:
            # Show the entire domain
            x_min, x_max = 0, model.Lx * 1000  # mm
            y_min, y_max = 0, model.Ly * 1000  # mm
        else:
            # Show the cropped view
            x_min, x_max = 0, model.Lx * 1000  # mm
            if y_crop is None:
                y_crop = (0.002, 0.008)  # default crop in meters
            y_min, y_max = y_crop[0] * 1000, y_crop[1] * 1000  # mm
        

        # --- Arc-length-based resampling for physically accurate animation ---
        def resample_path(path, v, dt):
            diffs = np.diff(path, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            arc_length = np.concatenate([[0], np.cumsum(seg_lengths)])
            total_length = arc_length[-1]
            n_points = int(np.ceil(total_length / (v * dt)))
            if n_points < 2:
                n_points = 2
            target_lengths = np.linspace(0, total_length, n_points)
            x = np.interp(target_lengths, arc_length, path[:, 0])
            y = np.interp(target_lengths, arc_length, path[:, 1])
            return np.stack([x, y], axis=1)

        # Use fine time step for noisy path
        dt = self.model.dt if hasattr(self.model, 'dt') else 1e-5
        sim_duration = self.model.nt * dt  # Actual simulation duration

        t_points_init1 = np.arange(0, sim_duration, dt)
        t_points_init2 = np.arange(0, sim_duration, dt)
        t_points_opt1 = np.arange(0, sim_duration, dt)
        t_points_opt2 = np.arange(0, sim_duration, dt)

        # Generate noisy paths
        initial_path1 = np.array([model.sawtooth_trajectory(t, initial_laser_params[0])[:2] for t in t_points_init1])
        initial_path2 = np.array([model.swirl_trajectory(t, initial_laser_params[1])[:2] for t in t_points_init2])
        optimized_path1 = np.array([model.sawtooth_trajectory(t, optimized_laser_params[0])[:2] for t in t_points_opt1])
        optimized_path2 = np.array([model.swirl_trajectory(t, optimized_laser_params[1])[:2] for t in t_points_opt2])

        # Resample at constant arc length intervals
        initial_path1 = resample_path(initial_path1, v_init1, dt)
        initial_path2 = resample_path(initial_path2, v_init2, dt)
        optimized_path1 = resample_path(optimized_path1, v_opt1, dt)
        optimized_path2 = resample_path(optimized_path2, v_opt2, dt)

        # Convert to mm for plotting
        initial_path1_mm = initial_path1 * 1000
        initial_path2_mm = initial_path2 * 1000
        optimized_path1_mm = optimized_path1 * 1000
        optimized_path2_mm = optimized_path2 * 1000
        
        # Setup figure and subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Dual Laser Heat Transfer Optimization', fontsize=16)

        # Configure axes with proper mm scale
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.grid(False)  # Remove grid lines

        axes[0].set_title('Initial Parameters', fontsize=14)
        axes[1].set_title('Optimized Parameters', fontsize=14)

        # Initialize temperature field images with fixed color scale
        init_field = np.ones((model.ny, model.nx)) * model.material['T0']

        image_initial = axes[0].imshow(init_field, extent=full_extent, origin='lower', 
                                    cmap=cmap_temp, vmin=T_min, vmax=T_max,
                                    interpolation='bilinear', aspect='auto')

        image_optimized = axes[1].imshow(init_field, extent=full_extent, origin='lower', 
                                    cmap=cmap_temp, vmin=T_min, vmax=T_max,
                                    interpolation='bilinear', aspect='auto')

        # Add colorbars with fixed limits
        cbar_initial = fig.colorbar(image_initial, ax=axes[0])
        cbar_initial.set_label('Temperature (°C)')
        cbar_optimized = fig.colorbar(image_optimized, ax=axes[1])
        cbar_optimized.set_label('Temperature (°C)')
        
        # Physical coordinates for contours
        x_phys = np.linspace(0, model.Lx * 1000, model.nx)
        y_phys = np.linspace(0, model.Ly * 1000, model.ny)
        X_phys, Y_phys = np.meshgrid(x_phys, y_phys)
        
        # Initialize laser markers
        source_marker_initial1, = axes[0].plot([], [], 'wo', markersize=10, markeredgecolor='black',
                                            markeredgewidth=1.5, zorder=10)
        source_marker_initial2, = axes[0].plot([], [], 'go', markersize=10, markeredgecolor='black',
                                            markeredgewidth=1.5, zorder=10)
        source_marker_optimized1, = axes[1].plot([], [], 'wo', markersize=10, markeredgecolor='black',
                                            markeredgewidth=1.5, zorder=10)
        source_marker_optimized2, = axes[1].plot([], [], 'go', markersize=10, markeredgecolor='black',
                                            markeredgewidth=1.5, zorder=10)
        
        # Initialize path lines
        path_line_initial1, = axes[0].plot([], [], 'w-', linewidth=2, alpha=0.8)
        path_line_initial2, = axes[0].plot([], [], 'g-', linewidth=2, alpha=0.8)
        path_line_optimized1, = axes[1].plot([], [], 'w-', linewidth=2, alpha=0.8)
        path_line_optimized2, = axes[1].plot([], [], 'g-', linewidth=2, alpha=0.8)
        
        # Add temperature info text
        temp_text_initial = axes[0].text(0.02, 0.97, '', transform=axes[0].transAxes,
                                    fontsize=10, color='white', verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        temp_text_optimized = axes[1].text(0.02, 0.97, '', transform=axes[1].transAxes,
                                        fontsize=10, color='white', verticalalignment='top',
                                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Add a title for time
        time_title = plt.figtext(0.5, 0.95, '', ha='center', va='top', fontsize=12)
        
        # Add figure legend
        legend_elements = [
            plt.Line2D([0], [0], color='white', marker='o', markersize=8, markerfacecolor='white',
                    markeredgecolor='black', label='Sawtooth Laser'),
            plt.Line2D([0], [0], color='green', marker='o', markersize=8, markerfacecolor='green',
                    markeredgecolor='black', label='Swirl Laser')
        ]
        fig.legend(handles=legend_elements, loc='lower center', 
                bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=12)
        
        # Initialize temperature fields
        T_initial = init_field.copy()
        T_optimized = init_field.copy()
        
        # Reset temperature fields before animation starts
        T_initial = np.ones((model.ny, model.nx)) * model.material['T0']
        T_optimized = np.ones((model.ny, model.nx)) * model.material['T0']

        # Store the variable for updating color limits
        T_max = T_max
        
        def update(frame):
            """Update function with temperature field and laser scan paths."""
            nonlocal T_initial, T_optimized
            if frame == 0:
                T_initial = np.ones((model.ny, model.nx)) * model.material['T0']
                T_optimized = np.ones((model.ny, model.nx)) * model.material['T0']
            # Get actual time value for this frame

            # For arc-length-based animation, use frame index as path index
            t = frame * dt
            time_title.set_text(f'Simulation Time: {t:.3f} s (Step {frame+1}/{len(initial_path1)})')

            # Get current laser source positions (arc-length-based)
            x_src_init1, y_src_init1 = initial_path1[frame] if frame < len(initial_path1) else initial_path1[-1]
            x_src_init2, y_src_init2 = initial_path2[frame] if frame < len(initial_path2) else initial_path2[-1]
            x_src_opt1, y_src_opt1 = optimized_path1[frame] if frame < len(optimized_path1) else optimized_path1[-1]
            x_src_opt2, y_src_opt2 = optimized_path2[frame] if frame < len(optimized_path2) else optimized_path2[-1]

            # Convert to mm for plotting
            x_src_init1_mm = x_src_init1 * 1000
            y_src_init1_mm = y_src_init1 * 1000
            x_src_init2_mm = x_src_init2 * 1000
            y_src_init2_mm = y_src_init2 * 1000
            x_src_opt1_mm = x_src_opt1 * 1000
            y_src_opt1_mm = y_src_opt1 * 1000
            x_src_opt2_mm = x_src_opt2 * 1000
            y_src_opt2_mm = y_src_opt2 * 1000

            # Update marker positions
            source_marker_initial1.set_data([x_src_init1_mm], [y_src_init1_mm])
            source_marker_initial2.set_data([x_src_init2_mm], [y_src_init2_mm])
            source_marker_optimized1.set_data([x_src_opt1_mm], [y_src_opt1_mm])
            source_marker_optimized2.set_data([x_src_opt2_mm], [y_src_opt2_mm])

            # Update path lines (accumulating trajectory history)
            current_frame = frame + 1
            path_line_initial1.set_data(initial_path1_mm[:current_frame, 0], initial_path1_mm[:current_frame, 1])
            path_line_initial2.set_data(initial_path2_mm[:current_frame, 0], initial_path2_mm[:current_frame, 1])
            path_line_optimized1.set_data(optimized_path1_mm[:current_frame, 0], optimized_path1_mm[:current_frame, 1])
            path_line_optimized2.set_data(optimized_path2_mm[:current_frame, 0], optimized_path2_mm[:current_frame, 1])
        
            # --- Update temperature fields for both initial and optimized ---
            # 1. Diffusion step
            lap_init = np.zeros_like(T_initial)
            lap_init[1:-1, 1:-1] = ((T_initial[1:-1, 2:] - 2*T_initial[1:-1, 1:-1] + T_initial[1:-1, :-2]) / model.dx**2 +
                                    (T_initial[2:, 1:-1] - 2*T_initial[1:-1, 1:-1] + T_initial[:-2, 1:-1]) / model.dy**2)
            lap_opt = np.zeros_like(T_optimized)
            lap_opt[1:-1, 1:-1] = ((T_optimized[1:-1, 2:] - 2*T_optimized[1:-1, 1:-1] + T_optimized[1:-1, :-2]) / model.dx**2 +
                                   (T_optimized[2:, 1:-1] - 2*T_optimized[1:-1, 1:-1] + T_optimized[:-2, 1:-1]) / model.dy**2)

            alpha = model.material.get('alpha', model.material['k'] / model.heat_capacity)
            T_init_new = T_initial + model.dt * alpha * lap_init
            T_opt_new = T_optimized + model.dt * alpha * lap_opt

            # 2. Heat source step (Gaussian)
            S_init1 = model._gaussian_source(x_src_init1, y_src_init1, initial_heat_params[0])
            S_init2 = model._gaussian_source(x_src_init2, y_src_init2, initial_heat_params[1])
            S_opt1 = model._gaussian_source(x_src_opt1, y_src_opt1, optimized_heat_params[0])
            S_opt2 = model._gaussian_source(x_src_opt2, y_src_opt2, optimized_heat_params[1])

            thickness = model.material.get('thickness', 0.00017)
            rho = model.material['rho']
            cp = model.material['cp']

            T_init_new += (S_init1 + S_init2) * (model.dt / (model.dx * model.dy * thickness * rho * cp))
            T_opt_new += (S_opt1 + S_opt2) * (model.dt / (model.dx * model.dy * thickness * rho * cp))

            # 3. Boundary conditions (Dirichlet: fixed at T0)
            T_init_new[0, :] = model.material['T0']
            T_init_new[-1, :] = model.material['T0']
            T_init_new[:, 0] = model.material['T0']
            T_init_new[:, -1] = model.material['T0']
            T_opt_new[0, :] = model.material['T0']
            T_opt_new[-1, :] = model.material['T0']
            T_opt_new[:, 0] = model.material['T0']
            T_opt_new[:, -1] = model.material['T0']

            # Update temperature fields
            T_initial = T_init_new
            T_optimized = T_opt_new

            # Update temperature images
            image_initial.set_data(T_initial)
            image_optimized.set_data(T_optimized)

            # Update color limits if needed
            image_initial.set_clim(vmin=T_min, vmax=T_max)
            image_optimized.set_clim(vmin=T_min, vmax=T_max)

            # Update temperature info text
            max_temp_init = np.max(T_initial)
            max_temp_opt = np.max(T_optimized)
            init_text = f'Max Temp: {max_temp_init:.0f}°C'
            opt_text = f'Max Temp: {max_temp_opt:.0f}°C'
            temp_text_initial.set_text(init_text)
            temp_text_optimized.set_text(opt_text)

            artists = [
                image_initial, image_optimized,
                source_marker_initial1, source_marker_initial2,
                source_marker_optimized1, source_marker_optimized2,
                path_line_initial1, path_line_initial2,
                path_line_optimized1, path_line_optimized2,
                temp_text_initial, temp_text_optimized
            ]
            return artists
        
        # Create animation with fewer frames for smoother performance

        # Use the minimum length among all arc-length-resampled paths for frame count
        frame_count = min(len(initial_path1), len(initial_path2), len(optimized_path1), len(optimized_path2))
        # Cap at reasonable number of frames
        if frame_count > max_frames:
            frame_skip = max(1, frame_count // max_frames)
            frames_to_use = list(range(0, frame_count, frame_skip))
        else:
            frames_to_use = list(range(frame_count))
        
        ani = animation.FuncAnimation(
            fig, update, frames=frames_to_use,
            interval=1000/fps, blit=False  # <--- set blit to False
        )
        
        # --- Save frames at every 0.01s of simulated time ---
        import os
        save_frames = True  # Set to True to enable saving
        frame_folder = os.path.join(os.getcwd(), "animation_frames")
        if save_frames:
            os.makedirs(frame_folder, exist_ok=True)
            saved_times = set()
            save_interval = 0.01  # seconds
            next_save_time = 0.0
            for idx, frame in enumerate(frames_to_use):
                t = frame * dt  # Approximate simulation time for this frame
                # Save frame if we've reached or passed the next interval
                if t >= next_save_time - 1e-6:
                    update(frame)  # Update the figure to this frame
                    fname = os.path.join(frame_folder, f"frame_{t:.3f}s.png")
                    plt.savefig(fname, dpi=150)
                    saved_times.add(round(t, 3))
                    next_save_time += save_interval
            print(f"Saved frames at times (s): {sorted(saved_times)}")
        
        # Save animation if requested
        if save_gif:
            print(f"Saving animation to {filename}...")
            ani.save(filename, writer='pillow', fps=fps, dpi=120)
            print(f"Animation saved to {filename}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(bottom=0.1, wspace=0.3)
       
        plt.show()
        
        return ani

def run_optimization():
    # 1. Setup material parameters for LPBF
    material_params = {
        'T0': 21.0,                # Initial temperature (°C)
        'alpha': 5e-6,             # Thermal diffusivity (m²/s)
        'rho': 7800.0,             # Density (kg/m³)
        'cp': 500.0,               # Specific heat capacity (J/kg·K)

        'thickness':  0.00017,       # Thickness (m)
        'T_melt': 1500.0,          # Melting temperature (°C)
        'k': 20.0,                 # Thermal conductivity (W/m·K)
        'absorptivity': 1,       # Laser absorptivity
    }
    
    # 2. Domain settings
    domain_size = (0.02, 0.0025)     # 20mm x 2.5mm domain
    grid_size = (201, 26)           # Increase grid points for new aspect ratio (optional, can adjust)
    dt = 1e-4                      # 10μs time step

    # 3. Initialize model
    model = HeatTransferModel(
        domain_size=domain_size,
        grid_size=grid_size,
        dt=dt,
        material_params=material_params
    )
    
    # 4. Define initial parameters
    initial_params = {
        # Sawtooth path parameters
        'sawtooth_v': 0.05,         # 50 mm/s scan speed
        'sawtooth_A': 0.0002,        # 0.2mm amplitude

        'sawtooth_y0': 0.00125,      # Center position (fixed)
        'sawtooth_period': 0.01,    # 10ms period
        'sawtooth_Q': 200.0,        # 200W laser power
        'sawtooth_r0': 5e-5,        # 50μm beam radius

        # Swirl/Spiral path parameters
        'swirl_v': 0.05,            # 50 mm/s scan speed
        'swirl_A': 0.0002,          # 0.2mm amplitude
        'swirl_y0': 0.00125,         # Center position (fixed)
        'swirl_fr': 20.0,           # 20 Hz frequency
        'swirl_Q': 200.0,           # 200W laser power
        'swirl_r0': 5e-5,           # 50μm beam radius

        # Noise parameter
        'noise_sigma': 0.000      # Initial value for noise
    }

    # 5. Define parameter bounds
    bounds = {
        'sawtooth_v': (0.01, 0.1),
        'swirl_v': (0.01, 0.1),
        'sawtooth_A': (0.00005, 0.00025),
        'swirl_A': (0.00005, 0.00025),
        'sawtooth_period': (0.01, 0.05),
        'swirl_fr': (5.0, 20.0),
        'sawtooth_Q': (50.0, 500.0),
        'swirl_Q': (50.0, 500.0),
        'sawtooth_r0': (3e-5, 8e-5),
        'swirl_r0': (3e-5, 8e-5),
        'noise_sigma': (0.0, 0.0005)  # Bounds for noise
    }

    # Set fixed laser parameters for simulation
    # (No need for fixed_heat here; heat parameters are set in the optimizer)

    # 6. Create trajectory optimizer
    optimizer = TrajectoryOptimizer(
        model=model,
        initial_params=initial_params,
        bounds=bounds,
        x_range=(0.0025, 0.0175)  # 2.5mm to 17.5mm scan length
    )
    
    # 7. Run optimization
    print("\n=============================================")
    print("Starting Laser Path Optimization")
    print("=============================================\n")
    
    # Choose method and objective
    method = 'Powell'  # Options: 'SLSQP', 'L-BFGS-B', 'Nelder-Mead', 'Powell'
    objective = 'standard'  # Options: 'standard', 'thermal_uniformity', 'max_gradient', etc.

    # Add a line describing what is being optimized for
    objective_descriptions = {
        'standard': "Minimizing the sum of squared temperature gradients (smoothness of temperature field).",
        'thermal_uniformity': "Promoting thermal uniformity (variance in melt pool temperature and gradients).",
        'max_gradient': "Minimizing the maximum temperature gradient (reducing hot spots).",
        'path_focused': "Minimizing gradients along the laser paths (melt pool quality along scan).",
        'max_temp_difference': "Minimizing the maximum temperature difference in the melt pool."
    }
    print(f"Optimization objective: {objective_descriptions.get(objective, 'Unknown objective')}")

    # Run the optimization
    try:
        result, optimized_params = optimizer.optimize(
            objective_type=objective,
            method=method,
            max_iterations=1000
        )
        
        optimization_successful = result.success
        
    except Exception as e:
        print(f"Optimization failed with error: {str(e)}")
        print("Falling back to derivative-free method (Nelder-Mead)")
        
        # Fall back to Nelder-Mead if other methods fail
        result, optimized_params = optimizer.optimize(
            objective_type=objective,
            method='Nelder-Mead',
            max_iterations=1000
        )
        
        optimization_successful = result.success
    
    # 8. Analyze and visualize results
    if optimization_successful:
        print("\n=============================================")
        print("Optimization Successful!")
        print("=============================================\n")
        
        # Print objective function improvement
        initial_obj = optimizer.objective_thermal_uniformity(
     optimizer.parameters_to_array(initial_params)
        )
        final_obj = optimizer.objective_thermal_uniformity(
            optimizer.parameters_to_array(optimized_params)
        )
        
        improvement = (initial_obj - final_obj) / initial_obj *  100
        print(f"Initial objective value: {initial_obj:.4e}")
        print(f"Final objective value: {final_obj:.4e}")
        print(f"Improvement: {improvement:.2f}%\n")
        
        # Compare initial and optimized parameters
        print("Parameter Comparison:")
        print("---------------------------------------------")
        print(f"{'Parameter':<15} {'Initial':<12} {'Optimized':<12} {'Change %':<10}")
        print("---------------------------------------------")
        for name in optimizer.param_names:
            init_val = initial_params[name]
            opt_val = optimized_params[name]
            change = (opt_val - init_val) / init_val * 100 if init_val != 0 else float('inf')
            print(f"{name:<15} {init_val:<12.6f} {opt_val:<12.6f} {change:<10.2f}")
        
        # Create visualization
        viz = Visualization(model)
        
        # Compare initial and optimized simulation results
        print("\nGenerating comparison visualizations...")
        viz.compare_simulations(initial_params, optimized_params)
        
        # Optional: Generate animation
        create_animation = input("Generate animation of optimization results? (y/n): ").lower()
        if create_animation == 'y':
            print("Generating animation (this may take a while)...")
            ani = viz.animate_optimization_results(
                model, 
                initial_params, 
                optimized_params, 
                fps=30,
                # Optionally, you can add arguments to crop the view to the scan region
                # show_full_domain=False, y_crop=None
            )
            
        # Optional: Perform sensitivity analysis
        run_sensitivity = input("Perform sensitivity analysis of optimized solution? (y/n): ")

        if run_sensitivity == 'y':
            print("\nPerforming sensitivity analysis...")
            sensitivity_data = optimizer.perform_sensitivity_analysis(
                optimized_params, 
                objective_type=objective
            )
            
        print("\nOptimization process complete!")
        
    else:
        print("\n=============================================")
        print("Optimization Failed")
        print("=============================================\n")
        print(f"Error message: {result.message}")
        print("Try with a different optimization method or adjust parameters.")
    
    return model, optimizer, result, optimized_params# Run the optimization if script is executed directly
if __name__ == "__main__":
    model, optimizer, result, optimized_params = run_optimization()


import os
def output_optimized_gcode(optimizer, optimized_params, filename="optimized_scan.gcode"):
    """
    Output the optimized scan path for both lasers as G-code.
    Args:
        optimizer: TrajectoryOptimizer instance
        optimized_params: dict of optimized parameters
        filename: output G-code filename
    """
    # Unpack parameters
    params_array = optimizer.parameters_to_array(optimized_params)
    laser_params, _ = optimizer.unpack_parameters(params_array)
    sawtooth_params, swirl_params = laser_params

    # Helper to resample a path at constant arc length intervals
    def resample_path(path, v, dt=1e-5):
        # Compute cumulative arc length
        diffs = np.diff(path, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        arc_length = np.concatenate([[0], np.cumsum(seg_lengths)])
        total_length = arc_length[-1]
        # Number of points based on total length and speed
        n_points = int(np.ceil(total_length / (v * dt)))
        if n_points < 2:
            n_points = 2
        target_lengths = np.linspace(0, total_length, n_points)
        # Interpolate x and y separately
        x = np.interp(target_lengths, arc_length, path[:, 0])
        y = np.interp(target_lengths, arc_length, path[:, 1])
        return np.stack([x, y], axis=1)

    # Determine scan duration (use slowest v)
    x_start, x_end = optimizer.x_range
    v1 = sawtooth_params['v']
    v2 = swirl_params['v']
    # Use a fine time step to capture noise
    dt = 1e-5
    total_time = max((x_end - x_start) / v1, (x_end - x_start) / v2)
    t_points = np.arange(0, total_time, dt)

    # Generate noisy paths at fine time intervals
    saw_path = np.array([optimizer.model.sawtooth_trajectory(t, sawtooth_params)[:2] for t in t_points])
    swirl_path = np.array([optimizer.model.swirl_trajectory(t, swirl_params)[:2] for t in t_points])

    # Resample paths at constant arc length intervals (matching the speed)
    saw_path_resampled = resample_path(saw_path, v1, dt)
    swirl_path_resampled = resample_path(swirl_path, v2, dt)

    # Write G-code
    with open(filename, 'w') as f:
        f.write("; Optimized dual-laser scan G-code\n")
        f.write("G21 ; Set units to mm\n")
        f.write("G90 ; Absolute positioning\n")
        f.write("M3 S255 ; Laser ON (example)\n")
        f.write("; Sawtooth laser path\n")
        for x, y in saw_path_resampled * 1000:  # convert to mm
            f.write(f"G1 X{x:.3f} Y{y:.3f} ; Sawtooth\n")
        f.write("; Swirl laser path\n")
        for x, y in swirl_path_resampled * 1000:
            f.write(f"G1 X{x:.3f} Y{y:.3f} ; Swirl\n")
        f.write("M5 ; Laser OFF\n")
    print(f"G-code written to {filename}")

if __name__ == "__main__":
    # ...existing code for setting up and running optimization...
    # Example:
    # optimizer = TrajectoryOptimizer(...)
    # result, optimized_params = optimizer.optimize(...)
    # (Replace above with your actual optimization code)

    # Prompt for G-code export after optimization
    user_input = input("\nWould you like to output the optimized scan as G-code? (y/n): ").strip().lower()
    if user_input == 'y':
        output_optimized_gcode(optimizer, optimized_params)
    else:
        print("G-code export skipped.")
