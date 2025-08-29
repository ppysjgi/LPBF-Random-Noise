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
            'thickness': 0.0041  # Plate thickness (m)
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

    def generate_raster_scan(self, x_start, x_end, y_start, y_end, n_lines, speed):
        """Generate a raster scan path covering the area."""
        import numpy as np
        x_vals = np.linspace(x_start, x_end, self.model.nx)
        y_vals = np.linspace(y_start, y_end, n_lines)
        path = []
        for i, y in enumerate(y_vals):
            xs = x_vals if i % 2 == 0 else x_vals[::-1]
            for x in xs:
                path.append((x, y))
        return np.array(path), speed

    def compare_simulations(self, initial_params, optimized_params, use_gaussian=True, y_crop=None, show_full_domain=True):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap

        # --- Raster scan setup ---
        x_start, x_end = 0, self.model.Lx
        y_start = 0.0010  # 1.0 mm
        y_end = 0.0015    # 1.5 mm
        hatch_spacing = 0.00005  # 0.05 mm
        n_lines = int(np.floor((y_end - y_start) / hatch_spacing)) + 1
        raster_speed = 0.05  # 50 mm/s in m/s
        raster_path, raster_v = self.generate_raster_scan(x_start, x_end, y_start, y_end, n_lines, raster_speed)

        # Simulate raster scan: treat as a single laser moving along raster_path
        heat_params = (
            {'Q': 200.0, 'r0': 0.00005},  # 200W, 0.05mm spot size
            {'Q': 200.0, 'r0': 0.00005}
        )
        def raster_trajectory(t, params):
            idx = int(t * raster_v * len(raster_path) / (x_end - x_start))
            idx = np.clip(idx, 0, len(raster_path) - 1)
            x, y = raster_path[idx]
            return x, y, 1, 0

        orig_sawtooth = self.model.sawtooth_trajectory
        orig_swirl = self.model.swirl_trajectory
        self.model.sawtooth_trajectory = raster_trajectory
        self.model.swirl_trajectory = raster_trajectory

        raster_laser_params = (
            {'v': raster_v, 'A': 0, 'y0': 0, 'period': 1, 'noise_sigma': 0},
            {'v': raster_v, 'A': 0, 'y0': 0, 'fr': 1, 'noise_sigma': 0}
        )
        T_raster = self.model.simulate((raster_laser_params, heat_params), use_gaussian=use_gaussian)

        # Restore original trajectory functions
        self.model.sawtooth_trajectory = orig_sawtooth
        self.model.swirl_trajectory = orig_swirl

        # --- Optimized simulation ---
        temp_optimizer = TrajectoryOptimizer(self.model, initial_params=initial_params, bounds=None)
        optimized_array = temp_optimizer.parameters_to_array(optimized_params)
        laser_opt, heat_opt = temp_optimizer.unpack_parameters(optimized_array)
        T_opt = self.model.simulate((laser_opt, heat_opt), use_gaussian=use_gaussian)

        # --- Generate scan paths for overlay ---
        dt = self.model.dt
        nt = self.model.nt
        t_points = np.linspace(0, nt * dt, nt)
        raster_x_time = []
        raster_y_time = []
        for t in t_points:
            x, y, _, _ = raster_trajectory(t, raster_laser_params[0])
            raster_x_time.append(x * 1000)
            raster_y_time.append(y * 1000)

        sawtooth_params, swirl_params = laser_opt
        saw_x = []
        saw_y = []
        swirl_x = []
        swirl_y = []
        for t in t_points:
            x1, y1, _, _ = self.model.sawtooth_trajectory(t, sawtooth_params)
            x2, y2, _, _ = self.model.swirl_trajectory(t, swirl_params)
            saw_x.append(x1 * 1000)
            saw_y.append(y1 * 1000)
            swirl_x.append(x2 * 1000)
            swirl_y.append(y2 * 1000)

        # --- Compute gradients ---
        _, _, grad_raster = self.model.compute_temperature_gradients(T_raster)
        _, _, grad_opt = self.model.compute_temperature_gradients(T_opt)

        # --- Visualization ---
        colors = [(0, 0, 0.3), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0), (1, 1, 1)]
        cmap_temp = LinearSegmentedColormap.from_list('thermal', colors)
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        extent = [0, self.model.Lx * 1000, 0, self.model.Ly * 1000]
        im1 = ax1.imshow(T_raster, extent=extent, origin='lower', cmap=cmap_temp, aspect='auto')
        ax1.set_title('Raster Scan Temperature')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        fig.colorbar(im1, ax=ax1, label='Temperature (°C)')
        ax1.plot(raster_x_time, raster_y_time, color='black', linewidth=1, label='Raster Laser Path')
        ax1.legend()

        im2 = ax2.imshow(T_opt, extent=extent, origin='lower', cmap=cmap_temp, aspect='auto')
        ax2.set_title('Optimized Scan Temperature')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        fig.colorbar(im2, ax=ax2, label='Temperature (°C)')
        ax2.plot(saw_x, saw_y, color='red', linewidth=1, label='Sawtooth Path')
        ax2.plot(swirl_x, swirl_y, color='black', linewidth=1, label='Swirl Path')
        ax2.legend();

        # --- Gradient plots with overlays ---
        grad_cmap = plt.get_cmap('viridis')
        im3 = ax3.imshow(grad_raster, extent=extent, origin='lower', cmap=grad_cmap, aspect='auto')
        ax3.set_title('Raster Scan Temperature Gradient')
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        fig.colorbar(im3, ax=ax3, label='|∇T| (°C/mm)')
        ax3.plot(raster_x_time, raster_y_time, color='black', linewidth=1, label='Raster Laser Path')
        ax3.legend()

        im4 = ax4.imshow(grad_opt, extent=extent, origin='lower', cmap=grad_cmap, aspect='auto')
        ax4.set_title('Optimized Scan Temperature Gradient')
        ax4.set_xlabel('X (mm)')
        ax4.set_ylabel('Y (mm)')
        fig.colorbar(im4, ax=ax4, label='|∇T| (°C/mm)')
        ax4.plot(saw_x, saw_y, color='red', linewidth=1, label='Sawtooth Path')
        ax4.plot(swirl_x, swirl_y, color='black', linewidth=1, label='Swirl Path')
        ax4.legend()

        plt.tight_layout()
        plt.show()
        return fig

    def animate_optimization_results(self, model, initial_params, optimized_params, fps=30, show_full_domain=True, y_crop=None, save_snapshots=True, snapshot_interval=0.01, snapshot_dir="animation_snapshots"):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
        import os

        # --- Raster scan setup ---
        x_start, x_end = 0, self.model.Lx
        y_start = 0.0010  # 1.0 mm
        y_end = 0.0015    # 1.5 mm
        hatch_spacing = 0.00005  # 0.05 mm
        n_lines = int(np.floor((y_end - y_start) / hatch_spacing)) + 1
        raster_speed = 0.2  # 200 mm/s in m/s
        raster_path, raster_v = self.generate_raster_scan(x_start, x_end, y_start, y_end, n_lines, raster_speed)

        heat_params = (
            {'Q': 200.0, 'r0': 0.00005},
            {'Q': 200.0, 'r0': 0.00005}
        )
        def raster_trajectory(t, params):
            idx = int(t * raster_v * len(raster_path) / (x_end - x_start))
            idx = np.clip(idx, 0, len(raster_path) - 1)
            x, y = raster_path[idx]
            return x, y, 1, 0

        # --- Optimized scan setup ---
        temp_optimizer = TrajectoryOptimizer(self.model, initial_params=initial_params, bounds=None)
        optimized_array = temp_optimizer.parameters_to_array(optimized_params)
        laser_opt, heat_opt = temp_optimizer.unpack_parameters(optimized_array)
        sawtooth_params, swirl_params = laser_opt

        # --- Animation setup ---
        dt = self.model.dt
        nt = self.model.nt
        t_points = np.linspace(0, nt * dt, nt)
        colors = [(0, 0, 0.3), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0), (1, 1, 1)]
        cmap_temp = LinearSegmentedColormap.from_list('thermal', colors)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        extent = [0, self.model.Lx * 1000, 0, self.model.Ly * 1000]

        # Initial temperature fields
        T_raster = self.model.material['T0'] * np.ones((self.model.ny, self.model.nx))
        T_opt = self.model.material['T0'] * np.ones((self.model.ny, self.model.nx))

        # Set temperature scale (vmin/vmax) for better visualization
        temp_vmin = self.model.material['T0']
        temp_vmax = 200  # Set to 200°C

        im1 = ax1.imshow(T_raster, extent=extent, origin='lower', cmap=cmap_temp, aspect='auto', vmin=temp_vmin, vmax=temp_vmax)
        ax1.set_title('Raster Scan Temperature')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        raster_laser_dot, = ax1.plot([], [], 'ko', markersize=6, label='Raster Laser')
        ax1.legend()

        im2 = ax2.imshow(T_opt, extent=extent, origin='lower', cmap=cmap_temp, aspect='auto', vmin=temp_vmin, vmax=temp_vmax)
        ax2.set_title('Optimized Scan Temperature')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        saw_dot, = ax2.plot([], [], 'ro', markersize=6, label='Sawtooth Laser')
        swirl_dot, = ax2.plot([], [], 'ko', markersize=6, label='Swirl Laser')
        ax2.legend()

        plt.tight_layout()

        # Precompute raster positions for speed
        raster_positions = [raster_trajectory(t, {'v': raster_v, 'A': 0, 'y0': 0, 'period': 1, 'noise_sigma': 0}) for t in t_points]
        # Precompute optimized positions for speed
        saw_positions = [self.model.sawtooth_trajectory(t, sawtooth_params) for t in t_points]
        swirl_positions = [self.model.swirl_trajectory(t, swirl_params) for t in t_points]

        # --- Snapshot saving setup ---
        if save_snapshots:
            if not os.path.exists(snapshot_dir):
                os.makedirs(snapshot_dir)
            snapshot_frames = set()
            total_time = nt * dt
            snapshot_times = np.arange(0, total_time, snapshot_interval)
            # Find closest frame indices for each snapshot time
            for snap_time in snapshot_times:
                frame_idx = int(np.round(snap_time / dt))
                if frame_idx < nt:
                    snapshot_frames.add(frame_idx)

        def update(frame):
            # Raster scan update
            frame_idx = frame
            if frame_idx >= len(raster_path):
                frame_idx = len(raster_path) - 1
            x_r, y_r = raster_path[frame_idx]
            S_r = self.model._gaussian_source(x_r, y_r, heat_params[0])
            if frame == 0:
                update.T_raster = self.model.material['T0'] * np.ones((self.model.ny, self.model.nx))
            lap_r = self.model._laplacian(update.T_raster)
            update.T_raster = update.T_raster + dt * self.model.material['alpha'] * lap_r
            update.T_raster = update.T_raster + dt * S_r / (self.model.dx * self.model.dy * self.model.thickness * self.model.material['rho'] * self.model.material['cp'])
            update.T_raster[0, :] = self.model.material['T0']
            update.T_raster[-1, :] = self.model.material['T0']
            update.T_raster[:, 0] = self.model.material['T0']
            update.T_raster[:, -1] = self.model.material['T0']
            im1.set_array(update.T_raster)
            raster_laser_dot.set_data([x_r * 1000], [y_r * 1000])

            # Optimized scan update (dual-laser)
            x_saw, y_saw, _, _ = saw_positions[frame]
            x_swirl, y_swirl, _, _ = swirl_positions[frame]
            S1 = self.model._gaussian_source(x_saw, y_saw, heat_opt[0])
            S2 = self.model._gaussian_source(x_swirl, y_swirl, heat_opt[1])
            S_total = S1 + S2
            if frame == 0:
                update.T_opt = self.model.material['T0'] * np.ones((self.model.ny, self.model.nx))
            lap_opt = self.model._laplacian(update.T_opt)
            update.T_opt = update.T_opt + dt * self.model.material['alpha'] * lap_opt
            update.T_opt = update.T_opt + dt * S_total / (self.model.dx * self.model.dy * self.model.thickness * self.model.material['rho'] * self.model.material['cp'])
            update.T_opt[0, :] = self.model.material['T0']
            update.T_opt[-1, :] = self.model.material['T0']
            update.T_opt[:, 0] = self.model.material['T0']
            update.T_opt[:, -1] = self.model.material['T0']
            im2.set_array(update.T_opt)
            saw_dot.set_data([x_saw * 1000], [y_saw * 1000])
            swirl_dot.set_data([x_swirl * 1000], [y_swirl * 1000])

            # --- Save snapshot every snapshot_interval seconds ---
            if save_snapshots and frame in snapshot_frames:
                fname = os.path.join(snapshot_dir, f"snapshot_{frame:05d}.png")
                fig.savefig(fname)
            return im1, raster_laser_dot, im2, saw_dot, swirl_dot

        update.T_raster = T_raster.copy()
        update.T_opt = T_opt.copy()

        ani = FuncAnimation(fig, update, frames=nt, interval=1000 / fps, blit=False)
        plt.show()
        return ani

# ===============================
# 4. Main Optimization Routine
# ===============================
def run_optimization():
    # 1. Setup material parameters for LPBF
    material_params = {
        'T0': 21.0,                # Initial temperature (°C)
        'alpha': 5e-6,             # Thermal diffusivity (m²/s)
        'rho': 7800.0,             # Density (kg/m³)
        'cp': 500.0,               # Specific heat capacity (J/kg·K)

        'thickness':  0.0041,       # Thickness (m)
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
