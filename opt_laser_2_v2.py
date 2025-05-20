import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.optimize import minimize
from autograd import grad, jacobian, hessian
import autograd.numpy as anp 
import cvxpy as cp

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
        
        # Default material parameters for LPBF simulation (all in SI units)
        default_params = {
            'T0': 20.0,           # Initial temperature (°C)
            'alpha': 5e-6,        # Thermal diffusivity (m²/s)
            'rho': 7800.0,        # Density (kg/m³)
            'cp': 500.0,          # Specific heat capacity (J/(kg·K))
            'k': 20.0,            # Thermal conductivity (W/(m·K)
            'T_melt': 1500.0,     # Melting temperature (°C)
            'thickness': 0.1    # Plate thickness (m)
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
        # added stuff to make differentiable if needed
        v = params['v']
        A = params['A']
        y0 = params['y0']
        period = params['period']
        
        raw_x = v * t
        k = 1000  # sharpness parameter
        x = raw_x - (raw_x - (self.Lx - 0.0005)) * (1 / (1 + anp.exp(-k * (raw_x - (self.Lx - 0.0005)))))
        y = y0 + A * (2/anp.pi) * anp.arcsin(anp.sin(2 * anp.pi * t / period))
        tx = v
        ty = (2 * A / period) * anp.cos(2 * anp.pi * t / period)
        norm = anp.sqrt(tx**2 + ty**2)
        return x, y, tx / norm, ty / norm

    def swirl_trajectory(self, t, params):
        v = params['v']
        A = params['A']
        y0 = params['y0']
        fr = params['fr']
        om = 2 * anp.pi * fr
        
        raw_x = v * t + A * anp.sin(om * t)
        k = 1000
        x = raw_x - (raw_x - (self.Lx - 0.0005)) * (1 / (1 + anp.exp(-k * (raw_x - (self.Lx - 0.0005)))))
        y = y0 + A * anp.cos(om * t)
        tx = v + A * om * anp.cos(om * t)
        ty = -A * om * anp.sin(om * t)
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

        power = heat_params.get('Q', 300.0)  # Power in Watts
        
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

        fixed_nt = 500
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
            source_increment = S_total / (self.dx * self.dy * self.thickness * self.material['rho'] * self.material['cp'])
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
        self.param_names = list(initial_params.keys()) if initial_params else None
        
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
        
        # Unpack laser path parameters for sawtooth and swirl trajectories
        sawtooth_params = {
            'v': params_dict['sawtooth_v'],
            'A': params_dict['sawtooth_A'],
            'y0': params_dict['sawtooth_y0'],
            'period': params_dict['sawtooth_period']
        }
        
        swirl_params = {
            'v': params_dict['swirl_v'],
            'A': params_dict['swirl_A'],
            'y0': params_dict['swirl_y0'],
            'fr': params_dict['swirl_fr']
        }
        
        # Heat params can use either Goldak or Gaussian model
        # If r0 is provided, add it to heat_params
        heat_param_keys = ['Q', 'r0'] if 'r0' in params_dict else ['Q', 'a_f', 'a_b', 'b_param', 'f_f', 'f_b']
        
        # Create heat parameters for both lasers (usually identical)
        heat_params = (
            {key: params_dict[key] for key in heat_param_keys if key in params_dict},
            {key: params_dict[key] for key in heat_param_keys if key in params_dict}
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

        # Run simulations
        T_init = self.model.simulate((laser_init, heat_init), use_gaussian=use_gaussian)
        T_opt = self.model.simulate((laser_opt, heat_opt), use_gaussian=use_gaussian)

        # Compute temperature gradients
        _, _, grad_init = self.model.compute_temperature_gradients(T_init)
        _, _, grad_opt = self.model.compute_temperature_gradients(T_opt)

        # Identify melt pools
        melt_temp = self.model.material.get('T_melt', self.model.material['T0'] + 100)
        melt_mask_init = T_init > melt_temp
        melt_mask_opt = T_opt > melt_temp

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
        ax6 = fig.add_subplot(gs[1, 2])  # Melt pool metrics
        
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
        
        # Generate path data
        path_duration = 0.05
        t_points = np.linspace(0, path_duration, 300)
        
        # Calculate paths (in physical coordinates)
        init_saw_path = np.array([self.model.sawtooth_trajectory(t, laser_init[0])[:2] for t in t_points])
        init_swirl_path = np.array([self.model.swirl_trajectory(t, laser_init[1])[:2] for t in t_points])
        opt_saw_path = np.array([self.model.sawtooth_trajectory(t, laser_opt[0])[:2] for t in t_points])
        opt_swirl_path = np.array([self.model.swirl_trajectory(t, laser_opt[1])[:2] for t in t_points])
        
        # Convert to mm
        init_saw_path_mm = init_saw_path * 1000
        init_swirl_path_mm = init_swirl_path * 1000
        opt_saw_path_mm = opt_saw_path * 1000
        opt_swirl_path_mm = opt_swirl_path * 1000
        
        # Generate physical coordinates for contours
        x_phys = np.linspace(0, self.model.Lx, self.model.nx) * 1000  # mm
        y_phys = np.linspace(0, self.model.Ly, self.model.ny) * 1000  # mm
        X_phys, Y_phys = np.meshgrid(x_phys, y_phys)
        
        # Display temperature field using the full domain for both rendering and extent
        im1 = ax1.imshow(T_init, extent=full_extent, origin='lower', 
                        cmap=cmap_temp, interpolation='bilinear', aspect='auto')
        cs1 = ax1.contour(X_phys, Y_phys, T_init, levels=[melt_temp], 
                        colors='cyan', linewidths=2)
        
        # Plot laser paths on temperature field with increased visibility
        ax1.plot(init_saw_path_mm[:, 0], init_saw_path_mm[:, 1], 'w-', linewidth=2.5, alpha=0.9)
        ax1.plot(init_swirl_path_mm[:, 0], init_swirl_path_mm[:, 1], 'g-', linewidth=2.5, alpha=0.9)
        
        # Add markers for current positions with increased visibility
        ax1.plot(init_saw_path_mm[-1, 0], init_saw_path_mm[-1, 1], 'wo', markersize=10, markeredgecolor='k', markeredgewidth=1.5)
        ax1.plot(init_swirl_path_mm[-1, 0], init_swirl_path_mm[-1, 1], 'go', markersize=10, markeredgecolor='k', markeredgewidth=1.5)
        
        # Add metrics text
        melt_count_init = np.sum(melt_mask_init)
        max_temp_init = np.max(T_init)
        
        if melt_count_init > 0:
            T_melt_init = T_init[melt_mask_init]
            cv_init = np.std(T_melt_init) / np.mean(T_melt_init) * 100
            metrics_text = f"Max Temp: {max_temp_init:.0f}°C\nMelt Pool: {melt_count_init} pts\nCV: {cv_init:.1f}%"
        else:
            metrics_text = f"Max Temp: {max_temp_init:.0f}°C\nNo melt pool"
        
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
        cs2 = ax2.contour(X_phys, Y_phys, T_opt, levels=[melt_temp], 
                        colors='cyan', linewidths=2)
        
        # Plot laser paths on temperature field with increased visibility
        ax2.plot(opt_saw_path_mm[:, 0], opt_saw_path_mm[:, 1], 'w-', linewidth=2.5, alpha=0.9)
        ax2.plot(opt_swirl_path_mm[:, 0], opt_swirl_path_mm[:, 1], 'g-', linewidth=2.5, alpha=0.9)
        
        # Add markers for current positions with increased visibility
        ax2.plot(opt_saw_path_mm[-1, 0], opt_saw_path_mm[-1, 1], 'wo', markersize=10, markeredgecolor='k', markeredgewidth=1.5)
        ax2.plot(opt_swirl_path_mm[-1, 0], opt_swirl_path_mm[-1, 1], 'go', markersize=10, markeredgecolor='k', markeredgewidth=1.5)
        
        # Add metrics text
        melt_count_opt = np.sum(melt_mask_opt)
        max_temp_opt = np.max(T_opt)
        
        if melt_count_opt > 0:
            T_melt_opt = T_opt[melt_mask_opt]
            cv_opt = np.std(T_melt_opt) / np.mean(T_melt_opt) * 100
            metrics_text = f"Max Temp: {max_temp_opt:.0f}°C\nMelt Pool: {melt_count_opt} pts\nCV: {cv_opt:.1f}%"
        else:
            metrics_text = f"Max Temp: {max_temp_opt:.0f}°C\nNo melt pool"
        
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
        ax4.contour(X_phys, Y_phys, T_init, levels=[melt_temp], 
                colors='cyan', linewidths=2)
        
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
        ax5.contour(X_phys, Y_phys, T_opt, levels=[melt_temp], 
                colors='cyan', linewidths=2)
        
        # Plot paths on gradient field
        ax5.plot(opt_saw_path_mm[:, 0], opt_saw_path_mm[:, 1], 'w-', linewidth=1.5, alpha=0.7)
        ax5.plot(opt_swirl_path_mm[:, 0], opt_swirl_path_mm[:, 1], 'g-', linewidth=1.5, alpha=0.7)
        
        ax5.set_title(f'Optimized Gradient (Max: {np.max(grad_opt)/1000:.0f}°C/mm)')
        ax5.set_xlabel('X (mm)')
        ax5.set_ylabel('Y (mm)')
        ax5.set_xlim(x_min, x_max)
        ax5.set_ylim(y_min, y_max)
        fig.colorbar(im5, ax=ax5, label='Gradient (°C/mm)')
        
        # Calculate metrics
        T_melt_init = T_init[melt_mask_init] if np.any(melt_mask_init) else np.array([0])
        T_melt_opt = T_opt[melt_mask_opt] if np.any(melt_mask_opt) else np.array([0])
        
        # Plot temperature distribution histogram
        ax6.hist(T_melt_init, bins=30, alpha=0.5, color='blue', label='Initial')
        ax6.hist(T_melt_opt, bins=30, alpha=0.5, color='red', label='Optimized')
        ax6.set_xlabel('Temperature (°C)')
        ax6.set_ylabel('Count')
        ax6.set_title('Melt Pool Temperature Distribution')
        ax6.legend()
        
        # Add metrics table
        metrics = {
            'Melt Pool Size (points)': [np.sum(melt_mask_init), np.sum(melt_mask_opt)],
            'Mean Temperature (°C)': [np.mean(T_melt_init), np.mean(T_melt_opt)],
            'Temp. Std Dev (°C)': [np.std(T_melt_init), np.std(T_melt_opt)],
            'CV (%)': [np.std(T_melt_init)/np.mean(T_melt_init)*100 if np.mean(T_melt_init) > 0 else 0,
                    np.std(T_melt_opt)/np.mean(T_melt_opt)*100 if np.mean(T_melt_opt) > 0 else 0],
            'Max Gradient (°C/mm)': [np.max(grad_init)/1000, np.max(grad_opt)/1000],
        }
        
        # Create table below the histogram instead of in the middle
        table_ax = fig.add_axes([0.67, -0.15, 0.25, 0.15])
        table_ax.axis('tight')
        table_ax.axis('off')
        table_data = [[metric] + [f"{val:.1f}" if isinstance(val, float) else f"{val}" 
                            for val in values] for metric, values in metrics.items()]
        table = table_ax.table(cellText=table_data,
                            colLabels=['Metric', 'Initial', 'Optimized'],
                            loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Add a legend for the entire figure (moved to left side to avoid overlapping with table)
        legend_ax = fig.add_axes([0.05, 0.02, 0.4, 0.02])
        legend_ax.axis('off')
        legend_elements = [
            plt.Line2D([0], [0], color='white', marker='o', markersize=8, markerfacecolor='white',
                    markeredgecolor='black', label='Sawtooth Laser'),
            plt.Line2D([0], [0], color='green', marker='o', markersize=8, markerfacecolor='green',
                    markeredgecolor='black', label='Swirl Laser'),
            plt.Line2D([0], [0], color='cyan', lw=2, label='Melt Pool Boundary')
        ]
        legend_ax.legend(handles=legend_elements, loc='center', ncol=3, frameon=False)
        
        plt.suptitle('LPBF Dual Laser Scan Optimization: Initial vs. Optimized', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.subplots_adjust(bottom=0.1, wspace=0.25, hspace=0.3)  # Adjust spacing
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
        
        # Generate trajectory points for longer simulation time
        # Use either the model's time steps or a custom duration (whichever is longer)
        model_time = model.nt * model.dt
        max_time = max(model_time, simulation_duration)
        num_steps = int(max_time / model.dt)
        times = np.linspace(0, max_time, num_steps)
        
        # Generate all paths
        initial_path1 = np.array([model.sawtooth_trajectory(t, initial_laser_params[0])[:2] 
                                for t in times])
        initial_path2 = np.array([model.swirl_trajectory(t, initial_laser_params[1])[:2] 
                                for t in times])
        
        optimized_path1 = np.array([model.sawtooth_trajectory(t, optimized_laser_params[0])[:2] 
                                for t in times])
        optimized_path2 = np.array([model.swirl_trajectory(t, optimized_laser_params[1])[:2] 
                                for t in times])
        
        # Convert to mm for plotting
        initial_path1_mm = initial_path1 * 1000
        initial_path2_mm = initial_path2 * 1000
        optimized_path1_mm = optimized_path1 * 1000
        optimized_path2_mm = optimized_path2 * 1000
        
        # Setup figure and subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        fig.suptitle('Dual Laser Heat Transfer Optimization', fontsize=16)
        
        # Set up temperature bounds
        T_min = model.material['T0']
        melt_temp = model.material.get('T_melt', model.material['T0'] + 100)
        initial_T_max = melt_temp * 1.5  # Initial max temperature
        
        # Configure axes with proper mm scale
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.grid(True, linestyle='--', alpha=0.3)
        
        axes[0].set_title('Initial Parameters', fontsize=14)
        axes[1].set_title('Optimized Parameters', fontsize=14)
        
        # Initialize temperature field images
        init_field = np.ones((model.ny, model.nx)) * model.material['T0']
        
        # Use the full domain extent for rendering the temperature field
        image_initial = axes[0].imshow(init_field, extent=full_extent, origin='lower', 
                                    cmap=cmap_temp, vmin=T_min, vmax=initial_T_max,
                                    interpolation='bilinear', aspect='auto')
        
        image_optimized = axes[1].imshow(init_field, extent=full_extent, origin='lower', 
                                    cmap=cmap_temp, vmin=T_min, vmax=initial_T_max,
                                    interpolation='bilinear', aspect='auto')
        
        # Add colorbars
        cbar_initial = fig.colorbar(image_initial, ax=axes[0])
        cbar_initial.set_label('Temperature (°C)')
        cbar_optimized = fig.colorbar(image_optimized, ax=axes[1])
        cbar_optimized.set_label('Temperature (°C)')
        
        # Physical coordinates for contours
        x_phys = np.linspace(0, model.Lx * 1000, model.nx)
        y_phys = np.linspace(0, model.Ly * 1000, model.ny)
        X_phys, Y_phys = np.meshgrid(x_phys, y_phys)
        
        # Initialize melt pool contours
        contour_initial = axes[0].contour(X_phys, Y_phys, init_field, 
                                        levels=[melt_temp], colors='cyan', linewidths=2)
        
        contour_optimized = axes[1].contour(X_phys, Y_phys, init_field, 
                                        levels=[melt_temp], colors='cyan', linewidths=2)
        
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
                    markeredgecolor='black', label='Swirl Laser'),
            plt.Line2D([0], [0], color='cyan', lw=2, label='Melt Pool Boundary')
        ]
        fig.legend(handles=legend_elements, loc='lower center', 
                bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=12)
        
        # Initialize temperature fields
        T_initial = init_field.copy()
        T_optimized = init_field.copy()
        
        # Store the variable for updating color limits
        T_max = initial_T_max
        
        # Function to update contours
        def update_contour(contour_obj, ax, T, melt_temp):
            # Remove old contour collections
            for coll in contour_obj.collections:
                coll.remove()
                
            # Create new contour
            return ax.contour(X_phys, Y_phys, T, 
                            levels=[melt_temp], colors='cyan', linewidths=2)
        
        def update(frame):
            """Update function with fixed heat source application."""
            nonlocal T_initial, T_optimized, contour_initial, contour_optimized, T_max
            
            # Get actual time value for this frame
            t = times[frame]
            
            # Update the time title
            time_title.set_text(f'Simulation Time: {t:.3f} s (Step {frame+1}/{len(frames_to_use)})')
            
            # Get current laser source positions
            x_src_init1, y_src_init1 = initial_path1[frame]
            x_src_init2, y_src_init2 = initial_path2[frame]
            x_src_opt1, y_src_opt1 = optimized_path1[frame]
            x_src_opt2, y_src_opt2 = optimized_path2[frame]
            
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
            
            # Compute tangent vectors at time t
            _, _, tx_init1, ty_init1 = model.sawtooth_trajectory(t, initial_laser_params[0])
            _, _, tx_init2, ty_init2 = model.swirl_trajectory(t, initial_laser_params[1])
            _, _, tx_opt1, ty_opt1 = model.sawtooth_trajectory(t, optimized_laser_params[0])
            _, _, tx_opt2, ty_opt2 = model.swirl_trajectory(t, optimized_laser_params[1])
            
            # First, handle the heat diffusion part (this part is correct)
            # 1. Initial parameters diffusion
            lap_init = np.zeros_like(T_initial)
            lap_init[1:-1, 1:-1] = ((T_initial[1:-1, 2:] - 2*T_initial[1:-1, 1:-1] + T_initial[1:-1, :-2]) / model.dx**2 +
                                (T_initial[2:, 1:-1] - 2*T_initial[1:-1, 1:-1] + T_initial[:-2, 1:-1]) / model.dy**2)
            
            # Get thermal diffusivity
            alpha = model.material.get('alpha', model.material['k'] / model.heat_capacity)
            
            # Apply heat diffusion
            T_init_new = T_initial.copy()
            T_init_new[1:-1, 1:-1] += model.dt * alpha * lap_init[1:-1, 1:-1]
            
            # 2. Optimized parameters diffusion
            lap_opt = np.zeros_like(T_optimized)
            lap_opt[1:-1, 1:-1] = ((T_optimized[1:-1, 2:] - 2*T_optimized[1:-1, 1:-1] + T_optimized[1:-1, :-2]) / model.dx**2 +
                                (T_optimized[2:, 1:-1] - 2*T_optimized[1:-1, 1:-1] + T_optimized[:-2, 1:-1]) / model.dy**2)
            
            T_opt_new = T_optimized.copy()
            T_opt_new[1:-1, 1:-1] += model.dt * alpha * lap_opt[1:-1, 1:-1]
            
            # Now, the critical part: apply heat sources using the model's own apply_heat_source function
            # This ensures we use the exact same heat application logic as the full simulation
            
            # Generate source positions for both initial and optimized
            init_positions = [(x_src_init1, y_src_init1), (x_src_init2, y_src_init2)]
            opt_positions = [(x_src_opt1, y_src_opt1), (x_src_opt2, y_src_opt2)]
            
            # For Gaussian sources, we don't need tangent vectors
            if use_gaussian:
                # Use the model's apply_heat_source function directly
                if hasattr(model, 'apply_heat_source'):
                    # Apply heat source to initial
                    T_init_new = model.apply_heat_source(T_init_new, init_positions, initial_heat_params)
                    
                    # Apply heat source to optimized
                    T_opt_new = model.apply_heat_source(T_opt_new, opt_positions, optimized_heat_params)
                else:
                    # Fallback if apply_heat_source is not available
                    #print("Warning: model.apply_heat_source not found, using direct heat application.")
                    # Direct application using _gaussian_source and heat equation
                    S_init1 = model._gaussian_source(x_src_init1, y_src_init1, initial_heat_params[0])
                    S_init2 = model._gaussian_source(x_src_init2, y_src_init2, initial_heat_params[1])
                    S_opt1 = model._gaussian_source(x_src_opt1, y_src_opt1, optimized_heat_params[0])
                    S_opt2 = model._gaussian_source(x_src_opt2, y_src_opt2, optimized_heat_params[1])
                    
                    # Convert to temperature change directly using the same formula as apply_heat_source
                    thickness = model.material.get('thickness', 0.001)  # meters
                    rho = model.material['rho']
                    cp = model.material['cp']
                    
                    # Apply initial heat sources
                    T_init_new += (S_init1 + S_init2) * (1 / (model.dx * model.dy * thickness * rho)) * (1 / cp)
                    
                    # Apply optimized heat sources  
                    T_opt_new += (S_opt1 + S_opt2) * (1 / (model.dx * model.dy * thickness * rho)) * (1 / cp)
            else:
                # For Goldak sources which need tangent vectors
                if hasattr(model, 'apply_heat_source'):
                    # Create source data with tangent vectors
                    init_source_data = [
                        (x_src_init1, y_src_init1, (tx_init1, ty_init1), initial_heat_params[0]),
                        (x_src_init2, y_src_init2, (tx_init2, ty_init2), initial_heat_params[1])
                    ]
                    opt_source_data = [
                        (x_src_opt1, y_src_opt1, (tx_opt1, ty_opt1), optimized_heat_params[0]),
                        (x_src_opt2, y_src_opt2, (tx_opt2, ty_opt2), optimized_heat_params[1])
                    ]
                    
                    # Apply heat directly using goldak_source and the same formula as apply_heat_source
                    for src_data in init_source_data:
                        x, y, tangent, params = src_data
                        source = model.goldak_source(x, y, tangent, params)
                        thickness = model.material.get('thickness', 0.001)
                        rho = model.material['rho']
                        cp = model.material['cp']
                        T_init_new += source * (1 / (model.dx * model.dy * thickness * rho)) * (1 / cp)
                        
                    for src_data in opt_source_data:
                        x, y, tangent, params = src_data
                        source = model.goldak_source(x, y, tangent, params)
                        thickness = model.material.get('thickness', 0.001)
                        rho = model.material['rho']
                        cp = model.material['cp']
                        T_opt_new += source * (1 / (model.dx * model.dy * thickness * rho)) * (1 / cp)
            
            # Apply boundary conditions (Neumann)
            # Initial
            T_init_new[0, :] = T_init_new[1, :]
            T_init_new[-1, :] = T_init_new[-2, :]
            T_init_new[:, 0] = T_init_new[:, 1]
            T_init_new[:, -1] = T_init_new[:, -2]
            
            # Optimized
            T_opt_new[0, :] = T_opt_new[1, :]
            T_opt_new[-1, :] = T_opt_new[-2, :]
            T_opt_new[:, 0] = T_opt_new[:, 1]
            T_opt_new[:, -1] = T_opt_new[:, -2]
            
            # Update temperature fields
            T_initial = T_init_new
            T_optimized = T_opt_new
            
            if frame % 20 == 0:  # Debug output every 20 frames
                print(f"Frame {frame}, Max temp: {np.max(T_initial):.2f}°C, {np.max(T_optimized):.2f}°C")
            
            # Identify melt pools
            melt_mask_init = T_initial > melt_temp
            melt_mask_opt = T_optimized > melt_temp
            
            # Calculate melt pool metrics
            melt_size_init = np.sum(melt_mask_init)
            melt_size_opt = np.sum(melt_mask_opt)
            
            max_temp_init = np.max(T_initial)
            max_temp_opt = np.max(T_optimized)
            
            # Enhanced melt pool metrics
            if melt_size_init > 0:
                T_melt_init = T_initial[melt_mask_init]
                temp_std_init = np.std(T_melt_init)
                cv_init = temp_std_init / np.mean(T_melt_init) * 100 if np.mean(T_melt_init) > 0 else 0
                init_text = (f'Max Temp: {max_temp_init:.0f}°C\n'
                        f'Melt Pool: {melt_size_init} pts\n'
                        f'CV: {cv_init:.1f}%')
            else:
                init_text = f'Max Temp: {max_temp_init:.0f}°C\nNo Melt Pool'
            
            if melt_size_opt > 0:
                T_melt_opt = T_optimized[melt_mask_opt]
                temp_std_opt = np.std(T_melt_opt)
                cv_opt = temp_std_opt / np.mean(T_melt_opt) * 100 if np.mean(T_melt_opt) > 0 else 0
                opt_text = (f'Max Temp: {max_temp_opt:.0f}°C\n'
                        f'Melt Pool: {melt_size_opt} pts\n'
                        f'CV: {cv_opt:.1f}%')
            else:
                opt_text = f'Max Temp: {max_temp_opt:.0f}°C\nNo Melt Pool'
            
            # Update text
            temp_text_initial.set_text(init_text)
            temp_text_optimized.set_text(opt_text)
            
            # Update temperature images
            image_initial.set_array(T_initial)
            image_optimized.set_array(T_optimized)
            
            # Update melt pool contours
            contour_initial = update_contour(contour_initial, axes[0], T_initial, melt_temp)
            contour_optimized = update_contour(contour_optimized, axes[1], T_optimized, melt_temp)
            
            # Adjust color scales if needed
            shared_max = max(np.max(T_initial), np.max(T_optimized))
            if shared_max > T_max:
                # Cap at reasonable value
                new_max = min(shared_max, melt_temp * 3)
                image_initial.set_clim(vmax=new_max)
                image_optimized.set_clim(vmax=new_max)
                T_max = new_max
            
            return [image_initial, image_optimized,
                    source_marker_initial1, source_marker_initial2,
                    source_marker_optimized1, source_marker_optimized2,
                    path_line_initial1, path_line_initial2,
                    path_line_optimized1, path_line_optimized2,
                    temp_text_initial, temp_text_optimized]
        
        # Create animation with fewer frames for smoother performance
        frame_count = len(times)
        
        # Cap at reasonable number of frames
        if frame_count > max_frames:
            frame_skip = max(1, frame_count // max_frames)
            frames_to_use = list(range(0, frame_count, frame_skip))
        else:
            frames_to_use = list(range(frame_count))
        
        ani = animation.FuncAnimation(
            fig, update, frames=frames_to_use,
            interval=1000/fps, blit=True
        )
        
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
        'T0': 20.0,                # Initial temperature (°C)
        'alpha': 5e-6,             # Thermal diffusivity (m²/s)
        'rho': 7800.0,             # Density (kg/m³)
        'cp': 500.0,               # Specific heat capacity (J/kg·K)
        'thickness': 0.01,       # Thickness (m) (1mm?)
        'T_melt': 1500.0,          # Melting temperature (°C)
        'k': 20.0,                 # Thermal conductivity (W/m·K)
        'absorptivity': 1,       # Laser absorptivity
    }
    
    # 2. Domain settings
    domain_size = (0.005, 0.005)     # 10mm x 10mm domain
    grid_size = (51, 51)           # Use a coarser grid for faster optimization
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
        'sawtooth_A': 0.001,        # 1mm amplitude 
        'sawtooth_y0': 0.0025,       # Center position (3mm)
        'sawtooth_period': 0.02,    # 20ms period
        
        # Swirl/Spiral path parameters
        'swirl_v': 0.05,            # 50 mm/s scan speed
        'swirl_A': 0.001,           # 1mm amplitude
        'swirl_y0': 0.0025,          # Center position (7mm)
        'swirl_fr': 10.0,           # 10 Hz frequency
        
        # Laser parameters for Gaussian model
        'Q': 300.0,                 # 300W laser power
        'r0': 5e-5,                 # 50μm beam radius
    }
    
    # 5. Define parameter bounds
    bounds = {
        # Velocity bounds (m/s)
        'sawtooth_v': (0.0001, 0.1),
        'swirl_v': (0.0001, 0.1),
        
        # Amplitude bounds (m)
        'sawtooth_A': (0.00005, 0.002),
        'swirl_A': (0.00005, 0.002),
        
        # Position bounds (m)
        'sawtooth_y0': (0.001, 0.003),
        'swirl_y0': (0.001, 0.003),
        
        # Time parameter bounds
        'sawtooth_period': (0.01, 0.05),
        'swirl_fr': (5.0, 20.0),
        
        # Laser parameter bounds
        'Q': (50.0, 500.0),
        'r0': (3e-5, 8e-5),
    }
    
    # 6. Create trajectory optimizer
    optimizer = TrajectoryOptimizer(
        model=model,
        initial_params=initial_params,
        bounds=bounds,
        x_range=(0.0, 0.008)  # 8mm scan length
    )
    
    # 7. Run optimization
    print("\n=============================================")
    print("Starting Laser Path Optimization")
    print("=============================================\n")
    
    # Choose method and objective
    method = 'Powell'  # Options: 'SLSQP', 'L-BFGS-B', 'Nelder-Mead', 'Powell', 'COBYLA'
    objective = 'standard'  # Options: 'standard', 'thermal_uniformity', 'max_gradient', etc.
    
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
        
        improvement = (initial_obj - final_obj) / initial_obj * 100
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
                fps=30
            )
            
        # Optional: Perform sensitivity analysis
        run_sensitivity = input("Perform sensitivity analysis of optimized solution? (y/n): ").lower()
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
    
    return model, optimizer, result, optimized_params

# Run the optimization if script is executed directly
if __name__ == "__main__":
    model, optimizer, result, optimized_params = run_optimization()