import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as anp 
import cvxpy as cp
import os
from matplotlib import animation
from scipy.optimize import minimize
from autograd import grad, jacobian, hessian
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage import measure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation

# ===============================
# 1. Heat Transfer Model Class
# ===============================

class HeatTransferModel:
    def __init__(self, domain_size=(0.01, 0.01), grid_size=(101, 101), dt=1e-5, material_params=None):
        # Set up simulation domain and grid
        self.Lx, self.Ly = domain_size
        self.nx, self.ny = grid_size
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)
        self.x = anp.linspace(0, self.Lx, self.nx)
        self.y = anp.linspace(0, self.Ly, self.ny)
        self.X, self.Y = anp.meshgrid(self.x, self.y)
        self.dt = dt
        self.nt = None  # Number of time steps, set in simulate()
        self.debug_counter = 0
        
        # Set material properties
        default_params = {
            'T0': 21.0,           # Initial temperature (°C)
            'alpha': 5e-6,        # Thermal diffusivity (m²/s)
            'rho': 7800.0,        # Density (kg/m³)
            'cp': 500.0,          # Specific heat capacity (J/(kg·K))
            'k': 20.0,            # Thermal conductivity (W/(m·K))
            'T_melt': 1500.0,     # Melting temperature (°C)
            'thickness': 0.00021  # Plate thickness (m)
        }
        self.material = default_params if material_params is None else material_params
        self.T = self.material['T0'] * anp.ones((self.ny, self.nx))
        self.heat_capacity = self.material['rho'] * self.material['cp']
        self.thickness = self.material['thickness']

    def reset(self):
        """Reset temperature field to initial value."""
        self.T = self.material['T0'] * anp.ones((self.ny, self.nx))
    
    def _laplacian(self, T):
        """Compute Laplacian of temperature field using finite differences."""
        lap_inner = ((T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / (self.dx**2) +
                     (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / (self.dy**2))
        lap = anp.pad(lap_inner, pad_width=((1,1),(1,1)), mode='constant', constant_values=0)
        return lap

    def sawtooth_trajectory(self, t, params):
        """Calculate sawtooth laser position and direction at time t."""
        v = params['v']
        A = params['A']
        y0 = params['y0']
        period = params['period']
        noise_sigma = params.get('noise_sigma', 0.0)
        max_noise = 0.00005
        noise_sigma = anp.clip(noise_sigma, 0.0, max_noise)
        omega = 2 * anp.pi / period
        avg_speed_factor = anp.sqrt(1 + (2 * A * omega / anp.pi) ** 2)
        t_scaled = t * avg_speed_factor
        raw_x = v * t_scaled
        k = 1000
        x = raw_x - (raw_x - (self.Lx - 0.0005)) * (1 / (1 + anp.exp(-k * (raw_x - (self.Lx - 0.0005)))))
        y = y0 + A * (2/anp.pi) * anp.arcsin(anp.sin(omega * t_scaled))
        if noise_sigma > 0:
            x = x + noise_sigma * np.random.randn()
            y = y + noise_sigma * np.random.randn()
        tx = v
        ty = (2 * A * omega / anp.pi) * anp.cos(omega * t_scaled)
        norm = anp.sqrt(tx**2 + ty**2)
        return x, y, tx / norm, ty / norm

    def swirl_trajectory(self, t, params):
        """Calculate swirl laser position and direction at time t."""
        v = params['v']
        A = params['A']
        y0 = params['y0']
        fr = params['fr']
        om = 2 * anp.pi * fr
        noise_sigma = params.get('noise_sigma', 0.0)
        max_noise = 0.00005
        noise_sigma = anp.clip(noise_sigma, 0.0, max_noise)
        avg_speed_factor = anp.sqrt(1 + (A * om / v) ** 2)
        t_scaled = t * avg_speed_factor
        raw_x = v * t_scaled + A * anp.sin(om * t_scaled)
        k = 1000
        x = raw_x - (raw_x - (self.Lx - 0.0005)) * (1 / (1 + anp.exp(-k * (raw_x - (self.Lx - 0.0005)))))
        y = y0 + A * anp.cos(om * t_scaled)
        if noise_sigma > 0:
            x = x + noise_sigma * np.random.randn()
            y = y + noise_sigma * np.random.randn()
        tx = v + A * om * anp.cos(om * t_scaled)
        ty = -A * om * anp.sin(om * t_scaled)
        norm = anp.sqrt(tx**2 + ty**2)
        return x, y, tx / norm, ty / norm

    def compute_temperature_gradients(self, T=None):
        """Compute spatial gradients and gradient magnitude of temperature field."""
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
        """Calculate Gaussian heat source centered at (x_src, y_src)."""
        power = heat_params.get('Q', 200.0)
        if 'r0' in heat_params:
            sigma_x = heat_params['r0'] / 2.0
            sigma_y = heat_params['r0'] / 2.0
        else:
            sigma_x = heat_params.get('sigma_x', 1.5e-3)
            sigma_y = heat_params.get('sigma_y', 1.5e-3)
        absorbed_power = power * self.material['absorptivity']
        G = np.exp(-(((self.X - x_src)**2) / (2 * sigma_x**2) +
                     ((self.Y - y_src)**2) / (2 * sigma_y**2)))
        return G * absorbed_power

    def simulate(self, parameters, start_x=0.0, end_x=None, use_gaussian=True, verbose=False):
        """
        Run heat transfer simulation for given laser and heat source parameters.
        Returns final temperature field.
        """
        laser_params, heat_params = parameters
        sawtooth_params, swirl_params = laser_params
        
        self.reset()
        if end_x is None:
            end_x = self.Lx

        fixed_nt = 3000  # Number of time steps
        self.nt = fixed_nt

        # Use thermal diffusivity from material properties
        if 'alpha' not in self.material and 'k' in self.material:
            alpha = self.material['k'] / (self.material['rho'] * self.material['cp'])
        else:
            alpha = self.material['alpha']
        
        T = self.T.copy()
        for n in range(self.nt):
            t = n * self.dt

            # Get laser positions and directions
            x1, y1, tx1, ty1 = self.sawtooth_trajectory(t, sawtooth_params)
            x2, y2, tx2, ty2 = self.swirl_trajectory(t, swirl_params)
            
            # Calculate heat source from both lasers
            S1 = self._gaussian_source(x1, y1, heat_params[0])
            S2 = self._gaussian_source(x2, y2, heat_params[1])
            S_total = S1 + S2
            
            # Update temperature field using explicit finite difference
            lap = self._laplacian(T)
            T_diff = T + self.dt * alpha * lap 
            source_increment = self.dt * S_total / (self.dx * self.dy * self.thickness * self.material['rho'] * self.material['cp'])
            T_new = T_diff + source_increment
            
            # Apply fixed boundary conditions
            T_new[0, :] = self.material['T0']
            T_new[-1, :] = self.material['T0']
            T_new[:, 0] = self.material['T0']
            T_new[:, -1] = self.material['T0']
            
            T = T_new
              
        
        self.T = T
        return self.T

    def compute_temperature_gradients(self, T=None):
        """Compute spatial gradients and gradient magnitude of temperature field."""
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

# ===============================
# 2. Trajectory Optimizer Class
# ===============================
    
class TrajectoryOptimizer:
    def __init__(self, model, initial_params=None, bounds=None, x_range=(0.0, 0.01)):
        # Store model and optimization settings
        self.model = model
        self.initial_params = initial_params
        self.bounds = bounds
        self.x_range = x_range

        # Select parameters to optimize (exclude fixed y0)
        self.param_names = [k for k in initial_params.keys() if k not in ['sawtooth_y0', 'swirl_y0']]
        # Always include noise_sigma if present in bounds
        if self.bounds and 'noise_sigma' in self.bounds and 'noise_sigma' not in self.param_names:
            self.param_names.append('noise_sigma')

    def parameters_to_array(self, params_dict):
        """Convert parameter dictionary to flat array for optimization."""
        return anp.array([params_dict[name] for name in self.param_names])
    
    def array_to_parameters(self, params_array):
        """Convert flat array to parameter dictionary."""
        return {name: params_array[i] for i, name in enumerate(self.param_names)}
    
    def unpack_parameters(self, params_array):
        """
        Convert flat parameter array to structured laser and heat source parameters.
        Returns (laser_params, heat_params).
        """
        params_dict = self.array_to_parameters(params_array)
        noise_sigma = params_dict.get('noise_sigma', self.initial_params.get('noise_sigma', 0.0))
        sawtooth_params = {
            'v': params_dict['sawtooth_v'],
            'A': params_dict['sawtooth_A'],
            'y0': self.initial_params['sawtooth_y0'],
            'period': params_dict['sawtooth_period'],
            'noise_sigma': noise_sigma
        }
        swirl_params = {
            'v': params_dict['swirl_v'],
            'A': params_dict['swirl_A'],
            'y0': self.initial_params['swirl_y0'],
            'fr': params_dict['swirl_fr'],
            'noise_sigma': noise_sigma
        }
        heat_params = (
            {'Q': params_dict['sawtooth_Q'], 'r0': params_dict['sawtooth_r0']},
            {'Q': params_dict['swirl_Q'], 'r0': params_dict['swirl_r0']}
        )
        laser_params = (sawtooth_params, swirl_params)
        return laser_params, heat_params
    
    def objective_function(self, params_array):
        """
        Objective: minimize sum of squared temperature gradients (smoothness).
        """
        # Penalize extreme parameter values
        for i, name in enumerate(self.param_names):
            if anp.abs(params_array[i]) > 1e6:
                print(f"Warning: Parameter {name} has extreme value: {params_array[i]}")
                return 1e10
        laser_params, heat_params = self.unpack_parameters(params_array)
        use_gaussian = 'r0' in heat_params[0]
        T = self.model.simulate((laser_params, heat_params), 
                                start_x=self.x_range[0], 
                                end_x=self.x_range[1],
                                use_gaussian=use_gaussian)
        _, _, grad_mag = self.model.compute_temperature_gradients(T)
        cost = anp.sum(grad_mag**2) / (self.model.nx * self.model.ny)
        return cost

    def objective_max_gradient(self, params_array):
        """
        Objective: minimize maximum temperature gradient (reduce hot spots).
        """
        laser_params, heat_params = self.unpack_parameters(params_array)
        use_gaussian = 'r0' in heat_params[0]
        T = self.model.simulate((laser_params, heat_params), 
                                start_x=self.x_range[0], 
                                end_x=self.x_range[1],
                                use_gaussian=use_gaussian)
        _, _, grad_mag = self.model.compute_temperature_gradients(T)
        cost = anp.max(grad_mag)
        return cost

    def objective_path_focused(self, params_array):
        """
        Objective: minimize gradients along laser paths only.
        """
        laser_params, heat_params = self.unpack_parameters(params_array)
        sawtooth_params, swirl_params = laser_params
        use_gaussian = 'r0' in heat_params[0]
        T = self.model.simulate((laser_params, heat_params), 
                                start_x=self.x_range[0], 
                                end_x=self.x_range[1],
                                use_gaussian=use_gaussian)
        _, _, grad_mag = self.model.compute_temperature_gradients(T)
        times = anp.linspace(0, self.model.nt * self.model.dt, 50)
        path_points = []
        for t in times:
            x1, y1, _, _ = self.model.sawtooth_trajectory(t, sawtooth_params)
            x2, y2, _, _ = self.model.swirl_trajectory(t, swirl_params)
            path_points.append((x1, y1))
            path_points.append((x2, y2))
        path_gradients = []
        dx, dy = self.model.dx, self.model.dy
        for x, y in path_points:
            i = int(anp.clip(y / dy, 0, self.model.ny - 1))
            j = int(anp.clip(x / dx, 0, self.model.nx - 1))
            path_gradients.append(grad_mag[i, j])
        cost = anp.mean(anp.array(path_gradients))
        return cost

    def objective_thermal_uniformity(self, params_array):
        """
        Objective: promote uniform melt pool temperature and minimize gradients.
        """
        laser_params, heat_params = self.unpack_parameters(params_array)
        use_gaussian = 'r0' in heat_params[0]
        T = self.model.simulate((laser_params, heat_params), 
                                start_x=self.x_range[0], 
                                end_x=self.x_range[1],
                                use_gaussian=use_gaussian)
        _, _, grad_mag = self.model.compute_temperature_gradients(T)
        grad_cost = anp.mean(grad_mag**2)
        melt_temp = self.model.material['T_melt']
        melt_mask = T > melt_temp
        if anp.sum(melt_mask) > 0:
            T_melt = T[melt_mask]
            T_variance = anp.var(T_melt)
        else:
            T_variance = 0.0
        cost = 0.7 * grad_cost + 0.3 * T_variance
        return cost

    def objective_max_temp_difference(self, params_array):
        """
        Objective: minimize maximum temperature difference in melt pool.
        """
        laser_params, heat_params = self.unpack_parameters(params_array)
        use_gaussian = 'r0' in heat_params[0]
        T = self.model.simulate((laser_params, heat_params), 
                            start_x=self.x_range[0], 
                            end_x=self.x_range[1],
                            use_gaussian=use_gaussian)
        melt_temp = self.model.material['T_melt']
        melt_mask = T > melt_temp
        if anp.sum(melt_mask) > 0:
            T_melt = T[melt_mask]
            max_temp_diff = anp.max(T_melt) - anp.min(T_melt)
        else:
            max_temp_diff = 1000.0  
        return max_temp_diff
    
    def get_inequality_constraints(self):
        """
        Create list of inequality constraint functions for optimizer.
        Each constraint must be g(x) <= 0.
        """
        inequality_constraints = []
        # Example constraints (distance, path bounds, energy density, parameter bounds)
        def min_distance_constraint(x):
            laser_params, _ = self.unpack_parameters(x)
            min_distance = self._calculate_min_laser_distance(laser_params)
            min_allowed_distance = 0.0005
            return min_allowed_distance - min_distance
        def sawtooth_max_y_constraint(x):
            params_dict = self.array_to_parameters(x)
            sawtooth_y_max = params_dict['sawtooth_y0'] + params_dict['sawtooth_A']
            return sawtooth_y_max - self.model.Ly
        def sawtooth_min_y_constraint(x):
            params_dict = self.array_to_parameters(x)
            sawtooth_y_min = params_dict['sawtooth_y0'] - params_dict['sawtooth_A']
            return -sawtooth_y_min
        def swirl_max_y_constraint(x):
            params_dict = self.array_to_parameters(x)
            swirl_y_max = params_dict['swirl_y0'] + params_dict['swirl_A']
            return swirl_y_max - self.model.Ly
        def swirl_min_y_constraint(x):
            params_dict = self.array_to_parameters(x)
            swirl_y_min = params_dict['swirl_y0'] - params_dict['swirl_A']
            return -swirl_y_min
        # Energy density constraints (example, not enforced by default)
        def max_energy_sawtooth_constraint(x):
            params_dict = self.array_to_parameters(x)
            Q = params_dict['Q']
            energy_density = Q / (params_dict['sawtooth_v'] * 0.0002)
            max_energy_density = 100.0
            return energy_density - max_energy_density
        def min_energy_sawtooth_constraint(x):
            params_dict = self.array_to_parameters(x)
            Q = params_dict['Q']
            energy_density = Q / (params_dict['sawtooth_v'] * 0.0002)
            min_energy_density = 10.0
            return min_energy_density - energy_density
        def max_energy_swirl_constraint(x):
            params_dict = self.array_to_parameters(x)
            Q = params_dict['Q']
            energy_density = Q / (params_dict['swirl_v'] * 0.0002)
            max_energy_density = 100.0
            return energy_density - max_energy_density
        def min_energy_swirl_constraint(x):
            params_dict = self.array_to_parameters(x)
            Q = params_dict['Q']
            energy_density = Q / (params_dict['swirl_v'] * 0.0002)
            min_energy_density = 10.0
            return min_energy_density - energy_density
        # Add parameter bounds as constraints
        bound_constraints = []
        for i, name in enumerate(self.param_names):
            if name in self.bounds:
                lb, ub = self.bounds[name]
                def make_lb_constraint(idx, bound):
                    return lambda x: bound - x[idx]
                def make_ub_constraint(idx, bound):
                    return lambda x: x[idx] - bound
                bound_constraints.append(make_lb_constraint(i, lb))
                bound_constraints.append(make_ub_constraint(i, ub))
        # Add selected constraints (commented out by default)
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
        inequality_constraints.extend(bound_constraints)
        return inequality_constraints
    
    def constraint_functions(self, params_array):
        """
        Evaluate all constraints for given parameter array.
        Returns array of constraint values (should be <= 0).
        """
        params_dict = self.array_to_parameters(params_array)
        laser_params, heat_params = self.unpack_parameters(params_array)
        sawtooth_params, swirl_params = laser_params
        constraints = []
        min_distance = self._calculate_min_laser_distance(laser_params)
        min_allowed_distance = 0.0005
        constraints.append(min_allowed_distance - min_distance)
        sawtooth_y_max = params_dict['sawtooth_y0'] + params_dict['sawtooth_A']
        sawtooth_y_min = params_dict['sawtooth_y0'] - params_dict['sawtooth_A']
        constraints.append(sawtooth_y_max - self.model.Ly)
        constraints.append(-sawtooth_y_min)
        swirl_y_max = params_dict['swirl_y0'] + params_dict['swirl_A']
        swirl_y_min = params_dict['swirl_y0'] - params_dict['swirl_A']
        constraints.append(swirl_y_max - self.model.Ly)
        constraints.append(-swirl_y_min)
        Q = params_dict['Q']
        energy_density_sawtooth = Q / (params_dict['sawtooth_v'] * 0.0002)
        energy_density_swirl = Q / (params_dict['swirl_v'] * 0.0002)
        max_energy_density = 100.0
        min_energy_density = 10.0
        #constraints.append(energy_density_sawtooth - max_energy_density)
        #constraints.append(min_energy_density - energy_density_sawtooth)
        #constraints.append(energy_density_swirl - max_energy_density)
        #constraints.append(min_energy_density - energy_density_swirl)
        return anp.array(constraints)

    def _calculate_min_laser_distance(self, laser_params):
        """
        Calculate minimum distance between lasers over simulation time.
        """
        sawtooth_params, swirl_params = laser_params
        slowest_v = anp.minimum(sawtooth_params['v'], swirl_params['v'])
        total_time = (self.x_range[1] - self.x_range[0]) / slowest_v
        n_samples = 500
        t_samples = anp.linspace(0, total_time, n_samples)
        distances = anp.zeros(n_samples)
        for i, t in enumerate(t_samples):
            x1, y1, _, _ = self.model.sawtooth_trajectory(t, sawtooth_params)
            x2, y2, _, _ = self.model.swirl_trajectory(t, swirl_params)
            distances[i] = anp.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        min_dist = anp.min(distances)
        return min_dist
    
    def optimize_with_scipy(self, objective_type='standard', method='SLSQP', max_iterations=100):
        """
        Run optimization using scipy's minimize with selected objective and constraints.
        """
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
        initial_params_array = self.parameters_to_array(self.initial_params)
        constraints = None
        if method.upper() in ['SLSQP', 'COBYLA']:
            constraints = [{
                'type': 'ineq',
                'fun': lambda x: self.constraint_functions(x)
            }]
        print(f"Starting optimization with {objective_type} objective using scipy method: {method}")
        bounds_list = [(self.bounds[name][0], self.bounds[name][1]) for name in self.param_names]
        result = minimize(
            objective_func,
            initial_params_array,
            method=method,
            constraints=constraints,
            bounds=bounds_list,
            options={
                'maxiter': max_iterations, 
                'disp': True,
                'xtol': 1e-10,
                'ftol': 1e-10
            }
        )
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
        """Convenience wrapper for optimize_with_scipy."""
        return self.optimize_with_scipy(
            objective_type=objective_type, 
            method=method, 
            max_iterations=max_iterations
        )
    
    def perform_sensitivity_analysis(self, best_params, objective_type='standard'):
        """
        Perform sensitivity analysis for each parameter around optimized solution.
        """
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
        # Store reference to heat transfer model
        self.model = model

    def generate_raster_scan(self, x_start, x_end, y_start, y_end, n_lines, speed):
        """Generate raster scan path covering the area at given speed."""
        x_vals = np.linspace(x_start, x_end, self.model.nx)
        y_vals = np.linspace(y_start, y_end, n_lines)
        path = []
        for i, y in enumerate(y_vals):
            xs = x_vals if i % 2 == 0 else x_vals[::-1]
            for x in xs:
                path.append((x, y))
        return np.array(path), speed

    def compare_simulations(self, initial_params, optimized_params, use_gaussian=True, y_crop=None, show_full_domain=True):
        """
        Compare temperature and gradient fields for raster scan vs. optimized scan.
        Overlays scan paths and melt pool contours.
        """
        # Raster scan setup
        x_start, x_end = 0, self.model.Lx
        y_start = 0.0010
        y_end = 0.0015
        hatch_spacing = 0.00005
        n_lines = int(np.floor((y_end - y_start) / hatch_spacing)) + 1
        raster_speed = 0.05
        raster_path, raster_v = self.generate_raster_scan(x_start, x_end, y_start, y_end, n_lines, raster_speed)

        # Simulate raster scan as single laser following raster_path
        heat_params = (
            {'Q': 200.0, 'r0': 0.00005},
            {'Q': 200.0, 'r0': 0.00005}
        )
        def raster_trajectory(t, params):
            idx = int(t * raster_v * len(raster_path) / (x_end - x_start))
            idx = np.clip(idx, 0, len(raster_path) - 1)
            x, y = raster_path[idx]
            return x, y, 1, 0

        # Override trajectory functions for raster scan simulation
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

        # Simulate optimized scan
        temp_optimizer = TrajectoryOptimizer(self.model, initial_params=initial_params, bounds=None)
        optimized_array = temp_optimizer.parameters_to_array(optimized_params)
        laser_opt, heat_opt = temp_optimizer.unpack_parameters(optimized_array)
        T_opt = self.model.simulate((laser_opt, heat_opt), use_gaussian=use_gaussian)

        # Generate scan paths for overlay
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

        # Compute temperature gradients
        _, _, grad_raster = self.model.compute_temperature_gradients(T_raster)
        _, _, grad_opt = self.model.compute_temperature_gradients(T_opt)

        # Visualization setup
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
        # Annotate gradient statistics
        max_grad_raster = np.max(grad_raster) / 1000
        mean_grad_raster = np.mean(grad_raster) / 1000
        ax1.text(
            0.02, 0.98,
            f"Max ∇T: {max_grad_raster:.2e} °C/mm\nMean ∇T: {mean_grad_raster:.2e} °C/mm",
            transform=ax1.transAxes, fontsize=10, color='white', verticalalignment='top',
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round')
        )

        im2 = ax2.imshow(T_opt, extent=extent, origin='lower', cmap=cmap_temp, aspect='auto')
        ax2.set_title('Optimized Scan Temperature')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        fig.colorbar(im2, ax=ax2, label='Temperature (°C)')
        ax2.plot(saw_x, saw_y, color='red', linewidth=1, label='Sawtooth Path')
        ax2.plot(swirl_x, swirl_y, color='black', linewidth=1, label='Swirl Path')
        ax2.legend();
        max_grad_opt = np.max(grad_opt) / 1000
        mean_grad_opt = np.mean(grad_opt) / 1000
        ax2.text(
            0.02, 0.98,
            f"Max ∇T: {max_grad_opt:.2e} °C/mm\nMean ∇T: {mean_grad_opt:.2e} °C/mm",
            transform=ax2.transAxes, fontsize=10, color='white', verticalalignment='top',
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))

        # Show gradient fields
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

        # Overlay melt pool contours
        T_melt = self.model.material['T_melt']
        melt_mask_raster = T_raster > T_melt
        melt_mask_opt = T_opt > T_melt
        ax1.contour(melt_mask_raster, levels=[0.5], colors='cyan', linewidths=2, extent=extent, origin='lower')
        ax2.contour(melt_mask_opt, levels=[0.5], colors='cyan', linewidths=2, extent=extent, origin='lower')

        plt.tight_layout()
        plt.show()
        return fig

    def animate_optimization_results(self, model, initial_params, optimized_params, fps=30, show_full_domain=True, y_crop=None, save_snapshots=True, snapshot_interval=0.01, snapshot_dir="animation_snapshots"):
        """
        Animate temperature evolution for raster and optimized scans.
        Shows laser positions and melt pool outlines at each frame.
        Optionally saves snapshots at intervals.
        """
        # Raster scan setup
        x_start, x_end = 0, self.model.Lx
        y_start = 0.0010
        y_end = 0.0015
        hatch_spacing = 0.00005
        n_lines = int(np.floor((y_end - y_start) / hatch_spacing)) + 1
        raster_speed = 0.2
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

        # Optimized scan setup
        temp_optimizer = TrajectoryOptimizer(self.model, initial_params=initial_params, bounds=None)
        optimized_array = temp_optimizer.parameters_to_array(optimized_params)
        laser_opt, heat_opt = temp_optimizer.unpack_parameters(optimized_array)
        sawtooth_params, swirl_params = laser_opt

        # Animation setup
        dt = self.model.dt
        nt = self.model.nt
        t_points = np.linspace(0, nt * dt, nt)
        colors = [(0, 0, 0.3), (0, 0, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0), (1, 1, 1)]
        cmap_temp = LinearSegmentedColormap.from_list('thermal', colors)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        extent = [0, self.model.Lx * 1000, 0, self.model.Ly * 1000]

        # Add inset axis for melt pool geometry
        ax_inset = inset_axes(ax2, width="30%", height="30%", loc='lower left', borderpad=2, bbox_to_anchor=(0.08, 0.05, 0.4, 0.4), bbox_transform=ax2.transAxes)
        ax_inset.set_title("Melt Pool Outline", fontsize=8)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_xlim(0, self.model.Lx * 1000)
        ax_inset.set_ylim(0, self.model.Ly * 1000)

        # Initial temperature fields
        T_raster = self.model.material['T0'] * np.ones((self.model.ny, self.model.nx))
        T_opt = self.model.material['T0'] * np.ones((self.model.ny, self.model.nx))

        temp_vmin = self.model.material['T0']
        temp_vmax = 200

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

        # Precompute raster and optimized positions
        raster_positions = [raster_trajectory(t, {'v': raster_v, 'A': 0, 'y0': 0, 'period': 1, 'noise_sigma': 0}) for t in t_points]
        saw_positions = [self.model.sawtooth_trajectory(t, sawtooth_params) for t in t_points]
        swirl_positions = [self.model.swirl_trajectory(t, swirl_params) for t in t_points]

        # Setup for saving snapshots
        if save_snapshots:
            if not os.path.exists(snapshot_dir):
                os.makedirs(snapshot_dir)
            snapshot_frames = set()
            total_time = nt * dt
            snapshot_times = np.arange(0, total_time, snapshot_interval)
            for snap_time in snapshot_times:
                frame_idx = int(np.round(snap_time / dt))
                if frame_idx < nt:
                    snapshot_frames.add(frame_idx)

        def update(frame):
            # Clear axes for new frame
            ax1.cla()
            ax2.cla()

            # Update temperature fields for raster scan
            im1 = ax1.imshow(update.T_raster, extent=extent, origin='lower', cmap=cmap_temp, aspect='auto', vmin=temp_vmin, vmax=5000)
            ax1.set_title('Raster Scan Temperature')
            ax1.set_xlabel('X (mm)')
            ax1.set_ylabel('Y (mm)')

            # Update temperature fields for optimized scan
            im2 = ax2.imshow(update.T_opt, extent=extent, origin='lower', cmap=cmap_temp, aspect='auto', vmin=temp_vmin, vmax=5000)
            ax2.set_title('Optimized Scan Temperature')
            ax2.set_xlabel('X (mm)')
            ax2.set_ylabel('Y (mm)')

            # Update laser positions
            raster_laser_dot, = ax1.plot([], [], 'ko', markersize=6, label='Raster Laser')
            saw_dot, = ax2.plot([], [], 'ro', markersize=6, label='Sawtooth Laser')
            swirl_dot, = ax2.plot([], [], 'ko', markersize=6, label='Swirl Laser')
            ax1.legend()
            ax2.legend()

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

            # Set temperature scale bar to fixed range
            im1.set_clim(vmin=temp_vmin, vmax=5000)
            im2.set_clim(vmin=temp_vmin, vmax=5000)

            # Annotate max temperature for each subplot
            max_temp_raster = np.max(update.T_raster)
            max_temp_opt = np.max(update.T_opt)
            update.raster_max_temp_text = ax1.text(
                0.98, 0.02,
                f"Max T: {max_temp_raster:.1f}°C",
                transform=ax1.transAxes, fontsize=10, color='white', ha='right', va='bottom',
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round')
            )
            update.opt_max_temp_text = ax2.text(
                0.98, 0.02,
                f"Max T: {max_temp_opt:.1f}°C",
                transform=ax2.transAxes, fontsize=10, color='white', ha='right', va='bottom',
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round')
            )

            # Melt pool outline in inset for optimized scan
            ax_inset.clear()
            ax_inset.set_title("Melt Pool Shape", fontsize=8, color='white')
            inset_xlim = (-0.5, 0.5)
            inset_ylim = (-0.5, 0.5)
            ax_inset.set_xlim(inset_xlim)
            ax_inset.set_ylim(inset_ylim)
            ax_inset.set_aspect('equal')
            ax_inset.set_xlabel("mm", fontsize=7, color='white', labelpad=2)
            ax_inset.set_ylabel("mm", fontsize=7, color='white', labelpad=2)
            ax_inset.tick_params(axis='both', which='major', labelsize=7, colors='white')
            T_melt = self.model.material['T_melt']
            melt_mask_opt = update.T_opt > T_melt

            if np.any(melt_mask_opt):
                contours = measure.find_contours(melt_mask_opt.astype(float), 0.5)
                if contours:
                    contour = max(contours, key=len)
                    y_idx, x_idx = contour[:, 0], contour[:, 1]
                    x_mm = np.interp(x_idx, np.arange(self.model.nx), np.linspace(0, self.model.Lx * 1e3, self.model.nx))
                    y_mm = np.interp(y_idx, np.arange(self.model.ny), np.linspace(0, self.model.Ly * 1e3, self.model.ny))
                    x_mm = x_mm - np.mean(x_mm)
                    y_mm = y_mm - np.mean(y_mm)
                    ax_inset.plot(x_mm, y_mm, color='cyan', linewidth=2)

            # Melt pool outline in inset for raster scan
            if not hasattr(update, "ax_inset_raster"):
                update.ax_inset_raster = inset_axes(ax1, width="30%", height="30%", loc='lower left', borderpad=2, bbox_to_anchor=(0.08, 0.05, 0.4, 0.4), bbox_transform=ax1.transAxes)
            ax_inset_raster = update.ax_inset_raster
            ax_inset_raster.clear()
            ax_inset_raster.set_title("Melt Pool Shape", fontsize=8, color='white')
            ax_inset_raster.set_xlim(inset_xlim)
            ax_inset_raster.set_ylim(inset_ylim)
            ax_inset_raster.set_aspect('equal')
            ax_inset_raster.set_xlabel("mm", fontsize=7, color='white', labelpad=2)
            ax_inset_raster.set_ylabel("mm", fontsize=7, color='white', labelpad=2)
            ax_inset_raster.tick_params(axis='both', which='major', labelsize=7, colors='white')
            melt_mask_raster = update.T_raster > T_melt

            if np.any(melt_mask_raster):
                contours = measure.find_contours(melt_mask_raster.astype(float), 0.5)
                if contours:
                    contour = max(contours, key=len)
                    y_idx, x_idx = contour[:, 0], contour[:, 1]
                    x_mm = np.interp(x_idx, np.arange(self.model.nx), np.linspace(0, self.model.Lx * 1e3, self.model.nx))
                    y_mm = np.interp(y_idx, np.arange(self.model.ny), np.linspace(0, self.model.Ly * 1e3, self.model.ny))
                    x_mm = x_mm - np.mean(x_mm)
                    y_mm = y_mm - np.mean(y_mm)
                    ax_inset_raster.plot(x_mm, y_mm, color='cyan', linewidth=2)

            # Save snapshot if required
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
    # Set up material parameters for LPBF simulation
    material_params = {
        'T0': 21.0,                # Initial temperature (°C)
        'alpha': 5e-6,             # Thermal diffusivity (m²/s)
        'rho': 7800.0,             # Density (kg/m³)
        'cp': 500.0,               # Specific heat capacity (J/kg·K)
        'thickness':  0.00021,     # Plate thickness (m)
        'T_melt': 1500.0,          # Melting temperature (°C)
        'k': 20.0,                 # Thermal conductivity (W/m·K)
        'absorptivity': 1,         # Laser absorptivity
    }
    
    # Define simulation domain and grid
    domain_size = (0.02, 0.0025)     # 20mm x 2.5mm domain
    grid_size = (201, 26)            # Grid resolution
    dt = 1e-4                        # Time step (s)

    # Initialize heat transfer model
    model = HeatTransferModel(
        domain_size=domain_size,
        grid_size=grid_size,
        dt=dt,
        material_params=material_params
    )
    
    # Initial laser and path parameters
    initial_params = {
        'sawtooth_v': 0.05,         # Sawtooth scan speed (m/s)
        'sawtooth_A': 0.0002,       # Sawtooth amplitude (m)
        'sawtooth_y0': 0.00125,     # Sawtooth center position (m)
        'sawtooth_period': 0.01,    # Sawtooth period (s)
        'sawtooth_Q': 200.0,        # Sawtooth laser power (W)
        'sawtooth_r0': 5e-5,        # Sawtooth beam radius (m)
        'swirl_v': 0.05,            # Swirl scan speed (m/s)
        'swirl_A': 0.0002,          # Swirl amplitude (m)
        'swirl_y0': 0.00125,        # Swirl center position (m)
        'swirl_fr': 20.0,           # Swirl frequency (Hz)
        'swirl_Q': 200.0,           # Swirl laser power (W)
        'swirl_r0': 5e-5,           # Swirl beam radius (m)
        'noise_sigma': 0.000        # Initial noise (m)
    }

    # Parameter bounds for optimization
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
        'noise_sigma': (0.0, 0.0005)
    }

    # Create optimizer for laser trajectories
    optimizer = TrajectoryOptimizer(
        model=model,
        initial_params=initial_params,
        bounds=bounds,
        x_range=(0.0025, 0.0175)  # Scan region (m)
    )
    
    # Print optimization objective description
    objective = 'standard'  # Choose objective type
    method = 'Powell'       # Choose optimization method
    objective_descriptions = {
        'standard': "Minimizing the sum of squared temperature gradients (smoothness of temperature field).",
        'thermal_uniformity': "Promoting thermal uniformity (variance in melt pool temperature and gradients).",
        'max_gradient': "Minimizing the maximum temperature gradient (reducing hot spots).",
        'path_focused': "Minimizing gradients along the laser paths (melt pool quality along scan).",
        'max_temp_difference': "Minimizing the maximum temperature difference in the melt pool."
    }
    print("\n=============================================")
    print("Starting Laser Path Optimization")
    print("=============================================\n")
    print(f"Optimization objective: {objective_descriptions.get(objective, 'Unknown objective')}")

    # Run optimization and handle errors
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
        result, optimized_params = optimizer.optimize(
            objective_type=objective,
            method='Nelder-Mead',
            max_iterations=1000
        )
        optimization_successful = result.success
    
    # Analyze and visualize results if optimization succeeded
    if optimization_successful:
        print("\n=============================================")
        print("Optimization Successful!")
        print("=============================================\n")
        
        # Show improvement in objective function
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
        
        # Print parameter changes
        print("Parameter Comparison:")
        print("---------------------------------------------")
        print(f"{'Parameter':<15} {'Initial':<12} {'Optimized':<12} {'Change %':<10}")
        print("---------------------------------------------")
        for name in optimizer.param_names:
            init_val = initial_params[name]
            opt_val = optimized_params[name]
            change = (opt_val - init_val) / init_val * 100 if init_val != 0 else float('inf')
            print(f"{name:<15} {init_val:<12.6f} {opt_val:<12.6f} {change:<10.2f}")
        
        # Visualize simulation results
        viz = Visualization(model)
        print("\nGenerating comparison visualizations...")
        viz.compare_simulations(initial_params, optimized_params)
        
        # Optionally animate results
        create_animation = input("Generate animation of optimization results? (y/n): ").lower()
        if create_animation == 'y':
            print("Generating animation (this may take a while)...")
            ani = viz.animate_optimization_results(
                model, 
                initial_params, 
                optimized_params, 
                fps=30,
            )
            
        # Optionally perform sensitivity analysis
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
    
    return model, optimizer, result, optimized_params

# ===============================
# 5. G-code Output Utility
# ===============================

def output_optimized_gcode(optimizer, optimized_params, filename="optimized_scan.gcode"):
    """
    Output the optimized scan path for both lasers as G-code.
    Args:
        optimizer: TrajectoryOptimizer instance
        optimized_params: dict of optimized parameters
        filename: output G-code filename
    """
    # Unpack optimized parameters
    params_array = optimizer.parameters_to_array(optimized_params)
    laser_params, _ = optimizer.unpack_parameters(params_array)
    sawtooth_params, swirl_params = laser_params

    # Helper to resample a path at constant arc length intervals
    def resample_path(path, v, dt=1e-5):
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

    # Generate scan paths for both lasers
    x_start, x_end = optimizer.x_range
    v1 = sawtooth_params['v']
    v2 = swirl_params['v']
    dt = 1e-5
    total_time = max((x_end - x_start) / v1, (x_end - x_start) / v2)
    t_points = np.arange(0, total_time, dt)
    saw_path = np.array([optimizer.model.sawtooth_trajectory(t, sawtooth_params)[:2] for t in t_points])
    swirl_path = np.array([optimizer.model.swirl_trajectory(t, swirl_params)[:2] for t in t_points])
    saw_path_resampled = resample_path(saw_path, v1, dt)
    swirl_path_resampled = resample_path(swirl_path, v2, dt)

    # Write G-code file
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

# Run optimization and optionally export G-code if script is executed directly
if __name__ == "__main__":
    model, optimizer, result, optimized_params = run_optimization()
    user_input = input("\nWould you like to output the optimized scan as G-code? (y/n): ").strip().lower()
    if user_input == 'y':
        output_optimized_gcode(optimizer, optimized_params)
    else:
        print("G-code export skipped.")
