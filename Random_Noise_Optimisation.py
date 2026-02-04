from logging import config
from networkx import omega
import time
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
    """
    Heat transfer simulation model for LPBF processes.
    
    Simulates 2D heat conduction with moving heat sources.
    Creates a base raster scan and compares to defined reference
    trajectories with optional noise.
    """
    
    def __init__(self, domain_size=(0.02, 0.0025), grid_size=(401, 51), dt=1e-4, material_params=None, noise_seed=None):
        """
        Initialize the heat transfer model.
    
        Args:
            domain_size: (Lx, Ly) domain dimensions in meters
            grid_size: (nx, ny) grid resolution
            dt: time step in seconds
            material_params: dictionary of material properties
            noise_seed: seed for noise RNG (Defaults random)
        """
        # Domain and grid setup
        self.Lx, self.Ly = domain_size
        self.nx, self.ny = grid_size
        self.dx = self.Lx / (self.nx - 1)
        self.dy = self.Ly / (self.ny - 1)
    
        # Create coordinate arrays
        self.x = anp.linspace(0, self.Lx, self.nx)
        self.y = anp.linspace(0, self.Ly, self.ny)
        self.X, self.Y = anp.meshgrid(self.x, self.y)
    
        # Time stepping
        self.dt = dt
        self.nt = None  # Set during simulation
    
        # Material properties with defaults
        self.material = self._set_material_properties(material_params)
    
        # Initialize temperature field
        self.T = self.material['T0'] * anp.ones((self.ny, self.nx))
    
        # Derived properties
        self.heat_capacity = self.material['rho'] * self.material['cp']
        self.thickness = self.material['thickness']
    
        # Initialize reproducible noise generator
        # Different each program run (if seed=None), but consistent within one run
        if noise_seed is None:
            import time
            self._noise_seed = int(time.time() * 1000000) % 100000000
        else:
            self._noise_seed = noise_seed
    
        self._rng = np.random.RandomState(self._noise_seed)
        print(f"Noise seed for this run: {self._noise_seed}")

    def _set_material_properties(self, material_params):
        """Set material properties with sensible defaults for steel."""
        default_params = {
            'T0': 21.0,           # Initial temperature (°C)
            'alpha': 5e-6,        # Thermal diffusivity (m²/s)
            'rho': 7800.0,        # Density (kg/m³)
            'cp': 500.0,          # Specific heat capacity (J/(kg·K))
            'k': 20.0,            # Thermal conductivity (W/(m·K))
            'T_melt': 1500.0,     # Melting temperature (°C)
            'thickness': 0.005,   # Grid thickness (m - define volume of cells)
            'absorptivity': 1.0   # Absorptivity (fraction)
        }
        
        if material_params is not None:
            default_params.update(material_params)
        
        return default_params

    def _laplacian(self, T):
        """
        Compute 2D Laplacian using central finite differences.
        
        Args:
            T: temperature field array
            
        Returns:
            Laplacian of T with zero padding at boundaries
        """
        # Interior points using central differences
        lap_x = (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / (self.dx**2)
        lap_y = (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / (self.dy**2)
        lap_inner = lap_x + lap_y
        
        # Pad with zeros for boundary conditions
        lap = anp.pad(lap_inner, pad_width=((1,1),(1,1)), mode='constant', constant_values=0)
        
        return lap

    def sawtooth_trajectory(self, t, params):
        """
        Calculate sawtooth laser position with continuous noise causing smooth path deviations.
        """
        v = params['v']
        A = params['A']
        y0 = params['y0']
        period = params['period']
        noise_sigma = params.get('noise_sigma', 0.0)

        max_noise = 0.00005
        noise_sigma = anp.clip(noise_sigma, 0.0, max_noise)

        omega = 2 * anp.pi / period
        x_start = 0.0015
        x_end = self.Lx - 0.0015

        v_y = (2 * A * omega / anp.pi)
        
        # Check if requested speed is achievable with this amplitude and period
        if v**2 > v_y**2:
            v_x = anp.sqrt(v**2 - v_y**2)
        else:
            # Speed too low for this amplitude/period - reduce amplitude
            v_x = v * 0.5
            A = A * 0.5
            v_y = (2 * A * omega / anp.pi)
    
        base_x = x_start + v_x * t
        base_y = y0 + A * (2/anp.pi) * anp.arcsin(anp.sin(omega * t))
    
        # Apply noise with reproducible RNG
        if noise_sigma > 0:
            # Initialize noise state if needed
            if not hasattr(self, '_noise_state'):
                self._noise_state = {
                    'x': 0.0,
                    'y': 0.0,
                    'last_t': -1.0
                }
    
            # Detect time discontinuities (reset or new trajectory)
            dt_noise = t - self._noise_state['last_t']
    
            # Use seeded RNG for reproducible noise
            if dt_noise > 2 * self.dt or dt_noise < 0:
                # Reset detected - reinitialize with small random perturbation
                self._noise_state['x'] = 0
                self._noise_state['y'] = 0
                self._noise_state['last_t'] = t - self.dt
                dt_noise = self.dt
    
            # Ornstein-Uhlenbeck process parameters
            theta = 200.0
            sigma_scale = 30.0
    
            # Generate random increments using seeded RNG
            diffusion_x = noise_sigma * sigma_scale * anp.sqrt(dt_noise) * self._rng.randn()
            diffusion_y = noise_sigma * sigma_scale * anp.sqrt(dt_noise) * self._rng.randn()
    
            # Update noise state with mean reversion
            drift_x = -theta * self._noise_state['x'] * dt_noise
            drift_y = -theta * self._noise_state['y'] * dt_noise
    
            self._noise_state['x'] += drift_x + diffusion_x
            self._noise_state['y'] += drift_y + diffusion_y
    
            # Apply noise to position
            x = base_x + self._noise_state['x']
            y = base_y + self._noise_state['y']
    
            # Update time
            self._noise_state['last_t'] = t
        else:
            x, y = base_x, base_y

        k = 1000
        x = x - (x - x_end) * (1 / (1 + anp.exp(-k * (x - x_end))))
        x = anp.maximum(x_start, x)
        y = anp.maximum(0, anp.minimum(y, self.Ly))

        tx_base = v_x
        ty_base = (2 * A * omega / anp.pi) * anp.cos(omega * t)
    
        norm = anp.sqrt(tx_base**2 + ty_base**2)
        tx = tx_base / norm if norm > 0 else 1.0
        ty = ty_base / norm if norm > 0 else 0.0

        return float(x), float(y), float(tx), float(ty)

    def swirl_trajectory(self, t, params):
        """
        Calculate swirl laser position with continuous noise causing smooth path deviations.
        """
        v = params['v']
        A = params['A']
        y0 = params['y0']
        fr = params['fr']
        noise_sigma = params.get('noise_sigma', 0.0)

        max_noise = 0.00005
        noise_sigma = anp.clip(noise_sigma, 0.0, max_noise)

        om = 2 * anp.pi * fr
        x_start = 0.0015
        x_end = self.Lx - 0.0015
        
        A_om = A * om
        if v**2 > A_om**2:
            v_base = anp.sqrt(v**2 - A_om**2)
        else:
            v_base = v * 0.7
            A = A * 0.5
            A_om = A * om
    
        # Calculate base trajectory position
        base_x = x_start + v_base * t + A * anp.cos(om * t)
        base_y = y0 + A * anp.sin(om * t)
    
        # Apply noise with reproducible RNG
        if noise_sigma > 0:
            # Initialize noise state if needed
            if not hasattr(self, '_noise_state_swirl'):
                self._noise_state_swirl = {
                    'x': 0.0,
                    'y': 0.0,
                    'last_t': -1.0
                }
    
            # Detect time discontinuities
            dt_noise = t - self._noise_state_swirl['last_t']
    
            # Use seeded RNG for reproducible noise
            if dt_noise > 2 * self.dt or dt_noise < 0:
                self._noise_state_swirl['x'] = 0
                self._noise_state_swirl['y'] = 0
                self._noise_state_swirl['last_t'] = t - self.dt
                dt_noise = self.dt
    
            # Ornstein-Uhlenbeck process
            theta = 200.0
            sigma_scale = 30.0
    
            # Generate random increments using seeded RNG
            diffusion_x = noise_sigma * sigma_scale * anp.sqrt(dt_noise) * self._rng.randn()
            diffusion_y = noise_sigma * sigma_scale * anp.sqrt(dt_noise) * self._rng.randn()
    
            # Update with mean reversion
            drift_x = -theta * self._noise_state_swirl['x'] * dt_noise
            drift_y = -theta * self._noise_state_swirl['y'] * dt_noise
    
            self._noise_state_swirl['x'] += drift_x + diffusion_x
            self._noise_state_swirl['y'] += drift_y + diffusion_y
    
            x = base_x + self._noise_state_swirl['x']
            y = base_y + self._noise_state_swirl['y']
    
            self._noise_state_swirl['last_t'] = t
        else:
            x, y = base_x, base_y

        # Apply boundary constraints
        k = 1000
        x = x - (x - x_end) * (1 / (1 + anp.exp(-k * (x - x_end))))
        x = anp.maximum(x_start, x)
        y = anp.maximum(0, anp.minimum(y, self.Ly))

        # Direction vector from base trajectory
        # dx/dt = v_base - A*om*sin(om*t), dy/dt = A*om*cos(om*t)
        tx_base = v_base - A * om * anp.sin(om * t)
        ty_base = A * om * anp.cos(om * t)
    
        # Normalize to unit vector
        norm = anp.sqrt(tx_base**2 + ty_base**2)
        tx = tx_base / norm if norm > 0 else 1.0
        ty = ty_base / norm if norm > 0 else 0.0

        return float(x), float(y), float(tx), float(ty)

    def straight_trajectory(self, t, params):
        """
        Calculate straight laser position with noise causing smooth path deviations.
        """
        v = params['v']
        y0 = params['y0']
        noise_sigma = params.get('noise_sigma', 0.0)

        max_noise = 0.00005
        noise_sigma = anp.clip(noise_sigma, 0.0, max_noise)

        x_start = 0.0015
        x_end = self.Lx - 0.0015

        # Base velocity
        base_x = x_start + v * t
        base_y = y0
    
        # Apply noise with reproducible RNG
        if noise_sigma > 0:
            # Initialize noise state if needed
            if not hasattr(self, '_noise_state_straight'):
                self._noise_state_straight = {
                    'x': 0.0,
                    'y': 0.0,
                    'last_t': -1.0
                }
    
            # Detect time discontinuities
            dt_noise = t - self._noise_state_straight['last_t']
    
            # Use seeded RNG for reproducible noise
            if dt_noise > 2 * self.dt or dt_noise < 0:
                self._noise_state_straight['x'] = 0
                self._noise_state_straight['y'] = 0
                self._noise_state_straight['last_t'] = t - self.dt
                dt_noise = self.dt
    
            # Ornstein-Uhlenbeck process
            theta = 200.0
            sigma_scale = 30.0
    
            # Generate random increments using seeded RNG
            diffusion_x = noise_sigma * sigma_scale * anp.sqrt(dt_noise) * self._rng.randn()
            diffusion_y = noise_sigma * sigma_scale * anp.sqrt(dt_noise) * self._rng.randn()
    
            # Update with mean reversion
            drift_x = -theta * self._noise_state_straight['x'] * dt_noise
            drift_y = -theta * self._noise_state_straight['y'] * dt_noise
    
            self._noise_state_straight['x'] += drift_x + diffusion_x
            self._noise_state_straight['y'] += drift_y + diffusion_y
    
            x = base_x + self._noise_state_straight['x']
            y = base_y + self._noise_state_straight['y']
    
            self._noise_state_straight['last_t'] = t
        else:
            x, y = base_x, base_y

        # Apply boundary constraints
        k = 1000
        x = x - (x - x_end) * (1 / (1 + anp.exp(-k * (x - x_end))))
        x = anp.maximum(x_start, x)
        y = anp.maximum(0, anp.minimum(y, self.Ly))
        # Direction vector is simply horizontal
        tx, ty = 1.0, 0.0

        return float(x), float(y), float(tx), float(ty)
    
    def reset(self):
        """Reset temperature field to initial conditions and clear noise states."""
        self.T = self.material['T0'] * anp.ones((self.ny, self.nx))

        # Delete noise states for all trajectory types
        if hasattr(self, '_noise_state'):
            del self._noise_state

        if hasattr(self, '_noise_state_swirl'):
            del self._noise_state_swirl

        if hasattr(self, '_noise_state_straight'):
            del self._noise_state_straight

   

    def precalculate_trajectory(self, laser_config, nt, dt):
        """
        Pre-calculate entire trajectory with noise evolution.
    
        Args:
            laser_config: laser configuration dictionary
            nt: number of time steps
            dt: time step size
        
        Returns:
            list of (x, y, tx, ty) tuples for each time step
        """
    
        # Extract trajectory type from laser configuration
        trajectory_type = laser_config['type']
        
        # Reset noise state for this trajectory type before precalculation
        if trajectory_type == 'sawtooth':
            if hasattr(self, '_noise_state'):
                del self._noise_state
        elif trajectory_type == 'swirl':
            if hasattr(self, '_noise_state_swirl'):
                del self._noise_state_swirl
        elif trajectory_type == 'straight':
            if hasattr(self, '_noise_state_straight'):
                del self._noise_state_straight
        
        # Calculate the raw trajectory points with noise
        raw_trajectory_points = []
        for n in range(nt):
            t = n * dt
            if trajectory_type == 'sawtooth':
                x, y, tx, ty = self.sawtooth_trajectory(t, laser_config['params'])
            elif trajectory_type == 'swirl':
                x, y, tx, ty = self.swirl_trajectory(t, laser_config['params'])
            elif trajectory_type == 'straight':
                x, y, tx, ty = self.straight_trajectory(t, laser_config['params'])
            else:
                raise ValueError(f"Unknown trajectory type: {trajectory_type}")
        
            raw_trajectory_points.append((x, y, tx, ty))
        
        # Check if noise is being used
        noise_sigma = laser_config['params'].get('noise_sigma', 0.0)
        
        # If noise is present, interpolate to maintain constant distance per time step
        if noise_sigma > 0:
            trajectory_points = self._interpolate_trajectory_arc_length(raw_trajectory_points, dt, laser_config['params'])
        else:
            trajectory_points = raw_trajectory_points
    
        return trajectory_points
    
    def _interpolate_trajectory_arc_length(self, raw_points, dt, params):
        """
        Interpolate trajectory points to maintain approximately constant arc length per time step.
        
        When noise is applied, the laser jumps varying distances between time steps.
        This function adds intermediate points so the laser travels the same distance 
        at each time step, moving smoothly between the noisy coordinates.
        
        Args:
            raw_points: list of (x, y, tx, ty) tuples from noisy trajectory
            dt: original time step
            params: trajectory parameters containing speed 'v'
            
        Returns:
            list of interpolated (x, y, tx, ty) tuples with consistent spacing
        """
        if len(raw_points) < 2:
            return raw_points
        
        # Get target speed from parameters
        v = params.get('v', 0.7)  # Default speed if not specified
        target_distance = v * dt  # Target distance per time step
        
        # Build the path as a list of segments with cumulative distances
        cumulative_distances = [0.0]
        total_distance = 0.0
        
        for i in range(len(raw_points) - 1):
            x1, y1, _, _ = raw_points[i]
            x2, y2, _, _ = raw_points[i + 1]
            segment_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += segment_distance
            cumulative_distances.append(total_distance)
        
        # Determine points needed
        num_steps = int(np.ceil(total_distance / target_distance))
        if num_steps == 0:
            return raw_points
        
        # Create evenly spaced distance samples
        interpolated_points = []
        interpolated_points.append(raw_points[0])
        
        for step in range(1, num_steps):
            target_dist = step * target_distance
            
            # Find segment distance falls in
            segment_idx = 0
            for i in range(len(cumulative_distances) - 1):
                if cumulative_distances[i] <= target_dist < cumulative_distances[i + 1]:
                    segment_idx = i
                    break
            
            # Handle edge case where target_dist equals total_distance
            if target_dist >= cumulative_distances[-1]:
                segment_idx = len(raw_points) - 2
            
            # Interpolate within this segment
            x1, y1, _, _ = raw_points[segment_idx]
            x2, y2, _, _ = raw_points[segment_idx + 1]
            
            # Calculate the fraction along this specific segment
            segment_start_dist = cumulative_distances[segment_idx]
            segment_end_dist = cumulative_distances[segment_idx + 1]
            segment_length = segment_end_dist - segment_start_dist
            
            if segment_length > 0:
                local_fraction = (target_dist - segment_start_dist) / segment_length
                local_fraction = np.clip(local_fraction, 0.0, 1.0)
                
                # Interpolate position
                x_new = x1 + local_fraction * (x2 - x1)
                y_new = y1 + local_fraction * (y2 - y1)
                
                # Calculate tangent direction from segment
                segment_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if segment_dist > 0:
                    tx = (x2 - x1) / segment_dist
                    ty = (y2 - y1) / segment_dist
                else:
                    tx, ty = 1.0, 0.0
                
                interpolated_points.append((x_new, y_new, tx, ty))
        
        # End point
        interpolated_points.append(raw_points[-1])
        
        return interpolated_points

    def get_trajectory_point_from_cache(self, t, laser_config, cached_trajectory=None):
        """
        Get trajectory point from pre-calculated cache or calculate on-the-fly.
    
        Args:
            t: time
            laser_config: laser configuration
            cached_trajectory: pre-calculated trajectory points (optional)
        
        Returns:
            (x, y, tx, ty) tuple
        """
        if cached_trajectory is not None:
            # Use cached trajectory
            dt = self.dt if hasattr(self, 'dt') else 1e-5
            frame = int(t / dt)
            nt = len(cached_trajectory)
            frame = min(frame, nt - 1)  # Clamp to valid range
            return cached_trajectory[frame]
        else:
            # Calculate on-the-fly
            trajectory_type = laser_config['type']
            if trajectory_type == 'sawtooth':
                return self.sawtooth_trajectory(t, laser_config['params'])
            elif trajectory_type == 'swirl':
                return self.swirl_trajectory(t, laser_config['params'])
            elif trajectory_type == 'straight':
                return self.straight_trajectory(t, laser_config['params'])
            else:
                raise ValueError(f"Unknown trajectory type: {trajectory_type}")

    def _gaussian_source(self, x_src, y_src, heat_params):
        """
        Calculate Gaussian heat source distribution centered at (x_src, y_src).
        
        Args:
            x_src, y_src: source center coordinates
            heat_params: dictionary with 'Q' (power) and 'r0' or 'sigma_x'/'sigma_y'
            
        Returns:
            Heat source distribution array
        """
        power = heat_params.get('Q', 200.0)
        
        # Determine Gaussian width parameters
        if 'r0' in heat_params:
            sigma_x = sigma_y = heat_params['r0'] / np.sqrt(2)
        else:
            sigma_x = heat_params.get('sigma_x', 1.5e-3)
            sigma_y = heat_params.get('sigma_y', 1.5e-3)
        
        # Apply absorptivity
        absorbed_power = power * self.material.get('absorptivity', 1.0)
        
        # Calculate Gaussian distribution
        G = np.exp(-(((self.X - x_src)**2) / (2 * sigma_x**2) +
                     ((self.Y - y_src)**2) / (2 * sigma_y**2)))
        
        return G * absorbed_power

    def _apply_boundary_conditions(self, T):
        """Apply convective boundary conditions."""
        # Convective heat transfer coefficient (W/m²·K)
        h = 10.0  # Typical value for air convection
        T_amb = self.material['T0']  # Ambient temperature
    
        # Calculate convective cooling rate
        dt_factor = self.dt * h / (self.material['rho'] * self.material['cp'] * self.thickness)
    
        # Apply convective cooling
        # Top boundary
        T[0, :] = T[0, :] - dt_factor * np.maximum(0, T[0, :] - T_amb)
        # Bottom boundary
        T[-1, :] = T[-1, :] - dt_factor * np.maximum(0, T[-1, :] - T_amb)
        # Left boundary
        T[:, 0] = T[:, 0] - dt_factor * np.maximum(0, T[:, 0] - T_amb)
        # Right boundary
        T[:, -1] = T[:, -1] - dt_factor * np.maximum(0, T[:, -1] - T_amb)
    
        # Ensure temperature doesn't go below ambient
        T = np.maximum(T, T_amb)
    
        return T

    def simulate(self, parameters, start_x=0.0, end_x=None, use_gaussian=True, verbose=False, nt=None):
        """
        Run heat transfer simulation with moving laser sources.
        
        Args:
            nt: Number of time steps (default: 6000)
        """
        laser_params, heat_params = parameters

        # Reset to initial conditions
        self.reset()

        # Set simulation domain
        if end_x is None:
            end_x = self.Lx

        # Fixed number of time steps for consistency (before interpolation)
        self.nt = nt if nt is not None else 4000

        # Get thermal diffusivity
        if 'alpha' not in self.material and 'k' in self.material:
            alpha = self.material['k'] / (self.material['rho'] * self.material['cp'])
        else:
            alpha = self.material['alpha']

        # Pre-calculate trajectories with continuous noise evolution
        # NOTE: After interpolation, the actual number of points may differ from self.nt
        if isinstance(laser_params, tuple):
            cached_trajectories = []
            for laser_config in laser_params:
                traj_cache = self.precalculate_trajectory(laser_config, self.nt, self.dt)
                cached_trajectories.append(traj_cache)
        else:
            cached_trajectories = self.precalculate_trajectory(laser_params, self.nt, self.dt)
    
        # Store cached trajectories for animation/visualization use
        self._cached_trajectories = cached_trajectories
        
        # Determine actual number of time steps from the trajectories
        if isinstance(cached_trajectories, list) and len(cached_trajectories) > 0:
            if isinstance(cached_trajectories[0], list):
                # Multiple lasers - use the maximum length
                actual_nt = max(len(traj) for traj in cached_trajectories)
            else:
                # Single laser
                actual_nt = len(cached_trajectories)
        else:
            actual_nt = self.nt
    
        # Initialize gradient tracking arrays
        max_gradients = []
        mean_gradients = []

        # Main simulation loop
        T = self.T.copy()
    
        # Boundary limits
        x_start_limit = 0.0015
        x_end_limit = self.Lx - 0.0015
    
        # Track when lasers have finished
        all_lasers_finished = False
        final_timestep = actual_nt

        for n in range(actual_nt):
            t = n * self.dt
        
            # Calculate heat source from laser(s) using cached trajectories
            S_total = anp.zeros_like(T)
            any_laser_active = False  # Track if any laser is still active
        
            if isinstance(laser_params, tuple):
                # Multiple lasers
                for i, laser_config in enumerate(laser_params):
                    # Get position from cache (handle varying trajectory lengths)
                    if n < len(cached_trajectories[i]):
                        x, y, _, _ = cached_trajectories[i][n]
                    else:
                        # Use last position if out of trajectory points
                        x, y, _, _ = cached_trajectories[i][-1]
            
                    # Check if laser is within valid range
                    laser_active = x_start_limit <= x <= x_end_limit
                
                    if laser_active:
                        any_laser_active = True
                
                    # Get heat parameters
                    heat_p = heat_params[i] if isinstance(heat_params, (list, tuple)) else heat_params

                    if use_gaussian and laser_active:
                        S = self._gaussian_source(x, y, heat_p)
                        S_total += S
            else:
                # Single laser - get position from cache
                if n < len(cached_trajectories):
                    x, y, _, _ = cached_trajectories[n]
                else:
                    # Use last position if run out of trajectory points
                    x, y, _, _ = cached_trajectories[-1]
        
                # Check if laser is within valid range
                laser_active = x_start_limit <= x <= x_end_limit
            
                if laser_active:
                    any_laser_active = True
        
                if use_gaussian and laser_active:
                    S_total = self._gaussian_source(x, y, heat_params)
        
            # Check if all lasers have finished and store temperature at that moment
            if not any_laser_active and not all_lasers_finished:
                all_lasers_finished = True
                final_timestep = n
                T_at_laser_finish = T.copy()  # Store temperature when laser turns off
        
            # Heat equation: dT/dt = alpha * laplacian(T) + S/(rho*cp*thickness)
            lap = self._laplacian(T)
            dT_diffusion = alpha * lap
        
            # Source term
            volume_factor = self.dx * self.dy * self.thickness
            dT_source = S_total / (volume_factor * self.heat_capacity)
        
            # Update temperature
            T = T + self.dt * (dT_diffusion + dT_source)
        
            # Apply boundary conditions
            T = self._apply_boundary_conditions(T)
        
            # Track gradients for analysis
            _, _, grad_mag = self.compute_temperature_gradients(T)
            max_gradients.append(anp.max(grad_mag))
            mean_gradients.append(anp.mean(grad_mag))
        
            # OPTIONAL: Stop simulation early once optimsed scan finishes, doesn't guarentee multiple raster vectors
            # Uncomment the following lines to stop immediately after lasers finish
            # if all_lasers_finished:
            #     print(f"Stopping simulation early at timestep {n}")
            #     break

        # Store temporal gradient statistics
        self.temporal_grad_stats = {
            'max_over_time': np.max(max_gradients),
            'mean_over_time': np.mean(mean_gradients),
            'max_gradients': max_gradients,
            'mean_gradients': mean_gradients,
            'final_timestep': final_timestep,  # Store when lasers actually finished
            'T_at_laser_finish': T_at_laser_finish if all_lasers_finished else T  # Temperature when laser turned off
        }

        self.T = T
        return self.T
    
    def _get_laser_position(self, t, laser_params):
        """Get laser position based on trajectory type."""
        if laser_params['type'] == 'sawtooth':
            return self.sawtooth_trajectory(t, laser_params['params'])
        elif laser_params['type'] == 'swirl':
            return self.swirl_trajectory(t, laser_params['params'])
        elif laser_params['type'] == 'straight':
            return self.straight_trajectory(t, laser_params['params'])
        else:
            raise ValueError(f"Unknown trajectory type: {laser_params['type']}")

    def compute_temperature_gradients(self, T=None):
        """
        Compute spatial temperature gradients using central differences.
        
        Args:
            T: temperature field (uses self.T if None)
            
        Returns:
            grad_x, grad_y, grad_mag: gradient components and magnitude
        """
        if T is None:
            T = self.T
        
        # X-gradient with boundary handling
        grad_x_left = (T[:, 1:2] - T[:, 0:1]) / self.dx
        grad_x_interior = (T[:, 2:] - T[:, :-2]) / (2 * self.dx)
        grad_x_right = (T[:, -1:] - T[:, -2:-1]) / self.dx
        grad_x = anp.concatenate([grad_x_left, grad_x_interior, grad_x_right], axis=1)
        
        # Y-gradient with boundary handling
        grad_y_top = (T[1:2, :] - T[0:1, :]) / self.dy
        grad_y_interior = (T[2:, :] - T[:-2, :]) / (2 * self.dy)
        grad_y_bottom = (T[-1:, :] - T[-2:-1, :]) / self.dy
        grad_y = anp.concatenate([grad_y_top, grad_y_interior, grad_y_bottom], axis=0)
        
        # Gradient magnitude
        grad_mag = anp.sqrt(grad_x**2 + grad_y**2)
        
        return grad_x, grad_y, grad_mag

# ===============================
# 2. Trajectory Optimizer Class
# ===============================
    
class TrajectoryOptimizer:
    """
    Optimization class for laser trajectory parameters in L-PBF processes.
    
    Supports both single and dual laser configurations with various
    objective functions and constraint handling.
    """
    
    def __init__(self, model, initial_params=None, bounds=None, x_range=(0.0, 0.01)):
        """
        Initialize the trajectory optimizer.
        
        Args:
            model: HeatTransferModel instance
            initial_params: dictionary of initial parameter values
            bounds: dictionary of parameter bounds (min, max)
            x_range: tuple of (x_start, x_end) for scan region
        """
        self.model = model
        self.initial_params = initial_params or {}
        self.bounds = bounds or {}
        self.x_range = x_range
        
        # Extract parameter names for optimization
        self.param_names = self._get_optimization_parameters()
        
    def _get_optimization_parameters(self):
        """Extract parameters to be optimized."""
        fixed_keys = {
            'sawtooth_y0', 'swirl_y0', 'straight_y0',
            'sawtooth2_y0', 'swirl2_y0', 'straight2_y0',
            'sawtooth_r0', 'swirl_r0', 'straight_r0',
            'sawtooth2_r0', 'swirl2_r0', 'straight2_r0'
        }

        param_names = [k for k in self.initial_params.keys() if k not in fixed_keys]

        # Always include noise_sigma if present in bounds
        if self.bounds and 'noise_sigma' in self.bounds and 'noise_sigma' not in param_names:
            param_names.append('noise_sigma')
    
        return param_names

    def parameters_to_array(self, params_dict):
        """Convert parameter dictionary to flat array for optimization."""
        return anp.array([params_dict[name] for name in self.param_names])
    
    def array_to_parameters(self, params_array):
        """Convert flat array to parameter dictionary."""
        return {name: params_array[i] for i, name in enumerate(self.param_names)}
    
    def unpack_parameters(self, params_array):
        """
        Convert flat parameter array to structured laser and heat source parameters.
        
        Args:
            params_array: flat array of optimization parameters
            
        Returns:
            tuple: (laser_params, heat_params) for simulation
        """
        params_dict = self.array_to_parameters(params_array)
        noise_sigma = params_dict.get('noise_sigma', self.initial_params.get('noise_sigma', 0.0))

        # Check if dual source configuration
        if self._is_dual_source(params_dict):
            return self._unpack_dual_source(params_dict, noise_sigma)
        else:
            return self._unpack_single_source(params_dict, noise_sigma)
    
    def _is_dual_source(self, params_dict):
        """Check if parameters define dual source configuration."""
        dual_keys = {'sawtooth2_v', 'swirl2_v', 'straight2_v'}
        return any(key in params_dict for key in dual_keys)
    
    def _unpack_dual_source(self, params_dict, noise_sigma):
        """Unpack parameters for dual laser configuration."""
        # Source 1
        laser1, heat1 = self._create_laser_config(params_dict, 1, noise_sigma)
        
        # Source 2  
        laser2, heat2 = self._create_laser_config(params_dict, 2, noise_sigma)
        
        return (laser1, laser2), (heat1, heat2)
    
    def _unpack_single_source(self, params_dict, noise_sigma):
        """Unpack parameters for single laser configuration."""
        laser_params, heat_params = self._create_laser_config(params_dict, 1, noise_sigma)
        return laser_params, heat_params
    
    def _create_laser_config(self, params_dict, source_num, noise_sigma):
        """
        Create laser configuration for specified source number.
        
        Args:
            params_dict: parameter dictionary
            source_num: 1 for primary source, 2 for secondary
            noise_sigma: noise level
            
        Returns:
            tuple: (laser_config, heat_config)
        """
        suffix = '' if source_num == 1 else '2'
        
        # Check trajectory type for this source
        if f'straight{suffix}_v' in params_dict:
            return self._create_straight_config(params_dict, suffix, noise_sigma)
        elif f'sawtooth{suffix}_v' in params_dict:
            return self._create_sawtooth_config(params_dict, suffix, noise_sigma)
        elif f'swirl{suffix}_v' in params_dict:
            return self._create_swirl_config(params_dict, suffix, noise_sigma)
        else:
            raise ValueError(f"No valid trajectory parameters found for source {source_num}")

    def _create_straight_config(self, params_dict, suffix, noise_sigma):
        """Create straight trajectory configuration."""
        prefix = f'straight{suffix}'
    
        laser_config = {
            'type': 'straight',
            'params': {
                'v': params_dict[f'{prefix}_v'],
                'y0': self.initial_params.get(f'{prefix}_y0', 0.0005),
                'noise_sigma': noise_sigma
            }
        }
    
        heat_config = {
            'Q': params_dict[f'{prefix}_Q'],
            'r0': self.initial_params.get(f'{prefix}_r0', 4e-5)
        }
    
        return laser_config, heat_config
    
    def _create_sawtooth_config(self, params_dict, suffix, noise_sigma):
        """Create sawtooth trajectory configuration."""
        prefix = f'sawtooth{suffix}'
        
        laser_config = {
            'type': 'sawtooth',
            'params': {
                'v': params_dict[f'{prefix}_v'],
                'A': params_dict[f'{prefix}_A'],
                'y0': self.initial_params.get(f'{prefix}_y0', 0.0005),
                'period': params_dict[f'{prefix}_period'],
                'noise_sigma': noise_sigma
            }
        }
        
        heat_config = {
            'Q': params_dict[f'{prefix}_Q'],
            'r0': self.initial_params.get(f'{prefix}_r0', 4e-5)
        }
        
        return laser_config, heat_config
    
    def _create_swirl_config(self, params_dict, suffix, noise_sigma):
        """Create swirl trajectory configuration."""
        prefix = f'swirl{suffix}'
        
        laser_config = {
            'type': 'swirl',
            'params': {
                'v': params_dict[f'{prefix}_v'],
                'A': params_dict[f'{prefix}_A'],
                'y0': self.initial_params.get(f'{prefix}_y0', 0.0005),
                'fr': params_dict[f'{prefix}_fr'],
                'noise_sigma': noise_sigma
            }
        }
        
        heat_config = {
            'Q': params_dict[f'{prefix}_Q'],
            'r0': self.initial_params.get(f'{prefix}_r0', 4e-5)
        }
        
        return laser_config, heat_config

    # ===============================
    # Objective Functions
    # ===============================

    def _validate_parameters(self, params_array):
        """Check for extreme parameter values that would cause numerical issues."""
        for i, name in enumerate(self.param_names):
            if anp.abs(params_array[i]) > 1e6:
                print(f"Warning: Parameter {name} has extreme value: {params_array[i]}")
                return False
        return True
    
    def _get_heat_source_type(self, heat_params):
        """Determine if heat source uses Gaussian distribution."""
        if isinstance(heat_params, (list, tuple)):
            return 'r0' in heat_params[0]
        else:
            return 'r0' in heat_params

    def objective_function(self, params_array):
        """
        Standard objective: minimize sum of squared temperature gradients (smoothness).
        
        Args:
            params_array: flat array of optimization parameters
            
        Returns:
            float: objective function value
        """
        if not self._validate_parameters(params_array):
            return 1e10
            
        laser_params, heat_params = self.unpack_parameters(params_array)
        use_gaussian = self._get_heat_source_type(heat_params)
        
        T = self.model.simulate(
            (laser_params, heat_params),
            start_x=self.x_range[0],
            end_x=self.x_range[1],
            use_gaussian=use_gaussian
        )
        
        _, _, grad_mag = self.model.compute_temperature_gradients(T)
        cost = anp.sum(grad_mag**2) / (self.model.nx * self.model.ny)
        
        return cost

    def objective_max_gradient(self, params_array):
        """
        Objective: minimize maximum temperature gradient (reduce hot spots).
        """
        if not self._validate_parameters(params_array):
            return 1e10
            
        laser_params, heat_params = self.unpack_parameters(params_array)
        use_gaussian = self._get_heat_source_type(heat_params)
        
        T = self.model.simulate(
            (laser_params, heat_params),
            start_x=self.x_range[0],
            end_x=self.x_range[1],
            use_gaussian=use_gaussian
        )
        
        _, _, grad_mag = self.model.compute_temperature_gradients(T)
        return anp.max(grad_mag)

    def objective_path_focused(self, params_array):
        """
        Objective: minimize gradients along laser paths only.
        """
        if not self._validate_parameters(params_array):
            return 1e10
            
        laser_params, heat_params = self.unpack_parameters(params_array)
        use_gaussian = self._get_heat_source_type(heat_params)
        
        T = self.model.simulate(
            (laser_params, heat_params),
            start_x=self.x_range[0],
            end_x=self.x_range[1],
            use_gaussian=use_gaussian
        )
        
        _, _, grad_mag = self.model.compute_temperature_gradients(T)
        
        # Sample points along laser paths
        path_gradients = self._sample_path_gradients(laser_params, grad_mag)
        
        return anp.mean(anp.array(path_gradients))
    
    def _sample_path_gradients(self, laser_params, grad_mag):
        """Sample temperature gradients along laser paths."""
        times = anp.linspace(0, self.model.nt * self.model.dt, 50)
        path_points = []
        
        # Handle single or dual laser configuration
        if isinstance(laser_params, tuple):
            for laser in laser_params:
                for t in times:
                    x, y, _, _ = self._get_trajectory_point(t, laser)
                    path_points.append((x, y))
        else:
            for t in times:
                x, y, _, _ = self._get_trajectory_point(t, laser_params)
                path_points.append((x, y))
        
        # Extract gradients at path points
        path_gradients = []
        dx, dy = self.model.dx, self.model.dy
        
        for x, y in path_points:
            i = int(anp.clip(y / dy, 0, self.model.ny - 1))
            j = int(anp.clip(x / dx, 0, self.model.nx - 1))
            path_gradients.append(grad_mag[i, j])
            
        return path_gradients
    
    def _get_trajectory_point(self, t, laser_config):
        """Get trajectory point for given time and laser configuration."""
        if laser_config['type'] == 'sawtooth':
            return self.model.sawtooth_trajectory(t, laser_config['params'])
        elif laser_config['type'] == 'swirl':
            return self.model.swirl_trajectory(t, laser_config['params'])
        elif laser_config['type'] == 'straight':
            return self.model.straight_trajectory(t, laser_config['params'])
        else:
            raise ValueError(f"Unknown trajectory type: {laser_config['type']}")

    def objective_thermal_uniformity(self, params_array):
        """
        Objective: promote uniform melt pool temperature and minimize gradients.
        """
        if not self._validate_parameters(params_array):
            return 1e10
            
        laser_params, heat_params = self.unpack_parameters(params_array)
        use_gaussian = self._get_heat_source_type(heat_params)
        
        T = self.model.simulate(
            (laser_params, heat_params),
            start_x=self.x_range[0],
            end_x=self.x_range[1],
            use_gaussian=use_gaussian
        )
        
        _, _, grad_mag = self.model.compute_temperature_gradients(T)
        grad_cost = anp.mean(grad_mag**2)
        
        # Calculate temperature variance in melt pool
        melt_temp = self.model.material['T_melt']
        melt_mask = T > melt_temp
        
        if anp.sum(melt_mask) > 0:
            T_melt = T[melt_mask]
            T_variance = anp.var(T_melt)
        else:
            T_variance = 0.0
        
        # Weighted combination of gradient and temperature variance
        cost = 0.7 * grad_cost + 0.3 * T_variance
        return cost

    def objective_max_temp_difference(self, params_array):
        """
        Objective: minimize maximum temperature difference in melt pool.
        """
        if not self._validate_parameters(params_array):
            return 1e10
            
        laser_params, heat_params = self.unpack_parameters(params_array)
        use_gaussian = self._get_heat_source_type(heat_params)
        
        T = self.model.simulate(
            (laser_params, heat_params),
            start_x=self.x_range[0],
            end_x=self.x_range[1],
            use_gaussian=use_gaussian
        )
        
        melt_temp = self.model.material['T_melt']
        melt_mask = T > melt_temp
        
        if anp.sum(melt_mask) > 0:
            T_melt = T[melt_mask]
            max_temp_diff = anp.max(T_melt) - anp.min(T_melt)
        else:
            max_temp_diff = 1000.0  # Penalty if no melt pool
            
        return max_temp_diff

    # ===============================
    # Constraint Functions
    # ===============================
    
    def get_inequality_constraints(self):
        """
        Create list of inequality constraint functions for optimizer.
        Each constraint must be g(x) <= 0.
        
        Returns:
            list: constraint functions
        """
        constraints = []
        
        # Add parameter bound constraints
        constraints.extend(self._create_bound_constraints())
        
        return constraints
    
    def _create_bound_constraints(self):
        """Create parameter bound constraints."""
        constraints = []
        
        for i, name in enumerate(self.param_names):
            if name in self.bounds:
                lb, ub = self.bounds[name]
                
                # Lower bound constraint: lb - x[i] <= 0
                def make_lb_constraint(idx, bound):
                    return lambda x: bound - x[idx]
                
                # Upper bound constraint: x[i] - ub <= 0
                def make_ub_constraint(idx, bound):
                    return lambda x: x[idx] - bound
                
                constraints.append(make_lb_constraint(i, lb))
                constraints.append(make_ub_constraint(i, ub))
        
        return constraints
    
    def _create_trajectory_constraints(self):
        """Create trajectory-specific constraints here if unique additions wanted."""
        constraints = []
        
        return constraints
    
    def constraint_functions(self, params_array):
        """
        Evaluate all constraints for given parameter array.
        
        Args:
            params_array: flat array of optimization parameters
            
        Returns:
            array: constraint values (should be <= 0)
        """
        params_dict = self.array_to_parameters(params_array)
        constraints = []
        
        # Domain boundary constraints
        constraints.extend(self._evaluate_domain_constraints(params_dict))
        
        # Optional: Add other constraint evaluations here
        
        return anp.array(constraints)
    
    def _evaluate_domain_constraints(self, params_dict):
        """Evaluate constraints to keep trajectories within domain."""
        constraints = []
        
        # Check sawtooth trajectory bounds
        if 'sawtooth_A' in params_dict:
            y0 = self.initial_params.get('sawtooth_y0', 0.00125)
            A = params_dict['sawtooth_A']
            
            # Maximum y constraint
            constraints.append((y0 + A) - self.model.Ly)
            # Minimum y constraint  
            constraints.append(-(y0 - A))
        
        # Check swirl trajectory bounds
        if 'swirl_A' in params_dict:
            y0 = self.initial_params.get('swirl_y0', 0.00125)
            A = params_dict['swirl_A']
            
            # Maximum y constraint
            constraints.append((y0 + A) - self.model.Ly)
            # Minimum y constraint
            constraints.append(-(y0 - A))
        
        # Similar for second laser if present
        if 'sawtooth2_A' in params_dict:
            y0 = self.initial_params.get('sawtooth2_y0', 0.00125)
            A = params_dict['sawtooth2_A']
            constraints.append((y0 + A) - self.model.Ly)
            constraints.append(-(y0 - A))
            
        if 'swirl2_A' in params_dict:
            y0 = self.initial_params.get('swirl2_y0', 0.00125)
            A = params_dict['swirl2_A']
            constraints.append((y0 + A) - self.model.Ly)
            constraints.append(-(y0 - A))
        
        return constraints

    def _calculate_min_laser_distance(self, laser_params):
        """
        Calculate minimum distance between lasers over simulation time.
        
        Args:
            laser_params: tuple of laser configurations
            
        Returns:
            float: minimum distance between lasers
        """
        if not isinstance(laser_params, tuple) or len(laser_params) != 2:
            return float('inf')  # No distance constraint for single laser
        
        laser1, laser2 = laser_params
        
        # Estimate simulation time based on slowest laser
        v1 = laser1['params']['v']
        v2 = laser2['params']['v']
        slowest_v = anp.minimum(v1, v2)
        total_time = (self.x_range[1] - self.x_range[0]) / slowest_v
        
        # Sample distances over time
        n_samples = 500
        t_samples = anp.linspace(0, total_time, n_samples)
        distances = anp.zeros(n_samples)
        
        for i, t in enumerate(t_samples):
            x1, y1, _, _ = self._get_trajectory_point(t, laser1)
            x2, y2, _, _ = self._get_trajectory_point(t, laser2)
            distances[i] = anp.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        return anp.min(distances)

    # ===============================
    # Optimization Methods
    # ===============================
    
    def optimize(self, objective_type='standard', method='SLSQP', max_iterations=100):
        """
        Run optimization using scipy's minimize.
        
        Args:
            objective_type: type of objective function to use
            method: optimization method
            max_iterations: maximum number of iterations
            
        Returns:
            tuple: (optimization_result, optimized_parameters)
        """
        return self.optimize_with_scipy(objective_type, method, max_iterations)
    
    def optimize_with_scipy(self, objective_type='standard', method='SLSQP', max_iterations=100):
        """
        Run optimization using scipy's minimize with selected objective and constraints.
        
        Args:
            objective_type: objective function type
            method: scipy optimization method
            max_iterations: maximum iterations
            
        Returns:
            tuple: (result, optimized_params)
        """
        # Available objective functions
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
        
        # Set up constraints
        constraints = None
        if method.upper() in ['SLSQP', 'COBYLA']:
            constraints = [{
                'type': 'ineq',
                'fun': lambda x: -self.constraint_functions(x)  # Convert to <= 0 form
            }]
        
        # Set up bounds
        bounds_list = [(self.bounds[name][0], self.bounds[name][1]) 
                      for name in self.param_names if name in self.bounds]
        
        print(f"Starting optimization with {objective_type} objective using {method}")
        
        # Run optimization
        result = minimize(
            objective_func,
            initial_params_array,
            method=method,
            constraints=constraints,
            bounds=bounds_list if bounds_list else None,
            options={
                'maxiter': max_iterations,
                'disp': True,
                'xtol': 1e-10,
                'ftol': 1e-10
            }
        )
        
        # Convert result back to parameter dictionary
        optimized_params = self.array_to_parameters(result.x)
        
        # Print optimization summary
        self._print_optimization_summary(result, optimized_params)
        
        return result, optimized_params
    
    def _print_optimization_summary(self, result, optimized_params):
        """Print optimization results summary."""
        print("\nOptimization complete.")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")
        
        print("\nOptimized Parameters:")
        for name, value in optimized_params.items():
            print(f"  {name}: {value:.6f}")

    # ===============================
    # Analysis Methods
    # ===============================
    
    def perform_sensitivity_analysis(self, best_params, objective_type='standard'):
        """
        Perform sensitivity analysis for each parameter around optimized solution.
        
        Args:
            best_params: dictionary of optimized parameters
            objective_type: objective function to use for analysis
            
        Returns:
            dict: sensitivity data for each parameter
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
            sensitivity_data[name] = self._calculate_parameter_sensitivity(
                name, i, best_params_array, base_obj, objective_func
            )
        
        # Print ranked sensitivity results
        self._print_sensitivity_ranking(sensitivity_data)
        
        return sensitivity_data
    
    def _calculate_parameter_sensitivity(self, name, index, base_params, base_obj, objective_func):
        """Calculate sensitivity for a single parameter."""
        # Perturbation amount (5% of parameter value)
        perturb_amount = max(base_params[index] * 0.05, 1e-6)
        
        # +5% perturbation
        params_plus = base_params.copy()
        params_plus[index] += perturb_amount
        obj_plus = objective_func(params_plus)
        
        # -5% perturbation
        params_minus = base_params.copy()
        params_minus[index] -= perturb_amount
        obj_minus = objective_func(params_minus)
        
        # Calculate sensitivity (finite difference approximation)
        sensitivity_plus = (obj_plus - base_obj) / perturb_amount
        sensitivity_minus = (base_obj - obj_minus) / perturb_amount
        
        # Average sensitivity
        avg_sensitivity = (sensitivity_plus + sensitivity_minus) / 2
        
        # Relative sensitivity
        rel_sensitivity = avg_sensitivity * base_params[index] / base_obj if base_obj != 0 else 0
        
        print(f"Parameter: {name}")
        print(f"  Value: {base_params[index]:.6f}")
        print(f"  Absolute Sensitivity: {avg_sensitivity:.6e}")
        print(f"  Relative Sensitivity: {rel_sensitivity:.6e}")
        
        return {
            'value': base_params[index],
            'sensitivity': avg_sensitivity,
            'rel_sensitivity': rel_sensitivity
        }
    
    def _print_sensitivity_ranking(self, sensitivity_data):
        """Print parameters ranked by sensitivity."""
        sorted_params = sorted(
            sensitivity_data.items(),
            key=lambda x: abs(x[1]['rel_sensitivity']),
            reverse=True
        )
        
        print("\nParameters ranked by sensitivity:")
        for i, (name, data) in enumerate(sorted_params):
            print(f"{i+1}. {name}: {abs(data['rel_sensitivity']):.6e}")
    
 
# ===============================
# 3. Visualization Class
# ===============================

class Visualization:
    """
    Visualization class for LPBF heat transfer simulation results.
    
    Provides methods for comparing simulations, generating animations,
    and visualizing temperature fields and laser trajectories.
    """
    
    def __init__(self, model):
        """
        Initialize visualization with heat transfer model.
        
        Args:
            model: HeatTransferModel instance
        """
        self.model = model
        
        # Define thermal colormap for consistent visualization
        self.thermal_colors = [(0, 0, 0.3), (0, 0, 1), (0, 1, 0), 
                              (1, 1, 0), (1, 0, 0), (1, 1, 1)]
        self.cmap_temp = LinearSegmentedColormap.from_list('thermal', self.thermal_colors)
    
    # ===============================
    # Raster Scan Generation
    # ===============================
    
    def generate_raster_scan(self, x_start, x_end, y_start, y_end, n_lines, speed):
        """
        Generate raster scan path covering specified area at given speed.
        
        Args:
            x_start, x_end: x-range for scan (meters)
            y_start, y_end: y-range for scan (meters)
            n_lines: number of scan lines
            speed: scan speed (m/s)
            
        Returns:
            tuple: (path_array, speed) where path_array is Nx2 array of (x,y) points
        """
        x_vals = np.linspace(x_start, x_end, self.model.nx)
        y_vals = np.linspace(y_start, y_end, n_lines)
        
        path = []
        for i, y in enumerate(y_vals):
            # Alternate scan direction for each line
            xs = x_vals if i % 2 == 0 else x_vals[::-1]
            for x in xs:
                path.append((x, y))
                
        return np.array(path), speed
    
    def _create_raster_trajectory_function(self, raster_path, raster_speed, x_end, x_start=0):
        def raster_trajectory(t, params):
            v = raster_speed  # Actual speed along the raster path
            noise_sigma = params.get('noise_sigma', 0.0)
        
            # Limit noise to reasonable bounds
            max_noise = 0.00005
            noise_sigma = anp.clip(noise_sigma, 0.0, max_noise)
        
            hatch_spacing = 0.00008
            y_start = 0.0002
            y_end = 0.0008

            # Arc-length based raster scan
            target_distance = v * t  # Total distance traveled along raster path
        
            horizontal_distance = x_end - x_start
            vertical_distance = hatch_spacing
        
            # Each pass consists of horizontal scan + vertical move
            pass_distance = horizontal_distance + vertical_distance
        
            # Current pass
            pass_num = int(target_distance / pass_distance)
            distance_in_pass = target_distance - pass_num * pass_distance
        
            y_current = y_start + pass_num * hatch_spacing

            # Check if end reached
            max_passes = int((y_end - y_start) / hatch_spacing) + 2
        
            if pass_num >= max_passes or y_current > y_end:
                final_y = min(y_end, y_start + (max_passes-1) * hatch_spacing)
                return x_end, final_y, 1.0, 0.0

            # Apply random noise
            if noise_sigma > 0:
                noise_x = noise_sigma * np.random.randn()
                noise_y = noise_sigma * np.random.randn()
            else:
                noise_x, noise_y = 0.0, 0.0

            if distance_in_pass <= horizontal_distance:
                # Horizontal scanning phase
                if pass_num % 2 == 0:
                    # Left to right
                    x = x_start + distance_in_pass + noise_x
                    y = y_current + noise_y
                    tx, ty = 1.0, 0.0
                else:
                    # Right to left
                    x = x_end - distance_in_pass + noise_x
                    y = y_current + noise_y
                    tx, ty = -1.0, 0.0
            
                x = max(x_start, min(x, x_end))
            else:
                # Vertical transition
                vertical_progress = distance_in_pass - horizontal_distance
            
                if pass_num % 2 == 0:
                    x = x_end + noise_x
                else:
                    x = x_start + noise_x
            
                y = y_current + vertical_progress + noise_y
                tx, ty = 0.0, 1.0

            # Apply boundary constraints
            x = max(x_start, min(x, x_end))
            y = max(0, min(y, y_end))

            return x, y, tx, ty
        return raster_trajectory
    
    # ===============================
    # Simulation Comparison
    # ===============================
    
    def compare_simulations(self, initial_params, optimized_params, use_gaussian=True, 
                        y_crop=None, show_full_domain=True):
        """
        Compare temperature and gradient fields for raster scan vs. optimized scan.

        Args:
            initial_params: dictionary of initial parameters
            optimized_params: dictionary of optimized parameters
            use_gaussian: whether to use Gaussian heat source
            y_crop: y-range for cropping display (unused, kept for compatibility)
            show_full_domain: whether to show full domain (unused, kept for compatibility)
    
        Returns:
            matplotlib.figure.Figure: comparison plot
        """
        # Setup raster scan reference
        raster_config = self._setup_raster_scan()
        T_raster, grad_raster, raster_positions, raster_temporal_stats = self._simulate_raster_scan(raster_config)

        # Setup and simulate optimized scan
        T_opt, grad_opt, laser_paths, opt_temporal_stats = self._simulate_optimized_scan(
            initial_params, optimized_params, use_gaussian
        )

        # Create comparison visualization
        fig = self._create_comparison_plots(
            T_raster, grad_raster, raster_positions, raster_temporal_stats,
            T_opt, grad_opt, laser_paths, optimized_params, opt_temporal_stats
        )

        plt.tight_layout()
        plt.show()
        return fig
    
    def _setup_raster_scan(self):
        """Setup raster scan configuration with 1.5mm margins."""
        x_start = 0.0015
        x_end = self.model.Lx - 0.0015
        y_start, y_end = 0.0002, 0.0008
        hatch_spacing = 0.00008  # 80 um
        n_lines = int(np.floor((y_end - y_start) / hatch_spacing)) + 1
        raster_speed = 0.7  # 700 mm/s
    
        raster_path, _ = self.generate_raster_scan(
            x_start, x_end, y_start, y_end, n_lines, raster_speed
        )
    
        heat_params = {'Q': 200.0, 'r0': 0.00004}
    
        return {
            'path': raster_path,
            'speed': raster_speed,
            'x_start': x_start,
            'x_end': x_end,
            'heat_params': heat_params
        }
    
    def _simulate_raster_scan(self, raster_config):
        """
        Simulate raster scan and extract results.
        """
        # Create raster trajectory function using the mathematical approach
        raster_trajectory = self._create_raster_trajectory_function(
            raster_config['path'], raster_config['speed'], 
            raster_config['x_end'], raster_config['x_start']
        )

        # Temporarily override trajectory functions
        orig_sawtooth = self.model.sawtooth_trajectory
        orig_swirl = self.model.swirl_trajectory

        self.model.sawtooth_trajectory = raster_trajectory
        self.model.swirl_trajectory = raster_trajectory

        try:
            raster_heat_params = {'Q': 200.0, 'r0': 0.00004}
        
            raster_laser_params = {'type': 'sawtooth', 'params': {
                'v': raster_config['speed'], 'A': 0, 'y0': 0, 
                'period': 1, 'noise_sigma': 0
            }}

            # Run simulation to update temporal_grad_stats
            T_raster = self.model.simulate(
                (raster_laser_params, raster_heat_params), 
                use_gaussian=True,
                nt=4000
            )

            # Calculate gradient from final temperature field
            _, _, grad_raster = self.model.compute_temperature_gradients(T_raster)
        
            # Get temporal statistics collected during simulation
            raster_temporal_stats = self.model.temporal_grad_stats
        
            # DIAGNOSTIC: Print to verify values
            print(f"\nRaster Scan Statistics:")
            print(f"  Final max gradient: {np.max(grad_raster):.2e} °C/m")
            print(f"  Temporal max gradient: {raster_temporal_stats['max_over_time']:.2e} °C/m")
            print(f"  Ratio (temporal/final): {raster_temporal_stats['max_over_time']/np.max(grad_raster):.2f}")

            raster_positions = self._extract_laser_positions_mathematical(
                raster_trajectory, raster_config['speed']
            )

        finally:
            # Restore original trajectory functions
            self.model.sawtooth_trajectory = orig_sawtooth
            self.model.swirl_trajectory = orig_swirl

        return T_raster, grad_raster, raster_positions, raster_temporal_stats
    
    def _extract_laser_positions_mathematical(self, trajectory_func, speed):
        """Extract laser positions using mathematical raster trajectory."""
        nt = self.model.nt if hasattr(self.model, 'nt') and self.model.nt else 3000
        dt = self.model.dt if hasattr(self.model, 'dt') else 1e-5
        t_points = np.linspace(0, nt * dt, nt)
    
        positions = []
        for t in t_points:
            x, y, _, _ = trajectory_func(t, {'v': speed})
            positions.append((x * 1000, y * 1000))  # Convert to mm
        
        return positions
    
    def _simulate_optimized_scan(self, initial_params, optimized_params, use_gaussian):
        """
        Simulate optimized scan and extract results.
        NOW uses cached trajectories for consistent paths.
        """
        # Setup optimizer and unpack parameters
        temp_optimizer = TrajectoryOptimizer(
            self.model, initial_params=initial_params, bounds=None
        )
        optimized_array = temp_optimizer.parameters_to_array(optimized_params)
        laser_opt, heat_opt = temp_optimizer.unpack_parameters(optimized_array)

        # Determine heat source type
        if isinstance(heat_opt, (list, tuple)):
            use_gaussian_opt = 'r0' in heat_opt[0] or 'sigma_x' in heat_opt[0]
        else:
            use_gaussian_opt = 'r0' in heat_opt or 'sigma_x' in heat_opt

        # Run simulation to create cached trajectories
        T_opt = self.model.simulate((laser_opt, heat_opt), use_gaussian=use_gaussian_opt)

        # Extract laser paths from the cached trajectories created during simulation
        laser_paths = {'straight': [], 'sawtooth': [], 'swirl': []}

        # Define boundary limits
        x_start = 0.0015
        x_end = self.model.Lx - 0.0015

        if hasattr(self.model, '_cached_trajectories'):
            cached_traj = self.model._cached_trajectories
    
            # Check if dual laser config (list of trajectory lists)
            # or single laser config (single trajectory list)
            if isinstance(laser_opt, tuple):
                # Multiple lasers - cached_traj is a list of trajectory lists
                for i, laser_config in enumerate(laser_opt):
                    traj_type = laser_config['type']
                    trajectory_points = cached_traj[i]  # Get the i-th trajectory list
            
                    # Convert to mm and store active points (before boundary)
                    for x, y, _, _ in trajectory_points:
                        if x < x_end:
                            laser_paths[traj_type].append((x * 1000, y * 1000))
            else:
                # Single laser - cached_traj is a single trajectory list
                traj_type = laser_opt['type']
        
                # Convert to mm and store active points (before boundary)
                for x, y, _, _ in cached_traj:
                    if x < x_end:
                        laser_paths[traj_type].append((x * 1000, y * 1000))

        # Calculate gradient from final temperature field
        # For optimized scan, use temperature at laser finish (not after full diffusion)
        T_opt_display = self.model.temporal_grad_stats.get('T_at_laser_finish', T_opt)
        _, _, grad_opt = self.model.compute_temperature_gradients(T_opt_display)
        opt_temporal_stats = self.model.temporal_grad_stats

        return T_opt_display, grad_opt, laser_paths, opt_temporal_stats
    
    def _track_positions_during_simulation(self, laser_opt):
        """
        Track laser positions using same time evolution as the simulation.
        This must be called after reset() to use the same initial noise state.
        Only records positions while laser is actively scanning (not clamped at boundary).
        """
        nt = 4000  # Must match self.model.nt in simulate()
        dt = self.model.dt
        t_points = np.linspace(0, nt * dt, nt)

        paths = {'straight': [], 'sawtooth': [], 'swirl': []}

        # Handle single or dual laser configuration
        laser_list = laser_opt if isinstance(laser_opt, tuple) else [laser_opt]
    
        # Domain boundaries
        x_start = 0.0015
        x_end = self.model.Lx - 0.0015
    
        for laser in laser_list:
            x_list, y_list = [], []
            scan_complete = False
        
            for t in t_points:
                # Call trajectory function to maintain noise state sync
                if laser['type'] == 'straight':
                    x, y, _, _ = self.model.straight_trajectory(t, laser['params'])
                elif laser['type'] == 'sawtooth':
                    x, y, _, _ = self.model.sawtooth_trajectory(t, laser['params'])
                else:  # Swirl
                    x, y, _, _ = self.model.swirl_trajectory(t, laser['params'])
        
                # Only record position if laser hasn't reached end
                if not scan_complete:
                    x_list.append(x * 1000)  # Convert to mm
                    y_list.append(y * 1000)
                
                    # Check if laser has reached end boundary
                    # Use a small tolerance to detect when clamping starts
                    if x >= (x_end - 1e-6):
                        scan_complete = True
        
            paths[laser['type']].append((x_list, y_list))

        return paths
    
    def _extract_laser_positions(self, trajectory_func, speed):
        """Extract laser positions over time for raster scan."""
        nt = self.model.nt if hasattr(self.model, 'nt') and self.model.nt else 3000
        dt = self.model.dt if hasattr(self.model, 'dt') else 1e-5
        t_points = np.linspace(0, nt * dt, nt)
        
        positions = []
        for t in t_points:
            x, y, _, _ = trajectory_func(t, {'v': speed})
            positions.append((x * 1000, y * 1000))  # Convert to mm
            
        return positions
    
    def _extract_optimized_paths(self, laser_params):
        """Extract optimized laser paths for plotting - now includes actual noisy positions."""
        nt = self.model.nt if hasattr(self.model, 'nt') and self.model.nt else 3000
        dt = self.model.dt if hasattr(self.model, 'dt') else 1e-5
        t_points = np.linspace(0, nt * dt, nt)

        paths = {'straight': [], 'sawtooth': [], 'swirl': []}

        # Handle single or dual laser configuration
        laser_list = laser_params if isinstance(laser_params, tuple) else [laser_params]

        for laser in laser_list:
            x_list, y_list = [], []
    
            # Reset noise state before extracting path to match simulation
            if laser['type'] == 'sawtooth':
                if hasattr(self.model, '_noise_state'):
                    self.model._noise_state = {'x': 0.0, 'y': 0.0, 'last_t': 0.0}
            elif laser['type'] == 'swirl':
                if hasattr(self.model, '_noise_state_swirl'):
                    self.model._noise_state_swirl = {'x': 0.0, 'y': 0.0, 'last_t': 0.0}
            elif laser['type'] == 'straight':
                if hasattr(self.model, '_noise_state_straight'):
                    self.model._noise_state_straight = {'x': 0.0, 'y': 0.0, 'last_t': 0.0}
    
            for t in t_points:
                # Get noisy position
                if laser['type'] == 'straight':
                    x, y, _, _ = self.model.straight_trajectory(t, laser['params'])
                elif laser['type'] == 'sawtooth':
                    x, y, _, _ = self.model.sawtooth_trajectory(t, laser['params'])
                else:  # Swirl
                    x, y, _, _ = self.model.swirl_trajectory(t, laser['params'])
        
                x_list.append(x * 1000)  # Convert to mm
                y_list.append(y * 1000)
    
            paths[laser['type']].append((x_list, y_list))

        return paths
    
    def _create_comparison_plots(self, T_raster, grad_raster, raster_positions, raster_temporal_stats,
                                T_opt, grad_opt, laser_paths, optimized_params, opt_temporal_stats):
        """
        Create 2x2 comparison plot layout.
        """
        fig = plt.figure(figsize=(16, 8))

        # Create subplots
        ax1 = fig.add_subplot(2, 2, 1)  # Raster temperature
        ax2 = fig.add_subplot(2, 2, 2)  # Optimized temperature
        ax3 = fig.add_subplot(2, 2, 3)  # Raster gradient
        ax4 = fig.add_subplot(2, 2, 4)  # Optimized gradient

        extent = [0, self.model.Lx * 1000, 0, self.model.Ly * 1000]

        # Plot temperature fields
        self._plot_temperature_field(ax1, T_raster, extent, "Raster Scan Temperature")
        self._plot_temperature_field(ax2, T_opt, extent, "Optimized Scan Temperature")

        # Plot gradient fields
        self._plot_gradient_field(ax3, grad_raster, extent, "Raster Scan Temperature Gradient")
        self._plot_gradient_field(ax4, grad_opt, extent, "Optimized Scan Temperature Gradient")

        # Add laser paths and annotations
        self._add_raster_paths(ax1, ax3, raster_positions)
        self._add_optimized_paths(ax2, ax4, laser_paths, optimized_params)

        # Add melt pool contours
        self._add_melt_pool_contours(ax1, T_raster, extent)
        self._add_melt_pool_contours(ax2, T_opt, extent)

        # Show peak gradients that occurred during simulation
        self._add_gradient_statistics(ax1, grad_raster, temporal_stats=raster_temporal_stats, 
                                      is_raster=True, label_type='temporal')
        self._add_gradient_statistics(ax2, grad_opt, temporal_stats=opt_temporal_stats, 
                                      is_raster=False, label_type='temporal')

        return fig
    
    def _plot_temperature_field(self, ax, T_field, extent, title):
        """Plot temperature field on given axis."""
        im = ax.imshow(T_field, extent=extent, origin='lower', 
                      cmap=self.cmap_temp, aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        plt.colorbar(im, ax=ax, label='Temperature (°C)')
        return im
    
    def _plot_gradient_field(self, ax, grad_field, extent, title):
        """Plot gradient field on given axis."""
        # Convert gradient field from °C/m to °C/mm for display
        grad_field_mm = grad_field / 1000
    
        grad_cmap = plt.get_cmap('viridis')
        im = ax.imshow(grad_field_mm, extent=extent, origin='lower', 
                      cmap=grad_cmap, aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        plt.colorbar(im, ax=ax, label='|∇T| (°C/mm)')
        return im
    
    def _add_raster_paths(self, temp_ax, grad_ax, raster_positions):
        """Add raster scan paths to temperature and gradient plots."""
        raster_x = [pos[0] for pos in raster_positions]
        raster_y = [pos[1] for pos in raster_positions]
    
        # Plot the scan path lines
        temp_ax.plot(raster_x, raster_y, color='black', linewidth=1, 
                    label='Raster Laser Path')
        grad_ax.plot(raster_x, raster_y, color='black', linewidth=1, 
                    label='Raster Laser Path')
    
    
        temp_ax.legend()
        grad_ax.legend()
    
    
    def _add_optimized_paths(self, temp_ax, grad_ax, laser_paths, optimized_params):
        """Add optimized laser paths to temperature and gradient plots."""
        colors = ['red', 'black']
        laser_types = ['straight', 'sawtooth', 'swirl']

        laser_assignments = []
    
        for laser_type in laser_types:
            if laser_paths[laser_type]:  # If this laser type has points
                # Extract x and y coordinates from the list of (x, y) tuples
                x_list = [point[0] for point in laser_paths[laser_type]]
                y_list = [point[1] for point in laser_paths[laser_type]]
            
                # Get color for this laser
                color_idx = len(laser_assignments)
                if color_idx >= len(colors):
                    # Fallback color
                    color = 'cyan'
                else:
                    color = colors[color_idx]
            
                # Plot actual simulated path
                temp_ax.plot(x_list, y_list, color=color, linewidth=1)
                grad_ax.plot(x_list, y_list, color=color, linewidth=1)
            
                # Create legend label
                heat_source_num = len(laser_assignments) + 1
                label = f"Heat Source {heat_source_num}: {laser_type.capitalize()}"
            
                # Add legend entry
                temp_ax.plot([], [], color=color, linewidth=2, label=label)
                grad_ax.plot([], [], color=color, linewidth=2, label=label)
            
                laser_assignments.append(laser_type)

        temp_ax.legend()
        grad_ax.legend()
    
    def _add_melt_pool_contours(self, ax, T_field, extent):
        """Add melt pool contours to temperature plot."""
        T_melt = self.model.material['T_melt']
        melt_mask = T_field > T_melt
        ax.contour(melt_mask, levels=[0.5], colors='cyan', linewidths=2, 
                  extent=extent, origin='lower')
    

    def _add_gradient_statistics(self, ax, grad_field, temporal_stats=None, is_raster=False, label_type='temporal'):
        """Add gradient statistics annotation to plot with correct unit conversions."""
    
        if label_type == 'temporal' and temporal_stats:
            # Temporal statistics stored in °C/m, convert to °C/mm for display
            max_grad_temporal = temporal_stats['max_over_time'] / 1000
            mean_grad_temporal = temporal_stats['mean_over_time'] / 1000
        
            stats_text = (
                f"Peak Over Simulation:\n"
                f"Max ∇T: {max_grad_temporal:.2e} °C/mm\n"
                f"Mean ∇T: {mean_grad_temporal:.2e} °C/mm"
            )
        else:
            # Final state statistics - grad_field is in °C/m, convert to °C/mm
            max_grad_final = np.max(grad_field) / 1000  
            mean_grad_final = np.mean(grad_field) / 1000
        
            stats_text = (
                f"Final State:\n"
                f"Max ∇T: {max_grad_final:.2e} °C/mm\n"
                f"Mean ∇T: {mean_grad_final:.2e} °C/mm"
            )

        # Position text box
        x_pos = 0.02
        y_pos = 0.98

        ax.text(x_pos, y_pos, stats_text, transform=ax.transAxes, fontsize=9, 
               color='white', verticalalignment='top',
               bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))

    def plot_temporal_gradients(self, raster_temporal_stats, opt_temporal_stats):
        """
        Plot temporal evolution of gradient statistics.
    
        Args:
            raster_temporal_stats: temporal gradient statistics for raster scan
            opt_temporal_stats: temporal gradient statistics for optimized scan
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
        # Create separate time arrays for each scan (they may have different lengths)
        dt = self.model.dt
        nt_raster = len(raster_temporal_stats['max_gradients'])
        nt_opt = len(opt_temporal_stats['max_gradients'])
        time_points_raster = np.arange(nt_raster) * dt
        time_points_opt = np.arange(nt_opt) * dt
    
        # Plot maximum gradients over time
        ax1.plot(time_points_raster, np.array(raster_temporal_stats['max_gradients'])/1000, 
                 'b-', label='Raster Scan', linewidth=2)
        ax1.plot(time_points_opt, np.array(opt_temporal_stats['max_gradients'])/1000, 
                 'r-', label='Optimized Scan', linewidth=2)
        ax1.set_ylabel('Maximum ∇T (°C/mm)')
        ax1.set_title('Maximum Temperature Gradient Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
        # Plot mean gradients over time
        ax2.plot(time_points_raster, np.array(raster_temporal_stats['mean_gradients'])/1000, 
                 'b-', label='Raster Scan', linewidth=2)
        ax2.plot(time_points_opt, np.array(opt_temporal_stats['mean_gradients'])/1000, 
                 'r-', label='Optimized Scan', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Mean ∇T (°C/mm)')
        ax2.set_title('Mean Temperature Gradient Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.show()
    
        # Print summary statistics
        print("\n=== Temporal Gradient Analysis ===")
        print(f"Raster Scan:")
        print(f"  Max gradient over all time: {raster_temporal_stats['max_over_time']/1000:.2e} °C/mm")
        print(f"  Mean gradient over all time: {raster_temporal_stats['mean_over_time']/1000:.2e} °C/mm")
        print(f"Optimized Scan:")
        print(f"  Max gradient over all time: {opt_temporal_stats['max_over_time']/1000:.2e} °C/mm")
        print(f"  Mean gradient over all time: {opt_temporal_stats['mean_over_time']/1000:.2e} °C/mm")
    
        # Calculate improvements
        max_improvement = ((raster_temporal_stats['max_over_time'] - opt_temporal_stats['max_over_time']) / 
                          raster_temporal_stats['max_over_time'] * 100)
        mean_improvement = ((raster_temporal_stats['mean_over_time'] - opt_temporal_stats['mean_over_time']) / 
                          raster_temporal_stats['mean_over_time'] * 100)
    
        print(f"\nImprovements:")
        print(f"  Max gradient reduction: {max_improvement:.2f}%")
        print(f"  Mean gradient reduction: {mean_improvement:.2f}%")
    
        return fig
    
    # ===============================
    # Animation Methods
    # ===============================
    
    def animate_optimization_results(self, model, initial_params, optimized_params, 
                                   fps=30, show_full_domain=True, y_crop=None, 
                                   save_snapshots=True, snapshot_interval=0.01, 
                                   snapshot_dir="animation_snapshots"):
        """
        Animate temperature evolution for raster and optimized scans.
        Shows laser positions and melt pool outlines at each frame.
        
        Args:
            model: HeatTransferModel instance
            initial_params: initial parameter dictionary
            optimized_params: optimized parameter dictionary  
            fps: animation frame rate
            show_full_domain: show full domain (kept for compatibility)
            y_crop: y-range for cropping (kept for compatibility)
            save_snapshots: whether to save snapshot images
            snapshot_interval: time interval for snapshots
            snapshot_dir: directory for snapshot images
            
        Returns:
            matplotlib.animation.FuncAnimation: animation object
        """
        # Setup animation components
        animation_config = self._setup_animation_config(
            initial_params, optimized_params, save_snapshots, 
            snapshot_interval, snapshot_dir
        )
        
        # Create animation figure and axes
        fig, axes_config = self._create_animation_figure()
        
        # Initialize animation data
        animation_data = self._initialize_animation_data(animation_config)
        
        # Create update function
        update_func = self._create_animation_update_function(
            animation_config, axes_config, animation_data
        )
        
        # Create and return animation
        nt = animation_config['nt']
        ani = FuncAnimation(fig, update_func, frames=nt, interval=1000/fps, blit=False)
        
        plt.show()
        return ani
    
    def _setup_animation_config(self, initial_params, optimized_params, 
                            save_snapshots, snapshot_interval, snapshot_dir):
        """Setup configuration for animation with synchronized timing."""
        # Raster scan configuration
        raster_config = self._setup_raster_scan()
        raster_speed = raster_config['speed']
        raster_config['speed'] = raster_speed
        # Optimized scan configuration
        temp_optimizer = TrajectoryOptimizer(
            self.model, initial_params=initial_params, bounds=None
        )
        optimized_array = temp_optimizer.parameters_to_array(optimized_params)
        laser_opt, heat_opt = temp_optimizer.unpack_parameters(optimized_array)

        # Calculate optimized scan speed to synchronize timing
        # Extract the actual scan speed from optimized parameters
        if isinstance(laser_opt, tuple):
            # Multiple lasers - use slowest speed for timing
            opt_speeds = [laser['params']['v'] for laser in laser_opt]
            opt_speed = min(opt_speeds)
        else:
            # Single laser
            opt_speed = laser_opt['params']['v']

        # Determine frame count from cached trajectories
        dt = self.model.dt
        if hasattr(self.model, '_cached_trajectories'):
            cached_traj = self.model._cached_trajectories
            if isinstance(cached_traj, list) and len(cached_traj) > 0:
                if isinstance(cached_traj[0], list):
                    # Multiple lasers - use maximum length
                    nt = max(len(traj) for traj in cached_traj)
                else:
                    # Single laser
                    nt = len(cached_traj)
            else:
                nt = 4000
        else:
            nt = 4000
    
        # Calculate scan distances and times
        x_start = raster_config['x_start']
        x_end = raster_config['x_end']
        scan_distance = x_end - x_start  # Horizontal distance per pass
        
        # Calculate total raster scan time including all passes
        # Raster scan parameters from _setup_raster_scan
        y_start, y_end = 0.0002, 0.0008
        hatch_spacing = 0.00008
        n_lines = int(np.floor((y_end - y_start) / hatch_spacing)) + 1
        
        # Each pass includes horizontal scan and vertical move
        horizontal_distance = scan_distance
        vertical_distance = hatch_spacing
        distance_per_pass = horizontal_distance + vertical_distance
        total_raster_distance = n_lines * distance_per_pass
        
        raster_time = total_raster_distance / raster_speed
        
        # Calculate optimized scan actual path length
        if hasattr(self.model, '_cached_trajectories'):
            cached_traj = self.model._cached_trajectories
            if isinstance(cached_traj, list) and isinstance(cached_traj[0], list):
                # Multiple lasers - calculate first laser path length
                opt_path_length = 0.0
                for i in range(len(cached_traj[0]) - 1):
                    x1, y1, _, _ = cached_traj[0][i]
                    x2, y2, _, _ = cached_traj[0][i + 1]
                    opt_path_length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            else:
                # Single laser
                opt_path_length = 0.0
                for i in range(len(cached_traj) - 1):
                    x1, y1, _, _ = cached_traj[i]
                    x2, y2, _, _ = cached_traj[i + 1]
                    opt_path_length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            opt_time = opt_path_length / opt_speed
        else:
            opt_path_length = scan_distance
            opt_time = scan_distance / opt_speed

        # Animation duration determined by the timesteps
        animation_duration = nt * dt

        print(f"\nAnimation Timing:")
        print(f"  Raster scan speed: {raster_speed:.3f} m/s")
        print(f"  Optimized scan speed: {opt_speed:.3f} m/s")
        print(f"  Raster path length: {total_raster_distance*1000:.2f} mm")
        print(f"  Optimized path length: {opt_path_length*1000:.2f} mm (with noise)")
        print(f"  Raster scan time: {raster_time:.3f} s")
        print(f"  Optimized scan time: {opt_time:.3f} s")
        print(f"  Animation duration: {animation_duration:.3f} s")
        print(f"  Time step (dt): {dt:.6f} s")
        print(f"  Number of frames: {nt}")
        print(f"  Note: With noise, path is longer so animation takes more time steps")

        # Snapshot configuration
        snapshot_frames = set()
        if save_snapshots:
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_frames = set(
                int(i * nt * snapshot_interval) for i in range(int(1/snapshot_interval) + 1)
            )

        return {
            'raster_config': raster_config,
            'laser_opt': laser_opt,
            'heat_opt': heat_opt,
            'dt': dt,
            'nt': nt,
            'animation_duration': animation_duration,
            'raster_time': raster_time,
            'opt_time': opt_time,
            'raster_speed': raster_speed,
            'opt_speed': opt_speed,
            'snapshot_frames': snapshot_frames,
            'snapshot_dir': snapshot_dir,
            'save_snapshots': save_snapshots
        }
    
    def _check_scan_completion(self, t, scan_time, speed, x_start, x_end):
        """
        Check if scan has completed and return final position if so.
    
        Args:
            t: current time
            scan_time: total time for this scan
            speed: scan speed
            x_start, x_end: x-range
        
        Returns:
            tuple: (is_complete, final_x, final_y)
        """
        if t >= scan_time:
            # Hold at final position if finished
            return True, x_end, 0.0005  # Return to center y position
        return False, None, None

    def _create_animation_figure(self):
        """Create figure and axes for animation."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Create inset axes for melt pool visualization
        ax_inset_opt = inset_axes(ax2, width="30%", height="30%", loc='lower left',
                                 borderpad=2, bbox_to_anchor=(0.08, 0.05, 0.4, 0.4),
                                 bbox_transform=ax2.transAxes)
        ax_inset_opt.set_title("Melt Pool Outline", fontsize=8)
        ax_inset_opt.set_xticks([])
        ax_inset_opt.set_yticks([])
        ax_inset_opt.set_xlim(0, self.model.Lx * 1000)
        ax_inset_opt.set_ylim(0, self.model.Ly * 1000)
        
        return fig, {
            'ax1': ax1,
            'ax2': ax2,
            'ax_inset_opt': ax_inset_opt,
            'ax_inset_raster': None  # Will be created dynamically
        }
    
    def _initialize_animation_data(self, config):
        """Initialize temperature fields and other animation data."""
        T_raster = self.model.material['T0'] * np.ones((self.model.ny, self.model.nx))
        T_opt = self.model.material['T0'] * np.ones((self.model.ny, self.model.nx))
    
        # Extract laser positions and trajectories
        laser_positions = self._extract_animation_laser_positions(config)
    
        # Initialize path history arrays to track actual noisy positions
        return {
            'T_raster': T_raster,
            'T_opt': T_opt,
            'laser_positions': laser_positions,
            'temp_vmin': self.model.material['T0'],
            'temp_vmax': 5000,
            'extent': [0, self.model.Lx * 1000, 0, self.model.Ly * 1000],
            'raster_path_history': [],
            'opt_path_history': []
        }
    
    def _extract_animation_laser_positions(self, config):
        """Extract laser positions for animation using corrected trajectory functions."""
        nt = config['nt']
        dt = config['dt']
        t_points = np.linspace(0, nt * dt, nt)

        # Raster positions using corrected mathematical approach
        raster_trajectory = self._create_raster_trajectory_function(
            config['raster_config']['path'], 
            config['raster_config']['speed'],
            config['raster_config']['x_end'],
            config['raster_config']['x_start']
        )

        # Store the trajectory function for real-time calculation
        raster_path_points = []
        for t in t_points:
            x, y, _, _ = raster_trajectory(t, {'v': config['raster_config']['speed'], 'noise_sigma': 0.0})
            raster_path_points.append((x, y))

        return {
            'raster_trajectory': raster_trajectory,
            'raster_path': raster_path_points,
            'optimized_positions': []  # Not needed since we calculate positions on-the-fly
        }
    
    def _create_animation_update_function(self, config, axes_config, animation_data):
        """Create the animation update function."""
        def update(frame):
            # Clear axes
            axes_config['ax1'].cla()
            axes_config['ax2'].cla()
            
            # Update temperature fields
            self._update_temperature_fields(frame, config, animation_data)
            
            # Plot updated fields
            self._plot_animation_frame(frame, config, axes_config, animation_data)
            
            # Update melt pool insets
            self._update_melt_pool_insets(frame, axes_config, animation_data)
            
            # Save snapshot if needed
            if (config['save_snapshots'] and 
                frame in config['snapshot_frames']):
                self._save_animation_snapshot(frame, config)
            
            return []
        
        return update
    
    def _update_temperature_fields(self, frame, config, animation_data):
        """Update temperature fields for current animation frame using pre-calculated trajectories."""
        dt = config['dt']
        t = frame * dt

        # Initialise on first frame
        if frame == 0:
            animation_data['T_raster'] = self.model.material['T0'] * np.ones((self.model.ny, self.model.nx))
            animation_data['T_opt'] = self.model.material['T0'] * np.ones((self.model.ny, self.model.nx))
    
            # Use same cached trajectories from the simulation
            if hasattr(self.model, '_cached_trajectories'):
                print("Using cached trajectories from simulation")
                animation_data['opt_trajectory_cache'] = self.model._cached_trajectories
            else:
                print("WARNING: No cached trajectories found! Running simulation first...")
                # Run simulation to generate cache
                self.model.simulate(
                    (config['laser_opt'], config['heat_opt']),
                    use_gaussian=True
                )
                if hasattr(self.model, '_cached_trajectories'):
                    animation_data['opt_trajectory_cache'] = self.model._cached_trajectories
                else:
                    raise RuntimeError("Failed to generate trajectory cache")

        # === RASTER Scan Update ===
        S_raster_total = 0
    
        raster_complete = False
        if frame < len(animation_data['laser_positions']['raster_path']):
            # Raster scan still active - use current frame position
            x_r, y_r, _, _ = animation_data['laser_positions']['raster_trajectory'](
                t, {'v': config['raster_config']['speed'], 'noise_sigma': 0.0}
            )
        
            # Apply heat at current position
            heat_params = config['raster_config']['heat_params']
            S_r = self.model._gaussian_source(x_r, y_r, heat_params)
            S_raster_total += S_r
        else:
            # Raster has completed - no more heat input
            raster_complete = True

        # Heat equation update for raster
        lap_r = self.model._laplacian(animation_data['T_raster'])
        T_diff_r = animation_data['T_raster'] + dt * self.model.material['alpha'] * lap_r

        volume_factor = self.model.dx * self.model.dy * self.model.thickness
        source_increment_r = (dt * S_raster_total) / (volume_factor * self.model.heat_capacity)
        T_new_r = T_diff_r + source_increment_r
        animation_data['T_raster'] = self.model._apply_boundary_conditions(T_new_r)
    
        # === OPTIMIZED Scan Update ===
        S_opt_total = 0
        heat_opt = config['heat_opt']
        
        # Define boundary limits
        x_start_limit = 0.0015
        x_end_limit = self.model.Lx - 0.0015
    
        if isinstance(config['laser_opt'], tuple):
            # Multiple lasers - use cached trajectories
            for i, laser_config in enumerate(config['laser_opt']):
                # Get position from cache
                x, y, _, _ = animation_data['opt_trajectory_cache'][i][frame]
        
                # Check if laser is within valid range
                laser_active = x_start_limit <= x <= x_end_limit
        
                heat_params = heat_opt[i] if isinstance(heat_opt, (list, tuple)) else heat_opt

                if laser_active:
                    S = self.model._gaussian_source(x, y, heat_params)
                    S_opt_total += S
        else:
            # Single laser - use cached trajectory
            x, y, _, _ = animation_data['opt_trajectory_cache'][frame]
    
            # Check if laser is within valid range
            laser_active = x_start_limit <= x <= x_end_limit
    
            if laser_active:
                S_opt_total = self.model._gaussian_source(x, y, heat_opt)

        # Heat equation update for optimized scan
        lap_opt = self.model._laplacian(animation_data['T_opt'])
        T_diff_opt = animation_data['T_opt'] + dt * self.model.material['alpha'] * lap_opt

        source_increment_opt = (dt * S_opt_total) / (volume_factor * self.model.heat_capacity)
        T_new_opt = T_diff_opt + source_increment_opt

        animation_data['T_opt'] = self.model._apply_boundary_conditions(T_new_opt)
    
    
    def _plot_animation_frame(self, frame, config, axes_config, animation_data):
        """Plot current animation frame with proper laser positions."""
        extent = animation_data['extent']
        vmin, vmax = animation_data['temp_vmin'], animation_data['temp_vmax']
    
        # Plot temperature fields
        im1 = axes_config['ax1'].imshow(
            animation_data['T_raster'], extent=extent, origin='lower',
            cmap=self.cmap_temp, aspect='auto', vmin=vmin, vmax=vmax
        )
        axes_config['ax1'].set_title('Raster Scan Temperature')
        axes_config['ax1'].set_xlabel('X (mm)')
        axes_config['ax1'].set_ylabel('Y (mm)')
    
        im2 = axes_config['ax2'].imshow(
            animation_data['T_opt'], extent=extent, origin='lower',
            cmap=self.cmap_temp, aspect='auto', vmin=vmin, vmax=vmax
        )
        axes_config['ax2'].set_title('Optimized Scan Temperature')
        axes_config['ax2'].set_xlabel('X (mm)')
        axes_config['ax2'].set_ylabel('Y (mm)')
    
        # Add laser position markers
        self._add_animation_laser_markers(frame, config, axes_config, animation_data)
    
        # Add temperature annotations
        self._add_animation_temperature_annotations(axes_config, animation_data)
    
    def _add_animation_laser_markers(self, frame, config, axes_config, animation_data):
        """Add laser position markers to animation frame using pre-calculated positions."""
        dt = config['dt']
        t = frame * dt
    
        # Raster laser position using corrected trajectory
        x_r, y_r, _, _ = animation_data['laser_positions']['raster_trajectory'](
            t, {'v': config['raster_config']['speed'], 'noise_sigma': 0.0}
        )
        axes_config['ax1'].plot(x_r * 1000, y_r * 1000, 'ko', markersize=8, label='Raster Laser')
    
        # Optimized laser positions using pre-calculated trajectories
        colors = ['red', 'black']
        color_codes = ['ro', 'ko']
    
        if isinstance(config['laser_opt'], tuple):
            # Multiple lasers - use cached positions
            for i, laser_config in enumerate(config['laser_opt']):
                frame_idx = min(frame, len(animation_data['opt_trajectory_cache'][i]) - 1)
                x, y, _, _ = animation_data['opt_trajectory_cache'][i][frame_idx]
            
                color_code = color_codes[i % len(color_codes)]
                axes_config['ax2'].plot(x * 1000, y * 1000, color_code, markersize=8)
        else:
            # Single laser - use cached position
            frame_idx = min(frame, len(animation_data['opt_trajectory_cache']) - 1)
            x, y, _, _ = animation_data['opt_trajectory_cache'][frame_idx]
        
            axes_config['ax2'].plot(x * 1000, y * 1000, 'ro', markersize=8)
    
        # Add legends
        axes_config['ax1'].legend()
        self._add_optimized_legend(axes_config['ax2'], config['laser_opt'], [])
    
    def _add_optimized_legend(self, ax, laser_opt, laser_positions):
        """Add legend for optimized laser configuration with consistent coloring."""
        colors = ['red', 'black']  # Heat Source 1: red, Heat Source 2: black
        
        if isinstance(laser_opt, tuple):
            # Multiple lasers
            for i, laser in enumerate(laser_opt):
                if i < len(laser_positions):  # Only add legend if laser is active
                    color = colors[i % len(colors)]
                    label = f"Heat Source {i+1}: {laser['type'].capitalize()}"
                    ax.plot([], [], marker='o', color=color, markersize=8, 
                           linestyle='None', label=label)
        else:
            # Single laser
            ax.plot([], [], marker='o', color='red', markersize=8, 
                   linestyle='None', label=f"Heat Source 1: {laser_opt['type'].capitalize()}")
        
        ax.legend()
    
    def _add_animation_temperature_annotations(self, axes_config, animation_data):
        """Add temperature annotations to animation frame."""
        max_temp_raster = np.max(animation_data['T_raster'])
        max_temp_opt = np.max(animation_data['T_opt'])
        
        axes_config['ax1'].text(
            0.98, 0.02, f"Max T: {max_temp_raster:.1f}°C",
            transform=axes_config['ax1'].transAxes, fontsize=10, color='white',
            ha='right', va='bottom',
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round')
        )
        
        axes_config['ax2'].text(
            0.98, 0.02, f"Max T: {max_temp_opt:.1f}°C",
            transform=axes_config['ax2'].transAxes, fontsize=10, color='white',
            ha='right', va='bottom',
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round')
        )
    
    def _update_melt_pool_insets(self, frame, axes_config, animation_data):
        """Update melt pool shape insets."""
        T_melt = self.model.material['T_melt']
        inset_lims = (-0.5, 0.5)
        
        # Update optimized scan inset
        self._update_single_melt_pool_inset(
            axes_config['ax_inset_opt'], animation_data['T_opt'], 
            T_melt, inset_lims, "Melt Pool Shape"
        )
        
        # Create/update raster scan inset if needed
        if axes_config['ax_inset_raster'] is None:
            axes_config['ax_inset_raster'] = inset_axes(
                axes_config['ax1'], width="30%", height="30%", loc='lower left',
                borderpad=2, bbox_to_anchor=(0.08, 0.05, 0.4, 0.4),
                bbox_transform=axes_config['ax1'].transAxes
            )
        
        self._update_single_melt_pool_inset(
            axes_config['ax_inset_raster'], animation_data['T_raster'],
            T_melt, inset_lims, "Melt Pool Shape"
        )
    
    def _update_single_melt_pool_inset(self, ax_inset, T_field, T_melt, inset_lims, title):
        """Update a single melt pool inset."""
        ax_inset.clear()
        ax_inset.set_title(title, fontsize=8, color='white')
        ax_inset.set_xlim(inset_lims)
        ax_inset.set_ylim(inset_lims)
        ax_inset.set_aspect('equal')
        ax_inset.set_xlabel("mm", fontsize=7, color='white', labelpad=2)
        ax_inset.set_ylabel("mm", fontsize=7, color='white', labelpad=2)
        ax_inset.tick_params(axis='both', which='major', labelsize=7, colors='white')
        
        # Find and plot melt pool contour
        melt_mask = T_field > T_melt
        
        if np.any(melt_mask):
            contours = measure.find_contours(melt_mask.astype(float), 0.5)
            if contours:
                # Use largest contour
                contour = max(contours, key=len)
                y_idx, x_idx = contour[:, 0], contour[:, 1]
                
                # Convert indices to physical coordinates
                x_mm = np.interp(x_idx, np.arange(self.model.nx), 
                               np.linspace(0, self.model.Lx * 1e3, self.model.nx))
                y_mm = np.interp(y_idx, np.arange(self.model.ny), 
                               np.linspace(0, self.model.Ly * 1e3, self.model.ny))
                
                # Center the coordinates
                x_mm = x_mm - np.mean(x_mm)
                y_mm = y_mm - np.mean(y_mm)
                
                ax_inset.plot(x_mm, y_mm, color='cyan', linewidth=2)
    
    def _save_animation_snapshot(self, frame, config):
        """Save animation snapshot if required."""
        if config['save_snapshots']:
            fname = os.path.join(config['snapshot_dir'], f"snapshot_{frame:05d}.png")
            plt.gcf().savefig(fname, dpi=150, bbox_inches='tight')

# ===============================
# 4. Main Optimization Routine
# ===============================

class OptimizationRunner:
    """
    Main optimization routine for LPBF laser trajectory optimization.
    
    Handles user interaction, parameter setup, optimization execution,
    and result analysis for both single and dual laser configurations.
    """
    
    def __init__(self):
        """Initialize with default configurations."""
        self.model = None
        self.optimizer = None
        self.result = None
        self.optimized_params = None
        
        # Default optimization settings
        self.default_objective = 'standard'
        self.default_method = 'Powell'
        self.default_max_iterations = 1000
        
        # Objective function descriptions
        self.objective_descriptions = {
            'standard': "Minimizing the sum of squared temperature gradients (smoothness of temperature field).",
            'thermal_uniformity': "Promoting thermal uniformity (variance in melt pool temperature and gradients).",
            'max_gradient': "Minimizing the maximum temperature gradient (reducing hot spots).",
            'path_focused': "Minimizing gradients along the laser paths (melt pool quality along scan).",
            'max_temp_difference': "Minimizing the maximum temperature difference in the melt pool."
        }
    
    # ===============================
    # User Input Methods
    # ===============================
    
    def get_user_configuration(self):
        """
        Collect user instructions for optimization configuration.
        
        Returns:
            dict: configuration dictionary with user choices
        """
        config = {}
        
        # Get number of heat sources
        config['num_sources'] = self._get_num_sources()
        
        # Get trajectory types based on number of sources
        if config['num_sources'] == 2:
            config['trajectory_types'] = self._get_dual_trajectories()
        else:
            config['trajectory_types'] = [self._get_single_trajectory()]
        
        return config
    
    def _get_num_sources(self):
        """Get number of heat sources from user."""
        while True:
           
            num_sources = input("Optimize for (1) single or (2) dual heat sources? Enter 1 or 2: ").strip()
            if num_sources in ['1', '2']:
                return int(num_sources)
            print("Invalid input. Please enter 1 or 2.")
    
    def _get_single_trajectory(self):
        """Get trajectory type for single laser configuration."""
        print("\nSelect trajectory for optimization:")
        print("1 - Straight")
        print("2 - Sawtooth")
        print("3 - Swirl")
    
        while True:
            choice = input("Enter 1, 2, or 3: ").strip()
            if choice == '1':
                return 'straight'
            elif choice == '2':
                return 'sawtooth'
            elif choice == '3':
                return 'swirl'
            print("Invalid input. Please enter 1, 2, or 3.")
    
    def _get_dual_trajectories(self):
        """Get trajectory types for dual laser configuration."""
        trajectories = []
    
        for source_num in [1, 2]:
            print(f"\nSelect reference trajectory for heat source {source_num}:")
            print("1 - Straight")
            print("2 - Sawtooth")
            print("3 - Swirl")
        
            while True:
                choice = input("Enter 1, 2, or 3: ").strip()
                if choice == '1':
                    trajectories.append('straight')
                    break
                elif choice == '2':
                    trajectories.append('sawtooth')
                    break
                elif choice == '3':
                    trajectories.append('swirl')
                    break
                print("Invalid input. Please enter 1, 2, or 3.")
    
        return trajectories
    
    # ===============================
    # Model Setup Methods
    # ===============================
    
    def setup_model(self):
        """
        Initialize heat transfer model.
        
        Returns:
            HeatTransferModel: configured model instance
        """
        # Material parameters for LPBF simulation
        material_params = self._get_material_parameters()
        
        # Domain and grid configuration
        domain_config = self._get_domain_configuration()
        
        # Create and return model
        return HeatTransferModel(
            domain_size=domain_config['domain_size'],
            grid_size=domain_config['grid_size'],
            dt=domain_config['dt'],
            material_params=material_params
        )
    
    def _get_material_parameters(self):
        """Get material parameters for LPBF simulation."""
        return {
            'T0': 21.0,                # Initial temperature (°C)
            'alpha': 5e-6,             # Thermal diffusivity (m²/s)
            'rho': 7800.0,             # Density (kg/m³)
            'cp': 500.0,               # Specific heat capacity (J/(kg·K))
            'k': 20.0,                 # Thermal conductivity (W/(m·K))
            'T_melt': 1500.0,          # Melting temperature (°C)
            'thickness': 0.005,        # Grid thickness (m - define volume of cells)
            'absorptivity': 1.0        # Absorptivity (fraction)
        }
    
    def _get_domain_configuration(self):
        """Get domain and discretization parameters."""
        return {
            'domain_size': (0.008, 0.001),     # Domain size (m) - 8mm x 1mm
            'grid_size': (401, 51),            # Grid resolution
            'dt': 1e-5                         # Time step (s)
        }
    
    # ===============================
    # Parameter Setup Methods
    # ===============================
    
    def setup_parameters(self, config): 
        """
        Setup initial parameters and bounds based on user configuration.
        
        Args:
            config: configuration dictionary from user input
            
        Returns:
            tuple: (initial_params, bounds) dictionaries
        """
        if config['num_sources'] == 2:
            return self._setup_dual_source_parameters(config['trajectory_types'])
        else:
            return self._setup_single_source_parameters(config['trajectory_types'][0])
    
    def _setup_single_source_parameters(self, trajectory_type):
        """Setup parameters for single laser configuration."""
        if trajectory_type == 'straight':
            initial_params, bounds = self._get_straight_parameters()
        elif trajectory_type == 'sawtooth':
            initial_params, bounds = self._get_sawtooth_parameters()
        else:  # swirl
            initial_params, bounds = self._get_swirl_parameters()
    
        # Add noise parameter
        initial_params['noise_sigma'] = 0.000
        bounds['noise_sigma'] = (0.0, 0.0005)
    
        return initial_params, bounds

    def _setup_dual_source_parameters(self, trajectory_types):
        """Setup parameters for dual laser configuration."""
        initial_params = {}
        bounds = {}
    
        # Setup parameters for each heat source
        for i, trajectory_type in enumerate(trajectory_types):
            suffix = '' if i == 0 else '2'
        
            if trajectory_type == 'straight':
                source_params, source_bounds = self._get_straight_parameters(suffix)
            elif trajectory_type == 'sawtooth':
                source_params, source_bounds = self._get_sawtooth_parameters(suffix)
            else:  # swirl
                source_params, source_bounds = self._get_swirl_parameters(suffix)
        
            initial_params.update(source_params)
            bounds.update(source_bounds)
    
        # Add noise parameter
        initial_params['noise_sigma'] = 0.000
        bounds['noise_sigma'] = (0.0, 0.0005)
    
        return initial_params, bounds
    
    def _get_sawtooth_parameters(self, suffix=''):
        """Get sawtooth trajectory parameters with optional suffix."""
        prefix = f'sawtooth{suffix}'
        
        initial_params = {
            f'{prefix}_v': 0.7,
            f'{prefix}_A': 0.0003,
            f'{prefix}_y0': 0.0005,
            f'{prefix}_period': 0.002,
            f'{prefix}_Q': 200.0,
            f'{prefix}_r0': 4e-5,
        }
        
        bounds = {
            f'{prefix}_v': (0.55, 0.85),
            f'{prefix}_A': (0.00025, 0.0004),
            f'{prefix}_period': (0.001, 0.003),
            f'{prefix}_Q': (175.0, 225.0),
        }

        # Add y0 as a fixed parameter
        initial_params[f'{prefix}_y0'] = 0.0005
        
        return initial_params, bounds
    
    def _get_swirl_parameters(self, suffix=''):
        """Get swirl trajectory parameters with optional suffix."""
        prefix = f'swirl{suffix}'
        
        initial_params = {
            f'{prefix}_v': 0.7,
            f'{prefix}_A': 0.0005,
            f'{prefix}_y0': 0.0005,
            f'{prefix}_fr': 900.0,
            f'{prefix}_Q': 200.0,
            f'{prefix}_r0': 4e-5,
        }
        
        bounds = {
            f'{prefix}_v': (0.55, 0.85),
            f'{prefix}_A': (0.0004, 0.0006),
            f'{prefix}_fr': (700.0, 1100.0),
            f'{prefix}_Q': (175.0, 225.0),
        }
        
        initial_params[f'{prefix}_y0'] = 0.0005
    
        return initial_params, bounds

    def _get_straight_parameters(self, suffix=''):
        """Get straight trajectory parameters with optional suffix."""
        prefix = f'straight{suffix}'
    
        initial_params = {
            f'{prefix}_v': 0.7,
            f'{prefix}_y0': 0.0005,
            f'{prefix}_Q': 200.0,
            f'{prefix}_r0': 4e-5,
        }
    
        bounds = {
            f'{prefix}_v': (0.55, 0.85),
            f'{prefix}_Q': (175.0, 225.0),
        }

        initial_params[f'{prefix}_y0'] = 0.0005
    
        return initial_params, bounds
    
    # ===============================
    # Optimization Execution
    # ===============================
    
    def run_optimization(self, model, initial_params, bounds):
        """
        Execute optimization with error handling and fallback methods.
        
        Args:
            model: HeatTransferModel instance
            initial_params: initial parameter dictionary
            bounds: parameter bounds dictionary
            
        Returns:
            tuple: (TrajectoryOptimizer, optimization_result, optimized_params)
        """
        # Create optimizer with updated scan region
        optimizer = TrajectoryOptimizer(
            model=model,
            initial_params=initial_params,
            bounds=bounds,
            x_range=(0.0015, model.Lx - 0.0015)  # 1.5mm margins on both sides
        )
        
        # Print optimization information
        self._print_optimization_header()
        
        # Attempt optimization with primary method
        result, optimized_params = self._attempt_optimization(
            optimizer, self.default_objective, self.default_method
        )
        
        # Try fallback method if primary failed
        if not result.success:
            print(f"Primary method ({self.default_method}) failed. Trying fallback method...")
            result, optimized_params = self._attempt_optimization(
                optimizer, self.default_objective, 'Nelder-Mead'
            )
        
        return optimizer, result, optimized_params
    
    def _print_optimization_header(self):
        """Print optimization header information."""
        print("\n=============================================")
        print("Starting Laser Path Optimization")
        print("=============================================\n")
        
        objective_desc = self.objective_descriptions.get(
            self.default_objective, 'Unknown objective'
        )
        print(f"Optimization objective: {objective_desc}")
    
    def _attempt_optimization(self, optimizer, objective_type, method):
        """
        Attempt optimization with specified method and handle errors.
        
        Args:
            optimizer: TrajectoryOptimizer instance
            objective_type: type of objective function
            method: optimization method
            
        Returns:
            tuple: (result, optimized_params)
        """
        try:
            result, optimized_params = optimizer.optimize(
                objective_type=objective_type,
                method=method,
                max_iterations=self.default_max_iterations
            )
            return result, optimized_params
        
        except Exception as e:
            print(f"Optimization failed with error: {str(e)}")
            if method != 'Nelder-Mead':
                print("Falling back to derivative-free method (Nelder-Mead)")
                return optimizer.optimize(
                    objective_type=objective_type,
                    method='Nelder-Mead',
                    max_iterations=self.default_max_iterations
                )
            else:
                # If fallback also fails, return failed result
                from scipy.optimize import OptimizeResult
                failed_result = OptimizeResult(
                    success=False,
                    message=f"All optimization methods failed: {str(e)}",
                    x=optimizer.parameters_to_array(optimizer.initial_params)
                )
                return failed_result, optimizer.initial_params
    
    # ===============================
    # Results Analysis
    # ===============================
    
    def analyze_results(self, optimizer, result, optimized_params, initial_params):
        """
        Analyze and display optimization results.
        
        Args:
            optimizer: TrajectoryOptimizer instance
            result: optimization result
            optimized_params: optimized parameter dictionary
            initial_params: initial parameter dictionary
        """
        if result.success:
            self._display_successful_results(
                optimizer, optimized_params, initial_params
            )
        else:
            self._display_failed_results(result)
    
    def _display_successful_results(self, optimizer, optimized_params, initial_params):
        """Display results for successful optimization."""
        print("\n=============================================")
        print("Optimization Successful!")
        print("=============================================\n")
        
        # Calculate and display improvement
        self._display_objective_improvement(optimizer, optimized_params, initial_params)
        
        # Display parameter changes
        self._display_parameter_changes(optimizer, optimized_params, initial_params)
    
    def _display_failed_results(self, result):
        """Display results for failed optimization."""
        print("\n=============================================")
        print("Optimization Failed")
        print("=============================================\n")
        print(f"Error message: {result.message}")
        print("Try with a different optimization method or adjust parameters.")
    
    def _display_objective_improvement(self, optimizer, optimized_params, initial_params):
        """Display improvement in objective function value."""
        # Use thermal_uniformity as standard comparison metric
        initial_obj = optimizer.objective_thermal_uniformity(
            optimizer.parameters_to_array(initial_params)
        )
        final_obj = optimizer.objective_thermal_uniformity(
            optimizer.parameters_to_array(optimized_params)
        )
        
        improvement = (initial_obj - final_obj) / initial_obj * 100 if initial_obj != 0 else 0
        
        print(f"Initial objective value: {initial_obj:.4e}")
        print(f"Final objective value: {final_obj:.4e}")
        print(f"Improvement: {improvement:.2f}%\n")
    
    def _display_parameter_changes(self, optimizer, optimized_params, initial_params):
        """Display table of parameter changes."""
        print("Parameter Comparison:")
        print("---------------------------------------------")
        print(f"{'Parameter':<15} {'Initial':<12} {'Optimized':<12} {'Change %':<10}")
        print("---------------------------------------------")
        
        for name in optimizer.param_names:
            init_val = initial_params[name]
            opt_val = optimized_params[name]
            
            # Calculate percentage change
            if init_val != 0:
                change = (opt_val - init_val) / init_val * 100
            else:
                change = float('inf') if opt_val != 0 else 0
            
            print(f"{name:<15} {init_val:<12.6f} {opt_val:<12.6f} {change:<10.2f}")
    
    # ===============================
    # Visualization and Analysis
    # ===============================
    
    def run_visualization_and_analysis(self, model, optimizer, result, 
                                     optimized_params, initial_params):
        """
        Run visualization and optional analysis steps.
        
        Args:
            model: HeatTransferModel instance
            optimizer: TrajectoryOptimizer instance
            result: optimization result
            optimized_params: optimized parameter dictionary
            initial_params: initial parameter dictionary
        """
        if not result.success:
            return
        
        # Generate comparison visualizations
        self._generate_comparison_plots(model, initial_params, optimized_params)
        
        # Optional animation
        self._optional_animation(model, initial_params, optimized_params)
        
        # Optional sensitivity analysis
        self._optional_sensitivity_analysis(optimizer, optimized_params)
        
        print("\nOptimization process complete!")
    
    def _generate_comparison_plots(self, model, initial_params, optimized_params):
        """Generate comparison visualization plots."""
        print("\nGenerating comparison visualizations...")
        viz = Visualization(model)
        viz.compare_simulations(initial_params, optimized_params)
    
        # Generate temporal gradient evolution plot
        create_temporal_plot = input("Generate temporal gradient evolution plot? (y/n): ").lower()
        if create_temporal_plot == 'y':
            print("Generating temporal gradient analysis...")
        
            # Run simulations to get temporal data
            raster_config = viz._setup_raster_scan()
            _, _, _, raster_temporal_stats = viz._simulate_raster_scan(raster_config)
            _, _, _, opt_temporal_stats = viz._simulate_optimized_scan(
                initial_params, optimized_params, True
            )
        
            # Plot temporal evolution
            viz.plot_temporal_gradients(raster_temporal_stats, opt_temporal_stats)
    
    def _optional_animation(self, model, initial_params, optimized_params):
        """Optionally generate optimization results animation."""
        create_animation = input("Generate animation of optimization results? (y/n): ").lower()
        if create_animation == 'y':
            print("Generating animation (this may take a while)...")
            viz = Visualization(model)
            ani = viz.animate_optimization_results(
                model, 
                initial_params, 
                optimized_params, 
                fps=30,
            )
    
    def _optional_sensitivity_analysis(self, optimizer, optimized_params):
        """Optionally perform sensitivity analysis."""
        run_sensitivity = input("Perform sensitivity analysis of optimized solution? (y/n): ").lower()
        if run_sensitivity == 'y':
            print("\nPerforming sensitivity analysis...")
            sensitivity_data = optimizer.perform_sensitivity_analysis(
                optimized_params, 
                objective_type=self.default_objective
            )
    
    # ===============================
    # Main Execution Method
    # ===============================
    
    def execute(self):
        """
        Execute complete optimization workflow.
        
        Returns:
            tuple: (model, optimizer, result, optimized_params) for further use
        """
        try:
            # Get user configuration
            config = self.get_user_configuration()
            
            # Setup model
            self.model = self.setup_model()
            
            # Setup parameters
            initial_params, bounds = self.setup_parameters(config)
            
            # Run optimization
            self.optimizer, self.result, self.optimized_params = self.run_optimization(
                self.model, initial_params, bounds
            )
            
            # Analyze results
            self.analyze_results(
                self.optimizer, self.result, self.optimized_params, initial_params
            )
            
            # Visualization and analysis
            self.run_visualization_and_analysis(
                self.model, self.optimizer, self.result, 
                self.optimized_params, initial_params
            )
            
            return self.model, self.optimizer, self.result, self.optimized_params
            
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user.")
            return None, None, None, None
        
        except Exception as e:
            import traceback
            print(f"\nUnexpected error during optimization: {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("Please check your inputs and try again.")

def run_optimization():
    """
    Legacy function for backward compatibility.
    
    Returns:
        tuple: (model, optimizer, result, optimized_params)
    """
    runner = OptimizationRunner()
    return runner.execute()

# ===============================
# 5. G-code Output Utility
# ===============================

class GCodeGenerator:
    """
    G-code generation for optimized laser trajectories.
    """
    
    def __init__(self, units='mm', positioning='absolute'):
        """
        Initialize G-code generator with configuration settings.
        
        Args:
            units: 'mm' or 'inch' for coordinate units
            positioning: 'absolute' or 'relative' positioning mode
        """
        self.units = units
        self.positioning = positioning
        
        # G-code configuration
        self.config = {
            'units_code': 'G21' if units == 'mm' else 'G20',
            'positioning_code': 'G90' if positioning == 'absolute' else 'G91',
            'feedrate_default': 1000,  # mm/min or in/min
            'laser_power_max': 255,    # Maximum laser power
            'precision': 3             # Decimal places for coordinates
        }
        
        # Trajectory metadata
        self.trajectory_info = {}
    
    # ===============================
    # Path Processing Methods
    # ===============================
    
    def generate_gcode(self, optimizer, optimized_params, filename="optimized_scan.gcode", 
                      include_metadata=True, separate_trajectories=False):
        """
        Generate G-code file from optimized laser parameters.
        
        Args:
            optimizer: TrajectoryOptimizer instance
            optimized_params: dictionary of optimized parameters
            filename: output G-code filename
            include_metadata: whether to include parameter metadata in comments
            separate_trajectories: whether to output separate files for each trajectory
            
        Returns:
            str or list: filename(s) of generated G-code file(s)
        """
        # Extract laser parameters
        params_array = optimizer.parameters_to_array(optimized_params)
        laser_params, heat_params = optimizer.unpack_parameters(params_array)
        
        # Process trajectories
        trajectory_data = self._process_trajectories(optimizer, laser_params, heat_params)
        
        if separate_trajectories and len(trajectory_data) > 1:
            return self._generate_separate_files(
                trajectory_data, filename, optimized_params, include_metadata
            )
        else:
            return self._generate_combined_file(
                trajectory_data, filename, optimized_params, include_metadata
            )
    
    def _process_trajectories(self, optimizer, laser_params, heat_params):
        """
        Process laser trajectories and generate path data.
        
        Args:
            optimizer: TrajectoryOptimizer instance
            laser_params: laser trajectory parameters
            heat_params: heat source parameters
            
        Returns:
            list: trajectory data dictionaries
        """
        trajectory_data = []
        x_start, x_end = optimizer.x_range
        
        # Handle single or dual laser configuration
        if isinstance(laser_params, tuple):
            # Multiple lasers
            for i, laser in enumerate(laser_params):
                heat_param = heat_params[i] if isinstance(heat_params, tuple) else heat_params
                traj_data = self._process_single_trajectory(
                    optimizer, laser, heat_param, x_start, x_end, i + 1
                )
                trajectory_data.append(traj_data)
        else:
            # Single laser
            traj_data = self._process_single_trajectory(
                optimizer, laser_params, heat_params, x_start, x_end, 1
            )
            trajectory_data.append(traj_data)
        
        return trajectory_data
    
    def _process_single_trajectory(self, optimizer, laser_params, heat_params, 
                                 x_start, x_end, laser_id):
        """
        Process a single laser trajectory.
        
        Args:
            optimizer: TrajectoryOptimizer instance
            laser_params: single laser parameters
            heat_params: heat source parameters
            x_start, x_end: x-range for trajectory
            laser_id: identifier for this laser
            
        Returns:
            dict: trajectory data
        """
        trajectory_type = laser_params['type']
        params = laser_params['params']
        
        # Calculate trajectory points
        dt = 1e-5  # Time step for trajectory sampling
        v = params['v']
        total_time = (x_end - x_start) / v
        t_points = np.arange(0, total_time, dt)
        
        # Generate path points
        if trajectory_type == 'straight':
            path_points = np.array([
                optimizer.model.straight_trajectory(t, params)[:2] 
                for t in t_points
            ])
        elif trajectory_type == 'sawtooth':
            path_points = np.array([
                optimizer.model.sawtooth_trajectory(t, params)[:2] 
                for t in t_points
            ])
        else:  # swirl
            path_points = np.array([
                optimizer.model.swirl_trajectory(t, params)[:2] 
                for t in t_points
            ])
        
        # Resample path for consistent speed
        resampled_path = self._resample_path_constant_speed(path_points, v, dt)
        
        # Calculate feedrate and laser power
        feedrate = self._calculate_feedrate(v)
        laser_power = self._calculate_laser_power(heat_params)
        
        # Store trajectory metadata
        self.trajectory_info[f'laser_{laser_id}'] = {
            'type': trajectory_type,
            'velocity': v,
            'power': heat_params.get('Q', 200.0),
            'spot_size': heat_params.get('r0', 8e-5),
            'points_count': len(resampled_path),
            'path_length': self._calculate_path_length(resampled_path)
        }
        
        return {
            'id': laser_id,
            'type': trajectory_type,
            'path': resampled_path,
            'feedrate': feedrate,
            'laser_power': laser_power,
            'params': params,
            'heat_params': heat_params
        }
    
    def _resample_path_constant_speed(self, path, velocity, dt):
        """
        Resample path to maintain constant speed between points.
        
        Args:
            path: array of (x, y) coordinates
            velocity: target velocity (m/s)
            dt: time step
            
        Returns:
            numpy.ndarray: resampled path points
        """
        if len(path) < 2:
            return path
        
        # Calculate cumulative arc length
        diffs = np.diff(path, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = arc_lengths[-1]
        
        # Calculate number of points for constant speed
        target_distance = velocity * dt
        n_points = max(2, int(np.ceil(total_length / target_distance)))
        
        # Create target arc lengths
        target_lengths = np.linspace(0, total_length, n_points)
        
        # Interpolate coordinates
        x_resampled = np.interp(target_lengths, arc_lengths, path[:, 0])
        y_resampled = np.interp(target_lengths, arc_lengths, path[:, 1])
        
        return np.column_stack([x_resampled, y_resampled])
    
    def _calculate_feedrate(self, velocity_ms):
        """
        Convert velocity from m/s to appropriate feedrate units.
        
        Args:
            velocity_ms: velocity in m/s
            
        Returns:
            float: feedrate in mm/min or in/min
        """
        if self.units == 'mm':
            # Convert m/s to mm/min
            feedrate = velocity_ms * 1000 * 60
        else:  # inches
            # Convert m/s to in/min
            feedrate = velocity_ms * 39.3701 * 60
        
        return round(feedrate, 1)
    
    def _calculate_laser_power(self, heat_params):
        """
        Calculate laser power setting from heat parameters.
        
        Args:
            heat_params: heat source parameters
            
        Returns:
            int: laser power setting (0-255 range)
        """
        power_watts = heat_params.get('Q', 200.0)
        max_power = 500.0  # Assumed maximum power in watts
        
        # Scale to 0-255 range
        power_setting = int((power_watts / max_power) * self.config['laser_power_max'])
        return max(1, min(power_setting, self.config['laser_power_max']))
    
    def _calculate_path_length(self, path):
        """Calculate total path length in meters."""
        if len(path) < 2:
            return 0.0
        
        diffs = np.diff(path, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        return np.sum(segment_lengths)
    
    # ===============================
    # G-code Generation Methods
    # ===============================
    
    def _generate_combined_file(self, trajectory_data, filename, optimized_params, include_metadata):
        """Generate single G-code file with all trajectories."""
        with open(filename, 'w') as f:
            # Write header
            self._write_gcode_header(f, optimized_params, include_metadata)
            
            # Write trajectories
            for i, traj_data in enumerate(trajectory_data):
                self._write_trajectory_gcode(f, traj_data, i + 1)
            
            # Write footer
            self._write_gcode_footer(f)
        
        print(f"G-code written to {filename}")
        self._print_generation_summary(trajectory_data, filename)
        
        return filename
    
    def _generate_separate_files(self, trajectory_data, base_filename, optimized_params, include_metadata):
        """Generate separate G-code files for each trajectory."""
        generated_files = []
        base_name, ext = os.path.splitext(base_filename)
        
        for i, traj_data in enumerate(trajectory_data):
            filename = f"{base_name}_laser{traj_data['id']}{ext}"
            
            with open(filename, 'w') as f:
                # Write header
                self._write_gcode_header(f, optimized_params, include_metadata, traj_data['id'])
                
                # Write single trajectory
                self._write_trajectory_gcode(f, traj_data, traj_data['id'])
                
                # Write footer
                self._write_gcode_footer(f)
            
            generated_files.append(filename)
            print(f"G-code written to {filename}")
        
        self._print_generation_summary(trajectory_data, generated_files)
        return generated_files
    
    def _write_gcode_header(self, file, optimized_params, include_metadata, laser_id=None):
        """Write G-code file header with setup commands and metadata."""
        file.write("; ===============================================\n")
        if laser_id:
            file.write(f"; Optimized Laser Trajectory G-code - Laser {laser_id}\n")
        else:
            file.write("; Optimized Laser Trajectory G-code\n")
        file.write("; Generated by LPBF Trajectory Optimizer\n")
        file.write(f"; Generated on: {self._get_timestamp()}\n")
        file.write("; ===============================================\n\n")
        
        # Include parameter metadata if requested
        if include_metadata:
            self._write_parameter_metadata(file, optimized_params, laser_id)
        
        # Setup commands
        file.write("; Setup commands\n")
        file.write(f"{self.config['units_code']} ; Set units to {self.units}\n")
        file.write(f"{self.config['positioning_code']} ; Set {'absolute' if self.positioning == 'absolute' else 'relative'} positioning\n")
        file.write("G94 ; Set feedrate per minute mode\n")
        file.write("M3 ; Spindle/Laser enable\n\n")
    
    def _write_parameter_metadata(self, file, optimized_params, laser_id=None):
        """Write optimized parameters as G-code comments."""
        file.write("; Optimized Parameters:\n")
        
        if laser_id:
            # Filter parameters for specific laser
            relevant_params = {k: v for k, v in optimized_params.items() 
                             if f'laser_{laser_id}' in self.trajectory_info or 
                                (laser_id == 1 and not any(c in k for c in ['2']))}
        else:
            relevant_params = optimized_params
        
        for param, value in relevant_params.items():
            if isinstance(value, float):
                file.write(f"; {param}: {value:.6f}\n")
            else:
                file.write(f"; {param}: {value}\n")
        
        file.write(";\n")
        
        # Write trajectory information
        for laser_key, info in self.trajectory_info.items():
            if laser_id is None or laser_key == f'laser_{laser_id}':
                file.write(f"; {laser_key.replace('_', ' ').title()} Information:\n")
                file.write(f";   Type: {info['type'].capitalize()}\n")
                file.write(f";   Velocity: {info['velocity']:.6f} m/s\n")
                file.write(f";   Power: {info['power']:.1f} W\n")
                file.write(f";   Spot Size: {info['spot_size']*1000:.3f} mm\n")
                file.write(f";   Path Length: {info['path_length']*1000:.2f} mm\n")
                file.write(f";   Points Count: {info['points_count']}\n")
                file.write(";\n")
    
    def _write_trajectory_gcode(self, file, traj_data, trajectory_num):
        """Write G-code commands for a single trajectory."""
        trajectory_type = traj_data['type']
        path = traj_data['path']
        feedrate = traj_data['feedrate']
        laser_power = traj_data['laser_power']
        
        file.write(f"; {trajectory_type.capitalize()} trajectory - Laser {trajectory_num}\n")
        file.write(f"F{feedrate:.1f} ; Set feedrate\n")
        file.write(f"S{laser_power} ; Set laser power\n")
        
        # Convert coordinates to appropriate units
        unit_multiplier = 1000 if self.units == 'mm' else 39.3701
        precision = self.config['precision']
        
        # Move to start position
        x_start, y_start = path[0] * unit_multiplier
        file.write(f"G0 X{x_start:.{precision}f} Y{y_start:.{precision}f} ; Rapid to start\n")
        file.write("M3 ; Laser ON\n")
        
        # Write path points
        for i, (x, y) in enumerate(path[1:], 1):
            x_coord = x * unit_multiplier
            y_coord = y * unit_multiplier
            
            file.write(f"G1 X{x_coord:.{precision}f} Y{y_coord:.{precision}f}")
            
            # Add comments for key points
            if i == 1:
                file.write(" ; Start trajectory")
            elif i == len(path) - 1:
                file.write(" ; End trajectory")
            elif i % 100 == 0:
                file.write(f" ; Point {i}")
            
            file.write("\n")
        
        file.write("M5 ; Laser OFF\n")
        file.write("G0 Z5 ; Lift Z (if applicable)\n\n")
    
    def _write_gcode_footer(self, file):
        """Write G-code file footer with cleanup commands."""
        file.write("; Cleanup commands\n")
        file.write("M5 ; Laser OFF\n")
        file.write("G0 X0 Y0 ; Return to origin\n")
        file.write("M30 ; Program end\n")
        file.write("\n; End of G-code\n")
    
    # ===============================
    # Utility Methods
    # ===============================
    
    def _get_timestamp(self):
        """Get current timestamp for file header."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _print_generation_summary(self, trajectory_data, filenames):
        """Print summary of G-code generation."""
        print("\n" + "="*50)
        print("G-code Generation Summary")
        print("="*50)
        
        if isinstance(filenames, list):
            print(f"Generated {len(filenames)} files:")
            for filename in filenames:
                print(f"  - {filename}")
        else:
            print(f"Generated file: {filenames}")
        
        print(f"\nTrajectory Summary:")
        total_length = 0
        total_points = 0
        
        for traj_data in trajectory_data:
            laser_id = traj_data['id']
            traj_type = traj_data['type']
            info = self.trajectory_info[f'laser_{laser_id}']
            
            print(f"  Laser {laser_id} ({traj_type.capitalize()}):")
            print(f"    - Path length: {info['path_length']*1000:.2f} mm")
            print(f"    - Points: {info['points_count']}")
            print(f"    - Velocity: {info['velocity']:.3f} m/s")
            print(f"    - Power: {info['power']:.1f} W")
            
            total_length += info['path_length']
            total_points += info['points_count']
        
        print(f"\nTotal path length: {total_length*1000:.2f} mm")
        print(f"Total points: {total_points}")
        print(f"Units: {self.units}")
        print(f"Positioning: {self.positioning}")
    
    # ===============================
    # Validation Methods
    # ===============================
    
    def validate_gcode(self, filename):
        """
        Perform basic validation of generated G-code file.
        
        Args:
            filename: G-code file to validate
            
        Returns:
            dict: validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Basic validation checks
            self._validate_gcode_structure(lines, validation_results)
            self._validate_gcode_coordinates(lines, validation_results)
            self._calculate_gcode_stats(lines, validation_results)
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Failed to validate file: {str(e)}")
        
        return validation_results
    
    def _validate_gcode_structure(self, lines, results):
        """Validate basic G-code structure."""
        has_start = any('M3' in line for line in lines)
        has_end = any('M5' in line or 'M30' in line for line in lines)
        
        if not has_start:
            results['warnings'].append("No laser/spindle start command (M3) found")
        
        if not has_end:
            results['warnings'].append("No program end command (M5/M30) found")
    
    def _validate_gcode_coordinates(self, lines, results):
        """Validate coordinate values are reasonable."""
        coords = []
        for line in lines:
            if line.startswith('G0') or line.startswith('G1'):
                # Extract X and Y coordinates
                try:
                    if 'X' in line and 'Y' in line:
                        x_pos = line.find('X') + 1
                        y_pos = line.find('Y') + 1
                        
                        x_end = min([i for i in [line.find(' ', x_pos), line.find(';', x_pos), len(line)] if i > x_pos])
                        y_end = min([i for i in [line.find(' ', y_pos), line.find(';', y_pos), len(line)] if i > y_pos])
                        
                        x_val = float(line[x_pos:x_end])
                        y_val = float(line[y_pos:y_end])
                        coords.append((x_val, y_val))
                except:
                    continue
        
        if coords:
            x_coords, y_coords = zip(*coords)
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            
            # Check for reasonable coordinate ranges
            max_reasonable = 1000 if self.units == 'mm' else 40  # mm or inches
            
            if x_range > max_reasonable or y_range > max_reasonable:
                results['warnings'].append(f"Large coordinate range detected: X={x_range:.1f}, Y={y_range:.1f}")
    
    def _calculate_gcode_stats(self, lines, results):
        """Calculate G-code file statistics."""
        results['stats'] = {
            'total_lines': len(lines),
            'comment_lines': len([l for l in lines if l.strip().startswith(';')]),
            'move_commands': len([l for l in lines if l.strip().startswith(('G0', 'G1'))]),
            'laser_commands': len([l for l in lines if 'M3' in l or 'M5' in l])
        }

def output_optimized_gcode(optimizer, optimized_params, filename="optimized_scan.gcode", 
                          units='mm', separate_files=False, include_metadata=True, validate=True):
    """
    Legacy function for backward compatibility with improved functionality.
    
    Args:
        optimizer: TrajectoryOptimizer instance
        optimized_params: dictionary of optimized parameters
        filename: output G-code filename
        units: 'mm' or 'inch' for coordinate units
        separate_files: whether to create separate files for each laser
        include_metadata: whether to include parameter metadata
        validate: whether to validate generated G-code
    
    Returns:
        str or list: filename(s) of generated G-code file(s)
    """
    # Create G-code generator
    generator = GCodeGenerator(units=units, positioning='absolute')
    
    # Generate G-code
    generated_files = generator.generate_gcode(
        optimizer, 
        optimized_params, 
        filename,
        include_metadata=include_metadata,
        separate_trajectories=separate_files
    )
    
    # Validate generated files if requested
    if validate:
        files_to_validate = generated_files if isinstance(generated_files, list) else [generated_files]
        
        for gcode_file in files_to_validate:
            validation_results = generator.validate_gcode(gcode_file)
            
            if validation_results['errors']:
                print(f"\nValidation errors for {gcode_file}:")
                for error in validation_results['errors']:
                    print(f"  - {error}")
            
            if validation_results['warnings']:
                print(f"\nValidation warnings for {gcode_file}:")
                for warning in validation_results['warnings']:
                    print(f"  - {warning}")
            
            if validation_results['valid'] and not validation_results['errors']:
                print(f"\n✓ {gcode_file} validated successfully")
    
    return generated_files


# ===============================
# Enhanced Main Execution
# ===============================

if __name__ == "__main__":
    # Run optimization
    model, optimizer, result, optimized_params = run_optimization()
    
    # Enhanced G-code export with options
    if model and optimizer and result and optimized_params and result.success:
        print("\n" + "="*50)
        print("G-code Export Options")
        print("="*50)
        
        export_gcode = input("Would you like to export G-code? (y/n): ").strip().lower()
        
        if export_gcode == 'y':
            # Get export options
            filename = input("Enter filename (default: optimized_scan.gcode): ").strip()
            if not filename:
                filename = "optimized_scan.gcode"
            
            units = input("Select units - (1) mm or (2) inches (default: mm): ").strip()
            units = 'inch' if units == '2' else 'mm'
            
            separate = input("Create separate files for each laser? (y/n, default: n): ").strip().lower()
            separate_files = separate == 'y'
            
            metadata = input("Include parameter metadata in comments? (y/n, default: y): ").strip().lower()
            include_metadata = metadata != 'n'
            
            # Generate G-code
            try:
                generated_files = output_optimized_gcode(
                    optimizer, 
                    optimized_params, 
                    filename,
                    units=units,
                    separate_files=separate_files,
                    include_metadata=include_metadata,
                    validate=True
                )
                
                print(f"\n✓ G-code export completed successfully!")
                
            except Exception as e:
                print(f"\n✗ G-code export failed: {str(e)}")
        else:
            print("G-code export skipped.")
    else:
        print("G-code export not available - optimization was not successful or incomplete.")
