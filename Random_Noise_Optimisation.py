#importing the necessary libraries
import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Laser trajectory parameters
A = 0.001  # Amplitude of the sinusoidal trajectory (1 mm radius)
loops = 10  # Number of loops
total_length = 0.01  # Total length in meters (10 mm)
T = 3 / 1000  # Simulation duration in seconds
V = total_length / T  # Linear speed in meters per second
fr = loops / T  # Frequency in Hz (10 loops over the simulation duration)
pitch = V / fr  # wobble pitch in m
n = pitch / A  # wobble radius-normalised pitch
om = 2 * np.pi * fr  # Angular velocity in rad/s
steps = 500  # number of steps to take
x_max = T * V  # total distance travelled [m]

# Parametric equations for the laser spot motion with linear speed
# Constants for noise
sigma_x_noise = A / 15  # Standard deviation for noise in x direction as fraction of Amplitude
sigma_y_noise = A / 15  # Standard deviation for noise in y direction as fraction of Amplitude

# parametric equations for the parent function of heat source 1
def xx(t):
    return A * np.sin(om * t) + V * t  # reference curve

def yy(t):
    return -A * np.cos(om * t)  # reference curve

# parametric equations for the parent function plus noise of heat source 1
def x(t):
    noise = np.random.normal(0, sigma_x_noise, t.shape)  # Add noise in the x direction
    return A * np.sin(om * t) + V * t + noise  # Reference curve with noise

def y(t):
    noise = np.random.normal(0, sigma_y_noise, t.shape)  # Add noise in the y direction
    return -A * np.cos(om * t) + noise  # Reference curve with noise

# equation for the parent function of heat source 2
def sawtooth(t, period=0.001, noise_amplitude=A / 10, subsample_factor=2):  # Match noise scale to sigma_y_noise
    base_sawtooth = 0.15 * (t / (period * 10) - np.floor(0.5 + t / (period * 10)))  # Sawtooth function
    noise = np.interp(
        t,
        t[::subsample_factor],
        np.random.normal(0, noise_amplitude, t[::subsample_factor].shape)
    )  # Subsample and interpolate noise
    return base_sawtooth + noise

# Define additional trajectories
def horizontal_line(t):
    return np.full_like(t, 0.05)  # A constant y-value (e.g., 0.05 meters)

def sine_wave(t):
    return 0.001 * np.sin(2 * np.pi * 10 * t / T)  # 10 waves over the trajectory, amplitude of 1 mm

def sine_wave_with_noise(t, noise_amplitude=0.0002, subsample_factor=3):  # Noise amplitude is 0.2 mm
    reference = 0.001 * np.sin(2 * np.pi * 10 * t / T)  # Reference sine wave (10 waves, 1 mm amplitude)
    
    # Subsample and interpolate noise
    noise = np.interp(
        t,
        t[::subsample_factor],
        np.random.normal(0, noise_amplitude, t[::subsample_factor].shape)
    )
    
    return reference, reference + noise  # Return both reference and noisy sine wave

# Define a noisy horizontal line trajectory with reduced noise amplitude
def noisy_horizontal_line(t):
    subsample_factor = 5  # Reduce the density of noise by subsampling
    reduced_noise_amplitude = sigma_y_noise / 2  # Reduce the noise amplitude by half
    noise = np.interp(
        t, 
        t[::subsample_factor], 
        np.random.normal(0, reduced_noise_amplitude, t[::subsample_factor].shape)
    )  # Interpolate noise to match the time array
    return np.full_like(t, 0.05) + noise  # Base y-value is 0.05 meters with added noise

# Time range for the motion
t = np.linspace(0, T, steps)

# Generate the trajectories for heat source 1 and the sawtooth
xx_traj1 = xx(t)
yy_traj1 = yy(t)
st_traj1 = sawtooth(t, period=0.00003, noise_amplitude=0.005)  # Higher frequency sawtooth with noise

# Generate the additional trajectories
horizontal_traj = horizontal_line(t)
sine_wave_traj = sine_wave(t)
noisy_horizontal_traj = noisy_horizontal_line(t)

# Center the trajectories around (0, 0)
xx_traj1_centered = xx_traj1 - np.mean(xx_traj1)
yy_traj1_centered = yy_traj1 - np.mean(yy_traj1)
st_traj1_centered = st_traj1 - np.mean(st_traj1)
horizontal_traj_centered = horizontal_traj - np.mean(horizontal_traj)
sine_wave_traj_centered = sine_wave_traj - np.mean(sine_wave_traj)
noisy_horizontal_traj_centered = noisy_horizontal_traj - np.mean(noisy_horizontal_traj)

# Scale the spiral trajectory to span 10 mm
xx_traj1_scaled = xx_traj1_centered * (10 / np.ptp(xx_traj1_centered))  # Scale x-axis to 10 mm
yy_traj1_scaled = yy_traj1_centered * (10 / np.ptp(xx_traj1_centered))  # Scale y-axis proportionally

# Generate the noisy spiral trajectory
x_noisy = x(t)
y_noisy = y(t)

# Center the noisy spiral trajectory around (0, 0)
x_noisy_centered = x_noisy - np.mean(x_noisy)
y_noisy_centered = y_noisy - np.mean(y_noisy)

# Scale the noisy spiral trajectory to span 10 mm
x_noisy_scaled = x_noisy_centered * (10 / np.ptp(xx_traj1_centered))  # Scale x-axis to 10 mm
y_noisy_scaled = y_noisy_centered * (10 / np.ptp(xx_traj1_centered))  # Scale y-axis proportionally

# Generate the sine wave trajectory with noise
sine_wave_reference, sine_wave_noisy = sine_wave_with_noise(t)

# Center the sine wave trajectories around (0, 0)
sine_wave_reference_centered = sine_wave_reference - np.mean(sine_wave_reference)
sine_wave_noisy_centered = sine_wave_noisy - np.mean(sine_wave_noisy)

# Scale the x-axis to span 10 mm
x_sine_wave = np.linspace(-5, 5, len(t))  # X-axis spans from -5 mm to 5 mm
y_sine_wave_reference = sine_wave_reference_centered * 1000  # Scale reference y-axis to mm
y_sine_wave_noisy = sine_wave_noisy_centered * 1000  # Scale noisy y-axis to mm

# Plot each trajectory in its own subplot
plt.figure(figsize=(12, 12))

# First subplot: Spiral Trajectory with Noise
plt.subplot(2, 2, 1)
plt.plot(xx_traj1_scaled, yy_traj1_scaled, linestyle='--', color='black', label='Reference Trajectory')
plt.plot(x_noisy_scaled, y_noisy_scaled, alpha=0.7, label='Noisy Trajectory')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.title('Spiral Trajectory with Noise')
plt.grid(True)

# Second subplot: Sawtooth Trajectory
plt.subplot(2, 2, 2)

# Generate the reference sawtooth trajectory without noise
st_traj1_reference = 0.15 * (t / (0.00003 * 10) - np.floor(0.5 + t / (0.00003 * 10)))  # Reference sawtooth
st_traj1_reference_centered = st_traj1_reference - np.mean(st_traj1_reference)  # Center the reference trajectory

# Scale the reference and noisy sawtooth trajectories
x_sawtooth = np.linspace(0, 10, len(t)) - 5  # Scale x-axis to 10 mm and center it at 0
y_sawtooth_reference = st_traj1_reference_centered * 1000  # Scale reference y-axis to 1 mm (amplitude)
y_sawtooth_noisy = st_traj1_centered * 1000  # Scale noisy y-axis to 1 mm (amplitude)

# Normalize and scale both trajectories to 1 mm
y_sawtooth_reference = y_sawtooth_reference / np.max(np.abs(y_sawtooth_reference)) * 1
y_sawtooth_noisy = y_sawtooth_noisy / np.max(np.abs(y_sawtooth_noisy)) * 1

# Plot the reference trajectory
plt.plot(x_sawtooth, y_sawtooth_reference, linestyle='--', color='black', label='Reference Trajectory')

# Plot the noisy trajectory
plt.plot(x_sawtooth, y_sawtooth_noisy, alpha=0.7, label='Noisy Trajectory')

# Add grid, labels, title, and legend
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.title('Sawtooth Trajectory')
plt.grid(True)

# Third subplot: Noisy Horizontal Line Trajectory with Reference
plt.subplot(2, 2, 3)
x_horizontal = np.linspace(-5, 5, len(t))  # X-axis spans from -5 mm to 5 mm
y_horizontal_reference = horizontal_traj_centered * 1000  # Reference horizontal line (scaled to mm)
y_horizontal_noisy = noisy_horizontal_traj_centered * 1000  # Noisy horizontal line (scaled to mm)

# Plot the reference trajectory
plt.plot(x_horizontal, y_horizontal_reference, linestyle='--', color='black', label='Reference Trajectory')

# Plot the noisy trajectory
plt.plot(x_horizontal, y_horizontal_noisy, alpha=0.7, label='Noisy Trajectory')

# Add grid, labels, and title
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.title('Noisy Horizontal Line Trajectory')
plt.grid(True)

# Fourth subplot: Sine Wave Trajectory
plt.subplot(2, 2, 4)
plt.plot(x_sine_wave, y_sine_wave_reference, linestyle='--', color='black', label='Reference Trajectory')
plt.plot(x_sine_wave, y_sine_wave_noisy, alpha=0.7, label='Noisy Trajectory')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.title('Sine Wave Trajectory with Noise')
plt.grid(True)

# Show the figure
plt.tight_layout()
plt.show()

def write_gcode(filename, x_values, y_values):
    """
    Writes G-code for a given trajectory to a file.
    
    Parameters:
        filename (str): The name of the output G-code file.
        x_values (array): Array of X coordinates.
        y_values (array): Array of Y coordinates.
    """
    with open(filename, 'w') as f:
        f.write("G21 ; Set units to mm\n")
        f.write("G90 ; Absolute positioning\n")
        f.write("G1 F1000 ; Set feedrate to 1000 mm/min\n")
        
        for x, y in zip(x_values, y_values):
            f.write(f"G1 X{x:.3f} Y{y:.3f}\n")
        
        f.write("M30 ; End of program\n")

# Generate G-code for each noisy trajectory
write_gcode("spiral_trajectory.gcode", x_noisy_scaled, y_noisy_scaled)
write_gcode("sawtooth_trajectory.gcode", x_sawtooth, y_sawtooth_noisy)
write_gcode("horizontal_line_trajectory.gcode", x_horizontal, y_horizontal_noisy)
write_gcode("sine_wave_trajectory.gcode", x_sine_wave, y_sine_wave_noisy)

print("G-code files generated successfully!")
