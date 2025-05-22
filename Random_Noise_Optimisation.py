# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
import matplotlib.animation as animation

#A program to predict the transient thermal response of a moving 2D heat source



# Define the Gaussian function
def gaussian(x, y, mu_x, mu_y, sigma):
    return np.exp(-((x - mu_x)**2 / (2 * sigma**2) + (y - mu_y)**2 / (2 * sigma**2)))



# Parameters
Lx, Ly = 7, 7  # Reduce domain to focus on 5 mm track (+1 mm margin each side)
Nx, Ny = 150, 150 
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = 0.000002
T = 0.05   # Time for 5 mm at 100 mm/s = 0.05 s (adjust as needed)
Nt = int(T / dt) + 1

density = 7.8e-6 # kg/mm^3
thermal_conductivity = 45e-3 # W/mmK
specific_heat = 420 # J/kgK

alpha = thermal_conductivity/(specific_heat*density)  # Thermal diffusivity mm^2/s

thickness = 1 #plate thickness 

# Parameters for the Gaussian#
D1 = 0.080 # 80um beam diameter
mu_x = 1  # Start 1 mm from left edge
mu_y = Ly / 2  # Centered in y
sigma = D1/2.355 # Standard deviation of the Gaussian beam


# Moving heat source parameters
source_amplitude = 100  # Peak power (W)
source_speed = 300  # mm/s

track_length = 5  # mm, length of the scan track

# Coordinate Matrices
x_mat, y_mat = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))

# Initial temperature distribution
u = np.zeros((Nt, Nx, Ny))
heat_source = np.zeros((Nt, Nx, Ny))
# Set Boundary Condition and initial state
initial_temperature = 20
u[0,:,:] = np.ones((Nx, Ny))*initial_temperature

   
# Starting heat source            
heat_source_temp = gaussian(x_mat, y_mat, mu_x, mu_y, sigma)    
heat_source_temp = heat_source_temp*source_amplitude
heat_source[0,:,:] = heat_source_temp
u_temp = heat_source_temp * dt / (dx*dy*thickness*density*specific_heat)
u[0, :, :] = u[0, :, :] + u_temp
mu_y = mu_y + (source_speed*dt)


# visualise the heat source
from mpl_toolkits.mplot3d import Axes3D

# Define the grid for the surface plot
X, Y = np.meshgrid(np.linspace(-D1*2, D1*2, 100), np.linspace(-D1*2, D1*2, 100))
Z1 = gaussian(X, Y, 0, 0, sigma) * source_amplitude

# Find the common color map range
vmin = Z1.min()
vmax = Z1.max()

fig = plt.figure(figsize=(12, 6))  # Reduced figure size for smaller plots

# 3D plot for Heat Source 1
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z1, cmap='hot', vmin=vmin, vmax=vmax)
ax1.set_title('Heat Intensity Distribution - Heat Source 1', fontsize=10)
ax1.set_xlabel('X Position (mm)', fontsize=8)
ax1.set_ylabel('Y Position (mm)', fontsize=8)
ax1.set_zlabel('Intensity (W/mm^2)', fontsize=8)
ax1.tick_params(axis='both', which='major', labelsize=6)

# 2D plot for Heat Source 1
ax2 = fig.add_subplot(122)
c = ax2.pcolormesh(X, Y, Z1, shading='auto', cmap='hot', vmin=vmin, vmax=vmax)
ax2.set_title('2D Heat Intensity Distribution - Heat Source 1', fontsize=10)
ax2.set_xlabel('X Position (mm)', fontsize=8)
ax2.set_ylabel('Y Position (mm)', fontsize=8)
ax2.tick_params(axis='both', which='major', labelsize=6)

# Add a color bar for the 2D plot
fig.colorbar(c, ax=ax2, label='Intensity (W/mm^2)', fraction=0.046, pad=0.04)

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.5, hspace=0.4)  # Add horizontal spacing between the two plots

plt.show()


# Coefficient matrix A
A = lil_matrix((Nx*Ny, Nx*Ny))
for i in range(1, Nx-1):
    for j in range(1, Ny-1):
        index = i * Ny + j
        A[index, index] = -2 * alpha * dt / dx**2 - 2 * alpha * dt / dy**2 - 1
        A[index, index - Ny] = alpha * dt / dx**2
        A[index, index + Ny] = alpha * dt / dx**2
        A[index, index - 1] = alpha * dt / dy**2
        A[index, index + 1] = alpha * dt / dy**2
        
T_track = 0

# Time-stepping loop
for n in range(1, Nt, 1):
    b = u[n-1,:,:].flatten()
    deltaT = A.dot(b)
    T_new_flattened = b + (deltaT * dt)
    u[n,:,:] = T_new_flattened.reshape((Nx, Ny))
    T_track = T_track + dt
    # Only apply heat source while within 5 mm track
    if mu_x <= 1 + track_length:
        heat_source_temp = gaussian(x_mat, y_mat, mu_x, mu_y, sigma)
        heat_source_temp = heat_source_temp * source_amplitude
        heat_source[n,:,:] = heat_source_temp
        u_temp = heat_source_temp * dt / (dx*dy*thickness*density*specific_heat)
        u[n, :, :] = u[n, :, :] + u_temp
        u[n, 0, :] = initial_temperature
        u[n, Nx-1, :] = initial_temperature
        u[n, :, 0] = initial_temperature
        u[n, :, Ny-1] = initial_temperature
        mu_x = mu_x + (source_speed * dt)
    print(n)

# After the simulation loop
max_temp = np.max(u)
print(f"Maximum temperature reached during the scan: {max_temp:.2f} °C")

# Set up the figure and axis for animation
fig, ax = plt.subplots(figsize=(7, 6))
vmin = 0
vmax = int(np.ceil(max_temp / 500.0)) * 500 # Set vmax to the next highest multiple of 500 above max_temp

im = ax.imshow(u[0, :, :].T, cmap='hot', vmin=vmin, vmax=vmax, origin='lower',
               extent=[0, Lx, 0, Ly], aspect='auto')
cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Temperature (deg C)', fontsize=10)
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_title('Temperature Distribution (Animated)')

def animate(frame):
    im.set_data(u[frame, :, :].T)
    ax.set_title(f'Temperature Distribution\nt={frame*dt:.2f}s')
    return [im]

frame_interval = 10
frames = range(0, Nt, frame_interval)

ani = animation.FuncAnimation(fig, animate, frames=frames, blit=False, interval=50)

plt.show()