# -*- coding: utf-8 -*-
"""
Spyder Editor
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix

#A program to predict the transient thermal response of a moving 2D heat source



# Define the Gaussian function
def gaussian(x, y, mu_x, mu_y, sigma):
    return np.exp(-((x - mu_x)**2 / (2 * sigma**2) + (y - mu_y)**2 / (2 * sigma**2)))



# Parameters
Lx, Ly = 100, 100  # Length of the plane in x and y directions
T = 4.0   # Total time
Nx, Ny = 500, 500  # Number of spatial points in x and y directions
Nt = (np.rint(40*T)).astype(int)  # Number of time points

density = 7.8e-6 # kg/mm^3
thermal_conductivity = 45e-3 # W/mmK
specific_heat = 420 # J/kgK

alpha = thermal_conductivity/(specific_heat*density)  # Thermal diffusivity mm^2/s

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = T / (Nt - 1)

thickness = 1 #plate thickness 

# Parameters for the Gaussian#
D1 = 0.080 # 80um beam diameter
mu_x = 30
mu_y = 30
sigma = D1/2.355 # Standard deviation of the Gaussian beam


# Moving heat source parameters
source_amplitude = 200  # W/mm^2

source_speed = 30 # mm/s



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
u_temp = heat_source_temp*(1/(dx*dy*thickness*density))*(1/specific_heat)
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
for n in range(1, Nt,1):
    b = u[n-1,:,:].flatten()
    deltaT=A.dot(b)
    T_new_flattened=b+(deltaT*dt)
    u[n,:,:]=T_new_flattened.reshape((Nx, Ny))
    T_track=T_track+dt
    if T_track<2:
        # Evaluate heat source
        heat_source_temp = gaussian(x_mat, y_mat, mu_x, mu_y, sigma)    
        heat_source_temp = heat_source_temp*source_amplitude
        heat_source[n,:,:] = heat_source_temp
        # Apply heat source to temperature field
        u_temp=heat_source_temp*(1/(dx*dy*thickness*density))*(1/specific_heat)
        u[n, :, :]=u[n, :, :]+u_temp
        #Ensure outer edge is constant temperature
        u[n, 0, :]=initial_temperature
        u[n, Nx-1, :]=initial_temperature
        u[n, :, 0]=initial_temperature
        u[n, :, Ny-1]=initial_temperature
        # Update heat source location
        mu_x=mu_x+(source_speed*dt)
        mu_y=mu_y+(source_speed*dt)
    print(n)


vmin = 0
vmax = 800
colorbar_ticks = np.round(np.linspace(vmin, vmax, 6))

# Create a single figure with subplots
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6), constrained_layout=True)  # Adjust rows/cols as needed
axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

# Plot the results
for idx, n in enumerate(range(0, Nt, 10)):
    if idx >= len(axes):  # Ensure we don't exceed the number of subplots
        break
    ax = axes[idx]
    contour = ax.contourf(x_mat, y_mat, u[n, :, :], cmap='hot', levels=np.linspace(vmin, vmax, 50), vmin=vmin, vmax=vmax)
    ax.set_title(f't={n*dt:.2f}s', fontsize=8)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('y', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)

# Add a single color bar for all subplots
cbar = fig.colorbar(contour, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1, ticks=colorbar_ticks)
cbar.ax.set_xticklabels([str(tick) for tick in colorbar_ticks])
cbar.set_label('Temperature (deg C)', fontsize=10)

plt.suptitle('Temperature Distribution Over Time', fontsize=12)
plt.show()
