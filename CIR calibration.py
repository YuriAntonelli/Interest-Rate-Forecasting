# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 11:05:07 2023

@author: yuria
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.special import iv
from scipy.optimize import minimize

#------------------------------------------------------------------------------
def simulate_CIR(k, theta, sigma, r0, T, N):
    dt = T / N
    interest_rate_paths = np.zeros(N+1)
    interest_rate_paths[0] = r0 
    for t in range(1, N+1):
        Z = np.random.randn()
        r = interest_rate_paths[t-1]
        interest_rate_paths[t] = r + k * (theta-r) * dt + sigma * np.sqrt(dt) * np.sqrt(max(0, r)) * Z    
    return interest_rate_paths
#------------------------------------------------------------------------------
def CIRols(data, dt):
    Nsteps = len(data)
    rs = data[:Nsteps - 1]  
    rt = data[1:Nsteps]
    
    model = LinearRegression()

    # feature engeneering to fit the theoretical model
    y = (rt - rs) / np.sqrt(rs)
    z1 = dt / np.sqrt(rs)
    z2 = dt * np.sqrt(rs)
    X = np.column_stack((z1, z2))

    # Build the model
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # Calculate the predicted values (y_hat), residuals and the parameters
    y_hat = model.predict(X)
    residuals = y - y_hat
    beta1 = model.coef_[0]        
    beta2 = model.coef_[1]

    # get the parameter of interest for CIR
    k0 = -beta2
    theta0 = beta1/k0
    sigma0 = np.std(residuals)/np.sqrt(dt)
    
    return k0, theta0, sigma0
#------------------------------------------------------------------------------
def CIRlog(data, dt, k, theta, sigma):
    Nsteps = len(data)
    rs = data[:Nsteps - 1]  # Empirical data
    rt = data[1:Nsteps]

    c = (2 * k) / ((sigma ** 2) * (1 - np.exp(-k * dt)))  # dt = t - s
    u = c * np.exp(-k * dt) * rs
    v = c * rt
    q = ((2 * k * theta) / (sigma ** 2)) - 1
    z = 2 * np.sqrt(u * v)
    bf = iv(q, z)  # Modified Bessel function
    logL = (Nsteps) * np.log(c) + np.sum(-u - v + 0.5 * q * np.log(v / u) + np.log(bf))
    return logL
#------------------------------------------------------------------------------
def neg_log_likelihood(params):
    return -CIRlog(data, dt, params[0], params[1], params[2])
#------------------------------------------------------------------------------

#---------------------------#
#-Simulation and Estimation-#
#---------------------------#

# Define the theoretical parameters
k_true = 7 # True mean reversion speed
theta_true = 0.05  # True long-term mean
sigma_true = 0.5  # True volatility of interest rates
r0_true = 0.3  # True initial interest rate
T = 1  # Time horizon
N = 100  # Number of time steps
dt = T/N

# simulated path
np.random.seed(123)
data = simulate_CIR(k_true, theta_true, sigma_true, r0_true, T, N)

# OLS estimation
initialvalues = CIRols(data, dt)
k_ols, theta_ols, sigma_ols = initialvalues

#MLE estimation
params_optim = minimize(neg_log_likelihood, initialvalues, method='Nelder-Mead').x
k_mle, theta_mle, sigma_mle = params_optim

# print results
print(f'The true parameters are: \n k:{k_true}, theta:{theta_true}, sigma:{sigma_true} \n -------------------')
print(f'The OLS estimates are: \n k:{k_ols}, theta:{theta_ols}, sigma:{sigma_ols} \n -------------------')
print(f'The MLE estimates are: \n k:{k_mle}, theta:{theta_mle}, sigma:{sigma_mle} \n -------------------')

#------------------------------------------------------------------------------
"""
time_index = np.linspace(0, 1, 101)
# Convert NumPy array to a Pandas DataFrame
df = pd.DataFrame({'Time': time_index, 'Value': data})

# Set the Date column as the index
df.set_index('Time', inplace=True)

# Create the time series plot using Seaborn
plt.figure(figsize=(13, 8))  # Adjust figure size if needed
sns.lineplot(data=df, x=df.index, y='Value')

# Set plot title and labels
plt.title('Simulated CIR', fontsize=(20))
plt.xlabel('Time')
plt.ylabel('Interest Rate')

# Show plot
plt.tight_layout()
plt.show()

#-----------------------------------------------------------------
#---------------------------------#
#-Plot of the likelihood function-#
#---------------------------------#
#---------#
#-3D plot-#
#---------#
# Define array length
array_length = 250
sigma_v = np.linspace(0.2, 0.6, array_length)
k_v = np.linspace(5, 12, array_length)
theta_v = np.linspace(0.01, 0.09, array_length)

def plot_with_fixed_sigma(sigma_value):
    # Create a meshgrid for 3D plotting
    grid1, grid2 = np.meshgrid(k_v, theta_v)

    # Create an empty grid for log-likelihood values
    log_lik = np.zeros((array_length, array_length))

    # Calculate log-likelihood for all combinations of k and theta
    for i in range(array_length):
        for j in range(array_length):
            params = (k_v[i], theta_v[j], sigma_value)
            log_lik[i, j] = -neg_log_likelihood(params)

    # Create a 3D surface plot of the log-likelihood
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(grid1, grid2, log_lik, cmap='viridis', rstride=1, cstride=1, alpha=0.8)

    # Customize the plot
    ax.set_xlabel('K')
    ax.set_ylabel('Theta')
    ax.set_zlabel('Log-Likelihood')
    plt.title('Log-likelihood with fixed Sigma', fontsize=24)

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Show the plot
    plt.show()

def plot_with_fixed_k(k_value):
    # Create a meshgrid for 3D plotting
    grid1, grid2 = np.meshgrid(sigma_v, theta_v)

    # Create an empty grid for log-likelihood values
    log_lik = np.zeros((array_length, array_length))

    # Calculate log-likelihood for all combinations of sigma and theta
    for i in range(array_length):
        for j in range(array_length):
            params = (k_value, theta_v[j], sigma_v[i])
            log_lik[i, j] = -neg_log_likelihood(params)

    # Create a 3D surface plot of the log-likelihood
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(grid1, grid2, log_lik, cmap='viridis', rstride=1, cstride=1, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Sigma')
    ax.set_ylabel('Theta')
    ax.set_zlabel('Log-Likelihood')
    plt.title('Log-likelihood with fixed K', fontsize=24)

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_with_fixed_theta(theta_value):
    # Create a meshgrid for 3D plotting
    grid1, grid2 = np.meshgrid(sigma_v, k_v)

    # Create an empty grid for log-likelihood values
    log_lik = np.zeros((array_length, array_length))

    # Calculate log-likelihood for all combinations of sigma and k
    for i in range(array_length):
        for j in range(array_length):
            params = (k_v[j], theta_value, sigma_v[i])
            log_lik[i, j] = -neg_log_likelihood(params)

    # Create a 3D surface plot of the log-likelihood
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(grid1, grid2, log_lik, cmap='viridis', rstride=1, cstride=1, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Sigma')
    ax.set_ylabel('K')
    ax.set_zlabel('Log-Likelihood')
    plt.title('Log-likelihood with fixed Theta', fontsize=24)

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

# Generate plots with fixed parameters
plot_with_fixed_sigma(sigma_mle)
plot_with_fixed_k(k_mle)
plot_with_fixed_theta(theta_mle)


#------------------------------------------------------------------------------

#---------#
#-2D Plot-#
#---------#

# 2D graph with theta
theta_v = np.linspace(0.01, 0.1, 10000)
likelihood = np.zeros(10000)
for n in range(len(theta_v)):
    params = (k_true, theta_v[n], sigma_true)
    likelihood[n] = CIRlog(data, dt, params[0], params[1], params[2])

plt.figure(figsize=(15,8))
plt.plot(theta_v, likelihood, marker='o', linestyle='-')
plt.xlabel('theta', fontsize=20)  # Replace with your x-axis label
plt.ylabel('log-likelihood', fontsize=20)  # Replace with your y-axis label
plt.grid(True)  # Add gridlines if needed
plt.show()

# 2D graph with k
k_v = np.linspace(-15, 15, 10000)
likelihood = np.zeros(10000)
for n in range(len(k_v)):
    params = (k_v[n], theta_true, sigma_true)
    likelihood[n] = CIRlog(data, dt, params[0], params[1], params[2])

plt.figure(figsize=(15,8))
plt.plot(k_v, likelihood, marker='o', linestyle='-')
plt.xlabel('k', fontsize=20)  # Replace with your x-axis label
plt.ylabel('log-likelihood', fontsize=20)  # Replace with your y-axis label
plt.grid(True)  # Add gridlines if needed
plt.show()

# 2D graph with sigma
sigma_v = np.linspace(0.3, 1, 10000)
likelihood = np.zeros(10000)
for n in range(len(sigma_v)):
    params = (k_true, theta_true, sigma_v[n])
    likelihood[n] = CIRlog(data, dt, params[0], params[1], params[2])

plt.figure(figsize=(15,8))
plt.plot(sigma_v, likelihood, marker='o', linestyle='-')
plt.xlabel('sigma', fontsize=20)  # Replace with your x-axis label
plt.ylabel('log-likelihood', fontsize=20)  # Replace with your y-axis label
plt.grid(True)  # Add gridlines if needed
plt.show()
"""























