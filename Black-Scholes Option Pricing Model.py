from gettext import install

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


# Black-Scholes Option Pricing Model
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


# Option Greeks
def delta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) * 0.01


def theta(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)

    return theta / 365  # Daily theta


def rho(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        rho_val = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
    else:
        rho_val = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01

    return rho_val


# EDIT PARAMETERS
# Change parameters here!
S = 130  # Spot price
r = 0.05  # Risk-free rate
sigma = 0.5  # Volatility
option_type = 'put'

# CREATES STRIKE AND MATURITY GRIDS
strikes = np.linspace(70, 130, 25)
maturities = np.linspace(0.1, 2, 25)  # 0.1 to 2 years
K_grid, T_grid = np.meshgrid(strikes, maturities)

# CALCULATES GREEK SYMBOLS
delta_grid = np.zeros_like(K_grid)
gamma_grid = np.zeros_like(K_grid)
vega_grid = np.zeros_like(K_grid)
theta_grid = np.zeros_like(K_grid)
rho_grid = np.zeros_like(K_grid)

for i in range(K_grid.shape[0]):
    for j in range(K_grid.shape[1]):
        delta_grid[i, j] = delta(S, K_grid[i, j], T_grid[i, j], r, sigma, option_type)
        gamma_grid[i, j] = gamma(S, K_grid[i, j], T_grid[i, j], r, sigma)
        vega_grid[i, j] = vega(S, K_grid[i, j], T_grid[i, j], r, sigma)
        theta_grid[i, j] = theta(S, K_grid[i, j], T_grid[i, j], r, sigma, option_type)
        rho_grid[i, j] = rho(S, K_grid[i, j], T_grid[i, j], r, sigma, option_type)

# CREATES HEATMAPS
greeks = {
    'Delta': delta_grid,
    'Gamma': gamma_grid,
    'Vega': vega_grid,
    'Theta': theta_grid,
    'Rho': rho_grid
}

plt.figure(figsize=(15, 10))
for i, (greek_name, greek_values) in enumerate(greeks.items(), 1):
    plt.subplot(2, 3, i)
    sns.heatmap(greek_values,
                xticklabels=[f"{k:.0f}" for k in strikes],
                yticklabels=[f"{t:.1f}" for t in maturities],
                cmap='viridis', annot=False, cbar=True)
    plt.title(f'{greek_name} Heatmap')
    plt.xlabel('Strike Price')
    plt.ylabel('Time to Maturity (years)')
    plt.gca().invert_yaxis()

plt.tight_layout()
plt.suptitle(f'Black-Scholes Option Greeks Heatmaps (S={S}, r={r}, σ={sigma}, {option_type})', y=1.02)
plt.figtext(0.95, 0.02, "Property of Thomas David Blomfield",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white",
                      edgecolor="black",
                      alpha=0.5),
            fontsize=8)

plt.show()
plt.show()
