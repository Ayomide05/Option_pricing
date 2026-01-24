"""
Visualization of Black-Scholes Option Prices and Greeks
This module creates the key plots that every options trader and risk manager
should understand intuitively:
1. Option price vs stock price (the "hockey stick" payoff)
2. Greeks across stock price
3. Greeks across time to expiry
4. Price surfaces (3D)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

IMAGE_DIR = "images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Import our modules
from black_scholes import black_scholes_call, black_scholes_put
from greeks import (delta_call, delta_put, gamma, theta_call, theta_put, vega, rho_call, rho_put)

def plot_option_payoff_and_price(K=100, T=1, r=0.05, sigma=0.2, save_path=None):
    """
    Plot option price vs stock price, showing the relationship between
    the current option value and the payoff at expiry.
    """
    S = np.linspace(50, 150, 200)

    # Calculate prices
    call_prices = [black_scholes_call(s, K, T, r, sigma)for s in S]
    put_prices = [black_scholes_put(s, K, T, r, sigma) for s in S]

    # Payoffs at expiry
    call_payoff = np.maximum(S - K, 0)
    put_payoff = np.maximum(K - S, 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Call option
    ax1 = axes[0]
    ax1.plot(S, call_payoff, 'b--', label='Payoff at Expiry', linewidth=2)
    ax1.plot(S, call_prices, 'b-', label=f'Price (T={T}y)', linewidth=2)
    ax1.axvline(x=K, color='gray', linestyle=':', alpha=0.7, label='Strike')
    ax1.fill_between(S, call_prices, call_payoff, alpha=0.2, color='blue')
    ax1.set_xlabel('Stock Price ($)', fontsize=12)
    ax1.set_ylabel('Option Value ($)', fontsize=12)
    ax1.set_title('European Call Option', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(50, 150)
    ax1.set_ylim(-2, 55)

    # Annotate the time value
    ax1.annotate('Time Value', xy=(85, 5), fontsize=10, color='blue')
    ax1.annotate('Intrinsic\nValue', xy=(125, 20), fontsize=10, color='blue')
    
    # Put option
    ax2 = axes[1]
    ax2.plot(S, put_payoff, 'r--', label='Payoff at Expiry', linewidth=2)
    ax2.plot(S, put_prices, 'r-', label=f'Price (T={T}y)', linewidth=2)
    ax2.axvline(x=K, color='gray', linestyle=':', alpha=0.7, label='Strike')
    ax2.fill_between(S, put_prices, put_payoff, alpha=0.2, color='red')
    ax2.set_xlabel('Stock Price ($)', fontsize=12)
    ax2.set_ylabel('Option Value ($)', fontsize=12)
    ax2.set_title('European Put Option', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(50, 150)
    ax2.set_ylim(-2, 55)

    plt.suptitle(f'Option Value vs Stock Price (K=${K}, r={r:.1%}, σ={sigma:.1%})', 
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    

def plot_greeks_vs_stock_price(K=100, T=1, r=0.05, sigma=0.2, save_path=None):
    """
    Plot all Greeks as functions of stock price.
    
    This shows how option sensitivity changes as the option moves
    from out-of-the-money to in-the-money.
    """
    S = np.linspace(50, 150, 200)

    # Calculate Greeks for calls
    deltas = [delta_call(s, K, T, r, sigma) for s in S]
    gammas = [gamma(s, K, T, r, sigma) for s in S]
    thetas = [theta_call(s, K, T, r, sigma) / 365 for s in S]  # Per day
    vegas = [vega(s, K, T, r, sigma) / 100 for s in S]  # Per 1% vol
    rhos = [rho_call(s, K, T, r, sigma) / 100 for s in S]  # Per 1% rate

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Delta
    ax = axes[0, 0]
    ax.plot(S, deltas, 'b-', linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(x=K, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Stock Price ($)')
    ax.set_ylabel('Delta')
    ax.set_title('Delta (Δ) - Hedge Ratio')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Gamma
    ax = axes[0, 1]
    ax.plot(S, gammas, 'g-', linewidth=2)
    ax.axvline(x=K, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Stock Price ($)')
    ax.set_ylabel('Gamma')
    ax.set_title('Gamma (Γ) - Delta Sensitivity')
    ax.grid(True, alpha=0.3)
    
    # Theta
    ax = axes[0, 2]
    ax.plot(S, thetas, 'r-', linewidth=2)
    ax.axvline(x=K, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Stock Price ($)')
    ax.set_ylabel('Theta ($/day)')
    ax.set_title('Theta (Θ) - Time Decay')
    ax.grid(True, alpha=0.3)
    
    # Vega
    ax = axes[1, 0]
    ax.plot(S, vegas, 'm-', linewidth=2)
    ax.axvline(x=K, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Stock Price ($)')
    ax.set_ylabel('Vega ($/1% vol)')
    ax.set_title('Vega (ν) - Volatility Sensitivity')
    ax.grid(True, alpha=0.3)
    
    # Rho
    ax = axes[1, 1]
    ax.plot(S, rhos, 'c-', linewidth=2)
    ax.axvline(x=K, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel('Stock Price ($)')
    ax.set_ylabel('Rho ($/1% rate)')
    ax.set_title('Rho (ρ) - Interest Rate Sensitivity')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Call Option Greeks vs Stock Price (K=${K}, T={T}y, σ={sigma:.1%})', 
                 fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    

def plot_greeks_vs_time(S=100, K=100, r=0.05, sigma=0.2, save_path=None):
    """
    Plot Greeks as functions of time to expiry.
    
    This shows how option behavior changes as expiry approaches.
    Critical for understanding time decay and gamma risk.
    """
    T = np.linspace(0.01, 2, 200)  # Avoid T=0
    
    # Calculate Greeks
    deltas = [delta_call(S, K, t, r, sigma) for t in T]
    gammas = [gamma(S, K, t, r, sigma) for t in T]
    thetas = [theta_call(S, K, t, r, sigma) / 365 for t in T]  # Per day
    vegas = [vega(S, K, t, r, sigma) / 100 for t in T]  # Per 1% vol
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Delta vs Time
    ax = axes[0, 0]
    ax.plot(T, deltas, 'b-', linewidth=2)
    ax.set_xlabel('Time to Expiry (years)')
    ax.set_ylabel('Delta')
    ax.set_title('Delta vs Time (ATM Call)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    
    # Gamma vs Time
    ax = axes[0, 1]
    ax.plot(T, gammas, 'g-', linewidth=2)
    ax.set_xlabel('Time to Expiry (years)')
    ax.set_ylabel('Gamma')
    ax.set_title('Gamma vs Time (ATM Call)')
    ax.grid(True, alpha=0.3)
    
    # Theta vs Time
    ax = axes[1, 0]
    ax.plot(T, thetas, 'r-', linewidth=2)
    ax.set_xlabel('Time to Expiry (years)')
    ax.set_ylabel('Theta ($/day)')
    ax.set_title('Theta vs Time (ATM Call)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Vega vs Time
    ax = axes[1, 1]
    ax.plot(T, vegas, 'm-', linewidth=2)
    ax.set_xlabel('Time to Expiry (years)')
    ax.set_ylabel('Vega ($/1% vol)')
    ax.set_title('Vega vs Time (ATM Call)')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Greeks vs Time to Expiry (S=${S}, K=${K}, ATM)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    


def plot_price_surface(K=100, r=0.05, sigma=0.2, save_path=None):
    """
    3D surface plot of option price as function of stock price and time.
    
    This visualization shows the entire option value landscape.
    """
    S = np.linspace(50, 150, 50)
    T = np.linspace(0.01, 2, 50)
    
    S_grid, T_grid = np.meshgrid(S, T)
    
    # Calculate call prices on the grid
    call_prices = np.zeros_like(S_grid)
    for i in range(len(T)):
        for j in range(len(S)):
            call_prices[i, j] = black_scholes_call(S[j], K, T[i], r, sigma)
    
    fig = plt.figure(figsize=(14, 6))
    
    # 3D Surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(S_grid, T_grid, call_prices, cmap=cm.viridis,
                            linewidth=0, antialiased=True, alpha=0.8)
    ax1.set_xlabel('Stock Price ($)')
    ax1.set_ylabel('Time to Expiry (years)')
    ax1.set_zlabel('Call Price ($)')
    ax1.set_title('Call Option Price Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(S_grid, T_grid, call_prices, levels=20, cmap=cm.viridis)
    ax2.axvline(x=K, color='white', linestyle='--', alpha=0.7, label='Strike')
    ax2.set_xlabel('Stock Price ($)')
    ax2.set_ylabel('Time to Expiry (years)')
    ax2.set_title('Call Price Contours')
    fig.colorbar(contour, ax=ax2, label='Call Price ($)')
    ax2.legend()
    
    plt.suptitle(f'Black-Scholes Call Price Surface (K=${K}, r={r:.1%}, σ={sigma:.1%})', 
                 fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    


def plot_gamma_surface(K=100, r=0.05, sigma=0.2, save_path=None):
    """
    3D surface plot of gamma - shows the "gamma explosion" near expiry.
    
    This is crucial for risk management: gamma becomes extreme for
    ATM options as expiry approaches, making delta hedging difficult.
    """
    S = np.linspace(80, 120, 50)
    T = np.linspace(0.01, 1, 50)
    
    S_grid, T_grid = np.meshgrid(S, T)
    
    gamma_values = np.zeros_like(S_grid)
    for i in range(len(T)):
        for j in range(len(S)):
            gamma_values[i, j] = gamma(S[j], K, T[i], r, sigma)
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D Surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(S_grid, T_grid, gamma_values, cmap=cm.plasma,
                            linewidth=0, antialiased=True, alpha=0.8)
    ax1.set_xlabel('Stock Price ($)')
    ax1.set_ylabel('Time to Expiry (years)')
    ax1.set_zlabel('Gamma')
    ax1.set_title('Gamma Surface')
    ax1.view_init(elev=25, azim=45)
    
    # Contour
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(S_grid, T_grid, gamma_values, levels=20, cmap=cm.plasma)
    ax2.axvline(x=K, color='white', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Stock Price ($)')
    ax2.set_ylabel('Time to Expiry (years)')
    ax2.set_title('Gamma Contours')
    fig.colorbar(contour, ax=ax2, label='Gamma')
    
    plt.suptitle('Gamma Explosion Near Expiry (ATM Options)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    

if __name__ == "__main__":
    print("Generating Option Pricing Visualizations...")
    print("=" * 50)
    
    # Generate all plots
    print("\n1. Option Payoff and Price...")
    plot_option_payoff_and_price(save_path=os.path.join(IMAGE_DIR, '01_option_payoff_price.png'))
    
    print("\n2. Greeks vs Stock Price...")
    plot_greeks_vs_stock_price(save_path=os.path.join(IMAGE_DIR, '02_greeks_vs_stock.png'))
    
    print("\n3. Greeks vs Time...")
    plot_greeks_vs_time(save_path=os.path.join(IMAGE_DIR, '03_greeks_vs_time.png'))
    
    print("\n4. Price Surface...")
    plot_price_surface(save_path=os.path.join(IMAGE_DIR, '04_price_surface.png'))
    
    print("\n5. Gamma Surface...")
    plot_gamma_surface(save_path=os.path.join(IMAGE_DIR, '05_gamma_surface.png'))
    
    print("\nAll visualizations complete!")
