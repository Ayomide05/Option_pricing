"""
Implied Volatility Calculator
This module implements implied volatiltiy calculation using the Newton-Raphson method.
Given a market price, we find the volatility that makes Black-scholes match that price.
The key insight: The derivative of option price with respect to volatility is vega,
which we already know how to calculate analytically (phase 1). This makes Newton-Raphson
converge very quickly (typically 2 - 4 iterations).
"""

import numpy as np
from scipy.stats import norm
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# Add parent folder to python's search path
sys.path.append(parent_dir)

from phase1_black_scholes.black_scholes import black_scholes_call, black_scholes_put
from phase1_black_scholes.greeks import vega

def implied_volatility_call(market_price: float, S: float, K: float, T: float, r: float, tol: float = 1e-6,
                            max_iter: int = 100, initial_guess: float = 0.2) -> float:
    """
    Calculate implied volatility for a European call option using Newton-Raphson.
    Parameters
    ------------
    market_price : float  (The observed market price of the call option)
    S : float   (Current stock price)
    K : float (Strike price)
    T : float ( Time to expiry in years)
    r : float ( risk-free interest rate)
    tol : float ( Convergence tolerance (default 1e-6))
    max_iter : int  (Maximum Iterations (default 100))
    initial_guess : float   (Starting volatility guess (defult 0.2 = 20%))

    Returns
    ----------
    float  (implied volatility ( as decimal, e.g 0.25 for 25%))
    
    Raises
    ------
    ValueError    if the algorithim fails to converge or inputs are invalid
        
    """

    # Validate inputs
    if market_price <= 0:
        raise ValueError("market price must be positive")
    if T <= 0:
        raise ValueError("Time to expiry must be positive")
    if S <= 0 or K <= 0:
        raise ValueError("Stock price and strike must be positive")
    
    # Check arbitrage bounds
    # Call price muse be at least max(S - K*exp(-rT), 0) and at most S
    lower_bound = max(S - K * np.exp(-r * T), 0)
    upper_bound = S

    if market_price < lower_bound:
        raise ValueError(f" Market price {market_price:.4f} is below arbitrage lower bound {lower_bound:.4f}")
    if market_price > upper_bound:
        raise ValueError(f"Market price {market_price:.4f} is above arbitrage upper bound {upper_bound:.4f}")
    
    # Newton-raphson iteration
    sigma = initial_guess

    for i in range(max_iter):
        # Calcultae BS price and vega at current sigma
        bs_price = black_scholes_call(S, K, T, r, sigma)
        bs_vega = vega(S, K, T, r, sigma)

        # Calculate the difference
        diff = bs_price - market_price

        # Check convergence
        if abs(diff) < tol:
            return sigma
        
        # Avoid division by zero (vega near zero for deep ITM/OTM)
        if bs_vega < 1e-10:
            # Switch to bisection or adjust guess
            if diff > 0:
                sigma = sigma * 0.5    # Price too high, reduce vol
            else:
                sigma = sigma * 1.5    # Price too low, increase vol
            continue

        # Newton-Raphson update
        sigma = sigma - diff / bs_vega

        # Keep sigma in reasonable bounds
        sigma = max(sigma, 1e-6)   # Minimum 0.0001%
        sigma = min(sigma, 5.0)    # Maximum 500%

    raise ValueError(f"Failed to converge after {max_iter} iterations. Last sigma: {sigma:.4f}")

def implied_volatility_put(market_price: float, S: float, K: float, T: float, r: float,
                           tol: float = 1e-6, max_iter: int = 100,
                           initial_guess: float = 0.2) -> float:
    """
    Calculate implied volatility for a European put option using Newton-Raphson.
    parameters
    ----------
    market_price : float
        The observed market price of the put option
    S : float   (Current stock price)
    K : float   ( Strike price)
    T : float   (Tikme to expiry in years)
    r : float   (Risk-free interest rate)
    tol : float (Convergence tolerance)
    max_iter : int (Maximum iterations)
    initial_guess : float  (Starting Volatility guess)
    
    Returns
    -------
    float : Implied volatility  (as decimal)
    """

    # Validate inputs
    if market_price <= 0:
        raise ValueError("Market price must be positive")
    if T <= 0:
        raise ValueError("Time to expiry must be positive")
    
    # Check arbitrage bounds for put
    # Put price must be at least maxmax(K*exp(-rT) - S, 0) and at most K*exp(-rT)
    lower_bound = max(K * np.exp(-r * T) - S, 0)
    upper_bound = K * np.exp(-r * T)

    if market_price < lower_bound:
        raise ValueError(f" Market price {market_price:.4f} is below arbitrage lower bound {lower_bound:.4f}")
    if market_price > upper_bound:
       raise ValueError(f"Market price {market_price:.4f} is above arbitrage upper bound {upper_bound:.4f}")
    
    # Newton-Raphson guess
    sigma = initial_guess

    for i in range(max_iter):
        bs_price = black_scholes_put(S, K, T, r, sigma)
        bs_vega = vega(S, K, T, r, sigma)   # Vega is same for calls and puts

        diff = bs_price - market_price

        if abs(diff) < tol:
            return sigma
        
        if bs_vega < 1e-10:
            if diff > 0:
                sigma = sigma * 0.5
            else:
                sigma = sigma * 1.5
            continue

        sigma = sigma - diff / bs_vega
        sigma = max(sigma, 1e-6)
        sigma = min(sigma, 5.0)

    raise ValueError(f"Failed to converge after {max_iter} iterations. Last sigma: {sigma:.4f}")

def implied_volatility(market_price: float, S:float, K:float, T: float,
                       r: float, option_type: str = 'call', **kwargs) -> float:
    
    """Calculate implied volatility for a European option
    ** kwargs    Additional arguements passed to the specific IV function
    """
    if option_type.lower() == 'call':
        return implied_volatility_call(market_price, S, K, T, r, **kwargs)
    elif option_type.lower() == 'put':
        return implied_volatility_put(market_price, S, K, T, r, **kwargs)
    else:
        raise ValueError("option_type must be 'call' or 'put")


def implied_volatility_bisection(market_price: float, S: float, K: float, T: float, r: float,
                                 option_type: str = 'call', tol: float = 1e-6,
                                 max_iter: int = 100) -> float:
    """
    Calculate implied volatility using bisection method.
    This method is more slower than Newton-Raphson but more robust.
    This is useful as a fallback when Newton-Raphson fails to converge
    """

    if option_type.lower() == 'call':
        price_func = black_scholes_call
    else:
        price_func = black_scholes_put

    # Initial bounds
    sigma_low = 1e-6
    sigma_high = 5.0

    for i in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        price_mid = price_func(S, K, T, r, sigma_mid)

        if abs(price_mid - market_price) < tol:
            return sigma_mid
        
        if price_mid > market_price:
            sigma_high = sigma_mid
        else: 
            sigma_low = sigma_mid

        if sigma_high - sigma_low < tol:
            return sigma_mid
        
    raise ValueError(f"Bisection failed to converge after {max_iter} iterations")

if __name__ == "__main__":
    print("Implied Volatility Calculator")
    print("=" * 50)

    # Test parameters
    S = 100
    K = 100
    T = 1
    r = 0.05

    # First, calculate BS price at known volatility
    known_vol = 0.20
    bs_price = black_scholes_call(S, K, T, r, known_vol)
    print(f"\nTest 1: Recovering known volatility")
    print(f" Known volatility: {known_vol:.1%}")
    print(f" BS Price at {known_vol:.1%} vol: ${bs_price:.4f}")

    # Now recover the volatility from the price
    recovered_vol = implied_volatility_call(bs_price, S, K, T, r)
    print(f"  Recovered volatility: {recovered_vol:.4%}")
    print(f"  Error: {abs(recovered_vol - known_vol):.2e}")

    # Test with different market prices
    print(f"\nTest 2: Different market prices")
    print("-" * 50)
    print(f"{'Market Price':>12} {'Implied Vol':>12} {'Verification':>12}")
    print("-" * 50)
    
    test_prices = [5.0, 8.0, 10.45, 15.0, 20.0]

    for price in test_prices:
        try:
            iv = implied_volatility_call(price, S, K, T, r)
            verify_price = black_scholes_call(S, K, T, r, iv)
            print(f"${price:>11.2f} {iv:>11.2%} ${verify_price:>11.2f}")
        except ValueError as e:
            print(f"${price:>11.2f} {'Error':>12}: {e}")

    
    # Test convergence speed
    print(f"\nTest 3: Convergence analysis")
    print("-" * 50)
    
    market_price = 10.45
    sigma = 0.5  # Start with wrong guess
    
    print(f"Starting guess: {sigma:.1%}")
    print(f"Target price: ${market_price:.2f}")
    print()

    for i in range(10):
        bs_price = black_scholes_call(S, K, T, r, sigma)
        bs_vega = vega(S, K, T, r, sigma)
        diff = bs_price - market_price

        print(f"Iteration {i+1}: σ = {sigma:.6f}, BS Price = ${bs_price:.4f}, Diff = {diff:+.6f}")
        
        if abs(diff) < 1e-6:
            print(f"\nConverged after {i+1} iterations!")
            break
        
        sigma = sigma - diff / bs_vega 
        
                  
    

    