"""Black-Scholes Option Pricing Model
This module implements the Black-scholes formula for European options, derived from the 
heat equation transformation
The key formula for a European call: 
    C = S * N(d1) - K * exp(-r*T) * N(d2)
where:
    d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T
"""

import numpy as np
from scipy.stats import norm

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """ 
    Calculate the Black-Scholes price for a European call option.
    Parameters
    ----------
    S: Spot Price (Current Stock Price)
        K: Strike Price
        T: Time to Expiry (in years)
        r: Risk-free interest rate (annualized as decimal, e.g 0.05 for 5%)
        sigma: Volatility (annualized as decimal e.g 0.20 for 20%)
    Returns
    -------
    float: The call option price
    """

    if T <= 0:
        # At expiry, the option is worth its intrinsic value
        return max(S- K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call_price

def black_scholes_put(S: float, K: float, T: float,  r: float, sigma: float) -> float:
    """calculate the Black-Scholes price for a European put option: This returns the put option price"""
    if T <= 0:
        return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return put_price

def black_scholes_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple:
    """
    Calculate d1 and d2 parameters used in Black-Scholes formula.
    These values are needed for Greeks calculation and have their own
    interpretation in terms of probabilities.
    Returns : tuple (d1, d2)
    """
    if T <= 0:
        # Handle edge case
        if S > K:
            return (np.inf, np.inf)
        elif S < K:
            return (-np.inf, -np.inf)
        else:
            return (0, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return d1, d2

def put_call_parity_check(S: float, K: float, T: float, r: float, call_price: float, put_price: float) -> float:
    """ 
    Verify put-call parity: C - P = S - K*exp(-rT)
    This returns the difference between the call price and the put price (This should be close to zerp if parity holds)

    put-call parity is a fundamental arbitrage relationship that must hold for European options.
    It's derived from the fact that a portolio of long call + short put has the same payoff as a forward contract.
    """
    left_side = call_price - put_price
    right_side = S - K * np.exp(-r * T)

    return left_side - right_side

def black_scholes_call_vectorized(S: np.ndarray, K:float, T: float, r: float, sigma: float) -> np.ndarray:
    """
    Vectorized Black-Scholes call price calculation. This is useful for computing option prices
    across a range of stock prices or for sensitivity analysis.
    """
    S = np.asarray(S)

    # Handle T = 0 case
    if T <= 0:
        return np.maximum(S - K, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

if __name__ == "__main__":
    # Test the implementation with standard parameters
    S = 100         # Stock price
    K = 100         # Strike price (at-the-money)
    T = 1           # One year to expiry
    r = 0.05        # 5% risk-free rate
    sigma = 0.2     # 20% volatility

    call = black_scholes_call(S, K, T, r, sigma)
    put = black_scholes_put(S, K, T, r, sigma)

    print("Black-scholes Option Pricing")
    print("=" * 40)
    print(f"Stock Price (S):     ${S:.2f}")
    print(f"Strike Price (K):    ${K:.2f}")
    print(f"Time to Expiry (T):  {T} year(s)")
    print(f"Risk-free Rate (r):  {r:.1%}")
    print(f"Volatility (σ):      {sigma:.1%}")
    print("-" * 40)
    print(f"Call Price:          ${call:.4f}")
    print(f"Put Price:           ${put:.4f}")
    print("-" * 40)

    # Verify put-call parity
    parity_error = put_call_parity_check(S, K, T, r, call, put)
    print(f"Put-Call Parity Error: {parity_error:.2e}")

    # show d1 and d2
    d1, d2 = black_scholes_d1_d2(S, K, T, r, sigma)
    print(f"\nd1 = {d1:.4f}")
    print(f"d2 = {d2:.4f}")
    print(f"N(d2) = {norm.cdf(d2):.4f} (risk-neutral prob. of exercise)")
    