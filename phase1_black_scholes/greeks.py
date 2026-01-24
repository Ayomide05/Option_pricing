"""
Option Greeks - Sensitivities of Option Prices
The Greeks measure how option prices change with respect to various inputs.
They are essential for:
1. Hedging: Delta tells you how many shares to hold to hedge an option
2. Risk Management: Understanding exposure to various market factors
3. Trading: Identifying mispricings and arbitrage opportunities
The five main Greeks:
- Delta (Δ): Sensitivity to stock price
- Gamma (Γ): Sensitivity of delta to stock price (second derivative)
- Theta (Θ): Sensitivity to time (time decay)
- Vega (ν): Sensitivity to volatility
- Rho (ρ): Sensitivity to interest rate

"""

import numpy as np
from scipy.stats import norm

def calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple:
    """Calculate d1 and d2 for use in Greeks formulas."""
    if T <= 0:
        return None, None
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

# Delta - First derivative with respect to stock price
def delta_call(S: float, K: float, T: float,  r: float, sigma: float) -> float:
    """ 
    Delta for a European call option = ∂C/∂S = N(d1)
    Interpretation:
    - Measures how much the call price changes when stock price changes by $1
    - Also the hedge ratio: hold Δ shares of stock to hedge one short call
    - For calls: 0 < Δ < 1 (always positive, option moves with stock)
    - At-the-money options have Δ ≈ 0.5
    
    Returns
    -------
    float
        Delta value between 0 and 1
    """
    if T <= 0:
        return 1.0 if S > K else 0.0
    
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)

def delta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Delta for a European put option = ∂P/∂S = N(d1) - 1 = -N(-d1)
    For puts: -1 < Δ < 0 (always negative, option moves opposite to stock)
    """
    if T <= 0:
        return -1.0 if S < K else 0.0
    
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) - 1

# Gamma - Second Derivative with respect to stock price
def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Gamma for European options (same for calls and puts).
    
    Gamma = ∂²C/∂S² = ∂Δ/∂S = n(d1) / (S * σ * √T)
    
    where n(x) is the standard normal PDF.
    
    Interpretation:
    - Measures how quickly delta changes as stock price moves
    - High gamma means the option's delta is unstable (harder to hedge)
    - Gamma is highest for at-the-money options near expiry
    - Always positive for long options
    
    Market Risk significance:
    - Gamma is a measure of "convexity risk"
    - Delta hedging alone doesn't work well for high-gamma positions
    - VaR models must account for gamma (delta-gamma approximation)
    """
    if T <= 0:
        return 0.0

    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# theta - Derivative with respect to time

def theta_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Theta for a European call option (per year).
    
    Theta = ∂C/∂t = -∂C/∂T
    
    The formula:
    Θ = -[S * n(d1) * σ / (2√T)] - r * K * exp(-rT) * N(d2)
    
    Interpretation:
    - Measures time decay: how much value the option loses per day/year
    - Usually negative for long options (time is the enemy)
    - Divide by 365 to get daily theta
    
    Note: The sign convention varies. We use Θ < 0 for long calls/puts,
    meaning the option loses value as time passes.
    """
    if T <= 0:
        return 0.0
    
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    
    return term1 + term2

def theta_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Theta for a European put option (per year).
    
    Θ = -[S * n(d1) * σ / (2√T)] + r * K * exp(-rT) * N(-d2)
    """
    if T <= 0:
        return 0.0
    
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    return term1 + term2

# Vega - Derivative with respect to volatility
def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Vega for European options (same for calls and puts).
    
    Vega = ∂C/∂σ = S * √T * n(d1)
    
    Note: Vega is not actually a Greek letter, but the name stuck.
    
    Interpretation:
    - Measures sensitivity to implied volatility
    - Always positive for long options (more volatility = more value)
    - Highest for at-the-money options with long time to expiry
    
    Convention: Often quoted per 1% change in volatility (divide by 100)
    
    Market Risk significance:
    - Vega exposure is critical for volatility risk
    - Implied volatility changes constantly (the volatility surface moves)
    - Large vega positions need careful volatility risk management
    """
    if T <= 0:
        return 0.0
    
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return S * np.sqrt(T) * norm.pdf(d1)

# Rho - Derivative with respect to interest rate

def rho_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Rho for a European call option.
    
    Rho = ∂C/∂r = K * T * exp(-rT) * N(d2)
    
    Interpretation:
    - Measures sensitivity to interest rate changes
    - Positive for calls (higher rates increase call value)
    - Usually the smallest Greek in magnitude for short-dated options
    
    Convention: Often quoted per 1% change in rates (divide by 100)
    """
    if T <= 0:
        return 0.0
    
    _, d2 = calculate_d1_d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * norm.cdf(d2)

def rho_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Rho for a European put option.
    
    Rho = ∂P/∂r = -K * T * exp(-rT) * N(-d2)
    
    Negative for puts (higher rates decrease put value)
    """
    if T <= 0:
        return 0.0
    
    _, d2 = calculate_d1_d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)

def all_greeks(S: float, K: float, T: float, r: float, sigma: float, 
        option_type: str = 'call') -> dict:
    """
    Calculate all Greeks for an option.
    
    Parameters
    ----------
    option_type : str
        'call' or 'put'
    
    Returns
    -------
    dict
        Dictionary containing all Greeks
    """
    if option_type.lower() == 'call':
        return {
            'delta': delta_call(S, K, T, r, sigma),
            'gamma': gamma(S, K, T, r, sigma),
            'theta': theta_call(S, K, T, r, sigma),
            'vega': vega(S, K, T, r, sigma),
            'rho': rho_call(S, K, T, r, sigma)
        }
    else:
        return {
            'delta': delta_put(S, K, T, r, sigma),
            'gamma': gamma(S, K, T, r, sigma),
            'theta': theta_put(S, K, T, r, sigma),
            'vega': vega(S, K, T, r, sigma),
            'rho': rho_put(S, K, T, r, sigma)
        }

if __name__ == "__main__":
    # Test with standard parameters
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    
    print("Option Greeks Analysis")
    print("=" * 50)
    print(f"Stock Price: ${S}, Strike: ${K}, T: {T}y, r: {r:.1%}, σ: {sigma:.1%}")
    print("=" * 50)
    
    # Call Greeks
    print("\nCALL OPTION GREEKS:")
    print("-" * 50)
    call_greeks = all_greeks(S, K, T, r, sigma, 'call')
    print(f"  Delta:  {call_greeks['delta']:>8.4f}  (hedge ratio)")
    print(f"  Gamma:  {call_greeks['gamma']:>8.4f}  (delta sensitivity)")
    print(f"  Theta:  {call_greeks['theta']:>8.4f}  (time decay per year)")
    print(f"  Theta:  {call_greeks['theta']/365:>8.4f}  (time decay per day)")
    print(f"  Vega:   {call_greeks['vega']:>8.4f}  (volatility sensitivity)")
    print(f"  Vega:   {call_greeks['vega']/100:>8.4f}  (per 1% vol change)")
    print(f"  Rho:    {call_greeks['rho']:>8.4f}  (rate sensitivity)")
    print(f"  Rho:    {call_greeks['rho']/100:>8.4f}  (per 1% rate change)")
    
    # Put Greeks
    print("\nPUT OPTION GREEKS:")
    print("-" * 50)
    put_greeks = all_greeks(S, K, T, r, sigma, 'put')
    print(f"  Delta:  {put_greeks['delta']:>8.4f}")
    print(f"  Gamma:  {put_greeks['gamma']:>8.4f}")
    print(f"  Theta:  {put_greeks['theta']:>8.4f}  (per year)")
    print(f"  Vega:   {put_greeks['vega']:>8.4f}")
    print(f"  Rho:    {put_greeks['rho']:>8.4f}")
    
    # Verify put-call relationships
    print("\nVERIFICATION:")
    print("-" * 50)
    print(f"  Call Delta - Put Delta = {call_greeks['delta'] - put_greeks['delta']:.4f} (should be 1)")
    print(f"  Call Gamma = Put Gamma: {call_greeks['gamma'] == put_greeks['gamma']}")
    print(f"  Call Vega = Put Vega: {call_greeks['vega'] == put_greeks['vega']}")   