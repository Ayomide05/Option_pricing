"""
Monte Carlo Option Pricing
This module implements Monte Carlo simulation for pricing options.
Monte Carlo can price any option, including path-dependent options 
that have no closed-form solution.
The method: 
1. Simulate many stock price paths using Geometric Brownian Motion
2. Calculate payoff for each path
3. Average the payoffs
4. Discount to present value
"""

import numpy as np
from typing import Tuple
import time

def simulate_gbm_paths(S0: float, r: float, sigma: float, T: float, 
                    n_steps: int, n_paths: int, antithetic: bool = False) -> np.ndarray:
    """Simulate stock price using Geometric Brownian Motion.
    The stock follows: dS = r*S*dt + sigma*S*dW
    Solution: S(t) = S(0) * exp((r - 0.5*sigma^2)*t + sigma*W(t))
    Parameters
    ----------
    S0 : float  (Initial Stock Price)
    r : float  (Risk-free rate (under risk,-neutral measure))
    sigma : float  (Volatiltiy)
    T : float  (Time to maturity (years))
    n_steps : int (Number of time steps)
    n_paths : int (Number of simulation paths)
    antithetic : bool  (if True, use antithetic variates for variance reduction)
    Returns
    ---------
    np.ndarray
        Stock price paths, shape (n_paths, n_steps + 1)
        First column is S0, last column is S(T)
    """

    dt = T / n_steps

    # Generate random increments
    if antithetic:
        # Generate half the paths, then mirror them
        half_paths = n_paths // 2
        Z = np.random.standard_normal((half_paths, n_steps))
        Z = np.vstack([Z, -Z])   # Antithetic pairs
        if n_paths % 2 == 1:
            # Add one more path if n_paths is odd
            Z = np.vstack([Z, np.random.standard_normal((1, n_steps))])
    else:
        Z = np.random.standard_normal((n_paths, n_steps))
    

    # Calculate log returns
    # ln(S(t+dt)/S(t)) = (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion

    # Build price paths
    log_S = np.zeros((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S0)
    log_S[:, 1:] = np.log(S0) + np.cumsum(log_returns, axis=1)

    S = np.exp(log_S)

    return S

def monte_carlo_european_call(S0: float, K: float, T: float, r: float, sigma: float,
                              n_paths: int = 100000, n_steps: int = 1,
                              antithetic: bool = True) -> Tuple[float, float, float]:
    """
    Price a European call option using Monte carlo simulation.
    Parameters
    ----------
    S0 : float  (Initial stock price)
    K : float   (Strike price)
    T : float   (Time to maturity)
    r : float   (Risk-free rate)
    sigma : float (Volatility)
    n_paths : int (Number of simulation paths)
    n_steps : int (Number of time steps (1 is sufficient for European options))
    antithetic : bool (Use antithetic variates for variance reduction)
    Returns
    -------
    Tuple[float, float, float]
        (price, standard_error, 95% confidence interval half-width)
    """
    # Simulate paths
    S = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, antithetic)
    # Get terminal prices
    S_T = S[:, -1]

    # calculate payoffs
    payoffs = np.maximum(S_T - K, 0)

    # Discount to present value
    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs

    # calculate statictics
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
    ci_95 = 1.96 * std_error

    return price, std_error, ci_95
    
def monte_carlo_european_put(S0: float, K: float, T: float, r: float, sigma: float,
                              n_paths: int = 100000, n_steps: int = 1,
                              antithetic: bool = True) -> Tuple[float, float, float]:
    """
    Price a European put option using Monte Carlo simulation.
    """
    S = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, antithetic)
    S_T = S[:, -1]
    
    payoffs = np.maximum(K - S_T, 0)
    
    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs
    
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
    ci_95 = 1.96 * std_error
    
    return price, std_error, ci_95

def monte_carlo_asian_call(S0: float, K: float, T: float, r: float, sigma: float,
                           n_paths: int = 100000, n_steps: int = 252,
                           antithetic: bool = True, 
                           average_type: str = 'arithmetic') -> Tuple[float, float, float]:
    """
    Price an Asian call option using Monte Carlo simulation.
    
    Asian options have payoff based on the average price over the life
    of the option, not just the terminal price.
    
    Payoff = max(Average(S) - K, 0)
    
    Parameters
    ----------
    average_type : str
        'arithmetic' or 'geometric'
        Arithmetic: (S1 + S2 + ... + Sn) / n
        Geometric: (S1 * S2 * ... * Sn)^(1/n)
    
    Note: Geometric Asian options have a closed-form solution,
    but arithmetic Asian options do not - Monte Carlo is required.
    """
    S = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, antithetic)

    # Calculate average price for each path
    if average_type == 'arithmetic':
        S_avg = np.mean(S[:, 1:], axis=1)   # Exclude S0
    elif average_type == "geometric":
        S_avg = np.exp(np.mean(np.log(S[:, 1:]), axis=1))
    else:
        raise ValueError("average_type must be 'arithmetic' or 'geometric'")
    
    # calculate payoffs
    payoffs = np.maximum(S_avg - K, 0)

    # Discount
    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs

    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
    ci_95 = 1.96 * std_error

    return price, std_error, ci_95

def monte_carlo_asian_put(S0: float, K: float, T: float, r: float, sigma: float,
                          n_paths: int = 100000, n_steps: int = 252,
                          antithetic: bool = True,
                          average_type: str = 'arithmetic') -> Tuple[float, float, float]:
    """
    Price an Asian put option using Monte Carlo simulation.
    
    Payoff = max(K - Average(S), 0)
    """
    S = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, antithetic)

    if average_type == "arithmetic":
        S_avg = np.mean(S[:, 1:], axis=1)
    elif average_type == 'geometric':
        S_avg = np.exp(np.mean(np.log(S[:, 1:]), axis=1))
    else:
        raise ValueError("average_type must be 'arithmetic' or 'geometric")
    
    payoffs = np.maximum(K - S_avg, 0)

    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs

    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
    ci_95 = 1.96 * std_error
    
    return price, std_error, ci_95

def monte_carlo_barrier_call(S0: float, K: float, T: float, r: float, sigma: float,
                            barrier: float, barrier_type: str = 'down-and-out',
                            n_paths: int = 100000, n_steps: int = 252,
                            antithetic: bool = True) -> Tuple[float, float, float]:
    """
    Price a barrier call option using Monte Carlo simulation.
    
    Barrier options are activated or deactivated when the underlying
    crosses a certain level (the barrier).
    
    Parameters
    ----------
    barrier : float
        Barrier level
    barrier_type : str
        'down-and-out': Option dies if S falls below barrier
        'down-and-in': Option activates if S falls below barrier
        'up-and-out': Option dies if S rises above barrier
        'up-and-in': Option activates if S rises above barrier
    
    Note: Barrier options are path-dependent - the entire path matters,
    not just the terminal value.
    """
    S = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, antithetic)
    S_T = S[:, -1]

    # Check barrier conditions
    if barrier_type == 'down-and-out':
        knocked_out = np.any(S <= barrier, axis=1)
        active = ~knocked_out
    elif barrier_type == 'down-and-in':
        knocked_in = np.any(S <= barrier, axis=1)
        active = knocked_in
    elif barrier_type == 'up-and-out':
        knocked_out = np.any(S >= barrier, axis=1)
        active = ~knocked_out
    elif barrier_type == 'up-and-in':
        knocked_in = np.any(S >= barrier, axis=1)
        active = knocked_in
    else:
        raise ValueError("Invalid barrier_type")
    
    payoffs = np.where(active, np.maximum(S_T - K, 0), 0)

    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs

    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
    ci_95 = 1.96 * std_error
    
    return price, std_error, ci_95

def monte_carlo_barrier_put(S0: float, K: float, T: float, r: float, sigma: float,
                             barrier: float, barrier_type: str = 'down-and-out',
                             n_paths: int = 100000, n_steps: int = 252,
                             antithetic: bool = True) -> Tuple[float, float, float]:
    """
    Price a barrier put option using Monte Carlo simulation.
    """
    S = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, antithetic)
    S_T = S[:, -1]
    
    if barrier_type == 'down-and-out':
        knocked_out = np.any(S <= barrier, axis=1)
        active = ~knocked_out
    elif barrier_type == 'down-and-in':
        knocked_in = np.any(S <= barrier, axis=1)
        active = knocked_in
    elif barrier_type == 'up-and-out':
        knocked_out = np.any(S >= barrier, axis=1)
        active = ~knocked_out
    elif barrier_type == 'up-and-in':
        knocked_in = np.any(S >= barrier, axis=1)
        active = knocked_in
    else:
        raise ValueError("Invalid barrier_type")
    
    payoffs = np.where(active, np.maximum(K - S_T, 0), 0)
    
    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs
    
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
    ci_95 = 1.96 * std_error
    
    return price, std_error, ci_95

def monte_carlo_lookback_call(S0: float, T: float, r: float, sigma: float,
                              n_paths: int = 100000, n_steps: int = 252,
                              antithetic: bool = True,
                              lookback_type: str = 'floating') -> Tuple[float, float, float]:
    """
    Price a lookback call option using Monte Carlo simulation.
    
    Lookback options have payoff based on the maximum or minimum
    price achieved during the life of the option.
    
    Parameters
    ----------
    lookback_type : str
        'floating': Payoff = S_T - S_min (floating strike)
        'fixed': Payoff = max(S_max - K, 0) (fixed strike, K = S0)
    """
    S = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, antithetic)
    S_T = S[:, -1]

    if lookback_type == 'floating':
        S_min = np.min(S, axis=1)
        payoffs = S_T - S_min
    elif lookback_type == 'fixed':
        S_max = np.max(S, axis=1)
        payoffs = np.maximum(S_max - S0, 0)
    else:
        raise ValueError("lookback_type muse be 'floating' or 'fixed'")
    
    discount_factor = np.exp(-r * T)
    discounted_payoffs = discount_factor * payoffs

    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
    ci_95 = 1.96 * std_error

    return price, std_error, ci_95

def convergence_analysis(S0: float, K: float, T: float, r: float, sigma: float,
                         true_price: float, path_counts: list = None) -> dict:
    """
    Analyze how Monte Carlo price converges as number of paths increases
    """
    if path_counts is None:
        path_counts = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]

    results = {
        'n_paths': [],
        'price' : [],
        'std_error' : [],
        'error': [],
        'time': []
    }

    for n in path_counts:
        start = time.time()
        price, std_err, _ = monte_carlo_european_call(S0, K, T, r, sigma, n_paths=n)
        elapsed = time.time() - start

        results['n_paths'] .append(n)
        results['price'].append(price)
        results['std_error'].append(std_err)
        results['error'].append(abs(price - true_price))
        results['time'].append(elapsed)

    return results


if __name__ == "__main__":
    print("Monte Carlo Option Pricing")
    print("=" * 60)

    # Parameters
    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2

    print(f"\nParameters: S0=${S0}, K=${K}, T={T}y, r={r:.1%}, sigma={sigma:.1%}")
    print("=" * 60)

    # Black-Scholes price (analytical)
    from scipy.stats import norm
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    bs_call = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    bs_put = K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    # 1. European Call
    print("\n1. EUROPEAN CALL OPTION")
    print("-" * 60)
    print(f"Black-Scholes Call Price: ${bs_call:.4f}")
    
    mc_call, mc_err, mc_ci = monte_carlo_european_call(S0, K, T, r, sigma, n_paths=100000)
    print(f"Monte Carlo Call Price:   ${mc_call:.4f} +/- ${mc_ci:.4f} (95% CI)")
    print(f"Standard Error:           ${mc_err:.6f}")
    print(f"Error vs BS:              ${abs(mc_call - bs_call):.6f}")
    
    # 2. European Put
    print("\n2. EUROPEAN PUT OPTION")
    print("-" * 60)
    print(f"Black-Scholes Put Price:  ${bs_put:.4f}")
    
    mc_put, mc_err, mc_ci = monte_carlo_european_put(S0, K, T, r, sigma, n_paths=100000)
    print(f"Monte Carlo Put Price:    ${mc_put:.4f} +/- ${mc_ci:.4f} (95% CI)")
    print(f"Error vs BS:              ${abs(mc_put - bs_put):.6f}")
    
    # 3. Asian Option
    print("\n3. ASIAN CALL OPTION (Arithmetic Average)")
    print("-" * 60)
    print("Note: No closed-form solution exists for arithmetic Asian options")
    
    asian_call, asian_err, asian_ci = monte_carlo_asian_call(
        S0, K, T, r, sigma, n_paths=100000, n_steps=252, average_type='arithmetic'
    )
    asian_diff = asian_call - mc_call
    asian_comparison = "cheaper" if asian_diff < 0 else "more expensive"
    print(f"Monte Carlo Asian Call:   ${asian_call:.4f} +/- ${asian_ci:.4f}")
    print(f"Compared to European:     ${asian_diff:+.4f} (Asian is {asian_comparison})")
    
    # 4. Barrier Option
    print("\n4. BARRIER CALL OPTION (Down-and-Out)")
    print("-" * 60)
    barrier = 80
    print(f"Barrier Level: ${barrier}")
    
    barrier_call, barrier_err, barrier_ci = monte_carlo_barrier_call(
        S0, K, T, r, sigma, barrier=barrier, barrier_type='down-and-out',
        n_paths=100000, n_steps=252
    )
    barrier_diff = barrier_call - mc_call
    barrier_comparison = "cheaper" if barrier_diff < 0 else "more expensive"
    print(f"Monte Carlo Barrier Call: ${barrier_call:.4f} +/- ${barrier_ci:.4f}")
    print(f"Compared to European:     ${barrier_diff:+.4f} (Barrier is {barrier_comparison})")
    
    # 5. Lookback Option
    print("\n5. LOOKBACK CALL OPTION (Floating Strike)")
    print("-" * 60)
    print("Payoff = S_T - S_min (you buy at the lowest price)")
    
    lookback_call, lookback_err, lookback_ci = monte_carlo_lookback_call(
        S0, T, r, sigma, n_paths=100000, n_steps=252, lookback_type='floating'
    )
    lookback_diff = lookback_call - mc_call
    lookback_comparison = "cheaper" if lookback_diff < 0 else "more expensive"
    print(f"Monte Carlo Lookback Call: ${lookback_call:.4f} +/- ${lookback_ci:.4f}")
    print(f"Compared to European:      ${lookback_diff:+.4f} (Lookback is {lookback_comparison})")
    
    # 6. Convergence Analysis
    print("\n6. CONVERGENCE ANALYSIS")
    print("-" * 60)
    print(f"{'Paths':>10} {'MC Price':>12} {'Std Error':>12} {'Error vs BS':>12} {'Time':>10}")
    print("-" * 60)
    
    results = convergence_analysis(S0, K, T, r, sigma, bs_call)
    for i in range(len(results['n_paths'])):
        print(f"{results['n_paths'][i]:>10,} ${results['price'][i]:>10.4f} "
              f"${results['std_error'][i]:>10.6f} ${results['error'][i]:>10.6f} "
              f"{results['time'][i]:>9.4f}s")
        
    # Analyze convergence rate
    print("\n" + "=" * 60)
    print("CONVERGENCE ANALYSIS INSIGHT")
    print("_" * 60)

    # Compare error reduction when paths increase
    # Theory: if paths increase by Nx, std error decreases by sqrt(N)x
    # We compare index 2 (1,000 paths) vs index 4 (10,000 paths)
    idx_low = 2
    idx_high = 4
    
    if len(results['n_paths']) > idx_high:
        se_low = results['std_error'][idx_low]
        se_high = results['std_error'][idx_high]
        paths_low = results['n_paths'][idx_low]
        paths_high = results['n_paths'][idx_high]
        
        paths_ratio = paths_high / paths_low
        se_ratio = se_low / se_high
        theoretical_ratio = np.sqrt(paths_ratio)
        
        print(f"Paths increased by:    {paths_ratio:.0f}x ({paths_low:,} -> {paths_high:,})")
        print(f"Std Error reduced by:  {se_ratio:.2f}x ({se_low:.6f} -> {se_high:.6f})")
        print(f"Theoretical reduction: {theoretical_ratio:.2f}x (1/sqrt(N) rule)")
        
        deviation = abs(se_ratio - theoretical_ratio) / theoretical_ratio
        if deviation < 0.3:
            print(f"\nResult: Convergence follows the 1/sqrt(N) rule (deviation: {deviation:.1%})")
        else:
            print(f"\nResult: Some deviation from theory ({deviation:.1%}) - normal due to randomness.")
    else:
        print("Not enough data points for convergence analysis.")


    






    