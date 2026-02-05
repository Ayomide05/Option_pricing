"""
Value-at_risk (VaR) Implementation
This module implements the three main VaR calculation methods which are:
1. Historical Simulation
2. Parametric (Variance-Covariance)
3. Monte carlo Simulation
This module also includes Expected Shorfall (ES)(CVaR) and backtesting functionality
VaR answers: "What is the maximum loss i can encounter over a given period of time at a given confidence level?"
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, List, Union
import warnings

# HISTORICAL SIMULATION VAR
def historical_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate VaR using Historical Simulation.
    
    Method: Sort historical returns and find the percentile corresponding
    to (1 - confidence_level).
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns (as decimals, e.g., 0.01 for 1%)
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
    
    Returns
    -------
    float
        VaR as a positive number (loss)
    
    Example
    -------
    >>> returns = np.random.normal(0, 0.02, 1000)  # 2% daily vol
    >>> var_95 = historical_var(returns, 0.95)
    >>> print(f"95% VaR: {var_95:.4f}")
    """

    if len(returns) == 0:
        raise ValueError("Returns array is empty")
    
    # VaR is the (1 - confidence_level) percetile of returns
    # We want losses, so we look at the left tail
    percentile = (1 - confidence_level) * 100
    var = -np.percentile(returns, percentile)

    return var

def historical_var_weighted(returns: np.ndarray, confidence_level: float = 0.95, decay_factor: float = 0.94) -> float:
    """
    Calculate VaR using Age-Weighted Historical Simulation.
    
    Recent observations get more weight than older ones.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns (oldest first)
    confidence_level : float
        Confidence level
    decay_factor : float
        Lambda parameter (0.94 typical for daily data)
    
    Returns
    -------
    float
        VaR as a positive number
    """

    n = len(returns)

    # Create weights: more recent = higher weight
    weights = np.array([decay_factor ** (n - 1 - i) for i in range(n)])
    weights = weights / weights.sum()     # Normalize

    # Sort returns and weights together
    sorted_indices = np.argsort(returns)
    sorted_returns = returns[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # FInd the return where cumulative weight reaches (1 - confidence_level)
    cumulative_weights = np.cumsum(sorted_weights)
    var_index = np.searchsorted(cumulative_weights, 1 - confidence_level)
    var_index = min(var_index, len(returns) - 1)

    var = -sorted_returns[var_index]

    return var

def parametric_var(returns: np.ndarray = None, volatility: float = None,
                   confidence_level: float = 0.95, mean: float = 0.0) -> float:
    """
    Calculate VaR using Parametric (Variance-Covariance) method. This method assumes
    returns are normally distributed
    Formula: VaR = -μ + σ × z_α
    Parameters
    ----------
    returns : np.ndarray, optional (Historical returns to estimate volatility)
    volatility : float, optional (Pre-calculated volatility (used if returns not provided))
    confidence_level : float  (Confidence level)
    mean : float  (Expected return (usually 0 for short horizons))
    Returns
    -------
    float  (VaR as a positive number)
    """
    if returns is not None:
        sigma = np.std(returns, ddof=1)
        if mean == 0.0:
            mu = 0.0    # Assume zero mean for short horizons
        else:
            mu = np.mean(returns)
    elif volatility is not None:
        sigma = volatility
        mu = mean
    else:
        raise ValueError("Must provide either returns or volatility")
    
    # Z-score for the confidence level
    z_score = stats.norm.ppf(confidence_level)

    #VaR formula (negative because we want loss as positive)
    var = -mu + sigma * z_score

    return var

def parametric_var_ewma(returns: np.ndarray, confidence_level: float = 0.95,
                        decay_factor: float = 0.94) -> float:
    """
    Calculate VaR using EWMA (Exponentially Weighted Moving Average) volatility.
    EWMA gives more weight to recent observations when estimating volatility.
    Parameters
    ----------
    returns : np.ndarray (Historical returns)
    confidence_level : float  (Confidence level)
    decay_factor : float  (Lambda parameter (0.94 for daily, 0.97 for monthly))
    Returns
    -------
    float : (VaR as a positive number)
    """
    # Calculate EWMA variance
    n = len(returns)
    weights = np.array([decay_factor ** i for i in range(n -1, -1, -1)])
    weights = weights / weights.sum()

    # EWMA variance
    mean_return = np.sum(weights * returns)
    variance = np.sum(weights * (returns - mean_return) ** 2)
    sigma = np.sqrt(variance)

    # Z-score
    z_score = stats.norm.ppf(confidence_level)

    var = sigma * z_score

    return var

def monte_carlo_var(returns: np.ndarray = None, volatility: float = None,
                    mean: float = 0.0, confidence_level: float = 0.95,
                    n_simulations: int = 10000, time_horizon: int = 1) -> float:
    """
    Calculate VaR using Monte Carlo Simulation. This Simulates future returns and finds the percentile.
    Parameters
    ----------
    returns : np.ndarray, optional  (Historical returns to estimate parameters)
    volatility : float, optional  (Pre-calculated volatility)
    mean : float  (Expected return)
    confidence_level : float  (Confidence level)
    n_simulations : int  (Number of Monte Carlo simulations)
    time_horizon : int (Number of days ahead)
    Returns
    -------
    float  (VaR as a positive number)
    """
    if returns is not None:
        sigma = np.std(returns, ddof=1)
        mu = 0.0   # Assume zero for short horizons
    elif volatility is not None:
        sigma = volatility
        mu = mean
    else:
        raise ValueError("Must provide either returns or volatility")
    
    # Scale for time horizon
    sigma_t = sigma * np.sqrt(time_horizon)
    mu_t = mu * time_horizon

    # simulate returns
    simulated_returns = np.random.normal(mu_t, sigma_t, n_simulations)

    # VaR is the (1 - confidence_level) percentile
    percentile = (1 - confidence_level) * 100
    var = -np.percentile(simulated_returns, percentile)

    return var

# Expected Shortfall (Conditional VAR)

def expected_shortfall_historical(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate Expected Shortfall using Historical Simulation.
    ES = Average of losses exceeding VaR
    Parameters
    ----------
    returns : np.ndarray  (Historical returns)
    confidence_level : float  (Confidence level)
    Returns
    -------
    float (Expected Shortfall as a positive number)
    """
    var = historical_var(returns, confidence_level)

    # Find returns worse that VaR 
    tail_returns = returns[returns < -var]

    if len(tail_returns) == 0:
        return var  # If no exceedances , ES = VaR
    
    es = -np.mean(tail_returns)

    return es

def expected_shortfall_parametric(returns: np.ndarray = None,
                                  volatility: float = None,
                                  confidence_level: float = 0.95) -> float:
    """
    Calculate Expected Shortfall assuming normal distribution.
    Formula: ES = σ × φ(z_α) / (1 - α)
    where φ is the standard normal PDF and α is the confidence level.
    Parameters
    ----------
    returns : np.ndarray, optional (Historical returns to estimate volatility)
    volatility : float, optional (Pre-calculated volatility)
    confidence_level : float  (Confidence level)
    Returns
    -------
    float  (Expected Shortfall as a positive number)
    """
    if returns is not None:
        sigma = np.std(returns, ddof=1)
    elif volatility is not None:
        sigma = volatility
    else:
        raise ValueError("Must provide either returns or volatility")
    
    z_score = stats.norm.ppf(confidence_level)

    # ES formula for normal distribution
    es = sigma * stats.norm.pdf(z_score) / (1 - confidence_level)

    return es

def expected_shortfall_monte_carlo(returns: np.ndarray = None,
                                   volatility: float = None,
                                   confidence_level: float = 0.95,
                                   n_simulations: int =10000) -> float:
    """ Calculate Expected Shortfall using Monte carlo Simulation."""
    if returns is not None:
        sigma = np.std(returns, ddof=1)
    elif volatility is not None:
        sigma = volatility
    else:
        raise ValueError("Must provide either returns or volatility")
    
    # Simulate returns
    simulated_returns = np.random.normal(0, sigma, n_simulations)

    # Find VaR
    percentile = (1 - confidence_level) * 100
    var = -np.percentile(simulated_returns, percentile)

    # ES = average of losses beyond VaR
    tail_returns = simulated_returns[simulated_returns < -var]

    if len(tail_returns) == 0:
        return var
    
    es = -np.mean(tail_returns)

    return es

# TIME SCALING

def scale_var(var_1day: float, time_horizon: int) -> float:
    """
    Scale 1-day VaR to a different time horizon using the square root of time rule.
    Parameters
    ----------
    var_1day : float  (1-day VaR)
    time_horizon : int (Target time horizon in days)
        Returns
    -------
    float   (Scaled VaR)
    """
    return var_1day * np.sqrt(time_horizon)

def portfolio_var_parametric(weights: np.ndarray, cov_matrix: np.ndarray,
                             confidence_level: float = 0.95,
                             portfolio_value: float = 1.0) -> dict:
    """
    Calculate Portfolio VaR using Variance-Covariance method.
    Parameters
    ----------
    weights : np.ndarray   (Portfolio weights (should sum to 1))
    cov_matrix : np.ndarray  (Covariance matrix of returns)
    confidence_level : float  (Confidence level)
    portfolio_value : float   (Total portfolio value (for dollar VaR))

    Returns
    -------
    dict
        Dictionary containing:
        - 'portfolio_var': Diversified VaR
        - 'undiversified_var': Sum of individual VaRs
        - 'diversification_benefit': Risk reduction from diversification
        - 'component_var': VaR contribution of each asset
    """
    n_assets = len(weights)

    # Portfolio variance
    portfolio_variance = weights @ cov_matrix @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)

    # Z-score
    z_score = stats.norm.ppf(confidence_level)

    # Diversified VaR
    portfolio_var = portfolio_volatility * z_score * portfolio_value

    # Individual VaRs (undiversified)
    individual_volatilities = np.sqrt(np.diag(cov_matrix))
    individual_vars = np.abs(weights) * individual_volatilities * z_score * portfolio_value
    undiversified_var = np.sum(individual_vars)

    # Diversification benefit
    diversification_benefit = undiversified_var - portfolio_var

    # Component VaR (marginal contribution)
    marginal_var = (cov_matrix @ weights) / portfolio_volatility * z_score * portfolio_value
    component_var = weights * marginal_var

    return {
        'portfolio_var': portfolio_var,
        'undiversified_var': undiversified_var,
        'diversification_benefit': diversification_benefit,
        'component_var': component_var,
        'individual_vars': individual_vars
    }

def calculate_covariance_matrix(returns_df: pd.DataFrame, 
                                method: str = 'sample') -> np.ndarray:
    """
    Calculate covariance matrix from returns.
    Parameters
    ----------
    returns_df : pd.DataFrame  (DataFrame with returns for each asset)
    method : str
        'sample' for sample covariance
        'ewma' for exponentially weighted (lambda=0.94)
    Returns
    -------
    np.ndarray (Covariance matrix)
    """
    if method == 'sample':
        return returns_df.cov().values
    elif method == 'ewma':
        lambda_param = 0.94
        n = len(returns_df)
        weights = np.array([lambda_param ** i for i in range(n -1, -1, -1)])
        weights = weights / weights.sum()

        # Weighted covariance
        returns_array = returns_df.values
        means = np.sum(weights[:, np.newaxis] * returns_array, axis=0)
        centered = returns_array - means
        cov_matrix = (weights[:, np.newaxis, np.newaxis] *
                     centered[:, :, np.newaxis] * 
                      centered[:, np.newaxis, :]).sum(axis=0)
        
        return cov_matrix
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
# BACKTESTING
def backtest_var(returns: np.ndarray, var_estimates: np.ndarray, 
                 confidence_level: float = 0.95) -> dict:
    """ 
    Backtest VaR model by comparing predictions to actual returns.
    Parameters
    ----------
    returns : np.ndarray (Actual returns)
    var_estimates : np.ndarray  (VaR estimates (same length as returns))
    onfidence_level : float (Confidence level used for VaR)
    Returns
    -------
    dict  (Backtesting results)
    """
    n = len(returns)

    # Count exceedances (actual loss > VaR)
    exceedances = returns < -var_estimates
    n_exceedances = np.sum(exceedances)

    # Expected exceedances
    expected_rate = 1 - confidence_level
    expected_exceedances = n * expected_rate

    # Exceedance rate
    actual_rate = n_exceedances / n

    # Kupiec test (proportion of failures)
    #HO: actual rate = expected rate
    if n_exceedances > 0 and n_exceedances < n:
        lr_pof = -2 * (np.log((1 - expected_rate) ** (n - n_exceedances) *
                              expected_rate ** n_exceedances) -
                               np.log((1 - actual_rate) ** (n - n_exceedances) *
                                      actual_rate ** n_exceedances))
        p_value = 1 - stats.chi2.cdf(lr_pof, 1)
    else:
        lr_pof = np.nan
        p_value = np.nan

    # Traffic light zone (Basel)
    if confidence_level == 0.99:
        # For 99% VaR over 250 days
        if n_exceedances <= 4:
            zone = "Green"
        elif n_exceedances <= 9:
            zone = "Yellow"
        else:
            zone = "Red"
    else:
        # Generic assessment
        ratio = actual_rate / expected_rate
        if ratio <= 1.5:
            zone = "Green"
        elif ratio <= 2.0:
            zone = "Yellow"
        else:
            zone = "Red"

    return {
        'n_observations': n,
        'n_exceedances': n_exceedances,
        'expected_exceedances': expected_exceedances,
        'exceedance_rate': actual_rate,
        'expected_rate': expected_rate,
        'kupiec_statistic': lr_pof,
        'kupiec_p_value': p_value,
        'zone': zone,
        'exceedance_dates': np.where(exceedances)[0]
    }

def rolling_var(returns: np.ndarray, window: int = 250,
                confidence_level: float = 0.95,
                method: str = 'historical') -> np.ndarray:
    """
    Calculate rolling VaR estimates.
    Parameters
    ----------
    returns : np.ndarray  (Historical returns)
    window : int  (Rolling window size)
    confidence_level : float  (Confidence level)
    method : str   ('historical' or 'parametric')
    Returns
    -------
    np.ndarray  (Array of VaR estimates (NaN for first window-1 observations))
    """
    n = len(returns)
    var_estimates = np.full(n, np.nan)
    for i in range(window, n):
        window_returns = returns[i - window:i]

        if method == 'historical':
            var_estimates[i] = historical_var(window_returns, confidence_level)
        elif method == 'parametric':
            var_estimates[i] = parametric_var(window_returns, confidence_level=confidence_level)
        else:
            raise ValueError(f"Unknown method: {method}")
    return var_estimates

