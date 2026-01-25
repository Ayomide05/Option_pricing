"""
Volatility Smile and Surface Visualization
Volatility Smile is a phenomenon that implied volatility vaires across strike 
prices, contradicting Black-Scholes assumption of constant volatility.
The smile reveals that:
    - The market doesn't believe in Black-Scholes' assumptions
    - Out-of-the-money puts are expensive (crash protection)
    - The distribution of returns has fat tails
This module fetches real option chain data and calculates the actual
implied volatility smile from market prices
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("warning: yfinance not installed. Run: pip install yfinance")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from phase1_black_scholes.black_scholes import black_scholes_call, black_scholes_put
from phase2_implied_volatility.implied_volatility import implied_volatility_call, implied_volatility_put

IMAGE_DIR = "images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def get_risk_free_rate(maturity: str = '3m') -> float:
    """
    Fetch current risk-free rate from Treasury yields.
    We uses yahoo Finance to get Treasury yield data which falls back
    to a default value if fetch fails
    Parameters
    -----------
    maturity : str
        '3m' for 3-month T-bill
        '5y' for 5-year Treasury
        '10y' for 10-year Treasury
    Returns
    --------
    float  : Risk-free rate as decimal (e.g., 0.05 for 5%)
    """
    # Treasury yield tickers on Yahoo Finance
    treasury_tickers = {
        '3m': '^IRX',   # 13-week Treasury Bill
        '5y': '^FVX',   # 5-year Treasury Note
        '10y': '^TNX',  # 10-year Treasury Note
        '30y': '^TYX',  # 30-year Treasury Bond           
    }

    ticker = treasury_tickers.get(maturity, '^IRX')

    try:
        if YFINANCE_AVAILABLE:
            treasury = yf.Ticker(ticker)
            hist = treasury.history(period='5d')

            if len(hist) > 0:
                # yahoo returns yield as percentage (e.g., 4.5 for 4.5%)
                rate = hist['Close'].iloc[-1] / 100
                print(f"  Risk-free rate ({maturity}): {rate:.2%} (from Treasury yield)")
                return rate
    except Exception as e:
        print(f"  Warning: Could not fetch Treasury rate: {e}")

    # Fallback to reasonable default
    default_rate = 0.045  # 4.45% as of late 2024/early 2025
    print(f" Risk-free rate: {default_rate:.2%} (default fallback)")
    return default_rate


def get_risk_free_rate_for_maturity(T: float) -> float:
    """Get appropriate risk-free rate based on option maturity."""
    if T <= 1.0:    # Up to 1 year - use 3-month rate
        return get_risk_free_rate('3m')
    elif T <= 5.0:    # 1 to 5 years - use 5-year rate
        return get_risk_free_rate('5y')
    elif T <= 15.0:  # 5 to 15 years - use 10-year rate
        return get_risk_free_rate('10y')
    else:    # Beyond 15 years - use 30-year rate
        return get_risk_free_rate('30y')
    
def fetch_option_chain(ticker: str = "SPY", expiry_index: int = 0) -> dict:
    """Fetch option chain data for a given ticker
    Parameters
    ----------
    ticker : str ( Stock ticker symbol (e.g., 'SPY', 'AAPL', 'QQQ'))
    expiry_index : int   ( Which expiry date to use (0 = nearest, 1 = next, etc))
    Returns
    -------
    dict  : Dictionary containing:
            - 'calls' : DataFrame of call options
            - 'puts' : DataFrame of put options
            - 'spot' : Current stock price
            - 'expiry' : Expiration date
            - 'T' : Time to expiry in years
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is required. Install with: pip install yfinance")
    
    # Fetch stock data
    stock = yf.Ticker(ticker)
    spot = stock.history(period="1d")['Close'].iloc[-1]
    # Get available expirayion dates
    expirations = stock.options

    if len(expirations) == 0:
        raise ValueError(f" No options available for {ticker}")
    
    # Select expiration
    expiry_index = min(expiry_index, len(expirations) - 1)
    expiry_date = expirations[expiry_index]

    # Fetch option chain for selected expiry
    opt_chain = stock.option_chain(expiry_date)
    
    # Calculate time to expiry
    expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
    today = datetime.now()
    days_to_expiry = (expiry_dt - today).days
    T = max(days_to_expiry / 365, 1/365)   # At least 1 day

    return {
        'calls': opt_chain.calls,
        'puts': opt_chain.puts,
        'spot': spot,
        'expiry': expiry_date,
        'T' : T,
        'ticker' : ticker
    }

def calculate_smile_from_market(option_data: dict, option_type: str = 'call',
                                r: float = None) -> pd.DataFrame:
    """Calculate implied volatility for each strike from market data.
    Parameters
    ----------
    option_data : dict  (Output from fetch_option_chain())
    option_type : str  ('call' or 'put')
    r : float  ( Risk-free rate (if None, fetches appropriate Treasury rate))
    Returns
    -------
    pd.DataFrame   (DataFrame with strikes, market prices and implied volatilities)
    """
    if r is None:
        T = option_data['T']
        r = get_risk_free_rate_for_maturity(T)
    S = option_data['spot']
    T =  option_data['T']

    if option_type == 'call':
        df = option_data['calls'].copy()
        iv_func = implied_volatility_call
    else:
        df = option_data['puts'].copy()
        iv_func = implied_volatility_put

    # Filter for liquid options (non-zero bid and ask)
    df = df[(df['bid'] > 0) & (df['ask'] > 0)].copy()
    # Use mid price
    df['mid_price'] = (df['bid'] + df['ask']) / 2
    # Calculate moneyness
    df['moneyness'] = df['strike'] / S

    # Calculate implied volatility for each strike
    ivs = []
    for idx, row in df.iterrows():
        K = row['strike']
        market_price = row['mid_price']

        try:
            iv = iv_func(market_price, S, K, T, r)
            ivs.append(iv)
        except ValueError:
            ivs.append(np.nan)
    df['implied_vol'] = ivs

    # Filter out failed calculations
    df = df.dropna(subset=['implied_vol'])
    
    # Filter reasonable IV range (1% to 200%)
    df = df[(df['implied_vol'] > 0.01) & (df['implied_vol'] < 2.0)]
    
    return df[['strike', 'moneyness', 'mid_price', 'bid', 'ask', 
               'volume', 'openInterest', 'implied_vol', 'impliedVolatility']]

def plot_market_smile(ticker: str = "SPY", expiry_index: int = 0,
                      save_path: str = None):
    """Fetch real market data and plot the actual volatility smile"""
    print(f"Fetching option data for {ticker}...")

    # Fetch data
    option_data = fetch_option_chain(ticker, expiry_index)

    S = option_data['spot']
    T = option_data['T']
    expiry = option_data['expiry']

    print(f" Spot price: ${S:.2f}")
    print(f" Expiry: {expiry} ({T*365:.0f} days)")

    # Calculate IV for calls and puts
    print("Calculating implied volatilities...")
    calls_df = calculate_smile_from_market(option_data, 'call')
    puts_df = calculate_smile_from_market(option_data, 'put')

    print(f"  Calls: {len(calls_df)} strikes")
    print(f"  Puts: {len(puts_df)} strikes")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: IV vs Strike
    ax1 = axes[0]
    ax1.scatter(calls_df['strike'], calls_df['implied_vol'] * 100,
                c ='blue', alpha=0.6, label='Calls', s=30)
    ax1.scatter(puts_df['strike'], puts_df['implied_vol'] * 100, 
                c='red', alpha=0.6, label='Puts', s=30)
    ax1.axvline(x=S, color='gray', linestyle='--', alpha=0.7, label=f'Spot (${S:.0f})')

    ax1.set_xlabel('Strike Price ($)', fontsize=12)
    ax1.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax1.set_title(f'{ticker} Volatility Smile', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: IV vs Moneyness
    ax2 = axes[1]
    ax2.scatter(calls_df['moneyness'], calls_df['implied_vol'] * 100,
                c='blue', alpha=0.6, label='Calls', s=30)
    ax2.scatter(puts_df['moneyness'], puts_df['implied_vol'] * 100,
                c='red', alpha=0.6, label='Puts', s=30)
    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='ATM')

    ax2.set_xlabel('Moneyness (K/S)', fontsize=12)
    ax2.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax2.set_title(f'{ticker} Smile by Moneyness', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{ticker} Options - Expiry: {expiry} ({T*365:.0f} days) - Spot: ${S:.2f}',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return calls_df, puts_df, option_data

def plot_smile_term_structure(ticker: str = "SPY", num_expiries: int = 4,
                              save_path: str = None):
    """Plot volatility smiles across multiple expiration dates.
    This shows how the smile shape changes with time to expiry"""

    print(f"Fetching option data for {ticker} across {num_expiries} expiries...")

    # Fetch stock to get available expiries
    stock = yf.Ticker(ticker)
    expirations = stock.options[:num_expiries]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(expirations)))

    spot = None

    for i, expiry in enumerate(expirations):
        try:
            option_data = fetch_option_chain(ticker, i)
            if spot is None:
                spot = option_data['spot']

            calls_df = calculate_smile_from_market(option_data, 'call')

            # Filter to reasonable moneyness range
            calls_df = calls_df[(calls_df['moneyness'] > 0.85) &
                                (calls_df['moneyness'] < 1.15)]
            days = int(option_data['T'] * 365)
            label = f"{expiry} ({days}d)"

            ax.plot(calls_df['moneyness'], calls_df['implied_vol'] * 100,
                    'o-', color=colors[i], label=label, alpha=0.7, markersize=4)
            
        except Exception as e:
            print(f" Skipping {expiry}: {e}")
            continue

    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Moneyness (K/S)', fontsize=12)
    ax.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax.set_title(f'{ticker} Volatility Smile Term Structure', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_market_vs_model(ticker: str = "SPY", expiry_index: int = 0,
                         save_path: str = None):
    """Compare actual market smile to our quadratic model fit.
    This shows how well a simple model captures market behavior"""
    print(f"Fetching and analyzing {ticker} options...")

    # Fetch data
    option_data = fetch_option_chain(ticker, expiry_index)
    calls_df = calculate_smile_from_market(option_data, 'call')

    S = option_data['spot']

    # Filter to reasonable change
    calls_df = calls_df[(calls_df['moneyness'] > 0.85) & 
                       (calls_df['moneyness'] < 1.15)].copy()
    
    # Fit quadratic model: IV = a + b*x + c*x^2 where x = moneyness - 1
    x = calls_df['moneyness'].values - 1  # Center at ATM
    y = calls_df['implied_vol'].values

    # Least squares fit
    coeffs = np.polyfit(x, y, 2)
    c, b, a = coeffs  # polyfit returns highest degree first
    
    print(f"\nQuadratic fit: IV = {a:.4f} + {b:.4f}*x + {c:.4f}*x²")
    print(f"  ATM vol (a): {a*100:.2f}%")
    print(f"  Skew (b): {b:.4f}")
    print(f"  Smile (c): {c:.4f}")

    # Generate fitted curve
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = a + b * x_fit + c * x_fit**2
    moneyness_fit = x_fit + 1
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(calls_df['moneyness'], calls_df['implied_vol'] * 100,
               c='blue', alpha=0.6, label='Market Data', s=40)
    ax.plot(moneyness_fit, y_fit * 100, 'r-', linewidth=2,
            label=f'Quadratic Fit')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Moneyness (K/S)', fontsize=12)
    ax.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax.set_title(f'{ticker} Market Smile vs Quadratic Model', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add equation to plot
    eq_text = f'IV = {a*100:.1f}% + {b*100:.1f}%·x + {c*100:.1f}%·x²\nwhere x = K/S - 1'
    ax.text(0.05, 0.95, eq_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return coeffs

def analyze_market_smile(ticker: str = "SPY"):
    """Comprehensive analysis of a ticker's volatility smile."""
    print("=" * 60)
    print(f"VOLATILITY SMILE ANALYSIS: {ticker}")
    print("=" * 60)

    # Fetch data
    option_data = fetch_option_chain(ticker, expiry_index=0)
    calls_df = calculate_smile_from_market(option_data, 'call')
    puts_df = calculate_smile_from_market(option_data, 'put')

    S = option_data['spot']
    T = option_data['T']

    print(f"\nMarket Data:")
    print(f"  Spot Price: ${S:.2f}")
    print(f"  Expiry: {option_data['expiry']}")
    print(f"  Days to Expiry: {T*365:.0f}")

    # Find ATM options
    calls_df['atm_distance'] = abs(calls_df['strike'] - S)
    atm_call = calls_df.loc[calls_df['atm_distance'].idxmin()]
    
    puts_df['atm_distance'] = abs(puts_df['strike'] - S)
    atm_put = puts_df.loc[puts_df['atm_distance'].idxmin()]

if __name__ == "__main__":
    if not YFINANCE_AVAILABLE:
        print("Please install yfinance: pip install yfinance")
        exit(1)
    
    print("Real Market Volatility Smile Analysis")
    print("=" * 50)
    
    # Analyze SPY (S&P 500 ETF) - most liquid options market
    ticker = "SPY"
    
    # 1. Basic smile analysis
    print("\n1. Comprehensive Analysis...")
    analyze_market_smile(ticker)
    
    # 2. Plot market smile
    print("\n2. Plotting Market Smile...")
    calls_df, puts_df, option_data = plot_market_smile(
        ticker, 
        expiry_index=1,  # Use second expiry for better liquidity
        save_path=os.path.join(IMAGE_DIR, '09_market_smile.png')
    )
    
    # 3. Compare to quadratic model
    print("\n3. Fitting Quadratic Model...")
    coeffs = plot_market_vs_model(
        ticker,
        expiry_index=1,
        save_path=os.path.join(IMAGE_DIR, '10_market_vs_model.png')
    )
    
    # 4. Term structure
    print("\n4. Plotting Term Structure...")
    plot_smile_term_structure(
        ticker,
        num_expiries=4,
        save_path=os.path.join(IMAGE_DIR, '11_smile_term_structure.png')
    )
    
    print("\n" + "=" * 50)
    print("Analysis complete!")
    print(f"Images saved to: {os.path.abspath(IMAGE_DIR)}")
   

