"""
Var with Real Market Data: This module calculates VaR using real stock data from yahoo Finance
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from var import (
    historical_var, parametric_var, monte_carlo_var,
    expected_shortfall_historical, expected_shortfall_parametric,
    portfolio_var_parametric, calculate_covariance_matrix,
    backtest_var, rolling_var, scale_var
)
import os

os.makedirs("images", exist_ok=True)

# Try to import yfinance
try: 
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("warning: yfinance not installed. Using simulated data.")

def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch hsitoriacl stcok data from Yahoo Finance.
    Parameters
    ----------
    ticker : str  (Stock ticker symbol)
    period : str  (Data period (e.g., "1y", "2y", "5y"))
    Returns
    -------
    pd.DataFrame  (DataFrame with Date, Close, and Returns)
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is required for real market data")
    
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)

    if hist.empty:
        raise ValueError(f"No data found for {ticker}")
    
    df = pd.DataFrame({
        'Date': hist.index,
        'Close': hist['Close'].values
    })
    df['Returns'] = df['Close'].pct_change()
    df = df.dropna()

    return df

def fetch_multiple_stocks(tickers: list, period: str = "2y") -> pd.DataFrame:
    """ 
    Fetch data for multiple stocks and return dataframe with returns for each stock"""
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is required for real market data")
    
    returns_dict = {}

    for ticker in tickers:
        try:
            df = fetch_stock_data(ticker, period)
            returns_dict[ticker] = df.set_index('Date')['Returns']
        except Exception as e:
            print(f"Warning: Could not fetch {ticker}: {e}")

    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna()

    return returns_df

def analyze_single_stock_var(ticker: str, confidence_level: float = 0.95,
                             period: str = "2y") -> dict:
    """Complete VaR analysis for a single stock"""
    # Fetch data
    df = fetch_stock_data(ticker, period)
    returns = df['Returns'].values

    # Calculate VaR using different methods
    var_hist = historical_var(returns, confidence_level)
    var_param = parametric_var(returns, confidence_level=confidence_level)
    var_mc = monte_carlo_var(returns, confidence_level=confidence_level, n_simulations=100000)
    
    # Calculate Expected Shortfall
    es_hist = expected_shortfall_historical(returns, confidence_level)
    es_param = expected_shortfall_parametric(returns, confidence_level=confidence_level)
    
    # Time scaling
    var_10day = scale_var(var_hist, 10)

    # Statistics
    results = {
        'ticker': ticker,
        'n_observations': len(returns),
        'start_date': df['Date'].iloc[0],
        'end_date': df['Date'].iloc[-1],
        'mean_return': np.mean(returns),
        'volatility': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'var_historical': var_hist,
        'var_parametric': var_param,
        'var_monte_carlo': var_mc,
        'es_historical': es_hist,
        'es_parametric': es_param,
        'var_10day': var_10day,
        'returns': returns,
        'dates': df['Date'].values
    }

    return results

def analyze_portfolio_var(tickers: list, weights: np.ndarray,
                          portfolio_value: float = 1000000,
                          confidence_level: float = 0.95,
                          period: str = "2y") -> dict:
    """Complete VaR analysis for a portfolio."""
    # Fetch data
    returns_df = fetch_multiple_stocks(tickers, period)

    # Calculate covariance matrix
    cov_matrix = calculate_covariance_matrix(returns_df, method='sample')

    # Calculate correlation matrix
    std_devs = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
    
    # Portfolio VaR
    portfolio_results = portfolio_var_parametric(
        weights, cov_matrix, confidence_level, portfolio_value
    )

    # Also calculate historical VaR for portfolio
    portfolio_returns = (returns_df.values @ weights)
    var_hist = historical_var(portfolio_returns, confidence_level) * portfolio_value
    es_hist = expected_shortfall_historical(portfolio_returns, confidence_level) * portfolio_value
    
    results = {
        'tickers': tickers,
        'weights': weights,
        'portfolio_value': portfolio_value,
        'n_observations': len(returns_df),
        'correlation_matrix': corr_matrix,
        'covariance_matrix': cov_matrix,
        'portfolio_var_parametric': portfolio_results['portfolio_var'],
        'portfolio_var_historical': var_hist,
        'undiversified_var': portfolio_results['undiversified_var'],
        'diversification_benefit': portfolio_results['diversification_benefit'],
        'component_var': portfolio_results['component_var'],
        'individual_vars': portfolio_results['individual_vars'],
        'es_historical': es_hist,
        'returns_df': returns_df,
        'portfolio_returns': portfolio_returns
    }
    
    return results

def plot_var_analysis(results: dict, save_path: str = None):
    """Create visualizations of VaR analysis for a single stock."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    returns = results['returns']
    var_hist = results['var_historical']
    es_hist = results['es_historical']

    # 1. return Distribution with VaR
    ax1 = axes[0, 0]
    ax1.hist(returns * 100, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(-var_hist * 100, color='red', linestyle='--', linewidth=2, label=f'95% VaR: {var_hist*100:.2f}%')
    ax1.axvline(-es_hist * 100, color='darkred', linestyle=':', linewidth=2, label=f'95% ES: {es_hist*100:.2f}%')
    ax1.set_xlabel('Daily Return (%)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{results["ticker"]} - Return Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Return Time Series
    ax2 = axes[0, 1]
    ax2.plot(results['dates'], returns * 100, color='steelblue', alpha=0.7, linewidth=0.8)
    ax2.axhline(-var_hist * 100, color='red', linestyle='--', linewidth=1.5, label=f'95% VaR')
    ax2.axhline(var_hist * 100, color='green', linestyle='--', linewidth=1.5)
    ax2.fill_between(results['dates'], -var_hist * 100, var_hist * 100, alpha=0.1, color='green')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Daily Return (%)')
    ax2.set_title(f'{results["ticker"]} - Returns Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. VaR Comparison
    ax3 = axes[1, 0]
    methods = ['Historical', 'Parametric', 'Monte Carlo']
    var_values = [results['var_historical'] * 100, 
                  results['var_parametric'] * 100, 
                  results['var_monte_carlo'] * 100]
    colors = ['steelblue', 'coral', 'seagreen']
    bars = ax3.bar(methods, var_values, color=colors, edgecolor='black')
    ax3.set_ylabel('VaR (%)')
    ax3.set_title('VaR by Method (95% Confidence)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, var_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 4. VaR vs ES Comparison
    ax4 = axes[1, 1]
    x = np.arange(2)
    width = 0.35
    var_vals = [results['var_historical'] * 100, results['var_parametric'] * 100]
    es_vals = [results['es_historical'] * 100, results['es_parametric'] * 100]
    
    bars1 = ax4.bar(x - width/2, var_vals, width, label='VaR', color='steelblue', edgecolor='black')
    bars2 = ax4.bar(x + width/2, es_vals, width, label='ES', color='coral', edgecolor='black')
    
    ax4.set_ylabel('Risk Measure (%)')
    ax4.set_title('VaR vs Expected Shortfall')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Historical', 'Parametric'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()

def plot_portfolio_var(results: dict, save_path: str = None):
    """
    Create visualization of portfolio VaR analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    tickers = results['tickers']
    weights = results['weights']
    
    # 1. Portfolio Weights
    ax1 = axes[0, 0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(tickers)))
    ax1.pie(weights, labels=tickers, autopct='%1.1f%%', colors=colors, 
            explode=[0.02]*len(tickers), shadow=True)
    ax1.set_title('Portfolio Allocation')
    
    # 2. Correlation Heatmap
    ax2 = axes[0, 1]
    corr_matrix = results['correlation_matrix']
    im = ax2.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(tickers)))
    ax2.set_yticks(range(len(tickers)))
    ax2.set_xticklabels(tickers)
    ax2.set_yticklabels(tickers)
    plt.colorbar(im, ax=ax2, label='Correlation')
    ax2.set_title('Correlation Matrix')
    
    # Add correlation values
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax2.text(j, i, f'{corr_matrix[i,j]:.2f}', ha='center', va='center', fontsize=10)
    
    # 3. Diversified vs Undiversified VaR
    ax3 = axes[1, 0]
    var_types = ['Diversified\nVaR', 'Undiversified\nVaR']
    var_values = [results['portfolio_var_parametric'], results['undiversified_var']]
    colors = ['seagreen', 'coral']
    bars = ax3.bar(var_types, var_values, color=colors, edgecolor='black')
    ax3.set_ylabel('VaR ($)')
    ax3.set_title(f'Diversification Benefit: ${results["diversification_benefit"]:,.0f}')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, var_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'${val:,.0f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Component VaR (contribution of each asset)
    ax4 = axes[1, 1]
    component_var = results['component_var']
    colors = plt.cm.Set3(np.linspace(0, 1, len(tickers)))
    bars = ax4.bar(tickers, component_var, color=colors, edgecolor='black')
    ax4.set_ylabel('Component VaR ($)')
    ax4.set_title('Risk Contribution by Asset')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()

def plot_backtest(returns: np.ndarray, var_estimates: np.ndarray,
                  confidence_level: float = 0.95, 
                  title: str = "VaR Backtest",
                  save_path: str = None):
    """
    Visualize VaR backtesting results.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    n = len(returns)
    dates = np.arange(n)
    
    # 1. Returns vs VaR
    ax1 = axes[0]
    ax1.plot(dates, returns * 100, color='steelblue', alpha=0.7, linewidth=0.8, label='Returns')
    ax1.plot(dates, -var_estimates * 100, color='red', linewidth=1.5, label='VaR (95%)')
    
    # Mark exceedances
    exceedances = returns < -var_estimates
    exceedance_dates = dates[exceedances]
    exceedance_returns = returns[exceedances]
    ax1.scatter(exceedance_dates, exceedance_returns * 100, color='red', s=50, 
                zorder=5, label=f'Exceedances ({np.sum(exceedances)})')
    
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Return (%)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative Exceedances
    ax2 = axes[1]
    cumulative_exceedances = np.cumsum(exceedances)
    expected_exceedances = (1 - confidence_level) * dates
    
    ax2.plot(dates, cumulative_exceedances, color='red', linewidth=2, label='Actual Exceedances')
    ax2.plot(dates, expected_exceedances, color='black', linestyle='--', linewidth=2, label='Expected Exceedances')
    ax2.fill_between(dates, expected_exceedances * 0.5, expected_exceedances * 1.5, 
                     alpha=0.2, color='gray', label='Acceptable Range')
    
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Cumulative Exceedances')
    ax2.set_title('Cumulative Exceedances vs Expected')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()

if __name__ == "__main__":
    print("VaR Analysis with Real Market Data")
    print("=" * 70)
    
    if not YFINANCE_AVAILABLE:
        print("\nyfinance not available. Please install: pip install yfinance")
        print("Exiting...")
        exit()
    
    # 1. Single Stock Analysis
    print("1. SINGLE STOCK VAR ANALYSIS")
        
    ticker = "AAPL"
    print(f"\nAnalyzing {ticker}...")
    
    try:
        results = analyze_single_stock_var(ticker, confidence_level=0.95, period="2y")
        
        print(f"\n{ticker} VaR Analysis")
        print("-" * 50)
        print(f"Period:              {results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')}")
        print(f"Observations:        {results['n_observations']} days")
        print(f"Mean daily return:   {results['mean_return']*100:.4f}%")
        print(f"Daily volatility:    {results['volatility']*100:.4f}%")
        print(f"Worst day:           {results['min_return']*100:.4f}%")
        print(f"Best day:            {results['max_return']*100:.4f}%")
        
        print(f"\n{'Method':<25} {'1-day VaR':>15} {'10-day VaR':>15}")
        print("-" * 55)
        print(f"{'Historical':<25} {results['var_historical']*100:>14.4f}% {scale_var(results['var_historical'], 10)*100:>14.4f}%")
        print(f"{'Parametric':<25} {results['var_parametric']*100:>14.4f}% {scale_var(results['var_parametric'], 10)*100:>14.4f}%")
        print(f"{'Monte Carlo':<25} {results['var_monte_carlo']*100:>14.4f}% {scale_var(results['var_monte_carlo'], 10)*100:>14.4f}%")
        
        print(f"\n{'Expected Shortfall':<25} {'ES (%)':>15} {'ES/VaR':>15}")
        print("-" * 55)
        print(f"{'Historical':<25} {results['es_historical']*100:>14.4f}% {results['es_historical']/results['var_historical']:>14.2f}x")
        print(f"{'Parametric':<25} {results['es_parametric']*100:>14.4f}% {results['es_parametric']/results['var_parametric']:>14.2f}x")
        
        # Create visualization
        plot_var_analysis(results, save_path="images/Single_stock_var.png")
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
    
    # 2. Portfolio Analysis
    print("2. PORTFOLIO VAR ANALYSIS")
        
    tickers = ["AAPL", "MSFT", "GOOGL", "JPM"]
    weights = np.array([0.30, 0.30, 0.25, 0.15])
    portfolio_value = 1000000
    
    print(f"\nPortfolio:")
    for t, w in zip(tickers, weights):
        print(f"  {t}: {w:.0%} (${portfolio_value * w:,.0f})")
    
    try:
        portfolio_results = analyze_portfolio_var(
            tickers, weights, portfolio_value, 
            confidence_level=0.95, period="2y"
        )
        
        print(f"\nPortfolio VaR Analysis")
        print("-" * 50)
        print(f"Portfolio Value:        ${portfolio_value:,}")
        print(f"Observations:           {portfolio_results['n_observations']} days")
        
        print(f"\n{'Metric':<30} {'Value':>20}")
        print("-" * 50)
        print(f"{'Diversified VaR (Parametric)':<30} ${portfolio_results['portfolio_var_parametric']:>19,.2f}")
        print(f"{'Diversified VaR (Historical)':<30} ${portfolio_results['portfolio_var_historical']:>19,.2f}")
        print(f"{'Undiversified VaR':<30} ${portfolio_results['undiversified_var']:>19,.2f}")
        print(f"{'Diversification Benefit':<30} ${portfolio_results['diversification_benefit']:>19,.2f}")
        print(f"{'Expected Shortfall':<30} ${portfolio_results['es_historical']:>19,.2f}")
        
        print(f"\nCorrelation Matrix:")
        corr_df = pd.DataFrame(portfolio_results['correlation_matrix'], 
                               index=tickers, columns=tickers)
        print(corr_df.round(3).to_string())
        
        print(f"\nComponent VaR (Risk Contribution):")
        for t, cv in zip(tickers, portfolio_results['component_var']):
            print(f"  {t}: ${cv:>15,.2f}")
        
        # Create visualization
        plot_portfolio_var(portfolio_results, save_path="images/Portfolio_var.png")
        
    except Exception as e:
        print(f"Error in portfolio analysis: {e}")
    
    # 3. Backtesting
    print("3. BACKTESTING")
    print("-" * 70)
    
    try:
        # Use single stock data for backtesting
        returns = results['returns']
        window = 250
        
        # Calculate rolling VaR
        var_estimates = rolling_var(returns, window=window, confidence_level=0.95, method='historical')
        
        # Backtest on period with VaR estimates
        test_returns = returns[window:]
        test_var = var_estimates[window:]
        
        backtest_results = backtest_var(test_returns, test_var, confidence_level=0.95)
        
        print(f"\nBacktest Results for {ticker}")
        print("-" * 50)
        print(f"Test period:            {backtest_results['n_observations']} days")
        print(f"Expected exceedances:   {backtest_results['expected_exceedances']:.1f}")
        print(f"Actual exceedances:     {backtest_results['n_exceedances']}")
        print(f"Exceedance rate:        {backtest_results['exceedance_rate']*100:.2f}% (expected: {backtest_results['expected_rate']*100:.2f}%)")
        print(f"Kupiec test p-value:    {backtest_results['kupiec_p_value']:.4f}")
        print(f"Traffic light zone:     {backtest_results['zone']}")
        
        if backtest_results['kupiec_p_value'] > 0.05:
            print("\nConclusion: Model passes backtest (not rejected at 5% significance)")
        else:
            print("\nConclusion: Model fails backtest (rejected at 5% significance)")
        
        # Create backtest visualization
        plot_backtest(test_returns, test_var, confidence_level=0.95,
                     title=f"{ticker} VaR Backtest", save_path="images/Backtest.png")
        
    except Exception as e:
        print(f"Error in backtesting: {e}")
    
    print("VaR Analysis Complete")
    


 
