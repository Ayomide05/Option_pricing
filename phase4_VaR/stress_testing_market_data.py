"""Stress Testing with Real Market Data: This module applies stress testing to real market data"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
import os

# Import from our modules
from var import (
    historical_var,
    parametric_var,
    parametric_var_ewma,
    expected_shortfall_historical
)
from stress_testing import (
    historical_stress_test,
    hypothetical_stress_test,
    reverse_stress_test,
    sensitivity_analysis,
    compare_var_methods,
    get_sector,
    get_sector_indices,
    HISTORICAL_SCENARIOS
)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed. Run: pip install yfinance")


def fetch_portfolio_data(tickers: List[str],
                         period: str = "2y") -> Dict:
    """
    Fetch real market data for a portfolio.
    Parameters
    ----------
    tickers : List of stock tickers
    period : Data period (e.g., "1y", "2y", "5y")
    Returns
    -------
    Dict with returns, prices, volatilities, correlation matrix
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is required. Install with: pip install yfinance")
    
    print(f"Fetching data for {tickers}...")

    #Fetch data
    data = yf.download(tickers, period=period, progress=False)['Close']

    # Handle single ticker case
    if len(tickers) == 1:
        data = data.to_frame(name=tickers[0])

    # Calculate returns
    returns = data.pct_change().dropna()

    # Calculate statistics
    volatilities = returns.std().values
    correlation_matrix = returns.corr().values
    covariance_matrix = returns.cov().values

    # Calculate betas (relative to equal-weighted portfolio as proxy for market)
    market_returns = returns.mean(axis=1)
    betas = []
    for ticker in tickers:
        cov_with_market = returns[ticker].cov(market_returns)
        market_var = market_returns.var()
        beta = cov_with_market / market_var if market_var > 0 else 1.0
        betas.append(beta)

    print(f"  Data period: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Trading days: {len(returns)}")
    
    return {
        'tickers': tickers,
        'prices': data,
        'returns': returns,
        'volatilities': volatilities,
        'correlation_matrix': correlation_matrix,
        'covariance_matrix': covariance_matrix,
        'betas': np.array(betas),
        'n_days': len(returns)
    }

# STRESS TESTING WITH REAL DATA

def run_stress_test_real_data(tickers: List[str],
                              weights: np.ndarray,
                              portfolio_value: float = 1_000_000,
                              period: str = "2y",
                              confidence_level: float = 0.95) -> Dict:
    """Run comprehensive stress testing on real market data.
    Parameters
    ----------
    tickers : List of stock tickers
    weights : Portfolio weights
    portfolio_value : Total portfolio value ($)
    period : Historical data period
    confidence_level : VaR confidence level
    Returns
    -------
    Dict with all stress test results
    """
    # Fetch real data
    data = fetch_portfolio_data(tickers, period)
    # Calculate portfolio returns
    portfolio_returns = (data['returns'].values @ weights)
    # Get sectors
    sectors = [get_sector(t) for t in tickers]

    print(f"\nPortfolio Configuration:")
    print(f"  Value: ${portfolio_value:,}")
    print(f"  Assets: {tickers}")
    print(f"  Sectors: {sectors}")
    print(f"  Weights: {[f'{w:.0%}' for w in weights]}")

    # 1. VaR Methods Comparison
    print("1. VAR METHODS (Real Market Data)")
    print("=" * 80)

    var_comparison = compare_var_methods(portfolio_returns, confidence_level)

    print(f"\n{'Method':<20} {'Value (%)':>12}")
    print("-" * 35)
    for _, row in var_comparison.iterrows():
        print(f"{row['Method']:<20} {row['Value (%)']:>11.4f}%")
    
    # 2. Historical Stress Testing
    print("2. HISTORICAL STRESS TESTING (Real Betas)")
    print("=" * 80)
    
    print(f"\n  Using real betas: {[f'{b:.2f}' for b in data['betas']]}")
    
    historical_results = historical_stress_test(
        portfolio_value, weights, data['betas'],
        scenarios=['covid_crash_2020', 'financial_crisis_worst_2008', 'lehman_2008',
                   'black_monday_1987', 'dotcom_crash_2000', 'sept_11_2001']
    )
    
    print(f"\n{'Scenario':<30} {'Market':>10} {'Portfolio':>12} {'Loss ($)':>15}")
    print("-" * 70)
    for _, row in historical_results.iterrows():
        print(f"{row['Scenario']:<30} {row['Market Return (%)']:>9.2f}% "
              f"{row['Portfolio Return (%)']:>11.2f}% ${row['Portfolio Loss ($)']:>14,.0f}")
        
    # 3. Hypothetical Stress Testing
    print("3. HYPOTHETICAL STRESS TESTING (Sector-Aware)")
    print("=" * 80)
    
    hypothetical_results = hypothetical_stress_test(portfolio_value, weights, tickers)
    
    print(f"\n{'Scenario':<45} {'Return':>12} {'Loss ($)':>15}")
    print("-" * 75)
    for _, row in hypothetical_results.iterrows():
        print(f"{row['Scenario']:<45} {row['Portfolio Return (%)']:>11.2f}% "
              f"${row['Portfolio Loss ($)']:>14,.0f}")
        
    # 4. Reverse Stress Testing
    print("\n" + "=" * 80)
    print("4. REVERSE STRESS TESTING")
    print("=" * 80)
    
    target_losses = [50_000, 100_000, 150_000, 200_000]
    reverse_results = reverse_stress_test(portfolio_value, target_losses, weights, tickers)
    
    print(f"\n{'Target Loss':>15} {'Uniform Drop':>15} {'Largest Position':>20}")
    print("-" * 55)
    for _, row in reverse_results.iterrows():
        print(f"${row['Target Loss ($)']:>14,.0f} {row['Uniform Decline (%)']:>14.2f}% "
              f"{row['Largest Position Only (%)']:>19.2f}%")
        
    # 5. Sensitivity Analysis with Real Parameters
    print("\n" + "=" * 80)
    print("5. SENSITIVITY ANALYSIS (Real Volatilities & Correlations)")
    print("=" * 80)
    
    print(f"\n  Real daily volatilities: {[f'{v*100:.2f}%' for v in data['volatilities']]}")
    print(f"  Real correlation range: {data['correlation_matrix'][np.triu_indices(len(tickers), 1)].min():.2f} to {data['correlation_matrix'][np.triu_indices(len(tickers), 1)].max():.2f}")
    
    sensitivity_results = sensitivity_analysis(
        portfolio_value, weights, 
        data['volatilities'], 
        data['correlation_matrix'],
        confidence_level
    )
    
    print(f"\n{'Scenario':<40} {'VaR ($)':>12} {'vs Base':>12}")
    print("-" * 65)
    for _, row in sensitivity_results.iterrows():
        change_str = f"{row['Change vs Base (%)']:+.1f}%" if row['Change vs Base (%)'] != 0 else "—"
        print(f"{row['Scenario']:<40} ${row['VaR ($)']:>11,.0f} {change_str:>12}")

    # 6. Compare VaR vs Worst Historical Day
    print("\n" + "=" * 80)
    print("6. VAR vs ACTUAL WORST DAYS")
    print("=" * 80)
    
    worst_days = portfolio_returns.argsort()[:5]
    worst_returns = portfolio_returns[worst_days]
    var_95 = historical_var(portfolio_returns, 0.95)
    var_99 = historical_var(portfolio_returns, 0.99)
    
    print(f"\n  95% VaR: {var_95*100:.2f}% (${var_95 * portfolio_value:,.0f})")
    print(f"  99% VaR: {var_99*100:.2f}% (${var_99 * portfolio_value:,.0f})")
    print(f"\n  Worst 5 actual days:")
    for i, (idx, ret) in enumerate(zip(worst_days, worst_returns)):
        date = data['returns'].index[idx].strftime('%Y-%m-%d')
        loss = -ret * portfolio_value
        exceeded_95 = "⚠️ Exceeded 95% VaR" if -ret > var_95 else ""
        print(f"    {i+1}. {date}: {ret*100:+.2f}% (${loss:,.0f}) {exceeded_95}")
    
    return {
        'data': data,
        'portfolio_returns': portfolio_returns,
        'var_comparison': var_comparison,
        'historical_results': historical_results,
        'hypothetical_results': hypothetical_results,
        'reverse_results': reverse_results,
        'sensitivity_results': sensitivity_results,
    }   

# VISUALIZATION
def  plot_stress_test_real_data(results: Dict, save_path: str = None):
    """
    Create visualization for real data stress testing.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    
    data = results['data']
    portfolio_returns = results['portfolio_returns']
    
    # 1. Portfolio Return Distribution with VaR lines
    ax1 = fig.add_subplot(gs[0, 0])
    
    var_95 = historical_var(portfolio_returns, 0.95)
    var_99 = historical_var(portfolio_returns, 0.99)
    es_95 = expected_shortfall_historical(portfolio_returns, 0.95)
    
    ax1.hist(portfolio_returns * 100, bins=50, density=True, alpha=0.7, 
             color='steelblue', edgecolor='black', label='Daily Returns')
    ax1.axvline(-var_95 * 100, color='orange', linestyle='--', linewidth=2, 
                label=f'95% VaR: {var_95*100:.2f}%')
    ax1.axvline(-var_99 * 100, color='red', linestyle='--', linewidth=2,
                label=f'99% VaR: {var_99*100:.2f}%')
    ax1.axvline(-es_95 * 100, color='darkred', linestyle=':', linewidth=2,
                label=f'95% ES: {es_95*100:.2f}%')
    ax1.set_xlabel('Daily Return (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Portfolio Return Distribution (Real Data)', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Correlation Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    
    corr = data['correlation_matrix']
    im = ax2.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(data['tickers'])))
    ax2.set_yticks(range(len(data['tickers'])))
    ax2.set_xticklabels(data['tickers'])
    ax2.set_yticklabels(data['tickers'])
    ax2.set_title('Real Correlation Matrix', fontweight='bold')
    
    for i in range(len(data['tickers'])):
        for j in range(len(data['tickers'])):
            ax2.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=10)
    
    plt.colorbar(im, ax=ax2, shrink=0.8)
    
    # 3. Historical Stress Scenarios
    ax3 = fig.add_subplot(gs[1, 0])
    
    hist_df = results['historical_results']
    scenarios = hist_df['Scenario'].head(6).values
    losses = hist_df['Portfolio Loss ($)'].head(6).values / 1000
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(scenarios)))
    
    bars = ax3.barh(range(len(scenarios)), losses, color=colors, edgecolor='black')
    ax3.set_yticks(range(len(scenarios)))
    ax3.set_yticklabels(scenarios, fontsize=9)
    ax3.set_xlabel('Portfolio Loss ($000s)')
    ax3.set_title('Historical Stress Scenarios', fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    
    for bar, loss in zip(bars, losses):
        ax3.text(loss + 2, bar.get_y() + bar.get_height()/2,
                 f'${loss:.0f}K', va='center', fontsize=9)
    
    # 4. Sensitivity Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    
    sens_df = results['sensitivity_results']
    sens_scenarios = sens_df['Scenario'].values
    var_values = sens_df['VaR ($)'].values / 1000
    
    colors4 = ['steelblue' if 'Base' in s else
               'orange' if 'Vol' in s else
               'green' if 'Corr' in s and 'Crisis' not in s else
               'red' for s in sens_scenarios]
    
    bars4 = ax4.bar(range(len(sens_scenarios)), var_values, color=colors4, edgecolor='black')
    ax4.set_xticks(range(len(sens_scenarios)))
    ax4.set_xticklabels(sens_scenarios, rotation=45, ha='right', fontsize=7)
    ax4.set_ylabel('VaR ($000s)')
    ax4.set_title('VaR Sensitivity (Real Parameters)', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. VaR Methods Comparison
    ax5 = fig.add_subplot(gs[2, 0])
    
    var_df = results['var_comparison']
    methods = var_df['Method'].values
    values = var_df['Value (%)'].values
    
    colors5 = ['steelblue', 'coral', 'seagreen', 'darkred']
    bars5 = ax5.bar(methods, values, color=colors5, edgecolor='black', width=0.6)
    ax5.set_ylabel('VaR / ES (%)')
    ax5.set_title('VaR Methods (Real Data)', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars5, values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. Hypothetical Scenarios
    ax6 = fig.add_subplot(gs[2, 1])
    
    hypo_df = results['hypothetical_results']
    hypo_scenarios = hypo_df['Scenario'].values[:6]
    hypo_losses = hypo_df['Portfolio Loss ($)'].values[:6] / 1000
    colors6 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(hypo_scenarios)))
    
    bars6 = ax6.barh(range(len(hypo_scenarios)), hypo_losses, color=colors6, edgecolor='black')
    ax6.set_yticks(range(len(hypo_scenarios)))
    ax6.set_yticklabels(hypo_scenarios, fontsize=8)
    ax6.set_xlabel('Portfolio Loss ($000s)')
    ax6.set_title('Hypothetical Scenarios (Sector-Aware)', fontweight='bold')
    ax6.invert_yaxis()
    ax6.grid(True, alpha=0.3, axis='x')
    
    for bar, loss in zip(bars6, hypo_losses):
        ax6.text(loss + 2, bar.get_y() + bar.get_height()/2,
                 f'${loss:.0f}K', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nSaved: {save_path}")
    
    plt.close()

if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    
    print("STRESS TESTING WITH REAL MARKET DATA")
    print("=" * 80)
    
    # Portfolio configuration
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM']
    weights = np.array([0.30, 0.30, 0.25, 0.15])
    portfolio_value = 1_000_000
    
    # Run stress testing
    results = run_stress_test_real_data(
        tickers=tickers,
        weights=weights,
        portfolio_value=portfolio_value,
        period="2y",
        confidence_level=0.95
    )
    
    # Generate visualization
    print("GENERATING VISUALIZATION")
    print("=" * 80)
    
    plot_stress_test_real_data(results, save_path="images/16_stress_testing_real_data.png")
    
    # Summary
    print("SUMMARY")
    print("=" * 80)
    
    var_df = results['var_comparison']
    hist_df = results['historical_results']
    sens_df = results['sensitivity_results']
    
    print(f"""
    REAL MARKET DATA ANALYSIS
    
    1. VaR (Real Data):
       - Historical: {var_df.iloc[0]['Value (%)']:.2f}%
       - Parametric: {var_df.iloc[1]['Value (%)']:.2f}%
       - EWMA:       {var_df.iloc[2]['Value (%)']:.2f}%
       - ES:         {var_df.iloc[3]['Value (%)']:.2f}%
    
    2. Worst Historical Scenario:
       {hist_df.iloc[0]['Scenario']} → ${hist_df.iloc[0]['Portfolio Loss ($)']:,.0f}
    
    3. Real Correlations:
       Range: {results['data']['correlation_matrix'][np.triu_indices(len(tickers), 1)].min():.2f} to {results['data']['correlation_matrix'][np.triu_indices(len(tickers), 1)].max():.2f}
    
    4. Crisis Impact:
       Normal VaR:  ${sens_df.iloc[0]['VaR ($)']:,.0f}
       Crisis VaR:  ${sens_df.iloc[-1]['VaR ($)']:,.0f}
       Increase:    {sens_df.iloc[-1]['Change vs Base (%)']:.0f}%
    """)
    
    print("Stress Testing with Real Data Complete")
    



    