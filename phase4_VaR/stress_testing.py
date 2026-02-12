""" Stress Testing Module.
Stress testing complements VaR by asking: "What happens under extreme but plausible scenarios
VaR tells us risk under normal conditions while Stress Testing tells us risk when things go wrong.
This module implements:
1. Historical scenario analysis (actual crisis periods)
2. Hypothetical scenario analysis (user-defined shocks)
3. Reverse stress testing (what would cause X loss?) 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from scipy import stats
import os

from var import (
    historical_var,
    historical_var_weighted,
    parametric_var,
    parametric_var_ewma,
    expected_shortfall_historical
)

# SECTOR CLASSIFICATION
SECTOR_MAPPING = {
    'tech' : ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AMD', 'TSLA',
             'AMZN', 'NFLX', 'ADBE', 'CRM', 'INTC', 'CSCO', 'ORCL', 'IBM'],
    'financial': ['JPM', 'BAC', 'GS', 'MS', 'C', 'WFC', 'BLK', 'AXP',
                  'V', 'MA', 'COF', 'USB', 'PNC', 'SCHW'],
    'healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'LLY',
                   'BMY', 'AMGN', 'GILD', 'CVS', 'CI', 'HUM'],
    'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO',
               'OXY', 'HAL', 'DVN', 'BKR'],
    'consumer': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE',
                 'SBUX', 'TGT', 'LOW', 'TJX', 'DG', 'DLTR'],
}

def get_sector(ticker: str) -> str:
    """Get the sector for a given ticker."""
    ticker = ticker.upper()
    for sector, tickers in SECTOR_MAPPING.items():
        if ticker in tickers:
            return sector
    return 'other'

def get_sector_indices(tickers: List[str], sector: str) -> List[int]:
    """Get indices of tickers that belong to a specific sector."""
    return [i for i, t in enumerate(tickers) if get_sector(t) == sector]


# HISTORICAL SCENARIOS
HISTORICAL_SCENARIOS = {
    'black_monday_1987': {
        'name' : 'Black Monday (1987)',
        'date' : '1987-10-19',
        'description': 'Largest single-day percentage decline in stock market history',
        'sp500_return': -0.2047,  # -20.47%
        'stock_multiplier': 1.0,  # Stocks fell with market
    },
    'asian_crisis_1997': {
        'name': 'Asian Financial Crisis (1997)',
        'date': '1997-10-27',
        'description': 'Currency crisis spread from Thailand across Asia',
        'sp500_return': -0.0685,
        'stock_multiplier': 1.2,  # Some stocks fell more
    },
    'dotcom_crash_2000': {
        'name': 'Dot-com Crash (2000)',
        'date': '2000-04-14',
        'description': 'Tech bubble burst',
        'sp500_return': -0.0584,
        'stock_multiplier': 1.5,  # Tech stocks fell much more than S&P
    },
    'sept_11_2001': {
        'name': '9/11 Attacks (2001)',
        'date': '2001-09-17',
        'description': 'First trading day after terrorist attacks',
        'sp500_return': -0.0489,
        'stock_multiplier': 1.0,  # Everything fell together
    },
    'lehman_2008': {
        'name': 'Lehman Brothers Collapse (2008)',
        'date': '2008-09-29',
        'description': 'Largest single-day point drop at the time',
        'sp500_return': -0.0879,
        'stock_multiplier': 1.2,  # Financials fell more
    },
    'financial_crisis_worst_2008': {
        'name': 'Financial Crisis Worst Day (2008)',
        'date': '2008-10-15',
        'description': 'Peak of 2008 financial crisis fear',
        'sp500_return': -0.0903,
        'stock_multiplier': 1.3,  # High beta stocks fell more
    },
    'flash_crash_2010': {
        'name': 'Flash Crash (2010)',
        'date': '2010-05-06',
        'description': 'Market dropped 9% in minutes before recovering',
        'sp500_return': -0.0347,
        'stock_multiplier': 1.5,  # Some stocks hit circuit breakers
    },
    'covid_crash_2020': {
        'name': 'COVID-19 Crash (2020)',
        'date': '2020-03-16',
        'description': 'Pandemic fears caused largest point drop in history',
        'sp500_return': -0.1198,
        'stock_multiplier': 1.0,  # Everything fell together
    },
}

# STRESS TESTING FUNCTIONS

def historical_stress_test(portfolio_value: float,
                           weights: np.ndarray = None, asset_betas: np.ndarray = None,
                           scenarios: List[str] = None) -> pd.DataFrame:
    """Apply historical stress scenarios to a portfolio.
    Parameters
    ----------
    portfolio_value :  (Total portfolio value ($))
    weights : Portfolio weights (if None, assumes single asset)
    asset_betas : Beta of each asset to S&P 500 (if None, assumes beta = 1)
    scenarios : List of scenario keys to test (if None, tests all)
    Returns
    --------
    Stress test results (DataFrame with scenario results, sorted by less)
    """

    if scenarios is None:
        scenarios = list(HISTORICAL_SCENARIOS.keys())
    if weights is None:
        weights = np.array([1.0])
    if asset_betas is None:
        asset_betas = np.ones(len(weights))

    if len(weights) != len(asset_betas):
        raise ValueError(f"Mismatch! You provided {len(weights)} weights but {len(asset_betas)} betas.")
    
    # Portfolio beta
    portfolio_beta = np.sum(weights * asset_betas)

    results = []

    for scenario_key in scenarios:
        scenario = HISTORICAL_SCENARIOS[scenario_key]

        # Estimate portfolio return based on beta and market return
        market_return = scenario['sp500_return']
        multiplier = scenario['stock_multiplier']

        portfolio_return = portfolio_beta * market_return * multiplier
        portfolio_loss = -portfolio_return * portfolio_value

        results.append({
            'Scenario': scenario['name'],
            'Date': scenario['date'],
            'Market Return (%)': market_return * 100,
            'Multiplier': multiplier,
            'Portfolio Return (%)': portfolio_return * 100,
            'Portfolio Loss ($)': portfolio_loss,
            'Description': scenario['description'],
        })

    df = pd.DataFrame(results)
    df = df.sort_values('Portfolio Loss ($)', ascending=False)

    return df

# HYPOTHETICAL SCENARIOS

def hypothetical_stress_test(portfolio_value: float,
                            weights: np.ndarray, 
                            tickers: List[str]) -> pd.DataFrame:
    """Apply historical scenarios with sector-aware shocks.
    Parameters
    ----------
    portfolio_value : Total portfolio value ($)
    weights : Portfolio weights
    tickers : List of ticker symbols
    Returns
    -------
    DataFrame with scenario results
    """
    n = len(weights)

    # Identify sector indices
    tech_indices = get_sector_indices(tickers, 'tech')
    financial_indices = get_sector_indices(tickers, 'financial')
    energy_indices = get_sector_indices(tickers, 'energy')

    scenarios = {}

    # Uniform scenarios
    scenarios['Mild Correction (-5%)'] = [-0.05] * n
    scenarios['Market correction (-10%)'] = [-0.10] * n
    scenarios['Sever Crash (-20%)'] = [-0.20] * n
    scenarios['Black Swan (-30%)'] = [-0.30] * n

    # Tech crash
    if tech_indices:
        tech_crash = [-0.10] * n
        for i in tech_indices:
            tech_crash[i] = -0.35
        scenarios['Tech Crash (Tech -35%, Others -10%)'] = tech_crash

    # Financial crisis
    if financial_indices:
        fin_crisis = [-0.10] * n
        for i in financial_indices:
            fin_crisis[i] = -0.45
        scenarios['Financial Crisis (Fin -45%, Others -10%)'] = fin_crisis
    
    # Energy shock
    if energy_indices:
        energy_shock = [-0.05] * n
        for i in energy_indices:
            energy_shock[i] = -0.30
        scenarios['Energy Crash (Energy -30%, Others -5%)'] = energy_shock

    # Interest rate shock
    rate_shock = [-0.08] * n
    for i in tech_indices:
        rate_shock[i] = -0.15
    for i in financial_indices:
        rate_shock[i] = -0.05
    if tech_indices or financial_indices:
        scenarios['Interest Rate Shock (+200bps)'] = rate_shock

    results = []

    for name, returns in scenarios.items():
        returns_array = np.array(returns)
        portfolio_return = np.sum(weights * returns_array)
        portfolio_loss = -portfolio_return * portfolio_value

        results.append({
            'Scenario': name,
            'Portfolio Return (%)': portfolio_return * 100,
            'Portfolio Loss ($)' : portfolio_loss
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('Portfolio Loss ($)', ascending=False)
    return df

def reverse_stress_test(portfolio_value: float,
                        target_losses: List[float],
                        weights: np.ndarray,
                        tickers: List[str] = None) -> pd.DataFrame:
    """Reverse stress test: This explains what market move that causes a specific loss?
    Parameters
    ----------
    portfolio_value : Total portfolio value ($)
    target_losses : List of target loss amounts
    weights : Portfolio weights
    tickers : List of ticker symbols (optional)
    Returns
    -------
    DataFrame with required market moves for each target loss
    """

    results = []

    for target_loss in target_losses:
        required_return = -target_loss / portfolio_value
        uniform_decline = required_return

        max_idx = np.argmax(weights)
        max_weight = weights[max_idx]
        single_asset_decline = required_return / max_weight

        result = {
            'Target Loss ($)' : target_loss,
            'Target Loss (%)' : (target_loss / portfolio_value) * 100,
            'Uniform Decline (%)': uniform_decline * 100,
            'Largest Position Only (%)': single_asset_decline * 100
        }

        if tickers:
            max_ticker = tickers[max_idx]
            result['Largest Position'] = max_ticker
            result['Sector'] = get_sector(max_ticker)

        results.append(result)

    return pd.DataFrame(results)

def sensitivity_analysis(portfolio_value: float,
                         weights: np.ndarray,
                         base_volatilities: np.ndarray,
                         base_correlation: np.ndarray,
                         confidence_level: float = 0.95) -> pd.DataFrame:
    """ Test how VaR changes when market conditions change.
    Parameters
    ----------
    portfolio_value : Total portfolio value ($)
    weights : Portfolio weights
    base_volatilities : Normal daily volatilities for each asset
    base_correlation : Normal correlation matrix
    confidence_level : VaR confidence level
    Returns
    -------
    DataFrame showing VaR under different stress conditions
    """
    z_score = stats.norm.ppf(confidence_level)

    def calculate_portfolio_var(vols, corr):
        cov_matrix = np.outer(vols, vols) * corr
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        return portfolio_vol * z_score * portfolio_value
    
    base_var = calculate_portfolio_var(base_volatilities, base_correlation)

    scenarios = []

    # Base case
    scenarios.append({
        'Scenario': 'Base Case (Normal Market)',
        'Volatility': 'Normal',
        'Correlation': 'Normal',
        'VaR ($)': base_var,
        'Change vs Base (%)' : 0,
    })

    # Volatility stress
    for vol_mult, label in [(1.5, '+50%'), (2.0, '+100%'), (3.0, '+200%')]:
        stressed_vol = base_volatilities * vol_mult
        var_stressed = calculate_portfolio_var(stressed_vol, base_correlation)
        scenarios.append({
            'Scenario': f'Volatility {label}',
            'Volatility': f'×{vol_mult}',
            'Correlation': 'Normal',
            'VaR ($)': var_stressed,
            'Change vs Base (%)': (var_stressed / base_var - 1) * 100,  # Percent change
        })

    # Correlation stress
    for corr_level in [0.6, 0.8, 0.95]:
        stressed_corr = np.ones_like(base_correlation) * corr_level
        np.fill_diagonal(stressed_corr, 1.0)
        var_stressed = calculate_portfolio_var(base_volatilities, stressed_corr)
        scenarios.append({
            'Scenario': f'Correlation → {corr_level}',
            'Volatility': 'Normal',
            'Correlation': f'All → {corr_level}',
            'VaR ($)': var_stressed,
            'Change vs Base (%)': (var_stressed / base_var - 1) * 100,
        })

    # Combined Stress
    crisis_scenarios = [
        (1.5, 0.7, 'Mild Crisis'),
        (2.0, 0.8, 'Moderate Crisis'),
        (3.0, 0.9, 'Severe Crisis'),
    ]

    for vol_mult, corr_level, label in crisis_scenarios:
        stressed_vol = base_volatilities * vol_mult
        stressed_corr = np.ones_like(base_correlation) * corr_level
        np.fill_diagonal(stressed_corr, 1.0)
        var_stressed = calculate_portfolio_var(stressed_vol, stressed_corr)
        scenarios.append({
            'Scenario': f'{label} (Vol×{vol_mult}, Corr→{corr_level})',
            'Volatility': f'×{vol_mult}',
            'Correlation': f'All → {corr_level}',
            'VaR ($)': var_stressed,
            'Change vs Base (%)': (var_stressed / base_var - 1) * 100,
        })
    
    return pd.DataFrame(scenarios)

def compare_var_methods(returns: np.ndarray, 
                        confidence_level: float = 0.95) -> pd.DataFrame:
    """Comapre vaR calculation methods (imported from var.py).
    Parameters
    ----------
    returns : Array of historical returns
    confidence_level : VaR confidence level
    Returns
    -------
    DataFrame comparing VaR methods
    """
    var_hist = historical_var(returns, confidence_level)
    var_param = parametric_var(returns, confidence_level)
    var_ewma = parametric_var_ewma(returns, confidence_level)
    es = expected_shortfall_historical(returns, confidence_level)
    
    results = pd.DataFrame({
        'Method': ['Historical', 'Parametric', 'EWMA', 'Expected Shortfall'],
        'Value (%)': [var_hist * 100, var_param * 100, var_ewma * 100, es * 100],
        'Best For': [
            'General use, captures fat tails',
            'Quick calculations, linear portfolios',
            'When recent volatility matters more',
            'Tail risk, Basel III compliance'
        ],
    })
    
    return results

#VISULAIZATION
def plot_stress_test_results(historical_df: pd.DataFrame,
                            hypothetical_df: pd.DataFrame,
                            sensitivity_df: pd.DataFrame,
                            var_comparison_df: pd.DataFrame,
                            save_path: str = None):
    """Create comprehensive stress testing visulaization."""
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # 1. Historical Stress Scenarios
    ax1 = fig.add_subplot(gs[0, 0])
    scenarios = historical_df['Scenario'].head(6).values
    losses = historical_df['Portfolio Loss ($)'].head(6).values / 1000
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(scenarios)))

    bars1 = ax1.barh(range(len(scenarios)), losses, color=colors, edgecolor='black')
    ax1.set_yticks(range(len(scenarios)))
    ax1.set_yticklabels(scenarios, fontsize=9)
    ax1.set_xlabel('Portfolio Loss ($000s)', fontsize=10)
    ax1.set_title('Historical Stress Scenarios', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    for bar, loss in zip(bars1, losses):
        ax1.text(loss + 2, bar.get_y() + bar.get_height()/2,
                 f'${loss:.0f}K', va='center', fontsize=9)
        
    # 2. Hypothetical Scenarios
    ax2 = fig.add_subplot(gs[0, 1])
    hypo_scenarios = hypothetical_df['Scenario'].values[:6]
    hypo_losses = hypothetical_df['Portfolio Loss ($)'].values[:6] / 1000
    colors2 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(hypo_scenarios)))
    
    bars2 = ax2.barh(range(len(hypo_scenarios)), hypo_losses, color=colors2, edgecolor='black')
    ax2.set_yticks(range(len(hypo_scenarios)))
    ax2.set_yticklabels(hypo_scenarios, fontsize=8)
    ax2.set_xlabel('Portfolio Loss ($000s)', fontsize=10)
    ax2.set_title('Hypothetical Stress Scenarios', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    for bar, loss in zip(bars2, hypo_losses):
        ax2.text(loss + 2, bar.get_y() + bar.get_height()/2,
                 f'${loss:.0f}K', va='center', fontsize=9)
        
    # 3. Sensitivity Analysis
    ax3 = fig.add_subplot(gs[1, 0])
    sens_scenarios = sensitivity_df['Scenario'].values
    var_values = sensitivity_df['VaR ($)'].values / 1000
    
    colors3 = ['steelblue' if 'Base' in s else
               'orange' if 'Vol' in s else
               'green' if 'Corr' in s and 'Crisis' not in s else
               'red' for s in sens_scenarios]
    
    bars3 = ax3.bar(range(len(sens_scenarios)), var_values, color=colors3, edgecolor='black')
    ax3.set_xticks(range(len(sens_scenarios)))
    ax3.set_xticklabels(sens_scenarios, rotation=45, ha='right', fontsize=7)
    ax3.set_ylabel('VaR ($000s)', fontsize=10)
    ax3.set_title('VaR Under Different Conditions', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. VaR Change vs Base
    ax4 = fig.add_subplot(gs[1, 1])
    changes = sensitivity_df['Change vs Base (%)'].values[1:]
    change_labels = sensitivity_df['Scenario'].values[1:]
    
    colors4 = ['green' if c < 50 else 'orange' if c < 100 else 'red' for c in changes]
    bars4 = ax4.bar(range(len(changes)), changes, color=colors4, edgecolor='black')
    ax4.set_xticks(range(len(changes)))
    ax4.set_xticklabels(change_labels, rotation=45, ha='right', fontsize=7)
    ax4.set_ylabel('Change vs Base (%)', fontsize=10)
    ax4.set_title('VaR Increase Under Stress', fontsize=12, fontweight='bold')
    ax4.axhline(y=50, color='orange', linestyle='--', alpha=0.7, linewidth=1)
    ax4.axhline(y=100, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. VaR Method Comparison
    ax5 = fig.add_subplot(gs[2, :])
    methods = var_comparison_df['Method'].values
    values = var_comparison_df['Value (%)'].values
    
    colors5 = ['steelblue', 'coral', 'seagreen', 'darkred']
    bars5 = ax5.bar(methods, values, color=colors5, edgecolor='black', width=0.5)
    ax5.set_ylabel('VaR / ES (%)', fontsize=10)
    ax5.set_title('VaR Methods Comparison (Imported from var.py)',
                  fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars5, values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()

    if save_path:
        full_path = os.path.join('images', save_path)
        plt.savefig(full_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {full_path}")
    
    plt.close()

if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)

    print("STRESS TESTING MODULE")
    print("=" * 80)

    portfolio_value = 1_000_000
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'JPM']
    weights = np.array([0.30, 0.30, 0.25, 0.15])
    betas = np.array([1.2, 1.1, 1.15, 1.0])
    volatilities = np.array([0.018, 0.016, 0.019, 0.015])

    correlation = np.array([
        [1.00, 0.43, 0.44, 0.31],
        [0.43, 1.00, 0.43, 0.29],
        [0.44, 0.43, 1.00, 0.28],
        [0.31, 0.29, 0.28, 1.00]
    ])

    # Simulate returns
    np.random.seed(42)
    n_days = 500
    cov_matrix = np.outer(volatilities, volatilities) * correlation
    asset_returns = np.random.multivariate_normal(np.zeros(4), cov_matrix, n_days)
    portfolio_returns = asset_returns @ weights

    print(f"\nPortfolio Configuration:")
    print(f"  Value: ${portfolio_value:,}")
    print(f"  Assets: {tickers}")
    print(f"  Sectors: {[get_sector(t) for t in tickers]}")
    print(f"  Weights: {[f'{w:.0%}' for w in weights]}")

    # 1. VaR Methods Comparison
    print("1. VAR METHODS COMPARISON (from var.py)")
    print("=" * 80)

    var_comparison = compare_var_methods(portfolio_returns, 0.95)
    
    print(f"\n{'Method':<20} {'Value (%)':>12} {'Best For':<35}")
    print("-" * 70)
    for _, row in var_comparison.iterrows():
        print(f"{row['Method']:<20} {row['Value (%)']:>11.4f}% {row['Best For']:<35}")

    # 2. Historical Stress Testing
    print("2. HISTORICAL STRESS TESTING")
    print("=" * 80)
    
    historical_results = historical_stress_test(
        portfolio_value, weights, betas,
        scenarios=['covid_crash_2020', 'financial_crisis_worst_2008', 'lehman_2008',
                   'black_monday_1987', 'dotcom_crash_2000', 'sept_11_2001']
    )
    
    print(f"\n{'Scenario':<30} {'Date':<12} {'Market':>10} {'Portfolio':>12} {'Loss ($)':>15}")
    print("-" * 80)
    for _, row in historical_results.iterrows():
        print(f"{row['Scenario']:<30} {row['Date']:<12} "
              f"{row['Market Return (%)']:>9.2f}% {row['Portfolio Return (%)']:>11.2f}% "
              f"${row['Portfolio Loss ($)']:>14,.0f}")
        
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
    print("4. REVERSE STRESS TESTING")
    print("=" * 80)
    
    target_losses = [50_000, 100_000, 150_000, 200_000]
    reverse_results = reverse_stress_test(portfolio_value, target_losses, weights, tickers)
    
    print(f"\n{'Target Loss':>15} {'As %':>10} {'Uniform Drop':>15} {'Largest Only':>15}")
    print("-" * 60)
    for _, row in reverse_results.iterrows():
        print(f"${row['Target Loss ($)']:>14,.0f} {row['Target Loss (%)']:>9.1f}% "
              f"{row['Uniform Decline (%)']:>14.2f}% "
              f"{row['Largest Position Only (%)']:>14.2f}%")
        
    # 5. Sensitivity Analysis
    print("5. SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    sensitivity_results = sensitivity_analysis(
        portfolio_value, weights, volatilities, correlation
    )
    
    print(f"\n{'Scenario':<40} {'VaR ($)':>12} {'vs Base':>12}")
    print("-" * 65)
    for _, row in sensitivity_results.iterrows():
        change_str = f"{row['Change vs Base (%)']:+.1f}%" if row['Change vs Base (%)'] != 0 else "—"
        print(f"{row['Scenario']:<40} ${row['VaR ($)']:>11,.0f} {change_str:>12}")

    # 6. Generate Visualization
    print("6. GENERATING VISUALIZATION")
    print("=" * 80)
    
    plot_stress_test_results(
        historical_results,
        hypothetical_results,
        sensitivity_results,
        var_comparison,
        save_path="15_stress_testing.png"
    )

    # Summary
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print(f"""
    1. VaR METHODS (imported from var.py):
       - Historical: {var_comparison.iloc[0]['Value (%)']:.2f}%
       - Parametric: {var_comparison.iloc[1]['Value (%)']:.2f}%
       - EWMA:       {var_comparison.iloc[2]['Value (%)']:.2f}%
       - ES:         {var_comparison.iloc[3]['Value (%)']:.2f}%
    
    2. WORST HISTORICAL SCENARIO:
       {historical_results.iloc[0]['Scenario']} → ${historical_results.iloc[0]['Portfolio Loss ($)']:,.0f} loss
    
    3. SECTOR CONCENTRATION:
       Tech (AAPL, MSFT, GOOGL) = {sum(weights[i] for i in get_sector_indices(tickers, 'tech')):.0%} of portfolio
    
    4. CRISIS CONDITIONS:
       Normal VaR: ${sensitivity_results.iloc[0]['VaR ($)']:,.0f}
       Severe Crisis VaR: ${sensitivity_results.iloc[-1]['VaR ($)']:,.0f}
       Increase: {sensitivity_results.iloc[-1]['Change vs Base (%)']:.0f}%
    """)
    
    print("Stress Testing Complete")
    


