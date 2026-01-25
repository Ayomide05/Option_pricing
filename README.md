# Option Pricing and Market Risk Analytics

A comprehensive implementation of option pricing models and market risk analytics in Python. This project covers the theoretical foundations and practical implementation of derivatives pricing, from the Black-Scholes model to Value-at-Risk calculations.

## Overview

This project is structured in phases, each building on the previous:

| Phase | Topic | Status |
|-------|-------|--------|
| 1 | Black-Scholes Model & Greeks | ✅ Complete |
| 2 | Implied Volatility & Market Data | ✅ Complete |
| 3 | Monte Carlo Pricing | 🔄 In Progress |
| 4 | Value-at-Risk (VaR) & Expected Shortfall | ⏳ Planned |

---

## Phase 1: Black-Scholes Model & Greeks

### Mathematical Foundation

The Black-Scholes model prices European options by solving a partial differential equation derived from no-arbitrage principles. The key insight is that an option can be perfectly hedged by continuously adjusting a position in the underlying stock, eliminating the need to estimate expected returns.

**The Black-Scholes PDE:**

```
∂V/∂t + rS(∂V/∂S) + ½σ²S²(∂²V/∂S²) = rV
```

**Solution for a European Call:**

```
C = S·N(d₁) - Ke^(-r(T-t))·N(d₂)
```

where:

```
d₁ = [ln(S/K) + (r + ½σ²)(T-t)] / [σ√(T-t)]
d₂ = d₁ - σ√(T-t)
```

### The Greeks

The Greeks measure option price sensitivity to various parameters:

| Greek | Symbol | Measures Sensitivity To | Formula (Call) |
|-------|--------|------------------------|----------------|
| Delta | Δ | Stock price | N(d₁) |
| Gamma | Γ | Stock price (2nd order) | n(d₁) / [Sσ√T] |
| Theta | Θ | Time decay | -[S·n(d₁)·σ] / [2√T] - rKe^(-rT)N(d₂) |
| Vega | ν | Volatility | S√T·n(d₁) |
| Rho | ρ | Interest rate | KTe^(-rT)N(d₂) |

### Implementation

- `black_scholes.py` — Core pricing functions for European calls and puts
- `greeks.py` — Analytical Greek calculations
- `visualizations.py` — Price and sensitivity visualizations

### Sample Output

```
Black-Scholes Option Pricing
========================================
Stock Price (S):     $100.00
Strike Price (K):    $100.00
Time to Expiry (T):  1 year(s)
Risk-free Rate (r):  5.0%
Volatility (σ):      20.0%
----------------------------------------
Call Price:          $10.4506
Put Price:           $5.5735
```

### Visualizations

#### Option Price vs Stock Price
![Option Payoff and Price](phase1_black_scholes/images/01_option_payoff_price.png)

#### Greeks vs Stock Price
![Greeks vs Stock Price](phase1_black_scholes/images/02_greeks_vs_stock.png)

#### Gamma Surface (The "Gamma Explosion")
![Gamma Surface](phase1_black_scholes/images/05_gamma_surface.png)

---

## Phase 2: Implied Volatility & Market Data

### What is Implied Volatility?

Implied volatility (IV) is the volatility that makes the Black-Scholes price match the observed market price. It represents the market's expectation of future volatility.

**The Problem:** Given market price, solve for σ in:

```
C_market = BS(S, K, T, r, σ)
```

There's no closed-form solution — we use numerical methods.

### Newton-Raphson Method

We use Newton-Raphson iteration with vega as the derivative:

```
σ_{n+1} = σ_n - (BS(σ_n) - C_market) / vega(σ_n)
```

Converges in 2-4 iterations typically.

### The Volatility Smile

Black-Scholes assumes constant volatility across all strikes. Reality disagrees.

**What the smile tells us:**
- **OTM puts are expensive** — Investors pay premium for crash protection
- **Negative skew** — Left side (puts) higher than right side (calls)
- **Fat tails** — Market expects more extreme moves than log-normal distribution

### Implementation

- `implied_volatility.py` — Newton-Raphson IV solver with convergence handling
- `market_data.py` — Real market data fetching and analysis using Yahoo Finance

### Real Market Data Analysis

The project fetches live option data and Treasury rates:

```
VOLATILITY SMILE ANALYSIS: SPY
============================================================
Risk-free rate (3m): 3.58% (from Treasury yield)

Market Data:
  Spot Price: $689.23
  Expiry: 2026-01-27
  Days to Expiry: 1

Quadratic fit: IV = 15.6% + -111.7%·x + 1848.0%·x²
  ATM vol: 15.63%
  Skew: -1.1170 (negative = crash protection premium)
  Smile: 18.4796 (curvature)
```

### Visualizations

#### Real Market Volatility Smile (SPY)
![Market Smile](phase2_implied_volatility/images/09_market_smile.png)

The classic equity skew: OTM puts (left) trade at much higher IV than OTM calls (right). This is the price of crash protection.

#### Market Data vs Quadratic Model Fit
![Model Fit](phase2_implied_volatility/images/10_market_vs_model.png)

A simple quadratic model captures the general shape of the smile, though market microstructure creates scatter.

#### Smile Term Structure
![Term Structure](phase2_implied_volatility/images/11_smile_term_structure.png)

How the smile shape changes across expiration dates. Shorter expiries show more pronounced effects.

---

## Phase 3: Monte Carlo Pricing

*Coming soon*

- Geometric Brownian Motion simulation
- European option pricing via Monte Carlo
- Path-dependent options (Asian, Barrier)
- Variance reduction techniques

---

## Phase 4: Value-at-Risk (VaR)

*Coming soon*

- Historical simulation VaR
- Parametric VaR (Delta-Normal, Delta-Gamma)
- Monte Carlo VaR
- Expected Shortfall (CVaR)
- Backtesting and model validation

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/option_pricing.git
cd option_pricing

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- yfinance (for real market data)
- pandas

## Project Structure

```
option_pricing/
├── README.md
├── requirements.txt
├── phase1_black_scholes/
│   ├── black_scholes.py
│   ├── greeks.py
│   ├── visualizations.py
│   └── images/
├── phase2_implied_volatility/
│   ├── implied_volatility.py
│   ├── market_data.py
│   └── images/
├── phase3_monte_carlo/
└── phase4_var/
```

## Key Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Derivatives Pricing | Black-Scholes analytical solution |
| Risk Sensitivities | Greeks (Delta, Gamma, Vega, Theta, Rho) |
| Numerical Methods | Newton-Raphson root finding |
| Market Data Integration | Yahoo Finance API, Treasury yields |
| Volatility Analysis | Smile, skew, term structure |

## References

- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637-654.
- Hull, J. C. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
- Wilmott, P. (2006). *Paul Wilmott on Quantitative Finance* (2nd ed.). Wiley.

## License

MIT License
## Author

Gabriel Justina Ayomide

---

*This project is part of my journey into quantitative finance and market risk. Feedback and contributions are welcome.*
