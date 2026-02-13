"""Generate a realistic sample mmm_data.csv for testing the ad-spend causal model."""

import numpy as np
import pandas as pd

np.random.seed(42)

n_days = 120
start_date = pd.Timestamp("2025-10-01")
dates = pd.date_range(start_date, periods=n_days, freq="D")

# Daily budget setting — fluctuates around 500 with some step changes
base_budget = 500.0
budget = np.full(n_days, base_budget)
budget[30:60] = 600.0   # budget increase period
budget[80:100] = 450.0  # budget decrease period
budget += np.random.normal(0, 10, n_days)
budget = np.clip(budget, 200, 800).round(2)

# Actual spend is a noisy fraction of budget (Google doesn't always spend it all)
spend_frac = np.random.beta(8, 2, n_days)  # mean ~0.8
cost = (budget * spend_frac).round(2)

# True response: conversions = f(adstock) with diminishing returns + DOW + trend
lam_true = 0.5
adstock = np.zeros(n_days)
adstock[0] = cost[0]
for t in range(1, n_days):
    adstock[t] = cost[t] + lam_true * adstock[t - 1]

dow = dates.dayofweek
dow_effect = np.array([-0.05, 0.0, 0.02, 0.03, 0.08, 0.15, 0.10])
trend = np.linspace(0, 0.15, n_days)

log_conv = (
    0.3
    + 0.45 * np.log(adstock + 1)
    + np.array([dow_effect[d] for d in dow])
    + trend
    + np.random.normal(0, 0.08, n_days)
)
conversions = np.exp(log_conv) - 1.0
conversions = np.clip(conversions, 0, None).round(2)

# Revenue & ROAS (Google-reported, slightly noisy)
aov = 116.80
revenue = conversions * aov * np.random.normal(1.0, 0.03, n_days)
roas = (revenue / cost).round(4)

df = pd.DataFrame({
    "date": dates.strftime("%Y-%m-%d"),
    "conversions": conversions,
    "cost": cost,
    "ad_budget": budget.round(2),
    "roas": roas,
})

df.to_csv("mmm_data.csv", index=False)
print(f"Wrote {len(df)} rows to mmm_data.csv")
print(df.describe().round(2))
