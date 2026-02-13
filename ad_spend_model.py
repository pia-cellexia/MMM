"""
Ad-Spend Causal Response Model
===============================
Fits a causal response curve (daily conversions vs ad spend) with adstock and
seasonality controls, then uses the fitted model to find the optimal daily
budget where marginal ROAS ≈ a configurable target.

Usage
-----
    python ad_spend_model.py                    # uses defaults
    python ad_spend_model.py --data my.csv --aov 120 --target-mroas 4.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------

def load_data(path: str | Path) -> pd.DataFrame:
    """Read CSV or Excel file and add time-trend and day-of-week features."""
    path = Path(path)
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    # Report and drop rows with NaN in required columns
    required = ["date", "conversions", "cost", "ad_budget"]
    n_before = len(df)
    df = df.dropna(subset=required)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"Warning: dropped {n_dropped} rows with NaN in {required}")

    df = df.sort_values("date").reset_index(drop=True)

    # Time trend (integer index)
    df["t"] = np.arange(len(df))

    # Day-of-week
    df["dow"] = df["date"].dt.dayofweek  # 0=Mon … 6=Sun
    dow_dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=True, dtype=float)
    df = pd.concat([df, dow_dummies], axis=1)

    return df


# ---------------------------------------------------------------------------
# 2. Adstock
# ---------------------------------------------------------------------------

def build_adstock(series: pd.Series, lam: float) -> np.ndarray:
    """Geometric adstock transform: adstock_t = cost_t + λ * adstock_{t-1}."""
    values = series.values.astype(float)
    adstock = np.zeros(len(values))
    adstock[0] = values[0]
    for t in range(1, len(values)):
        adstock[t] = values[t] + lam * adstock[t - 1]
    return adstock


def steady_state_adstock(spend: float, lam: float) -> float:
    """Steady-state adstock for a constant daily spend level."""
    return spend / (1.0 - lam)


# ---------------------------------------------------------------------------
# 3. Fit response model
# ---------------------------------------------------------------------------

def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of feature column names."""
    dow_cols = sorted(c for c in df.columns if c.startswith("dow_"))
    return ["log_adstock", "ad_budget", "t"] + dow_cols


def fit_response_model(
    df: pd.DataFrame,
    lam: float,
    alpha: float = 1.0,
    validation_split: float = 0.2,
) -> tuple[Ridge, list[str], float, float, float]:
    """
    Fit Ridge regression: log(conversions+1) ~ log(adstock+1) + controls.

    Returns
    -------
    model : fitted Ridge
    feature_cols : ordered feature names
    val_rmse : validation RMSE (on held-out tail)
    train_r2 : R² on the training set
    val_r2 : R² on the validation set
    """
    df = df.copy()
    df["adstock"] = build_adstock(df["cost"], lam)
    df["log_adstock"] = np.log(df["adstock"] + 1.0)

    feature_cols = _feature_columns(df)
    X = df[feature_cols].values
    y = np.log(df["conversions"] + 1.0).values

    # Time-series split
    split = int(len(df) * (1.0 - validation_split))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    train_r2 = float(r2_score(y_train, y_train_pred))
    val_r2 = float(r2_score(y_val, y_val_pred))

    return model, feature_cols, val_rmse, train_r2, val_r2


def select_lambda(
    df: pd.DataFrame,
    candidates: list[float] | None = None,
    alpha: float = 1.0,
) -> tuple[float, Ridge, list[str]]:
    """Try several λ values and return the best by validation RMSE."""
    if candidates is None:
        candidates = [0.3, 0.4, 0.5, 0.6, 0.7]

    best_lam = candidates[0]
    best_rmse = float("inf")
    best_model = None
    best_cols: list[str] = []

    print("λ selection:")
    for lam in candidates:
        model, cols, rmse, tr_r2, val_r2 = fit_response_model(df, lam, alpha=alpha)
        tag = ""
        if rmse < best_rmse:
            best_lam, best_rmse, best_model, best_cols = lam, rmse, model, cols
            tag = " <-- best"
        print(f"  λ={lam:.2f}  val RMSE={rmse:.4f}  train R²={tr_r2:.4f}  val R²={val_r2:.4f}{tag}")

    # Refit on all data with the best λ
    df = df.copy()
    df["adstock"] = build_adstock(df["cost"], best_lam)
    df["log_adstock"] = np.log(df["adstock"] + 1.0)
    feature_cols = _feature_columns(df)
    X = df[feature_cols].values
    y = np.log(df["conversions"] + 1.0).values

    final_model = Ridge(alpha=alpha)
    final_model.fit(X, y)

    # In-sample R² for the final model (fitted on all data)
    y_pred_all = final_model.predict(X)
    full_r2 = float(r2_score(y, y_pred_all))

    print(f"Selected λ={best_lam:.2f} (val RMSE={best_rmse:.4f}), refitted on all data.")
    print(f"Final model in-sample R²={full_r2:.4f}\n")
    return best_lam, final_model, feature_cols


# ---------------------------------------------------------------------------
# 4. Budget ↔ spend mapping
# ---------------------------------------------------------------------------

def fit_budget_model(df: pd.DataFrame) -> tuple[float, float]:
    """Linear regression: cost = θ₀ + θ₁ · ad_budget. Returns (θ₀, θ₁)."""
    model = LinearRegression()
    model.fit(df[["ad_budget"]].values, df["cost"].values)
    return float(model.intercept_), float(model.coef_[0])


def spend_to_budget(desired_spend: float, theta0: float, theta1: float) -> float:
    """Invert the linear cost–budget relationship."""
    if theta1 == 0:
        return 0.0
    return max(0.0, (desired_spend - theta0) / theta1)


# ---------------------------------------------------------------------------
# 5. Prediction helpers
# ---------------------------------------------------------------------------

def _typical_dow_vector(df: pd.DataFrame) -> dict[str, float]:
    """DOW dummy values for the most frequent day in the data."""
    dow_cols = sorted(c for c in df.columns if c.startswith("dow_"))
    mode_dow = int(df["dow"].mode().iloc[0])
    col_name = f"dow_{mode_dow}"
    return {c: (1.0 if c == col_name else 0.0) for c in dow_cols}


def predict_conversions(
    spend: float,
    lam: float,
    model: Ridge,
    feature_cols: list[str],
    theta0: float,
    theta1: float,
    typical_dow: dict[str, float],
    typical_t: float,
) -> float:
    """Predict daily conversions for a constant daily spend (steady state)."""
    adstock_ss = steady_state_adstock(spend, lam)
    log_adstock = np.log(adstock_ss + 1.0)
    budget = spend_to_budget(spend, theta0, theta1)

    feat = {"log_adstock": log_adstock, "ad_budget": budget, "t": typical_t}
    feat.update(typical_dow)

    x_vec = np.array([feat[c] for c in feature_cols]).reshape(1, -1)
    log_y = model.predict(x_vec)[0]
    return max(0.0, float(np.exp(log_y) - 1.0))


# ---------------------------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------------------------

def metrics_for_spend(
    spend: float,
    aov: float,
    **predict_kwargs,
) -> dict[str, float]:
    """Compute conversions, revenue, ROAS, and profit for a spend level."""
    conv = predict_conversions(spend, **predict_kwargs)
    revenue = conv * aov
    roas = revenue / spend if spend > 0 else float("nan")
    profit = revenue - spend
    return {
        "spend": spend,
        "conversions": conv,
        "revenue": round(revenue, 2),
        "roas": round(roas, 4),
        "profit": round(profit, 2),
    }


def marginal_roas(
    spend: float,
    aov: float,
    delta: float = 10.0,
    **predict_kwargs,
) -> float:
    """Numerical approximation of dRevenue / dSpend."""
    m1 = metrics_for_spend(spend, aov, **predict_kwargs)
    m2 = metrics_for_spend(spend + delta, aov, **predict_kwargs)
    return (m2["revenue"] - m1["revenue"]) / delta


# ---------------------------------------------------------------------------
# 7. Optimal-budget search
# ---------------------------------------------------------------------------

def find_optimal_spend(
    df: pd.DataFrame,
    aov: float,
    target_mroas: float,
    lam: float,
    model: Ridge,
    feature_cols: list[str],
    theta0: float,
    theta1: float,
    grid_size: int = 200,
) -> pd.DataFrame:
    """
    Grid-search over spend levels and return a DataFrame with metrics.

    The row where marginal ROAS is closest to *target_mroas* is the optimum.
    """
    typical_dow = _typical_dow_vector(df)
    typical_t = float(df["t"].max())

    predict_kwargs = dict(
        lam=lam,
        model=model,
        feature_cols=feature_cols,
        theta0=theta0,
        theta1=theta1,
        typical_dow=typical_dow,
        typical_t=typical_t,
    )

    max_cost = df["cost"].max()
    spend_grid = np.linspace(1.0, 2.0 * max_cost, grid_size)

    rows = []
    for s in spend_grid:
        m = metrics_for_spend(s, aov, **predict_kwargs)
        m["marginal_roas"] = round(marginal_roas(s, aov, **predict_kwargs), 4)
        m["budget"] = round(spend_to_budget(s, theta0, theta1), 2)
        rows.append(m)

    res_df = pd.DataFrame(rows)
    res_df["mroas_diff"] = (res_df["marginal_roas"] - target_mroas).abs()
    return res_df


# ---------------------------------------------------------------------------
# 8. Main workflow
# ---------------------------------------------------------------------------

def run(
    data_path: str = "mmm_data.csv",
    aov: float = 116.80,
    target_mroas: float = 3.5,
    grid_size: int = 200,
    output_csv: str | None = "budget_response_curve.csv",
) -> dict:
    """End-to-end pipeline: load → fit → optimise → report."""

    # --- Load ---
    df = load_data(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}\n")

    # --- Select λ and fit model ---
    lam, model, feature_cols = select_lambda(df)

    # Print model coefficients for diagnostics
    print("Model coefficients:")
    for name, coef in zip(feature_cols, model.coef_):
        print(f"  {name:>15s}: {coef:+.4f}")
    print(f"  {'intercept':>15s}: {model.intercept_:+.4f}\n")

    # --- Budget ↔ spend mapping ---
    theta0, theta1 = fit_budget_model(df)
    print(f"Budget→spend mapping: cost ≈ {theta0:.2f} + {theta1:.4f} × ad_budget\n")

    # --- Prepare adstock on df for later use ---
    df["adstock"] = build_adstock(df["cost"], lam)
    df["log_adstock"] = np.log(df["adstock"] + 1.0)

    # --- Grid search ---
    res_df = find_optimal_spend(
        df, aov, target_mroas, lam, model, feature_cols, theta0, theta1,
        grid_size=grid_size,
    )

    optimal_row = res_df.loc[res_df["mroas_diff"].idxmin()]

    print("=" * 60)
    print("OPTIMAL BUDGET RECOMMENDATION")
    print("=" * 60)
    print(f"  Target marginal ROAS    : {target_mroas}")
    print(f"  AOV                     : ${aov:.2f}")
    print(f"  Adstock λ               : {lam}")
    print(f"  ---")
    print(f"  Optimal daily spend     : ${optimal_row['spend']:.2f}")
    print(f"  Suggested daily budget  : ${optimal_row['budget']:.2f}")
    print(f"  Expected conversions    : {optimal_row['conversions']:.2f}")
    print(f"  Expected revenue        : ${optimal_row['revenue']:.2f}")
    print(f"  Expected ROAS           : {optimal_row['roas']:.2f}")
    print(f"  Expected marginal ROAS  : {optimal_row['marginal_roas']:.4f}")
    print(f"  Expected daily profit   : ${optimal_row['profit']:.2f}")
    print("=" * 60)

    # --- Export ---
    if output_csv:
        res_df.drop(columns=["mroas_diff"]).to_csv(output_csv, index=False)
        print(f"\nResponse-curve data exported to {output_csv}")

    return {
        "lam": lam,
        "optimal_spend": float(optimal_row["spend"]),
        "optimal_budget": float(optimal_row["budget"]),
        "conversions": float(optimal_row["conversions"]),
        "revenue": float(optimal_row["revenue"]),
        "roas": float(optimal_row["roas"]),
        "marginal_roas": float(optimal_row["marginal_roas"]),
        "profit": float(optimal_row["profit"]),
        "model": model,
        "feature_cols": feature_cols,
        "theta0": theta0,
        "theta1": theta1,
        "response_curve": res_df.drop(columns=["mroas_diff"]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ad-spend causal response model & budget optimiser",
    )
    parser.add_argument("--data", default="mmm_data.csv", help="Path to input CSV")
    parser.add_argument("--aov", type=float, default=116.80, help="Average Order Value")
    parser.add_argument(
        "--target-mroas", type=float, default=3.5, help="Target marginal ROAS"
    )
    parser.add_argument("--grid-size", type=int, default=200, help="Spend grid resolution")
    parser.add_argument("--output", default="budget_response_curve.csv", help="Output CSV")
    args = parser.parse_args()

    run(
        data_path=args.data,
        aov=args.aov,
        target_mroas=args.target_mroas,
        grid_size=args.grid_size,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
