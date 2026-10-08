"""
CODING HUMAN DISCRETION USING MACHINE LEARNING
Backtest v10: real market data.

Runs the same three strategies (Trend SMA50, Mean-Reversion, Opening Range
Breakout) x three execution styles (Rigid / Human Discretion / Coded
Discretion ML) x $10,000 portfolio with capital preservation, on REAL
historical data for three pairs: NASDAQ100, XAUUSD, EURUSD.

Trend and Mean-Reversion run on daily bars (their original swing-trading
horizon). ORB runs on 15-minute bars, opening range anchored to 09:30 in
each file's own recorded time (see note on timezone assumption).

This is a single real historical path per pair -- not a Monte Carlo
ensemble like the synthetic-data runs. Three independent real markets
(NDX, XAU, EUR) provide real-world cross-market variation instead.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from data_loaders import (
    load_ndx_daily, load_ndx_m15, load_xau_daily, load_xau_m15,
    load_eur_m15, load_eur_daily_from_m15,
)

STARTING_BALANCE = 10000.0
MAX_DRAWDOWN = 0.20
REDUCED_SIZE = 0.5
RECOVERY_BAND = 0.05
SESSION_START = pd.Timestamp("09:30:00").time()
OPENING_BARS = 4


# ---------- Feature engineering (daily) ----------

def build_features_daily(df):
    df = df.copy()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["roc5"] = df["close"].pct_change(5)
    df["roc10"] = df["close"].pct_change(10)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    df["roll_high20"] = df["high"].rolling(20).max()
    df["roll_low20"] = df["low"].rolling(20).min()
    df["dist_resistance"] = (df["roll_high20"] - df["close"]) / df["close"]
    df["dist_support"] = (df["close"] - df["roll_low20"]) / df["close"]
    df["volatility20"] = df["close"].pct_change().rolling(20).std()
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["zscore20"] = (df["close"] - bb_mid) / bb_std
    df["next_return"] = df["close"].shift(-1) / df["close"] - 1
    return df


DAILY_CONTEXT_FEATURES = ["roc5", "roc10", "rsi14", "dist_resistance", "dist_support", "volatility20"]


def signal_trend_sma50(df):
    return np.where(df["close"] > df["sma50"], 1, -1)


def signal_mean_reversion(df, entry_z=1.3):
    z = df["zscore20"]
    sig = np.zeros(len(df), dtype=int)
    sig[z > entry_z] = -1
    sig[z < -entry_z] = 1
    return sig


# ---------- Feature engineering (intraday / ORB) ----------

def build_features_intraday(df):
    df = df.copy()
    df["roc4"] = df["close"].pct_change(4)
    df["roc8"] = df["close"].pct_change(8)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    df["volatility20"] = df["close"].pct_change().rolling(20).std()
    df["bar_time"] = df["datetime"].dt.time
    df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df["next_return"] = df["close"].shift(-1) / df["close"] - 1
    return df


INTRADAY_CONTEXT_FEATURES = ["roc4", "roc8", "rsi14", "volatility20", "minute_of_day"]


def signal_orb_real(df, session_start=SESSION_START, opening_bars=OPENING_BARS):
    dates = df["datetime"].dt.date.values
    times = df["datetime"].dt.time.values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    signal = np.zeros(n, dtype=int)

    day_groups = pd.Series(np.arange(n)).groupby(dates).apply(list)
    for d, idxs in day_groups.items():
        session_positions = [i for i in idxs if times[i] >= session_start]
        if len(session_positions) < opening_bars + 1:
            continue
        range_idxs = session_positions[:opening_bars]
        day_high = high[range_idxs].max()
        day_low = low[range_idxs].min()
        trade_idxs = session_positions[opening_bars:]
        triggered, trig_dir = False, 0
        last_idx_of_day = idxs[-1]
        for i in trade_idxs:
            if i == last_idx_of_day:
                signal[i] = 0
                break
            if not triggered:
                if close[i] > day_high:
                    triggered, trig_dir = True, 1
                elif close[i] < day_low:
                    triggered, trig_dir = True, -1
            signal[i] = trig_dir if triggered else 0
    return signal


# ---------- Execution styles (same mechanics as v9) ----------

def rigid_positions(signal):
    return signal.copy()


def human_discretion_positions(signal, returns, seed,
                                p_skip=0.20, p_hesitate=0.35, entry_lag=2,
                                early_exit_profit=0.006, p_early_exit=0.15,
                                p_hold_loser=0.55, max_hold_extra=6):
    local_rng = np.random.default_rng(seed)
    n = len(signal)
    position = np.zeros(n, dtype=float)
    current_pos, entry_ret, lag_remaining, pending, hold_extra = 0.0, 0.0, 0, 0.0, 0
    for t in range(n):
        sig = signal[t]
        if lag_remaining > 0:
            lag_remaining -= 1
            if lag_remaining == 0:
                current_pos, pending, entry_ret = pending, 0.0, 0.0
            position[t] = 0.0
            continue
        if current_pos == 0 and sig != 0:
            if local_rng.random() < p_skip:
                position[t] = 0.0
                continue
            elif local_rng.random() < p_hesitate:
                pending, lag_remaining = sig, entry_lag
                position[t] = 0.0
                continue
            else:
                current_pos, entry_ret = sig, 0.0
        if current_pos != 0:
            entry_ret = (1 + entry_ret) * (1 + current_pos * returns[t]) - 1
            if sig != current_pos:
                if entry_ret > 0:
                    current_pos, entry_ret = sig, 0.0
                else:
                    if hold_extra == 0 and local_rng.random() < p_hold_loser:
                        hold_extra = int(local_rng.integers(1, max_hold_extra))
                    if hold_extra > 0:
                        hold_extra -= 1
                    else:
                        current_pos, entry_ret = sig, 0.0
            else:
                if entry_ret > early_exit_profit and local_rng.random() < p_early_exit:
                    current_pos, entry_ret = 0.0, 0.0
        position[t] = current_pos
    return position


def coded_discretion_positions(df, signal_col, folds, context_features, model_factory):
    positions = np.zeros(len(df))
    for tr_start, tr_end, te_start, te_end in folds:
        train = df.iloc[tr_start:tr_end]
        test = df.iloc[te_start:te_end]
        train_active = train[train[signal_col] != 0].copy()
        if len(train_active) < 30:
            continue
        train_active["filter_target"] = (train_active[signal_col] * train_active["next_return"] > 0).astype(int)
        model = model_factory()
        model.fit(train_active[context_features], train_active["filter_target"])
        test_active_idx = test.index[test[signal_col] != 0]
        if len(test_active_idx) > 0:
            preds = model.predict(df.loc[test_active_idx, context_features])
            take = preds == 1
            positions[test_active_idx[take]] = df.loc[test_active_idx[take], signal_col].values
    return positions


def walk_forward_folds(n, n_folds, min_train):
    test_span = (n - min_train) // n_folds
    folds = []
    for k in range(n_folds):
        train_end = min_train + k * test_span
        test_end = train_end + test_span
        if test_end > n:
            break
        folds.append((0, train_end, train_end, test_end))
    return folds


def equity_path_with_capital_preservation(position, returns, cooldown_periods,
                                           starting_balance=STARTING_BALANCE,
                                           max_dd=MAX_DRAWDOWN, reduced_size=REDUCED_SIZE,
                                           recovery_band=RECOVERY_BAND):
    n = len(position)
    equity, peak = starting_balance, starting_balance
    path = np.empty(n)
    state, cooldown_remaining = "normal", 0
    for t in range(n):
        size_mult = 1.0 if state == "normal" else (reduced_size if state == "reduced" else 0.0)
        ret = position[t] * size_mult * returns[t]
        equity = equity * (1 + ret)
        peak = max(peak, equity)
        dd = equity / peak - 1
        path[t] = equity
        if state == "normal":
            if dd <= -max_dd:
                state, cooldown_remaining = "paused", cooldown_periods
        elif state == "paused":
            cooldown_remaining -= 1
            if cooldown_remaining <= 0:
                state = "reduced"
        elif state == "reduced":
            if dd >= -recovery_band:
                state = "normal"
            elif dd <= -max_dd:
                state, cooldown_remaining = "paused", cooldown_periods
    return path


def sharpe_ratio(returns, periods_per_year):
    returns = pd.Series(returns).dropna()
    if len(returns) == 0 or returns.std() == 0:
        return np.nan
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    ann_return = (1 + returns.mean()) ** periods_per_year - 1
    return ann_return / ann_vol if ann_vol > 0 else np.nan


def run_daily_strategy(df, sig_func, strat_name, cooldown=15, n_folds=5, min_train_frac=0.25):
    sig = sig_func(df)
    returns = df["next_return"].values
    n = len(df)
    folds = walk_forward_folds(n, n_folds=n_folds, min_train=int(n * min_train_frac))
    eval_start = folds[0][2]
    idx = slice(eval_start, n)

    rigid = rigid_positions(sig)
    human = human_discretion_positions(sig, returns, seed=7)
    ml = coded_discretion_positions(
        df.assign(_sig=sig), "_sig", folds, DAILY_CONTEXT_FEATURES,
        lambda: RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    )

    results = {}
    for style, pos in [("Rigid", rigid), ("Human", human), ("ML", ml)]:
        pos_eval, ret_eval = pos[idx], returns[idx]
        path = equity_path_with_capital_preservation(pos_eval, ret_eval, cooldown)
        results[style] = {
            "path": path,
            "sharpe": sharpe_ratio(pos_eval * ret_eval, 252),
            "final_balance": path[-1],
        }
    return results


def run_orb_strategy(df, cooldown=150, n_folds=5, min_train_frac=0.2):
    sig = signal_orb_real(df)
    returns = df["next_return"].values
    n = len(df)
    folds = walk_forward_folds(n, n_folds=n_folds, min_train=int(n * min_train_frac))
    eval_start = folds[0][2]
    idx = slice(eval_start, n)

    rigid = rigid_positions(sig)
    human = human_discretion_positions(sig, returns, seed=11)
    ml = coded_discretion_positions(
        df.assign(_sig=sig), "_sig", folds, INTRADAY_CONTEXT_FEATURES,
        lambda: RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)
    )

    results = {}
    for style, pos in [("Rigid", rigid), ("Human", human), ("ML", ml)]:
        pos_eval, ret_eval = pos[idx], returns[idx]
        path = equity_path_with_capital_preservation(pos_eval, ret_eval, cooldown)
        results[style] = {
            "path": path,
            "sharpe": sharpe_ratio(pos_eval * ret_eval, 96 * 252),
            "final_balance": path[-1],
        }
    return results


def main():
    pairs_daily = {
        "NASDAQ100": load_ndx_daily,
        "XAUUSD": load_xau_daily,
        "EURUSD": load_eur_daily_from_m15,
    }
    pairs_m15 = {
        "NASDAQ100": load_ndx_m15,
        "XAUUSD": load_xau_m15,
        "EURUSD": load_eur_m15,
    }

    all_results = {}   # {(pair, strategy): {style: {...}}}

    for pair, loader in pairs_daily.items():
        print(f"\n=== {pair}: daily strategies ===")
        raw = loader()
        df = build_features_daily(raw).dropna().reset_index(drop=True)
        for strat_name, sig_func in [("Trend (SMA50)", signal_trend_sma50),
                                      ("Mean-Reversion", signal_mean_reversion)]:
            print(f"  running {strat_name} on {len(df)} daily bars...")
            res = run_daily_strategy(df, sig_func, strat_name)
            all_results[(pair, strat_name)] = res
            print(f"    Rigid Sharpe={res['Rigid']['sharpe']:.2f}  "
                  f"Human Sharpe={res['Human']['sharpe']:.2f}  ML Sharpe={res['ML']['sharpe']:.2f}")

    for pair, loader in pairs_m15.items():
        print(f"\n=== {pair}: ORB (15-min) ===")
        raw = loader()
        df = build_features_intraday(raw).dropna().reset_index(drop=True)
        print(f"  running ORB on {len(df)} 15-min bars...")
        res = run_orb_strategy(df)
        all_results[(pair, "ORB (15min)")] = res
        print(f"    Rigid Sharpe={res['Rigid']['sharpe']:.2f}  "
              f"Human Sharpe={res['Human']['sharpe']:.2f}  ML Sharpe={res['ML']['sharpe']:.2f}")

    # Summary table
    rows = []
    for (pair, strat), res in all_results.items():
        for style in ["Rigid", "Human", "ML"]:
            rows.append({
                "Pair": pair, "Strategy": strat, "Execution": style,
                "Sharpe": round(res[style]["sharpe"], 2),
                "Final Balance ($10k start)": f"${res[style]['final_balance']:,.0f}",
            })
    summary_df = pd.DataFrame(rows)
    print("\n\n=== FULL REAL-DATA SUMMARY ===")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("/home/claude/real_data_summary.csv", index=False)

    # Plot: 3 pairs x 3 strategies grid
    strategies = ["Trend (SMA50)", "Mean-Reversion", "ORB (15min)"]
    pairs = ["NASDAQ100", "XAUUSD", "EURUSD"]
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for row, pair in enumerate(pairs):
        for col, strat in enumerate(strategies):
            ax = axes[row][col]
            res = all_results[(pair, strat)]
            ax.plot(res["Rigid"]["path"], color="black", linestyle="--", linewidth=1.2, label="Rigid")
            ax.plot(res["Human"]["path"], color="firebrick", linewidth=1.5, label="Human")
            ax.plot(res["ML"]["path"], color="steelblue", linewidth=1.5, label="ML")
            ax.axhline(STARTING_BALANCE, color="gray", linewidth=0.7, linestyle=":")
            ax.set_title(f"{pair} — {strat}", fontsize=10)
            ax.set_ylabel("Portfolio ($)")
            if row == 2:
                ax.set_xlabel("Out-of-sample period")
            ax.legend(fontsize=7)
    plt.suptitle("Real Data: $10,000 Portfolios with Capital Preservation (20% max drawdown)", fontsize=13)
    plt.tight_layout()
    plt.savefig("/home/claude/real_data_portfolio_grid.png", dpi=150)
    print("\nSaved: real_data_summary.csv, real_data_portfolio_grid.png")


if __name__ == "__main__":
    main()
