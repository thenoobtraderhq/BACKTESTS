"""
CODING HUMAN DISCRETION USING MACHINE LEARNING
Backtest v11: block-bootstrap robustness check on REAL data.

WHY BLOCK BOOTSTRAP
--------------------
Real historical data gives one realized path per pair -- not enough to make
a robustness claim by itself (a single Sharpe ratio could be a fluke of that
particular history). Block bootstrap resampling repeatedly draws contiguous
chunks ("blocks") of the REAL historical returns, with replacement, and
concatenates them into surrogate return series. This preserves real
volatility clustering and short-range autocorrelation (unlike a naive
single-period shuffle, which would destroy that structure) while giving many
independent-ish paths to test across -- the real-data analogue of the
synthetic Monte Carlo robustness check from v7.

For each pair (NASDAQ100, XAUUSD, EURUSD):
  - Daily returns -> bootstrap surrogate daily paths (block = 20 trading days)
    -> Trend (SMA50) and Mean-Reversion, Rigid/Human/ML
  - M15 returns -> bootstrap surrogate intraday paths (block = 480 bars,
    ~5 days) -> Opening Range Breakout, Rigid/Human/ML

Surrogate series use close-only price reconstruction (no real high/low),
consistent with how the synthetic-data Monte Carlo in v7 was built.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from data_loaders import (
    load_ndx_daily, load_ndx_m15, load_xau_daily, load_xau_m15,
    load_eur_m15, load_eur_daily_from_m15,
)

N_BOOT_DAILY = 10
N_BOOT_ORB = 5
DAILY_TARGET_LEN = 3000
DAILY_BLOCK = 20
ORB_TARGET_LEN = 40000
ORB_BLOCK = 480
BARS_PER_DAY = 96
STARTING_BALANCE = 10000.0
MAX_DRAWDOWN = 0.20
REDUCED_SIZE = 0.5
RECOVERY_BAND = 0.05


def block_bootstrap_returns(real_returns, target_length, block_size, seed):
    rng = np.random.default_rng(seed)
    n = len(real_returns)
    chunks = []
    total = 0
    while total < target_length:
        start = rng.integers(0, n - block_size)
        chunk = real_returns[start:start + block_size]
        chunks.append(chunk)
        total += block_size
    return np.concatenate(chunks)[:target_length]


def prices_from_returns(returns, start_price=1.0):
    prices = np.empty(len(returns) + 1)
    prices[0] = start_price
    prices[1:] = start_price * np.cumprod(1 + returns)
    return prices


# ---------- Daily feature/strategy functions ----------

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
    df["roll_high20"] = df["close"].rolling(20).max()
    df["roll_low20"] = df["close"].rolling(20).min()
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


# ---------- Intraday (synthetic-style day structure for bootstrap paths) ----------

def build_features_intraday(df):
    df = df.copy()
    df["day_idx"] = df.index // BARS_PER_DAY
    df["bar_in_day"] = df.index % BARS_PER_DAY
    df["roc4"] = df["close"].pct_change(4)
    df["roc8"] = df["close"].pct_change(8)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    df["volatility20"] = df["close"].pct_change().rolling(20).std()
    df["next_return"] = df["close"].shift(-1) / df["close"] - 1
    return df


INTRADAY_CONTEXT_FEATURES = ["roc4", "roc8", "rsi14", "volatility20", "bar_in_day"]
OPENING_BARS = 4


def signal_orb(df):
    close = df["close"].values
    day_idx = df["day_idx"].values
    bar_in_day = df["bar_in_day"].values
    n = len(df)
    signal = np.zeros(n, dtype=int)
    current_day, day_high, day_low = -1, None, None
    triggered_today, trig_dir = False, 0
    for t in range(n):
        if day_idx[t] != current_day:
            current_day = day_idx[t]
            triggered_today, trig_dir = False, 0
            day_high, day_low = -np.inf, np.inf
        if bar_in_day[t] < OPENING_BARS:
            day_high = max(day_high, close[t])
            day_low = min(day_low, close[t])
            signal[t] = 0
            continue
        if not triggered_today:
            if close[t] > day_high:
                triggered_today, trig_dir = True, 1
            elif close[t] < day_low:
                triggered_today, trig_dir = True, -1
        is_last_bar = bar_in_day[t] == BARS_PER_DAY - 1
        signal[t] = trig_dir if (triggered_today and not is_last_bar) else 0
    return signal


# ---------- Execution styles ----------

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


def sharpe_ratio(returns, periods_per_year):
    returns = pd.Series(returns).dropna()
    if len(returns) == 0 or returns.std() == 0:
        return np.nan
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    ann_return = (1 + returns.mean()) ** periods_per_year - 1
    return ann_return / ann_vol if ann_vol > 0 else np.nan


def final_balance_with_capprotect(position, returns, cooldown):
    equity, peak = STARTING_BALANCE, STARTING_BALANCE
    state, cooldown_remaining = "normal", 0
    for t in range(len(position)):
        size_mult = 1.0 if state == "normal" else (REDUCED_SIZE if state == "reduced" else 0.0)
        ret = position[t] * size_mult * returns[t]
        equity *= (1 + ret)
        peak = max(peak, equity)
        dd = equity / peak - 1
        if state == "normal" and dd <= -MAX_DRAWDOWN:
            state, cooldown_remaining = "paused", cooldown
        elif state == "paused":
            cooldown_remaining -= 1
            if cooldown_remaining <= 0:
                state = "reduced"
        elif state == "reduced":
            if dd >= -RECOVERY_BAND:
                state = "normal"
            elif dd <= -MAX_DRAWDOWN:
                state, cooldown_remaining = "paused", cooldown
    return equity


def run_one_daily_bootstrap(real_returns, seed):
    boot_returns = block_bootstrap_returns(real_returns, DAILY_TARGET_LEN, DAILY_BLOCK, seed)
    prices = prices_from_returns(boot_returns)
    df = pd.DataFrame({"close": prices})
    df = build_features_daily(df).dropna().reset_index(drop=True)
    n = len(df)
    folds = walk_forward_folds(n, n_folds=5, min_train=int(n * 0.3))
    eval_start = folds[0][2]
    idx = slice(eval_start, n)
    returns = df["next_return"].values

    out = {}
    for strat_name, sig_func in [("Trend (SMA50)", signal_trend_sma50), ("Mean-Reversion", signal_mean_reversion)]:
        sig = sig_func(df)
        rigid = rigid_positions(sig)
        human = human_discretion_positions(sig, returns, seed=seed * 1000 + 7)
        ml = coded_discretion_positions(
            df.assign(_sig=sig), "_sig", folds, DAILY_CONTEXT_FEATURES,
            lambda: RandomForestClassifier(n_estimators=150, max_depth=5, random_state=seed)
        )
        for style, pos in [("Rigid", rigid), ("Human", human), ("ML", ml)]:
            out[(strat_name, style, "sharpe")] = sharpe_ratio(pos[idx] * returns[idx], 252)
            out[(strat_name, style, "balance")] = final_balance_with_capprotect(pos[idx], returns[idx], 15)
    return out


def run_one_orb_bootstrap(real_returns, seed):
    boot_returns = block_bootstrap_returns(real_returns, ORB_TARGET_LEN, ORB_BLOCK, seed)
    prices = prices_from_returns(boot_returns)
    df = pd.DataFrame({"close": prices})
    df = build_features_intraday(df).dropna().reset_index(drop=True)
    n = len(df)
    folds = walk_forward_folds(n, n_folds=5, min_train=int(n * 0.25))
    eval_start = folds[0][2]
    idx = slice(eval_start, n)
    returns = df["next_return"].values

    sig = signal_orb(df)
    rigid = rigid_positions(sig)
    human = human_discretion_positions(sig, returns, seed=seed * 1000 + 11)
    ml = coded_discretion_positions(
        df.assign(_sig=sig), "_sig", folds, INTRADAY_CONTEXT_FEATURES,
        lambda: RandomForestClassifier(n_estimators=120, max_depth=5, random_state=seed)
    )
    out = {}
    for style, pos in [("Rigid", rigid), ("Human", human), ("ML", ml)]:
        out[("ORB (15min)", style, "sharpe")] = sharpe_ratio(pos[idx] * returns[idx], BARS_PER_DAY * 252)
        out[("ORB (15min)", style, "balance")] = final_balance_with_capprotect(pos[idx], returns[idx], 150)
    return out


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

    all_records = []

    for pair, loader in pairs_daily.items():
        print(f"\n=== {pair}: block-bootstrap daily ({N_BOOT_DAILY} resamples) ===")
        raw = loader()
        real_returns = raw["close"].pct_change().dropna().values
        for b in range(1, N_BOOT_DAILY + 1):
            res = run_one_daily_bootstrap(real_returns, seed=b)
            for (strat, style, metric), val in res.items():
                all_records.append({"Pair": pair, "Strategy": strat, "Execution": style,
                                     "Metric": metric, "Value": val, "Boot": b})
            print(f"  bootstrap {b}/{N_BOOT_DAILY} done")

    for pair, loader in pairs_m15.items():
        print(f"\n=== {pair}: block-bootstrap ORB ({N_BOOT_ORB} resamples) ===")
        raw = loader()
        real_returns = raw["close"].pct_change().dropna().values
        for b in range(1, N_BOOT_ORB + 1):
            res = run_one_orb_bootstrap(real_returns, seed=b + 500)
            for (strat, style, metric), val in res.items():
                all_records.append({"Pair": pair, "Strategy": strat, "Execution": style,
                                     "Metric": metric, "Value": val, "Boot": b})
            print(f"  bootstrap {b}/{N_BOOT_ORB} done")

    records_df = pd.DataFrame(all_records)
    records_df.to_csv("/home/claude/bootstrap_raw_records.csv", index=False)

    sharpe_df = records_df[records_df["Metric"] == "sharpe"]
    summary_rows = []
    winrate_rows = []
    for pair in ["NASDAQ100", "XAUUSD", "EURUSD"]:
        for strat in ["Trend (SMA50)", "Mean-Reversion", "ORB (15min)"]:
            sub = sharpe_df[(sharpe_df["Pair"] == pair) & (sharpe_df["Strategy"] == strat)]
            if sub.empty:
                continue
            piv = sub.pivot(index="Boot", columns="Execution", values="Value")
            for style in ["Rigid", "Human", "ML"]:
                vals = piv[style].dropna()
                summary_rows.append({
                    "Pair": pair, "Strategy": strat, "Execution": style,
                    "Mean Sharpe": round(vals.mean(), 2), "Std Sharpe": round(vals.std(), 2),
                    "N Bootstraps": len(vals),
                })
            ml_beat_human = (piv["ML"] > piv["Human"]).mean()
            human_beat_rigid = (piv["Human"] > piv["Rigid"]).mean()
            ml_beat_rigid = (piv["ML"] > piv["Rigid"]).mean()
            winrate_rows.append({
                "Pair": pair, "Strategy": strat,
                "ML beat Human": f"{ml_beat_human:.0%}",
                "ML beat Rigid": f"{ml_beat_rigid:.0%}",
                "Human beat Rigid": f"{human_beat_rigid:.0%}",
            })

    summary_df = pd.DataFrame(summary_rows)
    winrate_df = pd.DataFrame(winrate_rows)
    print("\n=== Block-bootstrap Sharpe summary ===")
    print(summary_df.to_string(index=False))
    print("\n=== Block-bootstrap win rates ===")
    print(winrate_df.to_string(index=False))
    summary_df.to_csv("/home/claude/bootstrap_sharpe_summary.csv", index=False)
    winrate_df.to_csv("/home/claude/bootstrap_winrates.csv", index=False)

    strategies = ["Trend (SMA50)", "Mean-Reversion", "ORB (15min)"]
    pairs = ["NASDAQ100", "XAUUSD", "EURUSD"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for row, pair in enumerate(pairs):
        for col, strat in enumerate(strategies):
            ax = axes[row][col]
            sub = sharpe_df[(sharpe_df["Pair"] == pair) & (sharpe_df["Strategy"] == strat)]
            if sub.empty:
                ax.set_visible(False)
                continue
            piv = sub.pivot(index="Boot", columns="Execution", values="Value")
            data = [piv[style].dropna().values for style in ["Rigid", "Human", "ML"]]
            ax.boxplot(data, labels=["Rigid", "Human", "ML"])
            ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
            ax.set_title(f"{pair} — {strat}", fontsize=9)
            ax.set_ylabel("Sharpe")
    plt.suptitle("Block-Bootstrap Robustness: Sharpe Ratio Distributions (Real Data)", fontsize=13)
    plt.tight_layout()
    plt.savefig("/home/claude/bootstrap_boxplots.png", dpi=150)
    print("\nSaved: bootstrap_raw_records.csv, bootstrap_sharpe_summary.csv, bootstrap_winrates.csv, bootstrap_boxplots.png")


if __name__ == "__main__":
    main()
