# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import yfinance as yf
from lightgbm import LGBMClassifier
from datetime import datetime
import os


# =========================
# 1. 銘柄
# =========================
def get_tickers():
    return pd.read_csv("tickers.csv")["Ticker"].tolist()


# =========================
# 2. データ取得
# =========================
def download_data(tickers, start="2018-01-01"):

    dfs = []

    for t in tickers:

        try:

            df = yf.download(
                t,
                start=start,
                progress=False,
                auto_adjust=False
            )

            if df.empty:
                continue

            df = df.reset_index()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[["Date", "Close"]].copy()

            df["Close"] = pd.to_numeric(
                df["Close"],
                errors="coerce"
            )

            df["Ticker"] = t

            dfs.append(df)

        except:
            continue

    return pd.concat(dfs, ignore_index=True)


# =========================
# 3. 特徴量
# =========================
def create_features(df):

    df = df.sort_values(
        ["Ticker", "Date"]
    ).copy()

    g = df.groupby("Ticker")["Close"]

    df["ret_1"] = g.pct_change(fill_method=None)

    df["ret_5"] = g.pct_change(
        5,
        fill_method=None
    )

    df["ma20"] = g.transform(
        lambda x: x.rolling(20).mean()
    )

    df["ma60"] = g.transform(
        lambda x: x.rolling(60).mean()
    )

    df["trend"] = (
        df["ma20"] > df["ma60"]
    ).astype(int)

    df["volatility"] = g.transform(
        lambda x:
        x.pct_change(fill_method=None)
        .rolling(20)
        .std()
    )

    # 1週間後リターン
    df["future_ret"] = (
        g.pct_change(
            5,
            fill_method=None
        )
        .shift(-5)
    )

    df["target"] = (
        df["future_ret"] > 0
    ).astype(int)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    return df


# =========================
# 4. 学習
# =========================
def train_model(df):

    features = [
        "ret_1",
        "ret_5",
        "trend",
        "volatility"
    ]

    train = df[
        df["Date"] < "2023-01-01"
    ]

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(
        train[features],
        train["target"]
    )

    return model, features


# =========================
# 5. EV
# =========================
def build_ev(latest, df):

    stats = (
        df.groupby("Ticker")["future_ret"]
        .agg(["mean", "std"])
        .rename(
            columns={
                "mean": "mu",
                "std": "sigma"
            }
        )
    )

    latest = latest.merge(
        stats,
        left_on="Ticker",
        right_index=True,
        how="left"
    )

    latest["mu"] = (
        latest["mu"]
        .fillna(0)
    )

    latest["sigma"] = (
        latest["sigma"]
        .fillna(
            latest["sigma"].median()
        )
    )

    latest["expected_value"] = (
        latest["model_prob"]
        * latest["mu"]
    )

    latest["score"] = (
        latest["expected_value"]
        /
        (latest["sigma"] + 1e-6)
    )

    return latest


# =========================
# 6. シグナル
# =========================
def generate_signal(
    df,
    model,
    features
):

    latest = (
        df.sort_values("Date")
        .groupby("Ticker")
        .tail(1)
        .copy()
    )

    latest["model_prob"] = (
        model.predict_proba(
            latest[features]
        )[:, 1]
    )

    latest = build_ev(
        latest,
        df
    )

    latest = latest[
        latest["model_prob"] > 0.50
    ]

    latest = latest.sort_values(
        "score",
        ascending=False
    )

    buy = latest.head(5)

    return latest, buy


# =========================
# 7. 履歴保存
# =========================
def save_history(buy):

    today = datetime.today().strftime("%Y-%m-%d")

    history = pd.DataFrame({
        "signal_date": today,
        "Ticker": buy["Ticker"],
        "entry_price": buy["Close"],
        "model_prob": buy["model_prob"],
        "expected_value": buy["expected_value"],
        "score": buy["score"]
    })

    file = "signal_history.csv"

    if os.path.exists(file):

        old = pd.read_csv(file)

        history = pd.concat(
            [old, history],
            ignore_index=True
        )

    history.to_csv(
        file,
        index=False
    )

    print("history rows =", len(history))
    print(history.head())

def evaluate_history():

    import os

    if not os.path.exists("signal_history.csv"):
        return

    hist = pd.read_csv("signal_history.csv")

    results = []

    for _, row in hist.iterrows():

        try:

            ticker = row["Ticker"]

            px = yf.download(
                ticker,
                start=row["signal_date"],
                progress=False,
                auto_adjust=False
            )

            if len(px) < 6:
                continue

            entry = float(row["entry_price"])
            exit_price = float(px["Close"].iloc[5])

            ret = (exit_price / entry) - 1

            results.append({
                "signal_date": row["signal_date"],
                "Ticker": ticker,
                "entry_price": entry,
                "exit_price": exit_price,
                "return": ret
            })

        except Exception as e:
            print(e)

    if len(results) == 0:
        return

    perf = pd.DataFrame(results)

    perf.to_csv(
        "performance.csv",
        index=False
    )

    print("performance rows =", len(perf))

def performance_report():

    import os

    if not os.path.exists("performance.csv"):
        return

    perf = pd.read_csv("performance.csv")

    if len(perf) == 0:
        return

    avg_ret = perf["return"].mean()

    win_rate = (
        perf["return"] > 0
    ).mean()

    cum_ret = (
        (1 + perf["return"])
        .prod() - 1
    )

    print("\n=========================")
    print("PERFORMANCE REPORT")
    print("=========================\n")

    print(
        f"平均リターン : {avg_ret:.2%}"
    )

    print(
        f"勝率 : {win_rate:.2%}"
    )

    print(
        f"累積リターン : {cum_ret:.2%}"
    )
# =========================
# 8. 実行
# =========================
df = download_data(
    get_tickers()
)

df = create_features(df)

model, features = train_model(df)

all_sig, buy = generate_signal(
    df,
    model,
    features
)

save_history(buy)

evaluate_history()

performance_report()

# =========================
# 9. 出力
# =========================
print("\n=========================")
print("WEEKLY SWING SIGNAL")
print("=========================\n")

cols = [
    "Ticker",
    "model_prob",
    "expected_value",
    "score",
    "Close"
]

print("【買い推奨 TOP】")
print(buy[cols])

print("\n=========================\n")

print("【全ランキング】")
print(all_sig[cols])
