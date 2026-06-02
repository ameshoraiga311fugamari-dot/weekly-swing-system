# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:42:00 2026

@author: amesh
"""

# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import yfinance as yf
from lightgbm import LGBMClassifier


# =========================
# 1. 銘柄
# =========================
def get_tickers():
    return [
        "7203.T","6758.T","9984.T","9432.T",
        "6861.T","8035.T","4063.T","8306.T"
    ]


# =========================
# 2. データ取得
# =========================
def download_data(tickers, start="2018-01-01"):

    dfs = []

    for ticker in tickers:

        df = yf.download(
            ticker,
            start=start,
            progress=False,
            auto_adjust=False
        )

        if df is None or df.empty:
            continue

        df = df.reset_index()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Close" not in df.columns:
            continue

        df = df[["Date", "Close"]].copy()

        df["Close"] = pd.to_numeric(
            df["Close"],
            errors="coerce"
        )

        df["Ticker"] = ticker

        dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("データ取得失敗")

    return pd.concat(dfs, ignore_index=True)


# =========================
# 3. 特徴量
# =========================
def create_features(df):

    df = df.sort_values(
        ["Ticker", "Date"]
    ).copy()

    g = df.groupby("Ticker")["Close"]

    df["ret_1"] = g.pct_change(
        fill_method=None
    )

    df["ret_5"] = g.pct_change(
        5,
        fill_method=None
    )

    df["ret_20"] = g.pct_change(
        20,
        fill_method=None
    )

    df["ma20"] = g.transform(
        lambda x: x.rolling(20).mean()
    )

    df["ma60"] = g.transform(
        lambda x: x.rolling(60).mean()
    )

    df["ma_ratio"] = (
        df["ma20"] / df["ma60"] - 1
    )

    df["trend"] = (
        df["ma20"] > df["ma60"]
    ).astype(int)

    df["vol20"] = g.transform(
        lambda x:
        x.pct_change(fill_method=None)
         .rolling(20)
         .std()
    )

    # 5営業日後リターン
    df["future_ret"] = (
        g.pct_change(
            5,
            fill_method=None
        )
        .shift(-5)
    )

    # 市場平均超えターゲット
    market_ret = df.groupby("Date")[
        "future_ret"
    ].transform("mean")

    df["target"] = (
        df["future_ret"] > market_ret
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
        "ret_20",
        "trend",
        "vol20",
        "ma_ratio"
    ]

    train = df[
        df["Date"] < "2023-01-01"
    ]

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42
    )

    model.fit(
        train[features],
        train["target"]
    )

    return model, features


# =========================
# 5. EV統計
# =========================
def build_ev_stats(df):

    rows = []

    for ticker, grp in df.groupby("Ticker"):

        x = grp["future_ret"]

        mu = x.mean()

        sigma = x.std()

        win_rate = (x > 0).mean()

        wins = x[x > 0]
        losses = x[x < 0]

        avg_win = (
            wins.mean()
            if len(wins) > 0
            else 0
        )

        avg_loss = (
            abs(losses.mean())
            if len(losses) > 0
            else 0
        )

        rows.append([
            ticker,
            mu,
            sigma,
            win_rate,
            avg_win,
            avg_loss
        ])

    stats = pd.DataFrame(
        rows,
        columns=[
            "Ticker",
            "mu",
            "sigma",
            "win_rate",
            "avg_win",
            "avg_loss"
        ]
    )

    return stats


# =========================
# 6. シグナル
# =========================
def generate_signal(
    df,
    model,
    features,
    stats
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

    latest = latest.merge(
        stats,
        on="Ticker",
        how="left"
    )

    latest["expected_value"] = (
        latest["model_prob"]
        * latest["avg_win"]
        -
        (1 - latest["model_prob"])
        * latest["avg_loss"]
    )

    latest["score"] = (
        latest["expected_value"]
        /
        (latest["sigma"] + 1e-6)
    )

    # ロング専用フィルタ
    latest = latest[
        (latest["model_prob"] > 0.55)
        &
        (latest["score"] > 0)
    ]

    latest = latest.sort_values(
        "score",
        ascending=False
    )

    buy = latest.head(5)

    return latest, buy


# =========================
# 7. 実行
# =========================
df = download_data(
    get_tickers()
)

df = create_features(df)

model, features = train_model(df)

stats = build_ev_stats(df)

all_sig, buy = generate_signal(
    df,
    model,
    features,
    stats
)


# =========================
# 8. 出力
# =========================
print("\n=========================")
print("WEEKLY SWING SIGNAL")
print("=========================\n")

print("【買い推奨 TOP】")

print(
    buy[
        [
            "Ticker",
            "model_prob",
            "expected_value",
            "score",
            "Close"
        ]
    ]
)

print("\n=========================\n")

print("【全ランキング】")

print(
    all_sig[
        [
            "Ticker",
            "model_prob",
            "expected_value",
            "score",
            "Close"
        ]
    ]
)