"""
article_level_windowed.py (formerly data_loader_88_article.py)

Combines article/row-level sentiment (no daily aggregation) with the
20:00 UTC market-close windowing used in daily_avg_windowed.py.

See data_loaders/README.md for how this variant relates to the other five.
"""
import config as cfg
import os
import random
import pandas as pd
import datetime
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------
# Helper functions: sequences, datasets
# ------------------------------------------------------------------------
def make_sequences(df, lookback, cols):
    arr = df[cols].values.astype("float32")
    X, y = [], []
    for i in range(len(arr) - lookback):
        X.append(arr[i : i + lookback])
        y.append(arr[i + lookback, 0])
    return np.array(X), np.array(y)

def _assign_window_date(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Map UTC timestamp into the 'sentiment‐day' whose close at 20:00 UTC
    we want to predict:
      - Mon–Thu before 20:00 → same calendar day
      - Mon–Thu from 20:00 → next day
      - Fri before 20:00 → Fri
      - Fri from 20:00 → next Monday
      - Sat/Sun → next Monday
    """
    base = ts.normalize()
    wd   = base.weekday()  # Mon=0 … Fri=4, Sat=5, Sun=6
    cutoff = datetime.time(20, 0)

    if wd <= 3:  # Mon–Thu
        return base     if ts.time() < cutoff else base + pd.Timedelta(days=1)
    if wd == 4:  # Fri
        return base     if ts.time() < cutoff else base + pd.Timedelta(days=3)
    # Sat/Sun → next Monday
    return base + pd.Timedelta(days=(7 - wd))

def ds_xy(X, y, batch_size, shuffle=True, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(y), seed=seed)
    return ds.batch(min(batch_size, len(y))).prefetch(tf.data.AUTOTUNE)

def ds_multi(X1, X2, y, batch_size, shuffle=True, seed=42):
    ds = tf.data.Dataset.from_tensor_slices(((X1, X2), y))
    if shuffle:
        ds = ds.shuffle(len(y), seed=seed)
    return ds.batch(min(batch_size, len(y))).prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------------------------------
# loading & prep functions
# ------------------------------------------------------------------------
def seed_everything(seed=42):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def load_spy():
    print("  → load_spy()")
    spy = pd.read_csv(
        "data/spy_2012_2025.csv",
        skiprows=[1,2],
        parse_dates=["Price"],
        index_col="Price",
        usecols=["Price","price"]
    )
    spy.index.name = "date"
    spy.columns = ["price"]
    spy = spy.asfreq("B").ffill()
    return spy.loc[:"2020-07-29"], spy.loc["2020-07-30":]

def load_sentiment():
    print("  → load_sentiment() [20→20 UTC, Fri→Mon, article-level]")
    tmp = pd.read_csv(
        "data/3D_articles_new_version_hm.csv",
        usecols=["date","mean_positive","mean_neutral","mean_negative"],
        parse_dates=["date"]
    )
    # assign each article into its 20→20 window
    tmp["window_date"] = tmp["date"].apply(_assign_window_date)
    tmp = tmp.rename(columns={
        "mean_positive":"sent_pos",
        "mean_neutral":"sent_neu",
        "mean_negative":"sent_neg"
    })
    # normalize each article so pos+neu+neg = 1
    sums = tmp[["sent_pos","sent_neu","sent_neg"]].sum(axis=1).replace(0,1)
    tmp[["sent_pos","sent_neu","sent_neg"]] = tmp[["sent_pos","sent_neu","sent_neg"]].div(sums, axis=0)
    # index by window_date (so joining to SPY yields one row per article)
    out = tmp.set_index("window_date")[["sent_pos","sent_neu","sent_neg"]]
    out.index.name = "date"
    return out.sort_index()

def load_mag():
    print("  → load_mag() [20→20 UTC, Fri→Mon, article-level]")
    tmp = pd.read_csv(
        "data/3D_articles_Mag_7_hm.csv",
        usecols=["date","mean_positive","mean_neutral","mean_negative"],
        parse_dates=["date"]
    )
    tmp["window_date"] = tmp["date"].apply(_assign_window_date)
    tmp = tmp.rename(columns={
        "mean_positive":"mag_pos",
        "mean_neutral":"mag_neu",
        "mean_negative":"mag_neg"
    })
    sums = tmp[["mag_pos","mag_neu","mag_neg"]].sum(axis=1).replace(0,1)
    tmp[["mag_pos","mag_neu","mag_neg"]] = tmp[["mag_pos","mag_neu","mag_neg"]].div(sums, axis=0)
    out = tmp.set_index("window_date")[["mag_pos","mag_neu","mag_neg"]]
    out.index.name = "date"
    return out.sort_index().loc[:"2025-04-30"]

def load_tariff():
    print("  → load_tariff() [20→20 UTC, Fri→Mon, article-level]")
    tmp = pd.read_csv(
        "data/3D_articles_Tariff_hm.csv",
        usecols=["date","mean_positive","mean_neutral","mean_negative"],
        parse_dates=["date"]
    )
    tmp["window_date"] = tmp["date"].apply(_assign_window_date)
    tmp = tmp.rename(columns={
        "mean_positive":"tariff_pos",
        "mean_neutral":"tariff_neu",
        "mean_negative":"tariff_neg"
    })
    sums = tmp[["tariff_pos","tariff_neu","tariff_neg"]].sum(axis=1).replace(0,1)
    tmp[["tariff_pos","tariff_neu","tariff_neg"]] = tmp[["tariff_pos","tariff_neu","tariff_neg"]].div(sums, axis=0)
    out = tmp.set_index("window_date")[["tariff_pos","tariff_neu","tariff_neg"]]
    out.index.name = "date"
    return out.sort_index().loc[:"2025-04-30"]

def load_kalshi():
    print("  → load_kalshi() (optimized)")
    df = pd.read_csv(
        "data/KXINX_trades_range_vs_sp.csv",
        usecols=["trade_date","bucket_start","bucket_end","yes_price"],
        parse_dates=["trade_date"]
    )
    df_valid = df.dropna(subset=["bucket_start","bucket_end"])
    df_valid = (df_valid
                .sort_values(["trade_date","yes_price"], ascending=[True, False])
                .drop_duplicates("trade_date"))
    missing = set(df["trade_date"]) - set(df_valid["trade_date"])
    df_fallback = df[df["trade_date"].isin(missing)]
    df_fallback = (df_fallback
                   .sort_values(["trade_date","yes_price"], ascending=[True, False])
                   .drop_duplicates("trade_date"))
    best = pd.concat([df_valid, df_fallback]).sort_values("trade_date")
    best["avg_bucket"] = (best.bucket_start + best.bucket_end) / 2
    kalshi = (best.rename(columns={"trade_date":"date","avg_bucket":"kalshi_price"})
                [["date","kalshi_price"]]
                .set_index("date")
                .sort_index())
    return kalshi

# ------------------------------------------------------------------------
# Master loader: assemble pipelines + test data
# ------------------------------------------------------------------------
def get_datasets(
    cfg,
    use_sentiment=True,
    use_mag=False,
    use_tariff=False,
    use_kalshi=False
):
    print("→ get_datasets: start")
    seed_everything(cfg.get("seed",42))

    # 1) SPY prices
    spy_pre, spy_post = load_spy()
    df = spy_post.copy()

    # 2) optionally join each article‐row into the SPY index
    feat_cols = []
    if use_sentiment:
        sent = load_sentiment()
        df   = df.join(sent, how="left").fillna(0)
        feat_cols += ["sent_pos","sent_neu","sent_neg"]
    if use_mag:
        mag = load_mag()
        df  = df.join(mag, how="left").fillna(0)
        feat_cols += ["mag_pos","mag_neu","mag_neg"]
    if use_tariff:
        tar = load_tariff()
        df  = df.join(tar, how="left").fillna(0)
        feat_cols += ["tariff_pos","tariff_neu","tariff_neg"]

    # 3) always load Kalshi baseline
    kalshi_df = load_kalshi()
    if use_kalshi:
        df = df.join(kalshi_df, how="left").fillna(0)
        feat_cols += ["kalshi_price"]

    # 4) log‐scale & z‐score everything
    df["price"]   = np.log(df["price"])
    price_scl     = StandardScaler().fit(df[["price"]])
    df["price_z"] = price_scl.transform(df[["price"]])

    full_feats = ["price_z"]
    scalers     = {"price": price_scl}

    if use_sentiment:
        ssc = StandardScaler().fit(df[["sent_pos","sent_neu","sent_neg"]])
        df[["sent_pos_z","sent_neu_z","sent_neg_z"]] = ssc.transform(df[["sent_pos","sent_neu","sent_neg"]])
        full_feats += ["sent_pos_z","sent_neu_z","sent_neg_z"]
        scalers["sent"] = ssc

    if use_mag:
        msc = StandardScaler().fit(df[["mag_pos","mag_neu","mag_neg"]])
        df[["mag_pos_z","mag_neu_z","mag_neg_z"]] = msc.transform(df[["mag_pos","mag_neu","mag_neg"]])
        full_feats += ["mag_pos_z","mag_neu_z","mag_neg_z"]
        scalers["mag"] = msc

    if use_tariff:
        tsc = StandardScaler().fit(df[["tariff_pos","tariff_neu","tariff_neg"]])
        df[["tariff_pos_z","tariff_neu_z","tariff_neg_z"]] = tsc.transform(df[["tariff_pos","tariff_neu","tariff_neg"]])
        full_feats += ["tariff_pos_z","tariff_neu_z","tariff_neg_z"]
        scalers["tariff"] = tsc

    if use_kalshi:
        ksc = StandardScaler().fit(df[["kalshi_price"]])
        df["kalshi_price_z"] = ksc.transform(df[["kalshi_price"]])
        full_feats += ["kalshi_price_z"]
        scalers["kalshi"] = ksc

    # 5) make sliding windows
    X_all, y_all = make_sequences(df, cfg["lookback"], full_feats)
    X_price = X_all[:, :, :1]
    X_other = X_all[:, :, 1:] if X_all.shape[-1] > 1 else None

    # 6) train/val splits
    n_pre = int((1 - cfg["pre_val_frac"]) * len(X_price))
    n_ft  = int(cfg["train_frac"] * len(X_price))

    X_pre_tr, X_pre_val = X_price[:n_pre], X_price[n_pre:]
    y_pre_tr, y_pre_val = y_all[:n_pre],    y_all[n_pre:]

    Xp_tr, Xp_val = X_price[:n_ft], X_price[n_ft:]
    if X_other is not None:
        Xo_tr, Xo_val = X_other[:n_ft], X_other[n_ft:]
    else:
        Xo_tr = Xo_val = None
    y_tr, y_val = y_all[:n_ft], y_all[n_ft:]

    # 7) pipe into tf.data
    ds_pre_tr  = ds_xy(X_pre_tr, None if X_other is None else Xo_tr, y_pre_tr, cfg["batch_size"], shuffle=True,  seed=cfg["seed"]) if X_other else ds_xy(X_pre_tr, y_pre_tr, cfg["batch_size"], shuffle=True,  seed=cfg["seed"])
    ds_pre_val = ds_xy(X_pre_val, None if X_other is None else Xo_val, y_pre_val, cfg["batch_size"], shuffle=False)          if X_other else ds_xy(X_pre_val, y_pre_val, cfg["batch_size"], shuffle=False)
    if X_other is not None:
        ds_ft  = ds_multi(Xp_tr, Xo_tr, y_tr,    cfg["batch_size"], shuffle=True,  seed=cfg["seed"])
        ds_val = ds_multi(Xp_val, Xo_val, y_val, cfg["batch_size"], shuffle=False)
    else:
        ds_ft, ds_val = ds_pre_tr, ds_pre_val

    # 8) test arrays
    dates      = df.index[cfg["lookback"]:]
    test_dates = dates[n_ft:]
    Xp_test    = X_price[n_ft:]
    Xs_test    = X_other[n_ft:] if X_other is not None else None
    y_true_z   = y_all[n_ft:].reshape(-1,1)
    y_true     = np.exp(price_scl.inverse_transform(y_true_z)).flatten()

    print("→ get_datasets: done")
    return (
        ds_pre_tr, ds_pre_val, ds_ft, ds_val,
        Xp_test,   Xs_test,   y_true,   test_dates,
        scalers["price"], kalshi_df
    )
