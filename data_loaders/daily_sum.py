"""
daily_sum.py (formerly data_loader_sum.py)

Variant of daily_avg.py that sums (rather than averages) sentiment scores
within each calendar day, using naive calendar-date normalization (no
market-close windowing).

See data_loaders/README.md for how this variant relates to the other five.
"""
import config as cfg
import os
import random
import pandas as pd
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
    # change from business-day to calendar-day frequency
    spy = spy.asfreq("B").ffill()
    return spy.loc[:"2020-07-29"], spy.loc["2020-07-30":]


def load_sentiment():
    print("  → load_sentiment() | concatenating articles then normalizing")
    tmp = pd.read_csv(
        "data/3D_articles_new_version_hm.csv",
        usecols=["date","mean_positive","mean_neutral","mean_negative"],
        parse_dates=["date"]
    )
    # normalize date to midnight for alignment
    tmp["date"] = tmp["date"].dt.normalize()
    # rename columns
    tmp = tmp.rename(columns={
        "mean_positive":"sent_pos",
        "mean_neutral":"sent_neu",
        "mean_negative":"sent_neg"
    })
    # sum sentiment values per day instead of averaging across articles
    tmp = tmp.groupby("date", as_index=True).sum()  # <-- changed to .sum()  (was no grouping)
    # normalize each day's totals so sent_pos+sent_neu+sent_neg = 1
    row_sums = tmp[["sent_pos","sent_neu","sent_neg"]].sum(axis=1).replace(0, 1)
    tmp[["sent_pos","sent_neu","sent_neg"]] = (
        tmp[["sent_pos","sent_neu","sent_neg"]]
        .div(row_sums, axis=0)
    )
    return tmp.sort_index()


def load_mag():
    print("  → load_mag()")
    tmp = pd.read_csv(
        "data/3D_articles_Mag_7_hm.csv",
        usecols=["date","mean_positive","mean_neutral","mean_negative"],
        parse_dates=["date"]
    )
    tmp["date"] = tmp["date"].dt.normalize()
    # sum sentiment values per day instead of mean
    daily = (
        tmp
        .groupby("date", as_index=True)
        .sum()  # <-- changed from .mean() to .sum()
        .rename(columns={
            "mean_positive":"mag_pos",
            "mean_neutral":"mag_neu",
            "mean_negative":"mag_neg"
        })
    )
    s = daily[["mag_pos","mag_neu","mag_neg"]].sum(axis=1).replace(0,1)
    daily[["mag_pos","mag_neu","mag_neg"]] = (
        daily[["mag_pos","mag_neu","mag_neg"]].div(s, axis=0)
    )
    return daily.sort_index().loc[:"2025-04-30"]


def load_tariff():
    print("  → load_tariff()")
    tmp = pd.read_csv(
        "data/3D_articles_Tariff_hm.csv",
        usecols=["date","mean_positive","mean_neutral","mean_negative"],
        parse_dates=["date"]
    )
    tmp["date"] = tmp["date"].dt.normalize()
    # sum sentiment values per day instead of mean
    daily = (
        tmp
        .groupby("date", as_index=True)
        .sum()  # <-- changed from .mean() to .sum()
        .rename(columns={
            "mean_positive":"tariff_pos",
            "mean_neutral":"tariff_neu",
            "mean_negative":"tariff_neg"
        })
    )
    s = daily[["tariff_pos","tariff_neu","tariff_neg"]].sum(axis=1).replace(0,1)
    daily[["tariff_pos","tariff_neu","tariff_neg"]] = (
        daily[["tariff_pos","tariff_neu","tariff_neg"]].div(s, axis=0)
    )
    return daily.sort_index().loc[:"2025-04-30"]



def load_kalshi():
    print("  → load_kalshi() (optimized)")
    df = pd.read_csv(
        "data/KXINX_trades_range_vs_sp.csv",
        usecols=["trade_date","bucket_start","bucket_end","yes_price"],
        parse_dates=["trade_date"]
    )
    df_valid = df.dropna(subset=["bucket_start","bucket_end"])
    df_valid = (
        df_valid.sort_values(["trade_date","yes_price"], ascending=[True, False])
                .drop_duplicates("trade_date")
    )
    missing = set(df["trade_date"]) - set(df_valid["trade_date"])
    df_fallback = df[df["trade_date"].isin(missing)]
    df_fallback = (
        df_fallback.sort_values(["trade_date","yes_price"], ascending=[True, False])
                   .drop_duplicates("trade_date")
    )
    best = pd.concat([df_valid, df_fallback]).sort_values("trade_date")
    best["avg_bucket"] = (best.bucket_start + best.bucket_end) / 2
    kalshi = (
        best.rename(columns={"trade_date":"date","avg_bucket":"kalshi_price"})
            [["date","kalshi_price"]]
            .set_index("date")
            .sort_index()
    )
    return kalshi

# ------------------------------------------------------------------------
# Master loader: assemble pipelines and test data
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

    # ─── SPY ─────────────────────────────────────────────────────────────────
    print("→ get_datasets: loading SPY")
    spy_pre, spy_post = load_spy()
    print(f"   SPY post-train shape: {spy_post.shape}")
    df = spy_post.copy()

    # ─── SENTIMENT ───────────────────────────────────────────────────────────
    feat_cols = []
    if use_sentiment:
        print("→ joining raw sentiment rows")
        sent = load_sentiment()
        # fill missing with zeros
        df = df.join(sent, how="left").fillna(0)
        feat_cols += ["sent_pos","sent_neu","sent_neg"]

    if use_mag:
        print("→ joining Mag-7 sentiment")
        mag = load_mag()
        df = df.join(mag, how="left").fillna(0)
        feat_cols += ["mag_pos","mag_neu","mag_neg"]

    if use_tariff:
        print("→ joining Tariff sentiment")
        tar = load_tariff()
        df = df.join(tar, how="left").fillna(0)
        feat_cols += ["tariff_pos","tariff_neu","tariff_neg"]

    # ─── KALSHI AS INPUT ───────────────────────────────────────────────────────
    print("→ get_datasets: loading raw Kalshi for baseline")
    kalshi_df = load_kalshi()

    if use_kalshi:
        print("→ get_datasets: joining Kalshi as input feature")
        df = df.join(kalshi_df, how="left").fillna(0)
        feat_cols += ["kalshi_price"]

    # ─── SCALE ALL FEATURES ───────────────────────────────────────────────────
    print("→ scaling features")
    df["price"] = np.log(df["price"])
    price_scaler = StandardScaler().fit(df[["price"]])
    df["price_z"] = price_scaler.transform(df[["price"]])

    full_feat_cols = ["price_z"]

    if use_sentiment:
        sent_scaler = StandardScaler().fit(df[["sent_pos","sent_neu","sent_neg"]])
        df[["sent_pos_z","sent_neu_z","sent_neg_z"]] = sent_scaler.transform(
            df[["sent_pos","sent_neu","sent_neg"]]
        )
        full_feat_cols += ["sent_pos_z","sent_neu_z","sent_neg_z"]

    if use_mag:
        mag_scaler = StandardScaler().fit(df[["mag_pos","mag_neu","mag_neg"]])
        df[["mag_pos_z","mag_neu_z","mag_neg_z"]] = mag_scaler.transform(
            df[["mag_pos","mag_neu","mag_neg"]]
        )
        full_feat_cols += ["mag_pos_z","mag_neu_z","mag_neg_z"]

    if use_tariff:
        tar_scaler = StandardScaler().fit(df[["tariff_pos","tariff_neu","tariff_neg"]])
        df[["tariff_pos_z","tariff_neu_z","tariff_neg_z"]] = tar_scaler.transform(
            df[["tariff_pos","tariff_neu","tariff_neg"]]
        )
        full_feat_cols += ["tariff_pos_z","tariff_neu_z","tariff_neg_z"]

    if use_kalshi:
        ks_scaler = StandardScaler().fit(df[["kalshi_price"]])
        df["kalshi_price_z"] = ks_scaler.transform(df[["kalshi_price"]])
        full_feat_cols += ["kalshi_price_z"]

    # ─── SEQUENCES ─────────────────────────────────────────────────────────────
    print("→ making sequences")
    X_all, y_all = make_sequences(df, cfg["lookback"], full_feat_cols)
    X_price = X_all[:, :, :1]
    X_other = X_all[:, :, 1:] if X_all.shape[-1] > 1 else None

    # ─── SPLITS ────────────────────────────────────────────────────────────────
    print("→ splitting pretrain/fine-tune")
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

    # ─── DATASETS ─────────────────────────────────────────────────────────────
    print("→ creating tf.data datasets")
    ds_pre_tr  = ds_xy(  X_pre_tr,  y_pre_tr,  cfg["batch_size"], shuffle=True,  seed=cfg.get("seed",42))
    ds_pre_val = ds_xy(  X_pre_val, y_pre_val, cfg["batch_size"], shuffle=False)
    if Xo_tr is not None:
        ds_ft  = ds_multi(Xp_tr, Xo_tr, y_tr, cfg["batch_size"], shuffle=True,  seed=cfg.get("seed",42))
        ds_val = ds_multi(Xp_val, Xo_val, y_val, cfg["batch_size"], shuffle=False)
    else:
        ds_ft, ds_val = ds_pre_tr, ds_pre_val

    # ─── TEST DATA ────────────────────────────────────────────────────────────
    print("→ preparing test data")
    dates      = df.index[cfg["lookback"]:]
    test_dates = dates[n_ft:]
    Xp_test    = X_price[n_ft:]
    Xs_test    = X_other[n_ft:] if X_other is not None else None
    y_true_z   = y_all[n_ft:].reshape(-1,1)
    y_true     = np.exp(price_scaler.inverse_transform(y_true_z)).flatten()

    print("→ get_datasets: done")
    return (
        ds_pre_tr, ds_pre_val, ds_ft, ds_val,
        Xp_test,   Xs_test,   y_true,   test_dates,
        price_scaler, kalshi_df
    )
