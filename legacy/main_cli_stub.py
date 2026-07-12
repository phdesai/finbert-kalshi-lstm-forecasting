
#main.py

"""
NOTE: this is an early, incomplete draft of a CLI entry point for the
two-stage LSTM pipeline. It imports `data_utils`, `model_utils`, `train_utils`,
and `eval_utils`, none of which exist in this repo -- the project's actual
training/evaluation workflow lives in the notebooks under notebooks/ (backed
by data_loaders/, model.py, and config.py). Kept here for reference only;
it will not run as-is.

Original docstring:
Top-level script to train and evaluate the two-stage LSTM pipeline.
Usage:
  python main.py --mode train
  python main.py --mode eval
"""
import argparse
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Train or evaluate LSTM pipeline.")
    parser.add_argument(
        "--mode",
        choices=["train","eval"],
        required=True,
        help="Mode: 'train' to train and save models; 'eval' to evaluate and plot."
    )
    parser.add_argument(
        "--data-dir",
        default="./",
        help="Directory containing CSV data files."
    )
    parser.add_argument(
        "--output-dir",
        default="./artifacts/",
        help="Directory to save or load models, scalers, and npy files."
    )
    args = parser.parse_args()

    from data_utils import load_price_sentiment, load_kalshi, scale_features, make_sequences
    from model_utils import build_price_model, build_ft_model
    from train_utils import train_and_save
    from eval_utils import plot_forecasts, permutation_importance, compute_correlations

    # 1) Load and preprocess data
    price_csv = args.data_dir + "spy_2012_2025.csv"
    sent_csv  = args.data_dir + "3D_articles_new_version_hm.csv"
    df = load_price_sentiment(price_csv, sent_csv)

    # 2) Scale features
    df_z, price_scaler, sent_scaler = scale_features(df)

    # 3) Windowed sequences
    cfg = {
        "lookback":     30,
        "batch_size":   128,
        "pre_epochs":   40,
        "ft_epochs":    40,
        "lr_pre":       1e-3,
        "lr_ft":        1e-4,
        "train_frac":   0.9,
        "pre_val_frac": 0.1,
        "dropout_rate": 0.1,
        "l2_reg":       1e-5,
        "seed":         42
    }
    feat_cols = ["price_z","sent_pos_z","sent_neu_z","sent_neg_z"]
    X_all, y_all = make_sequences(df_z, cfg["lookback"], feat_cols)
    X_price_all = X_all[:, :, :1]
    X_sent_all  = X_all[:, :, 1:]

    if args.mode == "train":
        # build models
        price_model = build_price_model(
            cfg["lookback"], cfg["dropout_rate"], cfg["l2_reg"]
        )
        ft_model = build_ft_model(
            cfg["lookback"], X_sent_all.shape[2],
            cfg["dropout_rate"], cfg["l2_reg"]
        )
        # train and save
        train_and_save(
            price_model,
            ft_model,
            price_scaler,
            sent_scaler,
            X_price_all,
            X_sent_all,
            y_all,
            cfg,
            args.output_dir
        )
        print("Training complete; artifacts saved to", args.output_dir)
        return

    # -- eval mode --
    # reconstruct test_dates
    all_dates = df_z.index[cfg["lookback"]:]
    n_ft       = int(cfg["train_frac"] * len(X_price_all))
    test_dates = all_dates[n_ft:]

    # load artifacts
    from tensorflow.keras.models import load_model
    price_model = load_model(args.output_dir + "price_model.h5", compile=False)
    ft_model    = load_model(args.output_dir + "ft_model.h5",    compile=False)
    price_scaler = pd.read_pickle(args.output_dir + "price_scaler.pkl")
    sent_scaler  = pd.read_pickle(args.output_dir + "sent_scaler.pkl")

    # load test arrays
    Xp_test = np.load(args.output_dir + "Xp_test.npy")
    Xs_test = np.load(args.output_dir + "Xs_test.npy")
    y_test_z = np.load(args.output_dir + "y_test_z.npy")

    # invert transforms to get prices
    y_true = np.exp(price_scaler.inverse_transform(y_test_z).flatten())
    y_po   = np.exp(price_scaler.inverse_transform(
                 price_model.predict(Xp_test)
             )).flatten()
    y_ps   = np.exp(price_scaler.inverse_transform(
                 ft_model.predict([Xp_test, Xs_test])
             )).flatten()

    # load Kalshi
    kalshi = load_kalshi(args.data_dir + "kalshi_spy.csv", test_dates)

    # plot forecasts
    plot_forecasts(test_dates, y_true, y_po, y_ps, kalshi)

    # compute & print baseline correlations
    mae_po, r_po, dc_po = compute_correlations(y_true, y_po)
    mae_ps, r_ps, dc_ps = compute_correlations(y_true, y_ps)
    print(f"Price-only MAE={mae_po:.4f}, Pearson r={r_po:.4f}, dCorr={dc_po:.4f}")
    print(f"Price+Sent  MAE={mae_ps:.4f}, Pearson r={r_ps:.4f}, dCorr={dc_ps:.4f}")

    # permutation importance
    df_imp = permutation_importance(
        ft_model, price_scaler, Xp_test, Xs_test, y_true, seed=cfg['seed']
    )
    print("\nFeature importances (permutation):")
    print(df_imp)

    # plot importances
    from eval_utils import plot_importances
    plot_importances(df_imp)

if __name__ == "__main__":
    main()

