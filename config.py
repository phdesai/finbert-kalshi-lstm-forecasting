"""
config.py

Single shared hyperparameter dict (`cfg`) imported by model.py, the
data_loaders/ package, and the training notebooks. Edit values here to
change lookback window, batch size, learning rates, tuning behavior, etc.
across the whole pipeline.

Author's research notes, kept for context:
- For Daily Average sentiment training use batch_size = 16
- For Article Average sentiment training use batch_size = 128
- Previous lag returns and volatility, and maybe FinBERT isn't good, VADER
- Sum the scores (HOW DIDN'T WE THINK OF THIS EARLIER)
- Paper about how to aggregate the headlines based on previous headlines
- Could predict categorical variable
- Could break the day into smaller intervals since markets react to news
  instantly - price will not respond tomorrow. Would need intraday price
  data (picking the correct frequency is important).
"""

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
    "seed":         42,
    
    # New parameters for patience and hyperparameter tuning
    "patience":             10,      # Early stopping patience
    "tuning_max_trials":    20,      # Maximum number of hyperparameter combinations to try
    "tuning_executions":    1,       # Number of times to train each combination
    "use_tuning":          True,    # Whether to use hyperparameter tuning
}