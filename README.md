# Sentiment-Augmented LSTM Price Forecasting

http://hdl.handle.net/10230/71635

Thesis project exploring whether news/article sentiment (and Kalshi prediction
market data) improves short-horizon SPY price forecasts on top of a
price-only LSTM baseline. Each experiment trains a two-stage model: (1) an
LSTM pretrained on price history alone, then (2) a fine-tuned LSTM that
concatenates the price branch with a sentiment (or Kalshi) branch, for
several input variants: SPY-only sentiment, Mag 7, Tariffs, Kalshi-as-input,
and combinations excluding/including Kalshi.

## Repository structure

```
config.py                Shared hyperparameter dict (cfg) used everywhere below
model.py                 Model builders + keras_tuner search spaces for each variant
data_loaders/             Six data-loading variants (aggregation x windowing), see
                          data_loaders/README.md
notebooks/
  main_training_pipeline.ipynb    Primary pipeline: keras_tuner hyperparameter
                                  search + training for all variants (Colab or local)
  baseline_no_tuning.ipynb        Earlier, simpler version without hyperparameter
                                  tuning; useful for a quick local sanity check
  feature_analysis.ipynb          Correlation/feature-engineering exploration
                                  (distance correlation between sentiment and price)
  vader_sentiment_aggregation.ipynb  VADER-based sentiment aggregation used to
                                  produce some of the CSVs in data/
data/                     Input CSVs: SPY prices, per-topic article sentiment
                          (Mag 7 / Tariffs / general), Kalshi-derived series
results/                  Output metrics CSVs and plots from past runs
legacy/main_cli_stub.py   Early CLI draft, kept for reference (see note below)
snippets/                 Reference code not meant to be run standalone
keras_tuner/, tuner_results/  Hyperparameter-search checkpoints from past runs
                          (gitignored -- see "Large artifacts" below)
```

## Setup

```
pip install -r requirements.txt
```

Then open any notebook under `notebooks/` with Jupyter. Each notebook's first
cell sets the working directory to the project root, so paths like
`data/spy_2012_2025.csv` resolve correctly regardless of where Jupyter was
launched from.

To try a different sentiment aggregation strategy, edit the import line in
the training notebook's data-loading cell (commented alternatives are listed
inline, and documented in `data_loaders/README.md`):

```python
from data_loaders.daily_avg import get_datasets
# from data_loaders.daily_avg_windowed import get_datasets
# from data_loaders.daily_sum import get_datasets
# ...
```

## Large artifacts

`keras_tuner/` and `tuner_results/` hold checkpoint weights and trial metadata
from past hyperparameter searches (~350MB+) and are excluded from git via
`.gitignore`. They aren't needed to read or re-run the code -- rerunning the
tuning cells in `notebooks/main_training_pipeline.ipynb` will regenerate them.

Three of the raw data CSVs are also excluded via `.gitignore` because they
exceed (or are close to) GitHub's 100MB per-file limit:
`data/KXINX_trades_range_vs_sp.csv` (216MB), `data/3D_articles_new_version_hm.csv`
(157MB), and `data/3D_articles_Tariff_hm.csv` (60MB). They remain on disk in
this folder; if you fork/clone this repo elsewhere you'll need to supply your
own copies (or set up Git LFS) to rerun notebooks that depend on them.

## Known limitations

- `notebooks/feature_analysis.ipynb` references `kalshi_spy.csv`, which
  wasn't present in the original project folder. `data/KXINX_trades_range_vs_sp.csv`
  is the equivalent Kalshi-derived series used elsewhere in this repo; point
  the notebook at your own Kalshi export or adapt it to that file.
- `notebooks/vader_sentiment_aggregation.ipynb` was run against files on the
  original author's machine (hardcoded local paths) and is kept as a record
  of the VADER aggregation step rather than a turnkey script.
- `legacy/main_cli_stub.py` is an early, incomplete CLI entry point that
  imports modules (`data_utils`, `model_utils`, `train_utils`, `eval_utils`)
  that were never added to this repo. It does not run as-is; the notebooks
  are the actual working pipeline.
