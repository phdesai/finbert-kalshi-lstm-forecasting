"""
patience_estimation_reference.py (formerly training_script.py)

Reference snippet, not a standalone script: `estimate_patience()` is
self-contained and reusable, but the "usage" section below it assumes
variables (`model`, `ds_pre_tr_spy`, `ds_pre_val_spy`, `cfg`, `lr`,
`get_callbacks`) that are defined inline in the training notebooks
(see notebooks/main_training_pipeline.ipynb). Copy the pattern into a
notebook cell rather than running this file directly.
"""
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

def estimate_patience(model, 
                      train_ds, val_ds, 
                      warmup_epochs=50, 
                      patience_fraction=0.1,
                      min_patience=3):
    """
    1) Do a warm-up run of warmup_epochs (no early stopping), record val_loss.
    2) Find argmin of val_loss: epoch where the model first got its best score.
    3) Recommend patience = max(min_patience, int(argmin * patience_fraction)).
    """
    # 1) warm-up
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        verbose=0  # suppress epoch logs
    )
    
    val_losses = np.array(hist.history['val_loss'])
    best_epoch = val_losses.argmin() + 1   # +1 because epochs are 1-indexed
    
    # 2) compute patience
    pat = max(min_patience, int(best_epoch * patience_fraction))
    print(f" → Validation loss bottomed at epoch {best_epoch}.")
    print(f" → Setting patience to {pat} (={patience_fraction*100:.0f}% of {best_epoch}).")
    return pat

# ── usage ───────────────────────
# build & compile your model exactly as you will for final training:
model = build_price()   # or build_ft(...)
model.compile(Adam(lr), loss='mse')

# estimate a good patience
pat = estimate_patience(model, ds_pre_tr_spy, ds_pre_val_spy,
                        warmup_epochs=30,
                        patience_fraction=0.1,
                        min_patience=5)

# now retrain (or continue training) with EarlyStopping
es = EarlyStopping(
    monitor='val_loss',
    patience=pat,
    min_delta=1e-4,
    restore_best_weights=True
)

model.fit(
    ds_pre_tr_spy,
    validation_data=ds_pre_val_spy,
    epochs=cfg['pre_epochs'],
    callbacks=[es] + get_callbacks(...),
    verbose=1
)
