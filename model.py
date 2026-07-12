"""
model.py

Keras model builders and keras_tuner search spaces for the two-stage
(price-only pretrain -> price+sentiment fine-tune) LSTM pipeline, plus
per-experiment variants (SPY, Mag7, Tariffs, Kalshi-as-input,
AllExcKalshi, All). Imported by the training notebooks under notebooks/.
"""

from tensorflow.keras import Sequential, Model, Input, regularizers
from tensorflow.keras.layers import LSTM, Dropout, Dense, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from config import cfg


# --- Build price-only model ---
def build_price():
    model = Sequential()
    model.add(Input(shape=(cfg["lookback"], 1)))
    model.add(LSTM(cfg["lstm_units"], name="ps"))
    model.add(Dropout(cfg["dropout"]))
    model.add(Dense(1))
    return model

# --- Build fine-tune model ---
def build_ft(s_dims):
    lookback     = cfg["lookback"]
    dropout_rate = cfg["dropout_rate"]
    l2_reg       = cfg["l2_reg"]
    lstm_units   = cfg["lstm_units"]

    # Price input
    inp_p = Input(shape=(lookback, 1))
    x_p   = LSTM(lstm_units, name="ps")(inp_p)
    x_p   = Dropout(dropout_rate)(x_p)

    # Sentiment input
    inp_s = Input(shape=(lookback, s_dims))
    x_s   = LSTM(lstm_units // 2)(inp_s)
    x_s   = Dropout(dropout_rate)(x_s)

    # Merge and output
    m = Concatenate()([x_p, x_s])
    m = BatchNormalization()(m)
    x = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(m)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, name="o")(x)

    return Model(inputs=[inp_p, inp_s], outputs=out)



def build_model(hp, s_dims=None, use_sentiment=False):
    """
    Builds a tunable model for Keras Tuner.
    If `use_sentiment` is False, builds price-only model.
    If True, adds sentiment (or other auxiliary) input of shape `s_dims`.
    """
    lstm_units = hp.Int("units", 32, 256, step=32)
    dropout    = hp.Float("dropout", 0.0, 0.5, step=0.1)
    lr         = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])

    if not use_sentiment:
        model = Sequential()
        model.add(Input(shape=(cfg["lookback"], 1)))
        model.add(LSTM(lstm_units, name="ps"))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        model.compile(optimizer=Adam(lr), loss="mse")
        return model

    # Price input
    inp_p = Input(shape=(cfg["lookback"], 1))
    x_p   = LSTM(lstm_units, name="ps")(inp_p)
    x_p   = Dropout(dropout)(x_p)

    # Auxiliary input
    inp_s = Input(shape=(cfg["lookback"], s_dims))
    x_s   = LSTM(lstm_units // 2)(inp_s)
    x_s   = Dropout(dropout)(x_s)

    m = Concatenate()([x_p, x_s])
    m = BatchNormalization()(m)
    x = Dense(32, activation="relu")(m)
    x = Dropout(dropout)(x)
    out = Dense(1)(x)

    model = Model(inputs=[inp_p, inp_s], outputs=out)
    model.compile(optimizer=Adam(lr), loss="mse")
    return model

# model.py

from tensorflow.keras import Sequential, Model, Input, regularizers
from tensorflow.keras.layers import LSTM, Dropout, Dense, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from config import cfg


# ----------------------------------------
# 1. Base: price-only model
# ----------------------------------------
def build_price():
    """
    Build a price-only Sequential model. Uses:
      - Input: (lookback, 1)
      - One LSTM layer named "ps" with cfg["lstm_units"] units
      - One Dropout layer with rate cfg["dropout"]
      - One Dense(1) output
    """
    model = Sequential()
    model.add(Input(shape=(cfg["lookback"], 1)))
    model.add(LSTM(cfg["lstm_units"], name="ps"))
    model.add(Dropout(cfg["dropout"]))
    model.add(Dense(1))
    return model


# ----------------------------------------
# 2. Base: fine-tune model structure (shared code)
# ----------------------------------------
def build_ft(s_dims):
    """
    Build a fine-tune model that merges a “price” LSTM branch with
    an auxiliary branch of dimension s_dims (e.g. sentiment or Mag7 or tariff).
    - Both branches use cfg["lstm_units"], but the second branch uses half units if desired.
    - Dropout rate for both branches uses cfg["dropout_rate"] (copy it from cfg["dropout_s"] in notebook).
    - After merging, a Dense(32) + Dropout + Dense(1) output.
    """
    lookback     = cfg["lookback"]
    dropout_rate = cfg["dropout_rate"]   # NOTE: make sure notebook sets cfg["dropout_rate"] = cfg["dropout_s"]
    l2_reg       = cfg["l2_reg"]
    lstm_units   = cfg["lstm_units"]

    # Price branch
    inp_p = Input(shape=(lookback, 1), name="price_input")
    x_p   = LSTM(lstm_units, name="ps")(inp_p)
    x_p   = Dropout(dropout_rate)(x_p)

    # Auxiliary branch
    inp_s = Input(shape=(lookback, s_dims), name="aux_input")
    x_s   = LSTM(lstm_units // 2)(inp_s)
    x_s   = Dropout(dropout_rate)(x_s)

    # Merge & dense layers
    m = Concatenate()([x_p, x_s])
    m = BatchNormalization()(m)
    x = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(m)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, name="o")(x)

    return Model(inputs=[inp_p, inp_s], outputs=out)


# ----------------------------------------
# 3. Tuner models for each experiment
#    (Each returns a compiled model for Keras Tuner to search.)
# ----------------------------------------

def build_model_spy_sentiment(hp):
    """
    Build tuner model for SPY (price + sentiment). 
    Hyperparameters tuned: 
      - 'units'       : number of LSTM units in price branch
      - 'dropout'     : dropout after price LSTM
      - 'sent_units'  : number of LSTM units in sentiment branch
      - 'dropout_s'   : dropout after sentiment LSTM
      - 'learning_rate': choice of optimizer LR
    """
    lstm_units = hp.Int("units", 32, 256, step=32)
    dropout    = hp.Float("dropout", 0.0, 0.5, step=0.1)
    sent_units = hp.Int("sent_units", 16, 128, step=16)
    dropout_s  = hp.Float("dropout_s", 0.0, 0.5, step=0.1)
    lr         = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])

    # Price branch
    inp_p = Input(shape=(cfg["lookback"], 1), name="price_input")
    x_p   = LSTM(lstm_units, name="ps")(inp_p)
    x_p   = Dropout(dropout)(x_p)

    # Sentiment branch
    inp_s = Input(shape=(cfg["lookback"], cfg["sentiment_dim"]), name="sent_input")
    x_s   = LSTM(sent_units)(inp_s)
    x_s   = Dropout(dropout_s)(x_s)

    # Merge & output
    merged = Concatenate()([x_p, x_s])
    merged = BatchNormalization()(merged)
    out    = Dense(1)(merged)

    model = Model(inputs=[inp_p, inp_s], outputs=out)
    model.compile(optimizer=Adam(lr), loss="mse")
    return model


def build_model_mag7(hp, s_dims):
    """
    Build tuner model for Mag7 (price + Mag7). 
    Tunable hyperparams:
      - 'units'      : price LSTM units
      - 'dropout'    : price dropout
      - 'mag_units'  : Mag7 LSTM units
      - 'dropout_s'  : dropout after Mag7 LSTM
      - 'learning_rate': optimizer LR
    """
    lstm_units = hp.Int("units", 32, 256, step=32)
    dropout    = hp.Float("dropout", 0.0, 0.5, step=0.1)
    mag_units  = hp.Int("mag_units", 16, 128, step=16)
    dropout_s  = hp.Float("dropout_s", 0.0, 0.5, step=0.1)
    lr         = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])

    inp_p = Input(shape=(cfg["lookback"], 1), name="price_input")
    x_p   = LSTM(lstm_units, name="ps")(inp_p)
    x_p   = Dropout(dropout)(x_p)

    inp_m = Input(shape=(cfg["lookback"], s_dims), name="mag_input")
    x_m   = LSTM(mag_units)(inp_m)
    x_m   = Dropout(dropout_s)(x_m)

    merged = Concatenate()([x_p, x_m])
    merged = BatchNormalization()(merged)
    out    = Dense(1)(merged)

    model = Model(inputs=[inp_p, inp_m], outputs=out)
    model.compile(optimizer=Adam(lr), loss="mse")
    return model


def build_model_tariffs(hp, s_dims):
    """
    Build tuner model for Tariffs (price + tariff). 
    Tunable hyperparams:
      - 'units'      : price LSTM units
      - 'dropout'    : price dropout
      - 'tariff_units': tariff LSTM units
      - 'dropout_s'  : dropout after tariff LSTM
      - 'learning_rate': optimizer LR
    """
    lstm_units  = hp.Int("units", 32, 256, step=32)
    dropout     = hp.Float("dropout", 0.0, 0.5, step=0.1)
    tariff_units = hp.Int("tariff_units", 16, 128, step=16)
    dropout_s   = hp.Float("dropout_s", 0.0, 0.5, step=0.1)
    lr          = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])

    inp_p = Input(shape=(cfg["lookback"], 1), name="price_input")
    x_p   = LSTM(lstm_units, name="ps")(inp_p)
    x_p   = Dropout(dropout)(x_p)

    inp_t = Input(shape=(cfg["lookback"], s_dims), name="tariff_input")
    x_t   = LSTM(tariff_units)(inp_t)
    x_t   = Dropout(dropout_s)(x_t)

    merged = Concatenate()([x_p, x_t])
    merged = BatchNormalization()(merged)
    out    = Dense(1)(merged)

    model = Model(inputs=[inp_p, inp_t], outputs=out)
    model.compile(optimizer=Adam(lr), loss="mse")
    return model


def build_model_kalshi(hp, s_dims):
    """
    Build tuner model for KalshiInput (price + Kalshi). 
    Tunable hyperparams:
      - 'units'      : price LSTM units
      - 'dropout'    : price dropout
      - 'kalshi_units': Kalshi LSTM units
      - 'dropout_s'  : dropout after Kalshi LSTM
      - 'learning_rate': optimizer LR
    """
    lstm_units    = hp.Int("units", 32, 256, step=32)
    dropout       = hp.Float("dropout", 0.0, 0.5, step=0.1)
    kalshi_units  = hp.Int("kalshi_units", 16, 128, step=16)
    dropout_s     = hp.Float("dropout_s", 0.0, 0.5, step=0.1)
    lr            = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])

    inp_p = Input(shape=(cfg["lookback"], 1), name="price_input")
    x_p   = LSTM(lstm_units, name="ps")(inp_p)
    x_p   = Dropout(dropout)(x_p)

    inp_k = Input(shape=(cfg["lookback"], s_dims), name="kalshi_input")
    x_k   = LSTM(kalshi_units)(inp_k)
    x_k   = Dropout(dropout_s)(x_k)

    merged = Concatenate()([x_p, x_k])
    merged = BatchNormalization()(merged)
    out    = Dense(1)(merged)

    model = Model(inputs=[inp_p, inp_k], outputs=out)
    model.compile(optimizer=Adam(lr), loss="mse")
    return model


def build_model_allexc_kalshi(hp, s_dims):
    """
    Build tuner model for AllExcKalshi (price + sentiment+mag+tariff, no Kalshi input). 
    Tunable hyperparams:
      - 'units'      : price LSTM units
      - 'dropout'    : price dropout
      - 'aux_units'  : LSTM units for all‐aux branch
      - 'dropout_s'  : dropout after aux LSTM
      - 'learning_rate': optimizer LR
    """
    lstm_units  = hp.Int("units", 32, 256, step=32)
    dropout     = hp.Float("dropout", 0.0, 0.5, step=0.1)
    aux_units   = hp.Int("aux_units", 16, 128, step=16)
    dropout_s   = hp.Float("dropout_s", 0.0, 0.5, step=0.1)
    lr          = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])

    inp_p = Input(shape=(cfg["lookback"], 1), name="price_input")
    x_p   = LSTM(lstm_units, name="ps")(inp_p)
    x_p   = Dropout(dropout)(x_p)

    inp_s = Input(shape=(cfg["lookback"], s_dims), name="aux_input")
    x_s   = LSTM(aux_units)(inp_s)
    x_s   = Dropout(dropout_s)(x_s)

    merged = Concatenate()([x_p, x_s])
    merged = BatchNormalization()(merged)
    out    = Dense(1)(merged)

    model = Model(inputs=[inp_p, inp_s], outputs=out)
    model.compile(optimizer=Adam(lr), loss="mse")
    return model


def build_model_all(hp, s_dims):
    """
    Build tuner model for All (price + sentiment+mag+tariff+Kalshi). 
    Tunable hyperparams:
      - 'units'      : price LSTM units
      - 'dropout'    : price dropout
      - 'aux_units'  : LSTM units for combined all‐inputs branch
      - 'dropout_s'  : dropout after all‐inputs LSTM
      - 'learning_rate': optimizer LR
    """
    lstm_units = hp.Int("units", 32, 256, step=32)
    dropout    = hp.Float("dropout", 0.0, 0.5, step=0.1)
    aux_units  = hp.Int("aux_units", 16, 128, step=16)
    dropout_s  = hp.Float("dropout_s", 0.0, 0.5, step=0.1)
    lr         = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])

    inp_p = Input(shape=(cfg["lookback"], 1), name="price_input")
    x_p   = LSTM(lstm_units, name="ps")(inp_p)
    x_p   = Dropout(dropout)(x_p)

    inp_s = Input(shape=(cfg["lookback"], s_dims), name="aux_input")
    x_s   = LSTM(aux_units)(inp_s)
    x_s   = Dropout(dropout_s)(x_s)

    merged = Concatenate()([x_p, x_s])
    merged = BatchNormalization()(merged)
    out    = Dense(1)(merged)

    model = Model(inputs=[inp_p, inp_s], outputs=out)
    model.compile(optimizer=Adam(lr), loss="mse")
    return model
