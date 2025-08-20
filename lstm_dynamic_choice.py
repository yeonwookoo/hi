
"""
LSTM-based Dynamic Discrete Choice (continue vs stop) for taxi labor-supply.
- Reads the sequences pickle produced by prepare_taxi_sequences.py
- Trains an LSTM with masking on variable-length sequences
- Evaluates and plots basic diagnostics
- Provides a simple policy simulation (finite-difference) for revenue_per_hour_t

Usage:
  python lstm_dynamic_choice.py --data_pickle /path/to/dataset_taxi_sequences.pkl \
                                --epochs 10 --batch 64 --hidden 64

"""

import argparse, pickle, math
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

def pad_sequences(seqs, labels, pad_value=0.0):
    """Pad ragged sequences to [B, T, F] and labels to [B, T]. Also build masks (0 for padded)."""
    lens = np.array([s.shape[0] for s in seqs], dtype=np.int32)
    B = len(seqs); T = int(lens.max()); F = seqs[0].shape[1]
    X = np.full((B, T, F), pad_value, dtype=np.float32)
    y = np.zeros((B, T), dtype=np.float32)
    mask = np.zeros((B, T), dtype=np.float32)
    for i, (s, l) in enumerate(zip(seqs, labels)):
        t = s.shape[0]
        X[i, :t, :] = s
        y[i, :t] = l
        mask[i, :t] = 1.0
    # Mask last timestep in every sequence (no next choice to predict after terminal step)
    last_idx = (lens - 1).clip(min=0)
    for i in range(B):
        mask[i, last_idx[i]] = 0.0
    return X, y, mask

def build_model(n_features, hidden=64, lr=1e-3, dropout=0.1):
    inp = layers.Input(shape=(None, n_features), name="seq")
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.LSTM(hidden, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.TimeDistributed(layers.Dense(32, activation='relu'))(x)
    out = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'), name="p_continue")(x)
    model = models.Model(inp, out)
    opt = optimizers.Adam(lr)
    model.compile(optimizer=opt, loss="binary_crossentropy", sample_weight_mode="temporal", metrics=["accuracy"])
    return model

def temporal_sample_weights(mask):
    """Use mask as sample weights so padded steps (and terminal steps) don't contribute to loss."""
    return mask

def evaluate(model, X, y, w):
    pred = model.predict(X, verbose=0)[:, :, 0]
    # masked metrics
    eps = 1e-8
    y_masked = y[w>0.5]
    p_masked = pred[w>0.5]
    acc = ((p_masked>=0.5)==(y_masked>=0.5)).mean()
    bce = -(y_masked*np.log(p_masked+eps) + (1-y_masked)*np.log(1-p_masked+eps)).mean()
    return {"acc": float(acc), "bce": float(bce)}

def simulate_fd(model, X, w, feature_idx, delta=+0.1):
    """Finite-difference policy simulation: increase feature by 'delta' (e.g., revenue_per_hour + $0.1)
       Return average change in continue prob on valid (unmasked) steps."""
    X2 = X.copy()
    X2[:, :, feature_idx] += delta
    p0 = model.predict(X,  verbose=0)[:, :, 0]
    p1 = model.predict(X2, verbose=0)[:, :, 0]
    dm = (p1 - p0)[w>0.5]
    return float(dm.mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_pickle", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    with open(args.data_pickle, "rb") as f:
        data = pickle.load(f)
    X_tr_seqs, Y_tr_seqs = data["X_train"], data["Y_train"]
    X_va_seqs, Y_va_seqs = data["X_valid"], data["Y_valid"]
    feat_names = data["feature_names"]
    nF = len(feat_names)

    X_tr, y_tr, w_tr = pad_sequences(X_tr_seqs, Y_tr_seqs, pad_value=0.0)
    X_va, y_va, w_va = pad_sequences(X_va_seqs, Y_va_seqs, pad_value=0.0)

    model = build_model(nF, hidden=args.hidden, lr=args.lr, dropout=0.1)
    model.summary()

    cbs = [callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")]
    model.fit(
        X_tr, y_tr[...,None],
        validation_data=(X_va, y_va[...,None], temporal_sample_weights(w_va)),
        sample_weight=temporal_sample_weights(w_tr),
        epochs=args.epochs, batch_size=args.batch, verbose=2, shuffle=True
    )

    tr_metrics = evaluate(model, X_tr, y_tr, w_tr)
    va_metrics = evaluate(model, X_va, y_va, w_va)
    print("Train:", tr_metrics)
    print("Valid:", va_metrics)

    # Example policy sim: +$1/h revenue
    if "revenue_per_hour_t" in feat_names:
        idx = feat_names.index("revenue_per_hour_t")
        dP = simulate_fd(model, X_va, w_va, idx, delta=+1.0)
        print(f"PolicySim: +1 $/h revenue => ΔP(continue) ≈ {dP:.4f} (valid, masked)")

    # Save model
    out_path = Path(args.data_pickle).with_suffix(".h5")
    model.save(out_path)
    print(f"Saved model to: {out_path}")

if __name__ == "__main__":
    main()
