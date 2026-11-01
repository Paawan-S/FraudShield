"""Feature engineering shared by training (notebooks) and the app (Phase 10)."""
import numpy as np
import pandas as pd

IN_SCOPE_TYPES = ["TRANSFER", "CASH_OUT"]

# Set A: strict, no balances at all
SET_A_TREE = ["log_amount", "hour", "is_transfer"]
SET_A_LINEAR = ["log_amount", "hour_sin", "hour_cos", "is_transfer"]
SET_A_NO_HOUR = ["log_amount", "is_transfer"]

# Set B: Set A plus features built from the sender's balance BEFORE the transaction
SET_B_EXTRA = ["amt_to_balance", "empties_account", "old_zero"]
SET_B_TREE = SET_A_TREE + SET_B_EXTRA
SET_B_LINEAR = SET_A_LINEAR + SET_B_EXTRA


def build_features(df):
    """Needs columns: step, type, amount, oldbalanceOrg. Returns only the new feature columns."""
    out = pd.DataFrame(index=df.index)

    out["log_amount"] = np.log1p(df["amount"]).astype("float32")

    hour = (df["step"].astype("int32") - 1) % 24
    out["hour"] = hour.astype("int8")
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")

    out["is_transfer"] = (df["type"] == "TRANSFER").astype("int8")

    amt = df["amount"].to_numpy(dtype="float64")
    old = df["oldbalanceOrg"].to_numpy(dtype="float64")
    ratio = np.divide(amt, old, out=np.zeros_like(amt), where=old > 0)
    out["amt_to_balance"] = np.clip(ratio, 0, 10).astype("float32")
    out["empties_account"] = ((old > 0) & np.isclose(amt, old, rtol=0, atol=0.01)).astype("int8")
    out["old_zero"] = (old == 0).astype("int8")

    return out