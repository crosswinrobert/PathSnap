"""GPS track cleaning.

Design rule: the API must return one prediction per GPS observation, so we never silently drop
valid rows. Only rows that cannot be used at all (NaN / impossible coordinates) are removed, and
every row keeps `obs_idx` = its row number in the raw .track file so predictions map back.
"""
import numpy as np
import pandas as pd


def clean_track(track: pd.DataFrame):
    df = track.copy()
    df["obs_idx"] = np.arange(len(df))
    n_raw = len(df)

    valid = (df[["lon", "lat", "t"]].notna().all(axis=1)
             & df["lon"].between(-180, 180) & df["lat"].between(-90, 90))
    n_invalid = int((~valid).sum())
    df = df[valid]

    n_nonmono = int((np.diff(df["t"].to_numpy()) < 0).sum())
    df = df.sort_values("t", kind="stable")          # fixes ordering; no-op on the given data
    n_dup_t = int(df["t"].duplicated().sum())

    report = {"n_raw": n_raw, "n_invalid_dropped": n_invalid,
              "n_out_of_order_steps": n_nonmono, "n_duplicate_timestamps": n_dup_t}
    return df.reset_index(drop=True), report
