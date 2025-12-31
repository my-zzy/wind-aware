#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, ast
import numpy as np
import torch
import pandas as pd
from types import SimpleNamespace

from meta_pinn import MetaPINN, train_meta_pinn_multitask, plot_training_curves
from meta_pinn.utils import set_seed
from meta_pinn.config import DEFAULT_OPTIONS
from meta_pinn.physics import calibrate_beta_drag_vec

# ========== Data reading and formatting ==========
class FlightDataset:
    def __init__(self, csv_file, feature_keys, output_key):
        df = pd.read_csv(csv_file)
        def parse(x): return np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x)

        # Input X
        X_list = []
        for key in feature_keys:
            if key in df.columns:
                col = df[key].apply(parse).to_list()
                arr = np.vstack(col)
                X_list.append(arr)
        self.X = np.hstack(X_list)

        # Output Y (fa)
        Y_col = df[output_key].apply(parse).to_list()
        self.Y = np.vstack(Y_col)

        # Time t (optional)
        if "t" in df.columns:
            self.t = df["t"].to_numpy()
        else:
            self.t = np.arange(len(self.X))

# --- Packaged into meta-pinn compatible data objects ---
def wrap_as_taskdataset(ds, filename="unknown", method="manual", condition="wind"):
    return SimpleNamespace(
        X = ds.X.astype(np.float32),
        Y = ds.Y.astype(np.float32),
        meta = {
            "filename": filename,
            "method": method,
            "condition": condition,
            "t": ds.t.tolist()
        }
    )

def load_datasets(data_dir, feature_keys, output_key="fa"):
    """Load all CSV files in the folder as meta-pinn Task data"""
    datasets = []
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    for f in csv_files:
        ds = FlightDataset(f, feature_keys, output_key)
        wrapped = wrap_as_taskdataset(ds, filename=os.path.basename(f))
        datasets.append(wrapped)
        print(f"[Loaded] {f} | X.shape={ds.X.shape}, Y.shape={ds.Y.shape}")
    return datasets

def save_input_scaler(tasks, out_npz_path):
    X_all = np.concatenate([t.X for t in tasks], axis=0)
    x_mean = X_all.mean(axis=0).astype(np.float32)
    x_std  = X_all.std(axis=0).astype(np.float32)
    x_std[x_std < 1e-6] = 1.0  
    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
    np.savez(out_npz_path, x_mean=x_mean, x_std=x_std)
    print(f"[Scaler saved] {out_npz_path} | dim={x_mean.shape[0]}")
    return x_mean, x_std

# ========== Training ==========
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(DEFAULT_OPTIONS['seed'], deterministic=True)

    feature_keys = DEFAULT_OPTIONS['features']   # from options,e.g., ['v','q','pwm']
    output_key = "fa"
    train_dir = "data_pinn/train"
    test_dir  = "data_pinn/test"
    TrainData = load_datasets(train_dir, feature_keys, output_key)
    TestData  = load_datasets(test_dir,  feature_keys, output_key)

    input_dim = TrainData[0].X.shape[1]
    num_tasks = len(TrainData)

    model = MetaPINN(
        input_dim=input_dim, num_tasks=num_tasks,
        task_dim=128, hidden_dim=384, use_uncertainty=True,
        cond_dim=DEFAULT_OPTIONS.get('cond_dim', 1),
        use_cond_mod=DEFAULT_OPTIONS.get('use_cond_mod', True),
        cond_mod_from=DEFAULT_OPTIONS.get('cond_mod_from', 'target'),
        beta_min=DEFAULT_OPTIONS.get('beta_min', 0.15),
        beta_max=DEFAULT_OPTIONS.get('beta_max', 8.0)
    ).to(device)

    # save_dir = "saved_models/meta_pinn_offline"
    # os.makedirs(save_dir, exist_ok=True)
    # hist = train_meta_pinn_multitask(
    #     model, TrainData, TestData,
    #     DEFAULT_OPTIONS['UAV_mass'], DEFAULT_OPTIONS,
    #     save_path=save_dir
    # )
    save_dir = "saved_models/meta_pinn_offline"
    os.makedirs(save_dir, exist_ok=True)
    
    save_input_scaler(TrainData, os.path.join(save_dir, "x_scaler.npz"))
   
    hist = train_meta_pinn_multitask(
        model, TrainData, TestData,
        DEFAULT_OPTIONS['UAV_mass'], DEFAULT_OPTIONS,
        save_path=save_dir
    )

    torch.save(model.state_dict(), os.path.join(save_dir, "meta_pinn_last.pth"))
    np.save(os.path.join(save_dir, "history_train_total.npy"), np.array(hist['train_total']))
    np.save(os.path.join(save_dir, "history_test_mse.npy"),   np.array(hist['test_avg_mse']))

    import matplotlib.pyplot as plt
    fig = plot_training_curves(hist)
    fig.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close(fig)
    
    print("Offline Meta-PINN training finished!")

if __name__ == "__main__":
    main()
